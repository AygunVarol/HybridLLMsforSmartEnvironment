import pandas as pd
import torch
import re
import datetime
from sentence_transformers import SentenceTransformer, util
from log_manager import log_manager  # Import the global log manager

class RagHandler:
    def __init__(self):
        # We'll store each document as a dictionary with text and metadata.
        self.documents = []       # List of dicts, each with keys: "text" and "metadata"
        self.embeddings = None    # Precomputed embeddings as a torch.Tensor
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    def _generate_default_template(self, df):
        """
        Generates a default narrative template for any given DataFrame.
        This simply joins all column names with their corresponding values.
        """
        columns = df.columns.tolist()
        # Create a simple template like "col1: {col1}; col2: {col2}; ..."
        template = " ".join([f"{col}: {{{col}}};" for col in columns])
        return template

    def process_csv(self, filepath, template=None):
        """
        Processes a CSV file and builds a vector index for retrieval-augmented generation (RAG).
        This method is flexible: if no custom template is provided, it auto-generates one.
        It also stores row metadata (e.g., Date) for structured filtering.
        """
        try:
            df = pd.read_csv(filepath)
            if df.empty:
                msg = f"Error: CSV file '{filepath}' is empty."
                log_manager.add_log(msg)
                print(msg)
                return
            
            # If no custom template is provided, generate one automatically.
            if template is None:
                template = self._generate_default_template(df)
                msg = f"Auto-generated template: {template}"
            else:
                msg = f"Using provided template: {template}"
            log_manager.add_log(msg)
            print(msg)
            
            # Build documents: each row becomes a dict with a narrative text and its metadata.
            documents = []
            for _, row in df.iterrows():
                try:
                    text = template.format(**row)
                except Exception as e:
                    text = " ".join([f"{col}: {row[col]}" for col in df.columns])
                metadata = {}
                # Look for a column that represents the date (case-insensitive).
                for col in df.columns:
                    if col.lower() == "date":
                        # Normalize the date using our helper function.
                        metadata["Date"] = self.normalize_date(str(row[col]).strip())
                        break
                documents.append({"text": text, "metadata": metadata})
            
            self.documents = documents
            msg = f"Generated 'documents' from CSV with {len(documents)} entries."
            log_manager.add_log(msg)
            print(msg)
            
            # Compute embeddings for all document texts.
            texts = [doc["text"] for doc in documents]
            self.embeddings = self.embedding_model.encode(texts, convert_to_tensor=True)
            
            msg = "CSV processing complete. Data indexed for retrieval."
            log_manager.add_log(msg)
            print(msg)
        
        except pd.errors.EmptyDataError:
            msg = f"Error: CSV file '{filepath}' is empty."
            log_manager.add_log(msg)
            print(msg)
        except KeyError as e:
            msg = f"Error: Missing expected column in CSV - {str(e)}"
            log_manager.add_log(msg)
            print(msg)
        except Exception as e:
            msg = f"Unexpected error while processing CSV: {str(e)}"
            log_manager.add_log(msg)
            print(msg)

    def normalize_date(self, date_str):
        """
        Normalizes a date string into a standard format.
        Attempts to parse both dd.mm.yyyy and yyyy-mm-dd formats and returns dd.mm.yyyy.
        """
        # Try dd.mm.yyyy format first.
        try:
            dt = datetime.datetime.strptime(date_str, "%d.%m.%Y")
            return dt.strftime("%d.%m.%Y")
        except ValueError:
            pass
        # Try ISO format: yyyy-mm-dd.
        try:
            dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
            return dt.strftime("%d.%m.%Y")
        except ValueError:
            pass
        # Fallback: return the original string.
        return date_str
        
    def retrieve_context(self, query, top_k=5, score_threshold=0.65):
        """
        Retrieves the most relevant context passages for a given query.
        Supports queries with a single date (e.g. "On 24.11.2020 what was the parameter?")
        as well as date ranges (e.g. "On 24.11.2020 to 30.11.2020 how was the temperature change?").
        If no relevant documents are found based on date filtering, it falls back to dense retrieval.
        """
        if not self.documents or self.embeddings is None:
            msg = "Warning: No data available for retrieval."
            log_manager.add_log(msg)
            print(msg)
            return ""

        # Define date patterns: dd.mm.yyyy and yyyy-mm-dd.
        date_patterns = [r"\b\d{1,2}\.\d{1,2}\.\d{4}\b", r"\b\d{4}-\d{2}-\d{2}\b"]

        # Collect all date matches from the query.
        date_matches = []
        for pattern in date_patterns:
            matches = re.findall(pattern, query)
            date_matches.extend(matches)

        parsed_dates = []
        for date_str in date_matches:
            normalized = self.normalize_date(date_str)
            try:
                dt = datetime.datetime.strptime(normalized, "%d.%m.%Y")
                parsed_dates.append(dt)
            except Exception as e:
                # If parsing fails, skip this date.
                continue

        filtered_docs = []

        if parsed_dates:
            if len(parsed_dates) >= 2:
                # If two or more dates are found, treat them as a range.
                start_date = min(parsed_dates)
                end_date = max(parsed_dates)
                msg = f"Detected date range: {start_date.strftime('%d.%m.%Y')} to {end_date.strftime('%d.%m.%Y')}"
                log_manager.add_log(msg)
                print(msg)

                # Filter documents that have a "Date" in their metadata within the date range.
                for doc in self.documents:
                    doc_date_str = doc.get("metadata", {}).get("Date", "")
                    try:
                        doc_date = datetime.datetime.strptime(doc_date_str, "%d.%m.%Y")
                        if start_date <= doc_date <= end_date:
                            filtered_docs.append(doc)
                    except Exception:
                        # Skip if the document's date is not in the expected format.
                        continue

                if not filtered_docs:
                    msg = f"No documents found between {start_date.strftime('%d.%m.%Y')} and {end_date.strftime('%d.%m.%Y')}. Falling back to dense retrieval."
                    log_manager.add_log(msg)
                    print(msg)
            else:
                # Only one date was found, so filter for that exact date.
                query_date = parsed_dates[0].strftime("%d.%m.%Y")
                msg = f"Detected single date: {query_date}"
                log_manager.add_log(msg)
                print(msg)
                for doc in self.documents:
                    if doc.get("metadata", {}).get("Date", "") == query_date:
                        filtered_docs.append(doc)

                if not filtered_docs:
                    msg = f"No documents found for date '{query_date}'. Falling back to dense retrieval."
                    log_manager.add_log(msg)
                    print(msg)
        
        if filtered_docs:
            context = "\n".join([doc["text"] for doc in filtered_docs])
            msg = f"Date-filtered context for query '{query}': {context}"
            log_manager.add_log(msg)
            print(msg)
            return context

        # Fallback: perform dense retrieval if no date-based filtering produced results.
        try:
            query_embedding = self.embedding_model.encode(query, convert_to_tensor=True)
            scores = util.dot_score(query_embedding, self.embeddings)[0]
            scores_np = scores.cpu().numpy()  # Convert scores to a NumPy array for processing

            sorted_indices = scores_np.argsort()[::-1]
            filtered_contexts = [
                self.documents[i]["text"] for i in sorted_indices if scores_np[i] > score_threshold
            ][:top_k]

            context = "\n".join(filtered_contexts)
            msg = f"Dense retrieval context for query '{query}': {context if context else 'No relevant context found.'}"
            log_manager.add_log(msg)
            print(msg)
            return context

        except Exception as e:
            msg = f"Error in context retrieval: {str(e)}"
            log_manager.add_log(msg)
            print(msg)
            return ""