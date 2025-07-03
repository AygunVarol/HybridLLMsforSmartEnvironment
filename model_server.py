import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_cpp import Llama

def initialize_llama_model():
    """
    Initializes and returns the meta-llama model and tokenizer.
    Uses "meta-llama/Llama-3.2-1B-instruct" from Hugging Face.
    """
    try:
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        dtype = torch.float16 if device == "cuda" else (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32)
        model_name = "meta-llama/Llama-3.2-1B-instruct"
        print(f"Loading Llama model: {model_name} on {device} with dtype {dtype}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
        if torch.__version__ >= "2.0.0":
            try:
                model = torch.compile(model)
                print("Llama model compiled successfully.")
            except Exception as e:
                print(f"Warning: Llama model compilation failed: {str(e)}")
        print("Llama model and tokenizer loaded successfully.")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading Llama model: {str(e)}")
        return None, None

def initialize_tinyllama_model():
    """
    Initializes and returns the TinyLlama model using llama-cpp-python.
    Loads the model from the Hugging Face cache.
    """
    try:
        base_cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
        model_path = os.path.join(
            base_cache_dir,
            "models--TheBloke--TinyLlama-1.1B-Chat-v1.0-GGUF",
            "snapshots",
            "52e7645ba7c309695bec7ac98f4f005b139cf465",
            "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
        )
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"TinyLlama model file not found at: {model_path}")
        print(f"Loading TinyLlama model from: {model_path} ...")
        model = Llama(model_path=model_path, n_threads=4, n_ctx=1024)
        print("TinyLlama model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading TinyLlama model: {e}")
        return None

def initialize_phi3mini_model():
    """
    Initializes and returns the Phi-3-mini model using llama-cpp-python.
    Loads the model from the Hugging Face cache.
    """
    try:
        base_cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
        # Update the snapshot ID if necessary. The folder name is constructed from the repository id.
        model_path = os.path.join(
            base_cache_dir,
            "models--microsoft--Phi-3-mini-4k-instruct-gguf",
            "snapshots",
            "999f761fe19e26cf1a339a5ec5f9f201301cbb83",  # Replace with the actual snapshot ID from your download
            "Phi-3-mini-4k-instruct-q4.gguf"
        )
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Phi-3-mini model file not found at: {model_path}")
        print(f"Loading Phi-3-mini model from: {model_path} ...")
        model = Llama(model_path=model_path, n_threads=4, n_ctx=1024)
        print("Phi-3-mini model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading Phi-3-mini model: {e}")
        return None

def get_llama_response(prompt, model, tokenizer, temperature=0.3, max_length=1024, num_beams=3, top_k=40, top_p=0.95, repetition_penalty=1.1):
    """
    Generates a response using the meta-llama model (transformers) with extended parameters.
    Note: Custom stop sequences aren’t directly supported in Hugging Face's generate.
    """
    if model is None or tokenizer is None:
        return "Error: Llama model is not initialized."
    try:
        device = next(model.parameters()).device
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_length=max_length,
                num_beams=num_beams,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                early_stopping=True
            )
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        return response
    except Exception as e:
        return f"Error during Llama text generation: {str(e)}"

def get_tinyllama_response(prompt, model, temperature=0.3, max_tokens=1024, top_k=40, top_p=0.95, repeat_penalty=1.1, stop=["[TINY_STOP]"]):
    """
    Generates a response using the TinyLlama model (llama-cpp-python).
    """
    if model is None:
        return "Error: TinyLlama model is not initialized."
    try:
        result = model(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            top_k=top_k,
            top_p=top_p,
            repeat_penalty=repeat_penalty
        )
        response = result["choices"][0]["text"].strip()
        return response
    except Exception as e:
        return f"Error during TinyLlama text generation: {str(e)}"

def get_phi3mini_response(prompt, model, temperature=0.3, max_tokens=1024, top_k=40, top_p=0.95, repeat_penalty=1.1, stop=["\n"]):
    """
    Generates a response using the Phi-3-mini model (llama-cpp-python).
    """
    if model is None:
        return "Error: Phi-3-mini model is not initialized."
    try:
        result = model(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            repeat_penalty=repeat_penalty,
            top_k=top_k,
            top_p=top_p
        )
        response = result["choices"][0]["text"].strip()
        return response
    except Exception as e:
        return f"Error during Phi-3-mini text generation: {str(e)}"
