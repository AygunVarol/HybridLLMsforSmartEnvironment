import csv
import os
import time
import uuid
import threading

class MetricsManager:
    def __init__(self, filename="metrics_log.csv"):
        self.filename = filename
        self.lock = threading.Lock()
        # If the file does not exist or is empty, write the header.
        if not os.path.exists(self.filename) or os.path.getsize(self.filename) == 0:
            with self.lock, open(self.filename, mode="w", newline="") as f:
                writer = csv.writer(f, quoting=csv.QUOTE_ALL)
                writer.writerow([
                    "record_id",
                    "timestamp",
                    "model",
                    "latency_ms",
                    "tokens",
                    "latency_per_token_ms",
                    "energy_consumption",
                    "cpu_energy_joules",
                    "gpu_energy_joules",
                    "memory_usage_gb",
                    "cpu_usage_percent",
                    "gpu_usage_percent",
                    "throughput_tps",
                    "ttft_ms",
                    "accuracy",
                    "user_input",
                    "response",
                    "retrieved_context",
                ])
    
    def log_metrics(
        self,
        model,
        latency_ms,
        tokens,
        user_input=None,
        response=None,
        retrieved_context=None,
        accuracy="pending",
        energy_consumption=None,
        cpu_energy_joules=None,
        gpu_energy_joules=None,
        memory_usage_gb=None,
        cpu_usage_percent=None,
        gpu_usage_percent=None,
        throughput_tps=None,
        ttft_ms=None,
    ):
        record_id = str(uuid.uuid4())
        latency_per_token = latency_ms / tokens if tokens > 0 else 0
        if throughput_tps is None:
            throughput_tps = tokens / (latency_ms / 1000) if latency_ms > 0 else 0
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        with self.lock, open(self.filename, mode="a", newline="") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            if user_input is not None:
                user_input = user_input.replace("\n", " ")
            if response is not None:
                response = response.replace("\n", " ")
            if retrieved_context is not None:
                retrieved_context = retrieved_context.replace("\n", " ")
            writer.writerow([
                record_id,
                timestamp,
                model,
                latency_ms,
                tokens,
                latency_per_token,
                energy_consumption,
                cpu_energy_joules,
                gpu_energy_joules,
                memory_usage_gb,
                cpu_usage_percent,
                gpu_usage_percent,
                throughput_tps,
                ttft_ms,
                accuracy,
                user_input,
                response,
                retrieved_context,
            ])
        return record_id

    def update_accuracy(self, record_id, accuracy):
        with self.lock:
            records = []
            updated = False
            # Read all records
            with open(self.filename, mode="r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["record_id"] == record_id:
                        row["accuracy"] = accuracy
                        updated = True
                    records.append(row)
            # If a record was updated, write all records back
            if updated:
                with open(self.filename, mode="w", newline="") as f:
                    fieldnames = [
                        "record_id",
                        "timestamp",
                        "model",
                        "latency_ms",
                        "tokens",
                        "latency_per_token_ms",
                        "energy_consumption",
                        "cpu_energy_joules",
                        "gpu_energy_joules",
                        "memory_usage_gb",
                        "cpu_usage_percent",
                        "gpu_usage_percent",
                        "throughput_tps",
                        "ttft_ms",
                        "accuracy",
                        "user_input",
                        "response",
                        "retrieved_context",
                    ]
                    writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
                    writer.writeheader()
                    writer.writerows(records)
            return updated

metrics_manager = MetricsManager()