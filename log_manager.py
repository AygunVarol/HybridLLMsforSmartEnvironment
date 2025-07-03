import threading
import time
from collections import deque

class LogManager:
    def __init__(self, max_logs=1000):
        """Thread-safe log manager with an optional log size limit."""
        self.logs = deque(maxlen=max_logs)  # Keeps only the latest N logs
        self.lock = threading.Lock()

    def add_log(self, message):
        """Adds a new log entry with a timestamp."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        log_entry = f"[{timestamp}] {message}"
        
        with self.lock:
            self.logs.append(log_entry)

    def get_logs(self):
        """Returns a copy of logs to prevent concurrency issues."""
        with self.lock:
            return list(self.logs)  # Convert deque to list for safer access

    def clear_logs(self):
        """Clears all logs safely."""
        with self.lock:
            self.logs.clear()

# Create a global instance
log_manager = LogManager()