import queue
import threading
import time
from log_manager import log_manager  # Import the global log manager

class TaskQueue:
    def __init__(self):
        """Thread-safe task queue with graceful shutdown support."""
        self.tasks = queue.Queue()
        self.running = threading.Event()
        self.running.set()  # Flag to keep the worker running

    def add_task(self, func, *args, **kwargs):
        """Enqueue a task: a function plus its arguments."""
        self.tasks.put((func, args, kwargs))
        log_manager.add_log(f"Task added: {func.__name__} with args={args} kwargs={kwargs}")

    def run(self):
        """Continuously process tasks from the queue until stopped."""
        log_manager.add_log("Task queue started.")
        while self.running.is_set():
            try:
                func, args, kwargs = self.tasks.get(timeout=1)  # Non-blocking wait
                log_manager.add_log(f"Executing task: {func.__name__}")
                func(*args, **kwargs)
                self.tasks.task_done()
            except queue.Empty:
                continue  # No task in queue, loop again
            except Exception as e:
                log_manager.add_log(f"Task error: {str(e)}")

        log_manager.add_log("Task queue stopped.")

    def stop(self):
        """Stops the task queue gracefully."""
        self.running.clear()
        log_manager.add_log("Task queue stopping...")

    def run_in_thread(self):
        """Runs the task queue in a separate background thread."""
        worker_thread = threading.Thread(target=self.run, daemon=True)
        worker_thread.start()
        log_manager.add_log("Background worker thread started.")