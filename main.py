from flask import Flask, request, jsonify, render_template
import os
import csv
import threading
import time
import psutil
import subprocess
from model_server import (
    get_llama_response,
    get_tinyllama_response,
    get_phi3mini_response,
    initialize_llama_model,
    initialize_tinyllama_model,
    initialize_phi3mini_model
)
from rag import RagHandler
from background_worker import TaskQueue
from log_manager import log_manager  # Global log manager
from metrics_manager import metrics_manager  # Metrics logging

# --- Groq / Llama 3 Cloud Setup ---
from groq import Groq
groq_api_key = "GROQ_API"  # Replace with your Groq API key
groq_client = Groq(api_key=groq_api_key)

app = Flask(__name__)

SENSOR_LOG = "edge_sensor_data.csv"
SENSOR_FIELDS = [
    "timestamp",
    "location",
    "temperature",
    "pressure",
    "humidity",
    "gas",
]

if not os.path.exists(SENSOR_LOG) or os.path.getsize(SENSOR_LOG) == 0:
    with open(SENSOR_LOG, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(SENSOR_FIELDS)

def get_gpu_usage_percent():
    """Returns GPU utilization percentage or 0 if unavailable."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        usage_str = result.stdout.strip().split("\n")[0]
        return float(usage_str)
    except Exception:
        return 0.0

def read_cpu_energy_uj():
    """Read CPU energy in microjoules from RAPL if available."""
    try:
        with open("/sys/class/powercap/intel-rapl:0/energy_uj", "r") as f:
            return int(f.read().strip())
    except Exception:
        return None

def read_gpu_energy_mj():
    """Read GPU energy in millijoules using nvidia-smi if available."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=total_energy_consumption",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return float(result.stdout.strip().split("\n")[0])
    except Exception:
        return None

# Initialize components
task_queue = TaskQueue()
rag_handler = RagHandler()

# Initialize local models
llama_model, llama_tokenizer = initialize_llama_model()
tinyllama_model = initialize_tinyllama_model()
phi3mini_model = initialize_phi3mini_model()

# Start the background worker thread
worker_thread = threading.Thread(target=task_queue.run, daemon=True)
worker_thread.start()

# Global variable for sensor data
latest_sensor_data = {}

def get_cloud_response(prompt):
    """
    Generates a response from the Cloud model via Groq.
    Prepares a conversation history and calls the Groq API.
    """
    system_prompt = {
        "role": "system",
        "content": "You are a helpful assistant. Please answer concisely."
    }
    user_message = {
        "role": "user",
        "content": prompt
    }
    chat_history = [system_prompt, user_message]
    
    try:
        response = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=chat_history,
            max_tokens=150,
            temperature=0.7
        )
        cloud_response = response.choices[0].message.content
        return cloud_response
    except Exception as e:
        return f"Error during Cloud model text generation: {str(e)}"

@app.route('/sensor_data', methods=['POST'])
def sensor_data():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No sensor data provided"}), 400
    global latest_sensor_data
    latest_sensor_data.update(data)

    row = {
        "timestamp": data.get("timestamp", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
        "location": data.get("location", "unknown"),
        "temperature": data.get("temperature"),
        "pressure": data.get("pressure"),
        "humidity": data.get("humidity"),
        "gas": data.get("gas"),
    }

    with open(SENSOR_LOG, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SENSOR_FIELDS)
        writer.writerow(row)

    log_manager.add_log(f"Sensor data updated: {data}")
    return jsonify({"message": "Sensor data updated successfully."}), 200

@app.route('/')
def index():
    return render_template('index-mobile.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    if not data or 'user_input' not in data:
        return jsonify({'error': 'Invalid request format'}), 400

    user_input = data.get('user_input', '').strip()
    if not user_input:
        return jsonify({'error': 'Empty input'}), 400

    # Expected model values: "Cloud Model", "Llama Model", "TinyLlama Model", or "Phi-3-mini Model"
    selected_model = data.get('model', 'Llama Model')

    # Retrieve contextual information using RAG (if available)
    context = rag_handler.retrieve_context(user_input)
    if context:
        prompt = (
            f"Context: \"{context}\"\n\n"
            "Please provide only the final answer to the following question. Do not include any internal reasoning or chain-of-thought:\n"
            f"{user_input}\n"
            "Answer:"
        )
    else:
        prompt = (
            "Please provide only the final answer to the following question. Do not include any internal reasoning or chain-of-thought:\n"
            f"{user_input}\n"
            "Answer:"
        )

    # Incorporate real-time sensor data if available
    global latest_sensor_data
    if latest_sensor_data:
        sensor_context = "Real-time Sensor Data:\n" + "\n".join(
            [f"{key.capitalize()}: {value}" for key, value in latest_sensor_data.items()]
        )
        prompt = f"{sensor_context}\n\n{prompt}"

    try:
        start_cpu_energy = read_cpu_energy_uj()
        start_gpu_energy = read_gpu_energy_mj()
        start_time = time.time()
        if selected_model == "Cloud Model":
            response_text = get_cloud_response(prompt)
            model_type = "Cloud Model"
            prefixed_response = f"Cloud: {response_text}"
        elif selected_model == "TinyLlama Model":
            response_text = get_tinyllama_response(prompt, tinyllama_model)
            model_type = "TinyLlama Model"
            prefixed_response = f"TinyLlama: {response_text}"
        elif selected_model == "Phi-3-mini Model":
            response_text = get_phi3mini_response(prompt, phi3mini_model)
            model_type = "Phi-3-mini Model"
            prefixed_response = f"Phi-3-mini: {response_text}"
        else:  # default to Llama Model
            response_text = get_llama_response(prompt, llama_model, llama_tokenizer)
            model_type = "Llama Model"
            prefixed_response = f"Llama: {response_text}"
        end_time = time.time()
        end_cpu_energy = read_cpu_energy_uj()
        end_gpu_energy = read_gpu_energy_mj()
        
        latency_ms = (end_time - start_time) * 1000
        tokens = len(response_text.split())

        cpu_energy_j = None
        if start_cpu_energy is not None and end_cpu_energy is not None:
            cpu_energy_j = (end_cpu_energy - start_cpu_energy) / 1e6

        gpu_energy_j = None
        if start_gpu_energy is not None and end_gpu_energy is not None:
            gpu_energy_j = (end_gpu_energy - start_gpu_energy) / 1000

        memory_usage = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)
        cpu_usage = psutil.cpu_percent(interval=0.1)
        gpu_usage = get_gpu_usage_percent()
        throughput = tokens / (latency_ms / 1000) if latency_ms > 0 else 0
        ttft_ms = latency_ms  # Approximation since streaming is not implemented

        record_id = metrics_manager.log_metrics(
            model=model_type,
            latency_ms=latency_ms,
            tokens=tokens,
            user_input=user_input,
            response=response_text,
            retrieved_context=context,
            accuracy="pending",
            energy_consumption=None,
            cpu_energy_joules=cpu_energy_j,
            gpu_energy_joules=gpu_energy_j,
            memory_usage_gb=memory_usage,
            cpu_usage_percent=cpu_usage,
            gpu_usage_percent=gpu_usage,
            throughput_tps=throughput,
            ttft_ms=ttft_ms,
        )
        
        return jsonify({'response': prefixed_response, 'record_id': record_id})
    except Exception as e:
        return jsonify({'error': f'Model processing error: {str(e)}'}), 500

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'}), 400

    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    os.makedirs('uploads', exist_ok=True)
    filepath = os.path.join('uploads', file.filename)
    file.save(filepath)

    task_queue.add_task(rag_handler.process_csv, filepath)
    return jsonify({'message': 'File uploaded successfully and processing started'}), 200

@app.route('/logs', methods=['GET'])
def get_logs():
    logs = log_manager.get_logs()
    return jsonify({'logs': logs})

@app.route('/update_accuracy', methods=['POST'])
def update_accuracy():
    data = request.get_json()
    if not data or "record_id" not in data or "accuracy" not in data:
        return jsonify({"error": "Missing record_id or accuracy"}), 400
    
    record_id = data["record_id"]
    accuracy = data["accuracy"]
    
    success = metrics_manager.update_accuracy(record_id, str(accuracy))
    if success:
        return jsonify({"message": "Accuracy updated successfully."}), 200
    else:
        return jsonify({"error": "Record not found."}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=False, use_reloader=False)
