# HybridLLM
This repository contains a Flask based application for experimenting with large language models in smart environments. It logs various metrics such as latency, CPU and GPU usage and energy consumption.

## Overview
![Framework-Aygun-VAROL](https://github.com/user-attachments/assets/53191d9f-c915-4009-958b-5bf4e8a53d08)

## Update Notes
- This updated version stores data both on RPis and edge devices.
- It has comprehensive metrics (energy consumption, cpu usage percentage, etc.)
- It also stores chat history and retrieved context from the RAG

## Installation
Install Python dependencies using pip:
```bash
pip install -r requirements.txt
```

## Running
Start the Flask server with:
```bash
python main.py
```
The server runs on port 5000 by default.

