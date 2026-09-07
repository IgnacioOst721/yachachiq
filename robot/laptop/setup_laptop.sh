#!/bin/bash
# Laptop side. Only needed for the optional "remote" backends (Ollama + diffusers).
# For ComfyUI (the main image backend) see COMFYUI.md.
set -e
cd "$(dirname "$0")"
python3 -m venv venv
./venv/bin/pip install --upgrade pip
./venv/bin/pip install fastapi "uvicorn[standard]" requests torch diffusers transformers accelerate
echo "Run:  source venv/bin/activate && python ai_server.py"
