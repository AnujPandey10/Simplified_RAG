#!/bin/bash

# Ensure Ollama is running
echo "Checking if Ollama is running..."
if ! pgrep -x "ollama" > /dev/null; then
    echo "Ollama is not running. Starting Ollama..."
    open -a Ollama
    # Give Ollama time to start
    sleep 5
else
    echo "Ollama is already running."
fi

# Check if at least one model is available
echo "Checking for available models..."
MODEL_COUNT=$(ollama list | grep -c "^[a-zA-Z]")

if [ "$MODEL_COUNT" -eq 0 ]; then
    echo "No models found. Please download at least one model using 'ollama pull <model-name>'."
    echo "For example: ollama pull llama2"
    exit 1
fi

echo "Found $MODEL_COUNT model(s)."

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p data/uploaded_pdfs data/preloaded_pdfs data/vector_stores

# Install dependencies if needed
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Creating one and installing dependencies..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    echo "Using existing virtual environment."
    source venv/bin/activate
fi

# Start the application
echo "Starting the RAG-Enabled Chat Application..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
