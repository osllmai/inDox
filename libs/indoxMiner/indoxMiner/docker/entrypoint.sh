#!/bin/bash
set -e

# Function to log messages
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

log "Entrypoint script started."

# Check if MODEL_TYPE is provided
if [ -z "$MODEL_TYPE" ]; then
    log "ERROR: MODEL_TYPE environment variable is not set"
    exit 1
fi

# Check if the model directory exists
if [ ! -d "/app/models/$MODEL_TYPE" ]; then
    log "ERROR: Model directory /app/models/$MODEL_TYPE does not exist"
    exit 1
fi

# Install system dependencies if they exist
if [ -f "/app/models/$MODEL_TYPE/system_dependencies.txt" ]; then
    log "Installing system dependencies for model: $MODEL_TYPE"
    if [ "$(id -u)" -eq 0 ]; then
        apt-get update
        xargs apt-get install -y --no-install-recommends < "/app/models/$MODEL_TYPE/system_dependencies.txt" || {
            log "ERROR: Failed to install system dependencies for model: $MODEL_TYPE"
            exit 1
        }
        rm -rf /var/lib/apt/lists/*
    else
        sudo apt-get update
        sudo xargs apt-get install -y --no-install-recommends < "/app/models/$MODEL_TYPE/system_dependencies.txt" || {
            log "ERROR: Failed to install system dependencies for model: $MODEL_TYPE"
            exit 1
        }
        sudo rm -rf /var/lib/apt/lists/*
    fi
else
    log "No system dependencies found for model: $MODEL_TYPE"
fi

# Install Python dependencies if they exist
if [ -f "/app/models/$MODEL_TYPE/requirements.txt" ]; then
    log "Installing Python dependencies for model: $MODEL_TYPE"
    pip install --no-cache-dir -r "/app/models/$MODEL_TYPE/requirements.txt" || {
        log "ERROR: Failed to install Python dependencies for model: $MODEL_TYPE"
        exit 1
    }
else
    log "No Python dependencies found for model: $MODEL_TYPE"
fi

# Start the application
log "Starting API server..."
exec uvicorn api:app --host 0.0.0.0 --port 8001
