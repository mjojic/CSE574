#!/bin/bash
set -e

# ==================== CONFIG ====================

# Local model snapshot
MODEL_32B_PATH="${MODEL_32B_PATH:-/scratch/mjojic/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137}"

# Server config
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8011}"

# GPU config
GPU="${GPU:-0}"
TP_SIZE="${TP_SIZE:-1}"

# Runtime files
PID_FILE="${PID_FILE:-vllm_32b.pid}"
LOG_FILE="${LOG_FILE:-vllm_32b.log}"

# ==================== STARTUP INFO ====================

echo "============================================"
echo "Starting Qwen3-32B vLLM server"
echo "============================================"
echo "Model: $MODEL_32B_PATH"
echo "Host:  $HOST"
echo "Port:  $PORT"
echo "GPU:   $GPU"
echo "TP:    $TP_SIZE"
echo "Log:   $LOG_FILE"
echo "PID:   $PID_FILE"
echo "============================================"
echo ""

# ==================== ENVIRONMENT ====================

module load mamba/latest
source activate qwen3-fa2

echo "Checking GPU availability..."
nvidia-smi --query-gpu=index,name,memory.free,memory.total --format=csv
echo ""

# ==================== START SERVER ====================

echo "Starting Qwen3-32B vLLM server on GPU $GPU, port $PORT..."

CUDA_VISIBLE_DEVICES="$GPU" vllm serve "$MODEL_32B_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.9 \
    --tensor-parallel-size "$TP_SIZE" \
    --served-model-name Qwen3-32B \
> "$LOG_FILE" 2>&1 &

PID=$!
echo "$PID" > "$PID_FILE"

echo "Qwen3-32B PID: $PID"
echo "PID saved to $PID_FILE"
echo "To stop: kill \$(cat $PID_FILE)"
echo ""

# ==================== HEALTH CHECK ====================

echo "Waiting for Qwen3-32B server to become ready..."

while ! curl -sf "http://$HOST:$PORT/health" >/dev/null 2>&1; do
    sleep 5

    if ! kill -0 "$PID" 2>/dev/null; then
        echo "ERROR: Qwen3-32B server crashed. Check $LOG_FILE"
        tail -20 "$LOG_FILE"
        exit 1
    fi

    echo "  Still waiting for Qwen3-32B..."
done

echo ""
echo "============================================"
echo "Qwen3-32B vLLM server is running."
echo "URL:     http://$HOST:$PORT"
echo "Health:  http://$HOST:$PORT/health"
echo "Log:     $LOG_FILE"
echo "Stop:    kill \$(cat $PID_FILE)"
echo "============================================"