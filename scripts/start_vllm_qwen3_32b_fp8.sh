#!/usr/bin/env bash
# Start an OpenAI-compatible vLLM server for Qwen/Qwen3-32B-FP8.
#
# Run this on the machine that has GPUs (and the model cache). Same Hub layout as
# scripts/download_qwen3_32b_fp8.py: HF_HOME=/scratch/mjojic/huggingface → cache in .../hub.
# Note: `vllm serve --help` may fail on hosts without CUDA/NVML (device inference runs at CLI init).
#
# Prerequisites on the GPU host: vLLM installed (see https://docs.vllm.ai), CUDA drivers,
# and enough VRAM for the model (often tensor_parallel_size 2+ for 32B-class models).
#
# Examples:
#   export CUDA_VISIBLE_DEVICES=0,1   # use only GPUs you are allowed (e.g. 0–3)
#   export TENSOR_PARALLEL_SIZE=2
#   ./scripts/start_vllm_qwen3_32b_fp8.sh
#
# Optional:
#   export MAX_MODEL_LEN=8192         # lower context if you hit OOM
#   export HF_TOKEN=hf_...            # if Hub auth is required
#   ./scripts/start_vllm_qwen3_32b_fp8.sh --kv-cache-dtype fp8   # extra vLLM flags at end
#
# Once the server is up, point the planner at it:
#   export OPENAI_BASE_URL=http://localhost:8000/v1
#   python run_blocksworld_one_llm.py --problem-type generated_basic --instance 1 \
#       --llm-model Qwen/Qwen3-32B-FP8 --llm-response-format json_object

set -euo pipefail

export HF_HOME="${HF_HOME:-/scratch/mjojic/huggingface}"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-32B-FP8}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"

EXTRA_ARGS=()
if [[ -n "${MAX_MODEL_LEN:-}" ]]; then
  EXTRA_ARGS+=(--max-model-len "${MAX_MODEL_LEN}")
fi

exec vllm serve "${MODEL_ID}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
