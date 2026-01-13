#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# launch_engine  —  Launch one of several LLM inference engines via Docker
#
# Usage:
#   ./launch_engine --engine=<engine> --model=<model>
#
#   <engine> ∈ { vllm | sglang }
#   <model>  is the Hugging Face model ID (e.g. “mistralai/Mistral-7B-Instruct-v0.3”)
#
# Example:
#   ./launch_engine --engine=tgi --model=mistralai/Mistral-7B-Instruct-v0.3
#
# Notes:
#   • Expects HF_TOKEN or HUGGING_FACE_HUB_TOKEN in your environment.
#   • Always listens on 127.0.0.1:23333 inside the container→host.
#   • Uses $HOME/.cache/huggingface as cache Dir.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ─── Parse arguments “--engine=…” and “--model=…” ────────────────────────────
ENGINE=""
MODEL=""

for ARG in "$@"; do
  case "$ARG" in
    --engine=*)
      ENGINE="${ARG#--engine=}"
      ;;
    --model=*)
      MODEL="${ARG#--model=}"
      ;;
    *)
      echo "Unknown argument: $ARG"
      echo "Usage: $0 --engine=<vllm|sglang> --model=<your-org/your-model-name>"
      exit 1
      ;;
  esac
done

if [[ -z "$ENGINE" || -z "$MODEL" ]]; then
  echo "Error: both --engine and --model must be provided."
  echo "Usage: $0 --engine=<vllm|sglang> --model=<your-org/your-model-name>"
  exit 1
fi

# ─── Common variables ───────────────────────────────────────────────────────
PORT=23333
CACHE_DIR="$HOME/.cache/huggingface"

# Ensure at least one token is set
if [[ -z "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  echo "Error: You must export HF_TOKEN or HUGGING_FACE_HUB_TOKEN in your environment."
  exit 1
fi

# ─── Select and run the requested engine ────────────────────────────────────
case "$ENGINE" in

  vllm)
    docker run --rm \
      --runtime=nvidia --gpus all \
      -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
      -e HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
      -p 127.0.0.1:${PORT}:${PORT} \
      --ipc=host \
      --network=host \
      -e http_proxy="$HTTP_PROXY" \
      -e https_proxy="$HTTPS_PROXY" \
      -e no_proxy="$NO_PROXY" \
      -e HTTP_PROXY="$HTTP_PROXY" \
      -e HTTPS_PROXY="$HTTPS_PROXY" \
      -e NO_PROXY="$NO_PROXY" \
      --shm-size="20g" \
      vllm/vllm-openai:latest \
        --model "$MODEL" \
        --trust-remote-code \
        --max-model-len 4096 \
        --port "$PORT" \
        --gpu-memory-utilization 0.3
    ;;

  sglang)
    docker run --rm \
      --gpus all \
      -p 127.0.0.1:${PORT}:${PORT} \
      -v ~/.cache/huggingface:/root/.cache/huggingface \
      --ipc=host \
      --network=host \
      -e http_proxy="$HTTP_PROXY" \
      -e https_proxy="$HTTPS_PROXY" \
      -e no_proxy="$NO_PROXY" \
      -e HTTP_PROXY="$HTTP_PROXY" \
      -e HTTPS_PROXY="$HTTPS_PROXY" \
      -e NO_PROXY="$NO_PROXY" \
      --shm-size="20g" \
      -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN \
      lmsysorg/sglang:latest \
      bash -c "\
        pip install --no-cache-dir protobuf sentencepiece --break-system-packages && \
        python3 -m sglang.launch_server \
          --model-path $MODEL \
          --host 0.0.0.0 \
          --port $PORT \
        "
    ;;

  *)
    echo "Error: unsupported engine '$ENGINE'."
    echo "Please choose one of: vllm, sglang."
    exit 1
    ;;
esac
