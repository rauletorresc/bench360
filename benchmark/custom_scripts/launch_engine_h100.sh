#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# launch_engine  —  Launch one of several LLM inference engines via Docker
#
# Usage:
#   ./launch_engine --engine=<engine> --model=<model> --name<name>
#
#   <engine> ∈ { vllm | sglang }
#   <model>  is the Hugging Face model ID (e.g. “mistralai/Mistral-7B-Instruct-v0.3”)
#   <name>   is the name for the container
#
# Example:
#   ./launch_engine --engine=tgi --model=mistralai/Mistral-7B-Instruct-v0.3 -name="bench360_inference_engine"
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
    --name=*)
      NAME="${ARG#--name=}"
      ;;
    *)
      echo "Unknown argument: $ARG"
      echo "Usage: $0 --engine=<vllm|sglang> --model=<your-org/your-model-name> --name=<container-name>"
      exit 1
      ;;
  esac
done

if [[ -z "$ENGINE" || -z "$MODEL" || -z "$NAME" ]]; then
  echo "Error: All --engine, --model and --name must be provided."
  echo "Usage: $0 --engine=<vllm|sglang> --model=<your-org/your-model-name> --name=<container-name>"
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
      --name $NAME \
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
      -e VLLM_LOGGING_LEVEL=CRITICAL \
      --shm-size="20g" \
      vllm/vllm-openai:latest \
        --model "$MODEL" \
        --trust-remote-code \
        --max-model-len 4096 \
        --port "$PORT" \
        --gpu-memory-utilization 0.3 \
        --tensor-parallel-size 1 \
        --distributed-executor-backend="mp" \
        -cc.cudagraph_mode="FULL_DECODE_ONLY"
    ;;

  sglang)
    docker run --rm \
      --gpus all \
      --name $NAME \
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
          --enable-metrics
        "
    ;;

  xllm)
    set -o xtrace
    docker run --rm \
      --gpus all \
      --name $NAME \
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
      quay.io/jd_xllm/xllm-ai:xllm-dev-cuda-x86 \
        bash -c "git config --global http.sslverify "false" && \
        git clone https://github.com/jd-opensource/xllm && \
        cd xllm && \
        pip install pre-commit && \
        pre-commit install && \
        git submodule update --init && \
        python setup.py build && \
        .build/bin/xllm \
        --model '$MODEL' \
        --port '$PORT' \
        --master_node_addr='127.0.0.1:9748' \
        --nnodes=1 \
        --max_memory_utilization=0.3 \
        --block_size=128 \
        --enable_prefix_cache=false \
        --enable_chunked_prefill=true \
        --enable_schedule_overlap=true \
        --node_rank=0"
    ;;

  *)
    echo "Error: unsupported engine '$ENGINE'."
    echo "Please choose one of: vllm, sglang."
    exit 1
    ;;
esac
