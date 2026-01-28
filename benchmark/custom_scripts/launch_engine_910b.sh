#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# launch_engine  —  Launch one of several LLM inference engines via Docker
#
# Usage:
#   ./launch_engine --engine=<engine> --model=<model> --name<name>
#
#   <engine> ∈ { vllm | sglang | xllm }
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
      echo "Usage: $0 --engine=<vllm|sglang|xllm> --model=<your-org/your-model-name> --name=<container-name>"
      exit 1
      ;;
  esac
done

if [[ -z "$ENGINE" || -z "$MODEL" || -z "$NAME" ]]; then
  echo "Error: All --engine, --model and --name must be provided."
  echo "Usage: $0 --engine=<vllm|sglang|xllm> --model=<your-org/your-model-name> --name=<container-name>"
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
      --name $NAME \
      -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
      -e HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
      -p 127.0.0.1:${PORT}:${PORT} \
      --ipc=host \
      --network=host \
      --privileged=true \
      --device=/dev/davinci0 \
      --device=/dev/davinci1 \
      --device=/dev/davinci2 \
      --device=/dev/davinci3 \
      --device=/dev/davinci4 \
      --device=/dev/davinci5 \
      --device=/dev/davinci6 \
      --device=/dev/davinci7 \
      --device=/dev/davinci_manager \
      --device=/dev/devmm_svm \
      --device=/dev/hisi_hdc \
      -v /usr/local/sbin/:/usr/local/sbin/ \
      -v /var/log/npu/slog/:/var/log/npu/slog \
      -v /var/log/npu/profiling/:/var/log/npu/profiling \
      -v /var/log/npu/dump/:/var/log/npu/dump \
      -v /var/log/npu/:/usr/slog \
      -v /etc/hccn.conf:/etc/hccn.conf \
      -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
      -v /usr/local/dcmi:/usr/local/dcmi \
      -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
      -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
      -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
      -v /etc/ascend_install.info:/etc/ascend_install.info \
      -v /etc/vnpu.cfg:/etc/vnpu.cfg \
      -v /opt/shared/models/:/opt/shared/models/ \
      ${HOST_CA_CERT:+-v $HOST_CA_CERT:$HOST_CA_CERT:ro} \
      -e REQUESTS_CA_BUNDLE="$HOST_CA_CERT" \
      -e CURL_CA_BUNDLE="$HOST_CA_CERT" \
      -e SSL_CERT_FILE="$HOST_CA_CERT" \
      -e GIT_SSL_NO_VERIFY=true \
      -e GIT_SSL_CAINFO="$HOST_CA_CERT" \
      -e http_proxy="$HTTP_PROXY" \
      -e https_proxy="$HTTPS_PROXY" \
      -e no_proxy="$NO_PROXY" \
      -e HTTP_PROXY="$HTTP_PROXY" \
      -e HTTPS_PROXY="$HTTPS_PROXY" \
      -e NO_PROXY="$NO_PROXY" \
      -e VLLM_LOGGING_LEVEL=CRITICAL \
      --shm-size="250g" \
      quay.io/ascend/vllm-ascend:v0.13.0rc1 \
        vllm serve "$MODEL" \
        --trust-remote-code \
        --max-model-len 4096 \
        --port "$PORT" \
        --tensor-parallel-size 1 \
        --distributed-executor-backend="mp" \
        -cc.cudagraph_mode="FULL_DECODE_ONLY"
    ;;

sglang)
    docker run --rm \
      -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
      -e HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
      --name $NAME \
      -p 127.0.0.1:${PORT}:${PORT} \
      --ipc=host \
      --network=host \
      --privileged=true \
      --device=/dev/davinci0 \
      --device=/dev/davinci1 \
      --device=/dev/davinci2 \
      --device=/dev/davinci3 \
      --device=/dev/davinci4 \
      --device=/dev/davinci5 \
      --device=/dev/davinci6 \
      --device=/dev/davinci7 \
      --device=/dev/davinci_manager \
      --device=/dev/devmm_svm \
      --device=/dev/hisi_hdc \
      -v /usr/local/sbin/:/usr/local/sbin/ \
      -v /var/log/npu/slog/:/var/log/npu/slog \
      -v /var/log/npu/profiling/:/var/log/npu/profiling \
      -v /var/log/npu/dump/:/var/log/npu/dump \
      -v /var/log/npu/:/usr/slog \
      -v /etc/hccn.conf:/etc/hccn.conf \
      -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
      -v /usr/local/dcmi:/usr/local/dcmi \
      -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
      -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
      -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
      -v /etc/ascend_install.info:/etc/ascend_install.info \
      -v /etc/vnpu.cfg:/etc/vnpu.cfg \
      -v /opt/shared/models/:/opt/shared/models/ \
      ${HOST_CA_CERT:+-v $HOST_CA_CERT:$HOST_CA_CERT:ro} \
      -e REQUESTS_CA_BUNDLE="$HOST_CA_CERT" \
      -e CURL_CA_BUNDLE="$HOST_CA_CERT" \
      -e SSL_CERT_FILE="$HOST_CA_CERT" \
      -e GIT_SSL_NO_VERIFY=true \
      -e GIT_SSL_CAINFO="$HOST_CA_CERT" \
      -e http_proxy="$HTTP_PROXY" \
      -e https_proxy="$HTTPS_PROXY" \
      -e no_proxy="$NO_PROXY" \
      -e HTTP_PROXY="$HTTP_PROXY" \
      -e HTTPS_PROXY="$HTTPS_PROXY" \
      -e NO_PROXY="$NO_PROXY" \
      --shm-size="250g" \
      quay.io/ascend/sglang:v0.5.8-cann8.3.rc2-910b \
        bash -c "\
        pip install --no-cache-dir protobuf sentencepiece --break-system-packages && \
        sed -i -e 's/NPUUtils()\.get_arch()/NPUUtils()\.get_arch()\[:-2\]/g' /usr/local/python3.11.13/lib/python3.11/site-packages/triton/backends/ascend/compiler.py &&
        python3 -m sglang.launch_server \
          --model-path $MODEL \
          --host 0.0.0.0 \
          --port $PORT \
          --device npu \
          --attention-backend ascend \
          --mem-fraction-static 0.6 \
          --enable-metrics
        "
    ;;

xllm)
    docker run --rm \
      --name $NAME \
      -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
      -e HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
      -p 127.0.0.1:${PORT}:${PORT} \
      --ipc=host \
      --network=host \
      --privileged=true \
      --device=/dev/davinci0 \
      --device=/dev/davinci1 \
      --device=/dev/davinci2 \
      --device=/dev/davinci3 \
      --device=/dev/davinci4 \
      --device=/dev/davinci5 \
      --device=/dev/davinci6 \
      --device=/dev/davinci7 \
      --device=/dev/davinci_manager \
      --device=/dev/devmm_svm \
      --device=/dev/hisi_hdc \
      -v /usr/local/sbin/:/usr/local/sbin/ \
      -v /var/log/npu/slog/:/var/log/npu/slog \
      -v /var/log/npu/profiling/:/var/log/npu/profiling \
      -v /var/log/npu/dump/:/var/log/npu/dump \
      -v /var/log/npu/:/usr/slog \
      -v /etc/hccn.conf:/etc/hccn.conf \
      -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
      -v /usr/local/dcmi:/usr/local/dcmi \
      -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
      -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
      -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
      -v /etc/ascend_install.info:/etc/ascend_install.info \
      -v /etc/vnpu.cfg:/etc/vnpu.cfg \
      -v /opt/shared/models/:/opt/shared/models/ \
      -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
      ${HOST_CA_CERT:+-v $HOST_CA_CERT:$HOST_CA_CERT:ro} \
      -e REQUESTS_CA_BUNDLE="$HOST_CA_CERT" \
      -e CURL_CA_BUNDLE="$HOST_CA_CERT" \
      -e SSL_CERT_FILE="$HOST_CA_CERT" \
      -e GIT_SSL_NO_VERIFY=true \
      -e GIT_SSL_CAINFO="$HOST_CA_CERT" \
      -e http_proxy="$HTTP_PROXY" \
      -e https_proxy="$HTTPS_PROXY" \
      -e no_proxy="$NO_PROXY" \
      -e HTTP_PROXY="$HTTP_PROXY" \
      -e HTTPS_PROXY="$HTTPS_PROXY" \
      -e NO_PROXY="$NO_PROXY" \
      -e LD_PRELOAD=/usr/lib64/libtcmalloc.so.4 \
      --shm-size="250g" \
      quay.io/jd_xllm/xllm-ai:xllm-0.7.2-release-hb-rc2-arm \
        bash -c "python -c 'import torch_npu; [torch_npu.npu.set_device(i) for i in range(1)]' && /usr/local/bin/xllm \
        --model '$MODEL' \
        --devices='npu:0' \
        --port '$PORT' \
        --master_node_addr='127.0.0.1:9748' \
        --nnodes=1 \
        --max_memory_utilization=0.3 \
        --block_size=128 \
        --communication_backend='hccl' \
        --enable_prefix_cache=false \
        --enable_chunked_prefill=true \
        --enable_schedule_overlap=true \
        --node_rank=0"
    ;;

  *)
    echo "Error: unsupported engine '$ENGINE'."
    echo "Please choose one of: vllm, sglang, xllm."
    exit 1
    ;;
esac
