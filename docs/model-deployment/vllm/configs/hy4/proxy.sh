#!/bin/bash
set -euo pipefail

export VLLM_PLUGINS=__disabled__
export HY4_PROXY_RUN_DIR=/workspace/hy4-pd-eplb/proxy_run
unset PYTHONPATH HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
mkdir -p "$HY4_PROXY_RUN_DIR"

exec /usr/bin/python -u \
  /workspace/hy4-pd-eplb/proxy/text_proxy.py \
  --host 0.0.0.0 \
  --port 8002 \
  --prefill "http://${1:?usage: proxy.sh P_NODE_IP D_NODE_IP}:8000" 18998 \
  --decode "http://${2:?usage: proxy.sh P_NODE_IP D_NODE_IP}:8000"
