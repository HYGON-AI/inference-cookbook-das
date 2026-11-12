#!/bin/bash
set -euo pipefail

unset PYTHONPATH HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_USE_MODELSCOPE=1
export VLLM_HOST_IP="${1:?usage: D-server.sh D_NODE_IP}"
export VLLM_MOONCAKE_BOOTSTRAP_PORT=18999
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export MC_GID_INDEX=3
export NCCL_IB_GID_INDEX=3
export ROCSHMEM_IPC_MNVL=1
export VLLM_DEEPEP_BUFFER_SIZE_MB=0

export GPU_MAX_HW_QUEUES=3
export VLLM_HCU_USE_FUSED_RMS_QUANT=1
export VLLM_HCU_ENABLE_HYV4_ATTN_INPUT_OVERLAP=1
export VLLM_HCU_ENABLE_BYTELEVEL_DETOKENIZER_TAIL=1
export TOKENIZERS_PARALLELISM=false

exec vllm serve hygon/Hy4-preview-Channel-INT4-w4a8 \
  --trust-remote-code \
  --quantization compressed-tensors \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --enable-expert-parallel \
  --moe-backend deep_gemm \
  --enable-prefix-caching \
  --kv-cache-dtype fp8_e4m3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"use_local_argmax_reduction":true}' \
  --default-chat-template-kwargs '{"reasoning_effort":"no_think"}' \
  --reasoning-parser hy_v4 \
  --enable-auto-tool-choice \
  --tool-call-parser hy_v4 \
  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"mooncake_protocol":"rdma"}}' \
  --data-parallel-size 16 \
  --data-parallel-size-local 16 \
  --data-parallel-start-rank 0 \
  --data-parallel-address 127.0.0.1 \
  --data-parallel-rpc-port 28927 \
  --data-parallel-hybrid-lb \
  --api-server-count 1 \
  --all2all-backend deepep_low_latency \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 32 \
  --max-model-len 89088
