#!/bin/bash
set -euo pipefail

unset PYTHONPATH HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_USE_MODELSCOPE=1
export VLLM_HOST_IP="${1:?usage: P-server.sh P_NODE_IP}"
export VLLM_MOONCAKE_BOOTSTRAP_PORT=18998
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export MC_GID_INDEX=3
export NCCL_IB_GID_INDEX=3
export ROCSHMEM_GDR_DISABLE_XDP=1
export DEEP_EP_NORMAL_MNVL=1
export VLLM_DEEPEP_BUFFER_SIZE_MB=4096

export GPU_MAX_HW_QUEUES=4
export VLLM_HCU_ENABLE_BYTELEVEL_DETOKENIZER_TAIL=1
export VLLM_HCU_ENABLE_PREFILL_MQA_GROUPING=1
export VLLM_HCU_ENABLE_TRITON_CP_GATHER=1
export VLLM_HCU_HYV4_MOE_CHUNK_SIZE=1024
export VLLM_HCU_SHARED_EXPERTS_STREAM_FORCE=1
export VLLM_HCU_STATIC_HT_W4A8_COUNT_SCRATCH=1
export HY4_EXPERIMENTAL_STATIC_R16=1
export LIGHTOP_MQA_LOGITS_BETA=1
export TOKENIZERS_PARALLELISM=true
export RAYON_NUM_THREADS=4

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
  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_producer","kv_connector_extra_config":{"mooncake_protocol":"rdma","num_workers":80}}' \
  --prefill-context-parallel-size 16 \
  --all2all-backend deepep_high_throughput \
  --hf-overrides '{"head_dtype":"float32"}' \
  --gpu-memory-utilization 0.90 \
  --max-num-batched-tokens 98304 \
  --enforce-eager \
  --max-model-len 88000 \
  --no-async-scheduling \
  --renderer-num-workers 4 \
  --long-prefill-token-threshold 65536 \
  --enable-eplb \
  --eplb-config '{"expert_map_path":"/workspace/hy4-pd-eplb/r16.json","num_redundant_experts":16,"static_dispatch_policy":"nearest"}'
