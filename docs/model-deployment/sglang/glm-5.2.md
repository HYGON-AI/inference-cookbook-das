# GLM-5.2

## 模型列表

| 模型权重 | 量化方式 | SGLang 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | ----------- | -------- | ---- | -------- | -------- |
| [hygon/GLM-5.2-Channel-INT8-w8a8](https://www.modelscope.cn/models/hygon/GLM-5.2-Channel-INT8-w8a8) | INT8 W8A8 | [0.5.12](../docker_images.md) | BW1100 | 8 | IFB | [**`>_`**](#glm-52-channel-int8-w8a8-ifb-bw1100-8x-sglang-0512) |
|                                                                                                 | INT8 W8A8 | [0.5.12](../docker_images.md) | BW1000 | 16 | IFB | [**`>_`**](#glm-52-channel-int8-w8a8-ifb-bw1000-16x-sglang-0512) |
| [hygon/GLM-5.2-Channel-FP8-w8a8](https://www.modelscope.cn/models/hygon/GLM-5.2-Channel-FP8-w8a8) | FP8 W8A8 | [0.5.12](../docker_images.md) | BW1100 | 8 | IFB(tp8) | [**`>_`**](#glm-52-channel-fp8-w8a8-ifb-bw1100-8x-sglang-0512-tp8) |
|  | FP8 W8A8 | [0.5.12](../docker_images.md) | BW1100 | 8 | IFB(tp8ep8) | [**`>_`**](#glm-52-channel-fp8-w8a8-ifb-bw1100-8x-sglang-0512-tp8ep8) |
|  | FP8 W8A8 | [0.5.12](../docker_images.md) | scaleX40-3G | 24 | PD | [**`>_`**](#glm-52-channel-fp8-w8a8-pd-scalex40-3g-24x-sglang-0512) |
| [hygon/GLM-5.2-Channel-INT4-w4a8](https://www.modelscope.cn/models/hygon/GLM-5.2-Channel-INT4-w4a8) | INT4 W4A8 | 0.5.18 | BW1000 | 48 | 2P4D | [**`>_`**](#glm-52-channel-int4-w4a8-2p4d-bw1000-48x-sglang-0518) |
|  | INT4 W4A8 | [0.5.12](../docker_images.md) | BW1100 | 4 | IFB | [**`>_`**](#glm-52-channel-int4-w4a8-ifb-bw1100-4x-sglang-0512) |
|                                                                                                 | INT4 W4A8 | [0.5.12](../docker_images.md) | BW1000 | 8 | IFB | [**`>_`**](#glm-52-channel-int4-w4a8-ifb-bw1000-8x-sglang-0512) |

## 启动命令

### GLM-5.2-Channel-INT8-w8a8 IFB BW1100 8x SGLang 0.5.12

~~~bash
export SGLANG_ENABLE_SPEC_V2=1
export HSA_ENABLE_COREDUMP=1
export USE_DCU_CUSTOM_ALLREDUCE=0
export SGLANG_USE_AITER_AR=0
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export HIP_KERNEL_BATCH_CEILING=100
export GPU_FORCE_BLIT_COPY_SIZE=16
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_FP8_W8A8_MOE=0
export SGLANG_ROCM_USE_AITER_MOE=0
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export HIP_GRAPH_ACCUMULATE_DISPATCH=1
export HIP_GRAPH_USE_CMD_CACHE=0

sglang serve \
  --model-path hygon/GLM-5.2-Channel-INT8-w8a8 \
  --trust-remote-code \
  --tp-size 8 \
  --nsa-prefill-backend flashmla_auto \
  --nsa-decode-backend flashmla_kv \
  --dtype bfloat16 \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --page-size 64 \
  --kv-cache-dtype fp8_e4m3 \
  --chunked-prefill-size 16384 \
  --cuda-graph-max-bs 16 \
  --max-running-requests 32 \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --quantization slimquant_marlin \
  --mem-fraction-static 0.9
~~~

### GLM-5.2-Channel-INT8-w8a8 IFB BW1000 16x SGLang 0.5.12

#### Node 0

~~~bash
#!/bin/bash
set -e
export GLOO_SOCKET_IFNAME=ens61f1np1
export NCCL_SOCKET_IFNAME=ens61f1np1
export SGLANG_ENABLE_SPEC_V2=1
export HSA_ENABLE_COREDUMP=1
export USE_DCU_CUSTOM_ALLREDUCE=0
export SGLANG_USE_AITER_AR=0
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export HIP_KERNEL_BATCH_CEILING=100
export GPU_FORCE_BLIT_COPY_SIZE=16
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_FP8_W8A8_MOE=0
export SGLANG_ROCM_USE_AITER_MOE=0
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export HIP_GRAPH_ACCUMULATE_DISPATCH=1
export HIP_GRAPH_USE_CMD_CACHE=0

sglang serve \
  --model-path hygon/GLM-5.2-Channel-INT8-w8a8 \
  --trust-remote-code \
  --tp-size 16 \
  --pp-size 1 \
  --nnodes 2 \
  --node-rank 0 \
  --dist-init-addr <P_node0_ip>:<port0> \
  --host xxxxx \
  --port <port1>  \
  --nsa-prefill-backend flashmla_auto \
  --nsa-decode-backend flashmla_kv \
  --dtype bfloat16 \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --page-size 64 \
  --kv-cache-dtype fp8_e4m3 \
  --chunked-prefill-size 16384 \
  --cuda-graph-max-bs 64 \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --quantization slimquant_marlin \
  --mem-fraction-static 0.8
~~~

#### Node 1

~~~bash
#!/bin/bash
set -e
export GLOO_SOCKET_IFNAME=ens61f1np1
export NCCL_SOCKET_IFNAME=ens61f1np1
export SGLANG_ENABLE_SPEC_V2=1
export HSA_ENABLE_COREDUMP=1
export USE_DCU_CUSTOM_ALLREDUCE=0
export SGLANG_USE_AITER_AR=0
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export HIP_KERNEL_BATCH_CEILING=100
export GPU_FORCE_BLIT_COPY_SIZE=16
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_FP8_W8A8_MOE=0
export SGLANG_ROCM_USE_AITER_MOE=0
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export HIP_GRAPH_ACCUMULATE_DISPATCH=1
export HIP_GRAPH_USE_CMD_CACHE=0

sglang serve \
  --model-path hygon/GLM-5.2-Channel-INT8-w8a8 \
  --trust-remote-code \
  --tp-size 16 \
  --pp-size 1 \
  --nnodes 2 \
  --node-rank 1 \
  --dist-init-addr <P_node0_ip>:<port0> \
  --host xxxxx \
  --port <port1>  \
  --nsa-prefill-backend flashmla_auto \
  --nsa-decode-backend flashmla_kv \
  --dtype bfloat16 \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --page-size 64 \
  --kv-cache-dtype fp8_e4m3 \
  --chunked-prefill-size 16384 \
  --cuda-graph-max-bs 64 \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --quantization slimquant_marlin \
  --mem-fraction-static 0.8
~~~

### GLM-5.2-Channel-FP8-w8a8 IFB BW1100 8x SGLang 0.5.12 (tp8)

~~~bash
export SGLANG_ENABLE_SPEC_V2=1
export HSA_ENABLE_COREDUMP=1
export USE_DCU_CUSTOM_ALLREDUCE=0
export SGLANG_USE_AITER_AR=0
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export HIP_KERNEL_BATCH_CEILING=100
export GPU_FORCE_BLIT_COPY_SIZE=16
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_FP8_W8A8_MOE=1
export SGLANG_ROCM_USE_AITER_MOE=0
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export HIP_GRAPH_ACCUMULATE_DISPATCH=1
export HIP_GRAPH_USE_CMD_CACHE=0

sglang serve \
  --model-path hygon/GLM-5.2-Channel-FP8-w8a8 \
  --trust-remote-code \
  --tp-size 8 \
  --nsa-prefill-backend flashmla_auto \
  --nsa-decode-backend flashmla_kv \
  --dtype bfloat16 \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --page-size 64 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.8 \
  --chunked-prefill-size 8192 \
  --cuda-graph-max-bs 16 \
  --max-running-requests 16 \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4
~~~

### GLM-5.2-Channel-FP8-w8a8 IFB BW1100 8x SGLang 0.5.12 (tp8ep8)

~~~bash
export SGLANG_ENABLE_SPEC_V2=1
export HSA_ENABLE_COREDUMP=1
export USE_DCU_CUSTOM_ALLREDUCE=0
export SGLANG_USE_AITER_AR=0
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export HIP_KERNEL_BATCH_CEILING=100
export GPU_FORCE_BLIT_COPY_SIZE=16
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_FP8_W8A8_MOE=1
export SGLANG_ROCM_USE_AITER_MOE=0
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export HIP_GRAPH_ACCUMULATE_DISPATCH=1
export HIP_GRAPH_USE_CMD_CACHE=0
export ROCSHMEM_DISABLE_HDP_FLUSH=1
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_HEAP_SIZE=3173741824
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export SGLANG_USE_DEEPGEMM_MOE=1

sglang serve \
  --model-path hygon/GLM-5.2-Channel-FP8-w8a8 \
  --trust-remote-code \
  --tp-size 8 \
  --nsa-prefill-backend flashmla_auto \
  --nsa-decode-backend flashmla_kv \
  --dtype bfloat16 \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --page-size 64 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.8 \
  --chunked-prefill-size 8192 \
  --cuda-graph-max-bs 16 \
  --max-running-requests 16 \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --ep-size 8 \
  --moe-a2a-backend deepep \
  --moe-dense-tp-size 1 \
  --enable-dp-lm-head \
  --enable-dp-attention \
  --deepep-mode auto
~~~

### GLM-5.2-Channel-FP8-w8a8 PD scaleX40-3G 24x SGLang 0.5.12
#### DeepEP 配置

以下 `ep_config.json` 仅作为参考配置。使用时请将其保存为 `ep_config.json`

```json
{
  "normal_dispatch": {
    "num_sms": 48,
    "num_max_nvl_chunked_send_tokens": 6,
    "num_max_nvl_chunked_recv_tokens": 256,
    "num_max_rdma_chunked_send_tokens": 6,
    "num_max_rdma_chunked_recv_tokens": 128
  },
  "normal_combine": {
    "num_sms": 48,
    "num_max_nvl_chunked_send_tokens": 4,
    "num_max_nvl_chunked_recv_tokens": 256,
    "num_max_rdma_chunked_send_tokens": 6,
    "num_max_rdma_chunked_recv_tokens": 128
  }
}
```

#### EPLB 配置参考：[EPLB](../../optimization/static-eplb-sglang.md)

#### P node 0

~~~bash
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_TORCH_PROFILER_DIR=/workspace/profiling
export HSA_ENABLE_COREDUMP=1
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export HIP_KERNEL_BATCH_CEILING=100
export GPU_FORCE_BLIT_COPY_SIZE=16
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_FP8_W8A8_MOE=1
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export HIP_GRAPH_ACCUMULATE_DISPATCH=1
export HIP_GRAPH_USE_CMD_CACHE=0
export ROCSHMEM_DISABLE_HDP_FLUSH=1
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_HEAP_SIZE=3173741824
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export DEEP_EP_NORMAL_MNVL=1
export HIP_BUFFER_EXTRA_SIZE=0
export ROCSHMEM_GDR_DISABLE_XDP=1
export PYTHONUNBUFFERED=1
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1200
export ROCSHMEM_IPC_MNVL=1
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=em1
export GLOO_SOCKET_IFNAME=em1
export AMDGCN_USE_BUFFER_OPS=0
export ROCSHMEM_IB_GID_INDEX=0
export MC_IB_GID_INDEX=0
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH=1
export USE_DCU_CUSTOM_ALLREDUCE=0
export SGLANG_USE_AITER_AR=0
export SGLANG_USE_DEEPGEMM_MOE=1
export SGLANG_NSA_HCU_REUSE_SORTED_TOPK=1
export SGLANG_NSA_MQA_LOGITS_MEMORY_BUDGET_GB=2
export SGLANG_ENABLE_HCU_CONCAT_MLA_ABSORB_Q=1
export SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK=1
export SGLANG_LOGITS_PROCESSER_CHUNK_SIZE=2048

sglang serve \
  --model-path hygon/GLM-5.2-Channel-FP8-w8a8 \
  --trust-remote-code \
  --nsa-prefill-backend flashmla_auto \
  --nsa-decode-backend flashmla_kv \
  --context-length 524288 \
  --attn-cp-size 8 \
  --deepep-config /xxxxx/ep_config.json \
  --enable-nsa-prefill-context-parallel \
  --nsa-prefill-cp-mode round-robin-split \
  --disable-cuda-graph \
  --moe-a2a-backend deepep \
  --moe-dense-tp-size 1 \
  --enable-dp-lm-head \
  --enable-dp-attention \
  --deepep-mode normal \
  --init-expert-location /xxxxx/expert_distribution.pt  \ #见上述EPLB配置参考 
  --ep-dispatch-algorithm static \
  --ep-num-redundant-experts 16 \
  --disaggregation-ib-device shca_0,shca_1,shca_2,shca_4 \
  --disaggregation-mode prefill \
  --dist-init-addr <P_node0_ip>:<port0> \
  --nnodes 2 \
  --node-rank 0 \
  --host <P_node0_ip> \
  --port <port1> \
  --tp-size 8 \
  --pp-size 1 \
  --dp-size 1 \
  --ep-size 8 \
  --dtype bfloat16 \
  --dist-timeout 100000 \
  --watchdog-timeout 3600 \
  --page-size 64 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.8 \
  --chunked-prefill-size -1
~~~

#### P node 1

~~~bash
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_TORCH_PROFILER_DIR=/workspace/profiling
export HSA_ENABLE_COREDUMP=1
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export HIP_KERNEL_BATCH_CEILING=100
export GPU_FORCE_BLIT_COPY_SIZE=16
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_FP8_W8A8_MOE=1
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export HIP_GRAPH_ACCUMULATE_DISPATCH=1
export HIP_GRAPH_USE_CMD_CACHE=0
export ROCSHMEM_DISABLE_HDP_FLUSH=1
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_HEAP_SIZE=3173741824
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export DEEP_EP_NORMAL_MNVL=1
export HIP_BUFFER_EXTRA_SIZE=0
export ROCSHMEM_GDR_DISABLE_XDP=1
export PYTHONUNBUFFERED=1
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1200
export ROCSHMEM_IPC_MNVL=1
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=em1
export GLOO_SOCKET_IFNAME=em1
export AMDGCN_USE_BUFFER_OPS=0
export ROCSHMEM_IB_GID_INDEX=0
export MC_IB_GID_INDEX=0
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH=1
export USE_DCU_CUSTOM_ALLREDUCE=0
export SGLANG_USE_AITER_AR=0
export SGLANG_USE_DEEPGEMM_MOE=1
export SGLANG_NSA_HCU_REUSE_SORTED_TOPK=1
export SGLANG_NSA_MQA_LOGITS_MEMORY_BUDGET_GB=2
export SGLANG_ENABLE_HCU_CONCAT_MLA_ABSORB_Q=1
export SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK=1
export SGLANG_LOGITS_PROCESSER_CHUNK_SIZE=2048

sglang serve \
  --model-path hygon/GLM-5.2-Channel-FP8-w8a8 \
  --trust-remote-code \
  --nsa-prefill-backend flashmla_auto \
  --nsa-decode-backend flashmla_kv \
  --context-length 524288 \
  --attn-cp-size 8 \
  --deepep-config /xxxxx/ep_config.json \
  --enable-nsa-prefill-context-parallel \
  --nsa-prefill-cp-mode round-robin-split \
  --disable-cuda-graph \
  --moe-a2a-backend deepep \
  --moe-dense-tp-size 1 \
  --enable-dp-lm-head \
  --enable-dp-attention \
  --deepep-mode normal \
  --init-expert-location /xxxxx/expert_distribution.pt  \ #见上述EPLB配置参考 
  --ep-dispatch-algorithm static \
  --ep-num-redundant-experts 16 \
  --disaggregation-ib-device shca_0,shca_1,shca_2,shca_4 \
  --disaggregation-mode prefill \
  --dist-init-addr <P_node0_ip>:<port0> \
  --nnodes 2 \
  --node-rank 1 \
  --host <P_node1_ip> \
  --port <port1> \
  --tp-size 8 \
  --pp-size 1 \
  --dp-size 1 \
  --ep-size 8 \
  --dtype bfloat16 \
  --dist-timeout 100000 \
  --watchdog-timeout 3600 \
  --page-size 64 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.8 \
  --chunked-prefill-size -1
~~~

#### D node 0

~~~bash
export NCCL_IB_DISABLE=1
export HIP_BUFFER_EXTRA_SIZE=0
export ROCSHMEM_GDR_DISABLE_XDP=1
export SGLANG_ENABLE_SPEC_V2=1
export HSA_ENABLE_COREDUMP=1
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export HIP_KERNEL_BATCH_CEILING=100
export GPU_FORCE_BLIT_COPY_SIZE=16
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export SGLANG_USE_LIGHTOP=1
export SGLANG_NSA_FUSE_TOPK=1
export SGLANG_USE_DEEPGEMM_MOE=1
export SGLANG_USE_FP8_W8A8_MOE=1
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export HIP_GRAPH_ACCUMULATE_DISPATCH=1
export HIP_GRAPH_USE_CMD_CACHE=0
export ROCSHMEM_DISABLE_HDP_FLUSH=1
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_HEAP_SIZE=4737418240
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export ROCSHMEM_ALLOWED_IBV_DEVICES=shca_0,shca_1,shca_2,shca_4
export ROCSHMEM_IPC_MNVL=1
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export MC_TE_FILTERS=shca_0,shca_1,shca_2,shca_4
export MC_ALLOWED_IBV_DEVICES=shca_0,shca_1,shca_2,shca_4
export NCCL_SOCKET_IFNAME=em1
export GLOO_SOCKET_IFNAME=em1
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_USE_MODELSCOPE=1
export W8A8_SUPPORT_METHODS=3
export GPU_MAX_HW_QUEUES=3
export SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH=1
export SGLANG_SCHEDULER_SKIP_ALL_GATHER=1

sglang serve \
  --init-expert-location /xxxxx/expert_distribution.pt  \ #见上述EPLB配置参考 
  --ep-dispatch-algorithm static \
  --ep-num-redundant-experts 64 \
  --model-path hygon/GLM-5.2-Channel-FP8-w8a8 \
  --trust-remote-code \
  --host <D_node0_ip> \
  --port <port1> \
  --dist-init-addr <D_node0_ip>:<port0> \
  --nnodes 4 \
  --node-rank 0 \
  --tp-size 16 \
  --dp-size 16 \
  --ep-size 16 \
  --moe-dense-tp-size 1 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --deepep-mode low_latency \
  --enable-dp-lm-head \
  --nsa-prefill-backend flashmla_auto \
  --nsa-decode-backend flashmla_kv \
  --context-length 524288 \
  --dtype bfloat16 \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --page-size 64 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.8 \
  --chunked-prefill-size -1 \
  --quantization slimquant_marlin \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --cuda-graph-max-bs 32 \
  --max-running-requests 512 \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-bootstrap-port 8998 \
  --disaggregation-ib-device shca_0,shca_1,shca_2,shca_4
~~~

#### D node 1

~~~bash
export NCCL_IB_DISABLE=1
export HIP_BUFFER_EXTRA_SIZE=0
export ROCSHMEM_GDR_DISABLE_XDP=1
export SGLANG_ENABLE_SPEC_V2=1
export HSA_ENABLE_COREDUMP=1
export USE_DCU_CUSTOM_ALLREDUCE=1
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export HIP_KERNEL_BATCH_CEILING=100
export GPU_FORCE_BLIT_COPY_SIZE=16
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export SGLANG_USE_LIGHTOP=1
export SGLANG_NSA_FUSE_TOPK=1
export SGLANG_USE_DEEPGEMM_MOE=1
export SGLANG_USE_FP8_W8A8_MOE=1
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export HIP_GRAPH_ACCUMULATE_DISPATCH=1
export HIP_GRAPH_USE_CMD_CACHE=0
export ROCSHMEM_DISABLE_HDP_FLUSH=1
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_HEAP_SIZE=4737418240
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export ROCSHMEM_ALLOWED_IBV_DEVICES=shca_0,shca_1,shca_2,shca_4
export ROCSHMEM_IPC_MNVL=1
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export MC_TE_FILTERS=shca_0,shca_1,shca_2,shca_4
export MC_ALLOWED_IBV_DEVICES=shca_0,shca_1,shca_2,shca_4
export NCCL_SOCKET_IFNAME=em1
export GLOO_SOCKET_IFNAME=em1
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_USE_MODELSCOPE=1
export W8A8_SUPPORT_METHODS=3
export GPU_MAX_HW_QUEUES=3
export SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH=1
export SGLANG_SCHEDULER_SKIP_ALL_GATHER=1

sglang serve \
  --init-expert-location /xxxxx/expert_distribution.pt  \ #见上述EPLB配置参考 
  --ep-dispatch-algorithm static \
  --ep-num-redundant-experts 64 \
  --model-path hygon/GLM-5.2-Channel-FP8-w8a8 \
  --trust-remote-code \
  --host <D_node1_ip> \
  --port <port1> \
  --dist-init-addr <D_node0_ip>:<port0> \
  --nnodes 4 \
  --node-rank 1 \
  --tp-size 16 \
  --dp-size 16 \
  --ep-size 16 \
  --moe-dense-tp-size 1 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --deepep-mode low_latency \
  --enable-dp-lm-head \
  --nsa-prefill-backend flashmla_auto \
  --nsa-decode-backend flashmla_kv \
  --context-length 524288 \
  --dtype bfloat16 \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --page-size 64 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.8 \
  --chunked-prefill-size -1 \
  --quantization slimquant_marlin \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --cuda-graph-max-bs 32 \
  --max-running-requests 512 \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-bootstrap-port 8998 \
  --disaggregation-ib-device shca_0,shca_1,shca_2,shca_4
~~~

#### D node 2

~~~bash
export NCCL_IB_DISABLE=1
export HIP_BUFFER_EXTRA_SIZE=0
export ROCSHMEM_GDR_DISABLE_XDP=1
export SGLANG_ENABLE_SPEC_V2=1
export HSA_ENABLE_COREDUMP=1
export USE_DCU_CUSTOM_ALLREDUCE=1
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export HIP_KERNEL_BATCH_CEILING=100
export GPU_FORCE_BLIT_COPY_SIZE=16
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export SGLANG_USE_LIGHTOP=1
export SGLANG_NSA_FUSE_TOPK=1
export SGLANG_USE_DEEPGEMM_MOE=1
export SGLANG_USE_FP8_W8A8_MOE=1
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export HIP_GRAPH_ACCUMULATE_DISPATCH=1
export HIP_GRAPH_USE_CMD_CACHE=0
export ROCSHMEM_DISABLE_HDP_FLUSH=1
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_HEAP_SIZE=4737418240
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export ROCSHMEM_ALLOWED_IBV_DEVICES=shca_0,shca_1,shca_2,shca_4
export ROCSHMEM_IPC_MNVL=1
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export MC_TE_FILTERS=shca_0,shca_1,shca_2,shca_4
export MC_ALLOWED_IBV_DEVICES=shca_0,shca_1,shca_2,shca_4
export NCCL_SOCKET_IFNAME=em1
export GLOO_SOCKET_IFNAME=em1
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_USE_MODELSCOPE=1
export W8A8_SUPPORT_METHODS=3
export GPU_MAX_HW_QUEUES=3
export SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH=1
export SGLANG_SCHEDULER_SKIP_ALL_GATHER=1

sglang serve \
  --init-expert-location /xxxxx/expert_distribution.pt  \ #见上述EPLB配置参考 
  --ep-dispatch-algorithm static \
  --ep-num-redundant-experts 64 \
  --model-path hygon/GLM-5.2-Channel-FP8-w8a8 \
  --trust-remote-code \
  --host <D_node2_ip> \
  --port <port1> \
  --dist-init-addr <D_node0_ip>:<port0> \
  --nnodes 4 \
  --node-rank 2 \
  --tp-size 16 \
  --dp-size 16 \
  --ep-size 16 \
  --moe-dense-tp-size 1 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --deepep-mode low_latency \
  --enable-dp-lm-head \
  --nsa-prefill-backend flashmla_auto \
  --nsa-decode-backend flashmla_kv \
  --context-length 524288 \
  --dtype bfloat16 \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --page-size 64 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.8 \
  --chunked-prefill-size -1 \
  --quantization slimquant_marlin \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --cuda-graph-max-bs 32 \
  --max-running-requests 512 \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-bootstrap-port 8998 \
  --disaggregation-ib-device shca_0,shca_1,shca_2,shca_4
~~~

#### D node 3

~~~bash
export NCCL_IB_DISABLE=1
export HIP_BUFFER_EXTRA_SIZE=0
export ROCSHMEM_GDR_DISABLE_XDP=1
export SGLANG_ENABLE_SPEC_V2=1
export HSA_ENABLE_COREDUMP=1
export USE_DCU_CUSTOM_ALLREDUCE=1
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export HIP_KERNEL_BATCH_CEILING=100
export GPU_FORCE_BLIT_COPY_SIZE=16
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export SGLANG_USE_LIGHTOP=1
export SGLANG_NSA_FUSE_TOPK=1
export SGLANG_USE_DEEPGEMM_MOE=1
export SGLANG_USE_FP8_W8A8_MOE=1
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export HIP_GRAPH_ACCUMULATE_DISPATCH=1
export HIP_GRAPH_USE_CMD_CACHE=0
export ROCSHMEM_DISABLE_HDP_FLUSH=1
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_HEAP_SIZE=4737418240
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export ROCSHMEM_ALLOWED_IBV_DEVICES=shca_0,shca_1,shca_2,shca_4
export ROCSHMEM_IPC_MNVL=1
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export MC_TE_FILTERS=shca_0,shca_1,shca_2,shca_4
export MC_ALLOWED_IBV_DEVICES=shca_0,shca_1,shca_2,shca_4
export NCCL_SOCKET_IFNAME=em1
export GLOO_SOCKET_IFNAME=em1
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_USE_MODELSCOPE=1
export W8A8_SUPPORT_METHODS=3
export GPU_MAX_HW_QUEUES=3
export SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH=1
export SGLANG_SCHEDULER_SKIP_ALL_GATHER=1

sglang serve \
  --init-expert-location /xxxxx/expert_distribution.pt  \ #见上述EPLB配置参考 
  --ep-dispatch-algorithm static \
  --ep-num-redundant-experts 64 \
  --model-path hygon/GLM-5.2-Channel-FP8-w8a8 \
  --trust-remote-code \
  --host <D_node3_ip> \
  --port <port1> \
  --dist-init-addr <D_node0_ip>:<port0> \
  --nnodes 4 \
  --node-rank 3 \
  --tp-size 16 \
  --dp-size 16 \
  --ep-size 16 \
  --moe-dense-tp-size 1 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --deepep-mode low_latency \
  --enable-dp-lm-head \
  --nsa-prefill-backend flashmla_auto \
  --nsa-decode-backend flashmla_kv \
  --context-length 524288 \
  --dtype bfloat16 \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --page-size 64 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.8 \
  --chunked-prefill-size -1 \
  --quantization slimquant_marlin \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --cuda-graph-max-bs 32 \
  --max-running-requests 512 \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-bootstrap-port 8998 \
  --disaggregation-ib-device shca_0,shca_1,shca_2,shca_4
~~~

#### Router

~~~bash
python3 -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://prefill \
  --decode http://decode \
  --policy round_robin \
  --port 30001
~~~


### GLM-5.2-Channel-INT4-w4a8 2P4D BW1000 48x SGLang 0.5.18

使用 SGLang 0.5.18，共 6 台 BW1000 服务器，每台 8 卡，总计 48 卡。P 组为 2 节点 TP8+PP2，D 组为 4 节点 TP32+DP32+EP32。

将 `<P_node0_ip>`、`<P_node1_ip>` 和 `<D_node0_ip>` 至 `<D_node3_ip>` 替换为对应节点的实际 IP。`<port0>` 为分布式初始化端口，同组所有节点的 `--dist-init-addr` 均填写该组 node 0 的 IP 和相同端口；`<port1>` 为 P/D HTTP 服务端口，须与 Router 中对应地址一致；`<bootstrap_port>` 为 P 组 bootstrap 端口，两个 P 节点与 Router 的配置须一致。各服务端口应避免冲突。

通信网卡 `eno1`、IB 设备 `shca_0` 至 `shca_3` 以及 `LD_LIBRARY_PATH` 中的依赖库路径需按实际环境调整。网卡配置参考：[IB 网卡](../../troubleshooting/common-issues.md#ib网卡)。

先在对应服务器启动所有 P/D 节点，服务就绪后在 P node 0 上启动 Router。API 请求发送到 Router 的 30001 端口，`model` 使用所有节点统一设置的服务名 `glm`。

以下 `shca_topo.config` 仅供参考，PCI 地址、IB 网卡名称及映射编号需根据各节点实际硬件拓扑调整。将配置保存到各 D 节点，并将启动命令中的 `/xxxxx/shca_topo.config` 替换为实际文件的绝对路径。

```text
0000:09:00.0 shca_0 0
0000:36:00.0 shca_1 1
0000:55:00.0 shca_0 0
0000:77:00.0 shca_1 1
0000:85:00.0 shca_2 2
0000:b5:00.0 shca_3 3
0000:d5:00.0 shca_2 2
0000:f5:00.0 shca_3 3
```

#### P node 0

```bash
export PYTHONDONTWRITEBYTECODE=1
export SGLANG_OPT_USE_TOPK_V2=0
unset SGLANG_SIMULATE_ACC_LEN SGLANG_SIMULATE_ACC_METHOD
export PYTHONUNBUFFERED=1
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/mooncake:/usr/local/lib/python3.10/dist-packages/mooncake_transfer_engine_shca.libs:/opt/dtk/lib:/opt/dtk/hip/lib:${LD_LIBRARY_PATH:-}"
export SGLANG_SET_CPU_AFFINITY=1
export HIP_KERNEL_BATCH_CEILING=100
export GPU_MAX_HW_QUEUES=3
export SGLANG_USE_MODELSCOPE=1
export USE_DCU_CUSTOM_ALLREDUCE=0
export SGLANG_USE_AITER_AR=0
export HIP_H2D_DISABLE_COPY_BUFFER=0
export HIP_D2H_DISABLE_COPY_BUFFER=0
export HIP_H2D_DIRECT_COPY_THRESHOLD=32768
export HIP_H2D_HSAAPI_COPY_THRESHOLD=32768
export HIP_D2H_DIRECT_COPY_THRESHOLD=512
export HIP_D2H_HSAAPI_COPY_THRESHOLD=512
export HSA_KERNARG_POOL_SIZE=8388608
export HSA_FORCE_FINE_GRAIN_PCIE=1
export ROC_AQL_QUEUE_SIZE=131072
export NCCL_IB_HCA=shca_0,shca_1,shca_2,shca_3
export NCCL_SOCKET_IFNAME=eno1
export GLOO_SOCKET_IFNAME=eno1
unset NCCL_IB_GID_INDEX
export NCCL_NET_PLUGIN=shca
export NCCL_PLUGIN_P2P=ib
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export SGLANG_USE_LIGHTOP=1
export SGLANG_ROCM_USE_AITER_MOE=0
export W8A8_SUPPORT_METHODS=3
export HIP_GRAPH_ACCUMULATE_DISPATCH=1
export HIP_GRAPH_USE_CMD_CACHE=0
export SGLANG_DSA_HCU_INT8_INDEX_K_CACHE=1
export SGLANG_DEBUG_RADIX_CACHE_TIME=0
export SGLANG_UVICORN_WORKER_HEALTHCHECK_TIMEOUT=120
export SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1

sglang serve \
  --model-path hygon/GLM-5.2-Channel-INT4-w4a8 \
  --served-model-name glm \
  --trust-remote-code \
  --disaggregation-bootstrap-port "<bootstrap_port>" \
  --host "<P_node0_ip>" \
  --port "<port1>" \
  --dist-init-addr "<P_node0_ip>:<port0>" \
  --nnodes 2 \
  --node-rank 0 \
  --tp-size 8 \
  --pp-size 2 \
  --attn-cp-size 8 \
  --pp-max-micro-batch-size 2 \
  --enable-nsa-prefill-context-parallel \
  --nsa-prefill-cp-mode round-robin-split \
  --context-length 138240 \
  --kv-cache-dtype fp8_e4m3 \
  --dtype bfloat16 \
  --mem-fraction-static 0.75 \
  --chunked-prefill-size 16384 \
  --max-prefill-tokens 65536 \
  --page-size 64 \
  --nsa-prefill-backend flashmla_sparse \
  --nsa-decode-backend flashmla_kv \
  --quantization slimquant_w4a8_marlin \
  --disable-cuda-graph \
  --json-model-override-args '{"index_share_for_mtp_iteration": false}' \
  --disaggregation-ib-device shca_0,shca_1,shca_2,shca_3 \
  --tokenizer-worker-num 8 \
  --disaggregation-mode prefill
```

#### P node 1

```bash
export PYTHONDONTWRITEBYTECODE=1
export SGLANG_OPT_USE_TOPK_V2=0
unset SGLANG_SIMULATE_ACC_LEN SGLANG_SIMULATE_ACC_METHOD
export PYTHONUNBUFFERED=1
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/mooncake:/usr/local/lib/python3.10/dist-packages/mooncake_transfer_engine_shca.libs:/opt/dtk/lib:/opt/dtk/hip/lib:${LD_LIBRARY_PATH:-}"
export SGLANG_SET_CPU_AFFINITY=1
export HIP_KERNEL_BATCH_CEILING=100
export GPU_MAX_HW_QUEUES=3
export SGLANG_USE_MODELSCOPE=1
export USE_DCU_CUSTOM_ALLREDUCE=0
export SGLANG_USE_AITER_AR=0
export HIP_H2D_DISABLE_COPY_BUFFER=0
export HIP_D2H_DISABLE_COPY_BUFFER=0
export HIP_H2D_DIRECT_COPY_THRESHOLD=32768
export HIP_H2D_HSAAPI_COPY_THRESHOLD=32768
export HIP_D2H_DIRECT_COPY_THRESHOLD=512
export HIP_D2H_HSAAPI_COPY_THRESHOLD=512
export HSA_KERNARG_POOL_SIZE=8388608
export HSA_FORCE_FINE_GRAIN_PCIE=1
export ROC_AQL_QUEUE_SIZE=131072
export NCCL_IB_HCA=shca_0,shca_1,shca_2,shca_3
export NCCL_SOCKET_IFNAME=eno1
export GLOO_SOCKET_IFNAME=eno1
unset NCCL_IB_GID_INDEX
export NCCL_NET_PLUGIN=shca
export NCCL_PLUGIN_P2P=ib
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export SGLANG_USE_LIGHTOP=1
export SGLANG_ROCM_USE_AITER_MOE=0
export W8A8_SUPPORT_METHODS=3
export HIP_GRAPH_ACCUMULATE_DISPATCH=1
export HIP_GRAPH_USE_CMD_CACHE=0
export SGLANG_DSA_HCU_INT8_INDEX_K_CACHE=1
export SGLANG_DEBUG_RADIX_CACHE_TIME=0
export SGLANG_UVICORN_WORKER_HEALTHCHECK_TIMEOUT=120
export SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1

sglang serve \
  --model-path hygon/GLM-5.2-Channel-INT4-w4a8 \
  --served-model-name glm \
  --trust-remote-code \
  --disaggregation-bootstrap-port "<bootstrap_port>" \
  --host "<P_node1_ip>" \
  --port "<port1>" \
  --dist-init-addr "<P_node0_ip>:<port0>" \
  --nnodes 2 \
  --node-rank 1 \
  --tp-size 8 \
  --pp-size 2 \
  --attn-cp-size 8 \
  --pp-max-micro-batch-size 2 \
  --enable-nsa-prefill-context-parallel \
  --nsa-prefill-cp-mode round-robin-split \
  --context-length 138240 \
  --kv-cache-dtype fp8_e4m3 \
  --dtype bfloat16 \
  --mem-fraction-static 0.75 \
  --chunked-prefill-size 16384 \
  --max-prefill-tokens 65536 \
  --page-size 64 \
  --nsa-prefill-backend flashmla_sparse \
  --nsa-decode-backend flashmla_kv \
  --quantization slimquant_w4a8_marlin \
  --disable-cuda-graph \
  --json-model-override-args '{"index_share_for_mtp_iteration": false}' \
  --disaggregation-ib-device shca_0,shca_1,shca_2,shca_3 \
  --tokenizer-worker-num 8 \
  --disaggregation-mode prefill
```

#### D node 0

```bash
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=shca_0,shca_1,shca_2,shca_3
export NCCL_SOCKET_IFNAME=eno1
export GLOO_SOCKET_IFNAME=eno1
unset NCCL_IB_GID_INDEX
export NCCL_NET_PLUGIN=shca
export NCCL_PLUGIN_P2P=ib
unset NCCL_DEBUG
unset NCCL_DEBUG_SUBSYS
unset NCCL_DEBUG_FILE
unset RCCL_DEBUG
unset RCCL_DEBUG_SUBSYS
unset TORCH_NCCL_DEBUG
export TORCH_CPP_LOG_LEVEL=ERROR
export PYTHONWARNINGS="ignore"
export PYTHONDONTWRITEBYTECODE=1
export SGLANG_OPT_USE_TOPK_V2=0
unset SGLANG_SIMULATE_ACC_LEN SGLANG_SIMULATE_ACC_METHOD
export PYTHONUNBUFFERED=1
export SGLANG_ENABLE_SPEC_V2=1
export HSA_ENABLE_COREDUMP=1
export USE_DCU_CUSTOM_ALLREDUCE=0
export SGLANG_USE_AITER_AR=0
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/mooncake:/usr/local/lib/python3.10/dist-packages/mooncake_transfer_engine_shca.libs:/opt/dtk/lib:/opt/dtk/hip/lib:${LD_LIBRARY_PATH:-}"
export HIP_KERNEL_BATCH_CEILING=100
export GPU_FORCE_BLIT_COPY_SIZE=16
export HSA_KERNARG_POOL_SIZE=8388608
export HSA_FORCE_FINE_GRAIN_PCIE=1
export ROC_AQL_QUEUE_SIZE=131072
export SGLANG_USE_LIGHTOP=1
export SGLANG_ROCM_USE_AITER_MOE=0
export W8A8_SUPPORT_METHODS=3
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export HIP_GRAPH_ACCUMULATE_DISPATCH=1
export HIP_GRAPH_USE_CMD_CACHE=0
export SGLANG_USE_MODELSCOPE=1
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_HEAP_SIZE=419430400
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=32
export DEEPEP_ENABLE_LL_LAYERED_OPT=1
export ROCSHMEM_GDR_DISABLE_XDP=1
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_MAX_NUM_CONTEXTS=60
export ROCSHMEM_ALLOWED_IBV_DEVICES=shca_0,shca_1,shca_2,shca_3
export ROCSHMEM_TOPO_FILE_FORCE=/xxxxx/shca_topo.config
export HSA_USE_SVM=0
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export MC_ALLOWED_IBV_DEVICES=shca_0,shca_1,shca_2,shca_3
unset MC_GID_INDEX
export SGLANG_DEBUG_RADIX_CACHE_TIME=0
export SGLANG_DSA_HCU_INT8_INDEX_K_CACHE=1
export SGLANG_UVICORN_WORKER_HEALTHCHECK_TIMEOUT=120
export SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1

sglang serve \
  --model-path hygon/GLM-5.2-Channel-INT4-w4a8 \
  --trust-remote-code \
  --served-model-name glm \
  --host "<D_node0_ip>" \
  --port "<port1>" \
  --dist-init-addr "<D_node0_ip>:<port0>" \
  --nnodes 4 \
  --node-rank 0 \
  --tp-size 32 \
  --dcp-size 1 \
  --moe-dense-tp-size 1 \
  --dp-size 32 \
  --ep-size 32 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --enable-dp-lm-head \
  --deepep-mode low_latency \
  --page-size 64 \
  --nsa-prefill-backend flashmla_kv \
  --nsa-decode-backend flashmla_kv \
  --quantization slimquant_w4a8_marlin \
  --dtype bfloat16 \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.91 \
  --max-total-tokens 412736 \
  --chunked-prefill-size -1 \
  --cuda-graph-bs-decode 1 2 3 \
  --max-running-requests 128 \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --disaggregation-ib-device shca_0,shca_1,shca_2,shca_3 \
  --disaggregation-mode decode \
  --json-model-override-args '{"index_share_for_mtp_iteration": false}' \
  --tokenizer-worker-num 8
```

#### D node 1

```bash
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=shca_0,shca_1,shca_2,shca_3
export NCCL_SOCKET_IFNAME=eno1
export GLOO_SOCKET_IFNAME=eno1
unset NCCL_IB_GID_INDEX
export NCCL_NET_PLUGIN=shca
export NCCL_PLUGIN_P2P=ib
unset NCCL_DEBUG
unset NCCL_DEBUG_SUBSYS
unset NCCL_DEBUG_FILE
unset RCCL_DEBUG
unset RCCL_DEBUG_SUBSYS
unset TORCH_NCCL_DEBUG
export TORCH_CPP_LOG_LEVEL=ERROR
export PYTHONWARNINGS="ignore"
export PYTHONDONTWRITEBYTECODE=1
export SGLANG_OPT_USE_TOPK_V2=0
unset SGLANG_SIMULATE_ACC_LEN SGLANG_SIMULATE_ACC_METHOD
export PYTHONUNBUFFERED=1
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_DEBUG_RADIX_CACHE_TIME=1
export HSA_ENABLE_COREDUMP=1
export USE_DCU_CUSTOM_ALLREDUCE=0
export SGLANG_USE_AITER_AR=0
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/mooncake:/usr/local/lib/python3.10/dist-packages/mooncake_transfer_engine_shca.libs:/opt/dtk/lib:/opt/dtk/hip/lib:${LD_LIBRARY_PATH:-}"
export HIP_KERNEL_BATCH_CEILING=100
export GPU_FORCE_BLIT_COPY_SIZE=16
export HSA_KERNARG_POOL_SIZE=8388608
export HSA_FORCE_FINE_GRAIN_PCIE=1
export ROC_AQL_QUEUE_SIZE=131072
export SGLANG_USE_LIGHTOP=1
export SGLANG_ROCM_USE_AITER_MOE=0
export W8A8_SUPPORT_METHODS=3
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export HIP_GRAPH_ACCUMULATE_DISPATCH=1
export HIP_GRAPH_USE_CMD_CACHE=0
export SGLANG_USE_MODELSCOPE=1
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_HEAP_SIZE=419430400
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=32
export DEEPEP_ENABLE_LL_LAYERED_OPT=1
export ROCSHMEM_GDR_DISABLE_XDP=1
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_MAX_NUM_CONTEXTS=60
export ROCSHMEM_ALLOWED_IBV_DEVICES=shca_0,shca_1,shca_2,shca_3
export ROCSHMEM_TOPO_FILE_FORCE=/xxxxx/shca_topo.config
export HSA_USE_SVM=0
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export MC_ALLOWED_IBV_DEVICES=shca_0,shca_1,shca_2,shca_3
unset MC_GID_INDEX
unset SGLANG_SIMULATE_ACC_LEN
unset SGLANG_SIMULATE_ACC_METHOD
export SGLANG_DEBUG_RADIX_CACHE_TIME=0
export SGLANG_DSA_HCU_INT8_INDEX_K_CACHE=1
export SGLANG_UVICORN_WORKER_HEALTHCHECK_TIMEOUT=120
export SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1

sglang serve \
  --model-path hygon/GLM-5.2-Channel-INT4-w4a8 \
  --trust-remote-code \
  --served-model-name glm \
  --host "<D_node1_ip>" \
  --port "<port1>" \
  --dist-init-addr "<D_node0_ip>:<port0>" \
  --nnodes 4 \
  --node-rank 1 \
  --tp-size 32 \
  --dcp-size 1 \
  --moe-dense-tp-size 1 \
  --dp-size 32 \
  --ep-size 32 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --enable-dp-lm-head \
  --deepep-mode low_latency \
  --page-size 64 \
  --nsa-prefill-backend flashmla_kv \
  --nsa-decode-backend flashmla_kv \
  --quantization slimquant_w4a8_marlin \
  --dtype bfloat16 \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.91 \
  --max-total-tokens 412736 \
  --chunked-prefill-size -1 \
  --cuda-graph-bs-decode 1 2 3 \
  --max-running-requests 128 \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --disaggregation-ib-device shca_0,shca_1,shca_2,shca_3 \
  --disaggregation-mode decode \
  --json-model-override-args '{"index_share_for_mtp_iteration": false}' \
  --tokenizer-worker-num 8
```

#### D node 2

```bash
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=shca_0,shca_1,shca_2,shca_3
export NCCL_SOCKET_IFNAME=eno1
export GLOO_SOCKET_IFNAME=eno1
unset NCCL_IB_GID_INDEX
export NCCL_NET_PLUGIN=shca
export NCCL_PLUGIN_P2P=ib
unset NCCL_DEBUG
unset NCCL_DEBUG_SUBSYS
unset NCCL_DEBUG_FILE
unset RCCL_DEBUG
unset RCCL_DEBUG_SUBSYS
unset TORCH_NCCL_DEBUG
export TORCH_CPP_LOG_LEVEL=ERROR
export PYTHONWARNINGS="ignore"
export PYTHONDONTWRITEBYTECODE=1
export SGLANG_OPT_USE_TOPK_V2=0
unset SGLANG_SIMULATE_ACC_LEN SGLANG_SIMULATE_ACC_METHOD
export PYTHONUNBUFFERED=1
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_DEBUG_RADIX_CACHE_TIME=1
export HSA_ENABLE_COREDUMP=1
export USE_DCU_CUSTOM_ALLREDUCE=0
export SGLANG_USE_AITER_AR=0
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/mooncake:/usr/local/lib/python3.10/dist-packages/mooncake_transfer_engine_shca.libs:/opt/dtk/lib:/opt/dtk/hip/lib:${LD_LIBRARY_PATH:-}"
export HIP_KERNEL_BATCH_CEILING=100
export GPU_FORCE_BLIT_COPY_SIZE=16
export HSA_KERNARG_POOL_SIZE=8388608
export HSA_FORCE_FINE_GRAIN_PCIE=1
export ROC_AQL_QUEUE_SIZE=131072
export SGLANG_USE_LIGHTOP=1
export SGLANG_ROCM_USE_AITER_MOE=0
export W8A8_SUPPORT_METHODS=3
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export HIP_GRAPH_ACCUMULATE_DISPATCH=1
export HIP_GRAPH_USE_CMD_CACHE=0
export SGLANG_USE_MODELSCOPE=1
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_HEAP_SIZE=419430400
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=32
export DEEPEP_ENABLE_LL_LAYERED_OPT=1
export ROCSHMEM_GDR_DISABLE_XDP=1
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_MAX_NUM_CONTEXTS=60
export ROCSHMEM_ALLOWED_IBV_DEVICES=shca_0,shca_1,shca_2,shca_3
export ROCSHMEM_TOPO_FILE_FORCE=/xxxxx/shca_topo.config
export HSA_USE_SVM=0
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export MC_ALLOWED_IBV_DEVICES=shca_0,shca_1,shca_2,shca_3
unset MC_GID_INDEX
export SGLANG_DEBUG_RADIX_CACHE_TIME=0
export SGLANG_DSA_HCU_INT8_INDEX_K_CACHE=1
export SGLANG_UVICORN_WORKER_HEALTHCHECK_TIMEOUT=120
export SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1

sglang serve \
  --model-path hygon/GLM-5.2-Channel-INT4-w4a8 \
  --trust-remote-code \
  --served-model-name glm \
  --host "<D_node2_ip>" \
  --port "<port1>" \
  --dist-init-addr "<D_node0_ip>:<port0>" \
  --nnodes 4 \
  --node-rank 2 \
  --tp-size 32 \
  --dcp-size 1 \
  --moe-dense-tp-size 1 \
  --dp-size 32 \
  --ep-size 32 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --enable-dp-lm-head \
  --deepep-mode low_latency \
  --page-size 64 \
  --nsa-prefill-backend flashmla_kv \
  --nsa-decode-backend flashmla_kv \
  --quantization slimquant_w4a8_marlin \
  --dtype bfloat16 \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.91 \
  --max-total-tokens 412736 \
  --chunked-prefill-size -1 \
  --cuda-graph-bs-decode 1 2 3 \
  --max-running-requests 128 \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --disaggregation-ib-device shca_0,shca_1,shca_2,shca_3 \
  --disaggregation-mode decode \
  --json-model-override-args '{"index_share_for_mtp_iteration": false}' \
  --tokenizer-worker-num 8
```

#### D node 3

```bash
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=shca_0,shca_1,shca_2,shca_3
export NCCL_SOCKET_IFNAME=eno1
export GLOO_SOCKET_IFNAME=eno1
unset NCCL_IB_GID_INDEX
export NCCL_NET_PLUGIN=shca
export NCCL_PLUGIN_P2P=ib
unset NCCL_DEBUG
unset NCCL_DEBUG_SUBSYS
unset NCCL_DEBUG_FILE
unset RCCL_DEBUG
unset RCCL_DEBUG_SUBSYS
unset TORCH_NCCL_DEBUG
export TORCH_CPP_LOG_LEVEL=ERROR
export PYTHONWARNINGS="ignore"
export PYTHONDONTWRITEBYTECODE=1
export SGLANG_OPT_USE_TOPK_V2=0
unset SGLANG_SIMULATE_ACC_LEN SGLANG_SIMULATE_ACC_METHOD
export PYTHONUNBUFFERED=1
export SGLANG_ENABLE_SPEC_V2=1
export HSA_ENABLE_COREDUMP=1
export USE_DCU_CUSTOM_ALLREDUCE=0
export SGLANG_USE_AITER_AR=0
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/mooncake:/usr/local/lib/python3.10/dist-packages/mooncake_transfer_engine_shca.libs:/opt/dtk/lib:/opt/dtk/hip/lib:${LD_LIBRARY_PATH:-}"
export HIP_KERNEL_BATCH_CEILING=100
export GPU_FORCE_BLIT_COPY_SIZE=16
export HSA_KERNARG_POOL_SIZE=8388608
export HSA_FORCE_FINE_GRAIN_PCIE=1
export ROC_AQL_QUEUE_SIZE=131072
export SGLANG_USE_LIGHTOP=1
export SGLANG_ROCM_USE_AITER_MOE=0
export W8A8_SUPPORT_METHODS=3
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export HIP_GRAPH_ACCUMULATE_DISPATCH=1
export HIP_GRAPH_USE_CMD_CACHE=0
export SGLANG_USE_MODELSCOPE=1
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_HEAP_SIZE=419430400
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=32
export DEEPEP_ENABLE_LL_LAYERED_OPT=1
export ROCSHMEM_GDR_DISABLE_XDP=1
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_MAX_NUM_CONTEXTS=60
export ROCSHMEM_ALLOWED_IBV_DEVICES=shca_0,shca_1,shca_2,shca_3
export ROCSHMEM_TOPO_FILE_FORCE=/xxxxx/shca_topo.config
export HSA_USE_SVM=0
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export MC_ALLOWED_IBV_DEVICES=shca_0,shca_1,shca_2,shca_3
unset MC_GID_INDEX
export SGLANG_DEBUG_RADIX_CACHE_TIME=0
export SGLANG_DSA_HCU_INT8_INDEX_K_CACHE=1
export SGLANG_UVICORN_WORKER_HEALTHCHECK_TIMEOUT=120
export SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1

sglang serve \
  --model-path hygon/GLM-5.2-Channel-INT4-w4a8 \
  --trust-remote-code \
  --served-model-name glm \
  --host "<D_node3_ip>" \
  --port "<port1>" \
  --dist-init-addr "<D_node0_ip>:<port0>" \
  --nnodes 4 \
  --node-rank 3 \
  --tp-size 32 \
  --dcp-size 1 \
  --moe-dense-tp-size 1 \
  --dp-size 32 \
  --ep-size 32 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --enable-dp-lm-head \
  --deepep-mode low_latency \
  --page-size 64 \
  --nsa-prefill-backend flashmla_kv \
  --nsa-decode-backend flashmla_kv \
  --quantization slimquant_w4a8_marlin \
  --dtype bfloat16 \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.91 \
  --max-total-tokens 412736 \
  --chunked-prefill-size -1 \
  --cuda-graph-bs-decode 1 2 3 \
  --max-running-requests 128 \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --disaggregation-ib-device shca_0,shca_1,shca_2,shca_3 \
  --disaggregation-mode decode \
  --json-model-override-args '{"index_share_for_mtp_iteration": false}' \
  --tokenizer-worker-num 8
```

#### Router

```bash
sglang-router launch \
  --host "<P_node0_ip>" \
  --prometheus-port 29001 \
  --port 30001 \
  --pd-disaggregation \
  --prefill "http://<P_node0_ip>:<port1>" "<bootstrap_port>" \
  --decode "http://<D_node0_ip>:<port1>" \
  --health-check-timeout-secs 30 \
  --health-failure-threshold 10 \
  --cb-failure-threshold 50 \
  --cb-timeout-duration-secs 10 \
  --retry-max-retries 10 \
  --retry-initial-backoff-ms 200 \
  --retry-max-backoff-ms 5000 \
  --disable-health-check
```

#### API 验证

```bash
curl "http://<P_node0_ip>:30001/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model": "glm", "messages": [{"role": "user", "content": "你好"}], "max_tokens": 128}'
```

### GLM-5.2-Channel-INT4-w4a8 IFB BW1100 4x SGLang 0.5.12

~~~bash
export SGLANG_ENABLE_SPEC_V2=1
export HSA_ENABLE_COREDUMP=1
export USE_DCU_CUSTOM_ALLREDUCE=0
export SGLANG_USE_AITER_AR=0
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export HIP_KERNEL_BATCH_CEILING=100
export GPU_FORCE_BLIT_COPY_SIZE=16
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_FP8_W8A8_MOE=0
export SGLANG_ROCM_USE_AITER_MOE=0
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export HIP_GRAPH_ACCUMULATE_DISPATCH=1
export HIP_GRAPH_USE_CMD_CACHE=0
export SGLANG_W4A8_TPMOE_BACKEND=aiter

sglang serve \
  --model-path hygon/GLM-5.2-Channel-INT4-w4a8 \
  --trust-remote-code \
  --tp-size 4 \
  --nsa-prefill-backend flashmla_auto \
  --nsa-decode-backend flashmla_kv \
  --dtype bfloat16 \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --page-size 64 \
  --kv-cache-dtype fp8_e4m3 \
  --chunked-prefill-size 16384 \
  --cuda-graph-max-bs 16 \
  --max-running-requests 32 \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --quantization slimquant_w4a8_marlin \
  --mem-fraction-static 0.9
~~~

### GLM-5.2-Channel-INT4-w4a8 IFB BW1000 8x SGLang 0.5.12

~~~bash
export SGLANG_ENABLE_SPEC_V2=1
export HSA_ENABLE_COREDUMP=1
export USE_DCU_CUSTOM_ALLREDUCE=0
export SGLANG_USE_AITER_AR=0
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export HIP_KERNEL_BATCH_CEILING=100
export GPU_FORCE_BLIT_COPY_SIZE=16
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_FP8_W8A8_MOE=0
export SGLANG_ROCM_USE_AITER_MOE=0
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export HIP_GRAPH_ACCUMULATE_DISPATCH=1
export HIP_GRAPH_USE_CMD_CACHE=0
export SGLANG_W4A8_TPMOE_BACKEND=aiter

sglang serve \
  --model-path hygon/GLM-5.2-Channel-INT4-w4a8 \
  --trust-remote-code \
  --tp-size 8 \
  --nsa-prefill-backend flashmla_auto \
  --nsa-decode-backend flashmla_kv \
  --dtype bfloat16 \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --page-size 64 \
  --kv-cache-dtype fp8_e4m3 \
  --chunked-prefill-size 16384 \
  --cuda-graph-max-bs 16 \
  --max-running-requests 32 \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --quantization slimquant_w4a8_marlin \
  --mem-fraction-static 0.85
~~~

## API 调用

### IFB

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="hygon/GLM-5.2-Channel-INT8-w8a8",
    messages=[{"role": "user", "content": "你好"}],
    max_tokens=2048,
)
```

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "hygon/GLM-5.2-Channel-INT8-w8a8", "messages": [{"role": "user", "content": "你好"}], "max_tokens": 128}'
```

### PD 分离

PD 分离模式下，客户端请求发送到 SGLang Router。示例中 Router 端口为 `30001`。

```python
from openai import OpenAI

client = OpenAI(base_url="http://<router_ip>:30001/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="hygon/GLM-5.2-Channel-FP8-w8a8",
    messages=[{"role": "user", "content": "你好"}],
    max_tokens=2048,
)
```

```bash
curl http://<router_ip>:30001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "hygon/GLM-5.2-Channel-FP8-w8a8", "messages": [{"role": "user", "content": "你好"}], "max_tokens": 128}'
```
