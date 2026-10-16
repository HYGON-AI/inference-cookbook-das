# GLM-5.3

GLM-5.3 是智谱（Z.ai）推出的开放权重大语言模型，属于 GLM-5 系列，重点强化了编程、复杂推理和智能体任务能力。它采用 MoE（混合专家）架构，并与 GLM-5.2 使用相同基础模型，主要通过后训练提升效果，适合代码生成、调试、长流程任务规划和工具调用等场景。

## 模型列表

| 模型权重 | 量化方式 | SGLang 镜像 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | ----------- | -------- | ---- | -------- | -------- |
| [hygon/GLM-5.3-Channel-FP8-w8a8](https://www.modelscope.cn/models/hygon/GLM-5.3-Channel-FP8-w8a8) | FP8 W8A8 | 0.5.18 | scaleX40-3G | 8 | IFB | [**`>_`**](#glm-53-channel-fp8-w8a8-ifb-scalex40-3g-8x-sglang-0518) |
|                                                                                                 | FP8 W8A8 | 0.5.18 | scaleX40-3G | 24 | 1P1D | [**`>_`**](#glm-53-channel-fp8-w8a8-pd-scalex40-3g-24x-sglang-0518) |

| GLM-5.3-Flash（公开模型 ID TODO） | INT8 W8A8 | 0.5.18 | BW1000 | 24 | 1P1D | [**`>_`**](#glm-53-flash-channel-int8-w8a8-1p1d-bw1000-24x-sglang-0518) |

## 启动命令

### GLM-5.3-Channel-FP8-w8a8 IFB scaleX40-3G 8x SGLang 0.5.18

#### node 0

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3
export SGLANG_UVICORN_WORKER_HEALTHCHECK_TIMEOUT=120
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export LIGHTOP_SPARSE_MQA_GROUP6_ROWS_PER_CTA=3
export SGLANG_OPT_USE_TOPK_V2=0
export SGLANG_USE_LIGHTOP=1
export SGLANG_NSA_FUSE_TOPK=1
export SGLANG_DSA_HCU_LIGHTOP_MASK_TOPK=1
export SGLANG_ENABLE_HCU_CONCAT_MLA_ABSORB_Q=1
export SGLANG_USE_DEEPGEMM_MOE=1
export SGLANG_USE_FP8_W8A8_MOE=1
export NCCL_IB_DISABLE=1
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export HIP_BUFFER_EXTRA_SIZE=0
export HSA_ENABLE_COREDUMP=0
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export HIP_KERNEL_BATCH_CEILING=100
export GPU_FORCE_BLIT_COPY_SIZE=16
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export HIP_GRAPH_ACCUMULATE_DISPATCH=1
export HIP_GRAPH_USE_CMD_CACHE=1
export GPU_MAX_HW_QUEUES=3
export ROCSHMEM_GDR_DISABLE_XDP=1
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_HEAP_SIZE=4737418240
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1

sglang serve \
  --model-path hygon/GLM-5.3-Channel-FP8-w8a8 \
  --random-seed 536624121 \
  --trust-remote-code \
  --served-model-name GLM-5.3-Channel-FP8-w8a8 \
  --host "<ifb_node0_ip>" \
  --tokenizer-worker-num 16 \
  --dist-init-addr "<ifb_node0_ip>:5000" \
  --nnodes 2 \
  --node-rank 0 \
  --tp-size 8 \
  --dp-size 8 \
  --ep-size 8 \
  --moe-dense-tp-size 1 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --deepep-mode auto \
  --enable-dp-lm-head \
  --dsa-prefill-backend flashmla_sparse \
  --dsa-decode-backend flashmla_kv \
  --context-length 1048576 \
  --dtype bfloat16 \
  --dist-timeout 10000 \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --watchdog-timeout 3600 \
  --page-size 64 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.86 \
  --chunked-prefill-size 32768 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 5 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 6 \
  --cuda-graph-max-bs 8 \
  --max-running-requests 8 \
  --speculative-draft-lm-head-vp-size 8
```

#### node 1

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3
export SGLANG_UVICORN_WORKER_HEALTHCHECK_TIMEOUT=120
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export LIGHTOP_SPARSE_MQA_GROUP6_ROWS_PER_CTA=3
export SGLANG_OPT_USE_TOPK_V2=0
export SGLANG_USE_LIGHTOP=1
export SGLANG_NSA_FUSE_TOPK=1
export SGLANG_DSA_HCU_LIGHTOP_MASK_TOPK=1
export SGLANG_ENABLE_HCU_CONCAT_MLA_ABSORB_Q=1
export SGLANG_USE_DEEPGEMM_MOE=1
export SGLANG_USE_FP8_W8A8_MOE=1
export NCCL_IB_DISABLE=1
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export HIP_BUFFER_EXTRA_SIZE=0
export HSA_ENABLE_COREDUMP=0
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export HIP_KERNEL_BATCH_CEILING=100
export GPU_FORCE_BLIT_COPY_SIZE=16
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export HIP_GRAPH_ACCUMULATE_DISPATCH=1
export HIP_GRAPH_USE_CMD_CACHE=1
export GPU_MAX_HW_QUEUES=3
export ROCSHMEM_GDR_DISABLE_XDP=1
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_HEAP_SIZE=4737418240
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1

sglang serve \
  --model-path hygon/GLM-5.3-Channel-FP8-w8a8 \
  --random-seed 536624121 \
  --trust-remote-code \
  --served-model-name GLM-5.3-Channel-FP8-w8a8 \
  --host "<ifb_node1_ip>" \
  --tokenizer-worker-num 16 \
  --dist-init-addr "<ifb_node0_ip>:5000" \
  --nnodes 2 \
  --node-rank 1 \
  --tp-size 8 \
  --dp-size 8 \
  --ep-size 8 \
  --moe-dense-tp-size 1 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --deepep-mode auto \
  --enable-dp-lm-head \
  --dsa-prefill-backend flashmla_sparse \
  --dsa-decode-backend flashmla_kv \
  --context-length 1048576 \
  --dtype bfloat16 \
  --dist-timeout 10000 \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --watchdog-timeout 3600 \
  --page-size 64 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.86 \
  --chunked-prefill-size 32768 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 5 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 6 \
  --cuda-graph-max-bs 8 \
  --max-running-requests 8 \
  --speculative-draft-lm-head-vp-size 8
```

### GLM-5.3-Channel-FP8-w8a8 PD scaleX40-3G 24x SGLang 0.5.18

#### DeepEP 配置

以下 `ep_config.json` 为参考配置。将其保存为文件，并将 P 节点命令中的 `<deepep_config_path>` 替换为该文件的实际路径。

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

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3
export GLIBC_TUNABLES="glibc.rtld.optional_static_tls=0x40000"
export NCCL_SOCKET_IFNAME="<network_interface>"
export GLOO_SOCKET_IFNAME="<network_interface>"
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export SGLANG_USE_LIGHTOP=1
export SGLANG_NSA_FUSE_TOPK=1
export SGLANG_OPT_USE_TOPK_V2=0
export SGLANG_DSA_HCU_LIGHTOP_MASK_TOPK=1
export HSA_ENABLE_COREDUMP=0
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export HIP_KERNEL_BATCH_CEILING=100
export GPU_FORCE_BLIT_COPY_SIZE=16
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export HIP_GRAPH_ACCUMULATE_DISPATCH=1
export HIP_GRAPH_USE_CMD_CACHE=1
export SGLANG_ENABLE_HCU_CONCAT_MLA_ABSORB_Q=1
export SGLANG_NSA_MQA_LOGITS_MEMORY_BUDGET_GB=2
export SGLANG_USE_DEEPGEMM_MOE=1
export SGLANG_USE_FP8_W8A8_MOE=1
export DEEP_EP_NORMAL_MNVL=1
export ROCSHMEM_DISABLE_HDP_FLUSH=1
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_ALLOWED_IBV_DEVICES="<ibv_devices>"
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200

sglang serve \
  --model-path hygon/GLM-5.3-Channel-FP8-w8a8 \
  --trust-remote-code \
  --served-model-name GLM-5.3-Channel-FP8-w8a8 \
  --host "<p_node0_ip>" \
  --dist-init-addr "<p_node0_ip>:5000" \
  --nnodes 2 \
  --node-rank 0 \
  --tp-size 8 \
  --ep-size 8 \
  --attn-cp-size 8 \
  --moe-dense-tp-size 1 \
  --moe-a2a-backend deepep \
  --deepep-mode normal \
  --enable-prefill-cp \
  --cp-strategy interleave \
  --dsa-prefill-backend flashmla_sparse \
  --dsa-decode-backend flashmla_kv \
  --enable-dsa-cache-layer-split \
  --context-length 1048576 \
  --dtype bfloat16 \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --page-size 64 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.9 \
  --chunked-prefill-size 32768 \
  --max-prefill-tokens 32768 \
  --max-running-requests 512 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 5 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 6 \
  --disable-cuda-graph \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-bootstrap-port "<bootstrap_port>" \
  --disaggregation-ib-device "<ib_device>" \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --deepep-config "<deepep_config_path>"
```

#### P node 1

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3
export GLIBC_TUNABLES="glibc.rtld.optional_static_tls=0x40000"
export NCCL_SOCKET_IFNAME="<network_interface>"
export GLOO_SOCKET_IFNAME="<network_interface>"
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export SGLANG_USE_LIGHTOP=1
export SGLANG_NSA_FUSE_TOPK=1
export SGLANG_OPT_USE_TOPK_V2=0
export SGLANG_DSA_HCU_LIGHTOP_MASK_TOPK=1
export HSA_ENABLE_COREDUMP=0
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export HIP_KERNEL_BATCH_CEILING=100
export GPU_FORCE_BLIT_COPY_SIZE=16
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export HIP_GRAPH_ACCUMULATE_DISPATCH=1
export HIP_GRAPH_USE_CMD_CACHE=1
export SGLANG_ENABLE_HCU_CONCAT_MLA_ABSORB_Q=1
export SGLANG_NSA_MQA_LOGITS_MEMORY_BUDGET_GB=2
export SGLANG_USE_DEEPGEMM_MOE=1
export SGLANG_USE_FP8_W8A8_MOE=1
export DEEP_EP_NORMAL_MNVL=1
export ROCSHMEM_DISABLE_HDP_FLUSH=1
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_ALLOWED_IBV_DEVICES="<ibv_devices>"
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200

sglang serve \
  --model-path hygon/GLM-5.3-Channel-FP8-w8a8 \
  --trust-remote-code \
  --served-model-name GLM-5.3-Channel-FP8-w8a8 \
  --host "<p_node1_ip>" \
  --dist-init-addr "<p_node0_ip>:5000" \
  --nnodes 2 \
  --node-rank 1 \
  --tp-size 8 \
  --ep-size 8 \
  --attn-cp-size 8 \
  --moe-dense-tp-size 1 \
  --moe-a2a-backend deepep \
  --deepep-mode normal \
  --enable-prefill-cp \
  --cp-strategy interleave \
  --dsa-prefill-backend flashmla_sparse \
  --dsa-decode-backend flashmla_kv \
  --enable-dsa-cache-layer-split \
  --context-length 1048576 \
  --dtype bfloat16 \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --page-size 64 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.9 \
  --chunked-prefill-size 32768 \
  --max-prefill-tokens 32768 \
  --max-running-requests 512 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 5 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 6 \
  --disable-cuda-graph \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-bootstrap-port "<bootstrap_port>" \
  --disaggregation-ib-device "<ib_device>" \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --deepep-config "<deepep_config_path>"
```

#### D node 0

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3
export GLIBC_TUNABLES="glibc.rtld.optional_static_tls=0x40000"
export NCCL_SOCKET_IFNAME="<network_interface>"
export GLOO_SOCKET_IFNAME="<network_interface>"
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export SGLANG_USE_LIGHTOP=1
export SGLANG_NSA_FUSE_TOPK=1
export SGLANG_OPT_USE_TOPK_V2=0
export SGLANG_DSA_HCU_LIGHTOP_MASK_TOPK=1
export FLASH_MLA_SPARSE_DECODE_DISABLE_SPLITKV=0
export FLASH_MLA_SPARSE_PAGE64_SPECIALIZE=1
export FLASH_MLA_SPARSE_DECODE_NUM_SM_PARTS=32
export SGLANG_USE_DEEPGEMM_MOE=1
export SGLANG_USE_FP8_W8A8_MOE=1
export DEEP_EP_NORMAL_MNVL=1
export ROCSHMEM_DISABLE_HDP_FLUSH=1
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_ALLOWED_IBV_DEVICES="<ibv_devices>"
export HSA_ENABLE_COREDUMP=0
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export HIP_KERNEL_BATCH_CEILING=100
export GPU_FORCE_BLIT_COPY_SIZE=16
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export HIP_GRAPH_ACCUMULATE_DISPATCH=1
export HIP_GRAPH_USE_CMD_CACHE=1
export SGLANG_ENABLE_HCU_CONCAT_MLA_ABSORB_Q=1
export SGLANG_NSA_MQA_LOGITS_MEMORY_BUDGET_GB=2
export MC_GID_INDEX=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200

sglang serve \
  --model-path hygon/GLM-5.3-Channel-FP8-w8a8 \
  --trust-remote-code \
  --served-model-name GLM-5.3-Channel-FP8-w8a8 \
  --host "<d_node0_ip>" \
  --tokenizer-worker-num 16 \
  --dist-init-addr "<d_node0_ip>:5000" \
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
  --dsa-prefill-backend flashmla_auto \
  --dsa-decode-backend flashmla_kv \
  --context-length 1048576 \
  --dtype bfloat16 \
  --page-size 64 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.92 \
  --chunked-prefill-size -1 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 5 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 6 \
  --cuda-graph-max-bs 16 \
  --max-running-requests 256 \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-ib-device "<ib_device>" \
  --speculative-draft-lm-head-vp-size 16 \
  --reasoning-parser glm45 \
  --tool-call-parser glm47
```

#### D node 1

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3
export GLIBC_TUNABLES="glibc.rtld.optional_static_tls=0x40000"
export NCCL_SOCKET_IFNAME="<network_interface>"
export GLOO_SOCKET_IFNAME="<network_interface>"
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export SGLANG_USE_LIGHTOP=1
export SGLANG_NSA_FUSE_TOPK=1
export SGLANG_OPT_USE_TOPK_V2=0
export SGLANG_DSA_HCU_LIGHTOP_MASK_TOPK=1
export FLASH_MLA_SPARSE_DECODE_DISABLE_SPLITKV=0
export FLASH_MLA_SPARSE_PAGE64_SPECIALIZE=1
export FLASH_MLA_SPARSE_DECODE_NUM_SM_PARTS=32
export SGLANG_USE_DEEPGEMM_MOE=1
export SGLANG_USE_FP8_W8A8_MOE=1
export DEEP_EP_NORMAL_MNVL=1
export ROCSHMEM_DISABLE_HDP_FLUSH=1
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_ALLOWED_IBV_DEVICES="<ibv_devices>"
export HSA_ENABLE_COREDUMP=0
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export HIP_KERNEL_BATCH_CEILING=100
export GPU_FORCE_BLIT_COPY_SIZE=16
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export HIP_GRAPH_ACCUMULATE_DISPATCH=1
export HIP_GRAPH_USE_CMD_CACHE=1
export SGLANG_ENABLE_HCU_CONCAT_MLA_ABSORB_Q=1
export SGLANG_NSA_MQA_LOGITS_MEMORY_BUDGET_GB=2
export MC_GID_INDEX=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200

sglang serve \
  --model-path hygon/GLM-5.3-Channel-FP8-w8a8 \
  --trust-remote-code \
  --served-model-name GLM-5.3-Channel-FP8-w8a8 \
  --host "<d_node1_ip>" \
  --tokenizer-worker-num 16 \
  --dist-init-addr "<d_node0_ip>:5000" \
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
  --dsa-prefill-backend flashmla_auto \
  --dsa-decode-backend flashmla_kv \
  --context-length 1048576 \
  --dtype bfloat16 \
  --page-size 64 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.92 \
  --chunked-prefill-size -1 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 5 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 6 \
  --cuda-graph-max-bs 16 \
  --max-running-requests 256 \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-ib-device "<ib_device>" \
  --speculative-draft-lm-head-vp-size 16 \
  --reasoning-parser glm45 \
  --tool-call-parser glm47
```

#### D node 2

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3
export GLIBC_TUNABLES="glibc.rtld.optional_static_tls=0x40000"
export NCCL_SOCKET_IFNAME="<network_interface>"
export GLOO_SOCKET_IFNAME="<network_interface>"
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export SGLANG_USE_LIGHTOP=1
export SGLANG_NSA_FUSE_TOPK=1
export SGLANG_OPT_USE_TOPK_V2=0
export SGLANG_DSA_HCU_LIGHTOP_MASK_TOPK=1
export FLASH_MLA_SPARSE_DECODE_DISABLE_SPLITKV=0
export FLASH_MLA_SPARSE_PAGE64_SPECIALIZE=1
export FLASH_MLA_SPARSE_DECODE_NUM_SM_PARTS=32
export SGLANG_USE_DEEPGEMM_MOE=1
export SGLANG_USE_FP8_W8A8_MOE=1
export DEEP_EP_NORMAL_MNVL=1
export ROCSHMEM_DISABLE_HDP_FLUSH=1
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_ALLOWED_IBV_DEVICES="<ibv_devices>"
export HSA_ENABLE_COREDUMP=0
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export HIP_KERNEL_BATCH_CEILING=100
export GPU_FORCE_BLIT_COPY_SIZE=16
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export HIP_GRAPH_ACCUMULATE_DISPATCH=1
export HIP_GRAPH_USE_CMD_CACHE=1
export SGLANG_ENABLE_HCU_CONCAT_MLA_ABSORB_Q=1
export SGLANG_NSA_MQA_LOGITS_MEMORY_BUDGET_GB=2
export MC_GID_INDEX=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200

sglang serve \
  --model-path hygon/GLM-5.3-Channel-FP8-w8a8 \
  --trust-remote-code \
  --served-model-name GLM-5.3-Channel-FP8-w8a8 \
  --host "<d_node2_ip>" \
  --tokenizer-worker-num 16 \
  --dist-init-addr "<d_node0_ip>:5000" \
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
  --dsa-prefill-backend flashmla_auto \
  --dsa-decode-backend flashmla_kv \
  --context-length 1048576 \
  --dtype bfloat16 \
  --page-size 64 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.92 \
  --chunked-prefill-size -1 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 5 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 6 \
  --cuda-graph-max-bs 16 \
  --max-running-requests 256 \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-ib-device "<ib_device>" \
  --speculative-draft-lm-head-vp-size 16 \
  --reasoning-parser glm45 \
  --tool-call-parser glm47
```

#### D node 3

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3
export GLIBC_TUNABLES="glibc.rtld.optional_static_tls=0x40000"
export NCCL_SOCKET_IFNAME="<network_interface>"
export GLOO_SOCKET_IFNAME="<network_interface>"
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export SGLANG_USE_LIGHTOP=1
export SGLANG_NSA_FUSE_TOPK=1
export SGLANG_OPT_USE_TOPK_V2=0
export SGLANG_DSA_HCU_LIGHTOP_MASK_TOPK=1
export FLASH_MLA_SPARSE_DECODE_DISABLE_SPLITKV=0
export FLASH_MLA_SPARSE_PAGE64_SPECIALIZE=1
export FLASH_MLA_SPARSE_DECODE_NUM_SM_PARTS=32
export SGLANG_USE_DEEPGEMM_MOE=1
export SGLANG_USE_FP8_W8A8_MOE=1
export DEEP_EP_NORMAL_MNVL=1
export ROCSHMEM_DISABLE_HDP_FLUSH=1
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_ALLOWED_IBV_DEVICES="<ibv_devices>"
export HSA_ENABLE_COREDUMP=0
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export HIP_KERNEL_BATCH_CEILING=100
export GPU_FORCE_BLIT_COPY_SIZE=16
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export HIP_GRAPH_ACCUMULATE_DISPATCH=1
export HIP_GRAPH_USE_CMD_CACHE=1
export SGLANG_ENABLE_HCU_CONCAT_MLA_ABSORB_Q=1
export SGLANG_NSA_MQA_LOGITS_MEMORY_BUDGET_GB=2
export MC_GID_INDEX=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200

sglang serve \
  --model-path hygon/GLM-5.3-Channel-FP8-w8a8 \
  --trust-remote-code \
  --served-model-name GLM-5.3-Channel-FP8-w8a8 \
  --host "<d_node3_ip>" \
  --tokenizer-worker-num 16 \
  --dist-init-addr "<d_node0_ip>:5000" \
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
  --dsa-prefill-backend flashmla_auto \
  --dsa-decode-backend flashmla_kv \
  --context-length 1048576 \
  --dtype bfloat16 \
  --page-size 64 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.92 \
  --chunked-prefill-size -1 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 5 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 6 \
  --cuda-graph-max-bs 16 \
  --max-running-requests 256 \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-ib-device "<ib_device>" \
  --speculative-draft-lm-head-vp-size 16 \
  --reasoning-parser glm45 \
  --tool-call-parser glm47
```

#### Router

Router 连接 P node 0、D node 0 的默认服务端口 `30000`，自身监听 `30001`。将 `<p_node0_ip>`、`<d_node0_ip>` 替换为对应节点实际 IP。

```bash
python3 -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill "http://<p_node0_ip>:30000" \
  --decode "http://<d_node0_ip>:30000" \
  --policy round_robin \
  --port 30001
```

### GLM-5.3-Flash-Channel-INT8-w8a8 1P1D BW1000 24x SGLang 0.5.18

本节整理 GLM-5.3-Flash 的 PD 分离启动配置：P 使用单机 8 卡，D 使用两机共 16 卡（一个分布式 Decode 实例），合计 24 张 BW1000。Router 与 P 同机运行。

#### 部署前准备

当前为待完善配置，尚未完成硬件运行验证。完成以下 TODO 后再执行启动命令：

- TODO：确认量化权重版本。来源正文标记 v3，原配置默认 v1；所有 P/D 节点必须使用同一版本。
- TODO：补充 Hygon 官方 channelwise 量化模型的公开 ModelScope ID，并替换所有 `<TODO_MODEL_ID>`。
- TODO：补充可公开拉取的 SGLang 0.5.18 / DTK 26.04 镜像。来源使用专项测试镜像；通用 0.5.18 镜像的参数兼容性尚待验证。
- TODO：提供适配 BW1000 节点的 `topo.config`，并将 `<topo_config_path>` 替换为各节点容器内的实际路径。

将 `<P_node_ip>`、`<D_node0_ip>`、`<D_node1_ip>` 替换为对应节点互通的 IP。三台服务器各使用 8 张卡，容器工作目录应为非根目录。以下配置使用 `ib0` 与 `shca_0,shca_1,shca_2,shca_3`；执行前确认与实际 RDMA 网络一致。确保节点之间的服务端口 `30000`、分布式通信端口 `5000` 和 bootstrap 端口 `8998` 可达，客户端可访问 Router 的 `30001`。

将下面的配置保存为 `ep_config.json`，并替换 P 命令中的 `<deepep_config_path>`：

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

#### P node

```bash
export SGLANG_ENABLE_SPEC_V2=1
unset SGLANG_SIMULATE_ACC_LEN SGLANG_SIMULATE_ACC_METHOD SGLANG_SIMULATED_EXPERT_BALANCE
export SGLANG_HCU_SPEC_ASYNC_SCHEDULING=1
ulimit -s 67108864
export ROCSHMEM_TOPO_FILE_FORCE="<topo_config_path>"
export HSA_ENABLE_COREDUMP=1
export USE_DCU_CUSTOM_ALLREDUCE=0
export SGLANG_USE_AITER_AR=0
export SGLANG_CUSTOM_ALLREDUCE_INPUT_FENCE=thread0
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export HIP_KERNEL_BATCH_CEILING=100
export GPU_FORCE_BLIT_COPY_SIZE=16
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export SGLANG_USE_LIGHTOP=1
export SGLANG_ROCM_USE_AITER_MOE=0
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export HIP_GRAPH_ACCUMULATE_DISPATCH=1
export HIP_GRAPH_USE_CMD_CACHE=0
export SGLANG_ROCM_USE_AITER_TILELANG_MHC=1
export SGLANG_OPT_USE_TILELANG_MHC_PRE=1
export SGLANG_OPT_USE_TILELANG_MHC_POST=1
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
export SGLANG_OPT_SWIGLU_CLAMP_FUSION=0
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_MAX_NUM_CONTEXTS=60
export ROCSHMEM_ALLOWED_IBV_DEVICES=shca_0,shca_1,shca_2,shca_3
export SGLANG_USE_DEEPGEMM_MOE=1
export SGLANG_INT8_DEEPGEMM_ASM=1
export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=1
export W8A8_SUPPORT_METHODS=3
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_REUSE_W8A8_INT8_EP_MOE_WORKSPACE=0
export SGLANG_DSA_HCU_MQA_LOGITS_WORKSPACE_GB=1
export SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK=1
export SGLANG_LOGITS_PROCESSER_CHUNK_SIZE=1024
export SGLANG_USE_AITER_CHUNK_GATED_DELTA_H_HIP=1
export SGLANG_USE_FUSED_RMS_QUANT=1
export SGLANG_USE_FUSED_SILU_MUL_QUANT=1
export SGLANG_USE_LIGHTOP_PREFILL_DEQUANT=1
export SGLANG_USE_FUSED_SILU_MUL_CLAMP_QUANT=1
export SGLANG_DSA_KPOOL_AITER_TOPK=1
export SGLANG_KERNEL_KPOOL_TOPK_SORT_WRITEBACK=1
export SGLANG_USE_HICACHE_OPTIMIZATION_KERNEL=1
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export SGLANG_SCHEDULER_MAX_RECV_PER_POLL=8
export SGLANG_UVICORN_WORKER_STARTUP_TIMEOUT=1800
export SGLANG_UVICORN_WORKER_HEALTHCHECK_TIMEOUT=600
export SGLANG_PYSPY_DUMP_BEFORE_CRASH=0
unset SGLANG_SIMULATE_ACC_LEN SGLANG_SIMULATE_ACC_METHOD SGLANG_SIMULATED_EXPERT_BALANCE
unset PYTHONPATH
export NCCL_SOCKET_IFNAME=ib0
export GLOO_SOCKET_IFNAME=ib0
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
sglang serve \
    --model-path "<TODO_MODEL_ID>" \
    --served-model-name GLM-5.3-Flash \
    --trust-remote-code \
    --host "<P_node_ip>" \
    --random-seed 42 \
    --context-length 1048576 \
    --chunked-prefill-size 32768 \
    --max-prefill-tokens 32768 \
    --disable-shared-experts-fusion \
    --cuda-graph-backend-prefill disabled \
    --disable-chunked-prefix-cache \
    --disaggregation-bootstrap-port 8998 \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend mooncake \
    --disaggregation-ib-device shca_0,shca_1,shca_2,shca_3 \
    --mem-fraction-static 0.85 \
    --mamba-full-memory-ratio 0.9 \
    --max-running-requests 16 \
    --max-mamba-cache-size 96 \
    --mamba-radix-cache-strategy extra_buffer \
    --page-size 64 \
    --cuda-graph-backend-decode disabled \
    --tp-size 8 \
    --ep-size 8 \
    --attn-cp-size 8 \
    --enable-prefill-cp \
    --cp-strategy interleave \
    --moe-a2a-backend deepep \
    --deepep-mode normal \
    --attention-backend dsa \
    --dsa-prefill-backend flashmla_sparse \
    --dsa-decode-backend flashmla_kv \
    --kv-cache-dtype fp8_e4m3 \
    --quantization w8a8_int8 \
    --enable-dsa-cache-layer-split \
    --mla-kv-prefetch-ring-size 1 \
    --numa-node 0 3 2 1 4 7 6 5 \
    --enable-hierarchical-cache \
    --hicache-size 50 \
    --hicache-write-policy write_through \
    --hicache-io-backend kernel \
    --hicache-mem-layout layer_first \
    --reasoning-parser glm5 \
    --tool-call-parser glm5stream \
    --glm-decoding-constraint-module=sglang.srt.constrained.glm \
    --glm-ignore-decoding-constraint-exception \
    --grammar-backend xgrammar \
    --glm-special-token-escape-seed=42 \
    --glm-adaptive-max-tokens \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 5 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 6 \
    --enable-cache-report \
    --enable-metrics \
    --tokenizer-worker-num=8 \
    --json-model-override-args '{"index_share_for_mtp_iteration": true}' \
    --deepep-config "<deepep_config_path>"
```

#### D node 0

```bash
export PYTHONPYCACHEPREFIX=/tmp/glm53-pycache-fixed4
export SGLANG_HCU_SPEC_ASYNC_SCHEDULING=1
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_NSA_FUSE_TOPK=1
export ROCSHMEM_TOPO_FILE_FORCE="<topo_config_path>"
export SGLANG_DSA_FUSE_TOPK=1
unset SGLANG_KERNEL_API_LOGLEVEL
unset SGLANG_KERNEL_API_LOGDEST
export HSA_ENABLE_COREDUMP=1
export USE_DCU_CUSTOM_ALLREDUCE=0
export SGLANG_USE_AITER_AR=0
export SGLANG_CUSTOM_ALLREDUCE_INPUT_FENCE=thread0
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export HIP_KERNEL_BATCH_CEILING=100
export GPU_FORCE_BLIT_COPY_SIZE=16
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export SGLANG_USE_LIGHTOP=1
export SGLANG_ROCM_USE_AITER_MOE=0
export W8A8_SUPPORT_METHODS=3
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export HIP_GRAPH_ACCUMULATE_DISPATCH=0
export HIP_GRAPH_USE_CMD_CACHE=0
export SGLANG_ROCM_USE_AITER_TILELANG_MHC=1
export SGLANG_OPT_USE_TILELANG_MHC_PRE=1
export SGLANG_OPT_USE_TILELANG_MHC_POST=1
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
unset PRINT_MOE_ARGS
export SGLANG_OPT_SWIGLU_CLAMP_FUSION=0
unset ROCSHMEM_DISABLE_HDP_FLUSH
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_MAX_NUM_CONTEXTS=96
export ROCSHMEM_HEAP_SIZE=1073741824
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=96
export ROCSHMEM_ALLOWED_IBV_DEVICES=shca_0,shca_1,shca_2,shca_3
export ROCSHMEM_IB_GID_INDEX=0
export DEEPEP_ENABLE_LL_LAYERED_OPT=1
export SGLANG_USE_DEEPGEMM_MOE=1
export SGLANG_INT8_DEEPGEMM_ASM=1
export SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH=1
export SGLANG_SCHEDULER_SKIP_ALL_GATHER=1
export SGLANG_DSA_ENABLE_MTP_PRECOMPUTE_METADATA=0
export SGLANG_USE_LEGACY_FUSED_RMS_QUANT=0
export SGLANG_USE_FUSED_RMS_QUANT=1
export SGLANG_USE_FUSED_SILU_MUL_QUANT=1
export SGLANG_DSA_KPOOL_LIGHTOP_TOPK=1
export W8A8_SUPPORT_METHODS=3
export SGLANG_USE_LIGHTOP_PREFILL_DEQUANT=1
export SGLANG_REUSE_W8A8_INT8_EP_MOE_WORKSPACE=65536
export SGLANG_DSA_HCU_USE_LIGHTOP_DECODE_GATHER=1
export SGLANG_USE_AITER_CHUNK_GATED_DELTA_H_HIP=1
export SGLANG_USE_FUSED_SILU_MUL_CLAMP_QUANT=1
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export TOKENIZERS_PARALLELISM=false
export RAYON_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export SGLANG_PYSPY_DUMP_BEFORE_CRASH=0
export HSA_USE_SVM=0
unset SGLANG_SIMULATE_ACC_LEN SGLANG_SIMULATE_ACC_METHOD SGLANG_SIMULATED_EXPERT_BALANCE
unset PYTHONPATH
export NCCL_SOCKET_IFNAME=ib0
export GLOO_SOCKET_IFNAME=ib0
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
sglang serve \
    --model-path "<TODO_MODEL_ID>" \
    --served-model-name GLM-5.3-Flash \
    --trust-remote-code \
    --random-seed 648620208 \
    --nnodes 2 \
    --node-rank 0 \
    --host "<D_node0_ip>" \
    --dist-init-addr "<D_node0_ip>":5000 \
    --max-running-requests 128 \
    --disable-shared-experts-fusion \
    --disable-piecewise-cuda-graph \
    --disable-chunked-prefix-cache \
    --disable-radix-cache \
    --cuda-graph-bs 1 2 3 4 5 6 7 8 \
    --mem-fraction-static 0.80 \
    --context-length 1048576 \
    --page-size 64 \
    --dtype bfloat16 \
    --max-mamba-cache-size 320 \
    --mamba-scheduler-strategy no_buffer \
    --tp-size 16 \
    --dp-size 16 \
    --ep-size 16 \
    --moe-dense-tp-size 1 \
    --enable-dp-attention \
    --enable-dp-lm-head \
    --moe-a2a-backend deepep \
    --deepep-mode low_latency \
    --attention-backend nsa \
    --nsa-prefill-backend flashmla_auto \
    --nsa-decode-backend flashmla_kv \
    --linear-attn-backend triton \
    --kv-cache-dtype fp8_e4m3 \
    --quantization w8a8_int8 \
    --disaggregation-transfer-backend mooncake \
    --disaggregation-ib-device shca_0,shca_1,shca_2,shca_3 \
    --disaggregation-mode decode \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 5 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 6 \
    --reasoning-parser glm45 \
    --tool-call-parser glm47 \
    --grammar-backend xgrammar \
    --glm-adaptive-max-tokens \
    --enable-cache-report \
    --enable-metrics \
    --tokenizer-worker-num=16 \
    --json-model-override-args '{"index_share_for_mtp_iteration": true}' \
    --numa-node 0 3 2 1 4 7 6 5 \
    --enable-kda-replayssm-spec
```

#### D node 1

```bash
export PYTHONPYCACHEPREFIX=/tmp/glm53-pycache-fixed4
export SGLANG_HCU_SPEC_ASYNC_SCHEDULING=1
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_NSA_FUSE_TOPK=1
export ROCSHMEM_TOPO_FILE_FORCE="<topo_config_path>"
export SGLANG_DSA_FUSE_TOPK=1
unset SGLANG_KERNEL_API_LOGLEVEL
unset SGLANG_KERNEL_API_LOGDEST
export HSA_ENABLE_COREDUMP=1
export USE_DCU_CUSTOM_ALLREDUCE=0
export SGLANG_USE_AITER_AR=0
export SGLANG_CUSTOM_ALLREDUCE_INPUT_FENCE=thread0
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export HIP_KERNEL_BATCH_CEILING=100
export GPU_FORCE_BLIT_COPY_SIZE=16
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export SGLANG_USE_LIGHTOP=1
export SGLANG_ROCM_USE_AITER_MOE=0
export W8A8_SUPPORT_METHODS=3
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export HIP_GRAPH_ACCUMULATE_DISPATCH=0
export HIP_GRAPH_USE_CMD_CACHE=0
export SGLANG_ROCM_USE_AITER_TILELANG_MHC=1
export SGLANG_OPT_USE_TILELANG_MHC_PRE=1
export SGLANG_OPT_USE_TILELANG_MHC_POST=1
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
unset PRINT_MOE_ARGS
export SGLANG_OPT_SWIGLU_CLAMP_FUSION=0
unset ROCSHMEM_DISABLE_HDP_FLUSH
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_MAX_NUM_CONTEXTS=96
export ROCSHMEM_HEAP_SIZE=1073741824
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=96
export ROCSHMEM_ALLOWED_IBV_DEVICES=shca_0,shca_1,shca_2,shca_3
export DEEPEP_ENABLE_LL_LAYERED_OPT=1
export ROCSHMEM_IB_GID_INDEX=0
export SGLANG_USE_DEEPGEMM_MOE=1
export SGLANG_INT8_DEEPGEMM_ASM=1
export SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH=1
export SGLANG_SCHEDULER_SKIP_ALL_GATHER=1
export SGLANG_DSA_ENABLE_MTP_PRECOMPUTE_METADATA=0
export SGLANG_USE_LEGACY_FUSED_RMS_QUANT=0
export SGLANG_USE_FUSED_RMS_QUANT=1
export SGLANG_USE_FUSED_SILU_MUL_QUANT=1
export SGLANG_DSA_KPOOL_LIGHTOP_TOPK=1
export W8A8_SUPPORT_METHODS=3
export SGLANG_USE_LIGHTOP_PREFILL_DEQUANT=1
export SGLANG_REUSE_W8A8_INT8_EP_MOE_WORKSPACE=65536
export SGLANG_DSA_HCU_USE_LIGHTOP_DECODE_GATHER=1
export SGLANG_USE_AITER_CHUNK_GATED_DELTA_H_HIP=1
export SGLANG_USE_FUSED_SILU_MUL_CLAMP_QUANT=1
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export TOKENIZERS_PARALLELISM=false
export RAYON_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export SGLANG_PYSPY_DUMP_BEFORE_CRASH=0
export HSA_USE_SVM=0
unset SGLANG_SIMULATE_ACC_LEN SGLANG_SIMULATE_ACC_METHOD SGLANG_SIMULATED_EXPERT_BALANCE
unset PYTHONPATH
export NCCL_SOCKET_IFNAME=ib0
export GLOO_SOCKET_IFNAME=ib0
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
sglang serve \
    --model-path "<TODO_MODEL_ID>" \
    --served-model-name GLM-5.3-Flash \
    --trust-remote-code \
    --random-seed 648620208 \
    --nnodes 2 \
    --node-rank 1 \
    --host "<D_node1_ip>" \
    --dist-init-addr "<D_node0_ip>":5000 \
    --max-running-requests 128 \
    --disable-shared-experts-fusion \
    --disable-piecewise-cuda-graph \
    --disable-chunked-prefix-cache \
    --disable-radix-cache \
    --cuda-graph-bs 1 2 3 4 5 6 7 8 \
    --mem-fraction-static 0.80 \
    --context-length 1048576 \
    --page-size 64 \
    --dtype bfloat16 \
    --max-mamba-cache-size 320 \
    --mamba-scheduler-strategy no_buffer \
    --tp-size 16 \
    --dp-size 16 \
    --ep-size 16 \
    --moe-dense-tp-size 1 \
    --enable-dp-attention \
    --enable-dp-lm-head \
    --moe-a2a-backend deepep \
    --deepep-mode low_latency \
    --attention-backend nsa \
    --nsa-prefill-backend flashmla_auto \
    --nsa-decode-backend flashmla_kv \
    --linear-attn-backend triton \
    --kv-cache-dtype fp8_e4m3 \
    --quantization w8a8_int8 \
    --disaggregation-transfer-backend mooncake \
    --disaggregation-ib-device shca_0,shca_1,shca_2,shca_3 \
    --disaggregation-mode decode \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 5 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 6 \
    --reasoning-parser glm45 \
    --tool-call-parser glm47 \
    --grammar-backend xgrammar \
    --glm-adaptive-max-tokens \
    --enable-cache-report \
    --enable-metrics \
    --tokenizer-worker-num=16 \
    --json-model-override-args '{"index_share_for_mtp_iteration": true}' \
    --numa-node 0 3 2 1 4 7 6 5 \
    --enable-kda-replayssm-spec
```

#### Router

P/D 就绪后，在 P 节点的另一个终端启动 Router。两个 D 节点组成一个 Decode 实例，因此只注册 D node 0。

```bash
python3 -m sglang_router.launch_router --pd-disaggregation \
    --prefill "http://<P_node_ip>:30000" 8998 \
    --decode "http://<D_node0_ip>:30000" \
    --host "<P_node_ip>" \
    --port 30001 \
    --policy round_robin \
    --pool-idle-timeout-secs 4
```

#### 验证请求

在 P 节点执行：

```bash
curl http://localhost:30001/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"GLM-5.3-Flash","messages":[{"role":"user","content":"Hello"}],"max_tokens":128}'
```

## API 调用

### IFB

```python
from openai import OpenAI

client = OpenAI(base_url="http://<ifb_node0_ip>:30000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="GLM-5.3-Channel-FP8-w8a8",
    messages=[{"role": "user", "content": "你好"}],
    max_tokens=2048,
)
```

```bash
curl "http://<ifb_node0_ip>:30000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model": "GLM-5.3-Channel-FP8-w8a8", "messages": [{"role": "user", "content": "你好"}], "max_tokens": 128}'
```

### PD 分离

PD 分离模式下，客户端请求发送到 SGLang Router。将 `<router_ip>` 替换为 Router 所在节点的 IP，服务端口为 `30001`。

```python
from openai import OpenAI

client = OpenAI(base_url="http://<router_ip>:30001/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="GLM-5.3-Channel-FP8-w8a8",
    messages=[{"role": "user", "content": "你好"}],
    max_tokens=2048,
)
```

```bash
curl "http://<router_ip>:30001/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model": "GLM-5.3-Channel-FP8-w8a8", "messages": [{"role": "user", "content": "你好"}], "max_tokens": 128}'
```
