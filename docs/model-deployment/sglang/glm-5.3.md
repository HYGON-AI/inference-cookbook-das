# GLM-5.3

GLM-5.3 是智谱（Z.ai）推出的开放权重大语言模型，属于 GLM-5 系列，重点强化了编程、复杂推理和智能体任务能力。它采用 MoE（混合专家）架构，并与 GLM-5.2 使用相同基础模型，主要通过后训练提升效果，适合代码生成、调试、长流程任务规划和工具调用等场景。

## 模型列表

| 模型权重 | 量化方式 | SGLang 镜像 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | ----------- | -------- | ---- | -------- | -------- |
| [hygon/GLM-5.3-Channel-FP8-w8a8](https://www.modelscope.cn/models/hygon/GLM-5.3-Channel-FP8-w8a8) | FP8 W8A8 | 0.5.18 | scaleX40-3G | 8 | IFB | [**`>_`**](#glm-53-channel-fp8-w8a8-ifb-scalex40-3g-8x-sglang-0518) |
|                                                                                                 | FP8 W8A8 | 0.5.18 | scaleX40-3G | 24 | 1P1D | [**`>_`**](#glm-53-channel-fp8-w8a8-pd-scalex40-3g-24x-sglang-0518) |

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
export USE_DCU_CUSTOM_ALLREDUCE=1
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
export USE_DCU_CUSTOM_ALLREDUCE=1
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
export USE_DCU_CUSTOM_ALLREDUCE=1
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
export USE_DCU_CUSTOM_ALLREDUCE=1
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
export USE_DCU_CUSTOM_ALLREDUCE=1
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
export USE_DCU_CUSTOM_ALLREDUCE=1
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
export USE_DCU_CUSTOM_ALLREDUCE=1
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
export USE_DCU_CUSTOM_ALLREDUCE=1
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
