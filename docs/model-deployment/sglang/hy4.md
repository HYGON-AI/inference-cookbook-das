# Hy4 on SGLang

## 模型简介

Hy4-preview 是 Hy 系列混合专家模型。本文档提供 Channel FP8 W8A8 权重在 BW1100 上的 SGLang 0.5.18 部署命令。

## 模型列表

| 模型权重 | 量化方式 | SGLang 镜像 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | ----------- | -------- | ---- | -------- | -------- |
| [hygon/Hy4-preview-Channel-FP8-w8a8](https://www.modelscope.cn/models/hygon/Hy4-preview-Channel-FP8-w8a8) | FP8 W8A8 | [0.5.18](../docker_images.md) | BW1100 | 8 | IFB(TP8DP8) | [**`>_`**](#hy4-preview-channel-fp8-w8a8-ifb-bw1100-8x-sglang-0518-tp8dp8) |
|  | FP8 W8A8 | [0.5.18](../docker_images.md) | BW1100 | 16 | IFB(TP16DP16EP16) | [**`>_`**](#hy4-preview-channel-fp8-w8a8-ifb-bw1100-16x-sglang-0518-tp16dp16ep16) |

## 启动命令

### Hy4-preview-Channel-FP8-w8a8 IFB BW1100 8x SGLang 0.5.18 TP8DP8

须按本机 GPU 与 NUMA 亲和性设置 `--numa-node`。

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export SGLANG_USE_FP8_W8A8_MOE=1
export SGLANG_USE_DEEPGEMM_MOE=1
export SGLANG_OPT_USE_TOPK_V2=false
export SGLANG_ROCM_USE_AITER_MOE=1
export SGLANG_OPT_HY4_IHC_TILELANG=1

sglang serve \
  --model-path hygon/Hy4-preview-Channel-FP8-w8a8 \
  --trust-remote-code \
  --dtype bfloat16 \
  --tp-size 8 \
  --dp-size 8 \
  --enable-dp-attention \
  --enable-dp-lm-head \
  --moe-a2a-backend deepep \
  --deepep-mode auto \
  --moe-dense-tp-size 1 \
  --disable-shared-experts-fusion \
  --attention-backend dsa \
  --dsa-prefill-backend flashmla_sparse \
  --dsa-decode-backend flashmla_kv \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 2 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 3 \
  --page-size 64 \
  --chunked-prefill-size 8192 \
  --max-running-requests 64 \
  --numa-node 0 0 0 0 1 1 1 1 \
  --kv-cache-dtype fp8_e4m3 \
  --disable-radix-cache \
  --mem-fraction-static 0.95
```

### Hy4-preview-Channel-FP8-w8a8 IFB BW1100 16x SGLang 0.5.18 TP16DP16EP16

#### Node 0

```bash
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_SET_CPU_AFFINITY=1
export HIP_KERNEL_BATCH_CEILING=100
export GPU_FORCE_BLIT_COPY_SIZE=16
export GPU_MAX_HW_QUEUES=3
sysctl -w kernel.numa_balancing=0
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export HIP_GRAPH_ACCUMULATE_DISPATCH=0
export HIP_H2D_DISABLE_COPY_BUFFER=0
export HIP_D2H_DISABLE_COPY_BUFFER=0
export HIP_H2D_DIRECT_COPY_THRESHOLD=32768
export HIP_H2D_HSAAPI_COPY_THRESHOLD=32768
export HIP_D2H_DIRECT_COPY_THRESHOLD=512
export HIP_D2H_HSAAPI_COPY_THRESHOLD=512
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=512
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_OPT_HY4_IHC_TILELANG=1
export SGLANG_ROCM_USE_AITER_MOE=1
export SGLANG_USE_FP8_W8A8_MOE=1
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_DEEPGEMM_MOE=1
export ROCSHMEM_HEAP_SIZE=5368709120
export ROCSHMEM_MAX_NUM_CONTEXTS=64
export ROCSHMEM_GDR_DISABLE_XDP=1
export ROCSHMEM_IB_GID_INDEX=0
export SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH=1
export SGLANG_OPT_USE_TOPK_V2=false
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export NCCL_IB_HCA="shca_0,shca_1,shca_2,shca_3"

sglang serve \
  --model-path hygon/Hy4-preview-Channel-FP8-w8a8 \
  --dp-size 16 \
  --tp-size 16 \
  --ep-size 16 \
  --enable-dp-attention \
  --deepep-mode low_latency \
  --moe-a2a-backend deepep \
  --moe-dense-tp-size 1 \
  --enable-dp-lm-head \
  --dtype bfloat16 \
  --trust-remote-code \
  --mem-fraction-static 0.9 \
  --context-length 131072 \
  --page-size 64 \
  --attention-backend dsa \
  --dsa-prefill-backend flashmla_sparse \
  --dsa-decode-backend flashmla_kv \
  --kv-cache-dtype fp8_e4m3 \
  --max-running-requests 64 \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --watchdog-timeout 36000 \
  --disable-shared-experts-fusion \
  --enable-cache-report \
  --nnodes 2 \
  --node-rank 0 \
  --dist-init-addr "<node0_ip>:5000" \
  --host "<node0_ip>"
```

#### Node 1

```bash
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_SET_CPU_AFFINITY=1
export HIP_KERNEL_BATCH_CEILING=100
export GPU_FORCE_BLIT_COPY_SIZE=16
export GPU_MAX_HW_QUEUES=3
sysctl -w kernel.numa_balancing=0
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export HIP_GRAPH_ACCUMULATE_DISPATCH=0
export HIP_H2D_DISABLE_COPY_BUFFER=0
export HIP_D2H_DISABLE_COPY_BUFFER=0
export HIP_H2D_DIRECT_COPY_THRESHOLD=32768
export HIP_H2D_HSAAPI_COPY_THRESHOLD=32768
export HIP_D2H_DIRECT_COPY_THRESHOLD=512
export HIP_D2H_HSAAPI_COPY_THRESHOLD=512
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=512
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_OPT_HY4_IHC_TILELANG=1
export SGLANG_ROCM_USE_AITER_MOE=1
export SGLANG_USE_FP8_W8A8_MOE=1
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_DEEPGEMM_MOE=1
export ROCSHMEM_HEAP_SIZE=5368709120
export ROCSHMEM_MAX_NUM_CONTEXTS=64
export ROCSHMEM_GDR_DISABLE_XDP=1
export ROCSHMEM_IB_GID_INDEX=0
export SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH=1
export SGLANG_OPT_USE_TOPK_V2=false
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export NCCL_IB_HCA="shca_0,shca_1,shca_2,shca_3"

sglang serve \
  --model-path hygon/Hy4-preview-Channel-FP8-w8a8 \
  --dp-size 16 \
  --tp-size 16 \
  --ep-size 16 \
  --enable-dp-attention \
  --deepep-mode low_latency \
  --moe-a2a-backend deepep \
  --moe-dense-tp-size 1 \
  --enable-dp-lm-head \
  --dtype bfloat16 \
  --trust-remote-code \
  --mem-fraction-static 0.9 \
  --context-length 131072 \
  --page-size 64 \
  --attention-backend dsa \
  --dsa-prefill-backend flashmla_sparse \
  --dsa-decode-backend flashmla_kv \
  --kv-cache-dtype fp8_e4m3 \
  --max-running-requests 64 \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --watchdog-timeout 36000 \
  --disable-shared-experts-fusion \
  --enable-cache-report \
  --nnodes 2 \
  --node-rank 1 \
  --dist-init-addr "<node0_ip>:5000" \
  --host "<node1_ip>"
```
