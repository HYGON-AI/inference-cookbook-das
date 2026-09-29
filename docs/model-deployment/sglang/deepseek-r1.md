# DeepSeek-R1 on SGLang

## 模型简介

DeepSeek-R1 是 DeepSeek 推出的推理强化模型系列，面向复杂推理、数学与代码场景。SGLang 在 HCU 平台可通过张量并行部署 R1 系列模型，并兼容 OpenAI API 接口。

## 模型列表

| 模型权重 | 量化方式 | SGLang 镜像 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | ----------- | -------- | ---- | -------- | -------- |
| [deepseek-ai/DeepSeek-R1-Distill-Qwen-32B](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B) | BF16 | [0.5.18](../docker_images.md) | BW1100 | 2 | IFB | [**`>_`**](#deepseek-r1-distill-qwen-32b-ifb-bw1100-2x-sglang-0518) |
|  | BF16 | [0.5.18](../docker_images.md) | BW1000 | 2 | IFB | [**`>_`**](#deepseek-r1-distill-qwen-32b-ifb-bw1000-2x-sglang-0518) |
|  | BF16 | [0.5.12](../docker_images.md) | BW1100 | 2 | IFB | [**`>_`**](#deepseek-r1-distill-qwen-32b-ifb-bw1100-2x-sglang-0512) |
|  | BF16 | [0.5.12](../docker_images.md) | BW1000 | 4 | IFB | [**`>_`**](#deepseek-r1-distill-qwen-32b-ifb-bw1000-4x-sglang-0512) |
| [hygon/DeepSeek-R1-Channel-FP8-w8a8](https://www.modelscope.cn/models/hygon/DeepSeek-R1-Channel-FP8-w8a8) | FP8 W8A8 | [0.5.12](../docker_images.md) | BW1100 | 8 | IFB | [**`>_`**](#deepseek-r1-channel-fp8-w8a8-ifb-bw1100-8x-sglang-0512) |
| [deepseek-ai/DeepSeek-R1-Distill-Llama-70B](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Llama-70B) | BF16 | [0.5.12](../docker_images.md) | BW1100 | 8 | IFB | [**`>_`**](#deepseek-r1-distill-llama-70b-ifb-bw1100-8x-sglang-0512) |
|  | BF16 | [0.5.12](../docker_images.md) | BW1000 | 8 | IFB | [**`>_`**](#deepseek-r1-distill-llama-70b-ifb-bw1000-8x-sglang-0512) |
|  | BF16 | [0.5.12](../docker_images.md) | K100_AI | 4 | IFB | [**`>_`**](#deepseek-r1-distill-llama-70b-ifb-k100_ai-4x-sglang-0512) |
| [hygon/DeepSeek-R1-Distill-Llama-70B-Channel-INT8-w8a8](https://www.modelscope.cn/models/hygon/DeepSeek-R1-Distill-Llama-70B-Channel-INT8-w8a8) | INT8 W8A8 | [0.5.12](../docker_images.md) | BW1100 | 2 | IFB | [**`>_`**](#deepseek-r1-distill-llama-70b-channel-int8-w8a8-ifb-bw1100-2x-sglang-0512) |
|  | INT8 W8A8 | [0.5.12](../docker_images.md) | BW1000 | 4 | IFB | [**`>_`**](#deepseek-r1-distill-llama-70b-channel-int8-w8a8-ifb-bw1000-4x-sglang-0512) |
|  | INT8 W8A8 | [0.5.12](../docker_images.md) | K100_AI | 8 | IFB | [**`>_`**](#deepseek-r1-distill-llama-70b-channel-int8-w8a8-ifb-k100_ai-8x-sglang-0512) |
|  | FP8 W8A8 | [0.5.10](../docker_images.md) | BW1100 | 8 | IFB | [**`>_`**](#deepseek-r1-channel-fp8-w8a8-ifb-bw1100-8x-sglang-0510) |

## 启动命令

### DeepSeek-R1-Distill-Qwen-32B IFB BW1100 2x SGLang 0.5.18

```bash
sglang serve \
  --model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
  --trust-remote-code \
  --dtype bfloat16 \
  --tp 2 \
  --kv-cache-dtype bfloat16 \
  --page-size 64 \
  --mem-fraction-static 0.9 \
  --attention-backend fa3
```

### DeepSeek-R1-Distill-Qwen-32B IFB BW1000 2x SGLang 0.5.18

```bash
sglang serve \
  --model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
  --trust-remote-code \
  --dtype bfloat16 \
  --tp 2 \
  --kv-cache-dtype bfloat16 \
  --page-size 64 \
  --mem-fraction-static 0.9 \
  --attention-backend fa3
```

### DeepSeek-R1-Distill-Qwen-32B IFB BW1100 2x SGLang 0.5.12

```bash
unset NCCL_TOPO_FILE
export USE_DCU_CUSTOM_ALLREDUCE=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_TORCH_PROFILER_DIR=/home/profile
export SGLANG_SET_CPU_AFFINITY=1
export HIP_KERNEL_BATCH_CEILING=100
export GPU_MAX_HW_QUEUES=3
sysctl -w kernel.numa_balancing=0
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export HIP_H2D_DISABLE_COPY_BUFFER=0
export HIP_D2H_DISABLE_COPY_BUFFER=0
export HIP_H2D_DIRECT_COPY_THRESHOLD=32768
export HIP_H2D_HSAAPI_COPY_THRESHOLD=32768
export HIP_D2H_DIRECT_COPY_THRESHOLD=512
export HIP_D2H_HSAAPI_COPY_THRESHOLD=512
export SGLANG_USE_FUSED_RMSNORM_ROPE=1
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_MARLIN_W16A16_MOE=1

sglang serve \
  --model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
  --trust-remote-code \
  --numa-node 0 0 0 0 1 1 1 1 \
  --dtype bfloat16 \
  --kv-cache-dtype bfloat16 \
  --tensor-parallel-size 2 \
  --page-size 64 \
  --nnodes 1 \
  --node-rank 0 \
  --mem-fraction-static 0.9 \
  --attention-backend fa3
```

### DeepSeek-R1-Distill-Qwen-32B IFB BW1000 4x SGLang 0.5.12

```bash
unset NCCL_TOPO_FILE
export USE_DCU_CUSTOM_ALLREDUCE=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_TORCH_PROFILER_DIR=/home/profile
export SGLANG_SET_CPU_AFFINITY=1
export HIP_KERNEL_BATCH_CEILING=100
export GPU_MAX_HW_QUEUES=3
sysctl -w kernel.numa_balancing=0
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export HIP_H2D_DISABLE_COPY_BUFFER=0
export HIP_D2H_DISABLE_COPY_BUFFER=0
export HIP_H2D_DIRECT_COPY_THRESHOLD=32768
export HIP_H2D_HSAAPI_COPY_THRESHOLD=32768
export HIP_D2H_DIRECT_COPY_THRESHOLD=512
export HIP_D2H_HSAAPI_COPY_THRESHOLD=512
export SGLANG_USE_FUSED_RMSNORM_ROPE=1
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_MARLIN_W16A16_MOE=1

sglang serve \
  --model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
  --trust-remote-code \
  --numa-node 0 0 0 0 1 1 1 1 \
  --dtype bfloat16 \
  --kv-cache-dtype bfloat16 \
  --tensor-parallel-size 4 \
  --page-size 64 \
  --nnodes 1 \
  --node-rank 0 \
  --mem-fraction-static 0.9 \
  --attention-backend fa3
```

### DeepSeek-R1-Channel-FP8-w8a8 IFB BW1100 8x SGLang 0.5.12

```bash
export USE_HCU_CUSTOM_ALLREDUCE=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_USE_LIGHTOP=0
export SGLANG_USE_FUSED_RMS_QUANT=0
export SGLANG_USE_OPT_CAT=1
export SGLANG_USE_FP8_W8A8_MOE=1
export SGLANG_USE_RMS_QUANT_PATH=1
export USE_FUSED_RMS_QUANT_PATH=1
export SGLANG_USE_FUSED_RMSNORM_ROPE=0
export SGLANG_TORCH_PROFILER_DIR=/workspace/prof
export SGLANG_SET_CPU_AFFINITY=1
export HIP_KERNEL_BATCH_CEILING=100
export GPU_MAX_HW_QUEUES=4
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export HIP_H2D_DISABLE_COPY_BUFFER=0
export HIP_D2H_DISABLE_COPY_BUFFER=0
export HIP_H2D_DIRECT_COPY_THRESHOLD=32768
export HIP_H2D_HSAAPI_COPY_THRESHOLD=32768
export HIP_D2H_DIRECT_COPY_THRESHOLD=512
export HIP_D2H_HSAAPI_COPY_THRESHOLD=512
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export USE_SPE_MQP=1
export MC_ALLOWED_IBV_DEVICES=mlx5_6,mlx5_7,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_8,mlx5_9

sglang serve \
  --model-path hygon/DeepSeek-R1-Channel-FP8-w8a8 \
  --numa-node 0 0 0 0 1 1 1 1 \
  --chunked-prefill-size -1 \
  --max-running-requests 256 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 2 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 1 \
  --context-length 65536 \
  --quantization w8a8_fp8 \
  --kv-cache-dtype fp8_e4m3 \
  --trust-remote-code \
  --nnodes 1 \
  --node-rank 0 \
  --dtype bfloat16 \
  --tp-size 8 \
  --pp-size 1 \
  --mem-fraction-static 0.9 \
  --reasoning-parser deepseek-r1 \
  --tool-call-parser deepseekv31 \
  --attention-backend hcu_mla
```




### DeepSeek-R1-Channel-FP8-w8a8 IFB BW1100 8x SGLang 0.5.18

```bash
export USE_HCU_CUSTOM_ALLREDUCE=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_USE_LIGHTOP=0
export SGLANG_USE_FUSED_RMS_QUANT=0
export SGLANG_USE_OPT_CAT=1
export SGLANG_USE_FP8_W8A8_MOE=1
export SGLANG_USE_RMS_QUANT_PATH=1
export USE_FUSED_RMS_QUANT_PATH=1
export SGLANG_USE_FUSED_RMSNORM_ROPE=0
export SGLANG_TORCH_PROFILER_DIR=/workspace/prof
export SGLANG_SET_CPU_AFFINITY=1
export HIP_KERNEL_BATCH_CEILING=100
export GPU_MAX_HW_QUEUES=4
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export HIP_H2D_DISABLE_COPY_BUFFER=0
export HIP_D2H_DISABLE_COPY_BUFFER=0
export HIP_H2D_DIRECT_COPY_THRESHOLD=32768
export HIP_H2D_HSAAPI_COPY_THRESHOLD=32768
export HIP_D2H_DIRECT_COPY_THRESHOLD=512
export HIP_D2H_HSAAPI_COPY_THRESHOLD=512
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export USE_SPE_MQP=1
export MC_ALLOWED_IBV_DEVICES=mlx5_6,mlx5_7,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_8,mlx5_9

sglang serve \
  --model-path hygon/DeepSeek-R1-Channel-FP8-w8a8 \
  --numa-node 0 0 0 0 1 1 1 1 \
  --chunked-prefill-size -1 \
  --max-running-requests 256 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 2 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 1 \
  --context-length 65536 \
  --quantization w8a8_fp8 \
  --kv-cache-dtype fp8_e4m3 \
  --trust-remote-code \
  --nnodes 1 \
  --node-rank 0 \
  --dtype bfloat16 \
  --tp-size 8 \
  --pp-size 1 \
  --mem-fraction-static 0.9 \
  --reasoning-parser deepseek-r1 \
  --tool-call-parser deepseekv31 \
  --attention-backend hcu_mla
```



### DeepSeek-R1-Channel-int8 IFB BW1100 8x SGLang 0.5.18

```bash
export USE_HCU_CUSTOM_ALLREDUCE=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_USE_LIGHTOP=0
export SGLANG_USE_FUSED_RMS_QUANT=0
export SGLANG_USE_OPT_CAT=1
export SGLANG_USE_FP8_W8A8_MOE=1
export SGLANG_USE_RMS_QUANT_PATH=1
export USE_FUSED_RMS_QUANT_PATH=1
export SGLANG_USE_FUSED_RMSNORM_ROPE=0
export SGLANG_TORCH_PROFILER_DIR=/workspace/prof
export SGLANG_SET_CPU_AFFINITY=1
export HIP_KERNEL_BATCH_CEILING=100
export GPU_MAX_HW_QUEUES=4
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export HIP_H2D_DISABLE_COPY_BUFFER=0
export HIP_D2H_DISABLE_COPY_BUFFER=0
export HIP_H2D_DIRECT_COPY_THRESHOLD=32768
export HIP_H2D_HSAAPI_COPY_THRESHOLD=32768
export HIP_D2H_DIRECT_COPY_THRESHOLD=512
export HIP_D2H_HSAAPI_COPY_THRESHOLD=512
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export USE_SPE_MQP=1
export MC_ALLOWED_IBV_DEVICES=mlx5_6,mlx5_7,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_8,mlx5_9

sglang serve \
  --model-path /public/opendas/DL_DATA/llm-models/DeepSeek-R1-Channel-INT8 \
  --numa-node 0 0 0 0 1 1 1 1 \
  --chunked-prefill-size -1 \
  --max-running-requests 256 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 2 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 1 \
  --context-length 65536 \
  --quantization w8a8_fp8 \
  --kv-cache-dtype fp8_e4m3 \
  --trust-remote-code \
  --nnodes 1 \
  --node-rank 0 \
  --dtype bfloat16 \
  --tp-size 8 \
  --pp-size 1 \
  --mem-fraction-static 0.9 \
  --reasoning-parser deepseek-r1 \
  --tool-call-parser deepseekv31 \
  --attention-backend hcu_mla
```



### DeepSeek-R1-Channel-bf16 IFB BW1100 16x SGLang 0.5.18

```bash
export USE_HCU_CUSTOM_ALLREDUCE=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_USE_LIGHTOP=0
export SGLANG_USE_FUSED_RMS_QUANT=0
export SGLANG_USE_LIGHTOP_MOE_SUM_MUL_ADD=0
export SGLANG_USE_OPT_CAT=0
export SGLANG_TORCH_PROFILER_DIR=/workspace/prof
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_USE_FUSED_RMSNORM_ROPE=1
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export ALLREDUCE_STREAM_WITH_COMPUTE=1

model_path=/parastor/opendas/DL_DATA/llm-models/deepseek-r1/DeepSeek-R1-bf16
rank=0
host_ip=$(hostname -I | awk '{print $1}')

sglang serve \
  --model-path $model_path \
  --host $host_ip \
  --port 30000 \
  --numa-node 0 0 1 1 2 2 3 3 \
  --chunked-prefill-size -1 \
  --max-running-requests 512 \
  --context-length 40960 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --cuda-graph-max-bs 128 \
  --kv-cache-dtype fp8_e4m3 \
  --trust-remote-code \
  --dist-init-addr 10.16.1.8:5001 \
  --nnodes 2 \
  --node-rank $rank \
  --dtype bfloat16 \
  --tp-size 16 \
  --pp-size 1 \
  --reasoning-parser deepseek-r1 \
  --tool-call-parser deepseekv31 \
  --mem-fraction-static 0.85 \
  --attention-backend hcu_mla
```



### DeepSeek-R1-0528-W4A8-V2 IFB BW1100 4x SGLang 0.5.18

```bash
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export GPU_FORCE_BLIT_COPY_SIZE=16
export GPU_MAX_HW_QUEUES=3
export HIP_D2H_DIRECT_COPY_THRESHOLD=512
export HIP_D2H_DISABLE_COPY_BUFFER=0
export HIP_D2H_HSAAPI_COPY_THRESHOLD=512
export HIP_GRAPH_ACCUMULATE_DISPATCH=1
export HIP_GRAPH_USE_CMD_CACHE=0
export HIP_H2D_DIRECT_COPY_THRESHOLD=32768
export HIP_H2D_DISABLE_COPY_BUFFER=0
export HIP_H2D_HSAAPI_COPY_THRESHOLD=32768
export HIP_KERNEL_BATCH_CEILING=100
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export HSA_ENABLE_COREDUMP=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export HSA_KERNARG_POOL_SIZE=8388608
export HSA_USE_SVM=0
unset MC_GID_INDEX
unset NCCL_DEBUG
unset NCCL_DEBUG_FILE
unset NCCL_DEBUG_SUBSYS
export NCCL_IB_DISABLE=0
unset NCCL_IB_GID_INDEX
unset RCCL_DEBUG
unset RCCL_DEBUG_SUBSYS
unset ROCSHMEM_ALLOWED_IBV_DEVICES
export ROC_AQL_QUEUE_SIZE=131072
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_ROCM_USE_AITER_MOE=0
export SGLANG_SET_CPU_AFFINITY=1
export SGLANG_USE_AITER_AR=0
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_MODELSCOPE=1
export SGLANG_W4A8_TPMOE_BACKEND=aiter
export TORCH_CPP_LOG_LEVEL=ERROR
unset TORCH_NCCL_DEBUG
export USE_DCU_CUSTOM_ALLREDUCE=0
export W8A8_SUPPORT_METHODS=3

sglang serve \
  --context-length 32768 \
  --cuda-graph-max-bs 16 \
  --disable-radix-cache \
  --dist-timeout 10000 \
  --dp-size 1 \
  --dtype bfloat16 \
  --ep-size 1 \
  --kv-cache-dtype fp8_e4m3 \
  --max-running-requests 64 \
  --mem-fraction-static 0.90 \
  --model-path hygon/DeepSeek-R1-0528-W4A8-V2 \
  --moe-dense-tp-size 1 \
  --nsa-decode-backend flashmla_kv \
  --nsa-prefill-backend flashmla_auto \
  --page-size 64 \
  --quantization slimquant_w4a8_marlin \
  --reasoning-parser deepseek-r1 \
  --tool-call-parser deepseekv31 \
  --tp-size 8 \
  --trust-remote-code \
  --watchdog-timeout 3600 \
  --port 30000
```








### DeepSeek-R1-Distill-Llama-70B IFB BW1100 8x SGLang 0.5.12

```bash
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
export SGLANG_ROCM_USE_AITER_MOE=0
export W8A8_SUPPORT_METHODS=1
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export HIP_GRAPH_ACCUMULATE_DISPATCH=1
export HIP_GRAPH_USE_CMD_CACHE=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export NCCL_IB_GID_INDEX=3

sglang serve \
  --model-path deepseek-ai/DeepSeek-R1-Distill-Llama-70B \
  --trust-remote-code \
  --tp-size 8 \
  --attention-backend fa3 \
  --dtype bfloat16 \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --page-size 64 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.8 \
  --chunked-prefill-size 8192
```

### DeepSeek-R1-Distill-Llama-70B IFB BW1000 8x SGLang 0.5.12

```bash
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
export SGLANG_ROCM_USE_AITER_MOE=0
export W8A8_SUPPORT_METHODS=1
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export HIP_GRAPH_ACCUMULATE_DISPATCH=1
export HIP_GRAPH_USE_CMD_CACHE=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export NCCL_IB_GID_INDEX=3

sglang serve \
  --model-path deepseek-ai/DeepSeek-R1-Distill-Llama-70B \
  --trust-remote-code \
  --tp-size 8 \
  --attention-backend fa3 \
  --dtype bfloat16 \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --page-size 64 \
  --mem-fraction-static 0.8 \
  --chunked-prefill-size 8192
```

### DeepSeek-R1-Distill-Llama-70B IFB K100_AI 4x SGLang 0.5.12

```bash
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
export SGLANG_ROCM_USE_AITER_MOE=0
export W8A8_SUPPORT_METHODS=1
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export HIP_GRAPH_ACCUMULATE_DISPATCH=1
export HIP_GRAPH_USE_CMD_CACHE=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200

sglang serve \
  --model-path deepseek-ai/DeepSeek-R1-Distill-Llama-70B \
  --trust-remote-code \
  --tp-size 4 \
  --attention-backend fa3 \
  --dtype bfloat16 \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --page-size 64 \
  --kv-cache-dtype bf16 \
  --mem-fraction-static 0.8 \
  --chunked-prefill-size 8192 \
  --disable-custom-all-reduce
```

### DeepSeek-R1-Distill-Llama-70B-Channel-INT8-w8a8 IFB BW1100 2x SGLang 0.5.12

```bash
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
export SGLANG_ROCM_USE_AITER_MOE=0
export W8A8_SUPPORT_METHODS=1
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export HIP_GRAPH_ACCUMULATE_DISPATCH=1
export HIP_GRAPH_USE_CMD_CACHE=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export NCCL_IB_GID_INDEX=3
export HIP_VISIBLE_DEVICES=6,7

sglang serve \
  --model-path hygon/DeepSeek-R1-Distill-Llama-70B-Channel-INT8-w8a8 \
  --trust-remote-code \
  --tp-size 2 \
  --attention-backend fa3 \
  --dtype bfloat16 \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --page-size 64 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.8 \
  --chunked-prefill-size 8192 \
  --quantization slimquant_marlin
```

### DeepSeek-R1-Distill-Llama-70B-Channel-INT8-w8a8 IFB BW1000 4x SGLang 0.5.12

```bash
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
export SGLANG_ROCM_USE_AITER_MOE=0
export W8A8_SUPPORT_METHODS=1
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export HIP_GRAPH_ACCUMULATE_DISPATCH=1
export HIP_GRAPH_USE_CMD_CACHE=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export NCCL_IB_GID_INDEX=3

sglang serve \
  --model-path hygon/DeepSeek-R1-Distill-Llama-70B-Channel-INT8-w8a8 \
  --trust-remote-code \
  --tp-size 4 \
  --attention-backend fa3 \
  --dtype bfloat16 \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --page-size 64 \
  --mem-fraction-static 0.8 \
  --chunked-prefill-size 8192 \
  --quantization slimquant_marlin
```

### DeepSeek-R1-Distill-Llama-70B-Channel-INT8-w8a8 IFB K100_AI 8x SGLang 0.5.12

```bash
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
export SGLANG_ROCM_USE_AITER_MOE=0
export W8A8_SUPPORT_METHODS=1
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export HIP_GRAPH_ACCUMULATE_DISPATCH=1
export HIP_GRAPH_USE_CMD_CACHE=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export NCCL_IB_GID_INDEX=3

sglang serve \
  --model-path hygon/DeepSeek-R1-Distill-Llama-70B-Channel-INT8-w8a8 \
  --trust-remote-code \
  --tp-size 8 \
  --attention-backend fa3 \
  --dtype bfloat16 \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --page-size 64 \
  --kv-cache-dtype bf16 \
  --mem-fraction-static 0.8 \
  --chunked-prefill-size 8192 \
  --disable-custom-all-reduce \
  --quantization slimquant_marlin
```

### DeepSeek-R1-Channel-FP8-w8a8 IFB BW1100 8x SGLang 0.5.10
```bash
export USE_DCU_CUSTOM_ALLREDUCE=1
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_OPT_CAT=1
export SGLANG_USE_FP8_W8A8_MOE=1
export SGLANG_USE_RMS_QUANT_PATH=1
export USE_FUSED_RMS_QUANT_PATH=1
export SGLANG_USE_FUSED_RMSNORM_ROPE=1
export SGLANG_TORCH_PROFILER_DIR=/workspace/prof
export SGLANG_SET_CPU_AFFINITY=1
export HIP_KERNEL_BATCH_CEILING=100
export GPU_MAX_HW_QUEUES=4
export SGLANG_ENABLE_SPEC_V2=1

export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1

export HIP_H2D_DISABLE_COPY_BUFFER=0 # 同步异步强制走WriteBuffer
export HIP_D2H_DISABLE_COPY_BUFFER=0 # 同步异步强制走ReadBuffer
export HIP_H2D_DIRECT_COPY_THRESHOLD=32768 # 小于此值走CPUCopy
export HIP_H2D_HSAAPI_COPY_THRESHOLD=32768 # 大于此值走HSACOPY（CopyBuffer）
export HIP_D2H_DIRECT_COPY_THRESHOLD=512 # 小于此值走CPUCopy
export HIP_D2H_HSAAPI_COPY_THRESHOLD=512 # 大于此值走HSACOPY（CopyBuffer）

export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072

export ALLREDUCE_STREAM_WITH_COMPUTE=1
export USE_SPE_MQP=1
export MC_ALLOWED_IBV_DEVICES=mlx5_6,mlx5_7,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_8,mlx5_9

sglang serve \
    --model-path hygon/DeepSeek-R1-Channel-FP8-w8a8 \
    --numa-node 0 0 0 0 1 1 1 1 \
    --disable-radix-cache \
    --chunked-prefill-size -1 \
    --max-running-requests 256 \
    --speculative-algorithm EAGLE --speculative-num-steps 2  --speculative-eagle-topk 1  --speculative-num-draft-tokens 1 \
    --context-length 65536 \
    --quantization w8a8_fp8 \
    --kv-cache-dtype fp8_e4m3 \
    --trust-remote-code \
    --nnodes 1 \
    --node-rank 0 \
    --dtype bfloat16 \
    --tp-size 8 \
    --pp-size 1 \
    --mem-fraction-static 0.9 \
    --attention-backend dcu_mla

```

## API 调用

### IFB

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "DeepSeek-R1-Channel-FP8",
    "max_tokens": 1024,
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "请给出一个高并发服务限流方案。"}
    ]
  }'
```
