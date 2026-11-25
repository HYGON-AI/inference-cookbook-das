# GLM-4-32B-0414 on SGLang

## 模型列表

| 模型权重 | 量化方式 | SGLang 镜像 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | -------- | -------- | ---- | -------- | -------- |
| [ZhipuAI/GLM-4-32B-0414](https://www.modelscope.cn/models/ZhipuAI/GLM-4-32B-0414) | BF16 | [0.5.18](../docker_images.md) | BW1000 | 4 | IFB | [**`>_`**](#glm-4-32b-0414-ifb-bw1000-4x-sglang-0518) |
|  | BF16 | [0.5.18](../docker_images.md) | BW100 | 4 | IFB | [**`>_`**](#glm-4-32b-0414-ifb-bw100-4x-sglang-0518) |
|  | BF16 | [0.5.18](../docker_images.md) | K100_AI | 4 | IFB | [**`>_`**](#glm-4-32b-0414-ifb-k100_ai-4x-sglang-0518) |


## 启动命令

### GLM-4-32B-0414 IFB BW1000 4x SGLang 0.5.18

```bash
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_TORCH_PROFILER_DIR=/home/profile
export SGLANG_SET_CPU_AFFINITY=1
export HIP_KERNEL_BATCH_CEILING=100
export GPU_MAX_HW_QUEUES=3
sysctl -w kernel.numa_balancing=0
export SGLANG_ENABLE_SPEC_V2=1
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
  --model-path ZhipuAI/GLM-4-32B-0414 \
  --trust-remote-code \
  --dtype bfloat16 \
  --attention-backend triton \
  --tensor-parallel-size 4 \
  --page-size 64 \
  --mem-fraction-static 0.9
```

### GLM-4-32B-0414 IFB BW100 4x SGLang 0.5.18

```bash
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_TORCH_PROFILER_DIR=/home/profile
export SGLANG_SET_CPU_AFFINITY=1
export HIP_KERNEL_BATCH_CEILING=100
export GPU_MAX_HW_QUEUES=3
sysctl -w kernel.numa_balancing=0
export SGLANG_ENABLE_SPEC_V2=1
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
  --model-path ZhipuAI/GLM-4-32B-0414 \
  --trust-remote-code \
  --dtype bfloat16 \
  --attention-backend triton \
  --tensor-parallel-size 4 \
  --page-size 64 \
  --mem-fraction-static 0.9
```

### GLM-4-32B-0414 IFB K100_AI 4x SGLang 0.5.18

```bash
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_TORCH_PROFILER_DIR=/home/profile
export SGLANG_SET_CPU_AFFINITY=1
export HIP_KERNEL_BATCH_CEILING=100
export GPU_MAX_HW_QUEUES=3
sysctl -w kernel.numa_balancing=0
export SGLANG_ENABLE_SPEC_V2=1
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
  --model-path ZhipuAI/GLM-4-32B-0414 \
  --trust-remote-code \
  --dtype bfloat16 \
  --disable-custom-all-reduce \
  --attention-backend triton \
  --tensor-parallel-size 4 \
  --page-size 64 \
  --mem-fraction-static 0.85
```

## API 调用

### IFB

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "ZhipuAI/GLM-4-32B-0414", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 128}'
```
