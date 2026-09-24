# MiniMax-2.5 on SGLang

## 模型简介

MiniMax-M2.5-Channel-FP8-w8a8 是 MiniMax 推出的大规模 MoE（混合专家）语言模型系列，总参数量 230B，激活参数约 10B，在长文本理解和生成方面表现突出。



## 模型列表

| 模型权重 | 量化方式 | SGLang 镜像 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | ----------- | -------- | ---- | -------- | -------- |
| [hygon/MiniMax-M2.5-bf16](https://www.modelscope.cn/models/hygon/MiniMax-M2.5-bf16) | BF16 | [0.5.12](../docker_images.md) | BW1100 | 8 | IFB | [**`>_`**](#minimax-m25-bf16-ifb-bw1100-8x-sglang-0512) |
| [hygon/MiniMax-M2.5-Channel-INT8-w8a8](https://www.modelscope.cn/models/hygon/MiniMax-M2.5-Channel-INT8-w8a8) | INT8 W8A8 | 0.5.18 | BW1100 | 8 | IFB | [**`>_`**](#minimax-m25-channel-int8-w8a8-ifb-bw1100-8x-sglang-0518) |
|  | INT8 W8A8 | 0.5.18 | BW1100 | 16 | 1P1D | [**`>_`**](#minimax-m25-channel-int8-w8a8-1p1d-bw1100-16x-sglang-0518) |
|  | INT8 W8A8 | 0.5.18 | BW1000 | 8 | IFB | [**`>_`**](#minimax-m25-channel-int8-w8a8-ifb-bw1000-8x-sglang-0518) |
|  | INT8 W8A8 | 0.5.18 | BW1000 | 16 | 1P1D | [**`>_`**](#minimax-m25-channel-int8-w8a8-1p1d-bw1000-16x-sglang-0518) |
|  | INT8 W8A8 | [0.5.12](../docker_images.md) | BW1100 | 8 | IFB | [**`>_`**](#minimax-m25-channel-int8-w8a8-ifb-bw1100-8x-sglang-0512) |
|  | INT8 W8A8 | [0.5.12](../docker_images.md) | BW1000 | 8 | IFB | [**`>_`**](#minimax-m25-channel-int8-w8a8-ifb-bw1000-8x-sglang-0512) |
|  | INT8 W8A8 | 0.5.10 | BW1100 | 8 | IFB | [**`>_`**](#minimax-m2-5-channel-int8-w8a8-ifb-bw1100-8x) |
|                                                                                                 | INT8 W8A8 | 0.5.10 | BW1100 | 16 | 1P1D| [**`>_`**](#minimax-m2-5-channel-int8-w8a8-1p1d-bw1100-16x) |
|                                                                                                 | INT8 W8A8 | 0.5.10 | BW1000 | 8 | IFB | [**`>_`**](#minimax-m2-5-channel-int8-w8a8-ifb-bw1000-8x) |
|                                                                                                 | INT8 W8A8 | 0.5.10 | BW1000 | 16 | 1P1D| [**`>_`**](#minimax-m2-5-channel-int8-w8a8-1p1d-bw1000-16x) |
| [hygon/MiniMax-M2.5-Channel-FP8-w8a8](https://www.modelscope.cn/models/hygon/MiniMax-M2.5-Channel-FP8-w8a8) | FP8 W8A8 | [0.5.12](../docker_images.md) | scaleX40-3G | 8 | 1P1D | [**`>_`**](#minimax-m25-channel-fp8-w8a8-1p1d-scalex40-3g-8x-sglang-0512) |
|  | FP8 W8A8 | [0.5.12](../docker_images.md) | BW1100 | 8 | IFB | [**`>_`**](#minimax-m25-channel-fp8-w8a8-ifb-bw1100-8x-sglang-0512) |
|  | FP8 W8A8 | 0.5.10 | BW1100 | 8 | IFB | [**`>_`**](#minimax-m2-5-channel-fp8-w8a8-ifb-bw1100-8x) |
|                                                                                                 | FP8 W8A8 | 0.5.10 | BW1100 | 16 | 1P1D| [**`>_`**](#minimax-m2-5-channel-fp8-w8a8-1p1d-bw1100-16x) |


## 启动命令

### MiniMax-M2.5-bf16 IFB BW1100 8x SGLang 0.5.12

```bash
export USE_DCU_CUSTOM_ALLREDUCE=1
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_USE_LIGHTOP=1
export VLLM_USE_LIGHTOP_MOE_ALIGN=1
export LMSLIM_USE_LIGHTOP=1
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export ALLREDUCE_STREAM_WITH_COMPUTE=1

sglang serve \
  --model-path hygon/MiniMax-M2.5-bf16 \
  --kv-cache-dtype fp8_e4m3 \
  --trust-remote-code \
  --page-size 64 \
  --dtype bfloat16 \
  --tp-size 4 --pp-size 1 --data-parallel-size 2 \
  --tool-call-parser minimax-m2 \
  --reasoning-parser minimax-append-think \
  --mem-fraction-static 0.9 \
  --attention-backend fa3 \
  --chunked-prefill-size 16384 \
  --max-running-requests 512 \
  --context-length 131072
```

### MiniMax-M2.5-Channel-INT8-w8a8 IFB BW1100 8x SGLang 0.5.18

```bash
export USE_DCU_CUSTOM_ALLREDUCE=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export HSA_FORCE_FINE_GRAIN_PCIE=1
unset NCCL_TOPO_FILE
export SGLANG_USE_LIGHTOP=1
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=0
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export LMSLIM_USE_LIGHTOP=1
export VLLM_USE_LIGHTOP_MOE_ALIGN=1
export SGLANG_KV_LAYOUT_HCU_FA=1
export SGLANG_USE_FUSED_RMS_QUANT=0
export SGLANG_USE_FP32GEMM_GATE_CUSTOM=1

sglang serve \
  --model-path hygon/MiniMax-M2.5-Channel-INT8-w8a8 \
  --quantization slimquant_marlin \
  --json-model-override-args '{"quantization_config":{"config_groups":{"group_0":{"format":"int-quantized","input_activations":{"actorder":null,"block_structure":null,"dynamic":true,"group_size":null,"num_bits":8,"observer":null,"observer_kwargs":{},"scale_dtype":null,"strategy":"token","symmetric":true,"type":"int","zp_dtype":null},"output_activations":null,"targets":["Linear"],"weights":{"actorder":null,"block_structure":null,"dynamic":false,"group_size":null,"num_bits":8,"observer":"memoryless_minmax","observer_kwargs":{},"scale_dtype":null,"strategy":"channel","symmetric":true,"type":"int","zp_dtype":null}}},"format":"int-quantized","global_compression_ratio":null,"ignore":["model.layers.0.block_sparse_moe.gate","model.layers.1.block_sparse_moe.gate","model.layers.2.block_sparse_moe.gate","model.layers.3.block_sparse_moe.gate","model.layers.4.block_sparse_moe.gate","model.layers.5.block_sparse_moe.gate","model.layers.6.block_sparse_moe.gate","model.layers.7.block_sparse_moe.gate","model.layers.8.block_sparse_moe.gate","model.layers.9.block_sparse_moe.gate","model.layers.10.block_sparse_moe.gate","model.layers.11.block_sparse_moe.gate","model.layers.12.block_sparse_moe.gate","model.layers.13.block_sparse_moe.gate","model.layers.14.block_sparse_moe.gate","model.layers.15.block_sparse_moe.gate","model.layers.16.block_sparse_moe.gate","model.layers.17.block_sparse_moe.gate","model.layers.18.block_sparse_moe.gate","model.layers.19.block_sparse_moe.gate","model.layers.20.block_sparse_moe.gate","model.layers.21.block_sparse_moe.gate","model.layers.22.block_sparse_moe.gate","model.layers.23.block_sparse_moe.gate","model.layers.24.block_sparse_moe.gate","model.layers.25.block_sparse_moe.gate","model.layers.26.block_sparse_moe.gate","model.layers.27.block_sparse_moe.gate","model.layers.28.block_sparse_moe.gate","model.layers.29.block_sparse_moe.gate","model.layers.30.block_sparse_moe.gate","model.layers.31.block_sparse_moe.gate","model.layers.32.block_sparse_moe.gate","model.layers.33.block_sparse_moe.gate","model.layers.34.block_sparse_moe.gate","model.layers.35.block_sparse_moe.gate","model.layers.36.block_sparse_moe.gate","model.layers.37.block_sparse_moe.gate","model.layers.38.block_sparse_moe.gate","model.layers.39.block_sparse_moe.gate","model.layers.40.block_sparse_moe.gate","model.layers.41.block_sparse_moe.gate","model.layers.42.block_sparse_moe.gate","model.layers.43.block_sparse_moe.gate","model.layers.44.block_sparse_moe.gate","model.layers.45.block_sparse_moe.gate","model.layers.46.block_sparse_moe.gate","model.layers.47.block_sparse_moe.gate","model.layers.48.block_sparse_moe.gate","model.layers.49.block_sparse_moe.gate","model.layers.50.block_sparse_moe.gate","model.layers.51.block_sparse_moe.gate","model.layers.52.block_sparse_moe.gate","model.layers.53.block_sparse_moe.gate","model.layers.54.block_sparse_moe.gate","model.layers.55.block_sparse_moe.gate","model.layers.56.block_sparse_moe.gate","model.layers.57.block_sparse_moe.gate","model.layers.58.block_sparse_moe.gate","model.layers.59.block_sparse_moe.gate","model.layers.60.block_sparse_moe.gate","model.layers.61.block_sparse_moe.gate","lm_head"],"kv_cache_scheme":{"type":"float","num_bits":8,"strategy":"tensor","symmetric":true,"dynamic":false},"quant_method":"compressed-tensors","quantization_status":"compressed","sparsity_config":{},"transform_config":{},"version":"0.14.0.1"}}' \
  --trust-remote-code \
  --page-size 64 \
  --dtype bfloat16 \
  --kv-cache-dtype bfloat16 \
  --tp-size 8 \
  --pp-size 1 \
  --data-parallel-size 1 \
  --tool-call-parser minimax-m2 \
  --reasoning-parser minimax-append-think \
  --mem-fraction-static 0.9 \
  --attention-backend triton \
  --chunked-prefill-size 16384 \
  --max-running-requests 512 \
  --context-length 131072
```

### MiniMax-M2.5-Channel-INT8-w8a8 1P1D BW1100 16x SGLang 0.5.18

网卡配置参考：[IB 网卡](../../troubleshooting/common-issues.md#ib网卡)。

#### P node

```bash
export USE_DCU_CUSTOM_ALLREDUCE=1
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export HSA_FORCE_FINE_GRAIN_PCIE=1
export SGLANG_ROCM_USE_AITER_MOE=0
export SGLANG_USE_LIGHTOP=1
export SGLANG_TORCH_PROFILER_DIR=/home/profile
export VLLM_USE_LIGHTOP_MOE_ALIGN=1
export LMSLIM_USE_LIGHTOP=1
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=0
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export SGLANG_KV_LAYOUT_HCU_FA=1
export SGLANG_HOST_IP=12.12.12.46
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=mlx5_0
export MC_GID_INDEX=3
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export ROCSHMEM_DISABLE_HDP_FLUSH=1
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_HEAP_SIZE=3173741824
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export ROCSHMEM_IB_GID_INDEX=0

sglang serve \
  --model-path hygon/MiniMax-M2.5-Channel-INT8-w8a8 \
  --numa-node 0 0 0 0 1 1 1 1 \
  --max-running-requests 512 \
  --context-length 131072 \
  --disaggregation-mode prefill \
  --load-balance-method round_robin \
  --host 12.12.12.46 \
  --port 30000 \
  --disaggregation-transfer-backend mooncake_tcp \
  --quantization slimquant_marlin \
  --json-model-override-args '{"quantization_config":{"config_groups":{"group_0":{"format":"int-quantized","input_activations":{"actorder":null,"block_structure":null,"dynamic":true,"group_size":null,"num_bits":8,"observer":null,"observer_kwargs":{},"scale_dtype":null,"strategy":"token","symmetric":true,"type":"int","zp_dtype":null},"output_activations":null,"targets":["Linear"],"weights":{"actorder":null,"block_structure":null,"dynamic":false,"group_size":null,"num_bits":8,"observer":"memoryless_minmax","observer_kwargs":{},"scale_dtype":null,"strategy":"channel","symmetric":true,"type":"int","zp_dtype":null}}},"format":"int-quantized","global_compression_ratio":null,"ignore":["model.layers.0.block_sparse_moe.gate","model.layers.1.block_sparse_moe.gate","model.layers.2.block_sparse_moe.gate","model.layers.3.block_sparse_moe.gate","model.layers.4.block_sparse_moe.gate","model.layers.5.block_sparse_moe.gate","model.layers.6.block_sparse_moe.gate","model.layers.7.block_sparse_moe.gate","model.layers.8.block_sparse_moe.gate","model.layers.9.block_sparse_moe.gate","model.layers.10.block_sparse_moe.gate","model.layers.11.block_sparse_moe.gate","model.layers.12.block_sparse_moe.gate","model.layers.13.block_sparse_moe.gate","model.layers.14.block_sparse_moe.gate","model.layers.15.block_sparse_moe.gate","model.layers.16.block_sparse_moe.gate","model.layers.17.block_sparse_moe.gate","model.layers.18.block_sparse_moe.gate","model.layers.19.block_sparse_moe.gate","model.layers.20.block_sparse_moe.gate","model.layers.21.block_sparse_moe.gate","model.layers.22.block_sparse_moe.gate","model.layers.23.block_sparse_moe.gate","model.layers.24.block_sparse_moe.gate","model.layers.25.block_sparse_moe.gate","model.layers.26.block_sparse_moe.gate","model.layers.27.block_sparse_moe.gate","model.layers.28.block_sparse_moe.gate","model.layers.29.block_sparse_moe.gate","model.layers.30.block_sparse_moe.gate","model.layers.31.block_sparse_moe.gate","model.layers.32.block_sparse_moe.gate","model.layers.33.block_sparse_moe.gate","model.layers.34.block_sparse_moe.gate","model.layers.35.block_sparse_moe.gate","model.layers.36.block_sparse_moe.gate","model.layers.37.block_sparse_moe.gate","model.layers.38.block_sparse_moe.gate","model.layers.39.block_sparse_moe.gate","model.layers.40.block_sparse_moe.gate","model.layers.41.block_sparse_moe.gate","model.layers.42.block_sparse_moe.gate","model.layers.43.block_sparse_moe.gate","model.layers.44.block_sparse_moe.gate","model.layers.45.block_sparse_moe.gate","model.layers.46.block_sparse_moe.gate","model.layers.47.block_sparse_moe.gate","model.layers.48.block_sparse_moe.gate","model.layers.49.block_sparse_moe.gate","model.layers.50.block_sparse_moe.gate","model.layers.51.block_sparse_moe.gate","model.layers.52.block_sparse_moe.gate","model.layers.53.block_sparse_moe.gate","model.layers.54.block_sparse_moe.gate","model.layers.55.block_sparse_moe.gate","model.layers.56.block_sparse_moe.gate","model.layers.57.block_sparse_moe.gate","model.layers.58.block_sparse_moe.gate","model.layers.59.block_sparse_moe.gate","model.layers.60.block_sparse_moe.gate","model.layers.61.block_sparse_moe.gate","lm_head"],"kv_cache_scheme":{"type":"float","num_bits":8,"strategy":"tensor","symmetric":true,"dynamic":false},"quant_method":"compressed-tensors","quantization_status":"compressed","sparsity_config":{},"transform_config":{},"version":"0.14.0.1"}}' \
  --kv-cache-dtype bfloat16 \
  --trust-remote-code \
  --page-size 64 \
  --dtype bfloat16 \
  --tp-size 8 \
  --pp-size 1 \
  --dp-size 1 \
  --served-model-name hygon/MiniMax-M2.5-Channel-INT8-w8a8 \
  --tool-call-parser minimax-m2 \
  --reasoning-parser minimax-append-think \
  --mem-fraction-static 0.9 \
  --attention-backend triton
```

#### D node

```bash
export USE_DCU_CUSTOM_ALLREDUCE=1
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_USE_LIGHTOP=1
export SGLANG_TORCH_PROFILER_DIR=/home/profile
export SGLANG_ROCM_USE_AITER_MOE=0
export VLLM_USE_LIGHTOP_MOE_ALIGN=1
export LMSLIM_USE_LIGHTOP=1
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=0
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export SGLANG_KV_LAYOUT_HCU_FA=1
export SGLANG_HOST_IP=12.12.12.47
export NCCL_SOCKET_IFNAME=eth2
export GLOO_SOCKET_IFNAME=eth2
export NCCL_IB_HCA=mlx5_0
export MC_GID_INDEX=3
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export ROCSHMEM_DISABLE_HDP_FLUSH=1
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_HEAP_SIZE=3173741824
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export ROCSHMEM_IB_GID_INDEX=0

sglang serve \
  --model-path hygon/MiniMax-M2.5-Channel-INT8-w8a8 \
  --numa-node 0 0 0 0 1 1 1 1 \
  --max-running-requests 512 \
  --context-length 131072 \
  --host 12.12.12.47 \
  --port 30000 \
  --quantization slimquant_marlin \
  --json-model-override-args '{"quantization_config":{"config_groups":{"group_0":{"format":"int-quantized","input_activations":{"actorder":null,"block_structure":null,"dynamic":true,"group_size":null,"num_bits":8,"observer":null,"observer_kwargs":{},"scale_dtype":null,"strategy":"token","symmetric":true,"type":"int","zp_dtype":null},"output_activations":null,"targets":["Linear"],"weights":{"actorder":null,"block_structure":null,"dynamic":false,"group_size":null,"num_bits":8,"observer":"memoryless_minmax","observer_kwargs":{},"scale_dtype":null,"strategy":"channel","symmetric":true,"type":"int","zp_dtype":null}}},"format":"int-quantized","global_compression_ratio":null,"ignore":["model.layers.0.block_sparse_moe.gate","model.layers.1.block_sparse_moe.gate","model.layers.2.block_sparse_moe.gate","model.layers.3.block_sparse_moe.gate","model.layers.4.block_sparse_moe.gate","model.layers.5.block_sparse_moe.gate","model.layers.6.block_sparse_moe.gate","model.layers.7.block_sparse_moe.gate","model.layers.8.block_sparse_moe.gate","model.layers.9.block_sparse_moe.gate","model.layers.10.block_sparse_moe.gate","model.layers.11.block_sparse_moe.gate","model.layers.12.block_sparse_moe.gate","model.layers.13.block_sparse_moe.gate","model.layers.14.block_sparse_moe.gate","model.layers.15.block_sparse_moe.gate","model.layers.16.block_sparse_moe.gate","model.layers.17.block_sparse_moe.gate","model.layers.18.block_sparse_moe.gate","model.layers.19.block_sparse_moe.gate","model.layers.20.block_sparse_moe.gate","model.layers.21.block_sparse_moe.gate","model.layers.22.block_sparse_moe.gate","model.layers.23.block_sparse_moe.gate","model.layers.24.block_sparse_moe.gate","model.layers.25.block_sparse_moe.gate","model.layers.26.block_sparse_moe.gate","model.layers.27.block_sparse_moe.gate","model.layers.28.block_sparse_moe.gate","model.layers.29.block_sparse_moe.gate","model.layers.30.block_sparse_moe.gate","model.layers.31.block_sparse_moe.gate","model.layers.32.block_sparse_moe.gate","model.layers.33.block_sparse_moe.gate","model.layers.34.block_sparse_moe.gate","model.layers.35.block_sparse_moe.gate","model.layers.36.block_sparse_moe.gate","model.layers.37.block_sparse_moe.gate","model.layers.38.block_sparse_moe.gate","model.layers.39.block_sparse_moe.gate","model.layers.40.block_sparse_moe.gate","model.layers.41.block_sparse_moe.gate","model.layers.42.block_sparse_moe.gate","model.layers.43.block_sparse_moe.gate","model.layers.44.block_sparse_moe.gate","model.layers.45.block_sparse_moe.gate","model.layers.46.block_sparse_moe.gate","model.layers.47.block_sparse_moe.gate","model.layers.48.block_sparse_moe.gate","model.layers.49.block_sparse_moe.gate","model.layers.50.block_sparse_moe.gate","model.layers.51.block_sparse_moe.gate","model.layers.52.block_sparse_moe.gate","model.layers.53.block_sparse_moe.gate","model.layers.54.block_sparse_moe.gate","model.layers.55.block_sparse_moe.gate","model.layers.56.block_sparse_moe.gate","model.layers.57.block_sparse_moe.gate","model.layers.58.block_sparse_moe.gate","model.layers.59.block_sparse_moe.gate","model.layers.60.block_sparse_moe.gate","model.layers.61.block_sparse_moe.gate","lm_head"],"kv_cache_scheme":{"type":"float","num_bits":8,"strategy":"tensor","symmetric":true,"dynamic":false},"quant_method":"compressed-tensors","quantization_status":"compressed","sparsity_config":{},"transform_config":{},"version":"0.14.0.1"}}' \
  --kv-cache-dtype bfloat16 \
  --trust-remote-code \
  --page-size 64 \
  --dtype bfloat16 \
  --tp-size 8 \
  --pp-size 1 \
  --dp-size 8 \
  --served-model-name hygon/MiniMax-M2.5-Channel-INT8-w8a8 \
  --tool-call-parser minimax-m2 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --deepep-mode auto \
  --reasoning-parser minimax-append-think \
  --mem-fraction-static 0.9 \
  --attention-backend triton \
  --disaggregation-mode decode \
  --prefill-round-robin-balance \
  --disaggregation-transfer-backend mooncake_tcp
```

#### Router

```bash
python3 -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://12.12.12.46:30000 \
  --decode http://12.12.12.47:30000 \
  --policy cache_aware \
  --port 30001
```

### MiniMax-M2.5-Channel-INT8-w8a8 IFB BW1000 8x SGLang 0.5.18

```bash
export USE_DCU_CUSTOM_ALLREDUCE=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export HSA_FORCE_FINE_GRAIN_PCIE=1
unset NCCL_TOPO_FILE
export SGLANG_USE_LIGHTOP=1
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=0
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export LMSLIM_USE_LIGHTOP=1
export VLLM_USE_LIGHTOP_MOE_ALIGN=1
export SGLANG_KV_LAYOUT_HCU_FA=1
export SGLANG_USE_FUSED_RMS_QUANT=0
export SGLANG_USE_FP32GEMM_GATE_CUSTOM=1

sglang serve \
  --model-path hygon/MiniMax-M2.5-Channel-INT8-w8a8 \
  --quantization slimquant_marlin \
  --json-model-override-args '{"quantization_config":{"config_groups":{"group_0":{"format":"int-quantized","input_activations":{"actorder":null,"block_structure":null,"dynamic":true,"group_size":null,"num_bits":8,"observer":null,"observer_kwargs":{},"scale_dtype":null,"strategy":"token","symmetric":true,"type":"int","zp_dtype":null},"output_activations":null,"targets":["Linear"],"weights":{"actorder":null,"block_structure":null,"dynamic":false,"group_size":null,"num_bits":8,"observer":"memoryless_minmax","observer_kwargs":{},"scale_dtype":null,"strategy":"channel","symmetric":true,"type":"int","zp_dtype":null}}},"format":"int-quantized","global_compression_ratio":null,"ignore":["model.layers.0.block_sparse_moe.gate","model.layers.1.block_sparse_moe.gate","model.layers.2.block_sparse_moe.gate","model.layers.3.block_sparse_moe.gate","model.layers.4.block_sparse_moe.gate","model.layers.5.block_sparse_moe.gate","model.layers.6.block_sparse_moe.gate","model.layers.7.block_sparse_moe.gate","model.layers.8.block_sparse_moe.gate","model.layers.9.block_sparse_moe.gate","model.layers.10.block_sparse_moe.gate","model.layers.11.block_sparse_moe.gate","model.layers.12.block_sparse_moe.gate","model.layers.13.block_sparse_moe.gate","model.layers.14.block_sparse_moe.gate","model.layers.15.block_sparse_moe.gate","model.layers.16.block_sparse_moe.gate","model.layers.17.block_sparse_moe.gate","model.layers.18.block_sparse_moe.gate","model.layers.19.block_sparse_moe.gate","model.layers.20.block_sparse_moe.gate","model.layers.21.block_sparse_moe.gate","model.layers.22.block_sparse_moe.gate","model.layers.23.block_sparse_moe.gate","model.layers.24.block_sparse_moe.gate","model.layers.25.block_sparse_moe.gate","model.layers.26.block_sparse_moe.gate","model.layers.27.block_sparse_moe.gate","model.layers.28.block_sparse_moe.gate","model.layers.29.block_sparse_moe.gate","model.layers.30.block_sparse_moe.gate","model.layers.31.block_sparse_moe.gate","model.layers.32.block_sparse_moe.gate","model.layers.33.block_sparse_moe.gate","model.layers.34.block_sparse_moe.gate","model.layers.35.block_sparse_moe.gate","model.layers.36.block_sparse_moe.gate","model.layers.37.block_sparse_moe.gate","model.layers.38.block_sparse_moe.gate","model.layers.39.block_sparse_moe.gate","model.layers.40.block_sparse_moe.gate","model.layers.41.block_sparse_moe.gate","model.layers.42.block_sparse_moe.gate","model.layers.43.block_sparse_moe.gate","model.layers.44.block_sparse_moe.gate","model.layers.45.block_sparse_moe.gate","model.layers.46.block_sparse_moe.gate","model.layers.47.block_sparse_moe.gate","model.layers.48.block_sparse_moe.gate","model.layers.49.block_sparse_moe.gate","model.layers.50.block_sparse_moe.gate","model.layers.51.block_sparse_moe.gate","model.layers.52.block_sparse_moe.gate","model.layers.53.block_sparse_moe.gate","model.layers.54.block_sparse_moe.gate","model.layers.55.block_sparse_moe.gate","model.layers.56.block_sparse_moe.gate","model.layers.57.block_sparse_moe.gate","model.layers.58.block_sparse_moe.gate","model.layers.59.block_sparse_moe.gate","model.layers.60.block_sparse_moe.gate","model.layers.61.block_sparse_moe.gate","lm_head"],"kv_cache_scheme":{"type":"float","num_bits":8,"strategy":"tensor","symmetric":true,"dynamic":false},"quant_method":"compressed-tensors","quantization_status":"compressed","sparsity_config":{},"transform_config":{},"version":"0.14.0.1"}}' \
  --trust-remote-code \
  --page-size 64 \
  --dtype bfloat16 \
  --kv-cache-dtype bfloat16 \
  --tp-size 8 \
  --pp-size 1 \
  --data-parallel-size 1 \
  --tool-call-parser minimax-m2 \
  --reasoning-parser minimax-append-think \
  --mem-fraction-static 0.9 \
  --attention-backend triton \
  --chunked-prefill-size 16384 \
  --max-running-requests 512 \
  --context-length 131072
```

### MiniMax-M2.5-Channel-INT8-w8a8 1P1D BW1000 16x SGLang 0.5.18

网卡配置参考：[IB 网卡](../../troubleshooting/common-issues.md#ib网卡)。

#### P node

```bash
export USE_DCU_CUSTOM_ALLREDUCE=1
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export HSA_FORCE_FINE_GRAIN_PCIE=1
export SGLANG_ROCM_USE_AITER_MOE=0
export SGLANG_USE_LIGHTOP=1
export SGLANG_TORCH_PROFILER_DIR=/home/profile
export VLLM_USE_LIGHTOP_MOE_ALIGN=1
export LMSLIM_USE_LIGHTOP=1
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=0
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export SGLANG_KV_LAYOUT_HCU_FA=1
export SGLANG_HOST_IP=12.12.12.46
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=mlx5_0
export MC_GID_INDEX=3
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export ROCSHMEM_DISABLE_HDP_FLUSH=1
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_HEAP_SIZE=3173741824
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export ROCSHMEM_IB_GID_INDEX=0

sglang serve \
  --model-path hygon/MiniMax-M2.5-Channel-INT8-w8a8 \
  --numa-node 0 0 0 0 1 1 1 1 \
  --max-running-requests 512 \
  --context-length 131072 \
  --disaggregation-mode prefill \
  --load-balance-method round_robin \
  --host 12.12.12.46 \
  --port 30000 \
  --disaggregation-transfer-backend mooncake_tcp \
  --quantization slimquant_marlin \
  --json-model-override-args '{"quantization_config":{"config_groups":{"group_0":{"format":"int-quantized","input_activations":{"actorder":null,"block_structure":null,"dynamic":true,"group_size":null,"num_bits":8,"observer":null,"observer_kwargs":{},"scale_dtype":null,"strategy":"token","symmetric":true,"type":"int","zp_dtype":null},"output_activations":null,"targets":["Linear"],"weights":{"actorder":null,"block_structure":null,"dynamic":false,"group_size":null,"num_bits":8,"observer":"memoryless_minmax","observer_kwargs":{},"scale_dtype":null,"strategy":"channel","symmetric":true,"type":"int","zp_dtype":null}}},"format":"int-quantized","global_compression_ratio":null,"ignore":["model.layers.0.block_sparse_moe.gate","model.layers.1.block_sparse_moe.gate","model.layers.2.block_sparse_moe.gate","model.layers.3.block_sparse_moe.gate","model.layers.4.block_sparse_moe.gate","model.layers.5.block_sparse_moe.gate","model.layers.6.block_sparse_moe.gate","model.layers.7.block_sparse_moe.gate","model.layers.8.block_sparse_moe.gate","model.layers.9.block_sparse_moe.gate","model.layers.10.block_sparse_moe.gate","model.layers.11.block_sparse_moe.gate","model.layers.12.block_sparse_moe.gate","model.layers.13.block_sparse_moe.gate","model.layers.14.block_sparse_moe.gate","model.layers.15.block_sparse_moe.gate","model.layers.16.block_sparse_moe.gate","model.layers.17.block_sparse_moe.gate","model.layers.18.block_sparse_moe.gate","model.layers.19.block_sparse_moe.gate","model.layers.20.block_sparse_moe.gate","model.layers.21.block_sparse_moe.gate","model.layers.22.block_sparse_moe.gate","model.layers.23.block_sparse_moe.gate","model.layers.24.block_sparse_moe.gate","model.layers.25.block_sparse_moe.gate","model.layers.26.block_sparse_moe.gate","model.layers.27.block_sparse_moe.gate","model.layers.28.block_sparse_moe.gate","model.layers.29.block_sparse_moe.gate","model.layers.30.block_sparse_moe.gate","model.layers.31.block_sparse_moe.gate","model.layers.32.block_sparse_moe.gate","model.layers.33.block_sparse_moe.gate","model.layers.34.block_sparse_moe.gate","model.layers.35.block_sparse_moe.gate","model.layers.36.block_sparse_moe.gate","model.layers.37.block_sparse_moe.gate","model.layers.38.block_sparse_moe.gate","model.layers.39.block_sparse_moe.gate","model.layers.40.block_sparse_moe.gate","model.layers.41.block_sparse_moe.gate","model.layers.42.block_sparse_moe.gate","model.layers.43.block_sparse_moe.gate","model.layers.44.block_sparse_moe.gate","model.layers.45.block_sparse_moe.gate","model.layers.46.block_sparse_moe.gate","model.layers.47.block_sparse_moe.gate","model.layers.48.block_sparse_moe.gate","model.layers.49.block_sparse_moe.gate","model.layers.50.block_sparse_moe.gate","model.layers.51.block_sparse_moe.gate","model.layers.52.block_sparse_moe.gate","model.layers.53.block_sparse_moe.gate","model.layers.54.block_sparse_moe.gate","model.layers.55.block_sparse_moe.gate","model.layers.56.block_sparse_moe.gate","model.layers.57.block_sparse_moe.gate","model.layers.58.block_sparse_moe.gate","model.layers.59.block_sparse_moe.gate","model.layers.60.block_sparse_moe.gate","model.layers.61.block_sparse_moe.gate","lm_head"],"kv_cache_scheme":{"type":"float","num_bits":8,"strategy":"tensor","symmetric":true,"dynamic":false},"quant_method":"compressed-tensors","quantization_status":"compressed","sparsity_config":{},"transform_config":{},"version":"0.14.0.1"}}' \
  --kv-cache-dtype bfloat16 \
  --trust-remote-code \
  --page-size 64 \
  --dtype bfloat16 \
  --tp-size 8 \
  --pp-size 1 \
  --dp-size 1 \
  --served-model-name hygon/MiniMax-M2.5-Channel-INT8-w8a8 \
  --tool-call-parser minimax-m2 \
  --reasoning-parser minimax-append-think \
  --mem-fraction-static 0.9 \
  --attention-backend triton
```

#### D node

```bash
export USE_DCU_CUSTOM_ALLREDUCE=1
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_USE_LIGHTOP=1
export SGLANG_TORCH_PROFILER_DIR=/home/profile
export SGLANG_ROCM_USE_AITER_MOE=0
export VLLM_USE_LIGHTOP_MOE_ALIGN=1
export LMSLIM_USE_LIGHTOP=1
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=0
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export SGLANG_KV_LAYOUT_HCU_FA=1
export SGLANG_HOST_IP=12.12.12.47
export NCCL_SOCKET_IFNAME=eth2
export GLOO_SOCKET_IFNAME=eth2
export NCCL_IB_HCA=mlx5_0
export MC_GID_INDEX=3
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export ROCSHMEM_DISABLE_HDP_FLUSH=1
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_HEAP_SIZE=3173741824
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export ROCSHMEM_IB_GID_INDEX=0

sglang serve \
  --model-path hygon/MiniMax-M2.5-Channel-INT8-w8a8 \
  --numa-node 0 0 0 0 1 1 1 1 \
  --max-running-requests 512 \
  --context-length 131072 \
  --host 12.12.12.47 \
  --port 30000 \
  --quantization slimquant_marlin \
  --json-model-override-args '{"quantization_config":{"config_groups":{"group_0":{"format":"int-quantized","input_activations":{"actorder":null,"block_structure":null,"dynamic":true,"group_size":null,"num_bits":8,"observer":null,"observer_kwargs":{},"scale_dtype":null,"strategy":"token","symmetric":true,"type":"int","zp_dtype":null},"output_activations":null,"targets":["Linear"],"weights":{"actorder":null,"block_structure":null,"dynamic":false,"group_size":null,"num_bits":8,"observer":"memoryless_minmax","observer_kwargs":{},"scale_dtype":null,"strategy":"channel","symmetric":true,"type":"int","zp_dtype":null}}},"format":"int-quantized","global_compression_ratio":null,"ignore":["model.layers.0.block_sparse_moe.gate","model.layers.1.block_sparse_moe.gate","model.layers.2.block_sparse_moe.gate","model.layers.3.block_sparse_moe.gate","model.layers.4.block_sparse_moe.gate","model.layers.5.block_sparse_moe.gate","model.layers.6.block_sparse_moe.gate","model.layers.7.block_sparse_moe.gate","model.layers.8.block_sparse_moe.gate","model.layers.9.block_sparse_moe.gate","model.layers.10.block_sparse_moe.gate","model.layers.11.block_sparse_moe.gate","model.layers.12.block_sparse_moe.gate","model.layers.13.block_sparse_moe.gate","model.layers.14.block_sparse_moe.gate","model.layers.15.block_sparse_moe.gate","model.layers.16.block_sparse_moe.gate","model.layers.17.block_sparse_moe.gate","model.layers.18.block_sparse_moe.gate","model.layers.19.block_sparse_moe.gate","model.layers.20.block_sparse_moe.gate","model.layers.21.block_sparse_moe.gate","model.layers.22.block_sparse_moe.gate","model.layers.23.block_sparse_moe.gate","model.layers.24.block_sparse_moe.gate","model.layers.25.block_sparse_moe.gate","model.layers.26.block_sparse_moe.gate","model.layers.27.block_sparse_moe.gate","model.layers.28.block_sparse_moe.gate","model.layers.29.block_sparse_moe.gate","model.layers.30.block_sparse_moe.gate","model.layers.31.block_sparse_moe.gate","model.layers.32.block_sparse_moe.gate","model.layers.33.block_sparse_moe.gate","model.layers.34.block_sparse_moe.gate","model.layers.35.block_sparse_moe.gate","model.layers.36.block_sparse_moe.gate","model.layers.37.block_sparse_moe.gate","model.layers.38.block_sparse_moe.gate","model.layers.39.block_sparse_moe.gate","model.layers.40.block_sparse_moe.gate","model.layers.41.block_sparse_moe.gate","model.layers.42.block_sparse_moe.gate","model.layers.43.block_sparse_moe.gate","model.layers.44.block_sparse_moe.gate","model.layers.45.block_sparse_moe.gate","model.layers.46.block_sparse_moe.gate","model.layers.47.block_sparse_moe.gate","model.layers.48.block_sparse_moe.gate","model.layers.49.block_sparse_moe.gate","model.layers.50.block_sparse_moe.gate","model.layers.51.block_sparse_moe.gate","model.layers.52.block_sparse_moe.gate","model.layers.53.block_sparse_moe.gate","model.layers.54.block_sparse_moe.gate","model.layers.55.block_sparse_moe.gate","model.layers.56.block_sparse_moe.gate","model.layers.57.block_sparse_moe.gate","model.layers.58.block_sparse_moe.gate","model.layers.59.block_sparse_moe.gate","model.layers.60.block_sparse_moe.gate","model.layers.61.block_sparse_moe.gate","lm_head"],"kv_cache_scheme":{"type":"float","num_bits":8,"strategy":"tensor","symmetric":true,"dynamic":false},"quant_method":"compressed-tensors","quantization_status":"compressed","sparsity_config":{},"transform_config":{},"version":"0.14.0.1"}}' \
  --kv-cache-dtype bfloat16 \
  --trust-remote-code \
  --page-size 64 \
  --dtype bfloat16 \
  --tp-size 8 \
  --pp-size 1 \
  --dp-size 8 \
  --served-model-name hygon/MiniMax-M2.5-Channel-INT8-w8a8 \
  --tool-call-parser minimax-m2 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --deepep-mode auto \
  --reasoning-parser minimax-append-think \
  --mem-fraction-static 0.9 \
  --attention-backend triton \
  --disaggregation-mode decode \
  --prefill-round-robin-balance \
  --disaggregation-transfer-backend mooncake_tcp
```

#### Router

```bash
python3 -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://12.12.12.46:30000 \
  --decode http://12.12.12.47:30000 \
  --policy cache_aware \
  --port 30001
```

### MiniMax-M2.5-Channel-INT8-w8a8 IFB BW1100 8x SGLang 0.5.12

```bash
export USE_DCU_CUSTOM_ALLREDUCE=1
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_USE_LIGHTOP=1
export VLLM_USE_LIGHTOP_MOE_ALIGN=1
export LMSLIM_USE_LIGHTOP=1
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export ALLREDUCE_STREAM_WITH_COMPUTE=1

sglang serve \
  --model-path hygon/MiniMax-M2.5-Channel-INT8-w8a8 \
  --quantization slimquant_marlin \
  --kv-cache-dtype fp8_e4m3 \
  --trust-remote-code \
  --page-size 64 \
  --dtype bfloat16 \
  --tp-size 4 --pp-size 1 --data-parallel-size 2 \
  --tool-call-parser minimax-m2 \
  --reasoning-parser minimax-append-think \
  --mem-fraction-static 0.9 \
  --attention-backend fa3 \
  --chunked-prefill-size 16384 \
  --max-running-requests 512 \
  --context-length 131072
```

### MiniMax-M2.5-Channel-INT8-w8a8 IFB BW1000 8x SGLang 0.5.12

```bash
export USE_DCU_CUSTOM_ALLREDUCE=1
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_USE_LIGHTOP=1
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=0
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export LMSLIM_USE_LIGHTOP=1
export VLLM_USE_LIGHTOP_MOE_ALIGN=1
export SGLANG_KV_LAYOUT_HCU_FA=1
export SGLANG_USE_FUSED_RMS_QUANT=1
export SGLANG_USE_FP32GEMM_GATE_CUSTOM=1

sglang serve \
  --model-path hygon/MiniMax-M2.5-Channel-INT8-w8a8 \
  --quantization slimquant_marlin \
  --trust-remote-code \
  --page-size 64 \
  --dtype bfloat16 \
  --tp-size 8 --pp-size 1 --data-parallel-size 1 \
  --tool-call-parser minimax-m2 \
  --reasoning-parser minimax-append-think \
  --mem-fraction-static 0.9 \
  --attention-backend fa3 \
  --chunked-prefill-size 16384 \
  --max-running-requests 512 \
  --context-length 131072
```

### MiniMax-M2-5-Channel-INT8-w8a8 IFB BW1100 8x

网卡配置参考：[mlx]

```bash
export SGLANG_USE_MODELSCOPE=1
export USE_DCU_CUSTOM_ALLREDUCE=1
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_USE_LIGHTOP=1 
export VLLM_USE_LIGHTOP_MOE_ALIGN=1
export LMSLIM_USE_LIGHTOP=1
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export ALLREDUCE_STREAM_WITH_COMPUTE=1

sglang serve \
  --model-path hygon/MiniMax-M2.5-Channel-INT8-w8a8 \
  --quantization slimquant_marlin \
  --kv-cache-dtype fp8_e4m3 \
  --trust-remote-code \
  --page-size 64 \
  --dtype bfloat16 \
  --tp-size 4 --pp-size 1 --dp-size 2 \
  --tool-call-parser minimax-m2 \
  --reasoning-parser minimax-append-think \
  --mem-fraction-static 0.9 \
  --attention-backend fa3 \
  --numa-node 0 0 0 0 1 1 1 1 \
  --chunked-prefill-size 16384 \
  --max-running-requests 512 \
  --context-length 131072 \
  --port 30000
```

### MiniMax-M2-5-Channel-INT8-w8a8 IFB BW1000 8x

网卡配置参考：[mlx]

```bash
export SGLANG_USE_MODELSCOPE=1
export USE_DCU_CUSTOM_ALLREDUCE=1
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_USE_LIGHTOP=1 
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export LMSLIM_USE_LIGHTOP=1
export VLLM_USE_LIGHTOP_MOE_ALIGN=1

sglang serve \
  --model-path hygon/MiniMax-M2.5-Channel-INT8-w8a8 \
  --quantization slimquant_marlin \
  --kv-cache-dtype bfloat16 \
  --trust-remote-code \
  --page-size 64 \
  --dtype bfloat16 \
  --tp-size 8 --pp-size 1 --dp-size 1 \
  --tool-call-parser minimax-m2 \
  --reasoning-parser minimax-append-think \
  --mem-fraction-static 0.95  \
  --attention-backend fa3 \
  --numa-node 0 0 0 0 1 1 1 1 \
  --chunked-prefill-size 8192 \
  --max-running-requests 512 \
  --context-length 131072 \
  --port 30000
```

### MiniMax-M2.5-Channel-FP8-w8a8 1P1D scaleX40-3G 8x SGLang 0.5.12

网卡配置参考：[shca]

P/D 服务端口分别填写 `<P_port>` 和 `<D_port>`，Router 端口填写 `<router_port>`。

#### P node

```bash
export USE_DCU_CUSTOM_ALLREDUCE=0
export SGLANG_USE_AITER_AR=0
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_TORCH_PROFILER_DIR=/workspace/prof
export VLLM_USE_LIGHTOP_MOE_ALIGN=1
export LMSLIM_USE_LIGHTOP=1
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export NCCL_SIMPLE_CHANNELS=20
export RCCL_P2P_XHCL_CHANNEL_NUM=16
export RCCL_COLL_XHCL_CHANNEL_NUM=16
export HIP_H2D_DISABLE_COPY_BUFFER=0
export HIP_D2H_DISABLE_COPY_BUFFER=0
export HIP_H2D_DIRECT_COPY_THRESHOLD=32768
export HIP_H2D_HSAAPI_COPY_THRESHOLD=32768
export HIP_D2H_DIRECT_COPY_THRESHOLD=512
export HIP_D2H_HSAAPI_COPY_THRESHOLD=512
export HSA_KERNARG_POOL_SIZE=8388608
export HSA_FORCE_FINE_GRAIN_PCIE=1
export ROC_AQL_QUEUE_SIZE=131072
export NCCL_PLUGIN_P2P=ib
export NCCL_IB_DISABLE=0
unset NCCL_IB_GID_INDEX
export NCCL_NET_PLUGIN=shca
export NCCL_IB_HCA=shca_0:1,shca_1:1,shca_2:1,shca_4:1
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
unset MC_GID_INDEX
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export SGLANG_USE_LIGHTOP=1
export SGLANG_ROCM_USE_AITER_MOE=0
export W8A8_SUPPORT_METHODS=1
export HIP_GRAPH_ACCUMULATE_DISPATCH=1
export HIP_GRAPH_USE_CMD_CACHE=0
export MC_ALLOWED_IBV_DEVICES=shca_0,shca_1,shca_2,shca_4
export ROCSHMEM_ALLOWED_IBV_DEVICES=shca_0,shca_1,shca_2,shca_4
export SGLANG_USE_FUSED_RMS_QUANT=0

sglang serve \
  --model-path hygon/MiniMax-M2.5-Channel-FP8-w8a8 \
  --chunked-prefill-size 61568 \
  --max-prefill-tokens 61568 \
  --max-running-requests 256 \
  --context-length 131072 \
  --disaggregation-ib-device shca_0,shca_1,shca_2,shca_4 \
  --disaggregation-mode prefill \
  --load-balance-method round_robin \
  --host <P_node_ip> \
  --port <P_port> \
  --model-loader-extra-config '{"enable_multithread_load": "true","num_threads": 8}' \
  --quantization w8a8_fp8 \
  --kv-cache-dtype fp8_e4m3 \
  --trust-remote-code \
  --page-size 64 \
  --dtype bfloat16 \
  --tp-size 4 \
  --pp-size 1 \
  --dp-size 1 \
  --tool-call-parser minimax-m2 \
  --reasoning-parser minimax-append-think \
  --skip-server-warmup \
  --mem-fraction-static 0.9 \
  --attention-backend fa3
```

#### D node

```bash
export NCCL_NET_PLUGIN=shca
export NCCL_IB_HCA=shca_0:1,shca_1:1,shca_2:1,shca_4:1
export USE_DCU_CUSTOM_ALLREDUCE=0
export SGLANG_USE_AITER_AR=0
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_USE_LIGHTOP=1
export SGLANG_TORCH_PROFILER_DIR=/workspace/prof
export VLLM_USE_LIGHTOP_MOE_ALIGN=1
export LMSLIM_USE_LIGHTOP=1
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export NCCL_SIMPLE_CHANNELS=20
export RCCL_P2P_XHCL_CHANNEL_NUM=16
export RCCL_COLL_XHCL_CHANNEL_NUM=16
export SGLANG_USE_FP32GEMM_GATE_CUSTOM=1
export SGLANG_ROCM_USE_AITER_MOE=0
export MC_ALLOWED_IBV_DEVICES=shca_0,shca_1,shca_2,shca_4
export ROCSHMEM_ALLOWED_IBV_DEVICES=shca_0,shca_1,shca_2,shca_4

sglang serve \
  --model-path hygon/MiniMax-M2.5-Channel-FP8-w8a8 \
  --chunked-prefill-size 61568 \
  --max-prefill-tokens 61568 \
  --max-running-requests 512 \
  --context-length 131072 \
  --disaggregation-mode decode \
  --prefill-round-robin-balance \
  --host <D_node_ip> \
  --port <D_port> \
  --disaggregation-ib-device shca_0,shca_1,shca_2,shca_4 \
  --disable-radix-cache \
  --model-loader-extra-config '{"enable_multithread_load": "true","num_threads": 2}' \
  --quantization w8a8_fp8 \
  --kv-cache-dtype fp8_e4m3 \
  --trust-remote-code \
  --page-size 64 \
  --dtype bfloat16 \
  --tp-size 4 \
  --pp-size 1 \
  --dp-size 1 \
  --tool-call-parser minimax-m2 \
  --reasoning-parser minimax-append-think \
  --mem-fraction-static 0.90 \
  --attention-backend fa3 \
  --cuda-graph-max-bs 512
```

#### Router

```bash
python3 -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://<P_node_ip>:<P_port> \
  --decode http://<D_node_ip>:<D_port> \
  --policy cache_aware \
  --port <router_port>
```

### MiniMax-M2.5-Channel-FP8-w8a8 IFB BW1100 8x SGLang 0.5.12

```bash
export SGLANG_USE_MODELSCOPE=1
export USE_DCU_CUSTOM_ALLREDUCE=1
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_USE_LIGHTOP=1
export VLLM_USE_LIGHTOP_MOE_ALIGN=1
export LMSLIM_USE_LIGHTOP=1
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export ALLREDUCE_STREAM_WITH_COMPUTE=1

sglang serve \
  --model-path hygon/MiniMax-M2.5-Channel-FP8-w8a8 \
  --kv-cache-dtype fp8_e4m3 \
  --trust-remote-code \
  --quantization w8a8_fp8 \
  --page-size 64 \
  --dtype bfloat16 \
  --tp-size 4 --pp-size 1 --dp-size 2 \
  --tool-call-parser minimax-m2 \
  --reasoning-parser minimax-append-think \
  --mem-fraction-static 0.92 \
  --attention-backend fa3 \
  --numa-node 0 0 0 0 1 1 1 1 \
  --chunked-prefill-size 8192 \
  --max-running-requests 512 \
  --context-length 131072
```

### MiniMax-M2-5-Channel-FP8-w8a8 IFB BW1100 8x

网卡配置参考：[mlx]

```bash
export SGLANG_USE_MODELSCOPE=1
export USE_DCU_CUSTOM_ALLREDUCE=1
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_USE_LIGHTOP=1
export VLLM_USE_LIGHTOP_MOE_ALIGN=1
export LMSLIM_USE_LIGHTOP=1
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export ALLREDUCE_STREAM_WITH_COMPUTE=1

sglang serve \
  --model-path hygon/MiniMax-M2.5-Channel-FP8-w8a8/ \
  --kv-cache-dtype fp8_e4m3 \
  --trust-remote-code \
  --quantization w8a8_fp8 \
  --page-size 64 \
  --dtype bfloat16 \
  --tp-size 4 --pp-size 1 --dp-size 2 \
  --tool-call-parser minimax-m2 \
  --reasoning-parser minimax-append-think \
  --mem-fraction-static 0.92 \
  --attention-backend fa3 \
  --numa-node 0 0 0 0 1 1 1 1 \
  --chunked-prefill-size 8192 \
  --max-running-requests 512 \
  --context-length 131072 \
  --port 30000
```

### MiniMax-M2-5-Channel-INT8-w8a8 1P1D BW1100 16x

网卡配置参考：[mlx]

#### P node 0
```bash
export SGLANG_USE_MODELSCOPE=1
export USE_DCU_CUSTOM_ALLREDUCE=1
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_USE_LIGHTOP=1 
export VLLM_USE_LIGHTOP_MOE_ALIGN=1
export LMSLIM_USE_LIGHTOP=1
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export MC_GID_INDEX=0
export ALLREDUCE_STREAM_WITH_COMPUTE=1

sglang serve \
  --model-path hygon/MiniMax-M2.5-Channel-INT8-w8a8 \
  --quantization slimquant_marlin \
  --kv-cache-dtype fp8_e4m3 \
  --trust-remote-code \ 
  --page-size 64 \ 
  --dtype bfloat16 \ 
  --tp-size 2 --pp-size 4 --dp-size 1 \
  --tool-call-parser minimax-m2 \
  --reasoning-parser minimax-append-think \
  --mem-fraction-static 0.9 \ 
  --attention-backend fa3 \
  --numa-node 0 0 0 0 1 1 1 1 \ 
  --chunked-prefill-size 4096 \
  --max-running-requests 512 \ 
  --context-length 131072 \
  --disaggregation-mode prefill \
  --load-balance-method round_robin \
  --host 10.63.60.114 --port 30000 
```

#### D node 0
```bash
export SGLANG_USE_MODELSCOPE=1
export USE_DCU_CUSTOM_ALLREDUCE=1
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_USE_LIGHTOP=1 
export VLLM_USE_LIGHTOP_MOE_ALIGN=1
export LMSLIM_USE_LIGHTOP=1
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export MC_GID_INDEX=0
export ALLREDUCE_STREAM_WITH_COMPUTE=1
 
sglang serve \
  --model-path hygon/MiniMax-M2.5-Channel-INT8-w8a8 \
  --quantization slimquant_marlin \
  --kv-cache-dtype fp8_e4m3 \
  --trust-remote-code \ 
  --page-size 64 \
  --dtype bfloat16 \ 
  --tp-size 8 --pp-size 1 --dp-size 1 \
  --tool-call-parser minimax-m2 \
  --reasoning-parser minimax-append-think\
  --mem-fraction-static 0.9 --attention-backend fa3 \ 
  --numa-node 0 0 0 0 1 1 1 1 \ 
  --max-running-requests 512 \
  --context-length 131072 \
  --disaggregation-mode decode \
  --prefill-round-robin-balance \ 
  --host 10.63.60.113 --port 30001 
```

### MiniMax-M2-5-Channel-INT8-w8a8 1P1D BW1000 16x

网卡配置参考：[shca]

#### P node 0

```bash
export SGLANG_USE_MODELSCOPE=1
export USE_DCU_CUSTOM_ALLREDUCE=1
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_USE_LIGHTOP=1 
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export MC_ALLOWED_IBV_DEVICES=shca_0,shca_1,shca_2,shca_3
export NCCL_TOPO_FILE="/home/built-in-508-topo-input-tj-default.xml"
export LD_LIBRARY_PATH=/home/lib:$LD_LIBRARY_PATH
export ROCSHMEM_DISABLE_HDP_FLUSH=1
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_HEAP_SIZE=3173741824
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export ROCSHMEM_ALLOWED_IBV_DEVICES=shca_0,shca_1,shca_2,shca_3
export LMSLIM_USE_LIGHTOP=1
export VLLM_USE_LIGHTOP_MOE_ALIGN=1
export MC_ENABLE_DEST_DEVICE_AFFINITY=1

sglang serve \
  --model-path hygon/MiniMax-M2.5-Channel-INT8-w8a8 \
  --quantization slimquant_marlin \ 
  --kv-cache-dtype bfloat16 \
  --trust-remote-code \ 
  --page-size 64 \
  --dtype bfloat16 \ 
  --tp-size 8 --pp-size 1 --dp-size 1 \
  --tool-call-parser minimax-m2 \
  --reasoning-parser minimax-append-think \
  --mem-fraction-static 0.9 \
  --attention-backend fa3 \
  --numa-node 0 0 0 0 1 1 1 1 \
  --max-running-requests 512 \
  --context-length 131072 \
  --disaggregation-ib-device shca_0,shca_1,shca_2,shca_3 \
  --disaggregation-mode prefill \
  --load-balance-method round_robin \ 
  --host 10.212.16.171 --port 30000
```

#### D node 0
```bash
export SGLANG_USE_MODELSCOPE=1
export USE_DCU_CUSTOM_ALLREDUCE=1
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_USE_LIGHTOP=1 
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export MC_ALLOWED_IBV_DEVICES=shca_0,shca_1,shca_2,shca_3
export NCCL_TOPO_FILE="/home/topo_lib/built-in-508-topo-input-tj-default.xml"
export LD_LIBRARY_PATH=/home/topo_lib/lib:$LD_LIBRARY_PATH
export ROCSHMEM_DISABLE_HDP_FLUSH=1
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_HEAP_SIZE=3173741824
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export ROCSHMEM_ALLOWED_IBV_DEVICES=shca_0,shca_1,shca_2,shca_3
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export LMSLIM_USE_LIGHTOP=1
export VLLM_USE_LIGHTOP_MOE_ALIGN=1

sglang serve \
  --model-path hygon/MiniMax-M2.5-Channel-INT8-w8a8 \
  --quantization slimquant_marlin \ 
  --kv-cache-dtype bfloat16 \
  --trust-remote-code \ 
  --page-size 64 \
  --dtype bfloat16 \ 
  --tp-size 8 --pp-size 1 --dp-size 1 \
  --tool-call-parser minimax-m2 \
  --reasoning-parser minimax-append-think \
  --mem-fraction-static 0.9 \ 
  --attention-backend fa3 \
  --numa-node 0 0 0 0 1 1 1 1 \
  --chunked-prefill-size 16384 \
  --max-running-requests 512 \
  --context-length 131027 \
  --disaggregation-mode decode \
  --prefill-round-robin-balance \ 
  --host 10.212.16.172 --port 30001 \
  --disaggregation-ib-device shca_0,shca_1,shca_2,shca_3
```

### MiniMax-M2-5-Channel-FP8-w8a8 1P1D BW1100 16x

网卡配置参考：[mlx]

#### P节点 
```bash
export SGLANG_USE_MODELSCOPE=1
export USE_DCU_CUSTOM_ALLREDUCE=1
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_USE_LIGHTOP=1 
export VLLM_USE_LIGHTOP_MOE_ALIGN=1
export LMSLIM_USE_LIGHTOP=1
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export MC_GID_INDEX=0
export ALLREDUCE_STREAM_WITH_COMPUTE=1

sglang serve \
  --model-path hygon/MiniMax-M2.5-Channel-FP8-w8a8/ \
  --quantization w8a8_fp8 \
  --kv-cache-dtype fp8_e4m3 \
  --trust-remote-code \ 
  --page-size 64 \ 
  --dtype bfloat16 \ 
  --tp-size 2 --pp-size 4 --dp-size 1 \
  --tool-call-parser minimax-m2 \
  --reasoning-parser minimax-append-think \
  --mem-fraction-static 0.9 \ 
  --attention-backend fa3 \
  --numa-node 0 0 0 0 1 1 1 1 \ 
  --chunked-prefill-size 4096 \
  --max-running-requests 512 \ 
  --context-length 131072 \
  --disaggregation-mode prefill \
  --load-balance-method round_robin \
  --host 10.63.60.114 --port 30000 
```

#### D节点
```bash
export SGLANG_USE_MODELSCOPE=1
export USE_DCU_CUSTOM_ALLREDUCE=1
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_USE_LIGHTOP=1 
export VLLM_USE_LIGHTOP_MOE_ALIGN=1
export LMSLIM_USE_LIGHTOP=1
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export MC_GID_INDEX=0
export ALLREDUCE_STREAM_WITH_COMPUTE=1
 
sglang serve \
  --model-path hygon/MiniMax-M2.5-Channel-FP8-w8a8/ \
  --quantization w8a8_fp8 \
  --kv-cache-dtype fp8_e4m3 \
  --trust-remote-code \ 
  --page-size 64 \
  --dtype bfloat16 \ 
  --tp-size 8 --pp-size 1 --dp-size 1 \
  --tool-call-parser minimax-m2 \
  --reasoning-parser minimax-append-think\
  --mem-fraction-static 0.9 --attention-backend fa3 \ 
  --numa-node 0 0 0 0 1 1 1 1 \ 
  --max-running-requests 512 \
  --context-length 131072 \
  --disaggregation-mode decode \
  --prefill-round-robin-balance \ 
  --host 10.63.60.113 --port 30001 
```

## Router
```python
python3 -m sglang_router.launch_router \
--pd-disaggregation \
--prefill http://10.63.60.114:30000 \
--decode http://10.63.60.113:30001 \
--policy cache_aware --port 30002
```

## API 调用

### IFB

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="hygon/MiniMax-M2.5-Channel-FP8-w8a8",  # 替换为实际使用的模型名
    messages=[
        {"role": "system", "content": "你是一个有帮助的 AI 助手。"},
        {"role": "user", "content": "请分析一下当前中国 AI 芯片产业的发展现状"},
    ],
    max_tokens=2048,
)
print(response.choices[0].message.content)
```

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "hygon/MiniMax-M2.5-Channel-FP8-w8a8",
    "messages": [
      {"role": "system", "content": "你是一个有帮助的 AI 助手。"},
      {"role": "user", "content": "中国的首都是哪里？"}
    ],
    "max_tokens": 128
  }'
```

### PD 分离

PD 分离模式下，客户端请求发送到 SGLang Router，而非直接发送到 P/D 节点。Router 默认端口为 `30002`，若与 P node 0 部署在同一机器上需指定其他端口。

```python
from openai import OpenAI

client = OpenAI(base_url="http://<router_ip>:30002/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="hygon/MiniMax-M2.5-Channel-FP8-w8a8",  # 替换为实际使用的模型名
    messages=[
        {"role": "system", "content": "你是一个有帮助的 AI 助手。"},
        {"role": "user", "content": "请分析一下当前中国 AI 芯片产业的发展现状"},
    ],
    max_tokens=2048,
)
print(response.choices[0].message.content)
```

```bash
curl http://<router_ip>:30002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "hygon/MiniMax-M2.5-Channel-FP8-w8a8",
    "messages": [
      {"role": "system", "content": "你是一个有帮助的 AI 助手。"},
      {"role": "user", "content": "中国的首都是哪里？"}
    ],
    "max_tokens": 128
  }'
```
