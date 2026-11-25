# Qwen3.8 on vLLM

## 模型简介

Qwen3.8 系列模型面向长上下文推理与工具调用场景，支持 vLLM 部署。

## 模型列表

| 模型权重 | 量化方式 | vLLM 镜像 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | --------- | -------- | ---- | -------- | -------- |
| [Qwen/Qwen3.8-Flash-Next](https://www.modelscope.cn/models/Qwen/Qwen3.8-Flash-Next) | BF16 | 0.28.1 | BW1100 | 8 | IFB | [**`>_`**](#qwen38-flash-next-ifb-bw1100-8x-vllm-0281) |
|  | BF16 | 0.28.1 | BW1000 | 8 | IFB | [**`>_`**](#qwen38-flash-next-ifb-bw1000-8x-vllm-0281) |
|  | BF16 | 0.28.1 | K100_AI | 8 | IFB | [**`>_`**](#qwen38-flash-next-ifb-k100_ai-8x-vllm-0281) |
| [hygon/Qwen3.8-Flash-Next-Channel-FP8](https://www.modelscope.cn/models/hygon/Qwen3.8-Flash-Next-Channel-FP8) | FP8 | 0.28.1 | BW1100 | 2 | IFB | [**`>_`**](#qwen38-flash-next-channel-fp8-ifb-bw1100-2x-vllm-0281) |
|  | FP8 | 0.28.1 | BW1100（96G） | 4 | IFB | [**`>_`**](#qwen38-flash-next-channel-fp8-ifb-bw110096g-4x-vllm-0281) |
| [hygon/Qwen3.8-Flash-Next-Channel-INT8-w8a8](https://www.modelscope.cn/models/hygon/Qwen3.8-Flash-Next-Channel-INT8-w8a8) | W8A8 | 0.28.1 | BW1100 | 2 | IFB | [**`>_`**](#qwen38-flash-next-channel-int8-w8a8-ifb-bw1100-2x-vllm-0281) |
|  | W8A8 | 0.28.1 | BW1100（96G） | 4 | IFB | [**`>_`**](#qwen38-flash-next-channel-int8-w8a8-ifb-bw110096g-4x-vllm-0281) |
|  | W8A8 | 0.28.1 | BW1000 | 4 | IFB | [**`>_`**](#qwen38-flash-next-channel-int8-w8a8-ifb-bw1000-4x-vllm-0281) |
|  | W8A8 | 0.28.1 | K100_AI | 4 | IFB | [**`>_`**](#qwen38-flash-next-channel-int8-w8a8-ifb-k100_ai-4x-vllm-0281) |
| [hygon/Qwen3.8-Flash-Next-Mixed-w4a8](https://www.modelscope.cn/models/hygon/Qwen3.8-Flash-Next-Mixed-w4a8) | W4A8 | 0.28.1 | BW1100 | 2 | IFB | [**`>_`**](#qwen38-flash-next-mixed-w4a8-ifb-bw1100-2x-vllm-0281) |
|  | W4A8 | 0.28.1 | BW1000 | 4 | IFB | [**`>_`**](#qwen38-flash-next-mixed-w4a8-ifb-bw1000-4x-vllm-0281) |
|  | W4A8 | 0.28.1 | K100_AI | 4 | IFB | [**`>_`**](#qwen38-flash-next-mixed-w4a8-ifb-k100_ai-4x-vllm-0281) |
| [Qwen/Qwen3.8-27B](https://www.modelscope.cn/models/Qwen/Qwen3.8-27B) | BF16 | 0.28.1 | BW1100 | 1 | IFB | [**`>_`**](#qwen38-27b-ifb-bw1100-1x-vllm-0281) |
|  | BF16 | 0.28.1 | BW1000 | 2 | IFB | [**`>_`**](#qwen38-27b-ifb-bw1000-2x-vllm-0281) |
|  | BF16 | 0.28.1 | K100_AI | 2 | IFB | [**`>_`**](#qwen38-27b-ifb-k100_ai-2x-vllm-0281) |
|  | BF16 |  [0.21](../docker_images.md) | BW1100 | 1 | IFB | [**`>_`**](#qwen38-27b-ifb-bw1100-1x-vllm-021) |
|  | BF16 |  [0.21](../docker_images.md) | BW1000 | 2 | IFB | [**`>_`**](#qwen38-27b-ifb-bw1000-2x-vllm-021) |
|  | BF16 | [0.21](../docker_images.md) | K100_AI | 2 | IFB | [**`>_`**](#qwen38-27b-ifb-k100_ai-2x-vllm-021) |
|  | BF16 | [0.18-hotfix](../docker_images.md) | BW1000 | 2 | IFB | [**`>_`**](#qwen38-27b-ifb-bw1000-2x-vllm-018-hotfix) |
| [hygon/Qwen3.8-27B-Channel-FP8](https://www.modelscope.cn/models/hygon/Qwen3.8-27B-Channel-FP8) | FP8 | 0.28.1 | BW1100 | 1 | IFB | [**`>_`**](#qwen38-27b-channel-fp8-ifb-bw1100-1x-vllm-0281) |
| [hygon/Qwen3.8-27B-Channel-INT8-w8a8](https://www.modelscope.cn/models/hygon/Qwen3.8-27B-Channel-INT8-w8a8) | INT8 W8A8 | 0.28.1 | BW1100 | 1 | IFB | [**`>_`**](#qwen38-27b-channel-int8-w8a8-ifb-bw1100-1x-vllm-0281) |
|  | INT8 W8A8 | 0.28.1 | BW1000 | 1 | IFB | [**`>_`**](#qwen38-27b-channel-int8-w8a8-ifb-bw1000-1x-vllm-0281) |
|  | INT8 W8A8 | 0.28.1 | K100_AI | 2 | IFB | [**`>_`**](#qwen38-27b-channel-int8-w8a8-ifb-k100_ai-2x-vllm-0281) |
|  | W8A8 |  [0.21](../docker_images.md) | BW1100 | 1 | IFB | [**`>_`**](#qwen38-27b-channel-int8-w8a8-ifb-bw1100-1x-vllm-021) |
|  | W8A8 | 0.21 | BW1000 | 1 | IFB | [**`>_`**](#qwen38-27b-channel-int8-w8a8-ifb-bw1000-1x-vllm-021) |
|  | W8A8 | 0.21 | BW1000 | 2 | IFB | [**`>_`**](#qwen38-27b-channel-int8-w8a8-ifb-bw1000-2x-vllm-021) |
|  | W8A8 | [0.21](../docker_images.md) | K100_AI | 2 | IFB | [**`>_`**](#qwen38-27b-channel-int8-w8a8-ifb-k100_ai-2x-vllm-021) |
|  | W8A8 | [0.18-hotfix](../docker_images.md) | BW1000 | 1 | IFB | [**`>_`**](#qwen38-27b-channel-int8-w8a8-ifb-bw1000-1x-vllm-018-hotfix) |
|  | W8A8 | [0.18-hotfix](../docker_images.md) | BW1000 | 2 | IFB | [**`>_`**](#qwen38-27b-channel-int8-w8a8-ifb-bw1000-2x-vllm-018-hotfix) |



### 0.28.1 平台支持说明

BW1000 不支持本次发版所列 FP8 权重模型；以下 FP8 权重仅提供 BW1100 启动命令：`hygon/Qwen3.8-27B-Channel-FP8`、`hygon/Qwen3.8-Flash-Next-Channel-FP8`。

## 启动命令

### Qwen3.8-Flash-Next IFB BW1100 8x vLLM 0.28.1

```bash
export VLLM_HCU_USE_AITER_MOE_SHUFFLE=0
export VLLM_HCU_QSA_BACKEND=triton
vllm serve Qwen/Qwen3.8-Flash-Next \
  --trust-remote-code \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --tensor-parallel-size 8 \
  --moe-backend aiter \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --prefix-match-unit 80 \
  --enable-mamba-fine-grained-prefix-cache \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
```

### Qwen3.8-Flash-Next IFB BW1000 8x vLLM 0.28.1

```bash
export VLLM_HCU_USE_AITER_MOE_SHUFFLE=0
export VLLM_HCU_PLE_PREFETCH_STREAM=1
vllm serve Qwen/Qwen3.8-Flash-Next \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --moe-backend aiter \
  --kv-cache-dtype fp8_e5m2 \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --prefix-match-unit 80 \
  --enable-mamba-fine-grained-prefix-cache \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --engram-config.cpu_offload=true \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --reasoning-parser qwen3 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
```

### Qwen3.8-Flash-Next IFB K100_AI 8x vLLM 0.28.1

```bash
export VLLM_HCU_USE_AITER_MOE_SHUFFLE=0
export VLLM_HCU_PLE_PREFETCH_STREAM=1
export VLLM_HCU_USE_CUSTOM_OPS=0

vllm serve Qwen/Qwen3.8-Flash-Next \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --moe-backend triton \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --prefix-match-unit 80 \
  --enable-mamba-fine-grained-prefix-cache \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --engram-config.cpu_offload=true \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --reasoning-parser qwen3 \
  --max-model-len 131072 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.85
```

### Qwen3.8-Flash-Next-Channel-FP8 IFB BW1100 2x vLLM 0.28.1

启动环境不要预设 `VLLM_KV_CACHE_LAYOUT=HND`，由后端选择支持的布局。

```bash

vllm serve hygon/Qwen3.8-Flash-Next-Channel-FP8 \
  --trust-remote-code \
  --engram-config.embedding_across_dp=true \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --tensor-parallel-size 2 \
  --attention-backend FLASH_ATTN \
  --moe-backend aiter \
  --kv-cache-dtype fp8_e4m3 \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --prefix-match-unit 64 \
  --enable-mamba-fine-grained-prefix-cache \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --gpu-memory-utilization 0.92 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
```

### Qwen3.8-Flash-Next-Channel-FP8 IFB BW1100（96G） 4x vLLM 0.28.1

```bash
vllm serve hygon/Qwen3.8-Flash-Next-Channel-FP8 \
  --trust-remote-code \
  --engram-config.embedding_across_dp=true \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --tensor-parallel-size 4 \
  --attention-backend FLASH_ATTN \
  --moe-backend aiter \
  --kv-cache-dtype fp8_e4m3 \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --prefix-match-unit 64 \
  --enable-mamba-fine-grained-prefix-cache \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --gpu-memory-utilization 0.92 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
```

### Qwen3.8-Flash-Next-Channel-INT8-w8a8 IFB BW1100 2x vLLM 0.28.1

```bash
export VLLM_HCU_USE_AITER_MOE_SHUFFLE=0
export VLLM_HCU_PLE_PREFETCH_STREAM=1

vllm serve hygon/Qwen3.8-Flash-Next-Channel-INT8-w8a8 \
  -tp 2 \
  --moe-backend aiter \
  --dtype bfloat16 \
  --trust-remote-code \
  --kv-cache-dtype fp8_e5m2 \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --prefix-match-unit 64 \
  --enable-mamba-fine-grained-prefix-cache \
  --engram-config.cpu_offload=true \
  --engram-config.embedding_across_dp=true \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
```

### Qwen3.8-Flash-Next-Channel-INT8-w8a8 IFB BW1100（96G） 4x vLLM 0.28.1

```bash
export VLLM_HCU_USE_AITER_MOE_SHUFFLE=0
export VLLM_HCU_PLE_PREFETCH_STREAM=1

vllm serve hygon/Qwen3.8-Flash-Next-Channel-INT8-w8a8 \
  --tensor-parallel-size 4 \
  --moe-backend aiter \
  --dtype bfloat16 \
  --trust-remote-code \
  --kv-cache-dtype fp8_e5m2 \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --prefix-match-unit 64 \
  --enable-mamba-fine-grained-prefix-cache \
  --engram-config.cpu_offload=true \
  --engram-config.embedding_across_dp=true \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
```

### Qwen3.8-Flash-Next-Channel-INT8-w8a8 IFB BW1000 4x vLLM 0.28.1

```bash
export VLLM_HCU_USE_AITER_MOE_SHUFFLE=0
export VLLM_HCU_PLE_PREFETCH_STREAM=1

vllm serve hygon/Qwen3.8-Flash-Next-Channel-INT8-w8a8 \
  -tp 4 \
  --moe-backend aiter \
  --dtype bfloat16 \
  --trust-remote-code \
  --kv-cache-dtype fp8_e5m2 \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --prefix-match-unit 64 \
  --enable-mamba-fine-grained-prefix-cache \
  --engram-config.cpu_offload=true \
  --engram-config.embedding_across_dp=true \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
```

### Qwen3.8-Flash-Next-Channel-INT8-w8a8 IFB K100_AI 4x vLLM 0.28.1

```bash
export VLLM_HCU_USE_AITER_MOE_SHUFFLE=0
export VLLM_HCU_PLE_PREFETCH_STREAM=1
export VLLM_HCU_USE_CUSTOM_OPS=0

vllm serve hygon/Qwen3.8-Flash-Next-Channel-INT8-w8a8 \
  --tensor-parallel-size 4 \
  --moe-backend triton \
  --dtype bfloat16 \
  --trust-remote-code \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --enable-mamba-fine-grained-prefix-cache \
  --engram-config.cpu_offload=true \
  --engram-config.embedding_across_dp=true \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.80
```

### Qwen3.8-Flash-Next-Mixed-w4a8 IFB BW1100 2x vLLM 0.28.1

```bash
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_HCU_PLE_PREFETCH_STREAM=1

vllm serve hygon/Qwen3.8-Flash-Next-Mixed-w4a8 \
  -tp 1 \
  --enable-expert-parallel \
  --moe-backend aiter \
  -dp 2 \
  --trust-remote-code \
  --engram-config.cpu_offload=true \
  --engram-config.embedding_across_dp=true \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --kv-cache-dtype fp8_e4m3 \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --prefix-match-unit 64 \
  --enable-mamba-fine-grained-prefix-cache \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
```

### Qwen3.8-Flash-Next-Mixed-w4a8 IFB BW1000 4x vLLM 0.28.1
```bash
export VLLM_HCU_PLE_PREFETCH_STREAM=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600

vllm serve hygon/Qwen3.8-Flash-Next-Mixed-w4a8 \
  -tp 1 \
  --enable-expert-parallel \
  --moe-backend aiter \
  -dp 4 \
  --trust-remote-code \
  --engram-config.cpu_offload=true \
  --engram-config.embedding_across_dp=true \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --kv-cache-dtype fp8_e5m2 \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --prefix-match-unit 64 \
  --enable-mamba-fine-grained-prefix-cache \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
```

### Qwen3.8-Flash-Next-Mixed-w4a8 IFB K100_AI 4x vLLM 0.28.1

```bash
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_HCU_PLE_PREFETCH_STREAM=1
export VLLM_HCU_USE_CUSTOM_OPS=0

vllm serve hygon/Qwen3.8-Flash-Next-Mixed-w4a8 \
  -tp 1 \
  --enable-expert-parallel \
  --moe-backend triton \
  -dp 4 \
  --trust-remote-code \
  --engram-config.cpu_offload=true \
  --engram-config.embedding_across_dp=true \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --enable-mamba-fine-grained-prefix-cache \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
```

### Qwen3.8-27B IFB BW1100 1x vLLM 0.28.1

```bash
export VLLM_KV_CACHE_LAYOUT=HND

vllm serve Qwen/Qwen3.8-27B \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --safetensors-load-strategy prefetch \
  --attention-backend FLASH_ATTN \
  --kv-cache-dtype fp8_e4m3 \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --prefix-match-unit 64 \
  --enable-mamba-fine-grained-prefix-cache \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
```

### Qwen3.8-27B IFB BW1000 2x vLLM 0.28.1

```bash
export VLLM_KV_CACHE_LAYOUT=HND

vllm serve Qwen/Qwen3.8-27B \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --safetensors-load-strategy prefetch \
  --attention-backend FLASH_ATTN \
  --kv-cache-dtype fp8_e5m2 \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --prefix-match-unit 64 \
  --enable-mamba-fine-grained-prefix-cache \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
```

### Qwen3.8-27B IFB K100_AI 2x vLLM 0.28.1

```bash
export VLLM_HCU_USE_CUSTOM_OPS=0

vllm serve Qwen/Qwen3.8-27B \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --safetensors-load-strategy prefetch \
  --attention-backend TRITON_ATTN \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --enable-mamba-fine-grained-prefix-cache \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
```

### Qwen3.8-27B IFB BW1100 1x vLLM 0.21

```bash
vllm serve Qwen/Qwen3.8-27B \
  -tp 1 \
  --trust-remote-code \
  --attention-backend FLASH_ATTN_CUSTOM \
  --max-num-batched-tokens 10240 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 \
  --compilation-config '{"cudagraph_mode":"FULL","max_cudagraph_capture_size":2048}' \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3
```

### Qwen3.8-27B IFB BW1000 2x vLLM 0.21

```bash
vllm serve Qwen/Qwen3.8-27B \
  -tp 2 \
  --trust-remote-code \
  --attention-backend FLASH_ATTN_CUSTOM \
  --max-num-batched-tokens 10240 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 \
  --compilation-config '{"cudagraph_mode":"FULL","max_cudagraph_capture_size":2048}' \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3
```

### Qwen3.8-27B IFB K100_AI 2x vLLM 0.21

```bash
vllm serve Qwen/Qwen3.8-27B \
  -tp 2 \
  --trust-remote-code \
  --attention-backend FLASH_ATTN_CUSTOM \
  --max-num-batched-tokens 10240 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 \
  --compilation-config '{"cudagraph_mode":"FULL","max_cudagraph_capture_size":2048}' \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3
```

### Qwen3.8-27B IFB BW1000 2x vLLM 0.18-hotfix

```bash
vllm serve Qwen/Qwen3.8-27B \
  -tp 2 \
  --served-model-name "$SERVED_MODEL_NAME" \
  --trust-remote-code \
  --attention-backend FLASH_ATTN_CUSTOM \
  --no-enable-prefix-caching \
  --max-num-batched-tokens 10240 \
  --compilation-config '{"cudagraph_mode":"FULL","max_cudagraph_capture_size":2048}' \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3
```

### Qwen3.8-27B-Channel-FP8 IFB BW1100 1x vLLM 0.28.1

```bash
export VLLM_KV_CACHE_LAYOUT=HND

vllm serve hygon/Qwen3.8-27B-Channel-FP8 \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --attention-backend FLASH_ATTN \
  --kv-cache-dtype fp8_e4m3 \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --prefix-match-unit 64 \
  --enable-mamba-fine-grained-prefix-cache \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
```

### Qwen3.8-27B-Channel-INT8-w8a8 IFB BW1100 1x vLLM 0.28.1

```bash
export VLLM_KV_CACHE_LAYOUT=HND

vllm serve hygon/Qwen3.8-27B-Channel-INT8-w8a8 \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --attention-backend FLASH_ATTN \
  --kv-cache-dtype fp8_e4m3 \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --prefix-match-unit 64 \
  --enable-mamba-fine-grained-prefix-cache \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
```

### Qwen3.8-27B-Channel-INT8-w8a8 IFB BW1000 1x vLLM 0.28.1

```bash
export VLLM_KV_CACHE_LAYOUT=HND

vllm serve hygon/Qwen3.8-27B-Channel-INT8-w8a8 \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --attention-backend FLASH_ATTN \
  --kv-cache-dtype fp8_e5m2 \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --prefix-match-unit 64 \
  --enable-mamba-fine-grained-prefix-cache \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
```

### Qwen3.8-27B-Channel-INT8-w8a8 IFB K100_AI 2x vLLM 0.28.1

```bash
export VLLM_HCU_USE_CUSTOM_OPS=0

vllm serve hygon/Qwen3.8-27B-Channel-INT8-w8a8 \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --attention-backend TRITON_ATTN \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --enable-mamba-fine-grained-prefix-cache \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
```

### Qwen3.8-27B-Channel-INT8-w8a8 IFB BW1100 1x vLLM 0.21

```bash
vllm serve hygon/Qwen3.8-27B-Channel-INT8-w8a8 \
  -tp 1 \
  --trust-remote-code \
  --attention-backend FLASH_ATTN_CUSTOM \
  --max-num-batched-tokens 10240 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 \
  -q slimquant_marlin \
  --compilation-config '{"cudagraph_mode":"FULL","max_cudagraph_capture_size":2048}' \
  --speculative-config.quantization "slimquant_marlin" \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3
```

### Qwen3.8-27B-Channel-INT8-w8a8 IFB BW1000 1x vLLM 0.21

```bash
vllm serve hygon/Qwen3.8-27B-Channel-INT8-w8a8 \
  -tp 1 \
  --trust-remote-code \
  --attention-backend FLASH_ATTN_CUSTOM \
  --max-num-batched-tokens 10240 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 \
  -q slimquant_marlin \
  --compilation-config '{"cudagraph_mode":"FULL","max_cudagraph_capture_size":2048}' \
  --speculative-config.quantization "slimquant_marlin" \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3
```

#### 在 DTK 26.04 环境中通过 wheel 启动

如果已经有 DTK 26.04 和带 `venv` 的 Python 3.10，可以在独立的 Python 环境中启动。下面用的是超算互联网 Notebook 上的一张 64 GB BW（`gfx936`），系统为 Ubuntu 22.04 x86_64，Python 版本为 3.10.20。验证配置为纯文本、eager 模式、2048 token 上下文。请从同一个可写目录执行下面的命令。

```bash
unset PYTHONPATH PYTHONHOME
python3.10 -m venv .venv-vllm021
source .venv-vllm021/bin/activate

python -m pip install --index-url https://pypi.sourcefind.cn/das_release/dtk/+simple/ \
  --extra-index-url https://pypi.org/simple \
  'torch==2.10.0+das.opt1.dtk2604.2608131652.g28498c' \
  'vllm==0.21.0+das.dtk2604.torch2100.2606111143.g8c979d' \
  'vllm-hcu==0.21.0+das.dtk2604.torch2100.2608132222.g386d02' \
  'triton==3.6.0+das185.dtk2604.torch2100.2608131719.gb796b1' \
  'torchvision==0.25.0+das185.dtk2604.torch2100.2608130858.gdcd044' \
  'flash-attn==2.8.3+das185.dtk2604.torch2100.2608132155.g9febea' \
  'aiter==0.1.5+das185.dtk2604.torch2100.2608131714.gb0979d' \
  'causal-conv1d==1.5.4+das185.dtk2604.torch2100.2608131033.g44ef50' \
  'lightop==0.6.0+das.dtk2604.torch2100.2608121706.g09a4f3' \
  'lmslim==0.3.1+das.opt4.dtk2604.torch2100.2607241157.gf9a687' \
  'tilelang==0.1.9+das185.dtk2604.torch2100.2608121523.gf631c9'
```

这版 Torch wheel 在 NumPy 2.1.3 下导入时会报 `_ARRAY_API not found`，随后 `torch.from_numpy()` 失败。把 NumPy 1.26.4 和支持它的 `ml-dtypes` 0.5.4 放到单独目录，启动时通过 `PYTHONPATH` 加载。旧版 HCU 插件还依赖海光的 DeepGEMM。安装好后运行 `vllm-hcu-apply-patches`，补上 `FLASH_ATTN_CUSTOM` 所需的配置。

```bash
python -m pip install --no-deps --only-binary=:all: --target .vllm021-compat \
  --index-url https://pypi.org/simple numpy==1.26.4 ml-dtypes==0.5.4
python -m pip install --no-deps --only-binary=:all: --target .vllm021-compat \
  --index-url https://pypi.sourcefind.cn/das_release/dtk/+simple/ \
  'deepgemm==2.1.0+das185.dtk2604.torch2100.2608121826.g2b4d4e'
export PYTHONPATH="$PWD/.vllm021-compat" PYTHONNOUSERSITE=1
vllm-hcu-apply-patches
grep -q hcu_flash_attn_mode .venv-vllm021/lib/python3.10/site-packages/vllm/config/attention.py

python - <<'PY'
import ml_dtypes
import numpy as np
import torch
from deepgemm.m_group_gemm import pack_int8_weight_enk_to_w6_low_latency

x = np.array([1.0], dtype=np.float32)
assert np.array_equal(torch.from_numpy(x).to("cuda").cpu().numpy(), x)
assert callable(pack_int8_weight_enk_to_w6_low_latency)
print(torch.cuda.get_device_name(0), np.__version__, ml_dtypes.__version__)
PY
```

该版 Triton 还需要配套的 AI LLVM。基础镜像缺少它时，权重能加载，但 INT8 算子预热会失败。这里使用能编译该算子的 DTK 26.04 版本，将归档解包到当前目录。 [FlagOS 的版本说明](https://flagos-ai.github.io/release-info/base/hygon-dtk26.04/#230)

```bash
mkdir -p .vllm021-toolchain
(
  set -euo pipefail
  cd .vllm021-toolchain
  curl -fL --retry 2 \
    'https://resource.flagos.net/repository/flagos-filestore/hygon/dtk_llvm-1.0.0+dtk2604.2609221416.g5f9d5e-x86_64.run' \
    -o dtk_llvm.run
  printf '%s  %s\n' 296d8e21b9dce091f8514f701054df762cf55a85acbfb1b595ecd5d7422aa285 dtk_llvm.run | sha256sum -c -
  tail -n +"$(awk '/END_OF_COMPRESS_PACK_SCRIPT$/{print NR + 1; exit}' dtk_llvm.run)" dtk_llvm.run |
    tar -xzf - --no-same-owner --no-same-permissions
  test -x dtk_llvm/aillvm/bin/clang-18
) &&
export TRITON_HIP_CLANG_PATH="$PWD/.vllm021-toolchain/dtk_llvm/aillvm/bin/clang-18" &&
export LD_LIBRARY_PATH="$PWD/.vllm021-toolchain/dtk_llvm/aillvm/lib:/opt/dtk/dcc/gcvm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

使用 [ModelScope 上的海光 W8A8 权重](https://modelscope.cn/models/hygon/Qwen3.8-27B-Channel-INT8-w8a8)。下载到本地后，由 vLLM 读取权重自带的量化配置。Triton 可能先提示默认的 `/opt/dtk/aillvm` 不存在，随后使用明确指定的编译器。

```bash
env -u PYTHONPATH python -m pip install --index-url https://pypi.org/simple modelscope
modelscope download --model hygon/Qwen3.8-27B-Channel-INT8-w8a8 --local_dir ./qwen38-27b-w8a8

vllm serve ./qwen38-27b-w8a8 \
  --served-model-name hygon/Qwen3.8-27B-Channel-INT8-w8a8 \
  --tensor-parallel-size 1 --trust-remote-code \
  --attention-backend FLASH_ATTN_CUSTOM --language-model-only \
  --max-model-len 2048 --max-num-seqs 4 --enforce-eager --host 127.0.0.1
```

等服务启动后，从另一个 Shell 发一条请求。实测返回 `4`，服务日志显示 W8A8 线性层选择了 `TritonInt8ScaledMMLinearKernel`。

```bash
curl -fsS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "hygon/Qwen3.8-27B-Channel-INT8-w8a8",
    "messages": [{"role": "user", "content": "What is 2+2? Reply with just the number."}],
    "temperature": 0,
    "max_tokens": 24,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

### Qwen3.8-27B-Channel-INT8-w8a8 IFB BW1000 2x vLLM 0.21

```bash
vllm serve hygon/Qwen3.8-27B-Channel-INT8-w8a8 \
  -tp 2 \
  --trust-remote-code \
  --attention-backend FLASH_ATTN_CUSTOM \
  --max-num-batched-tokens 10240 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 \
  -q slimquant_marlin \
  --compilation-config '{"cudagraph_mode":"FULL","max_cudagraph_capture_size":2048}' \
  --speculative-config.quantization "slimquant_marlin" \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3
```

### Qwen3.8-27B-Channel-INT8-w8a8 IFB K100_AI 2x vLLM 0.21

```bash
export VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM=0
export VLLM_HCU_USE_CUSTOM_OPS=0
export VLLM_ROCM_USE_AITER=0
export VLLM_ROCM_USE_AITER_MOE=0

vllm serve hygon/Qwen3.8-27B-Channel-INT8-w8a8 \
  -tp 2 \
  --trust-remote-code \
  --attention-backend FLASH_ATTN_CUSTOM \
  --max-num-batched-tokens 10240 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 \
  -q slimquant_marlin \
  --compilation-config '{"cudagraph_mode":"FULL","max_cudagraph_capture_size":2048}' \
  --speculative-config.quantization "slimquant_marlin" \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3
```

### Qwen3.8-27B-Channel-INT8-w8a8 IFB BW1000 1x vLLM 0.18-hotfix

```bash
vllm serve hygon/Qwen3.8-27B-Channel-INT8-w8a8 \
  -tp 1 \
  --trust-remote-code \
  --attention-backend FLASH_ATTN_CUSTOM \
  --max-num-batched-tokens 10240 \
  --compilation-config '{"cudagraph_mode":"FULL","max_cudagraph_capture_size":2048}' \
  --speculative-config.method mtp \
  --speculative-config.quantization "slimquant_marlin" \
  -q slimquant_marlin \
  --speculative-config.num_speculative_tokens 3
```

### Qwen3.8-27B-Channel-INT8-w8a8 IFB BW1000 2x vLLM 0.18-hotfix

```bash
vllm serve hygon/Qwen3.8-27B-Channel-INT8-w8a8 \
  -tp 2 \
  --trust-remote-code \
  --attention-backend FLASH_ATTN_CUSTOM \
  --max-num-batched-tokens 10240 \
  --compilation-config '{"cudagraph_mode":"FULL","max_cudagraph_capture_size":2048}' \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 \
  --speculative-config.quantization "slimquant_marlin" \
  -q slimquant_marlin
```

## API 调用

### IFB

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="Qwen/Qwen3.8-27B",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "中国的首都是哪里？"},
    ],
    max_tokens=2048,
)
print(response.choices[0].message.content)
```

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.8-27B",
    "max_tokens": 2048,
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": [
        {"type": "text", "text": "中国的首都是哪里？"}
      ]}
    ]
  }'
```
