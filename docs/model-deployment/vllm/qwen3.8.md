# Qwen3.8 on vLLM

## 模型简介

Qwen3.8 系列模型面向长上下文推理与工具调用场景，支持 vLLM 部署。

## 模型列表

| 模型权重 | 量化方式 | vLLM 镜像 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | --------- | -------- | ---- | -------- | -------- |
| [Qwen/Qwen3.8-Flash-Next](https://www.modelscope.cn/models/Qwen/Qwen3.8-Flash-Next) | BF16 | 0.28.1 | BW1100 | 8 | IFB | [**`>_`**](#qwen38-flash-next-ifb-bw1100-8x-vllm-0281) |
|  | BF16 | 0.28.1 | BW1000 | 8 | IFB | [**`>_`**](#qwen38-flash-next-ifb-bw1000-8x-vllm-0281) |
| [hygon/Qwen3.8-Flash-Next-Channel-FP8](https://www.modelscope.cn/models/hygon/Qwen3.8-Flash-Next-Channel-FP8) | FP8 | 0.28.1 | BW1100 | 2 | IFB | [**`>_`**](#qwen38-flash-next-channel-fp8-ifb-bw1100-2x-vllm-0281) |
| [hygon/Qwen3.8-Flash-Next-Channel-INT8-w8a8](https://www.modelscope.cn/models/hygon/Qwen3.8-Flash-Next-Channel-INT8-w8a8) | W8A8 | 0.28.1 | BW1000 | 4 | IFB | [**`>_`**](#qwen38-flash-next-channel-int8-w8a8-ifb-bw1000-4x-vllm-0281) |
|  | W8A8 | 0.28.1 | K100_AI | 4 | IFB | [**`>_`**](#qwen38-flash-next-channel-int8-w8a8-ifb-k100_ai-4x-vllm-0281) |
| [hygon/Qwen3.8-Flash-Next-Mixed-w4a8](https://www.modelscope.cn/models/hygon/Qwen3.8-Flash-Next-Mixed-w4a8) | W4A8 | 0.28.1 | BW1100 | 2 | IFB | [**`>_`**](#qwen38-flash-next-mixed-w4a8-ifb-bw1100-2x-vllm-0281) |
|  | W4A8 | 0.28.1 | BW1000 | 8 | 1P1D | [**`>_`**](#qwen38-flash-next-mixed-w4a8-1p1d-bw1000-8x-vllm-0281) |
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
  -tp 4 \
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
  --max-num-seqs 64
```

### Qwen3.8-Flash-Next-Mixed-w4a8 IFB BW1100 2x vLLM 0.28.1

```bash
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
  --prefix-match-unit 64 \
  --enable-mamba-fine-grained-prefix-cache \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
```

### Qwen3.8-Flash-Next-Mixed-w4a8 1P1D BW1000 8x vLLM 0.28.1

单节点 8 卡部署：P 使用卡 `0,1,2,3`，D 使用卡 `4,5,6,7`，P、D 服务端口分别为 `8010`、`8011`，代理端口为 `8518`。启动顺序为 P node → D node → Mooncake Connector Proxy。

上下文长度不超过 128K 与超过 128K 时采用不同的并行方式，请按实际上下文长度选择对应小节。

#### 上下文 ≤ 128K

##### P node

```bash
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export VLLM_MOONCAKE_BOOTSTRAP_PORT=18998
export VLLM_ENGINE_READY_TIMEOUT_S=1800
export VLLM_HCU_PLE_PREFETCH_STREAM=1
export VLLM_HCU_DP_INT8_ALLGATHER=1
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_KV_CACHE_LAYOUT=BLHNC
export HIP_VISIBLE_DEVICES=0,1,2,3
export VLLM_DEEPEP_BUFFER_SIZE_MB=2048

vllm serve hygon/Qwen3.8-Flash-Next-Mixed-w4a8 \
  -tp 1 \
  --enable-expert-parallel \
  --enable-ep-weight-filter \
  -dp 4 \
  --all2all-backend deepep_high_throughput \
  --moe-backend deep_gemm \
  --port 8010 \
  --host 0.0.0.0 \
  --served-model-name qwen38-next \
  --trust-remote-code \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --prefix-match-unit 64 \
  --enable-mamba-fine-grained-prefix-cache \
  --engram-config.cpu_offload=true \
  --engram-config.embedding_across_dp=true \
  --max-model-len 262144 \
  --kv-cache-dtype fp8_e5m2 \
  --max-num-seqs 64 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.85 \
  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_producer","kv_connector_extra_config":{"mooncake_protocol":"rdma"}}'
```

##### D node

```bash
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export VLLM_ENGINE_READY_TIMEOUT_S=1800
export VLLM_HCU_PLE_PREFETCH_STREAM=1
export VLLM_HCU_DP_INT8_ALLGATHER=1
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_KV_CACHE_LAYOUT=BLHNC
export HIP_VISIBLE_DEVICES=4,5,6,7

vllm serve hygon/Qwen3.8-Flash-Next-Mixed-w4a8 \
  -tp 1 \
  --enable-expert-parallel \
  --enable-ep-weight-filter \
  -dp 4 \
  --all2all-backend deepep_low_latency \
  --moe-backend deep_gemm \
  --port 8011 \
  --host 0.0.0.0 \
  --served-model-name qwen38-next \
  --trust-remote-code \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --prefix-match-unit 64 \
  --enable-mamba-fine-grained-prefix-cache \
  --engram-config.cpu_offload=true \
  --engram-config.embedding_across_dp=true \
  --max-model-len 262144 \
  --kv-cache-dtype fp8_e5m2 \
  --max-num-seqs 64 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 \
  --max-num-batched-tokens 256 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --gpu-memory-utilization 0.85 \
  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"mooncake_protocol":"rdma"}}'
```

#### 上下文 > 128K

P、D 均按 `tp2`×`dp2`（每实例 4 卡，EP=4）切分并行度。

##### P node

```bash
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export VLLM_MOONCAKE_BOOTSTRAP_PORT=18998
export VLLM_ENGINE_READY_TIMEOUT_S=1800
export VLLM_HCU_PLE_PREFETCH_STREAM=1
export VLLM_HCU_DP_INT8_ALLGATHER=1
export VLLM_HCU_QWEN4_EXP_HC_SP=1
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_KV_CACHE_LAYOUT=BLHNC
export HIP_VISIBLE_DEVICES=0,1,2,3
export VLLM_DEEPEP_BUFFER_SIZE_MB=2048

vllm serve hygon/Qwen3.8-Flash-Next-Mixed-w4a8 \
  -tp 2 \
  -dp 2 \
  --enable-expert-parallel \
  --enable-ep-weight-filter \
  --all2all-backend deepep_high_throughput \
  --moe-backend deep_gemm \
  --port 8010 \
  --host 0.0.0.0 \
  --served-model-name qwen38-next \
  --trust-remote-code \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --prefix-match-unit 64 \
  --enable-mamba-fine-grained-prefix-cache \
  --engram-config.cpu_offload=true \
  --engram-config.embedding_across_dp=true \
  --max-model-len 262144 \
  --kv-cache-dtype fp8_e5m2 \
  --max-num-seqs 32 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.85 \
  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_producer","kv_connector_extra_config":{"mooncake_protocol":"rdma"}}'
```

##### D node

```bash
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export VLLM_ENGINE_READY_TIMEOUT_S=1800
export VLLM_HCU_PLE_PREFETCH_STREAM=1
export VLLM_HCU_DP_INT8_ALLGATHER=1
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_KV_CACHE_LAYOUT=BLHNC
export VLLM_HCU_QWEN4_EXP_HC_SP=1
export HIP_VISIBLE_DEVICES=4,5,6,7

vllm serve hygon/Qwen3.8-Flash-Next-Mixed-w4a8 \
  -tp 2 \
  -dp 2 \
  --enable-expert-parallel \
  --enable-ep-weight-filter \
  --moe-backend deep_gemm \
  --all2all-backend deepep_low_latency \
  --port 8011 \
  --host 0.0.0.0 \
  --served-model-name qwen38-next \
  --trust-remote-code \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --prefix-match-unit 64 \
  --enable-mamba-fine-grained-prefix-cache \
  --engram-config.cpu_offload=true \
  --engram-config.embedding_across_dp=true \
  --max-model-len 262144 \
  --kv-cache-dtype fp8_e5m2 \
  --max-num-seqs 64 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 \
  --max-num-batched-tokens 256 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --gpu-memory-utilization 0.85 \
  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"mooncake_protocol":"rdma"}}'
```

#### Mooncake Connector Proxy

P、D 服务就绪后启动代理：

```bash
python3 "<vllm_source_dir>/examples/disaggregated/mooncake_connector/mooncake_connector_proxy.py" \
  --host 0.0.0.0 \
  --prefill "http://127.0.0.1:8010" 18998 \
  --decode "http://127.0.0.1:8011" \
  --port 8518
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
  --prefix-match-unit 64 \
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

### PD 分离

vLLM PD 分离模式下，客户端请求发送到 P 节点的代理端口（示例中为 `10.16.1.36:8518`）。

```python
from openai import OpenAI

client = OpenAI(base_url="http://10.16.1.36:8518/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="qwen38-next",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "中国的首都是哪里？"},
    ],
    max_tokens=2048,
)
print(response.choices[0].message.content)
```

```bash
curl http://10.16.1.36:8518/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen38-next",
    "max_tokens": 2048,
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "中国的首都是哪里？"}
    ]
  }'
```
