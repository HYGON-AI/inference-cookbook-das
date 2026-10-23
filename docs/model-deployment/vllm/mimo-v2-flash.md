# MiMo-V2-Flash on vLLM

## 模型简介

本文提供 MiMo-V2-Flash 系列模型在 BW1100、BW1000、K100_AI 上的 vLLM 0.28.1 部署命令，具体权重规格和支持平台见模型列表。

## 模型列表

| 模型权重 | 量化方式 | vLLM 镜像 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | --------- | -------- | ---- | -------- | -------- |
| [hygon/MiMo-V2-Flash-Channel-INT8-w8a8](https://www.modelscope.cn/models/hygon/MiMo-V2-Flash-Channel-INT8-w8a8) | INT8 W8A8 | 0.28.1 | BW1100 | 8 | IFB | [**`>_`**](#mimo-v2-flash-channel-int8-w8a8-ifb-bw1100-8x-vllm-0281) |
|  | INT8 W8A8 | 0.28.1 | BW1000 | 8 | IFB | [**`>_`**](#mimo-v2-flash-channel-int8-w8a8-ifb-bw1000-8x-vllm-0281) |
|  | INT8 W8A8 | 0.28.1 | K100_AI | 8 | IFB | [**`>_`**](#mimo-v2-flash-channel-int8-w8a8-ifb-k100_ai-8x-vllm-0281) |
| [hygon/MiMo-V2-Flash-Channel-FP8-w8a8](https://www.modelscope.cn/models/hygon/MiMo-V2-Flash-Channel-FP8-w8a8) | FP8 W8A8 | 0.28.1 | BW1100 | 4 | IFB | [**`>_`**](#mimo-v2-flash-channel-fp8-w8a8-ifb-bw1100-4x-vllm-0281) |

第二批平台支持：BW1000 不支持以下 FP8 权重模型，仅提供 BW1100 命令：`hygon/MiMo-V2-Flash-Channel-FP8-w8a8`。

## 启动命令

### MiMo-V2-Flash-Channel-INT8-w8a8 IFB BW1100 8x vLLM 0.28.1

```bash
export VLLM_KV_CACHE_LAYOUT=HND

vllm serve hygon/MiMo-V2-Flash-Channel-INT8-w8a8 \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --dtype bfloat16 \
  --safetensors-load-strategy prefetch \
  --attention-backend FLASH_ATTN \
  --moe-backend aiter \
  --kv-cache-dtype fp8_e4m3 \
  --enable-prefix-caching \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 \
  --tool-call-parser mimo \
  --reasoning-parser qwen3 \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
```

### MiMo-V2-Flash-Channel-INT8-w8a8 IFB BW1000 8x vLLM 0.28.1

```bash
export VLLM_KV_CACHE_LAYOUT=HND

vllm serve hygon/MiMo-V2-Flash-Channel-INT8-w8a8 \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --dtype bfloat16 \
  --safetensors-load-strategy prefetch \
  --attention-backend FLASH_ATTN \
  --moe-backend aiter \
  --kv-cache-dtype fp8_e5m2 \
  --enable-prefix-caching \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 \
  --tool-call-parser mimo \
  --reasoning-parser qwen3 \
  --generation-config vllm \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
```

### MiMo-V2-Flash-Channel-INT8-w8a8 IFB K100_AI 8x vLLM 0.28.1

```bash
export VLLM_HCU_USE_CUSTOM_OPS=0

vllm serve hygon/MiMo-V2-Flash-Channel-INT8-w8a8 \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --dtype bfloat16 \
  --safetensors-load-strategy prefetch \
  --attention-backend TRITON_ATTN \
  --moe-backend triton \
  --enable-prefix-caching \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 \
  --tool-call-parser mimo \
  --reasoning-parser qwen3 \
  --generation-config vllm \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
```

### MiMo-V2-Flash-Channel-FP8-w8a8 IFB BW1100 4x vLLM 0.28.1

```bash

vllm serve hygon/MiMo-V2-Flash-Channel-FP8-w8a8 \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --moe-backend aiter \
  --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --tool-call-parser mimo \
  --reasoning-parser qwen3 \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
```

## API 调用

### IFB

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "hygon/MiMo-V2-Flash-Channel-INT8-w8a8", "messages": [{"role": "user", "content": "中国的首都是哪里？"}], "max_tokens": 128}'
```
