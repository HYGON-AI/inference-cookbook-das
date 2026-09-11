# Qwen3.8 on SGLang

## 模型简介

Qwen3.8 系列模型面向长上下文推理与工具调用场景，支持 SGLang 部署。

## 模型列表

| 模型权重 | 量化方式 | SGLang 镜像 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | ----------- | -------- | ---- | -------- | -------- |
| [Qwen/Qwen3.8-27B](https://www.modelscope.cn/models/Qwen/Qwen3.8-27B) | BF16 | 0.5.12 | BW1100 | 1 | IFB | [**`>_`**](#qwen38-27b-ifb-bw1100-1x-sglang-0512) |
|  | BF16 | [0.5.12](../docker_images.md) | BW1100 | 2 | IFB | [**`>_`**](#qwen38-27b-ifb-bw1100-2x-sglang-0512) |
|  | BF16 | [0.5.12](../docker_images.md) | BW1000 | 2 | IFB | [**`>_`**](#qwen38-27b-ifb-bw1000-2x-sglang-0512) |
|  | BF16 | [0.5.12](../docker_images.md) | K100_AI | 2 | IFB | [**`>_`**](#qwen38-27b-ifb-k100ai-2x-sglang-0512) |
| [hygon/Qwen3.8-27B-Channel-INT8-w8a8](https://www.modelscope.cn/models/hygon/Qwen3.8-27B-Channel-INT8-w8a8) | INT8 W8A8 | [0.5.12](../docker_images.md) | BW1100 | 2 | IFB | [**`>_`**](#qwen38-27b-channel-int8-w8a8-ifb-bw1100-2x-sglang-0512) |
|  | W8A8 | 0.5.12 | BW1000 | 1 | IFB | [**`>_`**](#qwen38-27b-channel-int8-w8a8-ifb-bw1000-1x-sglang-0512) |
|  | W8A8 | 0.5.12 | BW1000 | 2 | IFB | [**`>_`**](#qwen38-27b-channel-int8-w8a8-ifb-bw1000-2x-sglang-0512-262k) |
|  | INT8 W8A8 | [0.5.12](../docker_images.md) | BW1000 | 2 | IFB | [**`>_`**](#qwen38-27b-channel-int8-w8a8-ifb-bw1000-2x-sglang-0512) |
|  | INT8 W8A8 | [0.5.12](../docker_images.md) | K100_AI | 2 | IFB | [**`>_`**](#qwen38-27b-channel-int8-w8a8-ifb-k100ai-2x-sglang-0512) |
| [hygon/Qwen3.8-Flash-Next-Channel-INT8-w8a8](https://www.modelscope.cn/models/hygon/Qwen3.8-Flash-Next-Channel-INT8-w8a8) | INT8 W8A8 | 0.5.18 | BW1000 | 8 | IFB | [**`>_`**](#qwen38-flash-next-channel-int8-w8a8-ifb-bw1000-8x-sglang-0518) |
|  | INT8 W8A8 | 0.5.18 | BW1100 | 4 | IFB | [**`>_`**](#qwen38-flash-next-channel-int8-w8a8-ifb-bw1100-4x-sglang-0518) |
| [hygon/Qwen3.8-Flash-Next-Channel-FP8](https://modelscope.cn/models/hygon/Qwen3.8-Flash-Next-Channel-FP8) | FP8 | 0.5.18 | BW1100 | 4 | IFB | [**`>_`**](#qwen38-flash-next-channel-fp8-ifb-bw1100-4x-sglang-0518) |

## 启动命令

### Qwen3.8-27B IFB BW1100 1x SGLang 0.5.12

```bash
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_FUSED_TOPK_SOFTMAX=1
export SGLANG_USE_CAUSAL_CONV1D=1
export SGLANG_USE_AITER_LINEAR_ATTN=1
export SGLANG_ENABLE_SPEC_V2=1

sglang serve --model-path Qwen/Qwen3.8-27B \
    --mm-attention-backend fa3 \
    --speculative-algorithm NEXTN \
    --enable-piecewise-cuda-graph \
    --max-running-requests 96 \
    --chunked-prefill-size -1 \
    --cuda-graph-max-bs 256 \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --tp-size 1 --pp-size 1 \
    --attention-backend fa3 \
    --page-size 64 \
    --mem-fraction-static 0.9 \
    --trust-remote-code \
    --context-length 262144 \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --mamba-scheduler-strategy extra_buffer \
    --kv-cache-dtype fp8_e4m3
```

### Qwen3.8-27B IFB BW1100 2x SGLang 0.5.12

```bash
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_USE_FUSED_TOPK_SOFTMAX=1
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_CAUSAL_CONV1D=1
export SGLANG_USE_AITER_LINEAR_ATTN=1

sglang serve --model-path Qwen/Qwen3.8-27B \
  --attention-backend fa3 \
  --mm-attention-backend fa3 \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --tp-size 2 --pp-size 1 \
  --page-size 64 \
  --mamba-scheduler-strategy extra_buffer \
  --kv-cache-dtype fp8_e4m3 \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --trust-remote-code
```

### Qwen3.8-27B IFB BW1000 2x SGLang 0.5.12

```bash
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_USE_FUSED_TOPK_SOFTMAX=1
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_CAUSAL_CONV1D=1
export SGLANG_USE_AITER_LINEAR_ATTN=1

sglang serve --model-path Qwen/Qwen3.8-27B \
  --attention-backend fa3 \
  --mm-attention-backend fa3 \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --tp-size 2 --pp-size 1 \
  --page-size 64 \
  --mamba-scheduler-strategy extra_buffer \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --trust-remote-code
```

### Qwen3.8-27B IFB K100_AI 2x SGLang 0.5.12

```bash
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_FUSED_TOPK_SOFTMAX=1
export SGLANG_ROCM_USE_AITER_MOE=False
export SGLANG_USE_CUDA_IPC_TRANSPORT=1
export SGLANG_USE_AITER_LINEAR_ATTN=1

sglang serve --model-path Qwen/Qwen3.8-27B \
  --tp-size 2 \
  --attention-backend fa3 \
  --page-size 64 \
  --pp-size 1 \
  --mem-fraction-static 0.8 \
  --max-prefill-tokens 45000 \
  --mamba-scheduler-strategy extra_buffer \
  --disable-custom-all-reduce \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3
```

### Qwen3.8-27B-Channel-INT8-w8a8 IFB BW1100 2x SGLang 0.5.12

```bash
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_USE_FUSED_TOPK_SOFTMAX=1
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_CAUSAL_CONV1D=1
export SGLANG_USE_AITER_LINEAR_ATTN=1

sglang serve --model-path hygon/Qwen3.8-27B-Channel-INT8-w8a8 \
  --attention-backend fa3 \
  --mm-attention-backend fa3 \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --tp-size 2 --pp-size 1 \
  --page-size 64 \
  --mamba-scheduler-strategy extra_buffer \
  --kv-cache-dtype fp8_e4m3 \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --trust-remote-code
```

### Qwen3.8-27B-Channel-INT8-w8a8 IFB BW1000 1x SGLang 0.5.12

```bash
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_FUSED_TOPK_SOFTMAX=1
export SGLANG_USE_CAUSAL_CONV1D=1
export SGLANG_USE_AITER_LINEAR_ATTN=1
export SGLANG_ENABLE_SPEC_V2=1

sglang serve --model-path hygon/Qwen3.8-27B-Channel-INT8-w8a8 \
    --mm-attention-backend fa3 \
    --enable-piecewise-cuda-graph \
    --chunked-prefill-size -1 \
    --tp-size 1 --pp-size 1 \
    --attention-backend fa3 \
    --page-size 64 \
    --mem-fraction-static 0.9 \
    --cuda-graph-max-bs 256 \
    --trust-remote-code \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --mamba-scheduler-strategy extra_buffer \
    --kv-cache-dtype fp8_e5m2 \
    --max-running-requests 96 \
    --context-length 262144
```

### Qwen3.8-27B-Channel-INT8-w8a8 IFB BW1000 2x SGLang 0.5.12 262K

```bash
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_FUSED_TOPK_SOFTMAX=1
export SGLANG_USE_CAUSAL_CONV1D=1
export SGLANG_USE_AITER_LINEAR_ATTN=1
export SGLANG_ENABLE_SPEC_V2=1

sglang serve --model-path hygon/Qwen3.8-27B-Channel-INT8-w8a8 \
    --mm-attention-backend fa3 \
    --speculative-algorithm NEXTN \
    --enable-piecewise-cuda-graph \
    --max-running-requests 96 \
    --chunked-prefill-size -1 \
    --cuda-graph-max-bs 256 \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --tp-size 2 --pp-size 1 \
    --attention-backend fa3 \
    --page-size 64 \
    --mem-fraction-static 0.9 \
    --trust-remote-code \
    --context-length 262144 \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --mamba-scheduler-strategy extra_buffer \
    --kv-cache-dtype fp8_e5m2
```

### Qwen3.8-27B-Channel-INT8-w8a8 IFB BW1000 2x SGLang 0.5.12

```bash
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_USE_FUSED_TOPK_SOFTMAX=1
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_CAUSAL_CONV1D=1
export SGLANG_USE_AITER_LINEAR_ATTN=1

sglang serve --model-path hygon/Qwen3.8-27B-Channel-INT8-w8a8 \
  --attention-backend fa3 \
  --mm-attention-backend fa3 \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --tp-size 2 --pp-size 1 \
  --page-size 64 \
  --mamba-scheduler-strategy extra_buffer \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --trust-remote-code
```

### Qwen3.8-27B-Channel-INT8-w8a8 IFB K100_AI 2x SGLang 0.5.12

```bash
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_FUSED_TOPK_SOFTMAX=1
export SGLANG_ROCM_USE_AITER_MOE=False
export SGLANG_USE_CUDA_IPC_TRANSPORT=1
export SGLANG_USE_AITER_LINEAR_ATTN=1

sglang serve --model-path hygon/Qwen3.8-27B-Channel-INT8-w8a8 \
  --tp-size 2 \
  --attention-backend fa3 \
  --page-size 64 \
  --pp-size 1 \
  --mem-fraction-static 0.8 \
  --mamba-scheduler-strategy extra_buffer \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --disable-custom-all-reduce
```

### Qwen3.8-Flash-Next-Channel-INT8-w8a8 IFB BW1000 8x SGLang 0.5.18

```bash
export SGLANG_USE_MODELSCOPE=1
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_FUSED_TOPK_SOFTMAX=1
export SGLANG_USE_CAUSAL_CONV1D=1
export SGLANG_USE_AITER_LINEAR_ATTN=1
export SGLANG_USE_AITER_AR=1
export W8A8_SUPPORT_METHODS=3

sglang serve \
  --model-path hygon/Qwen3.8-Flash-Next-Channel-INT8-w8a8 \
  --trust-remote-code \
  --tp-size 8 \
  --pp-size 1 \
  --dp-size 1 \
  --ep-size 1 \
  --moe-runner-backend aiter \
  --speculative-moe-runner-backend aiter \
  --dtype bfloat16 \
  --context-length 32768 \
  --max-running-requests 32 \
  --language-model-only \
  --mem-fraction-static 0.85 \
  --chunked-prefill-size 4096 \
  --page-size 64 \
  --mamba-radix-cache-strategy extra_buffer \
  --kv-cache-dtype fp8_e5m2 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --watchdog-timeout 1200 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --numa-node 0 0 0 0 0 0 0 0
```

### Qwen3.8-Flash-Next-Channel-INT8-w8a8 IFB BW1100 4x SGLang 0.5.18

```bash
export SGLANG_USE_MODELSCOPE=1
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_FUSED_TOPK_SOFTMAX=1
export SGLANG_USE_CAUSAL_CONV1D=1
export SGLANG_USE_AITER_LINEAR_ATTN=1
export SGLANG_USE_AITER_AR=1
export W8A8_SUPPORT_METHODS=3

sglang serve \
  --model-path hygon/Qwen3.8-Flash-Next-Channel-INT8-w8a8 \
  --trust-remote-code \
  --tp-size 4 \
  --pp-size 1 \
  --dp-size 1 \
  --ep-size 1 \
  --moe-runner-backend aiter \
  --speculative-moe-runner-backend aiter \
  --dtype bfloat16 \
  --context-length 32768 \
  --max-running-requests 32 \
  --language-model-only \
  --mem-fraction-static 0.85 \
  --chunked-prefill-size 4096 \
  --page-size 64 \
  --mamba-radix-cache-strategy extra_buffer \
  --kv-cache-dtype fp8_e4m3 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --watchdog-timeout 1200 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --numa-node 0 0 0 0
```

### Qwen3.8-Flash-Next-Channel-FP8 IFB BW1100 4x SGLang 0.5.18

```bash
export SGLANG_USE_MODELSCOPE=1
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_FUSED_TOPK_SOFTMAX=1
export SGLANG_USE_CAUSAL_CONV1D=1
export SGLANG_USE_AITER_LINEAR_ATTN=1
export SGLANG_USE_AITER_AR=1

sglang serve \
  --model-path hygon/Qwen3.8-Flash-Next-Channel-FP8 \
  --trust-remote-code \
  --tp-size 4 \
  --pp-size 1 \
  --dp-size 1 \
  --ep-size 1 \
  --moe-runner-backend aiter \
  --speculative-moe-runner-backend aiter \
  --dtype bfloat16 \
  --language-model-only \
  --mem-fraction-static 0.85 \
  --chunked-prefill-size 4096 \
  --context-length 32768 \
  --page-size 64 \
  --mamba-radix-cache-strategy extra_buffer \
  --kv-cache-dtype fp8_e4m3 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --watchdog-timeout 1200 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --numa-node 0 0 0 0
```

## API 调用

### IFB

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="not-needed")

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
curl http://localhost:30000/v1/chat/completions \
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
