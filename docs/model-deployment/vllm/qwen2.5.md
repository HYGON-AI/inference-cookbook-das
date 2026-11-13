# Qwen2.5 on vLLM

## 模型简介

Qwen2.5 是阿里通义千问开源大语言模型系列，支持中英文、多语言、长文本与结构化输出。本页提供 Qwen2.5-14B-Instruct 在 HCU 上基于 vLLM 的推理部署方案。

## 模型列表

| 模型权重 | 量化方式 | vLLM 镜像 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | --------- | -------- | ---- | -------- | -------- |
| [Qwen/Qwen2.5-14B-Instruct](https://www.modelscope.cn/models/Qwen/Qwen2.5-14B-Instruct) | BF16 | 0.28.1 | BW1100 | 1 | IFB | [**`>_`**](#qwen25-14b-instruct-ifb-bw1100-1x-vllm-0281) |
|  | BF16 | 0.28.1 | BW1000 | 1 | IFB | [**`>_`**](#qwen25-14b-instruct-ifb-bw1000-1x-vllm-0281) |
|  | BF16 | 0.28.1 | K100_AI | 1 | IFB | [**`>_`**](#qwen25-14b-instruct-ifb-k100_ai-1x-vllm-0281) |

## 启动命令

### Qwen2.5-14B-Instruct IFB BW1100 1x vLLM 0.28.1

```bash
vllm serve Qwen/Qwen2.5-14B-Instruct \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --attention-backend FLASH_ATTN \
  --kv-cache-dtype bfloat16 \
  --enable-prefix-caching \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
```

### Qwen2.5-14B-Instruct IFB BW1000 1x vLLM 0.28.1

```bash
vllm serve Qwen/Qwen2.5-14B-Instruct \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --attention-backend FLASH_ATTN \
  --kv-cache-dtype bfloat16 \
  --enable-prefix-caching \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
```

### Qwen2.5-14B-Instruct IFB K100_AI 1x vLLM 0.28.1

```bash
export VLLM_HCU_USE_CUSTOM_OPS=0

vllm serve Qwen/Qwen2.5-14B-Instruct \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --attention-backend TRITON_ATTN \
  --kv-cache-dtype bfloat16 \
  --enable-prefix-caching \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
```

## API 调用

### IFB

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-14B-Instruct",
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
    "model": "Qwen/Qwen2.5-14B-Instruct",
    "max_tokens": 2048,
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "中国的首都是哪里？"}
    ]
  }'
```
