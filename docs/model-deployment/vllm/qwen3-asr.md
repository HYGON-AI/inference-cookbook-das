# Qwen3-ASR on vLLM

## 模型简介

[Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) 是 Qwen 团队开源的语音识别模型系列，支持音频转写与语言识别，提供 0.6B 和 1.7B 两种规模。本文整理原始权重及 Hugging Face 格式权重在 BW1100 上的 vLLM 部署命令。

需要将音频与已有文本对齐、获取词级时间戳时，请参阅 [Qwen3-ForcedAligner](qwen3-forcedaligner.md)。

---

## 模型列表

| 模型权重 | 量化方式 | vLLM 镜像 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | --------- | -------- | ---- | -------- | -------- |
| [Qwen/Qwen3-ASR-0.6B-hf](https://www.modelscope.cn/models/Qwen/Qwen3-ASR-0.6B-hf) | BF16 | 0.28.1 | BW1100 | 1 | IFB | [**`>_`**](#qwen3-asr-06b-hf-ifb-bw1100-1x-vllm-0281) |
|  | BF16 | [0.21](../docker_images.md) | BW1100 | 1x | IFB | [**`>_`**](#qwen3-asr-06b-hf-ifb-bw1100-1x-vllm-021) |
| [Qwen/Qwen3-ASR-1.7B-hf](https://www.modelscope.cn/models/Qwen/Qwen3-ASR-1.7B-hf) | BF16 | 0.28.1 | BW1100 | 1 | IFB | [**`>_`**](#qwen3-asr-17b-hf-ifb-bw1100-1x-vllm-0281) |
|  | BF16 | [0.21](../docker_images.md) | BW1100 | 1x | IFB | [**`>_`**](#qwen3-asr-17b-hf-ifb-bw1100-1x-vllm-021) |
| [Qwen/Qwen3-ASR-0.6B](https://www.modelscope.cn/models/Qwen/Qwen3-ASR-0.6B) | BF16 | 0.28.1 | BW1100 | 1 | IFB | [**`>_`**](#qwen3-asr-06b-ifb-bw1100-1x-vllm-0281) |
|  | BF16 | [0.21](../docker_images.md) | BW1100 | 1x | IFB | [**`>_`**](#qwen3-asr-06b-ifb-bw1100-1x-vllm-021) |
| [Qwen/Qwen3-ASR-1.7B](https://www.modelscope.cn/models/Qwen/Qwen3-ASR-1.7B) | BF16 | 0.28.1 | BW1100 | 1 | IFB | [**`>_`**](#qwen3-asr-17b-ifb-bw1100-1x-vllm-0281) |
|  | BF16 | [0.21](../docker_images.md) | BW1100 | 1x | IFB | [**`>_`**](#qwen3-asr-17b-ifb-bw1100-1x-vllm-021) |

## 启动命令

### Qwen3-ASR-0.6B-hf IFB BW1100 1x vLLM 0.28.1

```bash

vllm serve Qwen/Qwen3-ASR-0.6B-hf \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --attention-backend FLASH_ATTN \
  --no-enable-prefix-caching \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
```

### Qwen3-ASR-0.6B-hf IFB BW1100 1x vLLM 0.21

```bash
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE=1
export VLLM_HCU_USE_FLASH_ATTN_UNIFIED=1
export VLLM_HCU_USE_AITER_MOE_CONFIG=1

vllm serve Qwen/Qwen3-ASR-0.6B-hf \
  -tp 1 \
  --trust-remote-code \
  --no-enable-prefix-caching \
  --max-num-batched-tokens 8192
```

### Qwen3-ASR-1.7B-hf IFB BW1100 1x vLLM 0.28.1

```bash

vllm serve Qwen/Qwen3-ASR-1.7B-hf \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --attention-backend FLASH_ATTN \
  --no-enable-prefix-caching \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
```


### Qwen3-ASR-1.7B-hf IFB BW1100 1x vLLM 0.21

```bash
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE=1
export VLLM_HCU_USE_FLASH_ATTN_UNIFIED=1
export VLLM_HCU_USE_AITER_MOE_CONFIG=1

vllm serve Qwen/Qwen3-ASR-1.7B-hf \
  -tp 1 \
  --trust-remote-code \
  --no-enable-prefix-caching \
  --max-num-batched-tokens 8192
```

### Qwen3-ASR-0.6B IFB BW1100 1x vLLM 0.28.1

```bash

vllm serve Qwen/Qwen3-ASR-0.6B \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --attention-backend FLASH_ATTN \
  --no-enable-prefix-caching \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
```

### Qwen3-ASR-0.6B IFB BW1100 1x vLLM 0.21

```bash
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE=1
export VLLM_HCU_USE_FLASH_ATTN_UNIFIED=1
export VLLM_HCU_USE_AITER_MOE_CONFIG=1

vllm serve Qwen/Qwen3-ASR-0.6B \
  -tp 1 \
  --trust-remote-code \
  --no-enable-prefix-caching \
  --max-num-batched-tokens 8192
```

### Qwen3-ASR-1.7B IFB BW1100 1x vLLM 0.28.1

```bash

vllm serve Qwen/Qwen3-ASR-1.7B \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --attention-backend FLASH_ATTN \
  --no-enable-prefix-caching \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
```

### Qwen3-ASR-1.7B IFB BW1100 1x vLLM 0.21

```bash
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE=1
export VLLM_HCU_USE_FLASH_ATTN_UNIFIED=1
export VLLM_HCU_USE_AITER_MOE_CONFIG=1

vllm serve Qwen/Qwen3-ASR-1.7B \
  -tp 1 \
  --trust-remote-code \
  --no-enable-prefix-caching \
  --max-num-batched-tokens 8192
```

## API 调用

### 音频转写

以下以 `Qwen/Qwen3-ASR-0.6B-hf` 为例，使用其他模型时替换请求中的 `model`。

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="Qwen/Qwen3-ASR-0.6B-hf",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "audio_url",
                    "audio_url": {
                        "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav"
                    },
                }
            ],
        }
    ],
    temperature=0,
    max_tokens=1024,
)
print(response.choices[0].message.content)
```

```bash
curl --fail http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-ASR-0.6B-hf",
    "temperature": 0,
    "max_tokens": 1024,
    "messages": [
      {"role": "user", "content": [
        {"type": "audio_url", "audio_url": {"url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav"}}
      ]}
    ]
  }'
```

示例使用 Qwen 官方公开测试音频，服务端需要能够访问该 URL。Chat Completions 返回内容可能包含语言标记及 `<asr_text>`，转写文本位于该标记之后。

### 本地音频转写

也可通过转写接口提交本地音频文件。以下命令下载同一份公开测试音频，再提交至 `/v1/audio/transcriptions`：

```bash
curl --fail --location \
  https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav \
  --output asr_en.wav

curl --fail http://localhost:8000/v1/audio/transcriptions \
  -F 'model=Qwen/Qwen3-ASR-0.6B-hf' \
  -F 'file=@asr_en.wav' \
  -F 'response_format=json'
```
