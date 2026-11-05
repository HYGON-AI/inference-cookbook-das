# DeepSeek-V3.1 on vLLM

## 模型简介

DeepSeek-V3.1 是深度求索推出的混合推理 MoE 大模型（总参数约 671B，激活约 37B），支持 thinking / non-thinking 切换，并强化工具调用与 Agent 能力。
**DeepSeek-V3.1-Terminus** 是官方于 2025-09 发布的 V3.1 更新版：在保持原有能力的基础上，改善中英混杂与异常字符问题，并进一步优化 Code Agent / Search Agent 表现；开源权重为 FP8（UE8M0）格式。

## 模型列表

| 模型权重                                                                                                 | 量化方式 | vLLM 镜像                  | 推荐硬件 | 卡数 | 部署方式 | 启动命令                                                          |
| -------------------------------------------------------------------------------------------------------- | -------- | -------------------------- | -------- | ---- | -------- | ----------------------------------------------------------------- |
| [deepseek-ai/DeepSeek-V3.1-Terminus](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-V3.1-Terminus) | FP8 W8A8 | 0.28.1                     | BW1100   | 8    | IFB      | [**`>_`**](#deepseek-v31-terminus-ifb-bw1100-8x-vllm-0281) |
|                                                                                                          | FP8 W8A8 | [0.21](../docker_images.md) | BW1100   | 8    | IFB      | [**`>_`**](#deepseek-v31-terminus-ifb-bw1100-8x-vllm-021)  |

## 启动命令

### DeepSeek-V3.1-Terminus IFB BW1100 8x vLLM 0.28.1

```bash
vllm serve deepseek-ai/DeepSeek-V3.1-Terminus \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --dtype bfloat16 \
  --quantization slimquant_marlin \
  --attention-backend FLASHMLA \
  --moe-backend aiter \
  --kv-cache-dtype fp8_e4m3 \
  --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"quantization":"slimquant_marlin"}' \
  --enable-auto-tool-choice \
  --tool-call-parser deepseek_v31 \
  --reasoning-parser deepseek_v3 \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
```

### DeepSeek-V3.1-Terminus IFB BW1100 8x vLLM 0.21

```bash
vllm serve deepseek-ai/DeepSeek-V3.1-Terminus \
  --trust-remote-code \
  -q slimquant_marlin \
  -tp 8 \
  --dtype bfloat16 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.90 \
  --max-num-batched-tokens 16384 \
  --compilation-config '{"pass_config":{"fuse_act_quant":false}}' \
  --speculative_config '{"method":"deepseek_mtp","num_speculative_tokens":3,"quantization":"slimquant_marlin"}' \
  --no-enable-prefix-caching \
  --kv-cache-dtype fp8_e4m3 \
  --attention-backend FLASHMLA
```

## API 调用

### IFB

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-V3.1-Terminus",
    messages=[
        {"role": "system", "content": "你是一个专业的编程助手。"},
        {"role": "user", "content": "用 Python 实现一个高效的 LRU Cache"},
    ],
    max_tokens=2048,
    temperature=0.7,
)
print(response.choices[0].message.content)
```

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-V3.1-Terminus",
    "messages": [
      {"role": "system", "content": "你是一个专业的编程助手。"},
      {"role": "user", "content": "用 Python 实现一个高效的 LRU Cache"}
    ],
    "max_tokens": 2048
  }'
```
