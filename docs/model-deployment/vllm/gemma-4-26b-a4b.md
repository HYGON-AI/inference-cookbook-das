# Gemma-4-26B-A4B-it on vLLM

## 模型简介

Gemma-4-26B-A4B-it 是 Gemma4 系列指令模型，采用 MoE 架构，支持 sliding-window attention（SWA）和 full attention，本文档提供其在 vLLM 上的部署示例。

## 模型列表

| 模型权重 | 量化方式 | vLLM 镜像 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | --------- | -------- | ---- | -------- | -------- |
| gemma-4-26B-A4B-it | BF16 | 0.28.1 | BW | 1 | IFB | [**`>_`**](#gemma-4-26b-a4b-it-ifb-bw-1x-vllm-0281) |

### 0.28.1 平台支持说明

当前已在BW1000 HCU 单卡（GPU1，TP1）完成文本推理验证。容器使用 vLLM `0.28.1+das.29071fe.dtk2610`、PyTorch 2.11.0 和 Transformers 5.21.0，模型权重为 BF16，KV Cache 为 FP8。其他硬件、卡数和量化方式待补充。

Gemma4 的 SWA 层 head dimension 为 256，full attention 层 head dimension 为 512。当前版本启动日志报告异构 head dimension，自动选择 `TRITON_ATTN`。

真实问答验证使用 447 条本地 MMLU-Pro realQA 问题，MTP4 acceptance 为 66.81%，HTTP 错误和抢占均为 0。该数据用于推理负载验证，不是官方 MMLU-Pro 准确率复现。

## 启动命令

### Gemma-4-26B-A4B-it IFB BW 1x vLLM 0.28.1

```bash
export HIP_VISIBLE_DEVICES=1
export VLLM_ENGINE_READY_TIMEOUT_S=1800

vllm serve /mnt/models/gemma-4-26B-A4B-it \
  --served-model-name gemma-4-26b-a4b-it \
  --host 0.0.0.0 \
  --port 8091 \
  --tensor-parallel-size 1 \
  --max-model-len 16384 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 160 \
  --gpu-memory-utilization 0.92 \
  --block-size 64 \
  --enable-prefix-caching \
  --language-model-only \
  --dtype bfloat16 \
  --kv-cache-dtype fp8 \
  --speculative-config '{"method":"mtp","model":"/mnt/models/gemma-4-26B-A4B-it-assistant","num_speculative_tokens":4}'
```

## API 调用

### IFB

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8091/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="gemma-4-26b-a4b-it",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "中国的首都是哪里？"},
    ],
    temperature=0,
    max_tokens=2048,
)
print(response.choices[0].message.content)
```

```bash
curl http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4-26b-a4b-it",
    "temperature": 0,
    "max_tokens": 2048,
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": [
        {"type": "text", "text": "中国的首都是哪里？"}
      ]}
    ]
  }'
```
