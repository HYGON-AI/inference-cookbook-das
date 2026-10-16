# GLM-5.3 on vLLM

## 模型简介

GLM-5.3-Flash 是 Z.ai 推出的 GLM 系列大语言模型。本文档提供 Hygon Channelwise FP8 W8A8 量化模型在 BW1100 上的 vLLM 单节点部署示例。

## 模型列表

| 模型权重 | 量化方式 | vLLM 镜像 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | --------- | -------- | ---- | -------- | -------- |
| [hygon/GLM-5.3-Flash-Channel-FP8-w8a8](https://www.modelscope.cn/models/hygon/GLM-5.3-Flash-Channel-FP8-w8a8) | FP8 W8A8 | 0.28.1rc1 | BW1100 | 4 | IFB | [**`>_`**](#glm-53-flash-channel-fp8-w8a8-ifb-bw1100-4x-vllm-0281rc1) |
|  | FP8 W8A8 | 0.28.1rc1 | BW1100 | 8 | PCP8+EP8 | [**`>_`**](#glm-53-flash-channel-fp8-w8a8-pcp8ep8-bw1100-8x-vllm-0281rc1) |
|  | FP8 W8A8 | 0.28.1rc1 | BW1100 | 8 | DP8+EP8 | [**`>_`**](#glm-53-flash-channel-fp8-w8a8-dp8ep8-bw1100-8x-vllm-0281rc1) |

## 启动命令

### GLM-5.3-Flash-Channel-FP8-w8a8 IFB BW1100 4x vLLM 0.28.1rc1

```bash
export VLLM_HCU_INDEXER_KCACHE_LAYOUT=normal
export VLLM_USE_V2_MODEL_RUNNER=1

vllm serve hygon/GLM-5.3-Flash-Channel-FP8-w8a8 \
  --tensor-parallel-size 1 \
  --data-parallel-size 4 \
  --data-parallel-size-local 4 \
  --enable-expert-parallel \
  --all2all-backend allgather_reducescatter \
  --attention-backend FLASHMLA_SPARSE \
  --moe-backend aiter \
  --kv-cache-dtype fp8_ds_mla \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"use_local_argmax_reduction":true}' \
  --reasoning-parser glm45 \
  --enable-prefix-caching \
  --max-model-len 65536 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.9 \
  --language-model-only \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'
```

### GLM-5.3-Flash-Channel-FP8-w8a8 PCP8+EP8 BW1100 8x vLLM 0.28.1rc1

```bash
export VLLM_HCU_INDEXER_KCACHE_LAYOUT=normal
export VLLM_USE_V2_MODEL_RUNNER=1

vllm serve hygon/GLM-5.3-Flash-Channel-FP8-w8a8 \
  --tensor-parallel-size 1 \
  --prefill-context-parallel-size 8 \
  --decode-context-parallel-size 1 \
  --data-parallel-size 1 \
  --data-parallel-size-local 1 \
  --enable-expert-parallel \
  --all2all-backend deepep_low_latency \
  --attention-backend FLASHMLA_SPARSE \
  --kv-cache-dtype fp8_ds_mla \
  --moe-backend deep_gemm \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"use_local_argmax_reduction":true}' \
  --reasoning-parser glm45 \
  --enforce-eager \
  --language-model-only \
  --enable-prefix-caching \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 32
```

### GLM-5.3-Flash-Channel-FP8-w8a8 DP8+EP8 BW1100 8x vLLM 0.28.1rc1

```bash
export VLLM_HCU_INDEXER_KCACHE_LAYOUT=normal
export VLLM_USE_V2_MODEL_RUNNER=1

vllm serve hygon/GLM-5.3-Flash-Channel-FP8-w8a8 \
  --tensor-parallel-size 1 \
  --data-parallel-size 8 \
  --data-parallel-size-local 8 \
  --enable-expert-parallel \
  --all2all-backend deepep_low_latency \
  --attention-backend FLASHMLA_SPARSE \
  --kv-cache-dtype fp8_ds_mla \
  --moe-backend deep_gemm \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"use_local_argmax_reduction":true}' \
  --reasoning-parser glm45 \
  --enable-prefix-caching \
  --max-model-len 32768 \
  --max-num-batched-tokens 128 \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.90 \
  --language-model-only \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'
```

## API 调用

### IFB

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="hygon/GLM-5.3-Flash-Channel-FP8-w8a8",
    messages=[
        {"role": "system", "content": "你是一个有帮助的 AI 助手。"},
        {"role": "user", "content": "请分析一下当前中国 AI 芯片产业的发展现状"},
    ],
    max_tokens=2048,
)
print(response.choices[0].message.content)
```

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "hygon/GLM-5.3-Flash-Channel-FP8-w8a8",
    "messages": [
      {"role": "system", "content": "你是一个有帮助的 AI 助手。"},
      {"role": "user", "content": "中国的首都是哪里？"}
    ],
    "max_tokens": 128
  }'
```
