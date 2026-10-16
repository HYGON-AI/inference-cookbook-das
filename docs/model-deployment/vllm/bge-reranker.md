# bge-reranker-v2-m3 on vLLM

## 模型简介

bge-reranker-v2-m3 是 BAAI 发布的 reranker 模型，基于 M3 架构，用于对检索结果进行重排序，提升 RAG 系统的召回质量。

## 模型列表

| 模型权重 | 量化方式 | vLLM 镜像 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | --------- | -------- | ---- | -------- | -------- |
| [BAAI/bge-reranker-v2-m3](https://www.modelscope.cn/models/BAAI/bge-reranker-v2-m3) | BF16 | 0.28.1 | BW1000 | 1 | IFB | [**`>_`**](#bge-reranker-v2-m3-ifb-bw1000-1x-vllm-0281) |
|  | BF16 | 0.21 | BW1000 | 1 | IFB | [**`>_`**](#bge-reranker-v2-m3-ifb-bw1000-1x-vllm-021) |
|  | BF16 | 0.18 | BW1000 | 1 | IFB | [**`>_`**](#bge-reranker-v2-m3-ifb-bw1000-1x-vllm-018) |

## 启动命令

### bge-reranker-v2-m3 IFB BW1000 1x vLLM 0.28.1

重排序 pooling 任务；模型有效输入长度上限按 8192 配置。

```bash

vllm serve BAAI/bge-reranker-v2-m3 \
  --trust-remote-code \
  --runner pooling \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --attention-backend FLASH_ATTN \
  --no-enable-prefix-caching \
  --max-model-len 8192 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
```


### bge-reranker-v2-m3 IFB BW1000 1x vLLM 0.21

```bash
vllm serve BAAI/bge-reranker-v2-m3 --attention-backend FLASH_ATTN_CUSTOM
```

### bge-reranker-v2-m3 IFB BW1000 1x vLLM 0.18

```bash
export VLLM_HCU_USE_CUSTOM_FLASH_ATTN=1
vllm serve BAAI/bge-reranker-v2-m3
```

## API 调用

### IFB

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.post(
    "/rerank",
    json={
        "model": "BAAI/bge-reranker-v2-m3",
        "query": "什么是深度学习？",
        "documents": [
            "深度学习是机器学习的一个分支",
            "今天天气很好",
        ],
    },
)
```

```bash
curl http://localhost:8000/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{"model": "BAAI/bge-reranker-v2-m3", "query": "什么是深度学习？", "documents": ["深度学习是机器学习的一个分支", "今天天气很好"]}'
```
