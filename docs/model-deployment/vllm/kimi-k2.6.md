# Kimi-K2.6 on vLLM

## 模型简介

本文提供 Kimi-K2.6 系列模型在 BW1100、BW1000、K100_AI 上的 vLLM 0.28.1 部署命令，具体权重规格和支持平台见模型列表。

## 模型列表

| 模型权重 | 量化方式 | vLLM 镜像 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | --------- | -------- | ---- | -------- | -------- |
| [moonshotai/Kimi-K2.6](https://www.modelscope.cn/models/moonshotai/Kimi-K2.6) | INT4 W4A16 | 0.28.1 | BW1100 | 8 | IFB | [**`>_`**](#kimi-k26-ifb-bw1100-8x-vllm-0281) |
|  | INT4 W4A16 | 0.28.1 | BW1000 | 16 | IFB | [**`>_`**](#kimi-k26-ifb-bw1000-16x-vllm-0281) |
|  | INT4 W4A16 | 0.28.1 | K100_AI | 16 | IFB | [**`>_`**](#kimi-k26-ifb-k100_ai-16x-vllm-0281) |

## 启动命令

### Kimi-K2.6 IFB BW1100 8x vLLM 0.28.1

```bash

vllm serve moonshotai/Kimi-K2.6 \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --attention-backend FLASHMLA \
  --moe-backend aiter \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser kimi_k2 \
  --reasoning-parser kimi_k2 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
```

### Kimi-K2.6 IFB BW1000 16x vLLM 0.28.1

双节点部署，每个节点 8 卡，总计 16 卡。两个节点使用相同的 `MASTER_ADDR` 和 `MASTER_PORT`，分别执行对应 rank 命令。

#### rank 0

```bash
vllm serve moonshotai/Kimi-K2.6  \
  --trust-remote-code \
  --tensor-parallel-size 16 \
  --enable-expert-parallel \
  --distributed-executor-backend mp \
  --nnodes 2 \
  --node-rank 0 \
  --master-addr "${MASTER_ADDR}" \
  --master-port "${MASTER_PORT}" \
  --dtype bfloat16 \
  --attention-backend FLASHMLA \
  --moe-backend aiter \
  --all2all-backend allgather_reducescatter \
  --kv-cache-dtype fp8_e5m2 \
  --block-size 64 \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser kimi_k2 \
  --reasoning-parser kimi_k2 \
  --cpu-offload-gb 0 \
  --no-async-scheduling \
  --max-model-len 65536 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
```

#### rank 1

```bash
vllm serve moonshotai/Kimi-K2.6 \
  --trust-remote-code \
  --tensor-parallel-size 16 \
  --enable-expert-parallel \
  --distributed-executor-backend mp \
  --nnodes 2 \
  --node-rank 1 \
  --master-addr "${MASTER_ADDR}" \
  --master-port "${MASTER_PORT}" \
  --dtype bfloat16 \
  --attention-backend FLASHMLA \
  --moe-backend aiter \
  --all2all-backend allgather_reducescatter \
  --kv-cache-dtype fp8_e5m2 \
  --block-size 64 \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser kimi_k2 \
  --reasoning-parser kimi_k2 \
  --cpu-offload-gb 0 \
  --no-async-scheduling \
  --max-model-len 65536 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64 \
  --headless
```

### Kimi-K2.6 IFB K100_AI 16x vLLM 0.28.1

双节点部署，每个节点 8 卡，总计 16 卡。两个节点使用相同的 `MASTER_ADDR` 和 `MASTER_PORT`，分别执行对应 rank 命令。

#### rank 0

```bash
export NCCL_GRAPH_MIXING_SUPPORT=1
export VLLM_USE_BREAKABLE_CUDAGRAPH=1
export VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1
export VLLM_HCU_USE_CUSTOM_OPS=0

vllm serve moonshotai/Kimi-K2.6 \
  --trust-remote-code \
  --tensor-parallel-size 16 \
  --enable-expert-parallel \
  --distributed-executor-backend mp \
  --nnodes 2 \
  --node-rank 0 \
  --master-addr "${MASTER_ADDR}" \
  --master-port "${MASTER_PORT}" \
  --dtype bfloat16 \
  --attention-backend TRITON_MLA \
  --moe-backend triton \
  --all2all-backend allgather_reducescatter \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser kimi_k2 \
  --reasoning-parser kimi_k2 \
  --cpu-offload-gb 0 \
  --no-async-scheduling \
  --max-model-len 65536 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
```

#### rank 1

```bash
export NCCL_GRAPH_MIXING_SUPPORT=1
export VLLM_USE_BREAKABLE_CUDAGRAPH=1
export VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1
export VLLM_HCU_USE_CUSTOM_OPS=0

vllm serve moonshotai/Kimi-K2.6 \
  --trust-remote-code \
  --tensor-parallel-size 16 \
  --enable-expert-parallel \
  --distributed-executor-backend mp \
  --nnodes 2 \
  --node-rank 1 \
  --master-addr "${MASTER_ADDR}" \
  --master-port "${MASTER_PORT}" \
  --dtype bfloat16 \
  --attention-backend TRITON_MLA \
  --moe-backend triton \
  --all2all-backend allgather_reducescatter \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser kimi_k2 \
  --reasoning-parser kimi_k2 \
  --cpu-offload-gb 0 \
  --no-async-scheduling \
  --max-model-len 65536 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64 \
  --headless
```

## API 调用

### IFB

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "moonshotai/Kimi-K2.6", "messages": [{"role": "user", "content": "中国的首都是哪里？"}], "max_tokens": 128}'
```
