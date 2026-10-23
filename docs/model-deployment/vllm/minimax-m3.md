# MiniMax-M3 on vLLM

## 模型简介

本文提供 MiniMax/MiniMax-M3 BF16 权重在 BW1000 和 K100_AI 上的 vLLM 双节点部署命令。

## 模型列表

| 模型权重 | 量化方式 | vLLM 镜像 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | --------- | -------- | ---- | -------- | -------- |
| [MiniMax/MiniMax-M3](https://www.modelscope.cn/models/MiniMax/MiniMax-M3) | BF16 | 0.28.1 | BW1000 | 16 | IFB | [**`>_`**](#minimax-m3-ifb-bw1000-16x-vllm-0281) |
|  | BF16 | 0.28.1 | K100_AI | 16 | IFB | [**`>_`**](#minimax-m3-ifb-k100_ai-16x-vllm-0281) |

## 启动命令

### MiniMax-M3 IFB BW1000 16x vLLM 0.28.1

总计 16 卡，2 个节点；各节点执行对应 rank 命令，MASTER_ADDR 和 MASTER_PORT 在所有节点设置为相同值。

#### rank 0

```bash
export NCCL_GRAPH_MIXING_SUPPORT=1
export VLLM_KV_CACHE_LAYOUT=HND
export VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=0

vllm serve MiniMax/MiniMax-M3 \
  --trust-remote-code \
  --tensor-parallel-size 16 \
  --enable-expert-parallel \
  --distributed-executor-backend mp \
  --nnodes 2 \
  --node-rank 0 \
  --master-addr "${MASTER_ADDR}" \
  --master-port "${MASTER_PORT}" \
  --attention-backend FLASH_ATTN \
  --moe-backend triton \
  --all2all-backend allgather_reducescatter \
  --kv-cache-dtype bfloat16 \
  --block-size 128 \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser minimax_m3 \
  --reasoning-parser minimax_m3 \
  --no-async-scheduling \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
```

#### rank 1

```bash
export NCCL_GRAPH_MIXING_SUPPORT=1
export VLLM_KV_CACHE_LAYOUT=HND
export VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=0

vllm serve MiniMax/MiniMax-M3 \
  --trust-remote-code \
  --tensor-parallel-size 16 \
  --enable-expert-parallel \
  --distributed-executor-backend mp \
  --nnodes 2 \
  --node-rank 1 \
  --master-addr "${MASTER_ADDR}" \
  --master-port "${MASTER_PORT}" \
  --attention-backend FLASH_ATTN \
  --moe-backend triton \
  --all2all-backend allgather_reducescatter \
  --kv-cache-dtype bfloat16 \
  --block-size 128 \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser minimax_m3 \
  --reasoning-parser minimax_m3 \
  --no-async-scheduling \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64 \
  --headless
```


### MiniMax-M3 IFB K100_AI 16x vLLM 0.28.1

在各节点分别执行对应命令；`MASTER_ADDR` 为 rank 0 节点地址，各节点使用相同的 `MASTER_PORT`。数据并行地址和端口占位符如有使用，也应在各节点保持一致。

#### rank 0

```bash
export NCCL_GRAPH_MIXING_SUPPORT=1
export VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=0
export VLLM_HCU_USE_CUSTOM_OPS=0

vllm serve MiniMax/MiniMax-M3 \
  --trust-remote-code \
  --tensor-parallel-size 16 \
  --enable-expert-parallel \
  --distributed-executor-backend mp \
  --nnodes 2 \
  --node-rank 0 \
  --master-addr "${MASTER_ADDR}" \
  --master-port "${MASTER_PORT}" \
  --attention-backend TRITON_ATTN \
  --moe-backend triton \
  --all2all-backend allgather_reducescatter \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser minimax_m3 \
  --reasoning-parser minimax_m3 \
  --no-async-scheduling \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
```

#### rank 1

```bash
export NCCL_GRAPH_MIXING_SUPPORT=1
export VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=0
export VLLM_HCU_USE_CUSTOM_OPS=0

vllm serve MiniMax/MiniMax-M3 \
  --trust-remote-code \
  --tensor-parallel-size 16 \
  --enable-expert-parallel \
  --distributed-executor-backend mp \
  --nnodes 2 \
  --node-rank 1 \
  --master-addr "${MASTER_ADDR}" \
  --master-port "${MASTER_PORT}" \
  --attention-backend TRITON_ATTN \
  --moe-backend triton \
  --all2all-backend allgather_reducescatter \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser minimax_m3 \
  --reasoning-parser minimax_m3 \
  --no-async-scheduling \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64 \
  --headless
```

## API 调用

### IFB

```bash
curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"MiniMax/MiniMax-M3","messages":[{"role":"user","content":"中国的首都是什么？"}],"temperature":0,"max_tokens":400}'
```
