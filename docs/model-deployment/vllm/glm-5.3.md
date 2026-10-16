# GLM-5.3 on vLLM

## 模型简介

本文提供 GLM-5.3 系列模型在 BW1100、BW1000 上的 vLLM 0.28.1 部署命令，具体权重规格和支持平台见模型列表。

## 模型列表

| 模型权重 | 量化方式 | vLLM 镜像 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | --------- | -------- | ---- | -------- | -------- |
| [hygon/GLM-5.3-Flash-Channel-FP8-w8a8](https://www.modelscope.cn/models/hygon/GLM-5.3-Flash-Channel-FP8-w8a8) | FP8 W8A8 | 0.28.1 | BW1100 | 4 | IFB | [**`>_`**](#glm-53-flash-channel-fp8-w8a8-ifb-bw1100-4x-vllm-0281) |
| [hygon/GLM-5.3-Channel-FP8-w8a8](https://www.modelscope.cn/models/hygon/GLM-5.3-Channel-FP8-w8a8) | FP8 W8A8 | 0.28.1 | BW1100 | 8 | IFB | [**`>_`**](#glm-53-channel-fp8-w8a8-ifb-bw1100-8x-vllm-0281) |
| [hygon/GLM-5.3-Channel-INT4-w4a16](https://www.modelscope.cn/models/hygon/GLM-5.3-Channel-INT4-w4a16) | INT4 W4A16 | 0.28.1 | BW1100 | 8 | IFB | [**`>_`**](#glm-53-channel-int4-w4a16-ifb-bw1100-8x-vllm-0281) |
|  | INT4 W4A16 | 0.28.1 | BW1000 | 16 | IFB | [**`>_`**](#glm-53-channel-int4-w4a16-ifb-bw1000-16x-vllm-0281) |


### 0.28.1 平台支持说明

BW1000 不支持本次发版所列 FP8 权重模型；以下 FP8 权重仅提供 BW1100 启动命令：`hygon/GLM-5.3-Flash-Channel-FP8-w8a8`。

第二批平台支持：BW1000 不支持以下 FP8 权重模型，仅提供 BW1100 命令：`hygon/GLM-5.3-Channel-FP8-w8a8`。

## 启动命令

### GLM-5.3-Flash-Channel-FP8-w8a8 IFB BW1100 4x vLLM 0.28.1

```bash

vllm serve hygon/GLM-5.3-Flash-Channel-FP8-w8a8 \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --attention-backend FLASHMLA_SPARSE \
  --moe-backend aiter \
  --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"use_local_argmax_reduction":true}' \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm47 \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
```

### GLM-5.3-Channel-FP8-w8a8 IFB BW1100 8x vLLM 0.28.1

```bash

vllm serve hygon/GLM-5.3-Channel-FP8-w8a8 \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --attention-backend FLASHMLA_SPARSE \
  --moe-backend aiter \
  --kv-cache-dtype fp8_e4m3 \
  --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm47 \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
```

### GLM-5.3-Channel-INT4-w4a16 IFB BW1100 8x vLLM 0.28.1

```bash
export NCCL_GRAPH_MIXING_SUPPORT=1

vllm serve hygon/GLM-5.3-Channel-INT4-w4a16 \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --distributed-executor-backend mp \
  --dtype bfloat16 \
  --safetensors-load-strategy prefetch \
  --attention-backend FLASHMLA_SPARSE \
  --moe-backend aiter \
  --all2all-backend allgather_reducescatter \
  --kv-cache-dtype bfloat16 \
  --block-size 64 \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm47 \
  --no-async-scheduling \
  --api-server-count 1 \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
```

### GLM-5.3-Channel-INT4-w4a16 IFB BW1000 16x vLLM 0.28.1

双节点部署，每个节点 8 卡，总计 16 卡。TP8 × DP2，每个节点运行一个 TP8 副本；两条命令分别在对应节点执行。请将 MASTER_ADDR、MASTER_PORT、DATA_PARALLEL_ADDRESS、DATA_PARALLEL_RPC_PORT 设置为实际部署值，两个节点保持一致。

#### rank 0

```bash
export NCCL_GRAPH_MIXING_SUPPORT=1

vllm serve hygon/GLM-5.3-Channel-INT4-w4a16 \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --data-parallel-size 2 \
  --data-parallel-size-local 1 \
  --data-parallel-backend mp \
  --distributed-executor-backend mp \
  --nnodes 2 \
  --node-rank 0 \
  --master-addr "${MASTER_ADDR}" \
  --master-port "${MASTER_PORT}" \
  --data-parallel-address "${DATA_PARALLEL_ADDRESS}" \
  --data-parallel-rpc-port "${DATA_PARALLEL_RPC_PORT}" \
  --dtype bfloat16 \
  --safetensors-load-strategy prefetch \
  --attention-backend FLASHMLA_SPARSE \
  --moe-backend aiter \
  --all2all-backend allgather_reducescatter \
  --kv-cache-dtype bfloat16 \
  --block-size 64 \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm47 \
  --no-async-scheduling \
  --api-server-count 1 \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
```

#### rank 1

```bash
export NCCL_GRAPH_MIXING_SUPPORT=1

vllm serve hygon/GLM-5.3-Channel-INT4-w4a16 \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --data-parallel-size 2 \
  --data-parallel-size-local 1 \
  --data-parallel-backend mp \
  --distributed-executor-backend mp \
  --nnodes 2 \
  --node-rank 1 \
  --master-addr "${MASTER_ADDR}" \
  --master-port "${MASTER_PORT}" \
  --data-parallel-address "${DATA_PARALLEL_ADDRESS}" \
  --data-parallel-rpc-port "${DATA_PARALLEL_RPC_PORT}" \
  --headless \
  --dtype bfloat16 \
  --safetensors-load-strategy prefetch \
  --attention-backend FLASHMLA_SPARSE \
  --moe-backend aiter \
  --all2all-backend allgather_reducescatter \
  --kv-cache-dtype bfloat16 \
  --block-size 64 \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm47 \
  --no-async-scheduling \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
```

## API 调用

### IFB

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "hygon/GLM-5.3-Flash-Channel-FP8-w8a8", "messages": [{"role": "user", "content": "中国的首都是哪里？"}], "max_tokens": 128}'
```
