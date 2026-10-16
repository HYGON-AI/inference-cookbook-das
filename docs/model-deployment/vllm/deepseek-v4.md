# DeepSeek-V4 on vLLM

## 模型简介

DeepSeek-V4 是 DeepSeek 系列大语言模型，本文档提供其在 vLLM 上的部署示例。

## 模型列表

| 模型权重 | 量化方式 | vLLM 镜像 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | --------- | -------- | ---- | -------- | -------- |
| [hygon/DeepSeek-V4-Flash-Channel-FP8-w8a8](https://www.modelscope.cn/models/hygon/DeepSeek-V4-Flash-Channel-FP8-w8a8) | FP8 W8A8 | 0.28.1 | BW1100 | 8 | IFB | [**`>_`**](#deepseek-v4-flash-channel-fp8-w8a8-ifb-bw1100-8x-vllm-0281) |
| [hygon/DeepSeek-V4-Flash-0731-Channel-INT8-w8a8](https://www.modelscope.cn/models/hygon/DeepSeek-V4-Flash-0731-Channel-INT8-w8a8) | INT8 W8A8 | 0.28.1 | BW1100 | 8 | IFB | [**`>_`**](#deepseek-v4-flash-0731-channel-int8-w8a8-ifb-bw1100-8x-vllm-0281) |
|  | INT8 W8A8 | 0.28.1 | BW1000 | 16 | IFB | [**`>_`**](#deepseek-v4-flash-0731-channel-int8-w8a8-ifb-bw1000-16x-vllm-0281) |
|  | INT8 W8A8 | 0.25.1 | BW1000 | 8 | IFB | [**`>_`**](#deepseek-v4-flash-0731-channel-int8-w8a8-ifb-bw1000-8x-vllm-0251) |
| [hygon/DeepSeek-V4-Flash-0731-Channel-FP8-w8a8](https://www.modelscope.cn/models/hygon/DeepSeek-V4-Flash-0731-Channel-FP8-w8a8) | FP8 W8A8 | 0.28.1 | BW1100 | 8 | IFB | [**`>_`**](#deepseek-v4-flash-0731-channel-fp8-w8a8-ifb-bw1100-8x-vllm-0281) |
|  | FP8 W8A8 | 0.25.1 | BW1100 | 8 | IFB | [**`>_`**](#deepseek-v4-flash-0731-channel-fp8-w8a8-ifb-bw1100-8x-vllm-0251) |
| [hygon/DeepSeek-V4-Flash-0731-W4A8-INT4-Channel-Attn-W8A8-INT8-Channel](https://www.modelscope.cn/models/hygon/DeepSeek-V4-Flash-0731-W4A8-INT4-Channel-Attn-W8A8-INT8-Channel) | INT4 W4A8 | 0.28.1 | BW1100 | 8 | IFB | [**`>_`**](#deepseek-v4-flash-0731-w4a8-int4-channel-attn-w8a8-int8-channel-ifb-bw1100-8x-vllm-0281) |
|  | INT4 W4A8 | 0.28.1 | BW1000 | 8 | IFB | [**`>_`**](#deepseek-v4-flash-0731-w4a8-int4-channel-attn-w8a8-int8-channel-ifb-bw1000-8x-vllm-0281) |
| [hygon/DeepSeek-V4-Pro-0813-Channel-INT4-w4a8](https://www.modelscope.cn/models/hygon/DeepSeek-V4-Pro-0813-Channel-INT4-w4a8) | INT4 W4A8 | 0.28.1 | BW1100 | 8 | IFB | [**`>_`**](#deepseek-v4-pro-0813-channel-int4-w4a8-ifb-bw1100-8x-vllm-0281) |
| [hygon/DeepSeek-V4-Pro-0813-INT4-Channel](https://www.modelscope.cn/models/hygon/DeepSeek-V4-Pro-0813-INT4-Channel) | INT4 | 0.28.1 | BW1100 | 8 | IFB | [**`>_`**](#deepseek-v4-pro-0813-int4-channel-ifb-bw1100-8x-vllm-0281) |
| [hygon/DeepSeek-V4-Flash-Channel-INT8-w8a8](https://www.modelscope.cn/models/hygon/DeepSeek-V4-Flash-Channel-INT8-w8a8) | INT8 W8A8 | 0.28.1 | BW1100 | 8 | IFB | [**`>_`**](#deepseek-v4-flash-channel-int8-w8a8-ifb-bw1100-8x-vllm-0281) |
|  | INT8 W8A8 | 0.28.1 | BW1000 | 8 | IFB | [**`>_`**](#deepseek-v4-flash-channel-int8-w8a8-ifb-bw1000-8x-vllm-0281) |


### 0.28.1 平台支持说明

BW1000 不支持本次发版所列 FP8 权重模型；以下 FP8 权重仅提供 BW1100 启动命令：`hygon/DeepSeek-V4-Flash-0731-Channel-FP8-w8a8`、`hygon/DeepSeek-V4-Flash-Channel-FP8-w8a8`。

本次发版仅提供以下模型的 BW1100 命令，不提供 BW1000 启动命令：`hygon/DeepSeek-V4-Pro-0813-Channel-INT4-w4a8`、`hygon/DeepSeek-V4-Pro-0813-INT4-Channel`。

## 启动命令

### DeepSeek-V4-Flash-Channel-FP8-w8a8 IFB BW1100 8x vLLM 0.28.1

```bash

vllm serve hygon/DeepSeek-V4-Flash-Channel-FP8-w8a8 \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --distributed-executor-backend mp \
  --tokenizer-mode deepseek_v4 \
  --moe-backend aiter \
  --kv-cache-dtype fp8_e4m3 \
  --block-size 256 \
  --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --enable-auto-tool-choice \
  --tool-call-parser deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"","reasoning_end_str":""}' \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
```

### DeepSeek-V4-Flash-0731-Channel-INT8-w8a8 IFB BW1100 8x vLLM 0.28.1

```bash

vllm serve hygon/DeepSeek-V4-Flash-0731-Channel-INT8-w8a8 \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --distributed-executor-backend mp \
  --tokenizer-mode deepseek_v4 \
  --moe-backend aiter \
  --kv-cache-dtype fp8_e4m3 \
  --block-size 256 \
  --enable-prefix-caching \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}' \
  --enable-auto-tool-choice \
  --tool-call-parser deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"","reasoning_end_str":""}' \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
```

### DeepSeek-V4-Flash-0731-Channel-INT8-w8a8 IFB BW1000 16x vLLM 0.28.1

在各节点分别执行对应命令；`MASTER_ADDR` 为 rank 0 节点地址，各节点使用相同的 `MASTER_PORT`。数据并行地址和端口占位符如有使用，也应在各节点保持一致。

#### rank 0

```bash
export NCCL_GRAPH_MIXING_SUPPORT=1

vllm serve hygon/DeepSeek-V4-Flash-0731-Channel-INT8-w8a8 \
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
  --quantization compressed-tensors \
  --tokenizer-mode deepseek_v4 \
  --attention-backend FLASHMLA_SPARSE \
  --moe-backend aiter \
  --all2all-backend allgather_reducescatter \
  --kv-cache-dtype fp8_ds_mla \
  --block-size 256 \
  --enable-prefix-caching \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}' \
  --enable-auto-tool-choice \
  --tool-call-parser deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"","reasoning_end_str":""}' \
  --enforce-eager \
  --no-async-scheduling \
  --api-server-count 1 \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
```

#### rank 1

```bash
export NCCL_GRAPH_MIXING_SUPPORT=1

vllm serve hygon/DeepSeek-V4-Flash-0731-Channel-INT8-w8a8 \
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
  --quantization compressed-tensors \
  --tokenizer-mode deepseek_v4 \
  --attention-backend FLASHMLA_SPARSE \
  --moe-backend aiter \
  --all2all-backend allgather_reducescatter \
  --kv-cache-dtype fp8_ds_mla \
  --block-size 256 \
  --enable-prefix-caching \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}' \
  --enable-auto-tool-choice \
  --tool-call-parser deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"","reasoning_end_str":""}' \
  --enforce-eager \
  --no-async-scheduling \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
```

### DeepSeek-V4-Flash-0731-Channel-INT8-w8a8 IFB BW1000 8x vLLM 0.25.1

```bash
vllm serve hygon/DeepSeek-V4-Flash-0731-Channel-INT8-w8a8 \
  --dtype bfloat16 \
  --kv-cache-dtype fp8 \
  --tensor-parallel-size 8 \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64 \
  --block-size 256 \
  --trust-remote-code \
  --tokenizer-mode deepseek_v4 \
  --distributed-executor-backend mp \
  --moe-backend aiter \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}'
```

### DeepSeek-V4-Flash-0731-Channel-FP8-w8a8 IFB BW1100 8x vLLM 0.28.1

```bash

vllm serve hygon/DeepSeek-V4-Flash-0731-Channel-FP8-w8a8 \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --distributed-executor-backend mp \
  --tokenizer-mode deepseek_v4 \
  --moe-backend aiter \
  --kv-cache-dtype fp8_e4m3 \
  --block-size 256 \
  --enable-prefix-caching \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}' \
  --enable-auto-tool-choice \
  --tool-call-parser deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"","reasoning_end_str":""}' \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
```

### DeepSeek-V4-Flash-0731-Channel-FP8-w8a8 IFB BW1100 8x vLLM 0.25.1

```bash
vllm serve hygon/DeepSeek-V4-Flash-0731-Channel-FP8-w8a8 \
  --dtype bfloat16 \
  --kv-cache-dtype fp8 \
  --tensor-parallel-size 8 \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64 \
  --block-size 256 \
  --trust-remote-code \
  --tokenizer-mode deepseek_v4 \
  --distributed-executor-backend mp \
  --moe-backend aiter \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}'
```
### DeepSeek-V4-Flash-0731-W4A8-INT4-Channel-Attn-W8A8-INT8-Channel IFB BW1100 8x vLLM 0.28.1

```bash

vllm serve hygon/DeepSeek-V4-Flash-0731-W4A8-INT4-Channel-Attn-W8A8-INT8-Channel \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --distributed-executor-backend mp \
  --quantization slimquant_w4a8 \
  --tokenizer-mode deepseek_v4 \
  --moe-backend aiter \
  --kv-cache-dtype fp8_e4m3 \
  --block-size 256 \
  --enable-prefix-caching \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}' \
  --enable-auto-tool-choice \
  --tool-call-parser deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"","reasoning_end_str":""}' \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
```

### DeepSeek-V4-Flash-0731-W4A8-INT4-Channel-Attn-W8A8-INT8-Channel IFB BW1000 8x vLLM 0.28.1

```bash
export NCCL_GRAPH_MIXING_SUPPORT=1
export VLLM_USE_BREAKABLE_CUDAGRAPH=1

vllm serve hygon/DeepSeek-V4-Flash-0731-W4A8-INT4-Channel-Attn-W8A8-INT8-Channel \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --data-parallel-size 2 \
  --data-parallel-size-local 2 \
  --data-parallel-backend mp \
  --distributed-executor-backend mp \
  --quantization slimquant_w4a8 \
  --tokenizer-mode deepseek_v4 \
  --attention-backend FLASHMLA_SPARSE \
  --moe-backend auto \
  --all2all-backend deepep_auto \
  --kv-cache-dtype bfloat16 \
  --block-size 64 \
  --enable-prefix-caching \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}' \
  --enable-auto-tool-choice \
  --tool-call-parser deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"","reasoning_end_str":""}' \
  --no-async-scheduling \
  --api-server-count 1 \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
```

### DeepSeek-V4-Pro-0813-Channel-INT4-w4a8 IFB BW1100 8x vLLM 0.28.1

```bash

vllm serve hygon/DeepSeek-V4-Pro-0813-Channel-INT4-w4a8 \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --distributed-executor-backend mp \
  --quantization slimquant_w4a8 \
  --tokenizer-mode deepseek_v4 \
  --moe-backend aiter \
  --kv-cache-dtype fp8_e4m3 \
  --block-size 256 \
  --enable-prefix-caching \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}' \
  --enable-auto-tool-choice \
  --tool-call-parser deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
```

### DeepSeek-V4-Pro-0813-INT4-Channel IFB BW1100 8x vLLM 0.28.1

```bash

vllm serve hygon/DeepSeek-V4-Pro-0813-INT4-Channel \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --distributed-executor-backend mp \
  --quantization slimquant_w4a8 \
  --tokenizer-mode deepseek_v4 \
  --moe-backend aiter \
  --kv-cache-dtype fp8_e4m3 \
  --block-size 256 \
  --enable-prefix-caching \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}' \
  --enable-auto-tool-choice \
  --tool-call-parser deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
```

### DeepSeek-V4-Flash-Channel-INT8-w8a8 IFB BW1100 8x vLLM 0.28.1

```bash
export NCCL_GRAPH_MIXING_SUPPORT=1
export VLLM_USE_BREAKABLE_CUDAGRAPH=1

vllm serve hygon/DeepSeek-V4-Flash-Channel-INT8-w8a8 \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --distributed-executor-backend mp \
  --quantization compressed-tensors \
  --safetensors-load-strategy prefetch \
  --tokenizer-mode deepseek_v4 \
  --attention-backend FLASHMLA_SPARSE \
  --moe-backend aiter \
  --kv-cache-dtype fp8_ds_mla \
  --block-size 256 \
  --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"use_local_argmax_reduction":false}' \
  --enable-auto-tool-choice \
  --tool-call-parser deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"","reasoning_end_str":""}' \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
```

### DeepSeek-V4-Flash-Channel-INT8-w8a8 IFB BW1000 8x vLLM 0.28.1

```bash
export NCCL_GRAPH_MIXING_SUPPORT=1
export VLLM_USE_BREAKABLE_CUDAGRAPH=1

vllm serve hygon/DeepSeek-V4-Flash-Channel-INT8-w8a8 \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --distributed-executor-backend mp \
  --quantization compressed-tensors \
  --safetensors-load-strategy prefetch \
  --tokenizer-mode deepseek_v4 \
  --attention-backend FLASHMLA_SPARSE \
  --moe-backend aiter \
  --kv-cache-dtype fp8_ds_mla \
  --block-size 256 \
  --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"use_local_argmax_reduction":false}' \
  --enable-auto-tool-choice \
  --tool-call-parser deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"","reasoning_end_str":""}' \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
```

## API 调用

### IFB

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "hygon/DeepSeek-V4-Flash-Channel-FP8-w8a8", "messages": [{"role": "user", "content": "中国的首都是哪里？"}], "max_tokens": 128}'
```
