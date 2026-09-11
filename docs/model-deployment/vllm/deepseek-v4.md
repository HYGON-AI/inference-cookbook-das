# DeepSeek-V4 on vLLM

## 模型简介

DeepSeek-V4 是 DeepSeek 系列大语言模型，本文档提供其在 vLLM 上的部署示例。

## 模型列表

| 模型权重 | 量化方式 | vLLM 镜像 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | --------- | -------- | ---- | -------- | -------- |
| [hygon/DeepSeek-V4-Flash-0731-Channel-INT8-w8a8](https://www.modelscope.cn/models/hygon/DeepSeek-V4-Flash-0731-Channel-INT8-w8a8) | INT8 W8A8 | 0.28.1 | BW1000 | 8 | IFB | [**`>_`**](#deepseek-v4-flash-0731-channel-int8-w8a8-ifb-bw1000-8x-vllm-0281) |
| [hygon/DeepSeek-V4-Flash-0731-Channel-FP8-w8a8](https://www.modelscope.cn/models/hygon/DeepSeek-V4-Flash-0731-Channel-FP8-w8a8) | FP8 W8A8 | 0.28.1 | BW1000 | 8 | IFB | [**`>_`**](#deepseek-v4-flash-0731-channel-fp8-w8a8-ifb-bw1000-8x-vllm-0281) |

## 启动命令

### DeepSeek-V4-Flash-0731-Channel-INT8-w8a8 IFB BW1000 8x vLLM 0.28.1

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

### DeepSeek-V4-Flash-0731-Channel-FP8-w8a8 IFB BW1000 8x vLLM 0.28.1

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
