# GLM-5.3 on vLLM

## 模型简介

GLM-5.3 是 Z.ai 推出的 GLM 系列大语言模型，本文档提供其在 vLLM 上的部署示例。

## 模型列表

| 模型权重 | 量化方式 | vLLM 镜像 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | --------- | -------- | ---- | -------- | -------- |
| [hygon/GLM-5.3-Flash-Channel-FP8-w8a8](https://www.modelscope.cn/models/hygon/GLM-5.3-Flash-Channel-FP8-w8a8) | FP8 W8A8 | 0.28.1 | BW1100 | 4 | IFB | [**`>_`**](#glm-53-flash-channel-fp8-w8a8-ifb-bw1100-4x-vllm-0281) |

## 启动命令

### GLM-5.3-Flash-Channel-FP8-w8a8 IFB BW1100 4x vLLM 0.28.1

```bash
vllm serve hygon/GLM-5.3-Flash-Channel-FP8-w8a8 \
  --tensor-parallel-size 4 \
  --attention-backend FLASHMLA_SPARSE \
  --moe-backend aiter \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"use_local_argmax_reduction":true}' \
  --enable-prefix-caching \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64 \
  --default-chat-template-kwargs '{"reasoning_effort":"low"}'
```
