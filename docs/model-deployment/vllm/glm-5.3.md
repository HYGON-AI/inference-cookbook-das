# GLM-5.3 on vLLM

## 模型简介

本文提供 GLM-5.3 系列模型在 BW1100、BW1000 上的 vLLM 0.28.1 部署命令，具体权重规格和支持平台见模型列表。

## 模型列表

| 模型权重 | 量化方式 | vLLM 镜像 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | --------- | -------- | ---- | -------- | -------- |
| [hygon/GLM-5.3-Flash-Channel-FP8-w8a8](https://www.modelscope.cn/models/hygon/GLM-5.3-Flash-Channel-FP8-w8a8) | FP8 W8A8 | 0.28.1 | BW1100 | 4 | IFB | [**`>_`**](#glm-53-flash-channel-fp8-w8a8-ifb-bw1100-4x-vllm-0281) |


### 0.28.1 平台支持说明

BW1000 不支持本次发版所列 FP8 权重模型；以下 FP8 权重仅提供 BW1100 启动命令：`hygon/GLM-5.3-Flash-Channel-FP8-w8a8`。

## 启动命令

### GLM-5.3-Flash-Channel-FP8-w8a8 IFB BW1100 4x vLLM 0.28.1

```bash
export VLLM_USE_V2_MODEL_RUNNER=1

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

## API 调用

### IFB

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "hygon/GLM-5.3-Flash-Channel-FP8-w8a8", "messages": [{"role": "user", "content": "中国的首都是哪里？"}], "max_tokens": 128}'
```
