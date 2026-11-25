# GLM-4.7 on SGLang

## 模型简介

GLM-4.7 是智谱 AI 推出的大语言模型。GLM-4.7-FP8 是该模型的 FP8 精度权重版本。

## 模型列表

| 模型权重 | 量化方式 | SGLang 镜像 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | ----------- | -------- | ---- | -------- | -------- |
| [ZhipuAI/GLM-4.7-FP8](https://www.modelscope.cn/models/ZhipuAI/GLM-4.7-FP8) | FP8 | [0.5.18](../docker_images.md) | BW1100 | 8 | IFB | [**`>_`**](#glm-47-fp8-ifb-bw1100-8x-sglang-0518) |

## 启动命令

### GLM-4.7-FP8 IFB BW1100 8x SGLang 0.5.18

```bash
export ROCBLAS_USE_HIPBLASLT=1
export SGLANG_USE_FP8_W8A8_MOE=0
export SGLANG_USE_LIGHTOP=1
export SGLANG_ROCM_USE_AITER_MOE=0
export SGLANG_KV_LAYOUT_HCU_FA=true

sglang serve \
  --model-path ZhipuAI/GLM-4.7-FP8 \
  --chunked-prefill-size -1 \
  --max-running-requests 256 \
  --trust-remote-code \
  --nnodes 1 \
  --node-rank 0 \
  --tp-size 8 \
  --pp-size 1 \
  --mem-fraction-static 0.9 \
  --attention-backend fa3 \
  --page-size 64 \
  --moe-runner-backend triton \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --disable-custom-all-reduce \
  --disable-cuda-graph
```

## API 调用

### IFB

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "ZhipuAI/GLM-4.7-FP8", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 128}'
```
