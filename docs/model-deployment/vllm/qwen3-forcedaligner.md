# Qwen3-ForcedAligner on vLLM

## 模型简介

[Qwen3-ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B) 用于将音频与已有文本对齐，预测词级时间戳。它执行 token classification，输出时间 bin 分类结果，需要结合模型的对齐后处理转换为时间戳。

本文记录已完成服务启动验证的配置，音频与文本对齐的推理验证、时间戳精度评测仍待完成。

---

## 模型列表

| 模型权重 | 量化方式 | vLLM 镜像 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | --------- | -------- | ---- | -------- | -------- |
| [Qwen/Qwen3-ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B) | BF16 | [0.21](../docker_images.md) | BW1100 | 1x | IFB | [**`>_`**](#qwen3-forcedaligner-06b-ifb-bw1100-1x-vllm-021) |

使用与 [Qwen3-ASR 验证环境](qwen3-asr.md#验证环境) 相同的镜像：

```bash
docker pull harbor.sourcefind.cn:5443/hcu/admin/base/vllm:0.21.0-ubuntu22.04-dtk2604-py3.10
```

---

## 启动命令

### Qwen3-ForcedAligner-0.6B IFB BW1100 1x vLLM 0.21

```bash
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE=1
export VLLM_HCU_USE_FLASH_ATTN_UNIFIED=1
export VLLM_HCU_USE_AITER_MOE_CONFIG=1

vllm serve Qwen/Qwen3-ForcedAligner-0.6B \
  --hf-overrides '{"architectures": ["Qwen3ASRForcedAlignerForTokenClassification"]}' \
  -tp 1 \
  --max-num-batched-tokens 8192
```

`--hf-overrides` 指定时间戳分类架构。使用本地权重时，应选择 `Qwen3-ForcedAligner-0.6B` 的权重目录。

## API 调用

### IFB

#### 服务就绪检查

服务使用默认端口 `8000`。以下命令检查健康状态及模型列表：

```bash
curl --fail http://localhost:8000/health
curl --fail http://localhost:8000/v1/models
```

健康检查通过仅表示服务就绪，不能证明音频与文本对齐结果正确。此模型执行时间戳分类，不应使用 ASR 的 `/v1/chat/completions` 文本生成请求作为对齐测试。

#### 对齐验证状态

音频与已知文本的请求格式、时间 bin 到时间戳的后处理，以及时间戳误差评测尚待验证，可参考 [Qwen 官方 ForcedAligner 示例](https://github.com/QwenLM/Qwen3-ASR/blob/main/examples/example_qwen3_forced_aligner.py) 了解对齐任务的输入输出。

EvalScope 的 `wenet_speech` ASR 文本转写评测不能直接用于此模型的时间戳精度验证。完成对齐验证后，应补充对应客户端命令、评测数据和时间戳误差。
