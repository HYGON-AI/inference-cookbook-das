# Qwen3-Omni on vLLM-Omni

## 模型简介

Qwen3-Omni-30B-A3B-Instruct 是阿里通义千问推出的全模态（Omni）模型，基于 MoE 架构，总参数 30B、激活参数 3B，可同时处理文本、图像、音频和视频输入并输出文本。

## 模型列表

| 模型权重 | 量化方式 | vLLM-Omni 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | -------------- | -------- | ---- | -------- | -------- |
| [Qwen/Qwen3-Omni-30B-A3B-Instruct](https://www.modelscope.cn/models/Qwen/Qwen3-Omni-30B-A3B-Instruct) | BF16 | 0.29 | BW1000 | 2 | IFB | [**`>_`**](#qwen3-omni-30b-a3b-instruct-ifb-bw1000-2x-vllm-omni-029) |
|  | BF16 | 0.29 | BW1100 | 1 | IFB | [**`>_`**](#qwen3-omni-30b-a3b-instruct-ifb-bw1100-1x-vllm-omni-029) |

本文使用 `thinker-only` pipeline，提供纯文本生成和音频理解的部署及调用示例，只输出文本，不包含语音生成阶段。

## 部署准备

`--deploy-config` 接收 YAML 文件路径，请先将本文 [YAML 配置](#yaml-配置) 中对应硬件的内容保存为启动命令里使用的文件名：BW1000 使用 `qwen3_omni_moe_thinking_tp2.yaml`，BW1100 使用 `qwen3_omni_moe_thinking_tp1.yaml`。

本文启动和 API 示例统一使用官方模型名称。使用本地权重时，需要将启动命令和所有 API 请求中的模型名称同步替换为本地模型路径。可通过 `/v1/models` 确认实际服务模型名称。

通过 `HIP_VISIBLE_DEVICES` 选择空闲物理卡，YAML 中的 `devices` 使用所选卡的逻辑编号。以下示例分别使用物理卡 `0,1` 和 `0`；更换物理卡时，只修改 `HIP_VISIBLE_DEVICES`，保持对应 YAML 的逻辑设备及 TP 值不变。

启动前需要设置以下环境变量：

```bash
export VLLM_HCU_USE_CUSTOM_OPS="1"
export VLLM_ROCM_USE_AITER="1"
export VLLM_ROCM_USE_AITER_MOE="1"
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_FLASH_ATTN_UNIFIED=1
export VLLM_HCU_USE_AITER_MOE_CONFIG=1
export VLLM_HCU_USE_AITER_SAMPLER=1
```

## 启动 Server

### Qwen3-Omni-30B-A3B-Instruct IFB BW1000 2x vLLM-Omni 0.29

使用 `qwen3_omni_moe_thinking_tp2.yaml`，thinker stage 张量并行度为 2，占用设备 0、1：

```bash
export HIP_VISIBLE_DEVICES="0,1"
export VLLM_HCU_USE_CUSTOM_OPS="1"
export VLLM_ROCM_USE_AITER="1"
export VLLM_ROCM_USE_AITER_MOE="1"
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_FLASH_ATTN_UNIFIED=1
export VLLM_HCU_USE_AITER_MOE_CONFIG=1
export VLLM_HCU_USE_AITER_SAMPLER=1

vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --omni \
  --host 0.0.0.0 \
  --deploy-config ./qwen3_omni_moe_thinking_tp2.yaml
```

### Qwen3-Omni-30B-A3B-Instruct IFB BW1100 1x vLLM-Omni 0.29

使用 `qwen3_omni_moe_thinking_tp1.yaml`，thinker stage 张量并行度为 1，占用设备 0：

```bash
export HIP_VISIBLE_DEVICES="0"
export VLLM_HCU_USE_CUSTOM_OPS="1"
export VLLM_ROCM_USE_AITER="1"
export VLLM_ROCM_USE_AITER_MOE="1"
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_FLASH_ATTN_UNIFIED=1
export VLLM_HCU_USE_AITER_MOE_CONFIG=1
export VLLM_HCU_USE_AITER_SAMPLER=1

vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --omni \
  --host 0.0.0.0 \
  --deploy-config ./qwen3_omni_moe_thinking_tp1.yaml
```

本文示例使用 vLLM 默认端口 `8000`；如指定其他端口，请同步修改健康检查和 API 地址。

首次启动包含权重加载、编译和图捕获，请等待服务就绪后检查：

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/v1/models
```

## API 调用

### 音频理解

```python
import base64
from openai import OpenAI

with open("audio.wav", "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode()

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="Qwen/Qwen3-Omni-30B-A3B-Instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "请描述这段音频中的内容。"},
                {
                    "type": "audio_url",
                    "audio_url": {"url": f"data:audio/wav;base64,{audio_b64}"},
                },
            ],
        }
    ],
    max_tokens=2048,
)

print(response.choices[0].message.content)
```

### 纯文本

```python
response = client.chat.completions.create(
    model="Qwen/Qwen3-Omni-30B-A3B-Instruct",
    messages=[{"role": "user", "content": "你好，请用一句话介绍你自己。"}],
    max_tokens=512,
)
```

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    "messages": [{"role": "user", "content": "你好，请用一句话介绍你自己。"}],
    "max_tokens": 128
  }'
```

## YAML 配置

两份配置仅在 `stages[0].devices` 和 `stages[0].tensor_parallel_size` 上不同，用于匹配 BW1000 双卡与 BW1100 单卡。

### qwen3_omni_moe_thinking_tp2.yaml

用于 BW1000，张量并行度为 2：

```yaml
# Stage config for Qwen3-Omni-30B-A3B-Instruct audio-to-text output.
# This is a single thinker stage config with tensor parallel size 2.

pipeline: qwen3_omni_moe_thinker_only
distributed_executor_backend: mp
enable_prefix_caching: false
trust_remote_code: true

stages:
  - stage_id: 0
    devices: "0,1"
    tensor_parallel_size: 2
    max_num_seqs: 128
    gpu_memory_utilization: 0.9
    enforce_eager: false
    async_scheduling: false
    moe_backend: aiter
    attention_backend: FLASH_ATTN_CLASSIC
    default_sampling_params:
      temperature: 0.4
      top_p: 0.9
      top_k: 1
      max_tokens: 2048
      seed: 42
      detokenize: true
      repetition_penalty: 1.05
```

### qwen3_omni_moe_thinking_tp1.yaml

用于 BW1100，张量并行度为 1：

```yaml
# Stage config for Qwen3-Omni-30B-A3B-Instruct audio-to-text output.
# This is a single thinker stage config with tensor parallel size 1.

pipeline: qwen3_omni_moe_thinker_only
distributed_executor_backend: mp
enable_prefix_caching: false
trust_remote_code: true

stages:
  - stage_id: 0
    devices: "0"
    tensor_parallel_size: 1
    max_num_seqs: 128
    gpu_memory_utilization: 0.9
    enforce_eager: false
    async_scheduling: false
    moe_backend: aiter
    attention_backend: FLASH_ATTN_CLASSIC
    default_sampling_params:
      temperature: 0.4
      top_p: 0.9
      top_k: 1
      max_tokens: 2048
      seed: 42
      detokenize: true
      repetition_penalty: 1.05
```

## 关键配置说明

| 配置 | 作用 |
| --- | --- |
| `VLLM_HCU_USE_CUSTOM_OPS=1` | 启用 HCU 定制算子，替换通用实现。 |
| `VLLM_ROCM_USE_AITER=1` | 启用 AITER 算子库，加速注意力等计算。 |
| `VLLM_ROCM_USE_AITER_MOE=1` | 对 MoE 层启用 AITER 实现，Qwen3-Omni-30B-A3B 的专家计算量较大，该开关影响明显。 |
| `VLLM_HCU_USE_FLASH_ATTN=1` | 使用 FlashAttention 后端。 |
| `VLLM_HCU_USE_FLASH_ATTN_UNIFIED=1` | 使用统一 FlashAttention 路径。 |
| `VLLM_HCU_USE_AITER_MOE_CONFIG=1` | 启用 AITER MoE 的调优配置。 |
| `VLLM_HCU_USE_AITER_SAMPLER=1` | 启用 AITER sampler 路径；是否实际调用随机采样内核取决于请求采样参数及回退条件。 |
| `devices` / `tensor_parallel_size` | BW1000 使用 `"0,1"` 与 `2`；BW1100 使用 `"0"` 与 `1`，两者必须一致。 |
| `moe_backend: aiter` | MoE 层使用 AITER 后端，与 `VLLM_ROCM_USE_AITER_MOE` 配套。 |
| `attention_backend: FLASH_ATTN_CLASSIC` | 指定注意力后端为 FlashAttention classic。 |
| `gpu_memory_utilization: 0.9` | KV cache 与权重可使用的显存比例。 |
| `--allowed-local-media-path` | 允许服务端读取的本地媒体目录；不指定时请求中只能使用 URL 或 base64 data URL。 |

## 注意事项

- `devices` 中列出的卡数必须与 `tensor_parallel_size` 一致：BW1000 为 2 卡（`"0,1"` + `tensor_parallel_size: 2`），BW1100 为单卡（`"0"` + `tensor_parallel_size: 1`），混用会导致启动失败。
- 启动前确认所选物理卡空闲；不要覆盖其他服务正在使用的卡或端口。
- `default_sampling_params` 是部署默认采样参数，客户端请求可以显式覆盖这些参数。
