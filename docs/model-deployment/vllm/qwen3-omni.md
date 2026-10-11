# Qwen3-Omni on vLLM-Omni

## 模型简介

Qwen3-Omni-30B-A3B-Instruct 是阿里通义千问推出的全模态（Omni）模型，基于 MoE 架构，总参数 30B、激活参数 3B，可同时处理文本、图像、音频和视频输入并输出文本。

## 模型列表

| 模型权重 | 量化方式 | vLLM-Omni 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | -------------- | -------- | ---- | -------- | -------- |
| [Qwen/Qwen3-Omni-30B-A3B-Instruct](https://www.modelscope.cn/models/Qwen/Qwen3-Omni-30B-A3B-Instruct) | BF16 | 0.21 | BW1000 | 2 | IFB | [**`>_`**](#qwen3-omni-30b-a3b-instruct-ifb-bw1000-2x-vllm-021) |
|  | BF16 | 0.21 | BW1100 | 1 | IFB | [**`>_`**](#qwen3-omni-30b-a3b-instruct-ifb-bw1100-1x-vllm-021) |

## 部署准备

`--stage-configs-path` 接收 YAML 文件路径，请先将本文 [YAML 配置](#yaml-配置) 中对应硬件的内容保存为启动命令里使用的文件名：BW1000 使用 `qwen3_omni_moe_thinking_tp2.yaml`，BW1100 使用 `qwen3_omni_moe_thinking_tp1.yaml`。

启动前需要设置以下环境变量：

```bash
export VLLM_HCU_USE_CUSTOM_OPS="1"
export VLLM_ROCM_USE_AITER="1"
export VLLM_ROCM_USE_AITER_MOE="1"
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_FLASH_ATTN_UNIFIED=1
export VLLM_HCU_USE_AITER_MOE_CONFIG=1
```

## 启动 Server

### Qwen3-Omni-30B-A3B-Instruct IFB BW1000 2x vLLM 0.21

使用 `qwen3_omni_moe_thinking_tp2.yaml`，thinker stage 张量并行度为 2，占用设备 0、1：

```bash
export VLLM_HCU_USE_CUSTOM_OPS="1"
export VLLM_ROCM_USE_AITER="1"
export VLLM_ROCM_USE_AITER_MOE="1"
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_FLASH_ATTN_UNIFIED=1
export VLLM_HCU_USE_AITER_MOE_CONFIG=1

vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --omni \
  --host 0.0.0.0 \
  --port 8091 \
  --stage-configs-path ./qwen3_omni_moe_thinking_tp2.yaml
```

### Qwen3-Omni-30B-A3B-Instruct IFB BW1100 1x vLLM 0.21

使用 `qwen3_omni_moe_thinking_tp1.yaml`，thinker stage 张量并行度为 1，占用设备 0：

```bash
export VLLM_HCU_USE_CUSTOM_OPS="1"
export VLLM_ROCM_USE_AITER="1"
export VLLM_ROCM_USE_AITER_MOE="1"
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_FLASH_ATTN_UNIFIED=1
export VLLM_HCU_USE_AITER_MOE_CONFIG=1

vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --omni \
  --host 0.0.0.0 \
  --port 8091 \
  --stage-configs-path ./qwen3_omni_moe_thinking_tp1.yaml
```

服务启动完成后可检查：

```bash
curl http://127.0.0.1:8091/health
```

## API 调用

### 音频理解

```python
import base64
from openai import OpenAI

with open("audio.wav", "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode()

client = OpenAI(base_url="http://localhost:8091/v1", api_key="not-needed")

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
curl http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    "messages": [{"role": "user", "content": "你好，请用一句话介绍你自己。"}],
    "max_tokens": 128
  }'
```

## YAML 配置

两份配置结构一致，只有 `runtime.devices` 和 `tensor_parallel_size` 随硬件不同而变化，用于匹配 BW1000 与 BW1100 的单机容量。

### qwen3_omni_moe_thinking_tp2.yaml

用于 BW1000，张量并行度为 2：

```yaml
# Stage config for Qwen3-Omni-30B-A3B-Instruct audio-to-text output.
# This is a single thinker stage config with tensor parallel size 2.

stage_args:
  - stage_id: 0
    runtime:
      devices: "0,1"
    engine_args:
      model_stage: thinker
      max_num_seqs: 128
      model_arch: Qwen3OmniMoeForConditionalGeneration
      worker_type: ar
      scheduler_cls: vllm_omni.core.sched.omni_ar_scheduler.OmniARScheduler
      gpu_memory_utilization: 0.9
      enforce_eager: false
      trust_remote_code: true
      engine_output_type: text
      distributed_executor_backend: "mp"
      enable_prefix_caching: false
      hf_config_name: thinker_config
      tensor_parallel_size: 2
      compilation_config:
        pass_config:
          fuse_allreduce_rms: false
      attention_backend: FLASH_ATTN
      moe_backend: aiter
    final_output: true
    final_output_type: text
    is_comprehension: true
    default_sampling_params:
      temperature: 0.4
      top_p: 0.9
      top_k: 1
      max_tokens: 2048
      seed: 42
      detokenize: True
      repetition_penalty: 1.05
```

### qwen3_omni_moe_thinking_tp1.yaml

用于 BW1100，张量并行度为 1：

```yaml
# Stage config for Qwen3-Omni-30B-A3B-Instruct audio-to-text output.
# This is a single thinker stage config with tensor parallel size 1.

stage_args:
  - stage_id: 0
    runtime:
      devices: "0"
    engine_args:
      model_stage: thinker
      max_num_seqs: 128
      model_arch: Qwen3OmniMoeForConditionalGeneration
      worker_type: ar
      scheduler_cls: vllm_omni.core.sched.omni_ar_scheduler.OmniARScheduler
      gpu_memory_utilization: 0.9
      enforce_eager: false
      trust_remote_code: true
      engine_output_type: text
      distributed_executor_backend: "mp"
      enable_prefix_caching: false
      hf_config_name: thinker_config
      tensor_parallel_size: 1
      attention_backend: FLASH_ATTN
      moe_backend: aiter
    final_output: true
    final_output_type: text
    is_comprehension: true
    default_sampling_params:
      temperature: 0.4
      top_p: 0.9
      top_k: 1
      max_tokens: 2048
      seed: 42
      detokenize: True
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
| `model_stage: thinker` | 本配置只启动 thinker stage，输出类型为文本。 |
| `devices` / `tensor_parallel_size` | BW1000 使用 `"0,1"` 与 `2`；BW1100 使用 `"0"` 与 `1`，两者必须一致。 |
| `moe_backend: aiter` | MoE 层使用 AITER 后端，与 `VLLM_ROCM_USE_AITER_MOE` 配套。 |
| `attention_backend: FLASH_ATTN` | 指定注意力后端为 FlashAttention。 |
| `gpu_memory_utilization: 0.9` | KV cache 与权重可使用的显存比例。 |
| `--allowed-local-media-path` | 允许服务端读取的本地媒体目录；不指定时请求中只能使用 URL 或 base64 data URL。 |

## 注意事项

- `devices` 中列出的卡数必须与 `tensor_parallel_size` 一致：BW1000 为 2 卡（`"0,1"` + `tensor_parallel_size: 2`），BW1100 为单卡（`"0"` + `tensor_parallel_size: 1`），混用会导致启动失败。