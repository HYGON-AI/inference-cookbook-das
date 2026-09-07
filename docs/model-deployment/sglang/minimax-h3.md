# MiniMax-H3 on SGLang Diffusion

本文汇总 MiniMax-H3 在 BW1000 和 BW1100 上的 8 卡推荐部署策略。默认运行环境已包含 SGLang 0.5.18，本文不展开镜像安装和性能数据，只说明模型分区、8 卡并行方式、CacheDiT 使用边界和验证口径。

本文下方按 BW1000 和 BW1100 分别给出可复制的 8 卡启动、请求和验证模板。

## 模型列表

| 模型权重 | 量化方式 | SGLang 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| --- | --- | --- | --- | --- | --- | --- |
| [MiniMax/MiniMax-H3](https://www.modelscope.cn/models/MiniMax/MiniMax-H3) | BF16 | 0.5.18 | BW1000 | 8 | Online | [启动命令](#启动-server) |
| [MiniMax/MiniMax-H3](https://www.modelscope.cn/models/MiniMax/MiniMax-H3) | BF16 | 0.5.18 | BW1100 | 8 | Online | [启动命令](#启动-server) |


## 模型与场景

| 场景 | `model-variant` | 输入 | 输出 |
| --- | --- | --- | --- |
| T2VA | `fl2va` | 文本 | 视频 + 音频 |
| FL2VA | `fl2va` | 文本 + 首尾帧/关键帧 | 视频 + 音频 |
| Ref2VA | `ref2va` | 文本 + 参考图像、音频或视频 | 视频 + 音频 |

T2VA 与 FL2VA 可以共用 `fl2va` Server；切换到 Ref2VA 时需要以 `ref2va` 重新启动 Server。

## 8 卡推荐策略

总表只给一个稳定的 8 卡主配置。CacheDiT 是否开启属于请求/服务开关，不改变这里的推荐拓扑；需要专项调优时再查看对应硬件的详细文档。

| 硬件 | 推荐 8 卡布局 | 覆盖场景 | Offload 建议 | 说明 |
| --- | --- | --- | --- | --- |
| BW1000 | TP2 + SP4 | T2VA、FL2VA、Ref2VA | 默认全关；Ref2VA 复杂参考素材 OOM 时优先开启 Text Encoder offload | 64 GiB 卡的通用 8 卡部署入口 |
| BW1100 | SP8 | T2VA、FL2VA、Ref2VA | 默认全关 | 144 GiB 卡显存余量更大，先按纯序列并行作为统一 8 卡入口 |

## 启动前环境变量

以下环境变量用于固定 MiniMax-H3 的 HCU 性能路径。启动命令直接使用官方模型名 `MiniMax/MiniMax-H3` 和 SGLang 默认服务端口 `30000`。

```bash
export OMP_NUM_THREADS=32
export AllTOAll_STREAM_WITH_COMPUTE=1
export MINIMAX_H3_TORCH_SDPA_BACKEND=flash
export MINIMAX_H3_VAE_DECODER_STREAM_TEMPORAL_CAT=1
```

## 启动 Server

T2VA 和 FL2VA 使用 `fl2va` 分区；Ref2VA 需要单独启动 `ref2va` 分区。MiniMax-H3 建议显式指定组件级 attention backend：`transformer=fa` 给 DiT 主干，`text_encoder=torch_sdpa` 给文本/视觉编码器，避免不同组件都继承同一个默认 backend。

### BW1000：T2VA / FL2VA，8 卡 TP2 + SP4

```bash
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sglang serve \
  --model-type diffusion \
  --model-path MiniMax/MiniMax-H3 \
  --model-variant fl2va \
  --num-gpus 8 \
  --tp-size 2 \
  --sp-degree 4 \
  --ulysses-degree 4 \
  --ring-degree 1 \
  --encoder-parallel auto \
  --attention-backend fa \
  --component-attention-backends text_encoder=torch_sdpa,transformer=fa \
  --performance-mode manual \
  --dit-cpu-offload false \
  --dit-layerwise-offload false \
  --text-encoder-cpu-offload false \
  --image-encoder-cpu-offload false \
  --vae-cpu-offload false \
  --pin-cpu-memory false \
  --use-fsdp-inference false \
  --trust-remote-code \
  --warmup-mode server \
  --host 0.0.0.0 \
  --output-path ./outputs/minimax-h3
```

### BW1000：Ref2VA，8 卡 TP2 + SP4

```bash
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sglang serve \
  --model-type diffusion \
  --model-path MiniMax/MiniMax-H3 \
  --model-variant ref2va \
  --num-gpus 8 \
  --tp-size 2 \
  --sp-degree 4 \
  --ulysses-degree 4 \
  --ring-degree 1 \
  --encoder-parallel auto \
  --attention-backend fa \
  --component-attention-backends text_encoder=torch_sdpa,transformer=fa \
  --performance-mode manual \
  --dit-cpu-offload false \
  --dit-layerwise-offload false \
  --text-encoder-cpu-offload false \
  --image-encoder-cpu-offload false \
  --vae-cpu-offload false \
  --pin-cpu-memory false \
  --use-fsdp-inference false \
  --trust-remote-code \
  --warmup-mode server \
  --host 0.0.0.0 \
  --output-path ./outputs/minimax-h3
```

### BW1100：T2VA / FL2VA，8 卡 SP8

```bash
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sglang serve \
  --model-type diffusion \
  --model-path MiniMax/MiniMax-H3 \
  --model-variant fl2va \
  --num-gpus 8 \
  --tp-size 1 \
  --sp-degree 8 \
  --ulysses-degree 8 \
  --ring-degree 1 \
  --encoder-parallel auto \
  --attention-backend fa \
  --component-attention-backends text_encoder=torch_sdpa,transformer=fa \
  --performance-mode manual \
  --dit-cpu-offload false \
  --dit-layerwise-offload false \
  --text-encoder-cpu-offload false \
  --image-encoder-cpu-offload false \
  --vae-cpu-offload false \
  --pin-cpu-memory false \
  --use-fsdp-inference false \
  --trust-remote-code \
  --warmup-mode server \
  --host 0.0.0.0 \
  --output-path ./outputs/minimax-h3
```

### BW1100：Ref2VA，8 卡 SP8

```bash
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sglang serve \
  --model-type diffusion \
  --model-path MiniMax/MiniMax-H3 \
  --model-variant ref2va \
  --num-gpus 8 \
  --tp-size 1 \
  --sp-degree 8 \
  --ulysses-degree 8 \
  --ring-degree 1 \
  --encoder-parallel auto \
  --attention-backend fa \
  --component-attention-backends text_encoder=torch_sdpa,transformer=fa \
  --performance-mode manual \
  --dit-cpu-offload false \
  --dit-layerwise-offload false \
  --text-encoder-cpu-offload false \
  --image-encoder-cpu-offload false \
  --vae-cpu-offload false \
  --pin-cpu-memory false \
  --use-fsdp-inference false \
  --trust-remote-code \
  --warmup-mode server \
  --host 0.0.0.0 \
  --output-path ./outputs/minimax-h3
```

默认关闭组件 offload。BW1000 的 Ref2VA 复杂参考素材或更长视频 OOM 时，可以优先把对应启动命令中的 `--text-encoder-cpu-offload false` 改成 `true`，其他选项保持不变。

Server 是前台常驻进程。服务启动完成后，在另一个 Shell 中验活：

```bash
curl -sS http://127.0.0.1:30000/health
curl -sS http://127.0.0.1:30000/v1/models | python3 -m json.tool
```

## Warmup

Warmup 不能省略。严格验收时可以把启动命令中的 `--warmup-mode server` 改为 `--warmup-mode off`，手动执行 3 次 2-step 请求，然后再发送正式请求。

```bash
curl -sS -X POST http://127.0.0.1:30000/v1/videos \
  -H "Content-Type: application/json" \
  -d '{
    "task": "t2va",
    "prompt": "integrated_multimodal_description: A cat sitting on a windowsill watching snow fall outside.\noverall_soundscape: Quiet indoor ambience.\nnon_diegetic_music: N/A",
    "conditions": [],
    "target": {
      "short_edge": 768,
      "aspect_ratio": "16:9",
      "duration_seconds": 5
    },
    "seed": 1101,
    "n": 1,
    "num_inference_steps": 2,
    "flow_shift": 12,
    "audio_flow_shift": 3
  }'
```

## 三种场景请求

### T2VA

T2VA 使用 `model-variant=fl2va`。

```bash
curl -sS -X POST http://127.0.0.1:30000/v1/videos \
  -H "Content-Type: application/json" \
  -o create.json \
  -d '{
    "task": "t2va",
    "prompt": "integrated_multimodal_description: A cat sitting on a windowsill watching snow fall outside, soft indoor lighting, gentle ambient room tone.\noverall_soundscape: Quiet indoor ambience with soft snowfall.\nnon_diegetic_music: N/A",
    "conditions": [],
    "target": {
      "short_edge": 768,
      "aspect_ratio": "16:9",
      "duration_seconds": 5
    },
    "seed": 1101,
    "n": 1,
    "num_inference_steps": 50,
    "flow_shift": 12,
    "audio_flow_shift": 3
  }'
```

### FL2VA

FL2VA 使用 `model-variant=fl2va`。首帧使用 `frame_index=0`，尾帧使用 `frame_index=-1`；示例中的 `/inputs/fl2va_first_frame.png` 是客户或读者自备图片，替换成 Server 容器内可访问的实际路径即可。

```bash
curl -sS -X POST http://127.0.0.1:30000/v1/videos \
  -H "Content-Type: application/json" \
  -o create.json \
  -d '{
    "task": "fl2va",
    "prompt": "At 0.00 seconds, <Picture 1> is fully referenced. Generate a smooth cinematic camera orbit while preserving the subject and scene.",
    "conditions": [
      {
        "type": "image",
        "uri": "/inputs/fl2va_first_frame.png",
        "role": "keyframe",
        "frame_index": 0
      }
    ],
    "target": {
      "short_edge": 768,
      "aspect_ratio": "auto",
      "duration_seconds": 5
    },
    "seed": 2101,
    "n": 1,
    "num_inference_steps": 50,
    "flow_shift": 12,
    "audio_flow_shift": 3
  }'
```

### Ref2VA

Ref2VA 使用 `model-variant=ref2va`，需要单独启动 `ref2va` Server。示例中的 `/inputs/ref2va_image.png` 和 `/inputs/ref2va_audio.mp3` 是客户或读者自备参考素材，替换成 Server 容器内可访问的实际路径即可。

```bash
curl -sS -X POST http://127.0.0.1:30000/v1/videos \
  -H "Content-Type: application/json" \
  -o create.json \
  -d '{
    "task": "ref2va",
    "prompt": "<Subject 1> is the subject in <Picture 1>. <Audio 1> is the voice reference. Create a realistic video with precise lip sync while preserving the subject appearance.",
    "conditions": [
      {
        "type": "image",
        "uri": "/inputs/ref2va_image.png",
        "role": "reference"
      },
      {
        "type": "audio",
        "uri": "/inputs/ref2va_audio.mp3",
        "role": "reference"
      }
    ],
    "target": {
      "short_edge": 768,
      "aspect_ratio": "auto",
      "duration_seconds": 5
    },
    "seed": 3101,
    "n": 1,
    "num_inference_steps": 50,
    "flow_shift": 12,
    "audio_flow_shift": 3
  }'
```

## 查询和下载结果

接口会先返回 `queued` 和任务 ID。可以从 `create.json` 取出 `VIDEO_ID`，再查询状态或下载文件。

```bash
export VIDEO_ID=$(python3 - <<'PY'
import json
print(json.load(open("create.json"))["id"])
PY
)

curl -sS "http://127.0.0.1:30000/v1/videos/${VIDEO_ID}" | python3 -m json.tool
curl -sS -L "http://127.0.0.1:30000/v1/videos/${VIDEO_ID}/content" -o output.mp4
```

如果只看 Server 落盘结果，日志出现 `Output saved to ...mp4` 和 `Pixel data generated successfully` 后，视频也会保存在 `./outputs/minimax-h3` 下。

## 可选：开启 CacheDiT

CacheDiT 是 denoising 阶段的近似复用加速，不改变推荐并行拓扑。默认不开启；如果要开，需要在启动 Server 前设置：

```bash
export SGLANG_CACHE_DIT_ENABLED=true
export SGLANG_CACHE_DIT_FN=1
export SGLANG_CACHE_DIT_BN=0
export SGLANG_CACHE_DIT_WARMUP=4
export SGLANG_CACHE_DIT_RDT=0.24
export SGLANG_CACHE_DIT_MC=3
```

如果明确要关掉 CacheDiT，启动 Server 前执行：

```bash
export SGLANG_CACHE_DIT_ENABLED=false
```

## 可选：生成并启用 AdaLN cache

AdaLN cache 是给固定 T2VA 配置用的精确预计算 sidecar，不是 CacheDiT。它把固定 step/timestep plan 下的 AdaLN 结果提前算好，启动时用 sidecar 代替一部分 AdaLN 权重常驻显存，主要用于降低显存压力。

先准备模型权重，再生成 sidecar：

```bash
HIP_VISIBLE_DEVICES=0 python3 -m sglang.multimodal_gen.tools.build_minimax_h3_adaln_cache \
  --transformer-path /path/to/MiniMax-H3/FL2VA/transformer \
  --model-variant fl2va \
  --mode t2va \
  --num-inference-steps 50 \
  --flow-shift 12 \
  --audio-flow-shift 3 \
  --output /path/minimax_h3_t2va_50step_adaln.safetensors

ls -lh /path/minimax_h3_t2va_50step_adaln.safetensors
```

只有在本次请求仍是相同权重、T2VA、50 steps、`flow_shift=12`、`audio_flow_shift=3` 时，才可以在启动 Server 时额外加这一行：

```bash
  --minimax-h3-adaln-cache-path /path/minimax_h3_t2va_50step_adaln.safetensors \
```

例如把它放在 `--output-path ./outputs/minimax-h3` 前面；注意前一行要保留反斜杠。默认不启用 AdaLN cache 时，不需要设置 `MINIMAX_H3_ADALN_CACHE_PATH`，也不需要加这行参数。

## 日志检查

启动和请求完成后，建议确认日志中包含以下关键行：

```text
Using HCU FlashAttention-2 backend on HCU.
Using fa attention backend
Pipeline instantiated
[MiniMaxH3TextEncodingStage] finished in ... seconds
[MiniMaxH3DenoisingStage] finished in ... seconds
[MiniMaxH3DecodingStage] finished in ... seconds
Output saved to ...mp4
Pixel data generated successfully in ... seconds
```

同时确认启动参数中包含 `--component-attention-backends text_encoder=torch_sdpa,transformer=fa`：文本/视觉编码器保持 torch SDPA 路径，DiT transformer 使用 FA 路径。

开启 CacheDiT 后还要确认命中数大于 0，并确认 `residual_diffs` 没有 `NaN`。
