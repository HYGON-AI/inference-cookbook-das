# Qwen-Image 2.1 on SGLang Diffusion

BW1000 64 GB 单卡推荐使用 BF16、`torch_sdpa` 和 eager 执行。启动时显式关闭
`torch.compile` 和 Breakable CUDA Graph（BCG），将文本 Encoder、DiT 和 VAE 保留在卡上。

## 模型列表

| 模型权重 | 量化方式 | SGLang 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | ----------- | -------- | ---- | -------- | -------- |
| [Qwen/Qwen-Image-2.1](https://www.modelscope.cn/models/Qwen/Qwen-Image-2.1) | BF16 | 0.5.22 | BW1000 | 1 | Online | [**`>_`**](#qwen-image-21-online-bw1000-1x-sglang-0522) |

## 启动命令

### Qwen-Image 2.1 Online BW1000 1x SGLang 0.5.22

在已安装社区版 SGLang 0.5.22、`adapter-das` 和 `sglang-plugin-das` 的海光容器内运行。
插件使用 `release/sglang_v0.5.22`。以下命令使用 ModelScope 获取权重，容器需安装
`modelscope`。已有权重时，可把 `--model-path` 换成包含 `model_index.json` 的容器内目录。

```bash
export HIP_VISIBLE_DEVICES=0
export HCU_ADAPTER_HOST=sglang
export SGLANG_PLUGINS=hcu
export SGLANG_DIFFUSION_PLATFORM_OVERRIDE=cuda
export SGLANG_USE_MODELSCOPE=true

sglang serve \
  --model-type diffusion \
  --model-path Qwen/Qwen-Image-2.1 \
  --served-model-name Qwen/Qwen-Image-2.1 \
  --host 0.0.0.0 \
  --num-gpus 1 \
  --attention-backend torch_sdpa \
  --text-encoder-cpu-offload false \
  --dit-cpu-offload false \
  --dit-layerwise-offload false \
  --vae-cpu-offload false \
  --use-fsdp-inference false \
  --enable-torch-compile false \
  --enable-breakable-cuda-graph false
```

## API 调用

以下请求生成一张 1024×1024 的 PNG。响应中的 `data[0].b64_json` 包含图片数据。

```bash
curl -fsS http://localhost:30000/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen-Image-2.1",
    "prompt": "一只橘猫坐在窗边，午后阳光，写实摄影，细节清晰。",
    "size": "1024x1024",
    "n": 1,
    "num_inference_steps": 40,
    "guidance_scale": 1.0,
    "seed": 1000,
    "generator_device": "cuda",
    "quality": "exact",
    "enhance_prompt": false,
    "output_format": "png",
    "response_format": "b64_json"
  }' \
  | python3 -c 'import base64, json, sys; sys.stdout.buffer.write(base64.b64decode(json.load(sys.stdin)["data"][0]["b64_json"]))' \
  > qwen-image-2.1.png
```

## 性能参考

单张 BW1000，BF16、1024×1024、40 步、CFG=1，每请求一张图。预热后按下列顺序
串行测试两轮，每条提示词测 2 次，正式请求的 seed 依次为 1000–1009。
耗时从发出 HTTP 请求计到读完整个响应，预热不计入。

| 单卡模式 | 平均耗时 | P95 |
| -------- | -------- | --- |
| Eager | 10.702 s | 10.740 s |

### 固定提示词

1. 一只橘猫坐在窗边，午后阳光，写实摄影，细节清晰。
2. 雪山脚下的湖泊，松树林，清晨薄雾，电影级风景摄影。
3. 一张简约咖啡店海报，标题文字为“早安咖啡”，奶油色背景，一杯拿铁，精美平面设计。
4. A red sports car on a coastal road at sunset, realistic automotive photography, detailed reflections.
5. An isometric illustration of a small floating island with a wooden cottage, green trees and a waterfall.
