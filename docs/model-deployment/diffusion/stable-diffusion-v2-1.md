# stable-diffusion-v2-1 on Diffusers

本文提供 `stable-diffusion-v2-1` 在 BW1000 和 BW1100 上基于 Diffusers 的单卡文生图和图生图离线推理示例。

## 模型列表

| 模型权重 | Diffusers 版本 | 推荐硬件 | 卡数 | 部署方式 | 场景 | 启动命令 |
| --- | --- | --- | --- | --- | --- | --- |
| [stabilityai/stable-diffusion-2-1](https://modelscope.cn/models/stabilityai/stable-diffusion-2-1) | `0.37.0` | BW1100 | 1 | Offline | 文生图 | [**`>_`**](#stable-diffusion-v2-1-t2i-offline-bw1000-and-bw1100-1x-diffusers-0370) |
|  | `0.37.0` | BW1000 | 1 | Offline | 文生图 | [**`>_`**](#stable-diffusion-v2-1-t2i-offline-bw1000-and-bw1100-1x-diffusers-0370) |
|  | `0.37.0` | BW1100 | 1 | Offline | 图生图 | [**`>_`**](#stable-diffusion-v2-1-img2img-offline-bw1000-and-bw1100-1x-diffusers-0370) |
|  | `0.37.0` | BW1000 | 1 | Offline | 图生图 | [**`>_`**](#stable-diffusion-v2-1-img2img-offline-bw1000-and-bw1100-1x-diffusers-0370) |

## 模型与场景

| 场景 | Pipeline | 输入 | 输出 |
| --- | --- | --- | --- |
| 文生图 | `StableDiffusionPipeline` | 文本 Prompt | PNG 图片 |
| 图生图 | `StableDiffusionImg2ImgPipeline` | 文本 Prompt + 输入图片 | PNG 图片 |

## 启动命令

### stable-diffusion-v2-1 T2I Offline BW1000 and BW1100 1x Diffusers 0.37.0

创建 `text_to_image.py`：

```python
import argparse

import torch
from diffusers import StableDiffusionPipeline


parser = argparse.ArgumentParser()
parser.add_argument("--model-path", required=True)
parser.add_argument("--output", default="sd21-t2i.png")
args = parser.parse_args()

pipe = StableDiffusionPipeline.from_pretrained(
    args.model_path,
    torch_dtype=torch.float32,
    local_files_only=True,
    safety_checker=None,
).to("cuda")
pipe.unet = torch.compile(
    pipe.unet,
    backend="inductor",
    mode="max-autotune-no-cudagraphs",
    fullgraph=False,
)

generator = torch.Generator(device="cuda").manual_seed(42)
image = pipe(
    prompt="a cute cat sitting on the sofa",
    negative_prompt="blurry, low quality",
    num_inference_steps=20,
    guidance_scale=7.0,
    width=768,
    height=768,
    generator=generator,
).images[0]
image.save(args.output)
```

执行文生图推理：

```bash
python3 text_to_image.py \
  --model-path /path/to/stable-diffusion-2-1
```

### stable-diffusion-v2-1 Img2Img Offline BW1000 and BW1100 1x Diffusers 0.37.0

图生图的输出尺寸由输入图片尺寸决定。下面的脚本会先将输入图片调整为 768×768。

创建 `image_to_image.py`：

```python
import argparse

import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument("--model-path", required=True)
parser.add_argument("--input-image", required=True)
parser.add_argument("--output", default="sd21-img2img.png")
args = parser.parse_args()

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    args.model_path,
    torch_dtype=torch.float32,
    local_files_only=True,
    safety_checker=None,
).to("cuda")
pipe.unet = torch.compile(
    pipe.unet,
    backend="inductor",
    mode="max-autotune-no-cudagraphs",
    fullgraph=False,
)

with Image.open(args.input_image) as source:
    input_image = source.convert("RGB").resize((768, 768), Image.Resampling.LANCZOS)

generator = torch.Generator(device="cuda").manual_seed(42)
image = pipe(
    prompt="a watercolor painting of a cat sitting on a sofa",
    negative_prompt="blurry, low quality",
    image=input_image,
    strength=0.75,
    num_inference_steps=20,
    guidance_scale=7.0,
    generator=generator,
).images[0]
image.save(args.output)
```

执行图生图推理：

```bash
python3 image_to_image.py \
  --model-path /path/to/stable-diffusion-2-1 \
  --input-image /path/to/input.png
```

## 结果检查

可以使用 Pillow 检查图片是否能完整解码，并确认实际尺寸：

```bash
python3 - <<'PY'
from PIL import Image

for path in ("sd21-t2i.png", "sd21-img2img.png"):
    with Image.open(path) as image:
        image.load()
        print(path, image.format, image.mode, image.size)
PY
```

## 使用说明

- 两个离线推理脚本默认开启 `torch.compile`；首次推理会触发图编译。
- 相同 Prompt、Seed、推理参数和运行环境用于复现生成结果。
- 较大分辨率会增加推理时间和显存使用量。
