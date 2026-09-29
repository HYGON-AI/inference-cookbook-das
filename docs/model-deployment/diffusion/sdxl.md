# Stable Diffusion XL on HCU

## 模型简介和适用任务

[Stable Diffusion XL Base 1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) 是 Stable Diffusion 系列的高分辨率图像生成模型，适用于文生图和图生图任务。

## Diffusers 离线推理部署

使用 `StableDiffusionXLImg2ImgPipeline` 加载模型权重

### 图生图

```python
import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image

model_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
).to("cuda")

pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

init_image = load_image("input.png").convert("RGB").resize((1024, 1024))
with torch.inference_mode():
    image = pipe(
        prompt="A fantasy landscape, trending on artstation",
        image=init_image,
        strength=1.0,
        num_inference_steps=50,
        guidance_scale=7,
        negative_prompt="blurry, low quality",
    ).images[0]

image.save("sdxl_img2img.png")
```

## 性能优化

```python
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
```