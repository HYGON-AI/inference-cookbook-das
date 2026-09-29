# Stable Diffusion 1.5 on HCU

## 模型简介和适用任务

[Stable Diffusion 1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) 是基于 Latent Diffusion 的图像生成模型，适用于文生图（text-to-image）和图生图（image-to-image）任务

## Diffusers 离线推理部署

### 文生图

使用 `StableDiffusionPipeline` 加载 SD1.5 权重：

```python
import torch
from diffusers import StableDiffusionPipeline

model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
).to("cuda")

pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

with torch.inference_mode():
    image = pipe(
        prompt="A kitten running on the grass",
        num_inference_steps=50,
        guidance_scale=7.5,
        width=1024,
        height=1024,
    ).images[0]

image.save("sd15_txt2img.png")
```

### 图生图

使用 `StableDiffusionImg2ImgPipeline` 传入初始图片和文本指令：

```python
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers.utils import load_image

model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
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
        guidance_scale=7.5,
    ).images[0]

image.save("sd15_img2img.png")
```

## 性能优化

```python
# torch.compile
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

# 降低推理步数
image = pipe(prompt, num_inference_steps=20)  # 默认 50
```