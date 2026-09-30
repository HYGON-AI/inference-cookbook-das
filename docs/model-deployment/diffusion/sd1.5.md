# Stable Diffusion 1.5 on HCU

## 模型简介和适用任务

[Stable Diffusion 1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) 是基于 Latent Diffusion 的图像生成模型，适用于文生图（text-to-image）和图生图（image-to-image）任务。

该模型的原生训练分辨率为 512×512，建议生成尺寸保持在 512×512（上限不超过 768×768）。超出原生分辨率生成会出现主体重复、构图畸变等画质退化，需要更高分辨率时应先生成 512×512 再做超分。

## Diffusers 离线推理部署

### 文生图

使用 `StableDiffusionPipeline` 加载 SD1.5 权重：

```python
import torch
from diffusers import StableDiffusionPipeline

model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
).to("cuda")

pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

with torch.inference_mode():
    image = pipe(
        prompt="A kitten running on the grass",
        num_inference_steps=50,
        guidance_scale=7.5,
        width=512,
        height=512,
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
    torch_dtype=torch.float16,
).to("cuda")

pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

init_image = load_image("input.png").convert("RGB").resize((512, 512))
with torch.inference_mode():
    image = pipe(
        prompt="A fantasy landscape, trending on artstation",
        image=init_image,
        strength=0.75,
        num_inference_steps=50,
        guidance_scale=7.5,
    ).images[0]

image.save("sd15_img2img.png")
```

`strength` 决定对初始图加噪并去噪的比例，取值为 `0.0`–`1.0`。值越大改动越剧烈：`1.0` 会把初始图加噪到接近纯噪声，输出由 prompt 主导、基本不保留初始图内容；常规图生图建议 `0.4`–`0.8`。该值也会影响实际去噪步数，`num_inference_steps` 与 `strength` 的乘积即为实际执行的步数。

## 性能优化

```python
# torch.compile
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

# 降低推理步数
image = pipe(prompt, num_inference_steps=20)  # 默认 50
```