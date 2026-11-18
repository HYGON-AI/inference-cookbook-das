# Diffusion 模型部署 — 总览

## 框架对比

| 框架 | 特点 | 适用场景 | HCU 适配 |
|------|------|---------|---------|
| Diffusers | HuggingFace 生态，代码驱动 | 标准文生图/视频、API 服务 | ✅ 良好 |
| ComfyUI | 节点式可视化工作流 | 复杂 pipeline、灵活组合 | ✅ 良好 |

## 支持的模型

### 图像生成

| 模型 | 文档 |
|------|------|
| Stable Diffusion 1.5 | [sd1.5.md](sd1.5.md) |
| Stable Diffusion 3 | [sd3-flux.md](sd3-flux.md) |
| FLUX.1 | [sd3-flux.md](sd3-flux.md) |
| SDXL | [sdxl.md](sdxl.md) |
| ControlNet | [sd3-flux.md](sd3-flux.md) |

### 视频生成

| 模型 | 文档 |
|------|------|
| Wan2.1 | [wan2.1.md](wan2.1.md) |
| CogVideoX | [cogvideox.md](cogvideox.md) |
| MAGI-2-preview | [MAGI-2-preview.md](magi-2-preview/MAGI-2-preview.md) |

### 可视化部署

| 工具 | 文档 |
|------|------|
| ComfyUI | [comfyui-hcu.md](comfyui-hcu.md) |

## 环境准备

```bash
pip install diffusers transformers accelerate imageio[ffmpeg]
```
备注：MAGI-2-preview 使用专用镜像和源码补丁，请按其部署文档安装。

## 显存参考

| 模型 | 分辨率 | 最低显存 | 推荐 |
|------|--------|---------|------|
| SDXL | 1024x1024 | 12GB | 16GB+ |
| SD3-Medium | 1024x1024 | 16GB | 24GB+ |
| FLUX.1-dev | 1024x1024 | 24GB | 32GB+ |
| Wan2.1-1.3B | 832x480 | 8GB | 16GB+ |
| Wan2.1-14B | 832x480 | 32GB | 48GB+ |
| CogVideoX-5B | 720x480 | 32GB | 48GB+ |
| MAGI-2-preview | 1920x1088 | 64GB*8 | 64GB*8+ |
