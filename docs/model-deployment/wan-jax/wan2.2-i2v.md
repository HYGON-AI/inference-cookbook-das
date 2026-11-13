# Wan2.2-I2V on wan-jax

## 模型简介

Wan2.2-I2V-A14B 是阿里通义实验室推出的图生视频（Image-to-Video）模型，基于 DiT 双专家 MoE 架构，能够以输入首帧图像与提示词为引导，合成极具真实感与物理一致性的高动态视频。

本文档给出基于 **JAX / MaxDiffusion** 框架在 BW1000/BW1100 8 卡上的离线推理最佳实践，适用于 1280x720、1280x704 等高分辨率场景。

---

## 模型列表

| 模型权重 | 权重格式 | 推荐硬件 | 卡数 | 部署方式 | 启动链接 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| [Wan-AI/Wan2.2-I2V-A14B](https://www.modelscope.cn/models/Wan-AI/Wan2.2-I2V-A14B) | BF16 (Diffusers / JAX Array) | BW1000/BW1100 | 8x | Offline | [启动命令](#wan22-i2v-a14b-bf16) |

---

## 镜像与环境准备

推荐直接使用官方构建的离线部署镜像 `wan2.2_infer_jax:v1.0`：

```bash
# 1. 载入镜像（若使用归档包）
docker load -i /path/to/images/wan2.2_infer_jax_v1.0.tar.gz

# 2. 启动容器（映射 8 张加速卡、共享内存与权重目录）
docker run -it -d --name wan_jax_infer \
  --device=/dev/kfd --device=/dev/dri \
  --shm-size=64G \
  --security-opt seccomp=unconfined \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v /path/to/weights:/opt/weights:ro \
  wan2.2_infer_jax:v1.0 /bin/bash
```

进入容器并加载环境变量：
```bash
docker exec -it wan_jax_infer /bin/bash
source /etc/profile.d/wan_env.sh
```

---

## 代码获取与路径

代码已预置在镜像内 `/home/code/code_pr/maxdiffusion`，如需拉取最新优化分支：

```bash
git clone https://developer.sourcefind.cn/codes/OpenDAS/maxdiffusion.git
cd maxdiffusion
```

---

## 启动命令

### Wan2.2-I2V-A14B BF16

```bash
source /etc/profile.d/wan_env.sh

REPO=/home/code/code_pr/maxdiffusion
MODEL_DIR=/opt/weights/wan22-i2v-a14b-diffusers
OUTPUT_DIR=/tmp/wan22_i2v_run

mkdir -p ${OUTPUT_DIR}
cd ${OUTPUT_DIR}

python ${REPO}/src/maxdiffusion/generate_wan.py \
  ${REPO}/src/maxdiffusion/configs/base_wan_i2v_27b.yml \
  pretrained_model_name_or_path="${MODEL_DIR}" \
  hardware=gpu \
  attention=hipc_ulysses \
  height=720 width=1280 num_frames=61 num_inference_steps=20 \
  flow_shift=5.0 boundary_ratio=0.9 \
  guidance_scale_high=3.5 guidance_scale_low=3.5 \
  seed=0 per_device_batch_size=0.125 \
  ici_data_parallelism=1 ici_fsdp_parallelism=1 ici_context_parallelism=8 ici_tensor_parallelism=1 \
  dcn_data_parallelism=1 dcn_fsdp_parallelism=1 dcn_context_parallelism=1 dcn_tensor_parallelism=1 \
  prompt="Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard." \
  image_url="${MODEL_DIR}/examples/i2v_input.JPG" \
  output_dir="${OUTPUT_DIR}" \
  run_name="wan22_i2v_720p_run" \
  use_magcache=False use_kv_cache=False use_sen_cache=False \
  enable_lora=False
```
