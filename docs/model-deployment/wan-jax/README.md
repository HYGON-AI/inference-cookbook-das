# Wan on wan-jax 部署指南

本目录包含Wan2.2系列视频生成大模型在 **JAX / MaxDiffusion** 分布式推理框架与 BW1000/BW1100 硬件上的端到端最佳实践。

---

## 概述与核心特性

`wan-jax` 依托 Google JAX 的 `@jax.jit` 全图编译能力与 MaxDiffusion 分布式并行架构，结合专属通信与算子深度优化，具备以下特性：

1. **端到端纯图编译**：全流程 XLA 静态计算图，消除运行时调度抖动与 Python 循环开销。
2. **极速 VAE 编解码**：依托 8 卡空间切片并行与 3D 卷积自动融合，大幅压缩 VAE 图像编码与视频解码耗时。
3. **双专家物理 Scan 拆分**：针对 14B 双专家架构消除动态分支与显存重复驻留。
4. **统一 Ulysses 序列并行**：规整通信 Chunk 为 1，降低 RCCL 通信启动开销。

---

## 支持模型导航

| 模型架构 | 任务类型 | 推荐硬件 | 卡数 | 部署文档链接 |
| :--- | :---: | :---: | :---: | :---: |
| **Wan2.2-TI2V-5B** | 统一图生/文生视频 | BW1000/BW1100 | 8x | [wan2.2-ti2v.md](./wan2.2-ti2v.md) |
| **Wan2.2-T2V-A14B** | 双专家文生视频 | BW1000/BW1100 | 8x | [wan2.2-t2v.md](./wan2.2-t2v.md) |
| **Wan2.2-I2V-A14B** | 双专家图生视频 | BW1000/BW1100 | 8x | [wan2.2-i2v.md](./wan2.2-i2v.md) |

---

## 快速上手与通用环境配置

### 1. 运行环境镜像
* 官方标准镜像：`wan2.2_infer_jax:v1.0`

### 2. 容器启动示例
```bash
docker run -it -d --name wan_jax_infer \
  --device=/dev/kfd --device=/dev/dri \
  --shm-size=64G \
  --security-opt seccomp=unconfined \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v /opt/weights:/opt/weights:ro \
  wan2.2_infer_jax:v1.0 /bin/bash
```

进入容器后，加载 JAX 专属环境即可直接运行各大模型脚本：
```bash
source /etc/profile.d/wan_env.sh
```
