# MAGI-2-preview 在 BW1000 上部署

本文介绍了MAGI-2-preview在bw1000上的最优实践，在容器启动后安装模型依赖，从官方源码构建 MagiCompiler和经过HCU适配的 MagiAttention，再为官方 MAGI-2-preview应用模型优化补丁。


## 1. 拉取镜像

在宿主机执行:

```bash
docker pull harbor.sourcefind.cn:5443/hcu/admin/base/custom:dit-vla-wm-base-ubuntu22.04-dtk26.04-torch2.10-py3.10-20260929
```


## 2. 准备目录并启动容器

提前从 [sand-ai/MAGI-2-preview](https://huggingface.co/sand-ai/MAGI-2-preview) 下载完整权重到宿主机本地盘，本文以 `/data/models/Magi-2` 为例。权重约 307 GB，应预留足够磁盘空间。

通过以下命令启动容器：

```bash
docker run -itd \
  --device=/dev/kfd \
  --device=/dev/dri \
  --network=host \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --group-add video \
  --privileged=true \
  --shm-size=16G \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v /data/models:/home/models:ro \
  --name magi2_hcu \
  harbor.sourcefind.cn:5443/hcu/admin/base/custom:dit-vla-wm-base-ubuntu22.04-dtk26.04-torch2.10-py3.10-20260929

docker exec -it magi2_hcu bash
```

## 3. 获取材料并安装模型依赖

以下命令在容器内执行，需使用已包含本部署材料的 cookbook 版本。

```bash
mkdir -p /workspace
cd /workspace
git clone https://github.com/HYGON-AI/inference-cookbook-das.git
python3 -m pip install --no-deps \
  -r /workspace/inference-cookbook-das/docs/model-deployment/diffusion/magi-2-preview/requirements.txt \
  -i https://pypi.tuna.tsinghua.edu.cn/simple/
```


这里的 `requirements.txt` 是针对上述镜像的补充的模型依赖清单。

## 4. 拉取官方代码并应用补丁

三个仓库需使用以下固定 commit。

| 仓库 | 官方基线 commit | 处理方式 |
|---|---|---|
| MAGI-2-preview | `f68a0f9bbccbea56e0177a9bd912abc3b18ffe61` | 应用模型 patch |
| MagiAttention | `d12246a6f1a20736d3c7eee5eeced850281ac4ba` | 应用 HCU 适配 patch |
| MagiCompiler | `bfef5bc70226a0c0740e4c551e4f7245a974fb4f` | 官方源码直接构建 |

```bash
cd /workspace
git clone https://github.com/SandAI-org/MAGI-2-preview.git
git -C MAGI-2-preview checkout --detach f68a0f9bbccbea56e0177a9bd912abc3b18ffe61
git -C MAGI-2-preview apply --check /workspace/inference-cookbook-das/docs/model-deployment/diffusion/magi-2-preview/patches/magi2-preview-hcu.patch
git -C MAGI-2-preview apply --index /workspace/inference-cookbook-das/docs/model-deployment/diffusion/magi-2-preview/patches/magi2-preview-hcu.patch

git clone https://github.com/SandAI-org/MagiAttention.git
git -C MagiAttention checkout --detach d12246a6f1a20736d3c7eee5eeced850281ac4ba
git -C MagiAttention apply --check /workspace/inference-cookbook-das/docs/model-deployment/diffusion/magi-2-preview/patches/magiattention-hcu.patch
git -C MagiAttention apply --index /workspace/inference-cookbook-das/docs/model-deployment/diffusion/magi-2-preview/patches/magiattention-hcu.patch

git clone https://github.com/SandAI-org/MagiCompiler.git
git -C MagiCompiler checkout --detach bfef5bc70226a0c0740e4c551e4f7245a974fb4f
```


## 5. 源码构建并安装 MagiCompiler / MagiAttention

安装构建工具，然后编译并安装三个源码包：

```bash
python3 -m pip install --no-deps \
  setuptools==78.1.1 wheel==0.45.1 versioningit==3.3.0 tomli==2.2.1 \
  -i https://pypi.tuna.tsinghua.edu.cn/simple/

export MAX_JOBS=4
export PYTORCH_ROCM_ARCH=gfx936
mkdir -p /workspace/magi-wheels

cd /workspace/MagiCompiler
python3 -m pip wheel --no-deps --no-build-isolation --ignore-requires-python \
  -w /workspace/magi-wheels .

cd /workspace/MagiAttention
python3 -m pip wheel --no-deps --no-build-isolation -w /workspace/magi-wheels .
cd extensions
python3 -m pip wheel --no-deps --no-build-isolation -w /workspace/magi-wheels .

cd /workspace
python3 -m pip install --no-deps --ignore-requires-python /workspace/magi-wheels/magi_compiler-*.whl
python3 -m pip install --no-deps /workspace/magi-wheels/magi_attention-*.whl /workspace/magi-wheels/magi_attn_extensions-*.whl
```



## 6. 执行 warmup 和正式生成

```bash
cd /workspace/MAGI-2-preview
bash scripts/run_demo_hcu_opt.sh
```

默认使用 8 张卡生成 1920×1088（1080p）、10 秒视频，运行 5 个场景，每个场景各执行一次 warmup 和正式生成。

其他分辨率可依次运行：

```bash
RESOLUTION=540p OUTPUT_DIR=output/demo_hcu_opt_vllm_540p bash scripts/run_demo_hcu_opt.sh
RESOLUTION=272p OUTPUT_DIR=output/demo_hcu_opt_vllm_272p bash scripts/run_demo_hcu_opt.sh
```

## 7. 查看生成结果

默认输出目录为容器内的 `/workspace/MAGI-2-preview/output/demo_hcu_opt_vllm/`。需要导出到宿主机时执行：

```bash
docker cp magi2_hcu:/workspace/MAGI-2-preview/output ./magi2-output
```

上面的 `docker cp` 命令在宿主机执行。

| 文件 | 内容 |
|---|---|
| `sample_NNN.mp4` | 生成视频 |
| `sample_NNN_performance.json` | Preview、Refiner、VAE、编码等逐模块耗时 |
| `sample_e2e_timings.json` | 各样本端到端计时 |

脚本已经启用 `--benchmark`。复跑请更换 `OUTPUT_DIR`，保留历史结果。
