# Sapiens2 on MIGraphX

## 模型简介

Sapiens2 人体分割，输入 1024×768，FP16。用镜像里的 `migraphx-driver` 编译 ONNX 并计时。batch 写进 ONNX，换 batch 要换一份图。

规格：`0.4b`、`0.8b`、`1b`、`5b`。硬件：BW1100。一次用 1 张卡。

## 环境

| 项目 | 值 |
| --- | --- |
| 容器 | `sapiens2_migraphx` |
| 镜像 | `harbor.sourcefind.cn:5443/hcu/admin/base/custom:ubuntu-migraphx_5.3.1` |
| Python | 3.10.12 |
| 框架 | MIGraphX 5.3.1 |

模型代码从 [facebookresearch/sapiens2](https://github.com/facebookresearch/sapiens2) 克隆。权重按 [MODEL_ZOO.md](https://github.com/facebookresearch/sapiens2/blob/main/docs/MODEL_ZOO.md) 下载。编译读的是已经导出的 FP16 ONNX，文件名和下面命令里的一致，放在当前目录。

```bash
docker exec -it sapiens2_migraphx bash
source /opt/dtk/env.sh
export PYTHONPATH=/opt/dtk/lib:$PYTHONPATH
```

## 模型列表

| 模型权重 | 精度 | 分辨率 | 推荐硬件 | 卡数 | 任务 | 启动命令 |
| -------- | ---- | ------ | -------- | ---- | ---- | -------- |
| sapiens2_0.4b_seg | FP16 | 1024×768 | BW1100 | 1 | 吞吐 | [**`>_`**](#sapiens2-fp16-bw1100-1x-throughput) |
| sapiens2_0.8b_seg | FP16 | 1024×768 | BW1100 | 1 | 吞吐 | [**`>_`**](#sapiens2-fp16-bw1100-1x-throughput) |
| sapiens2_1b_seg | FP16 | 1024×768 | BW1100 | 1 | 吞吐 | [**`>_`**](#sapiens2-fp16-bw1100-1x-throughput) |
| sapiens2_5b_seg | FP16 | 1024×768 | BW1100 | 1 | 吞吐 | [**`>_`**](#sapiens2-fp16-bw1100-1x-throughput) |

## 启动命令

### Sapiens2 FP16 BW1100 1x throughput

先编译，再计时。计时不再加 `--fp16`。`Total time` 是一个 batch 的毫秒数，`images/s = 1000 / (Total time / batch)`。卡号换成空闲卡。把 `0.4b` 和 batch `8` 换成要测的规格。

```bash
HIP_VISIBLE_DEVICES=2 migraphx-driver compile \
  --onnx --gpu --fp16 --exhaustive-tune \
  -o sapiens2_0.4b_bs8_fp16.mxr \
  sapiens2_0.4b_seg_bs8_fp16.onnx

HIP_VISIBLE_DEVICES=2 migraphx-driver time \
  --migraphx -n 20 \
  sapiens2_0.4b_bs8_fp16.mxr
```

## 结果

BW1100 上的 MIGraphX FP16，对照 H20 上的 TensorRT FP16。吞吐单位是张/秒。BW/H20 是两边张/秒的比值。

| 模型 | MIGraphX bs4 | bs8 | bs16 | TensorRT bs4 | bs8 | bs16 | BW/H20 bs4 | bs8 | bs16 |
| ---- | ------------ | --- | ---- | ------------ | --- | ---- | ---------- | --- | ---- |
| 0.4B | 33.3731 | 34.0394 | 35.2692 | 34.1167 | 34.1423 | 34.3477 | 0.978 | 0.997 | 1.027 |
| 0.8B | 19.7105 | 18.5714 | 20.5763 | 16.6237 | 16.6293 | 16.6719 | 1.186 | 1.117 | 1.234 |
| 1B | 13.7200 | 13.5461 | 14.0388 | 11.2184 | 11.2885 | 11.3828 | 1.223 | 1.200 | 1.233 |
| 5B | 4.9048 | 4.9305 | 4.9314 | 3.2248 | 3.2507 | 3.2739 | 1.521 | 1.517 | 1.506 |

精度是 LIP 上 12 个共有类的 mIoU 和像素准确率。

| 模型 | bs4 mIoU | bs4 acc | bs8 mIoU | bs8 acc | bs16 mIoU | bs16 acc |
| ---- | -------- | ------- | -------- | ------- | --------- | -------- |
| 0.4B | 0.705401 | 0.893452 | 0.705342 | 0.893426 | 0.696372 | 0.884331 |
| 0.8B | 0.667889 | 0.880842 | 0.669172 | 0.881278 | 0.670280 | 0.882429 |
| 1B | 0.650380 | 0.875040 | 0.650362 | 0.875025 | 0.650375 | 0.875040 |
| 5B | 0.634061 | 0.860319 | 0.629331 | 0.859257 | 0.626700 | 0.858940 |

H20 TensorRT 同一套评测：

| 模型 | bs4 mIoU | bs4 acc | bs8 mIoU | bs8 acc | bs16 mIoU | bs16 acc |
| ---- | -------- | ------- | -------- | ------- | --------- | -------- |
| 0.4B | 0.699498 | 0.889904 | 0.699393 | 0.889869 | 0.699434 | 0.889889 |
| 0.8B | 0.519482 | 0.638048 | 0.519480 | 0.638047 | 0.519613 | 0.638100 |
| 1B | 0.565557 | 0.642702 | 0.565559 | 0.642714 | 0.565541 | 0.642699 |
| 5B | 0.034794 | 0.324774 | 0.034794 | 0.324772 | 0.034794 | 0.324771 |

