# Gemma-4-26B-A4B-it on vLLM

## 模型简介

Gemma-4-26B-A4B-it 是 Gemma4 系列指令模型，采用 MoE 架构，支持 sliding-window attention（SWA）和 full attention，本文档提供其在 vLLM 上的部署示例。

## 模型列表

| 模型权重 | 量化方式 | vLLM 镜像 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | --------- | -------- | ---- | -------- | -------- |
| gemma-4-26B-A4B-it | BF16 | 0.28.1 | BW | 1 | IFB | [**`>_`**](#gemma-4-26b-a4b-it-ifb-bw-1x-vllm-0281) |

### 0.28.1 平台支持说明

当前已在BW1000 HCU 单卡（GPU1，TP1）完成文本推理验证。容器使用 vLLM `0.28.1+das.29071fe.dtk2610`、PyTorch 2.11.0 和 Transformers 5.21.0，模型权重为 BF16，KV Cache 为 FP8。其他硬件、卡数和量化方式待补充。

Gemma4 的 SWA 层 head dimension 为 256，full attention 层 head dimension 为 512。当前版本启动日志报告异构 head dimension，自动选择 `TRITON_ATTN`。

真实问答验证使用 447 条本地 MMLU-Pro realQA 问题，MTP4 acceptance 为 66.81%，HTTP 错误和抢占均为 0。该数据用于推理负载验证，不是官方 MMLU-Pro 准确率复现。

## 启动命令

### Gemma-4-26B-A4B-it IFB BW 1x vLLM 0.28.1

```bash
export HIP_VISIBLE_DEVICES=1
export VLLM_ENGINE_READY_TIMEOUT_S=1800

vllm serve /mnt/models/gemma-4-26B-A4B-it \
  --served-model-name gemma-4-26b-a4b-it \
  --host 0.0.0.0 \
  --port 8091 \
  --tensor-parallel-size 1 \
  --max-model-len 16384 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 160 \
  --gpu-memory-utilization 0.92 \
  --block-size 64 \
  --enable-prefix-caching \
  --language-model-only \
  --dtype bfloat16 \
  --kv-cache-dtype fp8 \
  --speculative-config '{"method":"mtp","model":"/mnt/models/gemma-4-26B-A4B-it-assistant","num_speculative_tokens":4}'
```

## 性能优化开关（BW1000）

BW1000 单卡上的一组 Gemma4 专用优化由**一个**环境变量控制：

```bash
export VLLM_HCU_GEMMA_BW1000=1
```

打开后生效的内容：full attention（head 512）层走 FA 路径并直接从 FP8 KV 池按页
gather（替代整池 FP8→BF16 转换）、BW1000 注意力启动参数调优、SWA 层在 K/V scale 均为 1
时直接以 BF16 读取 FP8、LM head 的 rocBLAS 算法表 / softcap 查表 / 320 与 400 行桶 /
融合草稿 argmax。

### 适用范围

该开关是 opt-in，不默认开启。其安装路径断言 Gemma4 专有前提（词表 262144、
`soft_cap` 30.0、已知形状的 BF16 lm_head、TP1），并加载一张按 torch/HIP/rocBLAS/架构
校验过的 rocBLAS 表，因此**仅适用于 BW1000（gfx936）单卡、BF16 未量化权重的
Gemma-4-26B-A4B-it**。在其他模型、其他架构、TP>1，或 `gemma-4-26B-A4B-it-FP8_DYNAMIC`
这类量化权重上开启会在 `load_model` 阶段失败。

### 确认生效

启动日志应出现以下四行，缺任何一行表示未安装成功：

```
Gemma4 BW1000 head table loaded: {... 'verified': True ...}
Gemma4 BW1000 softcap lookup active: soft_cap=30.0, 65536-entry BF16 table
Gemma4 BW1000 head bucket active: rows=128..400 buckets=[320, 400] (all present in the rocBLAS table)
Gemma4 BW1000 fused draft argmax active: rows=33..256 vocab=262144 BM schedule=[32, 64, 128] (6 launch configs pre-compiled)
```

其中 `head table loaded: verified True` 只表示 CSV 通过校验器，不表示运行时的 GEMM
一定命中了表项。

### 带优化开关的启动命令

```bash
export VLLM_HCU_GEMMA_BW1000=1
export HIP_VISIBLE_DEVICES=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600

vllm serve /mnt/models/gemma-4-26B-A4B-it \
  --served-model-name gemma-4-26b-a4b-it \
  --host 0.0.0.0 \
  --port 8091 \
  --tensor-parallel-size 1 \
  --max-model-len 16384 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 160 \
  --gpu-memory-utilization 0.92 \
  --block-size 64 \
  --enable-prefix-caching \
  --language-model-only \
  --dtype bfloat16 \
  --kv-cache-dtype fp8 \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
  --speculative-config '{"method":"mtp","model":"/mnt/models/gemma-4-26B-A4B-it-assistant","num_speculative_tokens":4}'
```

首次启动会做 Triton 编译与 CUDA graph 捕获，耗时明显长于后续启动（同卡同参数实测
readiness 11.0 分钟 → 命中编译缓存后 3.5 分钟），因此
`VLLM_ENGINE_READY_TIMEOUT_S` 需要放宽。

### 已知边界

- 该开关只在上述单一配置（BW1000 单卡、TP1、BF16 权重、FP8 KV、MTP4）下验证过。
- 其中的 rocBLAS 算法表按 PyTorch / HIP / rocBLAS 版本和 GPU 架构逐字符校验；校验不通过
  时启动会直接失败。更换容器镜像或 ROCm 组件版本后需要重新调表。
- 本文不给总体加速比：各项优化的测量口径不同，不能相乘，也没有做跨配置的统一基线测量。
- 其中融合草稿 argmax 的累加顺序与库实现不同，顶端并列的 token 可能换胜者。在
  `temperature=0` 的贪心解码下这**不改变输出**：投机解码的贪心验证核写出的始终是目标
  模型的 argmax，草稿只决定这一步接受到第几个 token，因此草稿侧的差异只影响接受长度
  （即速度）。若使用 `temperature>0` 采样，被接受的草稿 token 会被直接写出，某一次带
  种子的采样结果可能与原路径不同（投机采样本身仍以目标分布为准）。

## API 调用

### IFB

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8091/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="gemma-4-26b-a4b-it",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "中国的首都是哪里？"},
    ],
    temperature=0,
    max_tokens=2048,
)
print(response.choices[0].message.content)
```

```bash
curl http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4-26b-a4b-it",
    "temperature": 0,
    "max_tokens": 2048,
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": [
        {"type": "text", "text": "中国的首都是哪里？"}
      ]}
    ]
  }'
```
