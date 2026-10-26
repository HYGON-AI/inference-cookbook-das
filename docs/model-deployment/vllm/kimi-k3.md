# Kimi-K3 on vLLM

## 模型简介

Kimi-K3 是 Moonshot AI 推出的 Kimi 系列模型，面向长上下文、工具调用和推理场景。本页提供基于 vLLM 的 BW1100 部署命令。

## 模型列表

| 模型权重 | 量化方式 | vLLM 镜像 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | --------- | -------- | ---- | -------- | -------- |
| [hygon/Kimi-K3-Channel-INT4-w4a8-INT8-w8a8](https://www.modelscope.cn/models/hygon/Kimi-K3-Channel-INT4-w4a8-INT8-w8a8) | INT4 W4A8 / INT8 W8A8 | 0.28.1 | BW1100 | 16 | IFB | [**`>_`**](#kimi-k3-channel-int4-w4a8-int8-w8a8-ifb-bw1100-16x-vllm-0281) |
| [hygon/kimi-k3-INT4](https://www.modelscope.cn/models/hygon/kimi-k3-INT4) | INT4 W4A8 | 0.28.1 | BW1100 | 16 | IFB | [**`>_`**](#kimi-k3-int4-ifb-bw1100-16x-vllm-0281) |
|  | INT4 W4A8 | 0.26 | BW1100 | 16 | IFB | [**`>_`**](#kimi-k3-int4-ifb-bw1100-16x-vllm-026) |
| [moonshotai/Kimi-K3](https://www.modelscope.cn/models/moonshotai/Kimi-K3) | BF16 | 0.26 | BW1100 | 16 | IFB | [**`>_`**](#kimi-k3-ifb-bw1100-16x-vllm-026) |

## 启动命令

### Kimi-K3-Channel-INT4-w4a8-INT8-w8a8 IFB BW1100 16x vLLM 0.28.1

在各节点分别执行对应命令；`MASTER_ADDR` 为 rank 0 节点地址，各节点使用相同的 `MASTER_PORT`。数据并行地址和端口占位符如有使用，也应在各节点保持一致。请从 [Inferact/Kimi-K3-DSpark](https://www.modelscope.cn/models/Inferact/Kimi-K3-DSpark) 下载模型，将 `DSPARK_MODEL` 替换为容器内实际绝对路径，并确保两个节点均可通过该路径访问完整模型文件。

#### rank 0

```bash
export VLLM_USE_NN=0

vllm serve hygon/Kimi-K3-Channel-INT4-w4a8-INT8-w8a8 \
  --trust-remote-code \
  --tensor-parallel-size 16 \
  --data-parallel-size 1 \
  --pipeline-parallel-size 1 \
  --nnodes 2 \
  --node-rank 0 \
  --master-addr "${MASTER_ADDR}" \
  --master-port "${MASTER_PORT}" \
  --quantization slimquant_w4a8 \
  --moe-backend aiter \
  -cc.pass_config.fuse_allreduce_rms=False \
  --load-format fastsafetensors \
  --disable-custom-all-reduce \
  --no-enable-flashinfer-autotune \
  --enable-auto-tool-choice \
  --tool-call-parser kimi_k3 \
  --reasoning-parser kimi_k3 \
  --speculative-config "{"model":"${DSPARK_MODEL}","num_speculative_tokens":7,"method":"dspark","draft_sample_method":"probabilistic","rejection_sample_method":"block"}" \
  --language-model-only \
  --no-async-scheduling \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
```

#### rank 1

```bash
export VLLM_USE_NN=0

vllm serve hygon/Kimi-K3-Channel-INT4-w4a8-INT8-w8a8 \
  --trust-remote-code \
  --tensor-parallel-size 16 \
  --data-parallel-size 1 \
  --pipeline-parallel-size 1 \
  --nnodes 2 \
  --node-rank 1 \
  --master-addr "${MASTER_ADDR}" \
  --master-port "${MASTER_PORT}" \
  --headless \
  --quantization slimquant_w4a8 \
  --moe-backend aiter \
  -cc.pass_config.fuse_allreduce_rms=False \
  --load-format fastsafetensors \
  --disable-custom-all-reduce \
  --no-enable-flashinfer-autotune \
  --enable-auto-tool-choice \
  --tool-call-parser kimi_k3 \
  --reasoning-parser kimi_k3 \
  --speculative-config "{"model":"${DSPARK_MODEL}","num_speculative_tokens":7,"method":"dspark","draft_sample_method":"probabilistic","rejection_sample_method":"block"}" \
  --language-model-only \
  --no-async-scheduling \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
```

### kimi-k3-INT4 IFB BW1100 16x vLLM 0.28.1

在各节点分别执行对应命令；`MASTER_ADDR` 为 rank 0 节点地址，各节点使用相同的 `MASTER_PORT`。请从 [Inferact/Kimi-K3-DSpark](https://www.modelscope.cn/models/Inferact/Kimi-K3-DSpark) 下载模型，将命令中的 `${DSPARK_MODEL}` 替换为容器内实际绝对路径，并确保两个节点均可通过该路径访问完整模型文件。

#### rank 0

```bash
export VLLM_USE_NN=0
export NCCL_GRAPH_MIXING_SUPPORT=1
export VLLM_USE_BREAKABLE_CUDAGRAPH=1

vllm serve hygon/kimi-k3-INT4 \
  --trust-remote-code \
  --tensor-parallel-size 16 \
  --data-parallel-size 1 \
  --pipeline-parallel-size 1 \
  --nnodes 2 \
  --node-rank 0 \
  --master-addr "${MASTER_ADDR}" \
  --master-port "${MASTER_PORT}" \
  --quantization slimquant_w4a8 \
  --moe-backend auto \
  -cc.pass_config.fuse_allreduce_rms=False \
  --load-format fastsafetensors \
  --disable-custom-all-reduce \
  --kv-cache-dtype fp8_e4m3 \
  --no-enable-flashinfer-autotune \
  --tool-call-parser kimi_k3 \
  --reasoning-parser kimi_k3 \
  --speculative-config '{"model":"${DSPARK_MODEL}", "num_speculative_tokens":7, "method": "dspark", "draft_sample_method": "probabilistic", "rejection_sample_method": "block"}' \
  --language-model-only \
  --max-model-len auto \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 128 \
  --gpu-memory-utilization 0.9
```

#### rank 1

```bash
export VLLM_USE_NN=0
export NCCL_GRAPH_MIXING_SUPPORT=1
export VLLM_USE_BREAKABLE_CUDAGRAPH=1

vllm serve hygon/kimi-k3-INT4 \
  --trust-remote-code \
  --tensor-parallel-size 16 \
  --data-parallel-size 1 \
  --pipeline-parallel-size 1 \
  --nnodes 2 \
  --node-rank 1 \
  --master-addr "${MASTER_ADDR}" \
  --master-port "${MASTER_PORT}" \
  --headless \
  --quantization slimquant_w4a8 \
  --moe-backend auto \
  -cc.pass_config.fuse_allreduce_rms=False \
  --load-format fastsafetensors \
  --disable-custom-all-reduce \
  --kv-cache-dtype fp8_e4m3 \
  --no-enable-flashinfer-autotune \
  --tool-call-parser kimi_k3 \
  --reasoning-parser kimi_k3 \
  --speculative-config '{"model":"${DSPARK_MODEL}", "num_speculative_tokens":7, "method": "dspark", "draft_sample_method": "probabilistic", "rejection_sample_method": "block"}' \
  --language-model-only \
  --max-model-len auto \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 128 \
  --gpu-memory-utilization 0.9
```

### Kimi-K3 IFB BW1100 16x vLLM 0.26

#### Node 0

```bash
export VLLM_KIMI_CONV1D_BACKEND=1
export VLLM_KIMI_FLASHKDA_BACKEND=1
export VLLM_USE_FLASHMLA=1

vllm serve moonshotai/Kimi-K3 \
  --trust-remote-code \
  --moe-backend auto \
  -cc.pass_config.fuse_allreduce_rms=False \
  --tensor-parallel-size 16 \
  --nnodes 2 \
  --node-rank 0 \
  --master-addr ip \
  --master-port 端口 \
  --load-format fastsafetensors \
  --gpu-memory-utilization 0.95 \
  --mm-encoder-tp-mode data \
  --limit-mm-per-prompt '{"image": 0}' \
  --max-num-seqs 128 \
  --max-num-batched-tokens 4096 \
  --enable-auto-tool-choice \
  --tool-call-parser kimi_k3 \
  --reasoning-parser kimi_k3
```

#### Node 1

```bash
export VLLM_KIMI_CONV1D_BACKEND=1
export VLLM_KIMI_FLASHKDA_BACKEND=1
export VLLM_USE_FLASHMLA=1

vllm serve moonshotai/Kimi-K3 \
  --trust-remote-code \
  --moe-backend auto \
  -cc.pass_config.fuse_allreduce_rms=False \
  --tensor-parallel-size 16 \
  --nnodes 2 \
  --node-rank 1 \
  --master-addr ip \
  --master-port 端口 \
  --headless \
  --load-format fastsafetensors \
  --gpu-memory-utilization 0.95 \
  --mm-encoder-tp-mode data \
  --limit-mm-per-prompt '{"image": 0}' \
  --max-num-seqs 128 \
  --max-num-batched-tokens 4096 \
  --enable-auto-tool-choice \
  --tool-call-parser kimi_k3 \
  --reasoning-parser kimi_k3
```

### kimi-k3-INT4 IFB BW1100 16x vLLM 0.26

#### Node 0

```bash
export VLLM_KIMI_CONV1D_BACKEND=1
export VLLM_KIMI_FLASHKDA_BACKEND=1
export VLLM_USE_FLASHMLA=1

vllm serve hygon/kimi-k3-INT4 \
  --trust-remote-code \
  --moe-backend auto \
  -cc.pass_config.fuse_allreduce_rms=False \
  --tensor-parallel-size 16 \
  --nnodes 2 \
  --node-rank 0 \
  --master-addr ip \
  --master-port 端口 \
  --load-format fastsafetensors \
  --gpu-memory-utilization 0.95 \
  --mm-encoder-tp-mode data \
  --limit-mm-per-prompt '{"image": 0}' \
  --max-num-seqs 128 \
  --max-num-batched-tokens 4096 \
  --enable-auto-tool-choice \
  --tool-call-parser kimi_k3 \
  --reasoning-parser kimi_k3
```

#### Node 1

```bash
export VLLM_KIMI_CONV1D_BACKEND=1
export VLLM_KIMI_FLASHKDA_BACKEND=1
export VLLM_USE_FLASHMLA=1

vllm serve hygon/kimi-k3-INT4 \
  --trust-remote-code \
  --moe-backend auto \
  -cc.pass_config.fuse_allreduce_rms=False \
  --tensor-parallel-size 16 \
  --nnodes 2 \
  --node-rank 1 \
  --master-addr ip \
  --master-port 端口 \
  --headless \
  --load-format fastsafetensors \
  --gpu-memory-utilization 0.95 \
  --mm-encoder-tp-mode data \
  --limit-mm-per-prompt '{"image": 0}' \
  --max-num-seqs 128 \
  --max-num-batched-tokens 4096 \
  --enable-auto-tool-choice \
  --tool-call-parser kimi_k3 \
  --reasoning-parser kimi_k3
```
