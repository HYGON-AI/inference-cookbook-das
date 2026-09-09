# Kimi-K3 on SGLang

## 模型简介

Kimi-K3 是 Moonshot AI 推出的 Kimi 系列模型，面向长上下文、工具调用和推理场景。本页提供基于 SGLang 的 BW1100 部署命令。

## 模型列表

| 模型权重 | 量化方式 | SGLang 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | ----------- | -------- | ---- | -------- | -------- |
| [moonshotai/Kimi-K3](https://www.modelscope.cn/models/moonshotai/Kimi-K3) | BF16 | 0.5.18 | BW1100 | 16 | IFB | [**`>_`**](#kimi-k3-ifb-bw1100-16x-sglang-0518) |
| [hygon/kimi-k3-INT4](https://www.modelscope.cn/models/hygon/kimi-k3-INT4) | INT4 W4A8 | 0.5.18 | BW1100 | 16 | IFB | [**`>_`**](#kimi-k3-int4-ifb-bw1100-16x-sglang-0518) |
| [hygon/kimi-k3-INT4](https://www.modelscope.cn/models/hygon/kimi-k3-INT4) | INT4 W4A8 | 0.5.18 | BW1100 | 32 | Prefill | [**`>_`**](#kimi-k3-int4-prefill-bw1100-32x-sglang-0518) |
| [hygon/kimi-k3-INT4](https://www.modelscope.cn/models/hygon/kimi-k3-INT4) | INT4 W4A8 | 0.5.18 | BW1100 | 32 | Decode | [**`>_`**](#kimi-k3-int4-decode-bw1100-32x-sglang-0518) |

## 启动命令

### Kimi-K3 IFB BW1100 16x SGLang 0.5.18

网卡配置参考：[IB 网卡](../../troubleshooting/common-issues.md#ib网卡)。

#### Node 0

```bash
export SGLANG_K3_ATTN_RESIDUAL_HCU=1
export SGLANG_KDA_USE_HCU_OP=1

sglang serve \
  --trust-remote-code \
  --model-path moonshotai/Kimi-K3 \
  --tp-size 16 \
  --port 11010 \
  --dist-init-addr <ip:端口> \
  --nnodes 2 \
  --node-rank 0 \
  --attention-backend hcu_mla \
  --linear-attn-backend triton \
  --page-size 64 \
  --linear-attn-prefill-backend flashkda \
  --reasoning-parser kimi_k3 \
  --tool-call-parser kimi_k3 \
  --mem-fraction-static 0.93 \
  --host 0.0.0.0 \
  --mm-attention-backend fa3 \
  --kv-cache-dtype fp8_e4m3 \
  --max-running-requests 128 \
  --mamba-full-memory-ratio 0.1 \
  --mamba-ssm-dtype bfloat16 \
  --load-format fastsafetensors \
  --disable-radix-cache
```

#### Node 1

```bash
export SGLANG_K3_ATTN_RESIDUAL_HCU=1
export SGLANG_KDA_USE_HCU_OP=1

sglang serve \
  --trust-remote-code \
  --model-path moonshotai/Kimi-K3 \
  --tp-size 16 \
  --port 11010 \
  --dist-init-addr <ip:端口> \
  --nnodes 2 \
  --node-rank 1 \
  --attention-backend hcu_mla \
  --linear-attn-backend triton \
  --page-size 64 \
  --linear-attn-prefill-backend flashkda \
  --reasoning-parser kimi_k3 \
  --tool-call-parser kimi_k3 \
  --mem-fraction-static 0.93 \
  --host 0.0.0.0 \
  --mm-attention-backend fa3 \
  --kv-cache-dtype fp8_e4m3 \
  --max-running-requests 128 \
  --mamba-full-memory-ratio 0.1 \
  --mamba-ssm-dtype bfloat16 \
  --load-format fastsafetensors \
  --disable-radix-cache
```

### kimi-k3-INT4 IFB BW1100 16x SGLang 0.5.18

网卡配置参考：[IB 网卡](../../troubleshooting/common-issues.md#ib网卡)。

#### Node 0

```bash
export SGLANG_K3_ATTN_RESIDUAL_HCU=1
export SGLANG_KDA_USE_HCU_OP=1
export SGLANG_USE_INT4_W4A8=1
export SGLANG_USE_LIGHTOP_W4A8_MARLIN_MOE=0

sglang serve \
  --trust-remote-code \
  --model-path hygon/kimi-k3-INT4 \
  --tp-size 16 \
  --port 11010 \
  --dist-init-addr <ip:port> \
  --nnodes 2 \
  --node-rank 0 \
  --attention-backend hcu_mla \
  --quantization slimquant_w4a8_marlin \
  --linear-attn-backend triton \
  --page-size 64 \
  --linear-attn-prefill-backend flashkda \
  --reasoning-parser kimi_k3 \
  --tool-call-parser kimi_k3 \
  --mem-fraction-static 0.93 \
  --host 0.0.0.0 \
  --mm-attention-backend fa3 \
  --kv-cache-dtype fp8_e4m3 \
  --max-running-requests 128 \
  --mamba-full-memory-ratio 0.1 \
  --mamba-ssm-dtype bfloat16 \
  --speculative-algorithm DSPARK \
  --speculative-draft-model-path RadixArk/Kimi-K3-DSpark \
  --speculative-dspark-block-size 7  \
  --speculative-draft-attention-backend triton \
  --speculative-draft-model-quantization unquant \
  --disable-radix-cache
```

#### Node 1

```bash
export SGLANG_K3_ATTN_RESIDUAL_HCU=1
export SGLANG_KDA_USE_HCU_OP=1
export SGLANG_USE_INT4_W4A8=1
export SGLANG_USE_LIGHTOP_W4A8_MARLIN_MOE=0

sglang serve \
  --trust-remote-code \
  --model-path hygon/kimi-k3-INT4 \
  --tp-size 16 \
  --port 11010 \
  --dist-init-addr <ip:port> \
  --nnodes 2 \
  --node-rank 1 \
  --attention-backend hcu_mla \
  --quantization slimquant_w4a8_marlin \
  --linear-attn-backend triton \
  --page-size 64 \
  --linear-attn-prefill-backend flashkda \
  --reasoning-parser kimi_k3 \
  --tool-call-parser kimi_k3 \
  --mem-fraction-static 0.93 \
  --host 0.0.0.0 \
  --mm-attention-backend fa3 \
  --kv-cache-dtype fp8_e4m3 \
  --max-running-requests 128 \
  --mamba-full-memory-ratio 0.1 \
  --mamba-ssm-dtype bfloat16 \
  --speculative-algorithm DSPARK \
  --speculative-draft-model-path RadixArk/Kimi-K3-DSpark \
  --speculative-dspark-block-size 7  \
  --speculative-draft-attention-backend triton \
  --speculative-draft-model-quantization unquant \
  --disable-radix-cache
```

### kimi-k3-INT4 Prefill BW1100 32x SGLang 0.5.18

网卡配置参考：[IB 网卡](../../troubleshooting/common-issues.md#ib网卡)。

```bash
NODE_RANK="${1:-}"

export SGLANG_USE_LIGHTOP=1
export SGLANG_K3_ATTN_RESIDUAL_HCU=1
export SGLANG_KDA_USE_HCU_OP=1
export SGLANG_USE_INT4_W4A8=1
export SGLANG_USE_LIGHTOP_W4A8_MARLIN_MOE=0
export SGLANG_USE_W4A8_CONTIGUOUS_HIPC=1

sglang serve \
  --trust-remote-code \
  --model-path hygon/kimi-k3-INT4 \
  --port 11010 \
  --dist-init-addr <ip:port> \
  --nnodes 8 \
  --node-rank "${NODE_RANK}" \
  --tp-size 4 \
  --dp-size 1 \
  --pp-size 8 \
  --attention-backend hcu_mla \
  --linear-attn-prefill-backend flashkda \
  --quantization slimquant_w4a8_marlin \
  --page-size 64 \
  --reasoning-parser kimi_k3 \
  --tool-call-parser kimi_k3 \
  --mem-fraction-static 0.85 \
  --host 0.0.0.0 \
  --mm-attention-backend fa3 \
  --kv-cache-dtype fp8_e4m3 \
  --max-running-requests 128 \
  --mamba-full-memory-ratio 0.15 \
  --mamba-ssm-dtype bfloat16 \
  --model-loader-extra-config '{"enable_multithread_load":"true","num_threads":64}' \
  --skip-server-warmup \
  --disable-radix-cache
```

### kimi-k3-INT4 Decode BW1100 32x SGLang 0.5.18

网卡配置参考：[IB 网卡](../../troubleshooting/common-issues.md#ib网卡)。

```bash
NODE_RANK="${1:-}"

export SGLANG_USE_LIGHTOP=1
export SGLANG_K3_ATTN_RESIDUAL_HCU=1
export SGLANG_KDA_USE_HCU_OP=1
export SGLANG_USE_INT4_W4A8=1
export SGLANG_USE_LIGHTOP_W4A8_MARLIN_MOE=0
export SGLANG_USE_W4A8_MASKED_HIPC=1

export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export ROCSHMEM_TOPO_FILE_FORCE=topo.config
export ROCSHMEM_ALLOWED_IBV_DEVICES=shca_1,shca_2,shca_3,shca_4
export ROCSHMEM_IPC_MNVL=1

sglang serve \
  --trust-remote-code \
  --model-path hygon/kimi-k3-INT4 \
  --port 11010 \
  --dist-init-addr 11.16.100.200:12115 \
  --nnodes 8 \
  --node-rank "${NODE_RANK}" \
  --tp-size 32 \
  --dp-size 8 \
  --pp-size 1 \
  --ep-size 32 \
  --enable-dp-attention \
  --enable-dp-lm-head \
  --deepep-mode auto \
  --moe-a2a-backend deepep \
  --attention-backend hcu_mla \
  --linear-attn-prefill-backend flashkda \
  --quantization slimquant_w4a8_marlin \
  --speculative-algorithm DSPARK \
  --speculative-draft-model-path RadixArk/Kimi-K3-DSpark \
  --speculative-dspark-block-size 7 \
  --speculative-draft-attention-backend triton \
  --speculative-draft-model-quantization unquant \
  --page-size 64 \
  --reasoning-parser kimi_k3 \
  --tool-call-parser kimi_k3 \
  --mem-fraction-static 0.88 \
  --host 0.0.0.0 \
  --mm-attention-backend fa3 \
  --kv-cache-dtype fp8_e4m3 \
  --max-running-requests 128 \
  --mamba-full-memory-ratio 0.9 \
  --mamba-ssm-dtype bfloat16 \
  --model-loader-extra-config '{"enable_multithread_load":"true","num_threads":64}' \
  --skip-server-warmup \
  2>&1 | tee "${log_file}"
```