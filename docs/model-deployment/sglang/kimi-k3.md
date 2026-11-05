# Kimi-K3 on SGLang

## 模型简介

Kimi-K3 是 Moonshot AI 推出的 Kimi 系列模型，面向长上下文、工具调用和推理场景。本页提供基于 SGLang 的 BW1100 部署命令。

## 模型列表

| 模型权重 | 量化方式 | SGLang 镜像 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | ----------- | -------- | ---- | -------- | -------- |
| [moonshotai/Kimi-K3](https://www.modelscope.cn/models/moonshotai/Kimi-K3) | FP4 W4A16 | 0.5.18 | BW1100 | 16 | IFB | [**`>_`**](#kimi-k3-ifb-bw1100-16x-sglang-0518) |
| [hygon/kimi-k3-INT4](https://www.modelscope.cn/models/hygon/kimi-k3-INT4) | INT4 W4A8 | 0.5.18 | BW1100 | 16 | IFB | [**`>_`**](#kimi-k3-int4-ifb-bw1100-16x-sglang-0518) |
| [hygon/kimi-k3-INT4](https://www.modelscope.cn/models/hygon/kimi-k3-INT4) | INT4 W4A8 | 0.5.18 | BW1100 | 32 | Prefill | [**`>_`**](#kimi-k3-int4-prefill-bw1100-32x-sglang-0518) |
| [hygon/kimi-k3-INT4](https://www.modelscope.cn/models/hygon/kimi-k3-INT4) | INT4 W4A8 | 0.5.18 | BW1100 | 32 | Decode | [**`>_`**](#kimi-k3-int4-decode-bw1100-32x-sglang-0518) |
| [hygon/Kimi-K3-Channel-W4A8-INT4-W8A8-INT8](https://www.modelscope.cn/models/hygon/Kimi-K3-Channel-W4A8-INT4-W8A8-INT8) | INT4 W4A8 + INT8 W8A8 | 0.5.18 | ScaleX40-3G | 32 | Prefill | [**`>_`**](#kimi-k3-channel-w4a8-int4-w8a8-int8-prefill-scalex40-3g-32x-sglang-0518) |
|  | INT4 W4A8 + INT8 W8A8 | 0.5.18 | ScaleX40-3G | 32 | Decode | [**`>_`**](#kimi-k3-channel-w4a8-int4-w8a8-int8-decode-scalex40-3g-32x-sglang-0518) |
|  | INT4 W4A8 + INT8 W8A8 | 0.5.18 | BW1100 | 16 | IFB | [**`>_`**](#kimi-k3-channel-w4a8-int4-w8a8-int8-ifb-bw1100-16x-sglang-0518) |

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
export SGLANG_KDA_DISABLE_TRITON_CHUNK_H=1

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
export NCCL_IB_HCA=shca_1:1,shca_2:1,shca_3:1,shca_4:1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export ROCSHMEM_TOPO_FILE_FORCE=topo.config
export ROCSHMEM_ALLOWED_IBV_DEVICES=shca_1,shca_2,shca_3,shca_4
export ROCSHMEM_IPC_MNVL=1
export SGLANG_SIMULATE_ACC_LEN=5
export SGLANG_SIMULATE_ACC_METHOD=match-expected
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export SGLANG_USE_AITER_AR=1
export SGLANG_ENABLE_ATTN_TP_USE_AITER_CUSTOM_COMM=1
export SGLANG_K3_DENSE_MLP_ATTN_TP=1
export SGLANG_ENABLE_HCU_FA_PREFIX_VALID=1

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
  --deepep-mode low_latency \
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
  --mem-fraction-static 0.86 \
  --host 0.0.0.0 \
  --mm-attention-backend fa3 \
  --kv-cache-dtype fp8_e4m3 \
  --max-running-requests 128 \
  --disaggregation-decode-extra-slots 16 \
  --mamba-full-memory-ratio 0.2 \
  --mamba-ssm-dtype bfloat16 \
  --disaggregation-mode decode \
  --disaggregation-ib-device shca_1,shca_2,shca_3,shca_4 \
  --disaggregation-transfer-backend fake \
  --skip-server-warmup \
  --disable-radix-cache
```

### Kimi-K3-Channel-W4A8-INT4-W8A8-INT8 Prefill ScaleX40-3G 32x SGLang 0.5.18

在 8 个节点上分别运行，每节点 4 卡（TP4 / PP8）。以下为 rank 0 命令，其余节点将 `--node-rank` 改为 1～7；所有节点的 `--dist-init-addr` 填写本组 rank 0 的地址。此命令用于单独测量 Prefill。

网卡配置参考：[IB 网卡](../../troubleshooting/common-issues.md#ib网卡)。

```bash
export SGLANG_USE_LIGHTOP=1
export SGLANG_K3_ATTN_RESIDUAL_HCU=1
export SGLANG_KDA_USE_HCU_OP=1
export SGLANG_USE_INT4_W4A8=1
export SGLANG_USE_LIGHTOP_W4A8_MARLIN_MOE=0
export SGLANG_USE_W4A8_CONTIGUOUS_HIPC=1
export SGLANG_KDA_DISABLE_TRITON_CHUNK_H=1

sglang serve \
  --trust-remote-code \
  --model-path hygon/Kimi-K3-Channel-W4A8-INT4-W8A8-INT8 \
  --dist-init-addr <node0_ip>:5001 \
  --nnodes 8 \
  --node-rank 0 \
  --tp-size 4 \
  --dp-size 1 \
  --pp-size 8 \
  --attention-backend hcu_mla \
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
  --disable-custom-all-reduce
```

### Kimi-K3-Channel-W4A8-INT4-W8A8-INT8 Decode ScaleX40-3G 32x SGLang 0.5.18

在 8 个节点上分别运行，每节点 4 卡（TP32 / DP16 / EP32）。以下为 rank 0 命令，其余节点将 `--node-rank` 改为 1～7；所有节点的 `--dist-init-addr` 填写本组 rank 0 的地址。本方案使用 fake transfer 和模拟接受长度 5，用于独立测量 Decode，不与上面的 Prefill 组成真实 PD 分离服务。启动前按实际机器调整 IB 网卡和设备，并从测试机器复制草稿模型、拓扑文件和调优文件，确保命令中的路径在各节点可读。

网卡配置参考：[IB 网卡](../../troubleshooting/common-issues.md#ib网卡)。

```bash
export NCCL_SOCKET_IFNAME=ibp129s0
export GLOO_SOCKET_IFNAME=ibp129s0
export NCCL_IB_HCA=shca_1:1,shca_2:1,shca_3:1,shca_4:1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export ROCSHMEM_TOPO_FILE_FORCE=/home/sglang-ws/109_kimi/config/topo.config
export ROCSHMEM_ALLOWED_IBV_DEVICES=shca_1,shca_2,shca_3,shca_4
export ROCSHMEM_HEAP_SIZE=4442450944
export ROCSHMEM_IPC_MNVL=1
export HIP_BUFFER_EXTRA_SIZE=0
export SGLANG_USE_LIGHTOP=1
export SGLANG_KDA_USE_HCU_OP=1
export SGLANG_USE_INT4_W4A8=1
export SGLANG_USE_LIGHTOP_W4A8_MARLIN_MOE=0
export SGLANG_USE_W4A8_CONTIGUOUS_HIPC=1
export SGLANG_USE_W4A8_MASKED_HIPC=1
export W8A8_SUPPORT_METHODS=3
export HIPBLASLT_TUNING_OVERRIDE_FILE=/home/sglang-ws/109_kimi/config/gemm_sizes_dp16_ep32_20261010_145700/blaslt_res/hipblaslt_tuning_override.csv
export ROCBLAS_TENSILE_LIBPATH=/home/sglang-ws/109_kimi/config/gemm_sizes_dp16_ep32_20261010_145700/blas_res/config/library_gpu6
export SGLANG_K3_FUSED_SITU_INT8=1
export SGLANG_K3_FUSED_LATENT_RMS_INT8=1
export SGLANG_K3_FUSED_ATTN_RES=1
export SGLANG_K3_REUSE_INPUT_INT8=1
export SGLANG_K3_FUSED_MLA_FRONT=1
export SGLANG_K3_FUSED_MLA_POST=1
export SGLANG_DSPARK_SPLITKV_V2=1
export SGLANG_ENABLE_HCU_FA_PREFIX_VALID=1
export SGLANG_USE_FUSED_REPLAY_STATE_INDICES=1
export SGLANG_K3_SBO_METHOD=serial
export SGLANG_USE_AITER_AR=1
export SGLANG_ENABLE_ATTN_TP_USE_AITER_CUSTOM_COMM=1
export SGLANG_K3_DENSE_MLP_ATTN_TP=1
export GPU_FORCE_BLIT_COPY_SIZE=16777218
export SGLANG_PREBUILT_ASYNC_TOKEN_H2D=1
export SGLANG_SIMULATE_ACC_LEN=5
export SGLANG_SIMULATE_ACC_METHOD=match-expected
export MC_ENABLE_DEST_DEVICE_AFFINITY=1

sglang serve \
  --model-path hygon/Kimi-K3-Channel-W4A8-INT4-W8A8-INT8 \
  --trust-remote-code \
  --host 0.0.0.0 \
  --nnodes 8 \
  --node-rank 0 \
  --dist-init-addr <node0_ip>:12115 \
  --tp-size 32 \
  --dp-size 16 \
  --pp-size 1 \
  --ep-size 32 \
  --enable-dp-attention \
  --enable-dp-lm-head \
  --attention-backend hcu_mla \
  --linear-attn-prefill-backend flashkda \
  --mm-attention-backend fa3 \
  --quantization slimquant_w4a8_marlin \
  --kv-cache-dtype fp8_e4m3 \
  --page-size 64 \
  --moe-a2a-backend deepep \
  --deepep-mode low_latency \
  --mem-fraction-static 0.85 \
  --max-running-requests 128 \
  --mamba-full-memory-ratio 0.2 \
  --mamba-ssm-dtype bfloat16 \
  --disaggregation-mode decode \
  --disaggregation-ib-device shca_1,shca_2,shca_3,shca_4 \
  --disaggregation-transfer-backend fake \
  --disaggregation-decode-extra-slots 0 \
  --speculative-algorithm DSPARK \
  --speculative-draft-model-path /public/opendas/DL_DATA/llm-models/Kimi-K3-DSpark \
  --speculative-dspark-block-size 7 \
  --speculative-draft-attention-backend triton \
  --speculative-draft-model-quantization unquant \
  --reasoning-parser kimi_k3 \
  --tool-call-parser kimi_k3 \
  --model-loader-extra-config '{"enable_multithread_load": "true", "num_threads": 64}' \
  --disable-radix-cache \
  --skip-server-warmup
```

### Kimi-K3-Channel-W4A8-INT4-W8A8-INT8 IFB BW1100 16x SGLang 0.5.18

在 2 个节点上分别运行，每节点 8 卡（TP16 / DP2 / EP16）。以下为 rank 0 命令，另一节点将 `--node-rank` 改为 1；两个节点的 `--dist-init-addr` 填写 rank 0 的地址。按实际机器调整 IB 网卡和设备，并准备命令中引用的草稿模型和拓扑文件。

网卡配置参考：[IB 网卡](../../troubleshooting/common-issues.md#ib网卡)。

```bash
export NCCL_SOCKET_IFNAME=ens66f0np0
export GLOO_SOCKET_IFNAME=ens66f0np0
export NCCL_IB_HCA=mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export ROCSHMEM_TOPO_FILE_FORCE=/home/sglang-ws/1012_kimi/kimi-sh/config/topo.config
export ROCSHMEM_ALLOWED_IBV_DEVICES=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
export ROCSHMEM_IB_GID_INDEX=0
export DEEPEP_ENABLE_LL_LAYERED_OPT=1
export SGLANG_USE_LIGHTOP=1
export SGLANG_KDA_USE_HCU_OP=1
export SGLANG_ROCM_USE_AITER_MOE=1
export SGLANG_USE_INT4_W4A8=1
export SGLANG_USE_LIGHTOP_W4A8_MARLIN_MOE=0
export SGLANG_USE_W4A8_CONTIGUOUS_HIPC=1
export SGLANG_USE_W4A8_MASKED_HIPC=1
export W8A8_SUPPORT_METHODS=3
export SGLANG_ENABLE_HCU_FA_PREFIX_VALID=1
export SGLANG_USE_FUSED_REPLAY_STATE_INDICES=1
export SGLANG_K3_FUSED_SITU_INT8=1
export SGLANG_K3_FUSED_LATENT_RMS_INT8=1
export SGLANG_K3_FUSED_ATTN_RES=1
export SGLANG_K3_REUSE_INPUT_INT8=1
export SGLANG_K3_FUSED_MLA_FRONT=1
export SGLANG_K3_FUSED_MLA_POST=1
export SGLANG_DSPARK_SPLITKV_V2=1
export SGLANG_USE_AITER_AR=1
export SGLANG_ENABLE_ATTN_TP_USE_AITER_CUSTOM_COMM=1
export SGLANG_K3_DENSE_MLP_ATTN_TP=1
export SGLANG_K3_SBO_METHOD=off
export GPU_FORCE_BLIT_COPY_SIZE=16777218
export SGLANG_PREBUILT_ASYNC_TOKEN_H2D=1

sglang serve \
  --model-path hygon/Kimi-K3-Channel-W4A8-INT4-W8A8-INT8 \
  --trust-remote-code \
  --host 0.0.0.0 \
  --nnodes 2 \
  --node-rank 0 \
  --dist-init-addr <node0_ip>:12115 \
  --tp-size 16 \
  --dp-size 2 \
  --pp-size 1 \
  --ep-size 16 \
  --enable-dp-attention \
  --enable-dp-lm-head \
  --attention-backend hcu_mla \
  --linear-attn-prefill-backend flashkda \
  --mm-attention-backend fa3 \
  --quantization slimquant_w4a8_marlin \
  --kv-cache-dtype fp8_e4m3 \
  --page-size 64 \
  --moe-a2a-backend deepep \
  --deepep-mode auto \
  --mem-fraction-static 0.88 \
  --max-running-requests 64 \
  --mamba-full-memory-ratio 0.15 \
  --mamba-ssm-dtype bfloat16 \
  --reasoning-parser kimi_k3 \
  --tool-call-parser kimi_k3 \
  --speculative-algorithm DSPARK \
  --speculative-draft-model-path /public/opendas/DL_DATA/llm-models/sglang-Kimi-K3-DSpark \
  --speculative-dspark-block-size 7 \
  --speculative-draft-attention-backend triton \
  --speculative-draft-model-quantization unquant \
  --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 64}' \
  --disable-radix-cache \
  --skip-server-warmup
```

## API 调用

### IFB

在 16 卡 IFB 的 rank 0 节点执行：

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"hygon/Kimi-K3-Channel-W4A8-INT4-W8A8-INT8","messages":[{"role":"user","content":"你好"}],"max_tokens":32}'
```
