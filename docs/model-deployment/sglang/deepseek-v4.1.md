# DeepSeek-V4.1 on SGLang

## 模型简介

DeepSeek-V4.1 是 DeepSeek 系列的混合专家模型。本页汇总 DeepSeek-V4.1 系列模型在 HCU 平台上使用 SGLang 的部署方式.

## 模型列表

| 模型权重                                                     | 量化方式  | SGLang 镜像 | 推荐硬件 | 卡数  | 部署方式 | 启动命令                                                     |
| ------------------------------------------------------------ | --------- | ----------- | -------- | ----- | -------- | ------------------------------------------------------------ |
| [hygon/DeepSeek-V4.1-Flash-1017-Channel-FP8-w8a8](https://modelscope.cn/models/hygon/DeepSeek-V4.1-Flash-1017-Channel-FP8-w8a8) | FP8 W8A8  | [0.5.18](../docker_images.md) | BW1100   | 8     | IFB      | [**`>_`**](#deepseek-v41-flash-1017-channel-fp8-w8a8-ifb-bw1100-8x-sglang-0518) |
|  | FP8 W8A8  | [0.5.18](../docker_images.md) | BW1100   | 16    | 1P1D     | [**`>_`**](#deepseek-v41-flash-1017-channel-fp8-w8a8-1p1d-bw1100-16x-sglang-0518) |
|                                                              | FP8 W8A8  | [0.5.18](../docker_images.md) | scalex40-3G | 40    | 1P1D     | [**`>_`**](#deepseek-v41-flash-1017-channel-fp8-w8a8-1p1d-scalex40-3g-40x-sglang-0518) |
| [hygon/DeepSeek-V4.1-Flash-1017-Channel-INT4-w4a8](https://modelscope.cn/models/hygon/DeepSeek-V4.1-Flash-1017-Channel-INT4-w4a8) | INT4-W4A8 | [0.5.18](../docker_images.md) | BW1000   | 8     | IFB      | [**`>_`**](#deepseek-v41-flash-1017-channel-int4-w4a8-ifb-bw1000-8x-sglang-0518) |
|                                                              | INT4-W4A8 | [0.5.18](../docker_images.md) | BW1000   | 16    | 1P1D     | [**`>_`**](#deepseek-v41-flash-1017-channel-int4-w4a8-1p1d-bw1000-16x-sglang-0518) |
|                                                              | INT4-W4A8 | [0.5.18](../docker_images.md) | BW1000   | 17 | EPD      | [**`>_`**](#deepseek-v41-flash-1017-channel-int4-w4a8-epd-bw1000-17x-sglang-0518) |
| [hygon/DeepSeek-V4.1-Flash-Channel-INT4-w4a8](https://modelscope.cn/models/hygon/DeepSeek-V4.1-Flash-Channel-INT4-w4a8) | INT4-W4A8 | [0.5.18](../docker_images.md) | BW1100   | 8     | 1P1D     | [**`>_`**](#deepseek-v41-flash-channel-int4-w4a8-1p1d-bw1100-8x-sglang-0518) |

## DeepEP 配置

以下 `deepep_IntraConfig.json` 仅作为参考配置。使用时请将其保存为 `deepep_IntraConfig.json`，并根据实际保存路径设置启动命令中的 `--deepep-config` 参数。

```json
{
  "normal_dispatch": {
    "num_sms": 48,
    "num_max_nvl_chunked_send_tokens": 6,
    "num_max_nvl_chunked_recv_tokens": 256,
    "num_max_rdma_chunked_send_tokens": 6,
    "num_max_rdma_chunked_recv_tokens": 128
  },
  "normal_combine": {
    "num_sms": 48,
    "num_max_nvl_chunked_send_tokens": 4,
    "num_max_nvl_chunked_recv_tokens": 256,
    "num_max_rdma_chunked_send_tokens": 6,
    "num_max_rdma_chunked_recv_tokens": 128
  }
}
```
## EPLB配置参考：[EPLB](../../optimization/static-eplb-sglang.md)

## 启动命令

运行前，请按实际环境替换各段中的 IP 占位符，并核对网卡名。`--host` 填写本机服务监听地址，`--dist-init-addr` 填写对应 P 或 D 分组的 Node 0 地址，同组节点使用同一初始化地址。

### DeepSeek-V4.1-Flash-1017-Channel-FP8-w8a8 IFB BW1100 8x SGLang 0.5.18

```bash
export NCCL_SOCKET_IFNAME=ens66f1np1
export GLOO_SOCKET_IFNAME=ens66f1np1
export SGLANG_TORCH_PROFILER_DIR=/home/proj_dpsk-v4/profile
export SGLANG_OPT_SWIGLU_CLAMP_FUSION=false
export SGLANG_USE_AITER_AG=0
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_LIGHTOP_GROUP_FP8_QUANT=1
export SGLANG_USE_DEEPGEMM_MOE=1
export SGLANG_USE_DPSKV4_LIGHTOP_QUANT_K_CACHE=1
export SGLANG_USE_DPSKV4_LIGHTOP_RMSNORM=1
export SGLANG_DSV4_SPLIT_PREFILL_DECODE_MLA=1
export SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256
export SGLANG_USE_LIGHTOP_PAGED_MQA_LOGITS_FP4=1
export SGLANG_USE_LIGHTOP_DSV41_PAGED_INDEXER=1
export NCCL_SIMPLE_CHANNELS=32
export NCCL_IB_HCA=mlx5_2
export ROCSHMEM_MAX_NUM_CONTEXTS=48
export SGLANG_USE_FP8_W8A8_MOE=1
export DEEPEP_ENABLE_LL_LAYERED_OPT=1
export ROCSHMEM_DISABLE_HDP_FLUSH=1

sglang serve \
 --model-path hygon/DeepSeek-V4.1-Flash-1017-Channel-FP8-w8a8 \
 --served-model-name DeepSeek-V4.1-Flash-1017-Channel-FP8-w8a8 \
 --tp-size 8 \
 --host 0.0.0.0 \
 --port 10015 \
 --trust-remote-code \
 --dist-timeout 10000 \
 --watchdog-timeout 3600 \
 --cuda-graph-backend-prefill disabled \
 --skip-server-warmup \
 --mem-fraction-static 0.85 \
 --max-running-requests 32 \
 --cuda-graph-max-bs-decode 32 \
 --speculative-algorithm DSPARK \
 --speculative-dspark-block-size 5 \
 --reasoning-parser auto \
 --tool-call-parser auto \
 --chunked-prefill-size 32768 \
 --dist-init-addr 127.0.0.1:10748 \
 --dp 8 \
 --enable-dp-attention \
 --enable-dp-attention-local-control-broadcast \
 --enable-dp-lm-head \
 --ep 8 \
 --moe-a2a-backend deepep \
 --deepep-config deepep_IntraConfig.json \
 --deepep-mode auto \
 --moe-runner-backend deep_gemm \
 --enable-single-batch-overlap \
 --speculative-moe-a2a-backend deepep \
 --speculative-moe-runner-backend deep_gemm \
 --disable-radix-cache \
 2>&1 | tee running_dpsk-v41_nmz20_decode_nodisagg.log
```

### DeepSeek-V4.1-Flash-1017-Channel-FP8-w8a8 1P1D BW1100 16x SGLang 0.5.18

网卡配置参考：[IB 网卡](../../troubleshooting/common-issues.md#ib网卡)。

表格中的 16 卡方案使用 8 卡 Prefill（CP8EP8）和 8 卡 Decode（DP8EP8）。将 Decode 换为本节的 DP16EP16 方案后，Prefill 仍使用 8 卡，合计 24 卡。

#### P node CP8EP8

```bash
export NCCL_SOCKET_IFNAME=ens61f1np1
export GLOO_SOCKET_IFNAME=ens61f1np1
export SGLANG_TORCH_PROFILER_DIR=/home/proj_dpsk-v4/profile
export SGLANG_OPT_SWIGLU_CLAMP_FUSION=false
export SGLANG_USE_AITER_AG=0
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_LIGHTOP_GROUP_FP8_QUANT=1
export SGLANG_USE_DEEPGEMM_MOE=1
export SGLANG_USE_DPSKV4_LIGHTOP_QUANT_K_CACHE=1
export SGLANG_USE_DPSKV4_LIGHTOP_RMSNORM=1
export SGLANG_DSV4_SPLIT_PREFILL_DECODE_MLA=1
export SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256
export SGLANG_USE_LIGHTOP_PAGED_MQA_LOGITS_FP4=1
export SGLANG_USE_LIGHTOP_DSV41_PAGED_INDEXER=1
export NCCL_SIMPLE_CHANNELS=32
export HSA_USERPTR_FOR_PAGED_MEM=0
export SGLANG_HCU_FUSED_MHC_BOUNDARY=1
export NCCL_IB_HCA=mlx5_1
export ROCSHMEM_MAX_NUM_CONTEXTS=48
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export UCX_NET_DEVICES=mlx5_1
export MC_ALLOWED_IBV_DEVICES=mlx5_1
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1
export SGLANG_HCU_MEGA_MOE_RUNTIME=deep_gemm
export SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=4096
export SGLANG_HCU_DSV41_BF16_MAIN_PROJ_PREPACK=1

sglang serve \
 --model-path hygon/DeepSeek-V4.1-Flash-1017-Channel-FP8-w8a8 \
 --tp-size 8 \
 --host 0.0.0.0 \
 --port 10015 \
 --trust-remote-code \
 --dist-timeout 10000 \
 --watchdog-timeout 3600 \
 --cuda-graph-backend-prefill disabled \
 --skip-server-warmup \
 --mem-fraction-static 0.80 \
 --max-running-requests 64 \
 --cuda-graph-max-bs-decode 32 \
 --reasoning-parser auto \
 --tool-call-parser auto \
 --chunked-prefill-size 32768 \
 --dist-init-addr "<P_node0_ip>:6144" \
 --speculative-algorithm DSPARK \
 --speculative-dspark-block-size 5 \
 --enable-prefill-cp \
 --cp-strategy interleave \
 --dp 1 \
 --attn-cp-size 8 \
 --enable-decoder-swa-bounded-replay \
 --enable-cp-cache-layer-split \
 --moe-a2a-backend megamoe \
 --max-prefill-tokens 32768 \
 --speculative-moe-a2a-backend megamoe \
 --disaggregation-ib-device mlx5_1 \
 --disaggregation-mode prefill \
 --disaggregation-transfer-backend mooncake \
 --disaggregation-bootstrap-port 8998 \
 --disable-radix-cache \
 --init-expert-location ./C-r32-ht.pt \
 --ep-num-redundant-experts 32 \
 --ep-dispatch-algorithm lp \
 2>&1 | tee running_dpsk-v41_nmz2_prefill.log
```

#### prefill(l3直通）- CP8EP8

##### mooncake-topo-custom.json

```json
{
  "cpu:0": [
    ["mlx5_4"],
    ["mlx5_5", "mlx5_2", "mlx5_3", "mlx5_7", "mlx5_9", "mlx5_6", "mlx5_8"]
  ],
  "cpu:1": [
    ["mlx5_2"],
    ["mlx5_3", "mlx5_5", "mlx5_4", "mlx5_8", "mlx5_6", "mlx5_7", "mlx5_9"]
  ],
  "cpu:2": [
    ["mlx5_5"],
    ["mlx5_4", "mlx5_2", "mlx5_3", "mlx5_7", "mlx5_9", "mlx5_6", "mlx5_8"]
  ],
  "cpu:3": [
    ["mlx5_3"],
    ["mlx5_2", "mlx5_4", "mlx5_5", "mlx5_7", "mlx5_8", "mlx5_9", "mlx5_6"]
  ],
  "cpu:4": [
    ["mlx5_9"],
    ["mlx5_8", "mlx5_7", "mlx5_6", "mlx5_4", "mlx5_5", "mlx5_2", "mlx5_3"]
  ],
  "cpu:5": [
    ["mlx5_7"],
    ["mlx5_6", "mlx5_9", "mlx5_8", "mlx5_2", "mlx5_3", "mlx5_5", "mlx5_4"]
  ],
  "cpu:6": [
    ["mlx5_8"],
    ["mlx5_9", "mlx5_7", "mlx5_6", "mlx5_3", "mlx5_5", "mlx5_4", "mlx5_2"]
  ],
  "cpu:7": [
    ["mlx5_6"],
    ["mlx5_7", "mlx5_9", "mlx5_8", "mlx5_5", "mlx5_2", "mlx5_4", "mlx5_3"]
  ],
  "hip:0": [
    ["mlx5_3"],
    ["mlx5_4", "mlx5_8", "mlx5_9", "mlx5_6", "mlx5_2", "mlx5_5", "mlx5_7"]
  ],
  "hip:1": [
    ["mlx5_5"],
    ["mlx5_9", "mlx5_2", "mlx5_8", "mlx5_3", "mlx5_4", "mlx5_7", "mlx5_6"]
  ],
  "hip:2": [
    ["mlx5_7"],
    ["mlx5_8", "mlx5_9", "mlx5_5", "mlx5_4", "mlx5_2", "mlx5_3", "mlx5_6"]
  ],
  "hip:3": [
    ["mlx5_4"],
    ["mlx5_5", "mlx5_2", "mlx5_7", "mlx5_9", "mlx5_8", "mlx5_3", "mlx5_6"]
  ],
  "hip:4": [
    ["mlx5_6"],
    ["mlx5_4", "mlx5_9", "mlx5_2", "mlx5_7", "mlx5_3", "mlx5_8", "mlx5_5"]
  ],
  "hip:5": [
    ["mlx5_8"],
    ["mlx5_6", "mlx5_7", "mlx5_4", "mlx5_3", "mlx5_9", "mlx5_2", "mlx5_5"]
  ],
  "hip:6": [
    ["mlx5_2"],
    ["mlx5_8", "mlx5_6", "mlx5_7", "mlx5_5", "mlx5_9", "mlx5_3", "mlx5_4"]
  ],
  "hip:7": [
    ["mlx5_9"],
    ["mlx5_6", "mlx5_8", "mlx5_2", "mlx5_5", "mlx5_7", "mlx5_3", "mlx5_4"]
  ]
}

```

##### mooncake_master.sh

在 Master 节点运行。将 `<master_ip>` 替换为该节点供其他节点访问的 IP。

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export NODE_IP="${NODE_IP:-<master_ip>}"
export MASTER_BIND_HOST="${MASTER_BIND_HOST:-$NODE_IP}"
export MASTER_PORT="${MASTER_PORT:-50156}"
export MASTER_METRICS_PORT="${MASTER_METRICS_PORT:-9014}"
export METADATA_BIND_HOST="${METADATA_BIND_HOST:-$NODE_IP}"
export METADATA_PORT="${METADATA_PORT:-8095}"
export CLIENT_TTL="${CLIENT_TTL:-60}"
export RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$SCRIPT_DIR/log/master"

command -v mooncake_master >/dev/null 2>&1 || {
  echo "mooncake_master was not found in PATH" >&2
  exit 1
}

exec mooncake_master \
  --rpc_address="$MASTER_BIND_HOST" \
  --rpc_port="$MASTER_PORT" \
  --metrics_port="$MASTER_METRICS_PORT" \
  --enable_http_metadata_server=true \
  --http_metadata_server_host="$METADATA_BIND_HOST" \
  --http_metadata_server_port="$METADATA_PORT" \
  --client_ttl="$CLIENT_TTL" \
  --logtostderr=true \
  --minloglevel=0 \
  2>&1 | tee "$SCRIPT_DIR/log/master/master_${RUN_TS}.log"

```

##### mooncake_store.sh

将 `<store_ip>` 替换为 Store 节点的 IP，并填写相同的 `<master_ip>`。

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export NODE_IP="${NODE_IP:-<store_ip>}"
export MASTER_HOST="${MASTER_HOST:-<master_ip>}"
export MASTER_PORT="${MASTER_PORT:-50156}"
export METADATA_PORT="${METADATA_PORT:-8095}"

export STORE_BIND_HOST="${STORE_BIND_HOST:-$NODE_IP}"
export STORE_PORT="${STORE_PORT:-50052}"
export STORE_HTTP_PORT="${STORE_HTTP_PORT:-9300}"
export STORE_GLOBAL_SEGMENT_SIZE="${STORE_GLOBAL_SEGMENT_SIZE:-100gb}" # 容量大小,根据节点实际容量大小设置
export MOONCAKE_PROTOCOL="${MOONCAKE_PROTOCOL:-rdma}"
export RDMA_DEVICES="${RDMA_DEVICES:-shca_0,shca_1,shca_2,shca_3}"
export MC_ALLOWED_IBV_DEVICES="${MC_ALLOWED_IBV_DEVICES:-$RDMA_DEVICES}"
export MC_ENABLE_DEST_DEVICE_AFFINITY="${MC_ENABLE_DEST_DEVICE_AFFINITY:-1}"
export MC_IB_GID_INDEX="${MC_IB_GID_INDEX:-0}"
export MC_IB_PCI_RELAXED_ORDERING="${MC_IB_PCI_RELAXED_ORDERING:-0}"
export RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
export MC_MS_AUTO_DISC=1
export MC_CUSTOM_TOPO_JSON=/mooncake-topo-custom.json

mkdir -p "$SCRIPT_DIR/log/store/p0"

command -v mooncake_client >/dev/null 2>&1 || {
  echo "mooncake_client was not found in PATH" >&2
  exit 1
}

exec env -u MC_FORCE_TCP -u LD_PRELOAD mooncake_client \
  --host="$STORE_BIND_HOST" \
  --port="$STORE_PORT" \
  --master_server_address="$MASTER_HOST:$MASTER_PORT" \
  --metadata_server="http://$MASTER_HOST:$METADATA_PORT/metadata" \
  --protocol="$MOONCAKE_PROTOCOL" \
  --device_names="$RDMA_DEVICES" \
  --global_segment_size="$STORE_GLOBAL_SEGMENT_SIZE" \
  --enable_http_server=true \
  --http_port="$STORE_HTTP_PORT" \
  2>&1 | tee "$SCRIPT_DIR/log/store/store_${RUN_TS}.log"

```

##### prefill.sh

`<P_l3_ip>` 填写 Prefill 节点用于 Mooncake 通信的本机地址，`<master_ip>` 填写 Mooncake Master 的地址。

```bash
export NCCL_SOCKET_IFNAME=ens66f1np1
export GLOO_SOCKET_IFNAME=ens66f1np1
export SGLANG_TORCH_PROFILER_DIR=/home/proj_dpsk-v4/profile
export SGLANG_OPT_SWIGLU_CLAMP_FUSION=false
export SGLANG_USE_AITER_AG=0
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_LIGHTOP_GROUP_FP8_QUANT=1
export SGLANG_USE_DEEPGEMM_MOE=1
export SGLANG_USE_DPSKV4_LIGHTOP_QUANT_K_CACHE=1
export SGLANG_USE_DPSKV4_LIGHTOP_RMSNORM=1
export SGLANG_DSV4_SPLIT_PREFILL_DECODE_MLA=1
export SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256
export SGLANG_USE_LIGHTOP_PAGED_MQA_LOGITS_FP4=1
export SGLANG_USE_LIGHTOP_DSV41_PAGED_INDEXER=1
export NCCL_SIMPLE_CHANNELS=32
export HSA_USERPTR_FOR_PAGED_MEM=0
export SGLANG_HCU_FUSED_MHC_BOUNDARY=1
export NCCL_IB_HCA=mlx5_3
export ROCSHMEM_MAX_NUM_CONTEXTS=48
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export UCX_NET_DEVICES=mlx5_3
export MC_ALLOWED_IBV_DEVICES=mlx5_3
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1
export SGLANG_HCU_MEGA_MOE_RUNTIME=deep_gemm
export SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=4096
export EPLB_LOCATION=/eplb_fp8_runB64_r32.pt
export EPLB_REDUNDANT=32
export SGLANG_LPLB_HIP_SOLVER=device
export EPLB_DISPATCH=lp

export MOONCAKE_LOCAL_HOSTNAME="<P_l3_ip>"
export MOONCAKE_TE_META_DATA_SERVER="http://<master_ip>:8095/metadata"
export MOONCAKE_MASTER="<master_ip>:50156"
export MOONCAKE_PROTOCOL=rdma
export MOONCAKE_GLOBAL_SEGMENT_SIZE=0GB
export MOONCAKE_CHECK_SERVER=0
export SGLANG_KV_LAYOUT_HCU_FA=0
export MOONCAKE_STANDALONE_STORAGE=0

sglang serve \
 --model-path hygon/DeepSeek-V4.1-Flash-1017-Channel-FP8-w8a8 \
 --tp-size 8 \
 --host "<P_l3_ip>" \
 --port 10015 \
 --trust-remote-code \
 --dist-timeout 10000 \
 --watchdog-timeout 3600 \
 --cuda-graph-backend-prefill disabled \
 --skip-server-warmup \
 --mem-fraction-static 0.80 \
 --max-running-requests 32 \
 --cuda-graph-max-bs-decode 32 \
 --reasoning-parser auto \
 --tool-call-parser auto \
 --chunked-prefill-size 32768 \
 --dist-init-addr "<P_node0_ip>:6144" \
 --speculative-algorithm DSPARK \
 --speculative-dspark-block-size 5 \
 --enable-prefill-cp \
 --cp-strategy interleave \
 --dp 1 \
 --attn-cp-size 8 \
 --enable-decoder-swa-bounded-replay \
 --enable-cp-cache-layer-split \
 --moe-a2a-backend megamoe \
 --speculative-moe-a2a-backend megamoe \
 --disaggregation-ib-device mlx5_3 \
 --disaggregation-mode prefill \
 --disaggregation-transfer-backend mooncake \
 --disaggregation-bootstrap-port 8998 \
 --disable-radix-cache \
 --init-expert-location eplb_fp8_runB64_r32.pt \
 --ep-num-redundant-experts 32 \
 --ep-dispatch-algorithm lp \
 --enable-unified-cache-external-linker \
 2>&1 | tee running_dpsk-v41_nmz20_prefill.log
```

#### D node DP8EP8

```bash
export NCCL_SOCKET_IFNAME=ens66f1np1
export GLOO_SOCKET_IFNAME=ens66f1np1
export SGLANG_TORCH_PROFILER_DIR=/home/proj_dpsk-v4/profile
export SGLANG_OPT_SWIGLU_CLAMP_FUSION=false
export SGLANG_USE_AITER_AG=0
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_LIGHTOP_GROUP_FP8_QUANT=1
export SGLANG_USE_DEEPGEMM_MOE=1
export SGLANG_USE_DPSKV4_LIGHTOP_QUANT_K_CACHE=1
export SGLANG_USE_DPSKV4_LIGHTOP_RMSNORM=1
export SGLANG_DSV4_SPLIT_PREFILL_DECODE_MLA=1
export SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256
export SGLANG_USE_LIGHTOP_PAGED_MQA_LOGITS_FP4=1
export SGLANG_USE_LIGHTOP_DSV41_PAGED_INDEXER=1
export NCCL_SIMPLE_CHANNELS=32
export NCCL_IB_HCA=mlx5_2
export ROCSHMEM_MAX_NUM_CONTEXTS=48
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export UCX_NET_DEVICES=mlx5_2
export MC_ALLOWED_IBV_DEVICES=mlx5_2
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1
export SGLANG_USE_FP8_W8A8_MOE=1
export DEEPEP_ENABLE_LL_LAYERED_OPT=1
export ROCSHMEM_DISABLE_HDP_FLUSH=1
export SGLANG_HCU_FUSED_MHC_BOUNDARY=1
export SGLANG_EXPERIMENTAL_LPLB_STATIC_PROBS=/lplb_decode_ep8_r32.pt

sglang serve \
 --model-path hygon/DeepSeek-V4.1-Flash-1017-Channel-FP8-w8a8 \
 --tp-size 8 \
 --host 0.0.0.0 \
 --port 10016 \
 --trust-remote-code \
 --dist-timeout 10000 \
 --watchdog-timeout 3600 \
 --cuda-graph-backend-prefill disabled \
 --skip-server-warmup \
 --mem-fraction-static 0.85 \
 --max-running-requests 32 \
 --cuda-graph-max-bs-decode 32 \
 --speculative-algorithm DSPARK \
 --speculative-dspark-block-size 5 \
 --reasoning-parser auto \
 --tool-call-parser auto \
 --chunked-prefill-size 32768 \
 --dist-init-addr "<D_node0_ip>:6244" \
 --dp 8 \
 --enable-dp-attention \
 --enable-dp-attention-local-control-broadcast \
 --enable-dp-lm-head \
 --ep 8 \
 --moe-a2a-backend deepep \
 --deepep-config /deepep_IntraConfig.json \
 --deepep-mode auto \
 --moe-runner-backend deep_gemm \
 --enable-single-batch-overlap \
 --speculative-moe-a2a-backend deepep \
 --speculative-moe-runner-backend deep_gemm \
 --disaggregation-ib-device mlx5_2 \
 --init-expert-location /eplb_decode_ep8_static_20261102.pt \
 --ep-num-redundant-experts 32 \
 --ep-dispatch-algorithm lp \
 --disaggregation-mode decode \
 --disaggregation-transfer-backend mooncake \
 --disaggregation-bootstrap-port 8998 \
 --disable-radix-cache \
 2>&1 | tee running_dpsk-v41_nmz22_decode.log
```

#### Decode - DP16EP16

```bash
export NCCL_SOCKET_IFNAME=enp28s0f1u1
export GLOO_SOCKET_IFNAME=enp28s0f1u1
export SGLANG_OPT_SWIGLU_CLAMP_FUSION=false
export SGLANG_USE_AITER_AG=0
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_LIGHTOP_GROUP_FP8_QUANT=1
export SGLANG_USE_DEEPGEMM_MOE=1
export SGLANG_USE_DPSKV4_LIGHTOP_QUANT_K_CACHE=1
export SGLANG_USE_DPSKV4_LIGHTOP_RMSNORM=1
export SGLANG_DSV4_SPLIT_PREFILL_DECODE_MLA=1
export SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256
export SGLANG_USE_LIGHTOP_PAGED_MQA_LOGITS_FP4=1
export SGLANG_USE_LIGHTOP_DSV41_PAGED_INDEXER=1
export NCCL_SIMPLE_CHANNELS=32
export ROCSHMEM_MAX_NUM_CONTEXTS=48
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1
export SGLANG_USE_FP8_W8A8_MOE=1
export DEEPEP_ENABLE_LL_LAYERED_OPT=0
export ROCSHMEM_DISABLE_HDP_FLUSH=1
export HSA_USERPTR_FOR_PAGED_MEM=0
export SGLANG_HCU_FUSED_MHC_BOUNDARY=1
export SGLANG_EXPERIMENTAL_LPLB_STATIC_PROBS=/lplb_decode_ep16_r32.pt
exec sglang serve \
 --model-path hygon/DeepSeek-V4.1-Flash-1017-Channel-FP8-w8a8 \
 --tp-size 16 \
 --host 0.0.0.0 \
 --port 10016 \
 --trust-remote-code \
 --dist-timeout 10000 \
 --watchdog-timeout 3600 \
 --cuda-graph-backend-prefill disabled \
 --skip-server-warmup \
 --mem-fraction-static 0.85 \
 --max-running-requests 32 \
 --cuda-graph-max-bs-decode 32 \
 --speculative-algorithm DSPARK \
 --speculative-dspark-block-size 5 \
 --reasoning-parser auto \
 --tool-call-parser auto \
 --chunked-prefill-size 32768 \
 --dist-init-addr "<D_node0_ip>:6244" \
 --dp 16 \
 --enable-dp-attention \
 --enable-dp-attention-local-control-broadcast \
 --enable-dp-lm-head \
 --ep 16 \
 --moe-a2a-backend deepep \
 --deepep-mode low_latency \
 --moe-runner-backend deep_gemm \
 --enable-single-batch-overlap \
 --speculative-moe-a2a-backend deepep \
 --speculative-moe-runner-backend deep_gemm \
 --disaggregation-mode decode \
 --disaggregation-transfer-backend mooncake \
 --disaggregation-bootstrap-port 8998 \
 --init-expert-location /home/eplb_decode_ep16_r32.pt \
 --ep-num-redundant-experts 32 \
 --ep-dispatch-algorithm lp \
 --disable-radix-cache \
 2>&1 | tee running_dpsk-v41_dp16_decode.log
```

### DeepSeek-V4.1-Flash-1017-Channel-FP8-w8a8 1P1D scalex40-3G 40x SGLang 0.5.18

#### P node

Prefill 使用两个节点，每节点 4 卡（TP8 / CP8 / DP1）。两节点均填写 `<P_node0_ip>`，并分别将 `--node-rank` 设为 `0` 和 `1`。

```bash
#!/usr/bin/env bash
set -euo pipefail

export LIGHTOP_PREFILL_SPARSE_PREP_FUSION=3
export NCCL_SOCKET_IFNAME=ibp129s0
export GLOO_SOCKET_IFNAME=ibp129s0
export SGLANG_TORCH_PROFILER_DIR=/home/profile/prefill
export SGLANG_OPT_SWIGLU_CLAMP_FUSION=false
export SGLANG_USE_AITER_AG=0
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_LIGHTOP_GROUP_FP8_QUANT=1
export SGLANG_USE_DEEPGEMM_MOE=1
export SGLANG_USE_DPSKV4_LIGHTOP_QUANT_K_CACHE=1
export SGLANG_USE_DPSKV4_LIGHTOP_RMSNORM=1
export SGLANG_DSV4_SPLIT_PREFILL_DECODE_MLA=1
export SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=384
export SGLANG_USE_LIGHTOP_PAGED_MQA_LOGITS_FP4=1
export SGLANG_USE_LIGHTOP_DSV41_PAGED_INDEXER=1
export NCCL_SIMPLE_CHANNELS=32
export NCCL_IB_HCA=shca_0,shca_1,shca_2,shca_3,shca_4
export ROCSHMEM_MAX_NUM_CONTEXTS=48
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export UCX_NET_DEVICES=shca_0,shca_1,shca_2,shca_3,shca_4
export MC_ALLOWED_IBV_DEVICES=shca_0,shca_1,shca_2,shca_3,shca_4
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1
export SGLANG_USE_FP8_W8A8_MOE=1
export DEEPEP_ENABLE_LL_LAYERED_OPT=1
export ROCSHMEM_DISABLE_HDP_FLUSH=1
export SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1
export SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=per_rank
export SGLANG_OPT_BF16_FP32_GEMM_ALGO=lightop
export DEEP_EP_NORMAL_MNVL=1
export ROCSHMEM_IPC_MNVL=1
export HIP_BUFFER_EXTRA_SIZE=0
export ROCSHMEM_GDR_DISABLE_XDP=1
export DEEPEP_CPU_TIMEOUT_SECS=300
export SGLANG_PROFILE_WITH_STACK=0
export SGLANG_PROFILE_RECORD_SHAPES=0
export NCCL_IB_DISABLE=0
export HIP_GRAPH_ACCUMULATE_DISPATCH=1
export HIP_GRAPH_USE_CMD_CACHE=0

exec sglang serve \
    --model-path hygon/DeepSeek-V4.1-Flash-1017-Channel-FP8-w8a8/ \
    --tp-size 8 \
    --host 0.0.0.0 \
    --port 10015 \
    --trust-remote-code \
    --dist-timeout 10000 \
    --watchdog-timeout 3600 \
    --cuda-graph-backend-prefill disabled \
    --skip-server-warmup \
    --mem-fraction-static 0.85 \
    --max-running-requests 32 \
    --cuda-graph-max-bs-decode 32 \
    --speculative-algorithm DSPARK \
    --speculative-dspark-block-size 5 \
    --reasoning-parser auto \
    --tool-call-parser auto \
    --chunked-prefill-size 32768 \
    --dist-init-addr "<P_node0_ip>:6144" \
    --nnodes 2 \
    --node-rank 0 \
    --enable-prefill-cp \
    --cp-strategy interleave \
    --dp 1 \
    --attn-cp-size 8 \
    --enable-cp-cache-layer-split \
    --enable-decoder-swa-bounded-replay \
    --moe-a2a-backend deepep \
    --deepep-config /deepep_IntraConfig.json \
    --deepep-mode auto \
    --moe-runner-backend deep_gemm \
    --enable-single-batch-overlap \
    --speculative-moe-a2a-backend none \
    --speculative-moe-runner-backend triton \
    --disaggregation-ib-device shca_0,shca_1,shca_2,shca_3,shca_4 \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend mooncake \
    --disaggregation-bootstrap-port 8998 \
    --disable-radix-cache \
    --numa-node 0 5 3 2
```

#### D node

在各节点上运行 `host_ip="<D_node0_ip>" nodes=8 rank=0 bash ep32.sh`，所有节点填写同一个 `<D_node0_ip>`，并分别将 `rank` 设为 `0` 到 `7`。

```bash
#!/usr/bin/env bash

export NCCL_SOCKET_IFNAME=ibp129s0
export GLOO_SOCKET_IFNAME=ibp129s0
export SGLANG_TORCH_PROFILER_DIR=/home/proj_dpsk-v4/profile
export SGLANG_OPT_SWIGLU_CLAMP_FUSION=false
export SGLANG_USE_AITER_AG=0
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_LIGHTOP_GROUP_FP8_QUANT=1
export SGLANG_USE_DEEPGEMM_MOE=1
export SGLANG_USE_DPSKV4_LIGHTOP_QUANT_K_CACHE=1
export SGLANG_USE_DPSKV4_LIGHTOP_RMSNORM=1
export SGLANG_DSV4_SPLIT_PREFILL_DECODE_MLA=1
export SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=1
export SGLANG_USE_LIGHTOP_PAGED_MQA_LOGITS_FP4=1
export SGLANG_USE_LIGHTOP_DSV41_PAGED_INDEXER=1
export NCCL_SIMPLE_CHANNELS=32
export NCCL_IB_HCA=shca_0,shca_1,shca_2,shca_3,shca_4
export ROCSHMEM_MAX_NUM_CONTEXTS=48
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export UCX_NET_DEVICES=shca_0,shca_1,shca_2,shca_3,shca_4
export MC_ALLOWED_IBV_DEVICES=shca_0,shca_1,shca_2,shca_3,shca_4
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1
export SGLANG_USE_FP8_W8A8_MOE=1
export DEEPEP_ENABLE_LL_LAYERED_OPT=1
export ROCSHMEM_DISABLE_HDP_FLUSH=1

export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=96

export NCCL_IB_DISABLE=0
export HIP_GRAPH_ACCUMULATE_DISPATCH=1
export HIP_GRAPH_USE_CMD_CACHE=0

export ROCSHMEM_IPC_MNVL=1
export HIP_BUFFER_EXTRA_SIZE=0
export ROCSHMEM_GDR_DISABLE_XDP=1
export DEEPEP_CPU_TIMEOUT_SECS=300
export SGLANG_SIMULATE_ACC_LEN=3.25
export SGLANG_SIMULATE_ACC_METHOD=match-expected
export SGLANG_HCU_FUSED_MHC_BOUNDARY=1
export DEEPEP_ENABLE_LL_LAYERED_OPT=1

mkdir -p ./log
exec sglang serve \
  --model-path hygon/DeepSeek-V4.1-Flash-1017-Channel-FP8-w8a8/ \
  --tp-size 32 \
  --host 0.0.0.0 \
  --port 10016 \
  --trust-remote-code \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --cuda-graph-backend-prefill disabled \
  --skip-server-warmup \
  --mem-fraction-static 0.85 \
  --max-running-requests 1024 \
  --cuda-graph-max-bs-decode 16 \
  --speculative-algorithm DSPARK \
  --speculative-dspark-block-size 3 \
  --reasoning-parser auto \
  --tool-call-parser auto \
  --chunked-prefill-size 32768 \
  --dist-init-addr "${host_ip}:6244" \
  --nnodes "${nodes}" --node-rank "${rank}" \
  --dp 32 \
  --enable-dp-attention \
  --enable-dp-attention-local-control-broadcast \
  --enable-dp-lm-head \
  --ep 32 \
  --moe-a2a-backend deepep \
  --deepep-config ./deepep_IntraConfig.json \
  --deepep-mode low_latency \
  --moe-runner-backend deep_gemm \
  --enable-single-batch-overlap \
  --speculative-moe-a2a-backend deepep \
  --speculative-moe-runner-backend deep_gemm \
  --disaggregation-ib-device shca_0,shca_1,shca_2,shca_3,shca_4 \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-bootstrap-port 8998 \
  --disable-radix-cache \
  --init-expert-location /selected.pt \
  --ep-dispatch-algorithm static \
  --ep-num-redundant-experts 32 \
  --eplb-algorithm auto \
  --numa-node 0 5 3 2 \
  --tokenizer-worker-num 4 \
  > "./log/node_${rank}_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S).log" 2>&1
```

### DeepSeek-V4.1-Flash-1017-Channel-INT4-w4a8 IFB BW1000 8x SGLang 0.5.18

```bash
#!/bin/bash

set -o pipefail
ulimit -c 0
export SGLANG_OPT_SWIGLU_CLAMP_FUSION=false
export SGLANG_USE_AITER=0
export SGLANG_USE_AITER_AG=0
export SGLANG_ROCM_USE_AITER_MOE=0
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_LIGHTOP_GROUP_FP8_QUANT=1
export SGLANG_USE_DPSKV4_LIGHTOP_QUANT_K_CACHE=1
export SGLANG_USE_DPSKV4_LIGHTOP_RMSNORM=1
export SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1
export SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=per_rank
export SGLANG_HCU_ENABLE_DSV41_W4A8_ENGRAM_LOAD_OPTIMIZATION=1
export SGLANG_DSV4_SPLIT_PREFILL_DECODE_MLA=1
export SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=1
export SGLANG_HCU_OPT_ENGRAM_GATE=1
export SGLANG_USE_W4A8_MASKED_HIPC=1
export SGLANG_USE_W4A8_CONTIGUOUS_HIPC=1
export LIGHTOP_PREFILL_SPARSE_PREP_FUSION=3
export SGLANG_DSV4_HC_BF16X3_MIN_ROWS=4096
export SGLANG_DSV41_PAGED_MQA_LOGITS_FREE_MEM_FRACTION=0.05
export SGLANG_KVALLOC_KERNEL=0
export HSA_USERPTR_FOR_PAGED_MEM=0
export SGLANG_HCU_FUSED_MHC_BOUNDARY=1

readonly PORT=${1:-30000}
readonly MODEL_PATH=${2:-/hygon/DeepSeek-V4.1-Flash-1017-Channel-INT4-w4a8}
readonly CPU_OFFLOAD_GB=${CPU_OFFLOAD_GB:-0}
readonly DIST_TIMEOUT=${DIST_TIMEOUT:-1800}
readonly MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-64}

sglang serve \
    --model-path "$MODEL_PATH" \
    --tp-size 8 \
    --dp-size 8 \
    --ep-size 8 \
    --enable-dp-attention \
    --enable-dp-lm-head \
    --host "<node_ip>" \
    --port "$PORT" \
    --trust-remote-code \
    --disable-radix-cache \
    --mem-fraction-static 0.89 \
    --cpu-offload-gb "$CPU_OFFLOAD_GB" \
    --dist-timeout "$DIST_TIMEOUT" \
    --max-running-requests "$MAX_RUNNING_REQUESTS" \
    --cuda-graph-backend-prefill disabled \
    --moe-runner-backend deep_gemm \
    --moe-a2a-backend deepep \
    --deepep-mode auto \
    --cuda-graph-max-bs-decode 64 \
    --speculative-algorithm DSPARK \
    --speculative-dspark-block-size 5 \
    --speculative-moe-a2a-backend deepep \
    --speculative-moe-runner-backend deep_gemm \
    2>&1 | tee "running_dpsk-v41_DP8EP8_${the_host}.log"
```

### DeepSeek-V4.1-Flash-1017-Channel-INT4-w4a8 1P1D BW1000 16x SGLang 0.5.18

#### P node

```bash
#!/bin/bash

set -o pipefail
ulimit -c 0

export ROCBLAS_TENSILE_LIBPATH=/opt/dtk/gemm_asm_gfx936_80cu_20261111/rocblas/library
export HIPBLASLT_TENSILE_LIBPATH=/opt/dtk/gemm_asm_gfx936_80cu_20261111/hipblaslt/library
export HIPBLASLT_TUNING_OVERRIDE_FILE=/opt/dtk/gemm_asm_gfx936_80cu_20261111/hipblaslt/tuning.txt
export SGLANG_DSV41_INT8_BLASLT_ALGO_CONFIG=/opt/dtk/gemm_asm_gfx936_80cu_20261111/wo_a_algo.json

export SGLANG_OPT_SWIGLU_CLAMP_FUSION=false
export SGLANG_USE_AITER=0
export SGLANG_USE_AITER_AG=0
export SGLANG_ROCM_USE_AITER_MOE=0
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_LIGHTOP_GROUP_FP8_QUANT=1
export SGLANG_USE_DPSKV4_LIGHTOP_QUANT_K_CACHE=1
export SGLANG_USE_DPSKV4_LIGHTOP_RMSNORM=1
export SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1
export SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=per_rank
export SGLANG_HCU_ENABLE_DSV41_W4A8_ENGRAM_LOAD_OPTIMIZATION=1
export SGLANG_DSV4_SPLIT_PREFILL_DECODE_MLA=1
export SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=1
export SGLANG_USE_W4A8_CONTIGUOUS_HIPC=1
export LIGHTOP_PREFILL_SPARSE_PREP_FUSION=3
export SGLANG_DSV4_HC_BF16X3_MIN_ROWS=4096
export SGLANG_HCU_MEGA_MOE_RUNTIME=megamoe
export SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=4096
export SGLANG_DSV41_PAGED_MQA_LOGITS_FREE_MEM_FRACTION=0.05
export SGLANG_KVALLOC_KERNEL=0
export HSA_USERPTR_FOR_PAGED_MEM=0
export SGLANG_HCU_DSV41_BF16_MAIN_PROJ_PREPACK=1

export NCCL_NET_PLUGIN=shca
export NCCL_IB_HCA=shca_0,shca_1,shca_2,shca_3
export ROCSHMEM_MAX_NUM_CONTEXTS=48
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export UCX_NET_DEVICES=shca_0,shca_1,shca_2,shca_3
export MC_ALLOWED_IBV_DEVICES=shca_0,shca_1,shca_2,shca_3
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1

readonly PORT=${1:-30000}
readonly MODEL_PATH=${2:-/hygon/DeepSeek-V4.1-Flash-1017-Channel-INT4-w4a8}
readonly CPU_OFFLOAD_GB=${CPU_OFFLOAD_GB:-0}
readonly DIST_TIMEOUT=${DIST_TIMEOUT:-1800}

the_host=$(hostname)

sglang serve \
    --model-path "$MODEL_PATH" \
    --tp-size 8 \
    --ep-size 8 \
    --enable-prefill-cp \
    --attn-cp-size 8 \
    --reasoning-parser auto \
    --tool-call-parser auto \
    --cp-strategy interleave \
    --enable-cp-cache-layer-split \
    --enable-decoder-swa-bounded-replay \
    --ep-num-redundant-experts 0 \
    --init-expert-location /prefill_static_load.pt \
    --host "<P_node_ip>" \
    --port "$PORT" \
    --trust-remote-code \
    --cuda-graph-backend-prefill disabled \
    --enable-cache-report \
    --mem-fraction-static 0.85 \
    --cpu-offload-gb "$CPU_OFFLOAD_GB" \
    --dist-timeout "$DIST_TIMEOUT" \
    --max-running-requests 32 \
    --deepep-mode normal \
    --moe-a2a-backend megamoe \
    --disaggregation-ib-device shca_0,shca_1,shca_2,shca_3 \
    --chunked-prefill-size 32768 \
    --max-prefill-tokens 32768 \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend mooncake \
    --speculative-algorithm DSPARK \
    --speculative-dspark-block-size 5 \
    --speculative-moe-a2a-backend deepep \
    --speculative-moe-runner-backend deep_gemm \
    2>&1 | tee "running_dpsk-v41_PD_P_CP8EP8_${the_host}.log"
```

#### Prefill(l3直通)

##### mooncake-topo-custom.json

```json
{
  "cpu:0": [
    [
      "shca_0"
    ],
    [
      "shca_1",
      "shca_2",
      "shca_3"
    ]
  ],
  "cpu:1": [
    [
      "shca_1"
    ],
    [
      "shca_0",
      "shca_2",
      "shca_3"
    ]
  ],
  "cpu:2": [
    [
      "shca_0"
    ],
    [
      "shca_1",
      "shca_3",
      "shca_2"
    ]
  ],
  "cpu:3": [
    [
      "shca_0"
    ],
    [
      "shca_1",
      "shca_3",
      "shca_2"
    ]
  ],
  "cpu:4": [
    [
      "shca_2"
    ],
    [
      "shca_3",
      "shca_0",
      "shca_1"
    ]
  ],
  "cpu:5": [
    [
      "shca_3"
    ],
    [
      "shca_2",
      "shca_0",
      "shca_1"
    ]
  ],
  "cpu:6": [
    [
      "shca_2"
    ],
    [
      "shca_3",
      "shca_1",
      "shca_0"
    ]
  ],
  "cpu:7": [
    [
      "shca_2"
    ],
    [
      "shca_3",
      "shca_0",
      "shca_1"
    ]
  ],
  "hip:0": [
    [
      "shca_0"
    ],
    [
      "shca_2",
      "shca_1",
      "shca_3"
    ]
  ],
  "hip:1": [
    [
      "shca_1"
    ],
    [
      "shca_0",
      "shca_2",
      "shca_3"
    ]
  ],
  "hip:2": [
    [
      "shca_0"
    ],
    [
      "shca_1",
      "shca_3",
      "shca_2"
    ]
  ],
  "hip:3": [
    [
      "shca_1"
    ],
    [
      "shca_3",
      "shca_0",
      "shca_2"
    ]
  ],
  "hip:4": [
    [
      "shca_2"
    ],
    [
      "shca_0",
      "shca_3",
      "shca_1"
    ]
  ],
  "hip:5": [
    [
      "shca_3"
    ],
    [
      "shca_2",
      "shca_0",
      "shca_1"
    ]
  ],
  "hip:6": [
    [
      "shca_2"
    ],
    [
      "shca_3",
      "shca_0",
      "shca_1"
    ]
  ],
  "hip:7": [
    [
      "shca_3"
    ],
    [
      "shca_1",
      "shca_2",
      "shca_0"
    ]
  ]
}

```

##### mooncake_master.sh

在 Master 节点运行。将 `<master_ip>` 替换为该节点供其他节点访问的 IP。

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export NODE_IP="${NODE_IP:-<master_ip>}"
export MASTER_BIND_HOST="${MASTER_BIND_HOST:-$NODE_IP}"
export MASTER_PORT="${MASTER_PORT:-50156}"
export MASTER_METRICS_PORT="${MASTER_METRICS_PORT:-9014}"
export METADATA_BIND_HOST="${METADATA_BIND_HOST:-$NODE_IP}"
export METADATA_PORT="${METADATA_PORT:-8095}"
export CLIENT_TTL="${CLIENT_TTL:-60}"
export RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$SCRIPT_DIR/log/master"

command -v mooncake_master >/dev/null 2>&1 || {
  echo "mooncake_master was not found in PATH" >&2
  exit 1
}

exec mooncake_master \
  --rpc_address="$MASTER_BIND_HOST" \
  --rpc_port="$MASTER_PORT" \
  --metrics_port="$MASTER_METRICS_PORT" \
  --enable_http_metadata_server=true \
  --http_metadata_server_host="$METADATA_BIND_HOST" \
  --http_metadata_server_port="$METADATA_PORT" \
  --client_ttl="$CLIENT_TTL" \
  --logtostderr=true \
  --minloglevel=0 \
  2>&1 | tee "$SCRIPT_DIR/log/master/master_${RUN_TS}.log"

```

##### mooncake_store.sh

将 `<store_ip>` 替换为 Store 节点的 IP，并填写相同的 `<master_ip>`。

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export NODE_IP="${NODE_IP:-<store_ip>}"
export MASTER_HOST="${MASTER_HOST:-<master_ip>}"
export MASTER_PORT="${MASTER_PORT:-50156}"
export METADATA_PORT="${METADATA_PORT:-8095}"

export STORE_BIND_HOST="${STORE_BIND_HOST:-$NODE_IP}"
export STORE_PORT="${STORE_PORT:-50052}"
export STORE_HTTP_PORT="${STORE_HTTP_PORT:-9300}"
export STORE_GLOBAL_SEGMENT_SIZE="${STORE_GLOBAL_SEGMENT_SIZE:-100gb}" # 容量大小,根据节点实际容量大小设置
export MOONCAKE_PROTOCOL="${MOONCAKE_PROTOCOL:-rdma}"
export RDMA_DEVICES="${RDMA_DEVICES:-shca_0,shca_1,shca_2,shca_3}"
export MC_ALLOWED_IBV_DEVICES="${MC_ALLOWED_IBV_DEVICES:-$RDMA_DEVICES}"
export MC_ENABLE_DEST_DEVICE_AFFINITY="${MC_ENABLE_DEST_DEVICE_AFFINITY:-1}"
export MC_IB_GID_INDEX="${MC_IB_GID_INDEX:-0}"
export MC_IB_PCI_RELAXED_ORDERING="${MC_IB_PCI_RELAXED_ORDERING:-0}"
export RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
export MC_MS_AUTO_DISC=1
export MC_CUSTOM_TOPO_JSON=/mooncake-topo-custom.json

mkdir -p "$SCRIPT_DIR/log/store/p0"

command -v mooncake_client >/dev/null 2>&1 || {
  echo "mooncake_client was not found in PATH" >&2
  exit 1
}

exec env -u MC_FORCE_TCP -u LD_PRELOAD mooncake_client \
  --host="$STORE_BIND_HOST" \
  --port="$STORE_PORT" \
  --master_server_address="$MASTER_HOST:$MASTER_PORT" \
  --metadata_server="http://$MASTER_HOST:$METADATA_PORT/metadata" \
  --protocol="$MOONCAKE_PROTOCOL" \
  --device_names="$RDMA_DEVICES" \
  --global_segment_size="$STORE_GLOBAL_SEGMENT_SIZE" \
  --enable_http_server=true \
  --http_port="$STORE_HTTP_PORT" \
  2>&1 | tee "$SCRIPT_DIR/log/store/store_${RUN_TS}.log"

```

##### prefill.sh

`<P_l3_ip>` 填写 Prefill 节点用于 Mooncake 通信的本机地址，`<master_ip>` 填写 Mooncake Master 的地址。

```bash
#!/bin/bash

set -o pipefail
ulimit -c 0
export SGLANG_OPT_SWIGLU_CLAMP_FUSION=false
export SGLANG_USE_AITER=0
export SGLANG_USE_AITER_AG=0
export SGLANG_ROCM_USE_AITER_MOE=0
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_LIGHTOP_GROUP_FP8_QUANT=1
export SGLANG_USE_DPSKV4_LIGHTOP_QUANT_K_CACHE=1
export SGLANG_USE_DPSKV4_LIGHTOP_RMSNORM=1
export SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1
export SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=per_rank
export SGLANG_HCU_ENABLE_DSV41_W4A8_ENGRAM_LOAD_OPTIMIZATION=1
export SGLANG_DSV4_SPLIT_PREFILL_DECODE_MLA=1
export SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=1
export SGLANG_USE_W4A8_CONTIGUOUS_HIPC=1
export LIGHTOP_PREFILL_SPARSE_PREP_FUSION=3
export SGLANG_HCU_MEGA_MOE_RUNTIME=megamoe
export SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=4096
export SGLANG_DSV41_PAGED_MQA_LOGITS_FREE_MEM_FRACTION=0.05
export SGLANG_KVALLOC_KERNEL=0
export HSA_USERPTR_FOR_PAGED_MEM=0

export NCCL_IB_HCA=shca_0,shca_1,shca_2,shca_3
export ROCSHMEM_MAX_NUM_CONTEXTS=48
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export UCX_NET_DEVICES=shca_0,shca_1,shca_2,shca_3
export MC_ALLOWED_IBV_DEVICES=shca_0,shca_1,shca_2,shca_3
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1

export MOONCAKE_LOCAL_HOSTNAME="<P_l3_ip>"
export MOONCAKE_TE_META_DATA_SERVER="http://<master_ip>:8095/metadata"
export MOONCAKE_MASTER="<master_ip>:50156"
export MOONCAKE_PROTOCOL=rdma
export MOONCAKE_GLOBAL_SEGMENT_SIZE=0GB
export MOONCAKE_CHECK_SERVER=0
export SGLANG_KV_LAYOUT_HCU_FA=0
export MOONCAKE_STANDALONE_STORAGE=0

readonly PORT=${1:-30000}
readonly MODEL_PATH=${2:-/hygon/DeepSeek-V4.1-Flash-1017-Channel-INT4-w4a8}
readonly CPU_OFFLOAD_GB=${CPU_OFFLOAD_GB:-0}
readonly DIST_TIMEOUT=${DIST_TIMEOUT:-1800}

the_host=$(hostname)

sglang serve \
    --model-path "$MODEL_PATH" \
    --tp-size 8 \
    --ep-size 8 \
    --enable-prefill-cp \
    --attn-cp-size 8 \
    --reasoning-parser auto \
    --tool-call-parser auto \
    --cp-strategy interleave \
    --enable-cp-cache-layer-split \
    --enable-decoder-swa-bounded-replay \
    --ep-num-redundant-experts 0 \
    --init-expert-location /prefill_static_load.pt \
    --host "<P_node_ip>" \
    --port "$PORT" \
    --trust-remote-code \
    --cuda-graph-backend-prefill disabled \
    --disable-radix-cache \
    --mem-fraction-static 0.85 \
    --cpu-offload-gb "$CPU_OFFLOAD_GB" \
    --dist-timeout "$DIST_TIMEOUT" \
    --max-running-requests 32 \
    --moe-a2a-backend megamoe \
    --disaggregation-ib-device shca_0,shca_1,shca_2,shca_3 \
    --chunked-prefill-size 32768 \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend mooncake \
    --speculative-algorithm DSPARK \
    --speculative-dspark-block-size 5 \
    --speculative-moe-a2a-backend deepep \
    --speculative-moe-runner-backend deep_gemm \
    --enable-unified-cache-external-linker \
    2>&1 | tee "running_dpsk-v41_PD_P_CP8EP8_${the_host}.log"

```

#### D node

```bash
#!/bin/bash

set -o pipefail
ulimit -c 0

export ROCBLAS_TENSILE_LIBPATH=/opt/dtk/gemm_asm_gfx936_80cu_20261111/rocblas/library
export HIPBLASLT_TENSILE_LIBPATH=/opt/dtk/gemm_asm_gfx936_80cu_20261111/hipblaslt/library
export HIPBLASLT_TUNING_OVERRIDE_FILE=/opt/dtk/gemm_asm_gfx936_80cu_20261111/hipblaslt/tuning.txt
export SGLANG_DSV41_INT8_BLASLT_ALGO_CONFIG=/opt/dtk/gemm_asm_gfx936_80cu_20261111/wo_a_algo.json

export SGLANG_OPT_SWIGLU_CLAMP_FUSION=false
export SGLANG_USE_AITER=0
export SGLANG_USE_AITER_AG=0
export SGLANG_ROCM_USE_AITER_MOE=0
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_LIGHTOP_GROUP_FP8_QUANT=1
export SGLANG_USE_DPSKV4_LIGHTOP_QUANT_K_CACHE=1
export SGLANG_USE_DPSKV4_LIGHTOP_RMSNORM=1
export SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1
export SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=per_rank
export SGLANG_HCU_ENABLE_DSV41_W4A8_ENGRAM_LOAD_OPTIMIZATION=1
export SGLANG_DSV4_SPLIT_PREFILL_DECODE_MLA=1
export SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=1
export SGLANG_HCU_OPT_ENGRAM_GATE=1
export SGLANG_USE_W4A8_MASKED_HIPC=1
export HSA_USERPTR_FOR_PAGED_MEM=0

export NCCL_NET_PLUGIN=shca
export NCCL_IB_HCA=shca_0,shca_1,shca_2,shca_3
export NCCL_SOCKET_IFNAME=ib0
export GLOO_SOCKET_IFNAME=ib0
export ROCSHMEM_MAX_NUM_CONTEXTS=48
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export UCX_NET_DEVICES=shca_0,shca_1,shca_2,shca_3
export MC_ALLOWED_IBV_DEVICES=shca_0,shca_1,shca_2,shca_3
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1
export SGLANG_HEALTH_CHECK_TIMEOUT=10000
export SGLANG_HCU_FUSED_MHC_BOUNDARY=1
export SGLANG_OPT_BF16_FP32_GEMM_ALGO=lightop

readonly PORT=${1:-30000}
readonly MODEL_PATH=${2:-/hygon/DeepSeek-V4.1-Flash-1017-Channel-INT4-w4a8}
readonly CPU_OFFLOAD_GB=${CPU_OFFLOAD_GB:-0}
readonly DIST_TIMEOUT=${DIST_TIMEOUT:-1800}
readonly BOOTSTRAP_PORT=${BOOTSTRAP_PORT:-8998}

the_host=$(hostname)

sglang serve \
    --model-path "$MODEL_PATH" \
    --tp-size 8 \
    --dp 8 \
    --ep 8 \
    --reasoning-parser auto \
    --tool-call-parser auto \
    --enable-dp-attention \
    --enable-dp-lm-head \
    --enable-single-batch-overlap \
    --host "<D_node_ip>" \
    --port "$PORT" \
    --trust-remote-code \
    --mem-fraction-static 0.86 \
    --ep-num-redundant-experts 0 \
    --init-expert-location /decode_static_load.pt \
    --cpu-offload-gb "$CPU_OFFLOAD_GB" \
    --dist-timeout "$DIST_TIMEOUT" \
    --moe-runner-backend deep_gemm \
    --moe-a2a-backend deepep \
    --deepep-mode low_latency \
    --disable-custom-all-reduce \
    --max-running-requests 64 \
    --cuda-graph-max-bs-decode 64 \
    --speculative-algorithm DSPARK \
    --speculative-dspark-block-size 5 \
    --speculative-moe-a2a-backend deepep \
    --speculative-moe-runner-backend deep_gemm \
    --disaggregation-ib-device shca_0,shca_1,shca_2,shca_3 \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend mooncake \
    --disaggregation-bootstrap-port "$BOOTSTRAP_PORT" \
    2>&1 | tee "running_dpsk-v41_PD_D_DP8EP8_${the_host}.log"
```

### DeepSeek-V4.1-Flash-1017-Channel-INT4-w4a8 EPD BW1000 17x SGLang 0.5.18

默认配置共 17 卡，包含 1 个 TP1 Encoder（1 卡）、CP8EP8 Prefill（8 卡）和 DP8EP8 Decode（8 卡）。16 卡是 Prefill 和 Decode 的小计。Encoder 每个实例占用 1 卡，调整 `NUM_ENCODERS` 后，总卡数为 `16 + NUM_ENCODERS`。

E、P、D 节点分别填写 `<E_node_ip>`、`<P_node_ip>`、`<D_node_ip>`，也可以通过 `ENCODER_HOST`、`PREFILL_HOST`、`DECODE_HOST` 传入对应地址。

#### E node

```bash
#!/bin/bash

set -o pipefail
ulimit -c 0

export SGLANG_USE_AITER=0

# ── VLM / 多模态 ─────────────────────────────────────────────────────────
export SGLANG_MM_BUFFER_SIZE_MB="${SGLANG_MM_BUFFER_SIZE_MB:-0}"
export SGLANG_VLM_CACHE_SIZE_MB="${SGLANG_VLM_CACHE_SIZE_MB:-4096}"            # --enable-prefix-mm-cache 的容量
export SGLANG_ENCODER_IMAGE_PROCESSOR_USE_GPU="${SGLANG_ENCODER_IMAGE_PROCESSOR_USE_GPU:-1}"
export SGLANG_ENCODER_MM_LOAD_WORKERS="${SGLANG_ENCODER_MM_LOAD_WORKERS:-8}"    # 默认 4
export SGLANG_ENCODER_PREPROC_WORKERS="${SGLANG_ENCODER_PREPROC_WORKERS:-8}"    # 默认 8
export SGLANG_ENCODER_MAX_BATCH_SIZE="${SGLANG_ENCODER_MAX_BATCH_SIZE:-8}"      # 默认 8，按并发实测调整
export SGLANG_ENCODER_MAX_PENDING_VIT="${SGLANG_ENCODER_MAX_PENDING_VIT:-32}"   # 默认 32

# ── ViT kernel / CUDA graph ──────────────────────────────────────────────
# 融合内核（lightop RMSNorm + fused SiLU-mul + Triton rotary + fused add-RMSNorm）
export SGLANG_DSV41_VIT_FUSED_KERNELS="${SGLANG_DSV41_VIT_FUSED_KERNELS:-1}"
export SGLANG_VIT_ENABLE_CUDA_GRAPH="${SGLANG_VIT_ENABLE_CUDA_GRAPH:-1}"
# 线上图按 patch 数分桶；padding 只进独立的 varlen 段，不影响真实 patch 的注意力。
export SGLANG_DSV41_VIT_CUDA_GRAPH_BUCKET_PATCHES="${SGLANG_DSV41_VIT_CUDA_GRAPH_BUCKET_PATCHES:-256}"
export SGLANG_DSV41_VIT_CUDA_GRAPH_MAX_PATCHES="${SGLANG_DSV41_VIT_CUDA_GRAPH_MAX_PATCHES:-8192}"
export SGLANG_DSV41_VIT_CUDA_GRAPH_CACHE_CAPACITY="${SGLANG_DSV41_VIT_CUDA_GRAPH_CACHE_CAPACITY:-24}"
export SGLANG_DSV41_VIT_CUDA_GRAPH_WARMUP_PACKS="${SGLANG_DSV41_VIT_CUDA_GRAPH_WARMUP_PACKS:-1:1536,1:2048,1:2304,1:3072,1:3584,1:4096,1:4608,1:5120,1:6144,1:7168,1:8192,2:1536,2:3072,3:2048}"
# ── 网络 / RDMA / Mooncake ───────────────────────────────────────────────
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export MOONCAKE_PROTOCOL=rdma
export MC_TCP_ENABLE_CONNECTION_POOL=true

readonly PORT=${1:-30000}
readonly MODEL_PATH=${2:-/hygon/DeepSeek-V4.1-Flash-1017-Channel-INT4-w4a8/}
readonly DIST_TIMEOUT=${DIST_TIMEOUT:-1800}
readonly WATCHDOG_TIMEOUT=${WATCHDOG_TIMEOUT:-3600}
# Prefill 节点的 EncoderBootstrapServer（prefill.sh 的 --language-only 自动开启，默认 8997）。
readonly ENCODER_HOST="${ENCODER_HOST:-<E_node_ip>}"
readonly PREFILL_HOST="${PREFILL_HOST:-<P_node_ip>}"
readonly PREFILL_BOOTSTRAP_PORT=${PREFILL_BOOTSTRAP_PORT:-8997}
readonly PREFILL_BOOTSTRAP_URL=${PREFILL_BOOTSTRAP_URL:-http://${PREFILL_HOST}:${PREFILL_BOOTSTRAP_PORT}}
readonly IB_DEVICE=${IB_DEVICE:-shca_0,shca_1,shca_2,shca_3}
readonly NET_IFNAME=${NET_IFNAME:-ib0}

# 多实例：每个实例占一张卡、TP1，端口 PORT+i
readonly NUM_ENCODERS=${NUM_ENCODERS:-1}
readonly GPU_IDS=${GPU_IDS:-$(seq -s, 0 $((NUM_ENCODERS - 1)))}

export NCCL_IB_HCA="${NCCL_IB_HCA:-$IB_DEVICE}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-$NET_IFNAME}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-$NET_IFNAME}"
export UCX_NET_DEVICES="${UCX_NET_DEVICES:-$IB_DEVICE}"
export MC_ALLOWED_IBV_DEVICES="${MC_ALLOWED_IBV_DEVICES:-$IB_DEVICE}"
export SGLANG_HOST_IP="${SGLANG_HOST_IP:-$ENCODER_HOST}"
export SGLANG_LOCAL_IP_NIC="${SGLANG_LOCAL_IP_NIC:-$NET_IFNAME}"

# ── ViT attention 后端 ──────────────────────────────────────────────────
#   HCU (gfx928/936/938): fa3 -> DTK flash_attn varlen（vision.py 在 HCU 上必导入 flash_attn）
#   AMD + aiter:          aiter_attn
#   其它:                  triton_attn
readonly PYTHON=${PYTHON:-python3}   # 需与 sglang 命令使用同一个解释器
MM_ATTN_BACKEND=${MM_ATTN_BACKEND:-}
if [[ -z "$MM_ATTN_BACKEND" ]]; then
    gcn_arch=$(HIP_VISIBLE_DEVICES="${GPU_IDS%%,*}" "$PYTHON" -c \
        "import torch; print(torch.cuda.get_device_properties(0).gcnArchName)" 2>/dev/null)
    if [[ "$gcn_arch" =~ gfx(928|936|938) ]]; then
        if import_err=$("$PYTHON" -c "from flash_attn import flash_attn_varlen_func" 2>&1); then
            MM_ATTN_BACKEND=fa3
        else
            echo "[encoder] HCU(${gcn_arch}) 但 flash_attn 导入失败：${import_err##*$'\n'}" >&2
            MM_ATTN_BACKEND=triton_attn
        fi
    elif import_err=$("$PYTHON" -c "from aiter import flash_attn_varlen_func" 2>&1); then
        MM_ATTN_BACKEND=aiter_attn
    else
        echo "[encoder] arch=${gcn_arch:-unknown}，aiter 导入失败：${import_err##*$'\n'}" >&2
        MM_ATTN_BACKEND=triton_attn
    fi
fi
echo "[encoder] mm-attention-backend=${MM_ATTN_BACKEND} max_batch=${SGLANG_ENCODER_MAX_BATCH_SIZE} instances=${NUM_ENCODERS} gpus=${GPU_IDS}"

IFS=',' read -r -a gpu_list <<< "$GPU_IDS"
if (( ${#gpu_list[@]} < NUM_ENCODERS )); then
    echo "[encoder] GPU_IDS(${GPU_IDS}) 少于 NUM_ENCODERS(${NUM_ENCODERS})" >&2
    exit 1
fi

the_host=$(hostname)

launch_encoder() {
    local idx=$1 gpu=$2 port=$3
    HIP_VISIBLE_DEVICES="$gpu" sglang serve \
        --model-path "$MODEL_PATH" \
        --tp-size 1 \
        --host "$ENCODER_HOST" \
        --port "$port" \
        --trust-remote-code \
        --encoder-only \
        --encoder-transfer-backend mooncake \
        --encoder-register-urls "$PREFILL_BOOTSTRAP_URL" \
        --disaggregation-ib-device "$IB_DEVICE" \
        --dtype bfloat16 \
        --mm-attention-backend "$MM_ATTN_BACKEND" \
        --enable-prefix-mm-cache \
        --dist-timeout "$DIST_TIMEOUT" \
        --watchdog-timeout "$WATCHDOG_TIMEOUT" \
        2>&1 | tee "running_dpsk-v41_PD_E_TP1_ENCODER_${the_host}_${idx}.log"
}

if (( NUM_ENCODERS == 1 )); then
    launch_encoder 0 "${gpu_list[0]}" "$PORT"
    exit $?
fi

pids=()
trap 'kill "${pids[@]}" 2>/dev/null; wait' INT TERM EXIT
for ((i = 0; i < NUM_ENCODERS; i++)); do
    launch_encoder "$i" "${gpu_list[$i]}" "$((PORT + i))" &
    pids+=($!)
    sleep 5   # 错开启动，避免自动选取的内部端口冲突
done
wait -n
echo "[encoder] 有实例退出，停止其余实例" >&2

```

#### P node

```bash
#!/bin/bash

set -o pipefail
ulimit -c 0

export ROCBLAS_TENSILE_LIBPATH=/opt/dtk/gemm_asm_gfx936_80cu_20261111/rocblas/library
export HIPBLASLT_TENSILE_LIBPATH=/opt/dtk/gemm_asm_gfx936_80cu_20261111/hipblaslt/library
export HIPBLASLT_TUNING_OVERRIDE_FILE=/opt/dtk/gemm_asm_gfx936_80cu_20261111/hipblaslt/tuning.txt
export SGLANG_DSV41_INT8_BLASLT_ALGO_CONFIG=/opt/dtk/gemm_asm_gfx936_80cu_20261111/wo_a_algo.json


export HSA_USERPTR_FOR_PAGED_MEM=0
export SGLANG_OPT_SWIGLU_CLAMP_FUSION=false
export SGLANG_USE_AITER=0
export SGLANG_USE_AITER_AG=0
export SGLANG_ROCM_USE_AITER_MOE=0
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_LIGHTOP_GROUP_FP8_QUANT=1
export SGLANG_USE_DPSKV4_LIGHTOP_QUANT_K_CACHE=1
export SGLANG_USE_DPSKV4_LIGHTOP_RMSNORM=1
# Disabled: triggers Engram._HostTable -> sgl_kernel.kvcacheio.get_device_accessible_ptr,
# which ImportErrors on the currently installed sgl_kernel (missing symbol).
# Re-enable once sgl_kernel is upgraded to a version that exports it.
export SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1
export SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=per_rank
export SGLANG_HCU_ENABLE_DSV41_W4A8_ENGRAM_LOAD_OPTIMIZATION=1
export SGLANG_DSV4_SPLIT_PREFILL_DECODE_MLA=1
export SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=1
export SGLANG_USE_W4A8_CONTIGUOUS_HIPC=1
export LIGHTOP_PREFILL_SPARSE_PREP_FUSION=3
export SGLANG_HCU_MEGA_MOE_RUNTIME=megamoe
export SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=4096
export SGLANG_DSV41_PAGED_MQA_LOGITS_FREE_MEM_FRACTION=0.05
export SGLANG_KVALLOC_KERNEL=0
export ROCSHMEM_MAX_NUM_CONTEXTS=48
export HSA_USERPTR_FOR_PAGED_MEM=0
export SGLANG_DSV4_HC_BF16X3_MIN_ROWS=4096
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1
export SGLANG_HCU_DSV41_BF16_MAIN_PROJ_PREPACK=1
readonly PORT=${1:-30000}
readonly MODEL_PATH=${2:-/hygon/DeepSeek-V4.1-Flash-1017-Channel-INT4-w4a8/}
readonly CPU_OFFLOAD_GB=${CPU_OFFLOAD_GB:-0}
readonly DIST_TIMEOUT=${DIST_TIMEOUT:-1800}
readonly PREFILL_HOST="${PREFILL_HOST:-<P_node_ip>}"
readonly PD_BOOTSTRAP_PORT=${PD_BOOTSTRAP_PORT:-8998}
readonly ENCODER_BOOTSTRAP_PORT=${ENCODER_BOOTSTRAP_PORT:-8997}
readonly ENCODER_HOST="${ENCODER_HOST:-<E_node_ip>}"
readonly ENCODER_PORT=${ENCODER_PORT:-30000}
readonly IB_DEVICE=${IB_DEVICE:-shca_0,shca_1,shca_2,shca_3}
readonly NET_IFNAME=${NET_IFNAME:-ib0}
readonly WATCHDOG_TIMEOUT=${WATCHDOG_TIMEOUT:-3600}

export NCCL_IB_HCA="${NCCL_IB_HCA:-$IB_DEVICE}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-$NET_IFNAME}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-$NET_IFNAME}"
export UCX_NET_DEVICES="${UCX_NET_DEVICES:-$IB_DEVICE}"
export MC_ALLOWED_IBV_DEVICES="${MC_ALLOWED_IBV_DEVICES:-$IB_DEVICE}"
export SGLANG_HOST_IP="${SGLANG_HOST_IP:-$PREFILL_HOST}"
export SGLANG_LOCAL_IP_NIC="${SGLANG_LOCAL_IP_NIC:-$NET_IFNAME}"
export SGLANG_EMBEDDING_POOL_SIZE_MB=512
the_host=$(hostname)

sglang serve \
    --model-path "$MODEL_PATH" \
    --tp-size 8 \
    --ep-size 8 \
    --enable-prefill-cp \
    --attn-cp-size 8 \
     --reasoning-parser auto \
    --tool-call-parser auto \
    --cp-strategy interleave \
    --enable-cp-cache-layer-split \
    --enable-decoder-swa-bounded-replay \
    --ep-num-redundant-experts 0 \
    --language-only \
    --init-expert-location /home/prefill_static_load.pt \
    --encoder-transfer-backend mooncake \
    --encoder-urls "http://${ENCODER_HOST}:${ENCODER_PORT}" \
    --encoder-bootstrap-port "$ENCODER_BOOTSTRAP_PORT" \
    --host "$PREFILL_HOST" \
    --port "$PORT" \
    --trust-remote-code \
    --cuda-graph-backend-prefill disabled \
    --disable-radix-cache \
    --mem-fraction-static 0.85 \
    --cpu-offload-gb "$CPU_OFFLOAD_GB" \
    --dist-timeout "$DIST_TIMEOUT" \
    --watchdog-timeout "$WATCHDOG_TIMEOUT" \
    --max-running-requests 32 \
    --moe-a2a-backend megamoe \
    --disaggregation-ib-device "$IB_DEVICE" \
    --disaggregation-bootstrap-port "$PD_BOOTSTRAP_PORT" \
    --chunked-prefill-size 32768 \
    --max-prefill-tokens 32768 \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend mooncake \
    --speculative-algorithm DSPARK \
    --speculative-dspark-block-size 5 \
    --speculative-moe-a2a-backend deepep \
    --speculative-moe-runner-backend deep_gemm \
    2>&1 | tee "running_dpsk-v41_PD_P_CP8EP8_${the_host}.log"

```

#### D node

```bash
#!/bin/bash

# EPD - D (decode) @ <decode-host>
set -o pipefail
ulimit -c 0

export ROCBLAS_TENSILE_LIBPATH=/opt/dtk/gemm_asm_gfx936_80cu_20261111/rocblas/library
export HIPBLASLT_TENSILE_LIBPATH=/opt/dtk/gemm_asm_gfx936_80cu_20261111/hipblaslt/library
export HIPBLASLT_TUNING_OVERRIDE_FILE=/opt/dtk/gemm_asm_gfx936_80cu_20261111/hipblaslt/tuning.txt
export SGLANG_DSV41_INT8_BLASLT_ALGO_CONFIG=/opt/dtk/gemm_asm_gfx936_80cu_20261111/wo_a_algo.json
export SGLANG_OPT_SWIGLU_CLAMP_FUSION=false
export SGLANG_USE_AITER=0
export SGLANG_USE_AITER_AG=0
export SGLANG_ROCM_USE_AITER_MOE=0
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_LIGHTOP_GROUP_FP8_QUANT=1
export SGLANG_USE_DPSKV4_LIGHTOP_QUANT_K_CACHE=1
export SGLANG_USE_DPSKV4_LIGHTOP_RMSNORM=1
export SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1
export SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=per_rank
export SGLANG_HCU_ENABLE_DSV41_W4A8_ENGRAM_LOAD_OPTIMIZATION=1
export SGLANG_DSV4_SPLIT_PREFILL_DECODE_MLA=1
export SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=1
export SGLANG_USE_W4A8_CONTIGUOUS_HIPC=1
export SGLANG_HCU_OPT_ENGRAM_GATE=1
export HSA_USERPTR_FOR_PAGED_MEM=0
export ROCSHMEM_MAX_NUM_CONTEXTS=48
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1
export SGLANG_HCU_FUSED_MHC_BOUNDARY=1
export SGLANG_OPT_BF16_FP32_GEMM_ALGO=lightop

readonly PORT=${1:-30000}
readonly MODEL_PATH=${2:-/hygon/DeepSeek-V4.1-Flash-1017-Channel-INT4-w4a8/}
readonly CPU_OFFLOAD_GB=${CPU_OFFLOAD_GB:-0}
readonly DIST_TIMEOUT=${DIST_TIMEOUT:-1800}
readonly DECODE_HOST="${DECODE_HOST:-<D_node_ip>}"
readonly PD_BOOTSTRAP_PORT=${PD_BOOTSTRAP_PORT:-8998}
readonly IB_DEVICE=${IB_DEVICE:-shca_0,shca_1,shca_2,shca_3}
readonly NET_IFNAME=${NET_IFNAME:-ib0}
readonly WATCHDOG_TIMEOUT=${WATCHDOG_TIMEOUT:-3600}
readonly MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-128}
readonly CUDA_GRAPH_MAX_BS_DECODE=${CUDA_GRAPH_MAX_BS_DECODE:-32}

export NCCL_IB_HCA="${NCCL_IB_HCA:-$IB_DEVICE}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-$NET_IFNAME}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-$NET_IFNAME}"
export UCX_NET_DEVICES="${UCX_NET_DEVICES:-$IB_DEVICE}"
export MC_ALLOWED_IBV_DEVICES="${MC_ALLOWED_IBV_DEVICES:-$IB_DEVICE}"
export SGLANG_HOST_IP="${SGLANG_HOST_IP:-$DECODE_HOST}"
export SGLANG_LOCAL_IP_NIC="${SGLANG_LOCAL_IP_NIC:-$NET_IFNAME}"
the_host=$(hostname)

sglang serve \
    --model-path "$MODEL_PATH" \
    --tp-size 8 \
    --ep-size 8 \
    --dp 8 \
     --reasoning-parser auto \
    --tool-call-parser auto \
    --enable-dp-attention \
    --enable-dp-lm-head \
    --disable-radix-cache \
    --enable-single-batch-overlap \
    --language-only \
   --init-expert-location /decode_static_load.pt \
    --host "$DECODE_HOST" \
    --port "$PORT" \
    --trust-remote-code \
    --mem-fraction-static 0.89 \
    --ep-num-redundant-experts 0 \
    --cpu-offload-gb "$CPU_OFFLOAD_GB" \
    --dist-timeout "$DIST_TIMEOUT" \
    --max-running-requests "$MAX_RUNNING_REQUESTS" \
    --cuda-graph-max-bs-decode "$CUDA_GRAPH_MAX_BS_DECODE" \
    --moe-runner-backend deep_gemm \
    --moe-a2a-backend deepep \
    --deepep-mode low_latency \
    --disable-custom-all-reduce \
    --disaggregation-ib-device "$IB_DEVICE" \
    --disaggregation-bootstrap-port "$PD_BOOTSTRAP_PORT" \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend mooncake \
    --watchdog-timeout "$WATCHDOG_TIMEOUT" \
    --speculative-algorithm DSPARK \
    --speculative-dspark-block-size 5 \
    --speculative-moe-a2a-backend deepep \
    --speculative-moe-runner-backend deep_gemm \

    2>&1 | tee "running_dpsk-v41_PD_D_TP8EP8_lowlatency_dspark_fractionstatic_0.80_${the_host}.log"

```

### DeepSeek-V4.1-Flash-Channel-INT4-w4a8 1P1D BW1100 8x SGLang 0.5.18

将 `<P_node_ip>` 和 `<D_node_ip>` 替换为 P、D 服务的实际监听 IP。同机部署时两处可填写同一地址，服务端口分别默认为 `30000` 和 `30001`。

#### P node

```bash
#!/bin/bash
set -o pipefail
ulimit -c 0

export HIP_VISIBLE_DEVICES=0,1,2,3
unset NCCL_TOPO_FILE
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_IB_DISABLE=1

export SGLANG_OPT_SWIGLU_CLAMP_FUSION=false
export SGLANG_USE_AITER=0
export SGLANG_USE_AITER_AG=0
export SGLANG_ROCM_USE_AITER_MOE=0
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_LIGHTOP_GROUP_FP8_QUANT=1
export SGLANG_USE_DPSKV4_LIGHTOP_QUANT_K_CACHE=1
export SGLANG_USE_DPSKV4_LIGHTOP_RMSNORM=1
export SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1
export SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=per_rank
export SGLANG_HCU_ENABLE_DSV41_W4A8_ENGRAM_LOAD_OPTIMIZATION=1
export SGLANG_DSV4_SPLIT_PREFILL_DECODE_MLA=1
export SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=1
export SGLANG_USE_W4A8_CONTIGUOUS_HIPC=1
export LIGHTOP_PREFILL_SPARSE_PREP_FUSION=3
export HSA_USERPTR_FOR_PAGED_MEM=0

readonly HOST_IP="<P_node_ip>"
readonly SOCKET_IFNAME=ens66f1np1
readonly RDMA_DEVICES=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_8

export SGLANG_HOST_IP="$HOST_IP"
export NCCL_SOCKET_IFNAME="$SOCKET_IFNAME"
export GLOO_SOCKET_IFNAME="$SOCKET_IFNAME"

export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_8:1
export MC_ALLOWED_IBV_DEVICES="$RDMA_DEVICES"

export ROCSHMEM_MAX_NUM_CONTEXTS=48
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1

readonly PORT=${1:-30000}
readonly MODEL_PATH=${2:-/hygon/DeepSeek-V4.1-Flash-Channel-INT4-w4a8}
readonly CPU_OFFLOAD_GB=${CPU_OFFLOAD_GB:-0}
readonly DIST_TIMEOUT=${DIST_TIMEOUT:-1800}

the_host=$(hostname)

sglang serve \
    --model-path "$MODEL_PATH" \
    --tp-size 4 \
    --ep-size 4 \
    --enable-prefill-cp \
    --attn-cp-size 4 \
    --reasoning-parser auto \
    --tool-call-parser auto \
    --cp-strategy interleave \
    --enable-cp-cache-layer-split \
    --enable-single-batch-overlap \
    --enable-decoder-swa-bounded-replay \
    --host "$HOST_IP" \
    --port "$PORT" \
    --trust-remote-code \
    --cuda-graph-backend-prefill disabled \
    --disable-radix-cache \
    --mem-fraction-static 0.85 \
    --chunked-prefill-size 32768 \
    --cpu-offload-gb "$CPU_OFFLOAD_GB" \
    --dist-timeout "$DIST_TIMEOUT" \
    --max-running-requests 32 \
    --moe-runner-backend deep_gemm \
    --moe-a2a-backend deepep \
    --deepep-mode normal \
    --speculative-algorithm DSPARK \
    --speculative-dspark-block-size 5 \
    --speculative-moe-a2a-backend deepep \
    --speculative-moe-runner-backend deep_gemm \
    --disaggregation-ib-device "$RDMA_DEVICES" \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend mooncake \
    2>&1 | tee "running_dpsk-v41_PD_P_CP4EP4_layersplit_sbo_dspark_fractionstatic_0.85_${the_host}.log"
```

#### D node

```bash
#!/bin/bash

set -o pipefail
ulimit -c 0

export HIP_VISIBLE_DEVICES=4,5,6,7
unset NCCL_TOPO_FILE
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_IB_DISABLE=1

export SGLANG_OPT_SWIGLU_CLAMP_FUSION=false
export SGLANG_USE_AITER=0
export SGLANG_USE_AITER_AG=0
export SGLANG_ROCM_USE_AITER_MOE=0
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_LIGHTOP_GROUP_FP8_QUANT=1
export SGLANG_USE_DPSKV4_LIGHTOP_QUANT_K_CACHE=1
export SGLANG_USE_DPSKV4_LIGHTOP_RMSNORM=1
export SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1
export SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=per_rank
export SGLANG_HCU_ENABLE_DSV41_W4A8_ENGRAM_LOAD_OPTIMIZATION=1
export SGLANG_DSV4_SPLIT_PREFILL_DECODE_MLA=1
export SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=1
export SGLANG_HCU_OPT_ENGRAM_GATE=1
export SGLANG_USE_W4A8_MASKED_HIPC=1
export SGLANG_HCU_FUSED_MHC_BOUNDARY=1
export HSA_USERPTR_FOR_PAGED_MEM=0
export SGLANG_OPT_BF16_FP32_GEMM_ALGO=lightop

readonly HOST_IP="<D_node_ip>"
readonly SOCKET_IFNAME=ens66f1np1
readonly RDMA_DEVICES=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_8

export SGLANG_HOST_IP="$HOST_IP"
export NCCL_SOCKET_IFNAME="$SOCKET_IFNAME"
export GLOO_SOCKET_IFNAME="$SOCKET_IFNAME"

export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_8:1
export MC_ALLOWED_IBV_DEVICES="$RDMA_DEVICES"

export ROCSHMEM_MAX_NUM_CONTEXTS=48
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1
export SGLANG_HEALTH_CHECK_TIMEOUT=10000

readonly PORT=${1:-30001}
readonly MODEL_PATH=${2:-/hygon/DeepSeek-V4.1-Flash-Channel-INT4-w4a8}
readonly CPU_OFFLOAD_GB=${CPU_OFFLOAD_GB:-0}
readonly DIST_TIMEOUT=${DIST_TIMEOUT:-1800}
readonly BOOTSTRAP_PORT=${BOOTSTRAP_PORT:-8998}

the_host=$(hostname)

sglang serve \
    --model-path "$MODEL_PATH" \
    --tp-size 4 \
    --dp 4 \
    --ep 4 \
    --enable-dp-attention \
    --enable-dp-lm-head \
    --reasoning-parser auto \
    --tool-call-parser auto \
    --enable-single-batch-overlap \
    --host "$HOST_IP" \
    --port "$PORT" \
    --trust-remote-code \
    --disable-radix-cache \
    --mem-fraction-static 0.86 \
    --cpu-offload-gb "$CPU_OFFLOAD_GB" \
    --dist-timeout "$DIST_TIMEOUT" \
    --max-running-requests 128 \
    --moe-runner-backend deep_gemm \
    --moe-a2a-backend deepep \
    --deepep-mode low_latency \
    --disable-custom-all-reduce \
    --cuda-graph-max-bs-decode 128 \
    --speculative-algorithm DSPARK \
    --speculative-dspark-block-size 5 \
    --speculative-moe-a2a-backend deepep \
    --speculative-moe-runner-backend deep_gemm \
    --disaggregation-ib-device "$RDMA_DEVICES" \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend mooncake \
    --disaggregation-bootstrap-port "$BOOTSTRAP_PORT" \
    2>&1 | tee "running_dpsk-v41_PD_D_DP4EP4_DSPARK_${the_host}.log"
```



#### Router

`<P_service_port>` 和 `<D_service_port>` 分别填写 P、D 的服务端口。Router 如与 D 同机运行，请将 Router 的监听端口改为未占用的端口，并同步调整客户端 URL。

```bash
python3 -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill "http://<P_node_ip>:<P_service_port>" \
  --decode "http://<D_node_ip>:<D_service_port>" \
  --request-timeout-secs 600 \
  --host 0.0.0.0 \
  --port 30001 \
  --prometheus-port 29001
```

## API 调用

### IFB

以下示例对应 BW1100 FP8 IFB 启动命令，直接访问 SGLang 服务的 `10015` 端口，使用命令中 `--served-model-name` 指定的模型名。

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:10015/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="DeepSeek-V4.1-Flash-1017-Channel-FP8-w8a8",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你好，请介绍一下你自己。"},
    ],
    max_tokens=2048,
)

print(response.choices[0].message.content)
```

```bash
curl http://localhost:10015/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "DeepSeek-V4.1-Flash-1017-Channel-FP8-w8a8", "messages": [{"role": "user", "content": "你好"}], "max_tokens": 128}'
```

### PD 分离

PD 分离模式下，客户端请求发送到 SGLang Router，而非直接发送到 P/D 节点。示例中 Router 端口为 `30001`。将 `<served_model_name>` 替换为实际部署返回的 `/v1/models` 模型 ID。

```python
from openai import OpenAI

client = OpenAI(base_url="http://<router_ip>:30001/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="<served_model_name>",
    messages=[
        {"role": "user", "content": "你好"},
    ],
    max_tokens=2048,
)

print(response.choices[0].message.content)
```

```bash
curl "http://<router_ip>:30001/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model": "<served_model_name>", "messages": [{"role": "user", "content": "你好"}], "max_tokens": 128}'
```
