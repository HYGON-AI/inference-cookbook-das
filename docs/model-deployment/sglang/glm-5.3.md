# GLM-5.3

## 模型列表

| 模型权重 | 量化方式 | SGLang 镜像 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | ----------- | -------- | ---- | -------- | -------- |
| [hygon/GLM-5.3-Flash-Channel-FP8-w8a8](https://modelscope.cn/models/hygon/GLM-5.3-Flash-Channel-FP8-w8a8) | FP8 W8A8 | 0.5.18 | BW1100 | 16 | 1P1D | [**`>_`**](#glm-53-flash-channel-fp8-w8a8-1p1d-bw1100-16x-sglang-0518) |
|  | FP8 W8A8 | 0.5.18 | BW1100 | 24 | 1P1D | [**`>_`**](#glm-53-flash-channel-fp8-w8a8-1p1d-bw1100-24x-sglang-0518) |

## 启动命令

两种配置共用 P 节点命令。16 卡配置为 P 8 卡 + D 8 卡；24 卡配置为 P 8 卡 + D 两节点共 16 卡。先将下列公共环境变量保存为 `common.sh`，在各节点启动命令所在目录放置该文件。节点 IP、网卡、ROCSHMEM 拓扑文件、DTK 库路径与 DeepEP 配置文件须按环境调整。网卡配置参考：[IB 网卡](../../troubleshooting/common-issues.md#ib网卡)。命令使用 `/home/work/code/sglang-das` 的源码版本。

### 公共环境变量：common.sh

```bash
#!/usr/bin/env bash
set -eo pipefail
export PYTHONPATH="${SGLANG_SOURCE_ROOT:-/home/work/code/sglang-das}/python${PYTHONPATH:+:${PYTHONPATH}}"
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=3000
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=3000
export SGLANG_DISAGGREGATION_TRANSFER_TIMEOUT=3000
export ROCSHMEM_TOPO_FILE_FORCE="${ROCSHMEM_TOPO_FILE_FORCE:-/home/work/deepseekv4/topo.config}"
export SGLANG_HEALTH_CHECK_TIMEOUT=3001
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-enp113s0f0np0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-enp113s0f0np0}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1}"
export MC_ALLOWED_IBV_DEVICES="${MC_ALLOWED_IBV_DEVICES:-mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9}"
export ROCSHMEM_IB_GID_INDEX="${ROCSHMEM_IB_GID_INDEX:-0}"
unset NCCL_TOPO_FILE RCCL_TOPO_FILE
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16
export HSA_ENABLE_COREDUMP=1
export USE_DCU_CUSTOM_ALLREDUCE=0
export SGLANG_USE_AITER_AR=0
export SGLANG_CUSTOM_ALLREDUCE_INPUT_FENCE=thread0
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export HIP_KERNEL_BATCH_CEILING=100
export GPU_FORCE_BLIT_COPY_SIZE=16
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export SGLANG_USE_LIGHTOP=1
export SGLANG_ROCM_USE_AITER_MOE=0
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export HIP_GRAPH_ACCUMULATE_DISPATCH=1
export HIP_GRAPH_USE_CMD_CACHE=0
export SGLANG_ROCM_USE_AITER_TILELANG_MHC=1
export SGLANG_OPT_USE_TILELANG_MHC_PRE=1
export SGLANG_OPT_USE_TILELANG_MHC_POST=1
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
export SGLANG_OPT_SWIGLU_CLAMP_FUSION=0
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_MAX_NUM_CONTEXTS=60
export ROCSHMEM_HEAP_SIZE=3173741824
export SGLANG_USE_DEEPGEMM_MOE=1
export SGLANG_USE_FP8_W8A8_MOE=1
export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=1
export W8A8_SUPPORT_METHODS=3
export SGLANG_USE_AITER_CHUNK_GATED_DELTA_H_HIP=1
export SGLANG_USE_FUSED_RMS_QUANT=1
export SGLANG_USE_FUSED_SILU_MUL_QUANT=1
export SGLANG_USE_LIGHTOP_PREFILL_DEQUANT=1
export SGLANG_USE_FUSED_SILU_MUL_CLAMP_QUANT=1
export SGLANG_USE_HICACHE_OPTIMIZATION_KERNEL=1
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export GLM_ENABLE_ENCODER_SESSION_ID_HEADER=1
export SGLANG_HTTP_WORKER_REUSE_PORT=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export RAYON_NUM_THREADS=1
export LD_LIBRARY_PATH=/home/work/dtk-26.10-blas-nightly-0914/lib/:$LD_LIBRARY_PATH
```


### 共用 P 节点

```bash
source ./common.sh
ulimit -s 67108864

sglang serve \
  --model-path hygon/GLM-5.3-Flash-Channel-FP8-w8a8 \
  --served-model-name hygon/GLM-5.3-Flash-Channel-FP8-w8a8 \
  --trust-remote-code \
  --host "<P_node_ip>" \
  --port 30000 \
  --random-seed 42 \
  --context-length 1048576 \
  --chunked-prefill-size 32768 \
  --max-prefill-tokens 32768 \
  --disable-shared-experts-fusion \
  --disable-piecewise-cuda-graph \
  --disable-chunked-prefix-cache \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend mooncake \
  --mem-fraction-static 0.8 \
  --max-running-requests 16 \
  --max-mamba-cache-size 96 \
  --mamba-scheduler-strategy extra_buffer \
  --page-size 64 \
  --disable-cuda-graph \
  --enable-single-batch-overlap \
  --tp-size 8 \
  --ep-size 8 \
  --attn-cp-size 8 \
  --enable-nsa-prefill-context-parallel \
  --nsa-prefill-cp-mode round-robin-split \
  --moe-a2a-backend deepep \
  --deepep-mode normal \
  --attention-backend nsa \
  --nsa-prefill-backend flashmla_auto \
  --nsa-decode-backend flashmla_kv \
  --kv-cache-dtype fp8_e4m3 \
  --numa-node 0 3 2 1 4 7 6 5 \
  --enable-hierarchical-cache \
  --hicache-size 64 \
  --hicache-write-policy write_through \
  --hicache-io-backend kernel \
  --hicache-mem-layout layer_first \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 5 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 6 \
  --enable-cache-report \
  --enable-metrics \
  --tokenizer-worker-num=8 \
  --deepep-config ./ep_config.json \
  --json-model-override-args '{"index_share_for_mtp_iteration": true}'
```


### GLM-5.3-Flash-Channel-FP8-w8a8 1P1D BW1100 16x SGLang 0.5.18

P 节点使用上方的共用命令。

#### D node

```bash
source ./common.sh
ulimit -s 67108864
export SGLANG_NSA_FUSE_TOPK=1
export DEEPEP_ENABLE_LL_LAYERED_OPT=1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=96
export SGLANG_DSA_ENABLE_MTP_PRECOMPUTE_METADATA=0
export SGLANG_DSA_KPOOL_LIGHTOP_TOPK=1

sglang serve \
  --model-path hygon/GLM-5.3-Flash-Channel-FP8-w8a8 \
  --served-model-name hygon/GLM-5.3-Flash-Channel-FP8-w8a8 \
  --trust-remote-code \
  --host "<D_node_ip>" \
  --port 30000 \
  --dist-init-addr "<D_node_ip>:5000" \
  --max-running-requests 64 \
  --tp-size 8 \
  --dp-size 8 \
  --ep-size 8 \
  --random-seed 42 \
  --disable-shared-experts-fusion \
  --disable-piecewise-cuda-graph \
  --disable-chunked-prefix-cache \
  --disable-radix-cache \
  --cuda-graph-bs 1 2 4 6 8 10 12 \
  --mem-fraction-static 0.85 \
  --context-length 1048576 \
  --page-size 64 \
  --dtype bfloat16 \
  --max-mamba-cache-size 320 \
  --mamba-scheduler-strategy no_buffer \
  --moe-dense-tp-size 1 \
  --enable-dp-attention \
  --enable-dp-lm-head \
  --moe-a2a-backend deepep \
  --deepep-mode low_latency \
  --attention-backend nsa \
  --nsa-prefill-backend flashmla_auto \
  --nsa-decode-backend flashmla_kv \
  --linear-attn-backend triton \
  --kv-cache-dtype fp8_e4m3 \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-mode decode \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 5 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 6 \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --enable-cache-report \
  --enable-metrics \
  --tokenizer-worker-num=8 \
  --json-model-override-args '{"index_share_for_mtp_iteration": true}' \
  --numa-node 0 3 2 1 4 7 6 5 \
  --deepep-config ./ep_config.json
```

#### Router

```bash
source ./common.sh
python3 -m sglang_router.launch_router --pd-disaggregation \
  --prefill "http://<P_node_ip>:30000" \
  --decode "http://<D_node_ip>:30000" \
  --policy cache_aware \
  --port 30001
```


### GLM-5.3-Flash-Channel-FP8-w8a8 1P1D BW1100 24x SGLang 0.5.18

P 节点使用上方的共用命令。D 组包含两个 8 卡节点，共同组成一个 D 服务。

#### D node 0

```bash
source ./common.sh
ulimit -s 67108864
export SGLANG_NSA_FUSE_TOPK=1
export DEEPEP_ENABLE_LL_LAYERED_OPT=1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=96
export SGLANG_DSA_ENABLE_MTP_PRECOMPUTE_METADATA=0
export SGLANG_DSA_KPOOL_LIGHTOP_TOPK=1

sglang serve \
  --model-path hygon/GLM-5.3-Flash-Channel-FP8-w8a8 \
  --served-model-name hygon/GLM-5.3-Flash-Channel-FP8-w8a8 \
  --trust-remote-code \
  --host "<D_node0_ip>" \
  --port 30000 \
  --dist-init-addr "<D_node0_ip>:5000" \
  --nnodes 2 \
  --node-rank 0 \
  --max-running-requests 128 \
  --tp-size 16 \
  --dp-size 16 \
  --ep-size 16 \
  --random-seed 42 \
  --disable-shared-experts-fusion \
  --disable-piecewise-cuda-graph \
  --disable-chunked-prefix-cache \
  --disable-radix-cache \
  --cuda-graph-bs 1 2 4 6 8 10 12 \
  --mem-fraction-static 0.85 \
  --context-length 1048576 \
  --page-size 64 \
  --dtype bfloat16 \
  --max-mamba-cache-size 320 \
  --mamba-scheduler-strategy no_buffer \
  --moe-dense-tp-size 1 \
  --enable-dp-attention \
  --enable-dp-lm-head \
  --moe-a2a-backend deepep \
  --deepep-mode low_latency \
  --attention-backend nsa \
  --nsa-prefill-backend flashmla_auto \
  --nsa-decode-backend flashmla_kv \
  --linear-attn-backend triton \
  --kv-cache-dtype fp8_e4m3 \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-mode decode \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 5 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 6 \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --enable-cache-report \
  --enable-metrics \
  --tokenizer-worker-num=8 \
  --json-model-override-args '{"index_share_for_mtp_iteration": true}' \
  --numa-node 0 3 2 1 4 7 6 5 \
  --deepep-config ./ep_config.json
```

#### D node 1

```bash
source ./common.sh
ulimit -s 67108864
export SGLANG_NSA_FUSE_TOPK=1
export DEEPEP_ENABLE_LL_LAYERED_OPT=1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=96
export SGLANG_DSA_ENABLE_MTP_PRECOMPUTE_METADATA=0
export SGLANG_DSA_KPOOL_LIGHTOP_TOPK=1

sglang serve \
  --model-path hygon/GLM-5.3-Flash-Channel-FP8-w8a8 \
  --served-model-name hygon/GLM-5.3-Flash-Channel-FP8-w8a8 \
  --trust-remote-code \
  --host "<D_node1_ip>" \
  --port 30000 \
  --dist-init-addr "<D_node0_ip>:5000" \
  --nnodes 2 \
  --node-rank 1 \
  --max-running-requests 128 \
  --tp-size 16 \
  --dp-size 16 \
  --ep-size 16 \
  --random-seed 42 \
  --disable-shared-experts-fusion \
  --disable-piecewise-cuda-graph \
  --disable-chunked-prefix-cache \
  --disable-radix-cache \
  --cuda-graph-bs 1 2 4 6 8 10 12 \
  --mem-fraction-static 0.85 \
  --context-length 1048576 \
  --page-size 64 \
  --dtype bfloat16 \
  --max-mamba-cache-size 320 \
  --mamba-scheduler-strategy no_buffer \
  --moe-dense-tp-size 1 \
  --enable-dp-attention \
  --enable-dp-lm-head \
  --moe-a2a-backend deepep \
  --deepep-mode low_latency \
  --attention-backend nsa \
  --nsa-prefill-backend flashmla_auto \
  --nsa-decode-backend flashmla_kv \
  --linear-attn-backend triton \
  --kv-cache-dtype fp8_e4m3 \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-mode decode \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 5 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 6 \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --enable-cache-report \
  --enable-metrics \
  --tokenizer-worker-num=8 \
  --json-model-override-args '{"index_share_for_mtp_iteration": true}' \
  --numa-node 0 3 2 1 4 7 6 5 \
  --deepep-config ./ep_config.json
```

#### Router

```bash
source ./common.sh
python3 -m sglang_router.launch_router --pd-disaggregation \
  --prefill "http://<P_node_ip>:30000" \
  --decode "http://<D_node0_ip>:30000" \
  --policy cache_aware \
  --port 30001
```

## API 调用

### PD 分离

客户端请求发送到 Router。以下以 Router 与 P 节点同机为例，实际部署时请使用 Router 地址。

```bash
curl "http://<P_node_ip>:30001/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"hygon/GLM-5.3-Flash-Channel-FP8-w8a8","messages":[{"role":"user","content":"你好"}],"max_tokens":128}'
```
