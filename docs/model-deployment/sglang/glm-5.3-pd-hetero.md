# GLM-5.3 PD 分离异构部署

本文介绍 GLM-5.3 的异构 PD 分离部署方案：Prefill 使用 BW1100，Decode 使用 B300，P/D 之间基于 SGLang Mooncake PD 分离方案传输 KV Cache。

## 模型列表

| 模型权重 | 量化方式 | SGLang 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | ----------- | -------- | ---- | -------- | -------- |
| [hygon/GLM-5.3-Channel-FP8-w8a8](https://www.modelscope.cn/models/hygon/GLM-5.3-Channel-FP8-w8a8) | FP8 W8A8 | 0.5.18 | BW1100 (P) | 48 | 6P1D | [**`>_`**](#glm-53-channel-fp8-w8a8-6p1d-bw1100b300-56x-sglang-0518) |
| [zai-org/GLM-5.3](https://huggingface.co/zai-org/GLM-5.3) (FP8) | FP8 | [0.5.18](https://hub.docker.com/r/lmsysorg/sglang/tags?name=v0.5.18-cu130) | B300 (D) | 8 | 6P1D | [**`>_`**](#glm-53-channel-fp8-w8a8-6p1d-bw1100b300-56x-sglang-0518) |

注： hygon SGLang 0.5.18 的版本目前是测试镜像，可以通过 'docker pull 42.228.13.241:5000/hcu/base_env/sglang:0.5.18-ubuntu22.04-dtk26041-py3.10-20261104-1735-patch' 获取

## 启动命令

### GLM-5.3-Channel-FP8-w8a8 6P1D BW1100+B300 56x SGLang 0.5.18

网卡配置参考：[IB 网卡](../../troubleshooting/common-issues.md#ib网卡)。

6 个 P 节点均为独立的单机 8 卡实例，采用 CP8EP8 部署；D 节点为单机 8 卡，采用 DP8EP8 部署。以下示例中各节点 IP 与端口如下，实际部署时请根据实际情况修改：

| 节点 | IP | 服务端口 |
| ---- | -- | -------- |
| P node 0 | `<p_node0_ip>` | `30023` |
| P node 1 | `<p_node1_ip>` | `30023` |
| P node 2 | `<p_node2_ip>` | `30023` |
| P node 3 | `<p_node3_ip>` | `30023` |
| P node 4 | `<p_node4_ip>` | `30023` |
| P node 5 | `<p_node5_ip>` | `30023` |
| D node 0 | `<d_node0_ip>` | `30000` |
| Router | — | `8000` |

#### DeepEP 配置

以下 `ep_config.json` 为参考配置。将其保存为文件，并将 P 节点命令中的 `<deepep_config_path>` 替换为该文件的实际路径。

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

#### P node 0

```bash
export NCCL_SOCKET_IFNAME=ib6s200p0
export GLOO_SOCKET_IFNAME=ib6s200p0
export ROCSHMEM_ALLOWED_IBV_DEVICES="ib7s400p0,ib7s400p1,ib7s400p2,ib7s400p3,ib7s400p4,ib7s400p5,ib7s400p6,ib7s400p7"
export ROCSHMEM_DISABLE_HDP_FLUSH=1
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export MC_TE_FILTERS="ib7s400p0,ib7s400p1,ib7s400p2,ib7s400p3,ib7s400p4,ib7s400p5,ib7s400p6,ib7s400p7"
export MC_ALLOWED_IBV_DEVICES="ib7s400p0,ib7s400p1,ib7s400p2,ib7s400p3,ib7s400p4,ib7s400p5,ib7s400p6,ib7s400p7"
export SGLANG_MOE_COPY_WEIGHT_VIEWS_BEFORE_H2D=1
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_DEEPGEMM_MOE=1
export SGLANG_USE_FP8_W8A8_MOE=1
export SGLANG_NSA_HCU_REUSE_SORTED_TOPK=1
export SGLANG_NSA_MQA_LOGITS_MEMORY_BUDGET_GB=2
export SGLANG_ENABLE_HCU_CONCAT_MLA_ABSORB_Q=1
export SGLANG_DEQUANT_K_CACHE_BLOCK_T=16
export SGLANG_OPT_USE_TOPK_V2=0
export SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK=1
export SGLANG_LOGITS_PROCESSER_CHUNK_SIZE=2048
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export HIP_KERNEL_BATCH_CEILING=100
export GPU_FORCE_BLIT_COPY_SIZE=16
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1800

sglang serve \
  --model-path hygon/GLM-5.3-Channel-FP8-w8a8 \
  --trust-remote-code \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --served-model-name GLM-5.3-Channel-FP8-w8a8 \
  --host <p_node0_ip> \
  --port 30023 \
  --dist-init-addr <p_node0_ip>:5000 \
  --tp-size 8 \
  --ep-size 8 \
  --attn-cp-size 8 \
  --moe-dense-tp-size 1 \
  --moe-a2a-backend deepep \
  --deepep-mode normal \
  --deepep-config "<deepep_config_path>" \
  --enable-prefill-cp \
  --enable-dsa-cache-layer-split \
  --cp-strategy interleave \
  --dsa-prefill-backend flashmla_sparse \
  --dsa-decode-backend flashmla_kv \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --context-length 1048576 \
  --dtype bfloat16 \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --page-size 64 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.9 \
  --chunked-prefill-size 32768 \
  --max-prefill-tokens 32768 \
  --max-running-requests 128 \
  --disable-cuda-graph \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-bootstrap-port 8998 \
  --disaggregation-ib-device "ib7s400p0,ib7s400p1,ib7s400p2,ib7s400p3,ib7s400p4,ib7s400p5,ib7s400p6,ib7s400p7"
```

#### P node 1

```bash
export NCCL_SOCKET_IFNAME=ib6s200p0
export GLOO_SOCKET_IFNAME=ib6s200p0
export ROCSHMEM_ALLOWED_IBV_DEVICES="ib7s400p0,ib7s400p1,ib7s400p2,ib7s400p3,ib7s400p4,ib7s400p5,ib7s400p6,ib7s400p7"
export ROCSHMEM_DISABLE_HDP_FLUSH=1
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export MC_TE_FILTERS="ib7s400p0,ib7s400p1,ib7s400p2,ib7s400p3,ib7s400p4,ib7s400p5,ib7s400p6,ib7s400p7"
export MC_ALLOWED_IBV_DEVICES="ib7s400p0,ib7s400p1,ib7s400p2,ib7s400p3,ib7s400p4,ib7s400p5,ib7s400p6,ib7s400p7"
export SGLANG_MOE_COPY_WEIGHT_VIEWS_BEFORE_H2D=1
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_DEEPGEMM_MOE=1
export SGLANG_USE_FP8_W8A8_MOE=1
export SGLANG_NSA_HCU_REUSE_SORTED_TOPK=1
export SGLANG_NSA_MQA_LOGITS_MEMORY_BUDGET_GB=2
export SGLANG_ENABLE_HCU_CONCAT_MLA_ABSORB_Q=1
export SGLANG_DEQUANT_K_CACHE_BLOCK_T=16
export SGLANG_OPT_USE_TOPK_V2=0
export SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK=1
export SGLANG_LOGITS_PROCESSER_CHUNK_SIZE=2048
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export HIP_KERNEL_BATCH_CEILING=100
export GPU_FORCE_BLIT_COPY_SIZE=16
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1800

sglang serve \
  --model-path hygon/GLM-5.3-Channel-FP8-w8a8 \
  --trust-remote-code \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --served-model-name GLM-5.3-Channel-FP8-w8a8 \
  --host <p_node1_ip> \
  --port 30023 \
  --dist-init-addr <p_node1_ip>:5000 \
  --tp-size 8 \
  --ep-size 8 \
  --attn-cp-size 8 \
  --moe-dense-tp-size 1 \
  --moe-a2a-backend deepep \
  --deepep-mode normal \
  --deepep-config "<deepep_config_path>" \
  --enable-prefill-cp \
  --enable-dsa-cache-layer-split \
  --cp-strategy interleave \
  --dsa-prefill-backend flashmla_sparse \
  --dsa-decode-backend flashmla_kv \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --context-length 1048576 \
  --dtype bfloat16 \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --page-size 64 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.9 \
  --chunked-prefill-size 32768 \
  --max-prefill-tokens 32768 \
  --max-running-requests 128 \
  --disable-cuda-graph \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-bootstrap-port 8998 \
  --disaggregation-ib-device "ib7s400p0,ib7s400p1,ib7s400p2,ib7s400p3,ib7s400p4,ib7s400p5,ib7s400p6,ib7s400p7"
```

#### P node 2

```bash
export NCCL_SOCKET_IFNAME=ib6s200p0
export GLOO_SOCKET_IFNAME=ib6s200p0
export ROCSHMEM_ALLOWED_IBV_DEVICES="ib7s400p0,ib7s400p1,ib7s400p2,ib7s400p3,ib7s400p4,ib7s400p5,ib7s400p6,ib7s400p7"
export ROCSHMEM_DISABLE_HDP_FLUSH=1
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export MC_TE_FILTERS="ib7s400p0,ib7s400p1,ib7s400p2,ib7s400p3,ib7s400p4,ib7s400p5,ib7s400p6,ib7s400p7"
export MC_ALLOWED_IBV_DEVICES="ib7s400p0,ib7s400p1,ib7s400p2,ib7s400p3,ib7s400p4,ib7s400p5,ib7s400p6,ib7s400p7"
export SGLANG_MOE_COPY_WEIGHT_VIEWS_BEFORE_H2D=1
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_DEEPGEMM_MOE=1
export SGLANG_USE_FP8_W8A8_MOE=1
export SGLANG_NSA_HCU_REUSE_SORTED_TOPK=1
export SGLANG_NSA_MQA_LOGITS_MEMORY_BUDGET_GB=2
export SGLANG_ENABLE_HCU_CONCAT_MLA_ABSORB_Q=1
export SGLANG_DEQUANT_K_CACHE_BLOCK_T=16
export SGLANG_OPT_USE_TOPK_V2=0
export SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK=1
export SGLANG_LOGITS_PROCESSER_CHUNK_SIZE=2048
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export HIP_KERNEL_BATCH_CEILING=100
export GPU_FORCE_BLIT_COPY_SIZE=16
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1800

sglang serve \
  --model-path hygon/GLM-5.3-Channel-FP8-w8a8 \
  --trust-remote-code \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --served-model-name GLM-5.3-Channel-FP8-w8a8 \
  --host <p_node2_ip> \
  --port 30023 \
  --dist-init-addr <p_node2_ip>:5000 \
  --tp-size 8 \
  --ep-size 8 \
  --attn-cp-size 8 \
  --moe-dense-tp-size 1 \
  --moe-a2a-backend deepep \
  --deepep-mode normal \
  --deepep-config "<deepep_config_path>" \
  --enable-prefill-cp \
  --enable-dsa-cache-layer-split \
  --cp-strategy interleave \
  --dsa-prefill-backend flashmla_sparse \
  --dsa-decode-backend flashmla_kv \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --context-length 1048576 \
  --dtype bfloat16 \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --page-size 64 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.9 \
  --chunked-prefill-size 32768 \
  --max-prefill-tokens 32768 \
  --max-running-requests 128 \
  --disable-cuda-graph \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-bootstrap-port 8998 \
  --disaggregation-ib-device "ib7s400p0,ib7s400p1,ib7s400p2,ib7s400p3,ib7s400p4,ib7s400p5,ib7s400p6,ib7s400p7"
```

#### P node 3

```bash
export NCCL_SOCKET_IFNAME=ib6s200p0
export GLOO_SOCKET_IFNAME=ib6s200p0
export ROCSHMEM_ALLOWED_IBV_DEVICES="ib7s400p0,ib7s400p1,ib7s400p2,ib7s400p3,ib7s400p4,ib7s400p5,ib7s400p6,ib7s400p7"
export ROCSHMEM_DISABLE_HDP_FLUSH=1
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export MC_TE_FILTERS="ib7s400p0,ib7s400p1,ib7s400p2,ib7s400p3,ib7s400p4,ib7s400p5,ib7s400p6,ib7s400p7"
export MC_ALLOWED_IBV_DEVICES="ib7s400p0,ib7s400p1,ib7s400p2,ib7s400p3,ib7s400p4,ib7s400p5,ib7s400p6,ib7s400p7"
export SGLANG_MOE_COPY_WEIGHT_VIEWS_BEFORE_H2D=1
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_DEEPGEMM_MOE=1
export SGLANG_USE_FP8_W8A8_MOE=1
export SGLANG_NSA_HCU_REUSE_SORTED_TOPK=1
export SGLANG_NSA_MQA_LOGITS_MEMORY_BUDGET_GB=2
export SGLANG_ENABLE_HCU_CONCAT_MLA_ABSORB_Q=1
export SGLANG_DEQUANT_K_CACHE_BLOCK_T=16
export SGLANG_OPT_USE_TOPK_V2=0
export SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK=1
export SGLANG_LOGITS_PROCESSER_CHUNK_SIZE=2048
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export HIP_KERNEL_BATCH_CEILING=100
export GPU_FORCE_BLIT_COPY_SIZE=16
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1800

sglang serve \
  --model-path hygon/GLM-5.3-Channel-FP8-w8a8 \
  --trust-remote-code \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --served-model-name GLM-5.3-Channel-FP8-w8a8 \
  --host <p_node3_ip> \
  --port 30023 \
  --dist-init-addr <p_node3_ip>:5000 \
  --tp-size 8 \
  --ep-size 8 \
  --attn-cp-size 8 \
  --moe-dense-tp-size 1 \
  --moe-a2a-backend deepep \
  --deepep-mode normal \
  --deepep-config "<deepep_config_path>" \
  --enable-prefill-cp \
  --enable-dsa-cache-layer-split \
  --cp-strategy interleave \
  --dsa-prefill-backend flashmla_sparse \
  --dsa-decode-backend flashmla_kv \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --context-length 1048576 \
  --dtype bfloat16 \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --page-size 64 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.9 \
  --chunked-prefill-size 32768 \
  --max-prefill-tokens 32768 \
  --max-running-requests 128 \
  --disable-cuda-graph \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-bootstrap-port 8998 \
  --disaggregation-ib-device "ib7s400p0,ib7s400p1,ib7s400p2,ib7s400p3,ib7s400p4,ib7s400p5,ib7s400p6,ib7s400p7"
```

#### P node 4

```bash
export NCCL_SOCKET_IFNAME=ib6s200p0
export GLOO_SOCKET_IFNAME=ib6s200p0
export ROCSHMEM_ALLOWED_IBV_DEVICES="ib7s400p0,ib7s400p1,ib7s400p2,ib7s400p3,ib7s400p4,ib7s400p5,ib7s400p6,ib7s400p7"
export ROCSHMEM_DISABLE_HDP_FLUSH=1
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export MC_TE_FILTERS="ib7s400p0,ib7s400p1,ib7s400p2,ib7s400p3,ib7s400p4,ib7s400p5,ib7s400p6,ib7s400p7"
export MC_ALLOWED_IBV_DEVICES="ib7s400p0,ib7s400p1,ib7s400p2,ib7s400p3,ib7s400p4,ib7s400p5,ib7s400p6,ib7s400p7"
export SGLANG_MOE_COPY_WEIGHT_VIEWS_BEFORE_H2D=1
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_DEEPGEMM_MOE=1
export SGLANG_USE_FP8_W8A8_MOE=1
export SGLANG_NSA_HCU_REUSE_SORTED_TOPK=1
export SGLANG_NSA_MQA_LOGITS_MEMORY_BUDGET_GB=2
export SGLANG_ENABLE_HCU_CONCAT_MLA_ABSORB_Q=1
export SGLANG_DEQUANT_K_CACHE_BLOCK_T=16
export SGLANG_OPT_USE_TOPK_V2=0
export SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK=1
export SGLANG_LOGITS_PROCESSER_CHUNK_SIZE=2048
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export HIP_KERNEL_BATCH_CEILING=100
export GPU_FORCE_BLIT_COPY_SIZE=16
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1800

sglang serve \
  --model-path hygon/GLM-5.3-Channel-FP8-w8a8 \
  --trust-remote-code \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --served-model-name GLM-5.3-Channel-FP8-w8a8 \
  --host <p_node4_ip> \
  --port 30023 \
  --dist-init-addr <p_node4_ip>:5000 \
  --tp-size 8 \
  --ep-size 8 \
  --attn-cp-size 8 \
  --moe-dense-tp-size 1 \
  --moe-a2a-backend deepep \
  --deepep-mode normal \
  --deepep-config "<deepep_config_path>" \
  --enable-prefill-cp \
  --enable-dsa-cache-layer-split \
  --cp-strategy interleave \
  --dsa-prefill-backend flashmla_sparse \
  --dsa-decode-backend flashmla_kv \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --context-length 1048576 \
  --dtype bfloat16 \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --page-size 64 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.9 \
  --chunked-prefill-size 32768 \
  --max-prefill-tokens 32768 \
  --max-running-requests 128 \
  --disable-cuda-graph \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-bootstrap-port 8998 \
  --disaggregation-ib-device "ib7s400p0,ib7s400p1,ib7s400p2,ib7s400p3,ib7s400p4,ib7s400p5,ib7s400p6,ib7s400p7"
```

#### P node 5

```bash
export NCCL_SOCKET_IFNAME=ib6s200p0
export GLOO_SOCKET_IFNAME=ib6s200p0
export ROCSHMEM_ALLOWED_IBV_DEVICES="ib7s400p0,ib7s400p1,ib7s400p2,ib7s400p3,ib7s400p4,ib7s400p5,ib7s400p6,ib7s400p7"
export ROCSHMEM_DISABLE_HDP_FLUSH=1
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export MC_TE_FILTERS="ib7s400p0,ib7s400p1,ib7s400p2,ib7s400p3,ib7s400p4,ib7s400p5,ib7s400p6,ib7s400p7"
export MC_ALLOWED_IBV_DEVICES="ib7s400p0,ib7s400p1,ib7s400p2,ib7s400p3,ib7s400p4,ib7s400p5,ib7s400p6,ib7s400p7"
export SGLANG_MOE_COPY_WEIGHT_VIEWS_BEFORE_H2D=1
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_DEEPGEMM_MOE=1
export SGLANG_USE_FP8_W8A8_MOE=1
export SGLANG_NSA_HCU_REUSE_SORTED_TOPK=1
export SGLANG_NSA_MQA_LOGITS_MEMORY_BUDGET_GB=2
export SGLANG_ENABLE_HCU_CONCAT_MLA_ABSORB_Q=1
export SGLANG_DEQUANT_K_CACHE_BLOCK_T=16
export SGLANG_OPT_USE_TOPK_V2=0
export SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK=1
export SGLANG_LOGITS_PROCESSER_CHUNK_SIZE=2048
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export HIP_KERNEL_BATCH_CEILING=100
export GPU_FORCE_BLIT_COPY_SIZE=16
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1800

sglang serve \
  --model-path hygon/GLM-5.3-Channel-FP8-w8a8 \
  --trust-remote-code \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --served-model-name GLM-5.3-Channel-FP8-w8a8 \
  --host <p_node5_ip> \
  --port 30023 \
  --dist-init-addr <p_node5_ip>:5000 \
  --tp-size 8 \
  --ep-size 8 \
  --attn-cp-size 8 \
  --moe-dense-tp-size 1 \
  --moe-a2a-backend deepep \
  --deepep-mode normal \
  --deepep-config "<deepep_config_path>" \
  --enable-prefill-cp \
  --enable-dsa-cache-layer-split \
  --cp-strategy interleave \
  --dsa-prefill-backend flashmla_sparse \
  --dsa-decode-backend flashmla_kv \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --context-length 1048576 \
  --dtype bfloat16 \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --page-size 64 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.9 \
  --chunked-prefill-size 32768 \
  --max-prefill-tokens 32768 \
  --max-running-requests 128 \
  --disable-cuda-graph \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-bootstrap-port 8998 \
  --disaggregation-ib-device "ib7s400p0,ib7s400p1,ib7s400p2,ib7s400p3,ib7s400p4,ib7s400p5,ib7s400p6,ib7s400p7"
```

#### D node 0

```bash
export TRITON_CACHE_DIR=/data/triton_cache
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_USE_FP8_W8A8_MOE=1
export NCCL_IB_DISABLE=0
export GLOO_SOCKET_IFNAME=ib6s200p0
export NCCL_SOCKET_IFNAME=ib6s200p0
export SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1
export SGLANG_DG_CACHE_DIR=/root/.cache/deep_gemm
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export NVSHMEM_ENABLE_NIC_PE_MAPPING=1
export NVSHMEM_HCA_PREFIX=ib7s
export NVSHMEM_HCA_LIST=ib7s400p0:1,ib7s400p1:1,ib7s400p2:1,ib7s400p3:1,ib7s400p4:1,ib7s400p5:1,ib7s400p6:1,ib7s400p7:1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=512
export SGLANG_DISABLE_DSA_INDEXER_FUSION=1

sglang serve \
  --model-path zai-org/GLM-5.3-FP8 \
  --trust-remote-code \
  --served-model-name GLM-5.3-Channel-FP8-w8a8 \
  --enable-cache-report \
  --tp 8 \
  --dp 8 \
  --enable-dp-attention \
  --ep 8 \
  --nccl-port 29500 \
  --dsa-prefill-backend flashmla_sparse \
  --dsa-decode-backend flashmla_kv \
  --page-size 64 \
  --kv-cache-dtype fp8_e4m3 \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend mooncake \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --disaggregation-ib-device "ib7s400p0,ib7s400p1,ib7s400p2,ib7s400p3,ib7s400p4,ib7s400p5,ib7s400p6,ib7s400p7" \
  --prefill-round-robin-balance \
  --mem-fraction-static 0.9 \
  --host <d_node0_ip> \
  --port 30000 \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --cuda-graph-max-bs-decode 128 \
  --max-running-requests 256 \
  --tokenizer-worker-num 8
```

#### Router

```bash
python3 -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://<p_node0_ip>:30023 \
  --prefill http://<p_node1_ip>:30023 \
  --prefill http://<p_node2_ip>:30023 \
  --prefill http://<p_node3_ip>:30023 \
  --prefill http://<p_node4_ip>:30023 \
  --prefill http://<p_node5_ip>:30023 \
  --decode http://<d_node0_ip>:30000 \
  --prefill-policy round_robin \
  --decode-policy round_robin \
  --host 0.0.0.0 \
  --port 8000
```

## API 调用

### PD 分离

PD 分离模式下，客户端请求发送到 SGLang Router，而非直接发送到 P/D 节点。将 `<router_ip>` 替换为 Router 所在节点的 IP，示例中 Router 端口为 `8000`。

```python
from openai import OpenAI

client = OpenAI(base_url="http://<router_ip>:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="GLM-5.3-Channel-FP8-w8a8",
    messages=[{"role": "user", "content": "你好"}],
    max_tokens=2048,
)
```

```bash
curl "http://<router_ip>:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model": "GLM-5.3-Channel-FP8-w8a8", "messages": [{"role": "user", "content": "你好"}], "max_tokens": 128}'
```
