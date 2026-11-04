# hy4

# Hy4

## 模型列表

| 模型权重 | 量化方式 | SGLang 镜像 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| --- | --- | --- | --- | --- | --- | --- |
| [hygon/Hy4-preview-Channel-INT4-w4a8](https://www.modelscope.cn/models/hygon/Hy4-preview-Channel-INT4-w4a8) | INT4 W4A8 | 0.5.18 | BW1000 | 48 | PD分离(P:PP2CP8EP8，D:DCP4DP8EP32) | **`>_`** |

## 启动命令

### Hy4-preview-Channel-INT4-w4a8 PD分离 BW1000 48 x SGLang 0.5.18 (P:PP2CP8EP8，D:DCP4DP8EP32)

#### P节点

##### P1

```bash
export NCCL_PXN_DISABLE=0   #开启PXN
export RCCL_PXN_GPU_BALANCE=1  #开启PXN的DCU负载均衡开关
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RCCL_NET_PLANE="shca_0,shca_3|shca_1,shca_2"  #双平面检测开关
export NCCL_PLUGIN_P2P=ib
export NCCL_NET_PLUGIN=shca
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_PP_LAYER_PARTITION=39,39
export SGLANG_PP_PREFILL_STEP_OVERLAP=1
export SGLANG_NSA_LAYER_SPLIT_ASYNC_BROADCAST=0
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
export ROCSHMEM_ALLOWED_IBV_DEVICES=shca_0,shca_1,shca_2,shca_3
export SGLANG_USE_DEEPGEMM_MOE=1
export SGLANG_INT8_DEEPGEMM_ASM=1
export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/amdsmi:$LD_LIBRARY_PATH
export W8A8_SUPPORT_METHODS=3 #切换blaslt的w8a8
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
##测试性能需要
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
# deepep的deepgemm HT的work space 
export SGLANG_REUSE_W8A8_INT8_EP_MOE_WORKSPACE=0
# mqa logits 申请0.5 GiB
export SGLANG_NSA_DCU_MQA_LOGITS_WORKSPACE_GB=1

# 对长序列input logprob的全词表logits进行分块，避免瞬时显存OOM；
export SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK=1
export SGLANG_LOGITS_PROCESSER_CHUNK_SIZE=1024

export SGLANG_USE_AITER_CHUNK_GATED_DELTA_H_HIP=1
# 指定sglang python代码
#hl_topk
export SGLANG_USE_FUSED_RMS_QUANT=1
export SGLANG_USE_FUSED_SILU_MUL_QUANT=1
export SGLANG_NSA_KPOOL_LIGHTOP_TOPK=1
export SGLANG_USE_LIGHTOP_PREFILL_DEQUANT=1
##
export SGLANG_USE_FUSED_SILU_MUL_CLAMP_QUANT=1
# aiter 的topk P用
export SGLANG_NSA_KPOOL_AITER_TOPK=1
#export GPU_MAX_HW_QUEUES=3
export PYTHONPATH=/jsyadmin/zhaole/hy4/1023/sglang-das/python:${PYTHONPATH:-}
export SGLANG_USE_HICACHE_OPTIMIZATION_KERNEL=1
export SGLANG_PROFILE_TP_RANKS=0

#补充
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/amdsmi:$LD_LIBRARY_PATH
export SGLANG_SET_CPU_AFFINITY=1
sysctl -w kernel.numa_balancing=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT="${SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT:-1200}"
export MC_IB_GID_INDEX=0
export MC_ALLOWED_IBV_DEVICES=shca_0,shca_1,shca_2,shca_3
export HIP_KERNEL_EVENT_SYSTEMFENCE="${HIP_KERNEL_EVENT_SYSTEMFENCE:-1}"
export VLLM_USE_LIGHTOP_MOE_ALIGN="${VLLM_USE_LIGHTOP_MOE_ALIGN:-1}"
export LMSLIM_USE_LIGHTOP="${LMSLIM_USE_LIGHTOP:-1}"
export MC_GID_INDEX="${MC_GID_INDEX:-0}"

export NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-16}"
export NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-16}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"

export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export SGLANG_HOST_IP=10.211.10.11
export SGLANG_USE_DEEPGEMM_MOE=1
export SGLANG_OPT_USE_TOPK_V2=false
export SGLANG_OPT_HY4_IHC_TILELANG=1
#export SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK=true

export SGLANG_USE_LIGHTOP_W4A8_MARLIN_MOE=0 #contiguous W4A8 packed weight must be [E,N/32,K*16] from pack_w4a8_moe_hipc_weight
export SGLANG_USE_W4A8_CONTIGUOUS_HIPC=1 #AssertionError: Hidden size mismatch w1
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_FUSED_RMS_ROTARY=0 # =1的时候decode阶段会报错，暂时设置为0


export SGLANG_ROCM_USE_AITER_MOE=0
export USE_DCU_CUSTOM_ALLREDUCE=0
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
export SGLANG_USE_FP8_W8A8_MOE=0
unset SGLANG_DEEPEP_BF16_DISPATCH
export SGLANG_GROUPGEMM=true

export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_MAX_NUM_CONTEXTS=60
export ROCSHMEM_HEAP_SIZE=3737418240
export ROCSHMEM_TOPO_FILE_FORCE=/workspace/configs/shca_topo.config
export DEEPEP_ENABLE_LL_LAYERED_OPT=1
export SGLANG_TORCH_PROFILER_DIR="${SGLANG_TORCH_PROFILER_DIR:-/workspace/test/prof/decode}"
export SGLANG_DSA_HCU_INT8_INDEX_K_CACHE=1

export SGLANG_DISAGGREGATION_QUEUE_SIZE=8
export SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=8
export SGLANG_DSA_LIGHTOP_PREFILL_DEQUANT_KV=1

time=$(date +"%Y%m%d-%H%M%S")
echo "当前时间：$time"

python3 -m sglang.launch_server \
  --model-path  /models/Hy4-preview-Channel-INT4-w4a8 \
  --trust-remote-code \
  --nnodes 2 \
  --node-rank 0 \
  --host 10.211.10.11 \
  --dist-init-addr 10.211.10.11:5001 \
  --port 30000 \
  --random-seed 42 \
  --chunked-prefill-size 32768 \
  --max-prefill-tokens 32768 \
  --disable-chunked-prefix-cache \
  --enable-cache-report \
  --disaggregation-mode prefill \
  --disaggregation-ib-device shca_0,shca_1,shca_2,shca_3 \
  --disaggregation-transfer-backend mooncake \
  --mem-fraction-static 0.80 \
  --max-running-requests 256 \
  --watchdog-timeout 1200 \
  --dist-timeout 3600 \
  --page-size 64 \
  --pp-max-micro-batch-size 8 \
  --pp-async-batch-depth 2 \
  --warmups prefill_shapes \
  --pp-size 2 \
  --tp-size 8 \
  --ep-size 8 \
  --attn-cp-size 8 \
  --moe-a2a-backend deepep \
  --deepep-mode normal \
  --attention-backend dsa \
  --dsa-prefill-backend flashmla_sparse \
  --dsa-decode-backend flashmla_kv \
  --enable-prefill-cp \
  --enable-dsa-cache-layer-split \
  --deepep-config ./ep_config.json \
  --speculative-algorithm NEXTN  \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --cp-strategy interleave \
  --reasoning-parser hunyuan \
  --kv-cache-dtype fp8_e4m3 \
  --quantization slimquant_w4a8_marlin \
  --numa-node 0 3 2 1 4 7 6 5 \
  --json-model-override-args '{"index_share_for_mtp_iteration": true}' \
  "$@" \
  2>&1 | tee "logs/hy4-w4a8-p1-$time.log"
```

##### P2

```bash
export NCCL_PXN_DISABLE=0   #开启PXN
export RCCL_PXN_GPU_BALANCE=1  #开启PXN的DCU负载均衡开关
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RCCL_NET_PLANE="shca_0,shca_3|shca_1,shca_2"  #双平面检测开关
export NCCL_PLUGIN_P2P=ib
export NCCL_NET_PLUGIN=shca
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_PP_LAYER_PARTITION=39,39
export SGLANG_PP_PREFILL_STEP_OVERLAP=1
export SGLANG_NSA_LAYER_SPLIT_ASYNC_BROADCAST=0
export HSA_ENABLE_COREDUMP=1
export USE_DCU_CUSTOM_ALLREDUCE=0DAS
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
export ROCSHMEM_ALLOWED_IBV_DEVICES=shca_0,shca_1,shca_2,shca_3
export SGLANG_USE_DEEPGEMM_MOE=1
export SGLANG_INT8_DEEPGEMM_ASM=1
export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/amdsmi:$LD_LIBRARY_PATH
export W8A8_SUPPORT_METHODS=3 #切换blaslt的w8a8
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
##测试性能需要
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
# deepep的deepgemm HT的work space 
export SGLANG_REUSE_W8A8_INT8_EP_MOE_WORKSPACE=0
# mqa logits 申请0.5 GiB
export SGLANG_NSA_DCU_MQA_LOGITS_WORKSPACE_GB=1

# 对长序列input logprob的全词表logits进行分块，避免瞬时显存OOM；
export SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK=1
export SGLANG_LOGITS_PROCESSER_CHUNK_SIZE=1024

export SGLANG_USE_AITER_CHUNK_GATED_DELTA_H_HIP=1
# 指定sglang python代码
#hl_topk
export SGLANG_USE_FUSED_RMS_QUANT=1
export SGLANG_USE_FUSED_SILU_MUL_QUANT=1
export SGLANG_NSA_KPOOL_LIGHTOP_TOPK=1
export SGLANG_USE_LIGHTOP_PREFILL_DEQUANT=1
##
export SGLANG_USE_FUSED_SILU_MUL_CLAMP_QUANT=1
# aiter 的topk P用
export SGLANG_NSA_KPOOL_AITER_TOPK=1
#export GPU_MAX_HW_QUEUES=3
export PYTHONPATH=/jsyadmin/zhaole/hy4/1023/sglang-das/python:${PYTHONPATH:-}
export SGLANG_USE_HICACHE_OPTIMIZATION_KERNEL=1
export SGLANG_PROFILE_TP_RANKS=0

#补充
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/amdsmi:$LD_LIBRARY_PATH
export SGLANG_SET_CPU_AFFINITY=1
sysctl -w kernel.numa_balancing=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT="${SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT:-1200}"
export MC_IB_GID_INDEX=0
export MC_ALLOWED_IBV_DEVICES=shca_0,shca_1,shca_2,shca_3
export HIP_KERNEL_EVENT_SYSTEMFENCE="${HIP_KERNEL_EVENT_SYSTEMFENCE:-1}"
export VLLM_USE_LIGHTOP_MOE_ALIGN="${VLLM_USE_LIGHTOP_MOE_ALIGN:-1}"
export LMSLIM_USE_LIGHTOP="${LMSLIM_USE_LIGHTOP:-1}"
export MC_GID_INDEX="${MC_GID_INDEX:-0}"

export NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-16}"
export NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-16}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export SGLANG_HOST_IP=10.211.10.12
export SGLANG_USE_DEEPGEMM_MOE=1
export SGLANG_OPT_USE_TOPK_V2=false
export SGLANG_OPT_HY4_IHC_TILELANG=1
#export SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK=true

export SGLANG_USE_LIGHTOP_W4A8_MARLIN_MOE=0 #contiguous W4A8 packed weight must be [E,N/32,K*16] from pack_w4a8_moe_hipc_weight
export SGLANG_USE_W4A8_CONTIGUOUS_HIPC=1
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_FUSED_RMS_ROTARY=0 # =1的时候decode阶段会报错，暂时设置为0

export SGLANG_ROCM_USE_AITER_MOE=0
export USE_DCU_CUSTOM_ALLREDUCE=0
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
export SGLANG_USE_FP8_W8A8_MOE=0
unset SGLANG_DEEPEP_BF16_DISPATCH
export SGLANG_GROUPGEMM=true

export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_MAX_NUM_CONTEXTS=60
export ROCSHMEM_HEAP_SIZE=3737418240
export ROCSHMEM_TOPO_FILE_FORCE=/workspace/configs/shca_topo.config
export DEEPEP_ENABLE_LL_LAYERED_OPT=1
export SGLANG_TORCH_PROFILER_DIR="${SGLANG_TORCH_PROFILER_DIR:-/workspace/test/prof/decode}"
export SGLANG_DSA_HCU_INT8_INDEX_K_CACHE=1

export SGLANG_DISAGGREGATION_QUEUE_SIZE=8
export SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=8
export SGLANG_DSA_LIGHTOP_PREFILL_DEQUANT_KV=1

time=$(date +"%Y%m%d-%H%M%S")
echo "当前时间：$time"

python3 -m sglang.launch_server \
  --model-path  /models/Hy4-preview-Channel-INT4-w4a8 \
  --trust-remote-code \
  --nnodes 2 \
  --node-rank 1 \
  --host 10.211.10.12 \
  --dist-init-addr 10.211.10.11:5001 \
  --port 30000 \
  --random-seed 42 \
  --chunked-prefill-size 32768 \
  --max-prefill-tokens 32768 \
  --disable-chunked-prefix-cache \
  --enable-cache-report \
  --disaggregation-mode prefill \
  --disaggregation-ib-device shca_0,shca_1,shca_2,shca_3 \
  --disaggregation-transfer-backend mooncake \
  --mem-fraction-static 0.80 \
  --max-running-requests 256 \
  --watchdog-timeout 1200 \
  --dist-timeout 3600 \
  --page-size 64 \
  --pp-max-micro-batch-size 8 \
  --pp-async-batch-depth 2 \
  --warmups prefill_shapes \
  --pp-size 2 \
  --tp-size 8 \
  --ep-size 8 \
  --attn-cp-size 8 \
  --moe-a2a-backend deepep \
  --deepep-mode normal \
  --kv-cache-dtype fp8_e4m3 \
  --quantization slimquant_w4a8_marlin \
  --attention-backend dsa \
  --dsa-prefill-backend flashmla_sparse \
  --dsa-decode-backend flashmla_kv \
  --enable-prefill-cp \
  --enable-dsa-cache-layer-split \
  --deepep-config ./ep_config.json \
  --speculative-algorithm NEXTN  \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --cp-strategy interleave \
  --reasoning-parser hunyuan \
  --numa-node 0 3 2 1 4 7 6 5 \
  --json-model-override-args '{"index_share_for_mtp_iteration": true}' \
  "$@" \
  2>&1 | tee "logs/hy4-w4a8-p2-$time.log"
```

#### D节点

##### D1

```bash
#!/bin/bash
set -euo pipefail
export PYTHONPATH=/hy4_work/sglang/python:${PYTHONPATH:-} 
# SGLANG_ROOT="${SGLANG_ROOT:-/hy4_work/sglang}"
MODEL_PATH="${MODEL_PATH:-/models/Hy4-preview-Channel-INT4-w4a8}"
DECODE_HOST="${DECODE_HOST:-10.211.10.8}"
DECODE_BIND_HOST="${DECODE_BIND_HOST:-${DECODE_HOST}}"
DECODE_PORT="${DECODE_PORT:-30001}"
DIST_INIT_ADDR="${DIST_INIT_ADDR:-10.211.10.8:5000}"
export SGLANG_HOST_IP="${SGLANG_HOST_IP:-${DECODE_HOST}}"
#export ROCBLAS_TENSILE_LIBPATH=/hy4_work/auto_select_tools/optimization_configs/new/config/library_gpu5
export LIGHTOP_PAGED_MQA_LOGITS_BACKEND=1
export LIGHTOP_PAGED_MQA_LOGITS_GROUP_SIZE=3
# shared indexer 显存优化
# export SGLANG_HY4_COMPACT_PD_INDEX_K=1
# ===================== RCCL / NCCL =====================
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=shca_0,shca_1,shca_2,shca_3
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"

unset NCCL_IB_GID_INDEX
export NCCL_NET_PLUGIN=shca
export NCCL_PLUGIN_P2P=ib
#export NCCL_TOPO_FILE=/opt/topo/built-in-508-topo-input-tj-default.xml

# 关闭 debug 刷屏
unset NCCL_DEBUG
unset NCCL_DEBUG_SUBSYS
unset NCCL_DEBUG_FILE
unset RCCL_DEBUG
unset RCCL_DEBUG_SUBSYS
unset TORCH_NCCL_DEBUG
export TORCH_CPP_LOG_LEVEL=ERROR
export PYTHONWARNINGS="ignore"
# Import the repository directly; disable bytecode writes on shared storage.
SGLANG_PYTHON="${SGLANG_ROOT}/python"
if [[ ! -f "${SGLANG_PYTHON}/sglang/launch_server.py" ]]; then
  printf 'SGLang Python source not found: %s\n' "${SGLANG_PYTHON}" >&2
  exit 1
fi
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  printf 'HY4 model config not found: %s/config.json\n' "${MODEL_PATH}" >&2
  exit 1
fi
export PYTHONPATH="${SGLANG_PYTHON}"
export PYTHONDONTWRITEBYTECODE=1
unset PYTHONPYCACHEPREFIX

# Put runtime files on the work volume, not the container root filesystem.
# Override SGLANG_RUNTIME_DIR with an absolute path on another writable volume.
SGLANG_RUNTIME_DIR="${SGLANG_RUNTIME_DIR:-${SGLANG_ROOT%/*}/runtime/node0}"
if [[ "${SGLANG_RUNTIME_DIR}" != /* ]]; then
  printf 'SGLANG_RUNTIME_DIR must be an absolute path: %s\n' "${SGLANG_RUNTIME_DIR}" >&2
  exit 1
fi
export TMPDIR="${SGLANG_RUNTIME_DIR}/tmp"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
export XDG_CACHE_HOME="${SGLANG_RUNTIME_DIR}/cache"
export TRITON_CACHE_DIR="${XDG_CACHE_HOME}/triton"
export TORCH_EXTENSIONS_DIR="${XDG_CACHE_HOME}/torch_extensions"
export TORCHINDUCTOR_CACHE_DIR="${XDG_CACHE_HOME}/torchinductor"
mkdir -p "${TMPDIR}" "${TRITON_CACHE_DIR}" "${TORCH_EXTENSIONS_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"
cd "${SGLANG_RUNTIME_DIR}"
ulimit -c 0
# Check real writes before loading torch or spawning GPU workers.
python3 - <<'PY_TMP_CHECK'
import os
import tempfile
with tempfile.TemporaryFile(dir=os.environ["TMPDIR"]) as probe:
    probe.write(b"sglang temporary directory check")
    probe.flush()
    os.fsync(probe.fileno())
PY_TMP_CHECK

# Keep service logs on the work volume.
SGLANG_LOG_DIR="${SGLANG_LOG_DIR:-/hy4_work/hy4_w4a8/dcp/dcp4/logs}"
mkdir -p "${SGLANG_LOG_DIR}"
LOG_FILE="${SGLANG_LOG_DIR}/hy4_w4a8_dcp2_dp16_ep32_node0_$(date +%Y%m%d_%H%M%S).log"
export PYTHONUNBUFFERED=1

# rocm tuning
#export ROCBLAS_TENSILE_LIBPATH=/hy4_work/auto_select_tools/optimization_configs/new/config/library_gpu5

# ===================== SGLang / 通用性能参数 =====================
unset SGLANG_ENABLE_SPEC_V2
unset SGLANG_ENABLE_PD_DECODE_STEPINFO_SYNC
export HSA_ENABLE_COREDUMP=0
export USE_DCU_CUSTOM_ALLREDUCE=0
export SGLANG_USE_AITER_AR=0
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/amdsmi:$LD_LIBRARY_PATH
export HIP_KERNEL_BATCH_CEILING=100
export GPU_FORCE_BLIT_COPY_SIZE=16
export HSA_KERNARG_POOL_SIZE=8388608
export HSA_FORCE_FINE_GRAIN_PCIE=1
export ROC_AQL_QUEUE_SIZE=131072
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_LIGHTOP_W4A8_MARLIN_MOE=0
export SGLANG_ROCM_USE_AITER_MOE=0
export W8A8_SUPPORT_METHODS=3
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export HIP_GRAPH_ACCUMULATE_DISPATCH=1
export HIP_H2D_DISABLE_COPY_BUFFER="${HIP_H2D_DISABLE_COPY_BUFFER:-0}"
export HIP_D2H_DISABLE_COPY_BUFFER="${HIP_D2H_DISABLE_COPY_BUFFER:-0}"
export HIP_H2D_DIRECT_COPY_THRESHOLD="${HIP_H2D_DIRECT_COPY_THRESHOLD:-32768}"
export HIP_H2D_HSAAPI_COPY_THRESHOLD="${HIP_H2D_HSAAPI_COPY_THRESHOLD:-32768}"
export HIP_D2H_DIRECT_COPY_THRESHOLD="${HIP_D2H_DIRECT_COPY_THRESHOLD:-512}"
export HIP_D2H_HSAAPI_COPY_THRESHOLD="${HIP_D2H_HSAAPI_COPY_THRESHOLD:-512}"

# export HIP_GRAPH_USE_CMD_CACHE=0
export SGLANG_USE_MODELSCOPE=1
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=3600
#export ROCSHMEM_DISABLE_HDP_FLUSH=1
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288

export ROCSHMEM_HEAP_SIZE=419430400 # 400 MB
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=64
export DEEPEP_ENABLE_LL_LAYERED_OPT=1 # deepep分层
export DEEPEP_ENABLE_LL_LAYERED_DEVELOP=0
export ROCSHMEM_GDR_DISABLE_XDP=1
#export DEEPEP_ENABLE_MORI_EP_LL_LAYERED=1 #mori-ep分层
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_MAX_NUM_CONTEXTS=60
export ROCSHMEM_ALLOWED_IBV_DEVICES=shca_0,shca_1,shca_2,shca_3
export ROCSHMEM_TOPO_FILE_FORCE="${ROCSHMEM_TOPO_FILE_FORCE:-/hy4_work/configs/shca_topo.config}"
export HSA_USE_SVM=0

export GPU_MAX_HW_QUEUES="${GPU_MAX_HW_QUEUES:-4}"
export SGLANG_USE_W4A8_CONTIGUOUS_HIPC=1
# ===================== Mooncake / Disaggregation =====================
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export MC_IB_GID_INDEX=0
export MC_ALLOWED_IBV_DEVICES=shca_0,shca_1,shca_2,shca_3
unset MC_GID_INDEX
unset SGLANG_SIMULATE_ACC_LEN SGLANG_SIMULATE_ACC_METHOD SGLANG_SIMULATE_ACC_TOKEN_MODE
export SGLANG_DEBUG_RADIX_CACHE_TIME=0

# SBO
export SGLANG_ROCM_USE_MULTI_STREAM=1

# Match the current P-side BF16 index-K transfer ABI (main KV remains FP8).
# unset SGLANG_NSA_HCU_INT8_INDEX_K_CACHE
export SGLANG_DSA_HCU_INT8_INDEX_K_CACHE=1
export SGLANG_DSA_FUSE_TOPK=false
export SGLANG_USE_W4A8_MASKED_HIPC=1
export SGLANG_UVICORN_WORKER_HEALTHCHECK_TIMEOUT=120
export SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1

export SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH="${SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH:-1}"
export SGLANG_SET_CPU_AFFINITY="${SGLANG_SET_CPU_AFFINITY:-1}"
export SGLANG_OPT_HY4_IHC_PRENORM=1
export SGLANG_OPT_HY4_IHC_TILELANG=1
export MC_TCP_ENABLE_CONNECTION_POOL=1
export SGLANG_DSA_HCU_LIGHTOP_TOPK="${SGLANG_DSA_HCU_LIGHTOP_TOPK:-1}"
export SGLANG_TORCH_PROFILER_DIR="${SGLANG_TORCH_PROFILER_DIR:-/hy4_work/prof}"

export HYV4_GATE_TUNING=off
export HYV4_MTP_INPUT_TUNING=1
export HYV4_HADAMARD_TUNING=1

export SGLANG_HY4_DCP2_FUSED_POST_ATTN=1
export SGLANG_HY4_DCP2_FUSED_VPROJ=1

export SGLANG_HY4_DCP_FUSED_KV_WRITE_LOCS=1
export SGLANG_HY4_HCU_DSA_SHORT_CONTEXT_K_ONLY=1

# topk_length
export SGLANG_ENABLE_DSA_FLASHMLA_DCP_GROUPED_TOPK=1

# # g_proj overlap
export SGLANG_ENABLE_HYV4_GATE_PROJ_PARALLEL=0
export SGLANG_HY4_DCP2_GATE_A2A_OVERLAP=1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 上下文长度读取 HY4 config.json，KV 容量按可用显存计算。
# 如需限制，可在调用脚本时追加 --context-length / --max-total-tokens。
# ===================== 启动 HY4 Decode：node-rank 0 =====================
# --cuda-graph-config '{"decode":{"backend":"full","bs":[1,2,4,8,16,32]},"prefill":{"backend":"disabled"}}' \
python3 -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --trust-remote-code \
  --served-model-name "${SERVED_MODEL_NAME:-hy4}" \
  --host "${DECODE_BIND_HOST}" \
  --port "${DECODE_PORT}" \
  --dist-init-addr "${DIST_INIT_ADDR}" \
  --nnodes 4 \
  --node-rank 0 \
  --tp-size 32 \
  --dp-size 8 \
  --dcp-size 4 \
  --dcp-comm-backend a2a \
  --dcp-replicate-q-proj \
  --moe-dense-tp-size 1 \
  --ep-size 16 \
  --pp-size 1 \
  --attn-cp-size 1 \
  --enable-dp-attention \
  --custom-all-reduce-backend native \
  --moe-a2a-backend deepep \
  --deepep-mode low_latency \
  --page-size 64 \
  --disaggregation-transfer-backend mooncake \
  --json-model-override-args '{"index_share_for_mtp_iteration": true}' \
  --cuda-graph-config '{"decode":{"backend":"full","bs":[1,2,4,8,16,32,64,128]},"prefill":{"backend":"disabled"}}' \
  --attention-backend dsa \
  --dsa-decode-backend flashmla_kv \
  --dsa-paged-mqa-logits-backend lightop \
  --quantization slimquant_w4a8_marlin \
  --cuda-graph-backend-decode full \
  --cuda-graph-backend-prefill disabled \
  --dtype bfloat16 \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.80 \
  --skip-server-warmup \
  --max-running-requests 128 \
  --reasoning-parser hunyuan \
  --disaggregation-ib-device shca_0,shca_1,shca_2,shca_3 \
  --deepep-config /hy4_work/configs/ep_config.json \
  --enable-dp-lm-head \
  --disaggregation-mode decode \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --tokenizer-worker-num 8 \
  --enable-cache-report \
  --load-balance-method total_tokens \
  "$@" \
  2>&1 | tee -a "${LOG_FILE}"
```

##### D2

```bash
#!/bin/bash
set -euo pipefail
export PYTHONPATH=/hy4_work/sglang/python:${PYTHONPATH:-} 
# SGLANG_ROOT="${SGLANG_ROOT:-/hy4_work/sglang}"
MODEL_PATH="${MODEL_PATH:-/models/Hy4-preview-Channel-INT4-w4a8}"
DECODE_HOST="${DECODE_HOST:-10.211.10.9}"
DECODE_BIND_HOST="${DECODE_BIND_HOST:-${DECODE_HOST}}"
DECODE_PORT="${DECODE_PORT:-30001}"
DIST_INIT_ADDR="${DIST_INIT_ADDR:-10.211.10.8:5000}"
export SGLANG_HOST_IP="${SGLANG_HOST_IP:-${DECODE_HOST}}"
#export ROCBLAS_TENSILE_LIBPATH=/hy4_work/auto_select_tools/optimization_configs/new/config/library_gpu5
export LIGHTOP_PAGED_MQA_LOGITS_BACKEND=1
export LIGHTOP_PAGED_MQA_LOGITS_GROUP_SIZE=3
# shared indexer 显存优化
# export SGLANG_HY4_COMPACT_PD_INDEX_K=1
# ===================== RCCL / NCCL =====================
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=shca_0,shca_1,shca_2,shca_3
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"

unset NCCL_IB_GID_INDEX
export NCCL_NET_PLUGIN=shca
export NCCL_PLUGIN_P2P=ib
#export NCCL_TOPO_FILE=/opt/topo/built-in-508-topo-input-tj-default.xml

# 关闭 debug 刷屏
unset NCCL_DEBUG
unset NCCL_DEBUG_SUBSYS
unset NCCL_DEBUG_FILE
unset RCCL_DEBUG
unset RCCL_DEBUG_SUBSYS
unset TORCH_NCCL_DEBUG
export TORCH_CPP_LOG_LEVEL=ERROR
export PYTHONWARNINGS="ignore"
# Import the repository directly; disable bytecode writes on shared storage.
SGLANG_PYTHON="${SGLANG_ROOT}/python"
if [[ ! -f "${SGLANG_PYTHON}/sglang/launch_server.py" ]]; then
  printf 'SGLang Python source not found: %s\n' "${SGLANG_PYTHON}" >&2
  exit 1
fi
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  printf 'HY4 model config not found: %s/config.json\n' "${MODEL_PATH}" >&2
  exit 1
fi
export PYTHONPATH="${SGLANG_PYTHON}"
export PYTHONDONTWRITEBYTECODE=1
unset PYTHONPYCACHEPREFIX

# Put runtime files on the work volume, not the container root filesystem.
# Override SGLANG_RUNTIME_DIR with an absolute path on another writable volume.
SGLANG_RUNTIME_DIR="${SGLANG_RUNTIME_DIR:-${SGLANG_ROOT%/*}/runtime/node1}"
if [[ "${SGLANG_RUNTIME_DIR}" != /* ]]; then
  printf 'SGLANG_RUNTIME_DIR must be an absolute path: %s\n' "${SGLANG_RUNTIME_DIR}" >&2
  exit 1
fi
export TMPDIR="${SGLANG_RUNTIME_DIR}/tmp"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
export XDG_CACHE_HOME="${SGLANG_RUNTIME_DIR}/cache"
export TRITON_CACHE_DIR="${XDG_CACHE_HOME}/triton"
export TORCH_EXTENSIONS_DIR="${XDG_CACHE_HOME}/torch_extensions"
export TORCHINDUCTOR_CACHE_DIR="${XDG_CACHE_HOME}/torchinductor"
mkdir -p "${TMPDIR}" "${TRITON_CACHE_DIR}" "${TORCH_EXTENSIONS_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"
cd "${SGLANG_RUNTIME_DIR}"
ulimit -c 0
# Check real writes before loading torch or spawning GPU workers.
python3 - <<'PY_TMP_CHECK'
import os
import tempfile
with tempfile.TemporaryFile(dir=os.environ["TMPDIR"]) as probe:
    probe.write(b"sglang temporary directory check")
    probe.flush()
    os.fsync(probe.fileno())
PY_TMP_CHECK

# Keep service logs on the work volume.
SGLANG_LOG_DIR="${SGLANG_LOG_DIR:-/hy4_work/hy4_w4a8/dcp/dcp4/logs}"
mkdir -p "${SGLANG_LOG_DIR}"
LOG_FILE="${SGLANG_LOG_DIR}/hy4_w4a8_dcp2_dp16_ep32_node1_$(date +%Y%m%d_%H%M%S).log"
export PYTHONUNBUFFERED=1
# rocm tuning
#export ROCBLAS_TENSILE_LIBPATH=/hy4_work/auto_select_tools/optimization_configs/new/config/library_gpu5
# ===================== SGLang / 通用性能参数 =====================
unset SGLANG_ENABLE_SPEC_V2
unset SGLANG_ENABLE_PD_DECODE_STEPINFO_SYNC
export HSA_ENABLE_COREDUMP=0
export USE_DCU_CUSTOM_ALLREDUCE=0
export SGLANG_USE_AITER_AR=0
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/amdsmi:$LD_LIBRARY_PATH
export HIP_KERNEL_BATCH_CEILING=100
export GPU_FORCE_BLIT_COPY_SIZE=16
export HSA_KERNARG_POOL_SIZE=8388608
export HSA_FORCE_FINE_GRAIN_PCIE=1
export ROC_AQL_QUEUE_SIZE=131072
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_LIGHTOP_W4A8_MARLIN_MOE=0
export SGLANG_ROCM_USE_AITER_MOE=0
export W8A8_SUPPORT_METHODS=3
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export HIP_GRAPH_ACCUMULATE_DISPATCH=1
export HIP_H2D_DISABLE_COPY_BUFFER="${HIP_H2D_DISABLE_COPY_BUFFER:-0}"
export HIP_D2H_DISABLE_COPY_BUFFER="${HIP_D2H_DISABLE_COPY_BUFFER:-0}"
export HIP_H2D_DIRECT_COPY_THRESHOLD="${HIP_H2D_DIRECT_COPY_THRESHOLD:-32768}"
export HIP_H2D_HSAAPI_COPY_THRESHOLD="${HIP_H2D_HSAAPI_COPY_THRESHOLD:-32768}"
export HIP_D2H_DIRECT_COPY_THRESHOLD="${HIP_D2H_DIRECT_COPY_THRESHOLD:-512}"
export HIP_D2H_HSAAPI_COPY_THRESHOLD="${HIP_D2H_HSAAPI_COPY_THRESHOLD:-512}"
# export HIP_GRAPH_USE_CMD_CACHE=0
export SGLANG_USE_MODELSCOPE=1
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=3600
#export ROCSHMEM_DISABLE_HDP_FLUSH=1
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_HEAP_SIZE=419430400 # 400 MB
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=64
export DEEPEP_ENABLE_LL_LAYERED_OPT=1 # deepep分层
export DEEPEP_ENABLE_LL_LAYERED_DEVELOP=0
export ROCSHMEM_GDR_DISABLE_XDP=1
#export DEEPEP_ENABLE_MORI_EP_LL_LAYERED=1 #mori-ep分层
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_MAX_NUM_CONTEXTS=60
export ROCSHMEM_ALLOWED_IBV_DEVICES=shca_0,shca_1,shca_2,shca_3
export ROCSHMEM_TOPO_FILE_FORCE="${ROCSHMEM_TOPO_FILE_FORCE:-/hy4_work/configs/shca_topo.config}"
export HSA_USE_SVM=0

export GPU_MAX_HW_QUEUES="${GPU_MAX_HW_QUEUES:-4}"
export SGLANG_USE_W4A8_CONTIGUOUS_HIPC=1
# ===================== Mooncake / Disaggregation =====================
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export MC_ALLOWED_IBV_DEVICES=shca_0,shca_1,shca_2,shca_3
unset MC_GID_INDEX
unset SGLANG_SIMULATE_ACC_LEN SGLANG_SIMULATE_ACC_METHOD SGLANG_SIMULATE_ACC_TOKEN_MODE
export SGLANG_DEBUG_RADIX_CACHE_TIME=0

# SBO
export SGLANG_ROCM_USE_MULTI_STREAM=1

export SGLANG_DSA_HCU_INT8_INDEX_K_CACHE=1
export SGLANG_DSA_FUSE_TOPK=false
export SGLANG_USE_W4A8_MASKED_HIPC=1
export SGLANG_UVICORN_WORKER_HEALTHCHECK_TIMEOUT=120
export SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1
export MC_IB_GID_INDEX=0
export SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH="${SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH:-1}"
export SGLANG_SET_CPU_AFFINITY="${SGLANG_SET_CPU_AFFINITY:-1}"
export SGLANG_OPT_HY4_IHC_PRENORM=1
export SGLANG_OPT_HY4_IHC_TILELANG=1
export MC_TCP_ENABLE_CONNECTION_POOL=1
export SGLANG_DSA_HCU_LIGHTOP_TOPK="${SGLANG_DSA_HCU_LIGHTOP_TOPK:-1}"
export SGLANG_TORCH_PROFILER_DIR="${SGLANG_TORCH_PROFILER_DIR:-/hy4_work/prof}"

export HYV4_GATE_TUNING=off
export HYV4_MTP_INPUT_TUNING=1
export HYV4_HADAMARD_TUNING=1

export SGLANG_HY4_DCP2_FUSED_POST_ATTN=1
export SGLANG_HY4_DCP2_FUSED_VPROJ=1

export SGLANG_HY4_DCP_FUSED_KV_WRITE_LOCS=1
export SGLANG_HY4_HCU_DSA_SHORT_CONTEXT_K_ONLY=1

# topk_length
export SGLANG_ENABLE_DSA_FLASHMLA_DCP_GROUPED_TOPK=1

# # g_proj overlap
export SGLANG_ENABLE_HYV4_GATE_PROJ_PARALLEL=0
export SGLANG_HY4_DCP2_GATE_A2A_OVERLAP=1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# export ROCBLAS_LAYER=3
# export ROCBLAS_LOG_SPLIT_BY_HANDLE=1
# export ROCBLAS_LOG_BENCH_PATH=rocblas-bench.log
# # blast
# export HIPBLASLT_LOG_MASK=32
# export HIPBLASLT_LOG_FILE=hipblaslt.log

#export SGLANG_OPT_HY4_IHC_TILELANG=1
# 上下文长度读取 HY4 config.json，KV token 上限与 d1.sh 保持一致。
# 如需限制，可在调用脚本时追加 --context-length / --max-total-tokens。
# ===================== 启动 HY4 Decode：node-rank 1 =====================

python3 -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --trust-remote-code \
  --served-model-name "${SERVED_MODEL_NAME:-hy4}" \
  --host "${DECODE_BIND_HOST}" \
  --port "${DECODE_PORT}" \
  --dist-init-addr "${DIST_INIT_ADDR}" \
  --nnodes 4 \
  --node-rank 1 \
  --tp-size 32 \
  --dp-size 8 \
  --dcp-size 4 \
  --dcp-comm-backend a2a \
  --dcp-replicate-q-proj \
  --moe-dense-tp-size 1 \
  --ep-size 16 \
  --pp-size 1 \
  --attn-cp-size 1 \
  --enable-dp-attention \
  --custom-all-reduce-backend native \
  --moe-a2a-backend deepep \
  --deepep-mode low_latency \
  --page-size 64 \
  --disaggregation-transfer-backend mooncake \
  --json-model-override-args '{"index_share_for_mtp_iteration": true}' \
  --cuda-graph-config '{"decode":{"backend":"full","bs":[1,2,4,8,16,32,64,128]},"prefill":{"backend":"disabled"}}' \
  --attention-backend dsa \
  --dsa-decode-backend flashmla_kv \
  --dsa-paged-mqa-logits-backend lightop \
  --quantization slimquant_w4a8_marlin \
  --cuda-graph-backend-decode full \
  --cuda-graph-backend-prefill disabled \
  --dtype bfloat16 \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.80 \
  --skip-server-warmup \
  --max-running-requests 128 \
  --reasoning-parser hunyuan \
  --disaggregation-ib-device shca_0,shca_1,shca_2,shca_3 \
  --deepep-config /hy4_work/configs/ep_config.json \
  --enable-dp-lm-head \
  --disaggregation-mode decode \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --tokenizer-worker-num 8 \
  --enable-cache-report \
  --load-balance-method total_tokens \
  "$@" \
  2>&1 | tee -a "${LOG_FILE}"
```

##### D3

```bash
#!/bin/bash
set -euo pipefail
export PYTHONPATH=/hy4_work/sglang/python:${PYTHONPATH:-} 
# SGLANG_ROOT="${SGLANG_ROOT:-/hy4_work/sglang}"
MODEL_PATH="${MODEL_PATH:-/models/Hy4-preview-Channel-INT4-w4a8}"
DECODE_HOST="${DECODE_HOST:-10.211.10.13}"
DECODE_BIND_HOST="${DECODE_BIND_HOST:-${DECODE_HOST}}"
DECODE_PORT="${DECODE_PORT:-30001}"
DIST_INIT_ADDR="${DIST_INIT_ADDR:-10.211.10.8:5000}"
export SGLANG_HOST_IP="${SGLANG_HOST_IP:-${DECODE_HOST}}"
#export ROCBLAS_TENSILE_LIBPATH=/hy4_work/auto_select_tools/optimization_configs/new/config/library_gpu5
export LIGHTOP_PAGED_MQA_LOGITS_BACKEND=1
export LIGHTOP_PAGED_MQA_LOGITS_GROUP_SIZE=3
# shared indexer 显存优化
# export SGLANG_HY4_COMPACT_PD_INDEX_K=1
# ===================== RCCL / NCCL =====================
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=shca_0,shca_1,shca_2,shca_3
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"

unset NCCL_IB_GID_INDEX

export NCCL_NET_PLUGIN=shca
export NCCL_PLUGIN_P2P=ib
#export NCCL_TOPO_FILE=/opt/topo/built-in-508-topo-input-tj-default.xml

# 关闭 debug 刷屏
unset NCCL_DEBUG
unset NCCL_DEBUG_SUBSYS
unset NCCL_DEBUG_FILE
unset RCCL_DEBUG
unset RCCL_DEBUG_SUBSYS
unset TORCH_NCCL_DEBUG
export TORCH_CPP_LOG_LEVEL=ERROR
export PYTHONWARNINGS="ignore"
# Import the repository directly; disable bytecode writes on shared storage.
SGLANG_PYTHON="${SGLANG_ROOT}/python"
if [[ ! -f "${SGLANG_PYTHON}/sglang/launch_server.py" ]]; then
  printf 'SGLang Python source not found: %s\n' "${SGLANG_PYTHON}" >&2
  exit 1
fi
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  printf 'HY4 model config not found: %s/config.json\n' "${MODEL_PATH}" >&2
  exit 1
fi
export PYTHONPATH="${SGLANG_PYTHON}"
export PYTHONDONTWRITEBYTECODE=1
unset PYTHONPYCACHEPREFIX

# Put runtime files on the work volume, not the container root filesystem.
# Override SGLANG_RUNTIME_DIR with an absolute path on another writable volume.
SGLANG_RUNTIME_DIR="${SGLANG_RUNTIME_DIR:-${SGLANG_ROOT%/*}/runtime/node2}"
if [[ "${SGLANG_RUNTIME_DIR}" != /* ]]; then
  printf 'SGLANG_RUNTIME_DIR must be an absolute path: %s\n' "${SGLANG_RUNTIME_DIR}" >&2
  exit 1
fi
export TMPDIR="${SGLANG_RUNTIME_DIR}/tmp"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
export XDG_CACHE_HOME="${SGLANG_RUNTIME_DIR}/cache"
export TRITON_CACHE_DIR="${XDG_CACHE_HOME}/triton"
export TORCH_EXTENSIONS_DIR="${XDG_CACHE_HOME}/torch_extensions"
export TORCHINDUCTOR_CACHE_DIR="${XDG_CACHE_HOME}/torchinductor"
mkdir -p "${TMPDIR}" "${TRITON_CACHE_DIR}" "${TORCH_EXTENSIONS_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"
cd "${SGLANG_RUNTIME_DIR}"
ulimit -c 0
# Check real writes before loading torch or spawning GPU workers.
python3 - <<'PY_TMP_CHECK'
import os
import tempfile
with tempfile.TemporaryFile(dir=os.environ["TMPDIR"]) as probe:
    probe.write(b"sglang temporary directory check")
    probe.flush()
    os.fsync(probe.fileno())
PY_TMP_CHECK

# Keep service logs on the work volume.
SGLANG_LOG_DIR="${SGLANG_LOG_DIR:-/hy4_work/hy4_w4a8/dcp/dcp4/logs}"
mkdir -p "${SGLANG_LOG_DIR}"
LOG_FILE="${SGLANG_LOG_DIR}/hy4_w4a8_dcp2_dp16_ep32_node2_$(date +%Y%m%d_%H%M%S).log"
export PYTHONUNBUFFERED=1
# rocm tuning
#export ROCBLAS_TENSILE_LIBPATH=/hy4_work/auto_select_tools/optimization_configs/new/config/library_gpu5
# ===================== SGLang / 通用性能参数 =====================
unset SGLANG_ENABLE_SPEC_V2
unset SGLANG_ENABLE_PD_DECODE_STEPINFO_SYNC
export HSA_ENABLE_COREDUMP=0
export USE_DCU_CUSTOM_ALLREDUCE=0
export SGLANG_USE_AITER_AR=0
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/amdsmi:$LD_LIBRARY_PATH
export HIP_KERNEL_BATCH_CEILING=100
export GPU_FORCE_BLIT_COPY_SIZE=16
export HSA_KERNARG_POOL_SIZE=8388608
export HSA_FORCE_FINE_GRAIN_PCIE=1
export ROC_AQL_QUEUE_SIZE=131072
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_LIGHTOP_W4A8_MARLIN_MOE=0
export SGLANG_ROCM_USE_AITER_MOE=0
export W8A8_SUPPORT_METHODS=3
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export HIP_GRAPH_ACCUMULATE_DISPATCH=1
export HIP_H2D_DISABLE_COPY_BUFFER="${HIP_H2D_DISABLE_COPY_BUFFER:-0}"
export HIP_D2H_DISABLE_COPY_BUFFER="${HIP_D2H_DISABLE_COPY_BUFFER:-0}"
export HIP_H2D_DIRECT_COPY_THRESHOLD="${HIP_H2D_DIRECT_COPY_THRESHOLD:-32768}"
export HIP_H2D_HSAAPI_COPY_THRESHOLD="${HIP_H2D_HSAAPI_COPY_THRESHOLD:-32768}"
export HIP_D2H_DIRECT_COPY_THRESHOLD="${HIP_D2H_DIRECT_COPY_THRESHOLD:-512}"
export HIP_D2H_HSAAPI_COPY_THRESHOLD="${HIP_D2H_HSAAPI_COPY_THRESHOLD:-512}"
# export HIP_GRAPH_USE_CMD_CACHE=0
export SGLANG_USE_MODELSCOPE=1
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=3600
#export ROCSHMEM_DISABLE_HDP_FLUSH=1
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_HEAP_SIZE=419430400 # 400 MB
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=64
export DEEPEP_ENABLE_LL_LAYERED_OPT=1 # deepep分层
export DEEPEP_ENABLE_LL_LAYERED_DEVELOP=0
export ROCSHMEM_GDR_DISABLE_XDP=1
#export DEEPEP_ENABLE_MORI_EP_LL_LAYERED=1 #mori-ep分层
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_MAX_NUM_CONTEXTS=60
export ROCSHMEM_ALLOWED_IBV_DEVICES=shca_0,shca_1,shca_2,shca_3
export ROCSHMEM_TOPO_FILE_FORCE="${ROCSHMEM_TOPO_FILE_FORCE:-/hy4_work/configs/shca_topo.config}"
export HSA_USE_SVM=0

export GPU_MAX_HW_QUEUES="${GPU_MAX_HW_QUEUES:-4}"
export SGLANG_USE_W4A8_CONTIGUOUS_HIPC=1
# ===================== Mooncake / Disaggregation =====================
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export MC_ALLOWED_IBV_DEVICES=shca_0,shca_1,shca_2,shca_3
unset MC_GID_INDEX
unset SGLANG_SIMULATE_ACC_LEN SGLANG_SIMULATE_ACC_METHOD SGLANG_SIMULATE_ACC_TOKEN_MODE
#export SGLANG_OPT_HY4_IHC_TILELANG=1
export SGLANG_DEBUG_RADIX_CACHE_TIME=0

# SBO
export SGLANG_ROCM_USE_MULTI_STREAM=1

#unset SGLANG_NSA_HCU_INT8_INDEX_K_CACHE
export SGLANG_DSA_HCU_INT8_INDEX_K_CACHE=1
export SGLANG_DSA_FUSE_TOPK=false
export SGLANG_USE_W4A8_MASKED_HIPC=1
export SGLANG_UVICORN_WORKER_HEALTHCHECK_TIMEOUT=120
export SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1
export MC_IB_GID_INDEX=0
export SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH="${SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH:-1}"
export SGLANG_SET_CPU_AFFINITY="${SGLANG_SET_CPU_AFFINITY:-1}"
export SGLANG_OPT_HY4_IHC_PRENORM=1
export SGLANG_OPT_HY4_IHC_TILELANG=1
export MC_TCP_ENABLE_CONNECTION_POOL=1
export SGLANG_DSA_HCU_LIGHTOP_TOPK="${SGLANG_DSA_HCU_LIGHTOP_TOPK:-1}"
export SGLANG_TORCH_PROFILER_DIR="${SGLANG_TORCH_PROFILER_DIR:-/hy4_work/prof}"

export HYV4_GATE_TUNING=off
export HYV4_MTP_INPUT_TUNING=1
export HYV4_HADAMARD_TUNING=1

export SGLANG_HY4_DCP2_FUSED_POST_ATTN=1
export SGLANG_HY4_DCP2_FUSED_VPROJ=1

export SGLANG_HY4_DCP_FUSED_KV_WRITE_LOCS=1
export SGLANG_HY4_HCU_DSA_SHORT_CONTEXT_K_ONLY=1

# topk_length
export SGLANG_ENABLE_DSA_FLASHMLA_DCP_GROUPED_TOPK=1

# # g_proj overlap
export SGLANG_ENABLE_HYV4_GATE_PROJ_PARALLEL=0
export SGLANG_HY4_DCP2_GATE_A2A_OVERLAP=1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# export ROCBLAS_LAYER=3
# export ROCBLAS_LOG_SPLIT_BY_HANDLE=1
# export ROCBLAS_LOG_BENCH_PATH=rocblas-bench.log
# # blast
# export HIPBLASLT_LOG_MASK=32
# export HIPBLASLT_LOG_FILE=hipblaslt.log

# 上下文长度读取 HY4 config.json，KV token 上限与 d1.sh 保持一致。
# 如需限制，可在调用脚本时追加 --context-length / --max-total-tokens。
# ===================== 启动 HY4 Decode：node-rank 2 =====================
python3 -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --trust-remote-code \
  --served-model-name "${SERVED_MODEL_NAME:-hy4}" \
  --host "${DECODE_BIND_HOST}" \
  --port "${DECODE_PORT}" \
  --dist-init-addr "${DIST_INIT_ADDR}" \
  --nnodes 4 \
  --node-rank 2 \
  --tp-size 32 \
  --dp-size 8 \
  --dcp-size 4 \
  --dcp-comm-backend a2a \
  --dcp-replicate-q-proj \
  --moe-dense-tp-size 1 \
  --ep-size 16 \
  --pp-size 1 \
  --attn-cp-size 1 \
  --enable-dp-attention \
  --custom-all-reduce-backend native \
  --moe-a2a-backend deepep \
  --deepep-mode low_latency \
  --page-size 64 \
  --disaggregation-transfer-backend mooncake \
  --json-model-override-args '{"index_share_for_mtp_iteration": true}' \
  --cuda-graph-config '{"decode":{"backend":"full","bs":[1,2,4,8,16,32,64,128]},"prefill":{"backend":"disabled"}}' \
  --attention-backend dsa \
  --dsa-decode-backend flashmla_kv \
  --dsa-paged-mqa-logits-backend lightop \
  --quantization slimquant_w4a8_marlin \
  --cuda-graph-backend-decode full \
  --cuda-graph-backend-prefill disabled \
  --dtype bfloat16 \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.80 \
  --skip-server-warmup \
  --max-running-requests 128 \
  --reasoning-parser hunyuan \
  --disaggregation-ib-device shca_0,shca_1,shca_2,shca_3 \
  --deepep-config /hy4_work/configs/ep_config.json \
  --enable-dp-lm-head \
  --disaggregation-mode decode \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --tokenizer-worker-num 8 \
  --enable-cache-report \
  --load-balance-method total_tokens \
  "$@" \
  2>&1 | tee -a "${LOG_FILE}"
```

##### D4

```bash
#!/bin/bash
set -euo pipefail
export PYTHONPATH=/hy4_work/sglang/python:${PYTHONPATH:-} 
# SGLANG_ROOT="${SGLANG_ROOT:-/hy4_work/sglang}"
MODEL_PATH="${MODEL_PATH:-/models/Hy4-preview-Channel-INT4-w4a8}"
DECODE_HOST="${DECODE_HOST:-10.211.10.14}"
DECODE_BIND_HOST="${DECODE_BIND_HOST:-${DECODE_HOST}}"
DECODE_PORT="${DECODE_PORT:-30001}"
DIST_INIT_ADDR="${DIST_INIT_ADDR:-10.211.10.8:5000}"
export SGLANG_HOST_IP="${SGLANG_HOST_IP:-${DECODE_HOST}}"
# shared indexer 显存优化
# export SGLANG_HY4_COMPACT_PD_INDEX_K=1
#export ROCBLAS_TENSILE_LIBPATH=/hy4_work/auto_select_tools/optimization_configs/new/config/library_gpu5
export LIGHTOP_PAGED_MQA_LOGITS_BACKEND=1
export LIGHTOP_PAGED_MQA_LOGITS_GROUP_SIZE=3

# ===================== RCCL / NCCL =====================
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=shca_0,shca_1,shca_2,shca_3
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"

unset NCCL_IB_GID_INDEX

export NCCL_NET_PLUGIN=shca
export NCCL_PLUGIN_P2P=ib
#export NCCL_TOPO_FILE=/opt/topo/built-in-508-topo-input-tj-default.xml

# 关闭 debug 刷屏
unset NCCL_DEBUG
unset NCCL_DEBUG_SUBSYS
unset NCCL_DEBUG_FILE
unset RCCL_DEBUG
unset RCCL_DEBUG_SUBSYS
unset TORCH_NCCL_DEBUG
export TORCH_CPP_LOG_LEVEL=ERROR
export PYTHONWARNINGS="ignore"
# Import the repository directly; disable bytecode writes on shared storage.
SGLANG_PYTHON="${SGLANG_ROOT}/python"
if [[ ! -f "${SGLANG_PYTHON}/sglang/launch_server.py" ]]; then
  printf 'SGLang Python source not found: %s\n' "${SGLANG_PYTHON}" >&2
  exit 1
fi
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  printf 'HY4 model config not found: %s/config.json\n' "${MODEL_PATH}" >&2
  exit 1
fi
export PYTHONPATH="${SGLANG_PYTHON}"
export PYTHONDONTWRITEBYTECODE=1
unset PYTHONPYCACHEPREFIX

# Put runtime files on the work volume, not the container root filesystem.
# Override SGLANG_RUNTIME_DIR with an absolute path on another writable volume.
SGLANG_RUNTIME_DIR="${SGLANG_RUNTIME_DIR:-${SGLANG_ROOT%/*}/runtime/node3}"
if [[ "${SGLANG_RUNTIME_DIR}" != /* ]]; then
  printf 'SGLANG_RUNTIME_DIR must be an absolute path: %s\n' "${SGLANG_RUNTIME_DIR}" >&2
  exit 1
fi
export TMPDIR="${SGLANG_RUNTIME_DIR}/tmp"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
export XDG_CACHE_HOME="${SGLANG_RUNTIME_DIR}/cache"
export TRITON_CACHE_DIR="${XDG_CACHE_HOME}/triton"
export TORCH_EXTENSIONS_DIR="${XDG_CACHE_HOME}/torch_extensions"
export TORCHINDUCTOR_CACHE_DIR="${XDG_CACHE_HOME}/torchinductor"
mkdir -p "${TMPDIR}" "${TRITON_CACHE_DIR}" "${TORCH_EXTENSIONS_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"
cd "${SGLANG_RUNTIME_DIR}"
ulimit -c 0
# Check real writes before loading torch or spawning GPU workers.
python3 - <<'PY_TMP_CHECK'
import os
import tempfile
with tempfile.TemporaryFile(dir=os.environ["TMPDIR"]) as probe:
    probe.write(b"sglang temporary directory check")
    probe.flush()
    os.fsync(probe.fileno())
PY_TMP_CHECK

# Keep service logs on the work volume.
SGLANG_LOG_DIR="${SGLANG_LOG_DIR:-/hy4_work/hy4_w4a8/dcp/dcp4/logs}"
mkdir -p "${SGLANG_LOG_DIR}"
LOG_FILE="${SGLANG_LOG_DIR}/hy4_w4a8_dcp2_dp16_ep32_node3_$(date +%Y%m%d_%H%M%S).log"
export PYTHONUNBUFFERED=1
# rocm tuning
#export ROCBLAS_TENSILE_LIBPATH=/hy4_work/auto_select_tools/optimization_configs/new/config/library_gpu5
# ===================== SGLang / 通用性能参数 =====================
unset SGLANG_ENABLE_SPEC_V2
unset SGLANG_ENABLE_PD_DECODE_STEPINFO_SYNC
export HSA_ENABLE_COREDUMP=0
export USE_DCU_CUSTOM_ALLREDUCE=0
export SGLANG_USE_AITER_AR=0
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/amdsmi:$LD_LIBRARY_PATH
export HIP_KERNEL_BATCH_CEILING=100
export GPU_FORCE_BLIT_COPY_SIZE=16
export HSA_KERNARG_POOL_SIZE=8388608
export HSA_FORCE_FINE_GRAIN_PCIE=1
export ROC_AQL_QUEUE_SIZE=131072
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_LIGHTOP_W4A8_MARLIN_MOE=0
export SGLANG_ROCM_USE_AITER_MOE=0
export W8A8_SUPPORT_METHODS=3
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export HIP_GRAPH_ACCUMULATE_DISPATCH=1
export HIP_H2D_DISABLE_COPY_BUFFER="${HIP_H2D_DISABLE_COPY_BUFFER:-0}"
export HIP_D2H_DISABLE_COPY_BUFFER="${HIP_D2H_DISABLE_COPY_BUFFER:-0}"
export HIP_H2D_DIRECT_COPY_THRESHOLD="${HIP_H2D_DIRECT_COPY_THRESHOLD:-32768}"
export HIP_H2D_HSAAPI_COPY_THRESHOLD="${HIP_H2D_HSAAPI_COPY_THRESHOLD:-32768}"
export HIP_D2H_DIRECT_COPY_THRESHOLD="${HIP_D2H_DIRECT_COPY_THRESHOLD:-512}"
export HIP_D2H_HSAAPI_COPY_THRESHOLD="${HIP_D2H_HSAAPI_COPY_THRESHOLD:-512}"
# export HIP_GRAPH_USE_CMD_CACHE=0
export SGLANG_USE_MODELSCOPE=1
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=3600
#export ROCSHMEM_DISABLE_HDP_FLUSH=1
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_HEAP_SIZE=419430400 # 400 MB
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=64
export DEEPEP_ENABLE_LL_LAYERED_OPT=1 # deepep分层
export DEEPEP_ENABLE_LL_LAYERED_DEVELOP=0
export ROCSHMEM_GDR_DISABLE_XDP=1
#export DEEPEP_ENABLE_MORI_EP_LL_LAYERED=1 #mori-ep分层
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_MAX_NUM_CONTEXTS=60
export ROCSHMEM_ALLOWED_IBV_DEVICES=shca_0,shca_1,shca_2,shca_3
export ROCSHMEM_TOPO_FILE_FORCE="${ROCSHMEM_TOPO_FILE_FORCE:-/hy4_work/configs/shca_topo.config}"
export HSA_USE_SVM=0

# ===================== Mooncake / Disaggregation =====================
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export MC_ALLOWED_IBV_DEVICES=shca_0,shca_1,shca_2,shca_3
unset MC_GID_INDEX
unset SGLANG_SIMULATE_ACC_LEN SGLANG_SIMULATE_ACC_METHOD SGLANG_SIMULATE_ACC_TOKEN_MODE
# export SGLANG_OPT_HY4_IHC_TILELANG=1
export SGLANG_DEBUG_RADIX_CACHE_TIME=0

# SBO
export SGLANG_ROCM_USE_MULTI_STREAM=1

#unset SGLANG_NSA_HCU_INT8_INDEX_K_CACHE
export SGLANG_DSA_HCU_INT8_INDEX_K_CACHE=1
export SGLANG_DSA_FUSE_TOPK=false
export SGLANG_USE_W4A8_MASKED_HIPC=1
export SGLANG_UVICORN_WORKER_HEALTHCHECK_TIMEOUT=120
export SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1
export MC_IB_GID_INDEX=0
export SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH="${SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH:-1}"
export SGLANG_SET_CPU_AFFINITY="${SGLANG_SET_CPU_AFFINITY:-1}"
export SGLANG_OPT_HY4_IHC_PRENORM=1
export SGLANG_OPT_HY4_IHC_TILELANG=1
export MC_TCP_ENABLE_CONNECTION_POOL=1
export GPU_MAX_HW_QUEUES="${GPU_MAX_HW_QUEUES:-4}"
export SGLANG_USE_W4A8_CONTIGUOUS_HIPC=1
export SGLANG_DSA_HCU_LIGHTOP_TOPK="${SGLANG_DSA_HCU_LIGHTOP_TOPK:-1}"
export SGLANG_TORCH_PROFILER_DIR="${SGLANG_TORCH_PROFILER_DIR:-/hy4_work/prof}"

export HYV4_GATE_TUNING=off
export HYV4_MTP_INPUT_TUNING=1
export HYV4_HADAMARD_TUNING=1

export SGLANG_HY4_DCP2_FUSED_POST_ATTN=1
export SGLANG_HY4_DCP2_FUSED_VPROJ=1

export SGLANG_HY4_DCP_FUSED_KV_WRITE_LOCS=1
export SGLANG_HY4_HCU_DSA_SHORT_CONTEXT_K_ONLY=1

# topk_length
export SGLANG_ENABLE_DSA_FLASHMLA_DCP_GROUPED_TOPK=1

# # g_proj overlap
export SGLANG_ENABLE_HYV4_GATE_PROJ_PARALLEL=0
export SGLANG_HY4_DCP2_GATE_A2A_OVERLAP=1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# export ROCBLAS_LAYER=3
# export ROCBLAS_LOG_SPLIT_BY_HANDLE=1
# export ROCBLAS_LOG_BENCH_PATH=rocblas-bench.log
# # blast
# export HIPBLASLT_LOG_MASK=32
# export HIPBLASLT_LOG_FILE=hipblaslt.log

# 上下文长度读取 HY4 config.json，KV token 上限与 d1.sh 保持一致。
# 如需限制，可在调用脚本时追加 --context-length / --max-total-tokens。
# ===================== 启动 HY4 Decode：node-rank 3 =====================
python3 -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --trust-remote-code \
  --served-model-name "${SERVED_MODEL_NAME:-hy4}" \
  --host "${DECODE_BIND_HOST}" \
  --port "${DECODE_PORT}" \
  --dist-init-addr "${DIST_INIT_ADDR}" \
  --nnodes 4 \
  --node-rank 3 \
  --tp-size 32 \
  --dp-size 8 \
  --dcp-size 4 \
  --dcp-comm-backend a2a \
  --dcp-replicate-q-proj \
  --moe-dense-tp-size 1 \
  --ep-size 16 \
  --pp-size 1 \
  --attn-cp-size 1 \
  --enable-dp-attention \
  --custom-all-reduce-backend native \
  --moe-a2a-backend deepep \
  --deepep-mode low_latency \
  --page-size 64 \
  --disaggregation-transfer-backend mooncake \
  --json-model-override-args '{"index_share_for_mtp_iteration": true}' \
  --cuda-graph-config '{"decode":{"backend":"full","bs":[1,2,4,8,16,32,64,128]},"prefill":{"backend":"disabled"}}' \
  --attention-backend dsa \
  --dsa-decode-backend flashmla_kv \
  --dsa-paged-mqa-logits-backend lightop \
  --quantization slimquant_w4a8_marlin \
  --cuda-graph-backend-decode full \
  --cuda-graph-backend-prefill disabled \
  --dtype bfloat16 \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.80 \
  --skip-server-warmup \
  --max-running-requests 128 \
  --reasoning-parser hunyuan \
  --disaggregation-ib-device shca_0,shca_1,shca_2,shca_3 \
  --deepep-config /hy4_work/configs/ep_config.json \
  --enable-dp-lm-head \
  --disaggregation-mode decode \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --tokenizer-worker-num 8 \
  --enable-cache-report \
  --load-balance-method total_tokens \
  "$@" \
  2>&1 | tee -a "${LOG_FILE}"
```

#### Router

```bash
python -m sglang_router.launch_router     --pd-disaggregation     --policy round_robin     --host 0.0.0.0     --port 8000     --prefill http://10.211.10.11:30000     --decode http://10.211.10.8:30001
```

## API 调用

### IFB

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="hygon/Hy4-preview-Channel-INT4-w4a8",
    messages=[{"role": "user", "content": "中国的首都是哪里？"}],
    max_tokens=128,
)
```

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "hygon/Hy4-preview-Channel-INT4-w4a8", "messages": [{"role": "user", "content": "中国的首都是哪里？"}], "max_tokens": 128}'
```
