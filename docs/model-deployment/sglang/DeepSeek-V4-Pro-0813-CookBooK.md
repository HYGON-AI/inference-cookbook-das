# DeepSeek\-V4\-Pro\-0813 CookBooK

# 模型和环境配置

## 模型路径

郑州集群：

- W8A8\_FP8: https://www.modelscope.cn/models/hygon/DeepSeek-V4-Pro-0813-Channel-FP8-w8a8
- W4A8\_INF8:https://www.modelscope.cn/models/hygon/DeepSeek-V4-Pro-0813-Channel-INT4-w4a8

## 环境配置

1. SGLang仓库：https://github\.com/Sherlolo/sglang\-das/tree/main\-yangyn
2. EP有关文件配置：

   1. EPLB的权重

      [expert\_distribution\_recorder\_1782542259\.408945\.pt](图片和附件/expert_distribution_recorder_1782542259.408945.pt)
   2. deepep的参数

      [deepep\_config\.json](图片和附件/deepep_config.json)

# W4A8 风冷节点部署方案

## IFB部署

### ifb\-tp8 / 8 卡

```SQL
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16
export SGLANG_OPT_USE_FUSED_STORE_CACHE=false
export SGLANG_OPT_USE_FUSED_HASH_TOPK=true
export SGLANG_OPT_SWIGLU_CLAMP_FUSION=false
export SGLANG_TOPK_TRANSFORM_512_TORCH=false
export SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK=true
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export SGLANG_USE_AITER_AG=0
export ROCSHMEM_DISABLE_HDP_FLUSH=1
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_HEAP_SIZE=3173741824
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export SGLANG_ROCM_USE_AITER_MOE=1
export SGLANG_USE_OPT_CAT=1
export SGLANG_USE_FUSED_MLA_CAT=1
export SGLANG_USE_LIGHTOP_GROUP_FP8_QUANT=0
export SGLANG_USE_FUSED_DPSKV4_SILU_MUL_FP8_QUANT=0
export SGLANG_USE_LINEAR_BF16_FP32_USE_BLASLT=1
export SGLANG_APPLY_CONFIG_BACKUP=none
export SGLANG_ROCM_USE_AITER_TILELANG_MHC=1
export SGLANG_DSV4_SPLIT_PREFILL_DECODE_MLA=1
export SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=0
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_DPSKV4_LIGHTOP_QUANT_K_CACHE=1
export SGLANG_USE_DPSKV4_LIGHTOP_RMSNORM=1
export SGLANG_USE_FUSED_DPSKV4_QNORM_ROPE_KV_ROPE_QUANT=1

export SGLANG_DSV4_HCU_INT8_INDEX_K_CACHE=1
export SGLANG_LIGHTOP_TOPK=1


export SGLANG_USE_LIGHTOP_EP_MOE_ALIGN=1
export SGLANG_USE_LIGHTOP_EP_SCATTER=1
export SGLANG_USE_LIGHTOP_EP_GATHER=1
export SGLANG_USE_LIGHTOP_TOPK_IDS_POSTPROCESS=1
export SGLANG_OPT_FP8_WO_A_GEMM=0
export SGLANG_RAGGED_VERIFY_MODE=static
export SGLANG_DSPARK_CONFIDENCE_RELAY_LAG_STEPS=2
export SGLANG_DSPARK_OPT_MARKOV_W2_TP_SHARD=1
export SGLANG_DSPARK_ENABLE_MULTI_STREAM=1
export SGLANG_DSPARK_FAST_KERNEL=1
export SGLANG_DSPARK_FAST_SAMPLING=1
export SGLANG_W4A8_TPMOE_BACKEND=aiter
 
sglang serve \
  --model-path /public/home/wanglong3/DeepSeek-V4-Pro-0813-INT4-Channel \
  --model-loader-extra-config '{"enable_multithread_load": "true","num_threads": 4}' \
  --trust-remote-code \
  --quantization slimquant_marlin \
  --moe-runner-backend aiter \
  --dtype bfloat16 \
  --tp-size 8 \
  --pp-size 1 \
  --dp-size 1 \
  --host 0.0.0.0 \
  --port 31888 \
  --numa-node 0 0 0 0 1 1 1 1 \
  --mem-fraction-static 0.85 \
  --attention-backend fa3 \
  --chunked-prefill-size 4096 \
  --max-prefill-tokens 4096 \
  --max-running-requests 4 \
  --context-length 32768 \
  --disable-custom-all-reduce \
  --disable-shared-experts-fusion \
  2>&1 | tee running_dpsk-v4.log
```

### P\-IFB\(cp8ep8\) / 8 卡

```Markdown
#!/usr/bin/env bash
set -o pipefail

# --- paths / network ---
MODEL_PATH=/public/home/wanglong3/DeepSeek-V4-Pro-0813-INT4-Channel
DEEPEP_CONFIG=/public/home/yangyn1/work_space/2026/sglang_main/package/deepep_config.json


# --- sglang runtime ---
export SGLANG_SET_CPU_AFFINITY=1
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_USE_LIGHTOP=1
export SGLANG_ROCM_USE_AITER_MOE=1
export SGLANG_ROCM_USE_AITER_TILELANG_MHC=1
export SGLANG_GROUPGEMM=true
export SGLANG_USE_FP8_W8A8_MOE=0
unset SGLANG_DEEPEP_BF16_DISPATCH
export SGLANG_USE_LIGHTOP_EP_MOE_ALIGN=1
export SGLANG_USE_LIGHTOP_EP_SCATTER=1
export SGLANG_USE_LIGHTOP_EP_GATHER=1
export SGLANG_USE_LIGHTOP_TOPK_IDS_POSTPROCESS=1
export SGLANG_USE_LIGHTOP_GROUP_FP8_QUANT=0
export SGLANG_OPT_USE_FUSED_HASH_TOPK=true
export SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK=true
export SGLANG_TOPK_TRANSFORM_512_TORCH=false
export SGLANG_OPT_SWIGLU_CLAMP_FUSION=false
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export USE_DCU_CUSTOM_ALLREDUCE=0
export SGLANG_OPT_FP8_WO_A_GEMM=0
export SGLANG_USE_LINEAR_BF16_FP32_USE_BLASLT=1
export SGLANG_USE_OPT_CAT=1
export SGLANG_USE_FUSED_MLA_CAT=1
export SGLANG_USE_DPSKV4_LIGHTOP_QUANT_K_CACHE=1
export SGLANG_USE_DPSKV4_LIGHTOP_RMSNORM=1
export SGLANG_USE_FUSED_DPSKV4_QNORM_ROPE_KV_ROPE_QUANT=1
export SGLANG_USE_FUSED_DPSKV4_SILU_MUL_FP8_QUANT=0
export SGLANG_APPLY_CONFIG_BACKUP=none
export SGLANG_USE_AITER_AG=0
export SGLANG_W4A8_EP_USE_GROUPGEMM=true
export SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=1
export SGLANG_DSV4_SPLIT_PREFILL_DECODE_MLA=1
export SGLANG_DSV4_HCU_USE_BF16_FLASH_MLA=0
export SGLANG_DSV4_HCU_USE_LIGHTOP_BF16_GATHER=0
export SGLANG_USE_W4A8_CONTIGUOUS_HIPC=1
export SGLANG_USE_LIGHTOP_W4A8_MARLIN_MOE=false
export SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1

export SGLANG_LIGHTOP_KVALLOC_KERNEL=1
export W8A8_SUPPORT_METHODS=3

export ROCSHMEM_MAX_NUM_CONTEXTS=48
export MC_ENABLE_DEST_DEVICE_AFFINITY=1


sglang serve \
  --reasoning-parser deepseek-v4 \
  --tool-call-parser deepseekv4 \
  --model-path ${MODEL_PATH} \
  --model-loader-extra-config "{\"enable_multithread_load\": \"true\",\"num_threads\": 8}" \
  --trust-remote-code \
  --quantization slimquant_marlin \
  --tp 8 \
  --dp 1 \
  --enable-prefill-cp \
  --cp-strategy interleave \
  --ep 8 \
  --moe-dense-tp-size 1 \
  --moe-a2a-backend deepep \
  --moe-runner-backend aiter \
  --deepep-mode normal \
  --deepep-config ${DEEPEP_CONFIG} \
  --host 0.0.0.0 \
  --port 31200 \
  --watchdog-timeout 3600 \
  --max-running-requests 32 \
  --disable-cuda-graph \
  --disable-radix-cache \
  --disable-flashinfer-autotune \
  --chunked-prefill-size 8192 \
  --max-total-tokens 60000 \
  --max-prefill-tokens 8192 \
  --mem-fraction-static 0.96 \
  --swa-full-tokens-ratio 0.55 \
  --kv-cache-dtype auto \
  2>&1 | tee ifb-p-cp8ep8.log
```

### D\-IFB\(tp8ep8\+dspark\) / 8 卡

备注:

- dp8ep8\+dspark方案显存不够 切换为 tp8 的方案
- dp16ep16\+dspark 可以正常运行 待节点验证

```Bash
#!/usr/bin/env bash
set -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_PYTHON_DIR="$(cd -- "${SCRIPT_DIR}/../../sglang-das/python" && pwd)"
export PYTHONPATH="${SGLANG_PYTHON_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

# P+D DSpark uses Prefill-generated Draft KV. Do not allocate hidden pools.
unset SGLANG_PD_HIDDEN_POOL_TOKENS
unset SGLANG_PD_HIDDEN_RECV_POOL_TOKENS

export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
# Keep Decode radix cache disabled until the experimental DSV4/DSpark path
# preserves the no-cache accuracy baseline.
export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0
export SGLANG_UVICORN_WORKER_HEALTHCHECK_TIMEOUT=120
export SGLANG_SET_CPU_AFFINITY=1
export SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_OPT_USE_FUSED_STORE_CACHE=false
export SGLANG_OPT_USE_FUSED_HASH_TOPK=true
export SGLANG_OPT_SWIGLU_CLAMP_FUSION=false
export SGLANG_TOPK_TRANSFORM_512_TORCH=false
export SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK=true
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export SGLANG_USE_AITER_AG=0
export ROCSHMEM_DISABLE_HDP_FLUSH=1
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288

export ROCSHMEM_HEAP_SIZE=1610612736
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=64

export SGLANG_ROCM_USE_AITER_MOE=1
export SGLANG_USE_OPT_CAT=1
export SGLANG_USE_FUSED_MLA_CAT=1
export SGLANG_USE_LIGHTOP_GROUP_FP8_QUANT=0
export SGLANG_USE_FUSED_DPSKV4_SILU_MUL_FP8_QUANT=0
export SGLANG_USE_LINEAR_BF16_FP32_USE_BLASLT=1
export SGLANG_APPLY_CONFIG_BACKUP=none
export SGLANG_ROCM_USE_AITER_TILELANG_MHC=1
export SGLANG_DSV4_SPLIT_PREFILL_DECODE_MLA=0
export SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=0
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_DPSKV4_LIGHTOP_QUANT_K_CACHE=1
export SGLANG_USE_DPSKV4_LIGHTOP_RMSNORM=1
export SGLANG_USE_FUSED_DPSKV4_QNORM_ROPE_KV_ROPE_QUANT=1
export SGLANG_USE_LIGHTOP_EP_MOE_ALIGN=1
export SGLANG_USE_LIGHTOP_EP_SCATTER=1
export SGLANG_USE_LIGHTOP_EP_GATHER=1
export SGLANG_USE_LIGHTOP_TOPK_IDS_POSTPROCESS=1
export SGLANG_OPT_FP8_WO_A_GEMM=0

export SGLANG_RAGGED_VERIFY_MODE=static
export SGLANG_DSPARK_CONFIDENCE_RELAY_LAG_STEPS=2
export SGLANG_DSPARK_OPT_MARKOV_W2_TP_SHARD=1
export SGLANG_DSPARK_ENABLE_MULTI_STREAM=1
export SGLANG_DSPARK_FAST_KERNEL=1
export SGLANG_DSPARK_FAST_SAMPLING=1
export DEEPEP_ENABLE_LL_LAYERED_OPT=1

export SGLANG_LIGHTOP_KVALLOC_KERNEL=1
export W8A8_SUPPORT_METHODS=3
export SGLANG_USE_W4A8_MASKED_HIPC=1
export SGLANG_USE_LIGHTOP_W4A8_MARLIN_MOE=false


sglang serve \
  --tp-size 8 \
  --dp 1 \
  --ep 8 \
  --watchdog-timeout 3600 \
  --port 31200 \
  --model-path /public/home/wanglong3/DeepSeek-V4-Pro-0813-INT4-Channel \
  --model-loader-extra-config '{"enable_multithread_load":"true","num_threads":8}' \
  --trust-remote-code \
  --chunked-prefill-size 8192 \
  --max-total-tokens 65172 \
  --disable-flashinfer-autotune \
  --cuda-graph-max-bs 32 \
  --mem-fraction-static 0.91 \
  --max-running-requests 32 \
  --enable-metrics \
  --quantization slimquant_marlin \
  --load-balance-method auto \
  --moe-a2a-backend deepep \
  --moe-runner-backend deep_gemm \
  --deepep-mode low_latency \
  --speculative-algorithm DSPARK \
  --speculative-num-steps 1 \
  --speculative-eagle-topk 1 \
  --speculative-moe-a2a-backend deepep \
  --speculative-moe-runner-backend deep_gemm \
  2>&1 | tee ifb-p-tp8ep8.log
```

## PD部署

PD 分离脚本完整继承上面的 Pro W4A8 P-IFB 和 D-IFB 配置，仅增加
Mooncake PD 分离所需参数。P、D 节点的 RDMA 设备名必须按各自机器实际情况设置。

### Prefill（P-IFB cp8ep8）

```Bash
#!/usr/bin/env bash
set -o pipefail

# --- paths / network ---
MODEL_PATH=/public/home/wanglong3/DeepSeek-V4-Pro-0813-INT4-Channel
DEEPEP_CONFIG=/public/home/yangyn1/work_space/2026/sglang_main/package/deepep_config.json
BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-8998}"
DISAGGREGATION_IB_DEVICE="${DISAGGREGATION_IB_DEVICE:?请设置 P 节点的 RDMA 网卡列表}"


# --- sglang runtime ---
export SGLANG_SET_CPU_AFFINITY=1
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_USE_LIGHTOP=1
export SGLANG_ROCM_USE_AITER_MOE=1
export SGLANG_ROCM_USE_AITER_TILELANG_MHC=1
export SGLANG_GROUPGEMM=true
export SGLANG_USE_FP8_W8A8_MOE=0
unset SGLANG_DEEPEP_BF16_DISPATCH
export SGLANG_USE_LIGHTOP_EP_MOE_ALIGN=1
export SGLANG_USE_LIGHTOP_EP_SCATTER=1
export SGLANG_USE_LIGHTOP_EP_GATHER=1
export SGLANG_USE_LIGHTOP_TOPK_IDS_POSTPROCESS=1
export SGLANG_USE_LIGHTOP_GROUP_FP8_QUANT=0
export SGLANG_OPT_USE_FUSED_HASH_TOPK=true
export SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK=true
export SGLANG_TOPK_TRANSFORM_512_TORCH=false
export SGLANG_OPT_SWIGLU_CLAMP_FUSION=false
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export USE_DCU_CUSTOM_ALLREDUCE=0
export SGLANG_OPT_FP8_WO_A_GEMM=0
export SGLANG_USE_LINEAR_BF16_FP32_USE_BLASLT=1
export SGLANG_USE_OPT_CAT=1
export SGLANG_USE_FUSED_MLA_CAT=1
export SGLANG_USE_DPSKV4_LIGHTOP_QUANT_K_CACHE=1
export SGLANG_USE_DPSKV4_LIGHTOP_RMSNORM=1
export SGLANG_USE_FUSED_DPSKV4_QNORM_ROPE_KV_ROPE_QUANT=1
export SGLANG_USE_FUSED_DPSKV4_SILU_MUL_FP8_QUANT=0
export SGLANG_APPLY_CONFIG_BACKUP=none
export SGLANG_USE_AITER_AG=0
export SGLANG_W4A8_EP_USE_GROUPGEMM=true
export SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=1
export SGLANG_DSV4_SPLIT_PREFILL_DECODE_MLA=1
export SGLANG_DSV4_HCU_USE_BF16_FLASH_MLA=0
export SGLANG_DSV4_HCU_USE_LIGHTOP_BF16_GATHER=0
export SGLANG_USE_W4A8_CONTIGUOUS_HIPC=1
export SGLANG_USE_LIGHTOP_W4A8_MARLIN_MOE=false
export SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1

export SGLANG_LIGHTOP_KVALLOC_KERNEL=1
export W8A8_SUPPORT_METHODS=3

export ROCSHMEM_MAX_NUM_CONTEXTS=48
export MC_ENABLE_DEST_DEVICE_AFFINITY=1


sglang serve \
  --reasoning-parser deepseek-v4 \
  --tool-call-parser deepseekv4 \
  --model-path ${MODEL_PATH} \
  --model-loader-extra-config "{\"enable_multithread_load\": \"true\",\"num_threads\": 8}" \
  --trust-remote-code \
  --quantization slimquant_marlin \
  --tp 8 \
  --dp 1 \
  --enable-prefill-cp \
  --cp-strategy interleave \
  --ep 8 \
  --moe-dense-tp-size 1 \
  --moe-a2a-backend deepep \
  --moe-runner-backend aiter \
  --deepep-mode normal \
  --deepep-config ${DEEPEP_CONFIG} \
  --host 0.0.0.0 \
  --port 31200 \
  --watchdog-timeout 3600 \
  --max-running-requests 32 \
  --disable-cuda-graph \
  --disable-radix-cache \
  --disable-flashinfer-autotune \
  --chunked-prefill-size 8192 \
  --max-total-tokens 60000 \
  --max-prefill-tokens 8192 \
  --mem-fraction-static 0.96 \
  --swa-full-tokens-ratio 0.55 \
  --kv-cache-dtype auto \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-bootstrap-port "${BOOTSTRAP_PORT}" \
  --disaggregation-ib-device "${DISAGGREGATION_IB_DEVICE}" \
  2>&1 | tee ifb-p-cp8ep8.log
```

### Decode（D-IFB tp8ep8+DSpark）

```Bash
#!/usr/bin/env bash
set -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_PYTHON_DIR="$(cd -- "${SCRIPT_DIR}/../../sglang-das/python" && pwd)"
export PYTHONPATH="${SGLANG_PYTHON_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-8998}"
DISAGGREGATION_IB_DEVICE="${DISAGGREGATION_IB_DEVICE:?请设置 D 节点的 RDMA 网卡列表}"

# P+D DSpark uses Prefill-generated Draft KV. Do not allocate hidden pools.
unset SGLANG_PD_HIDDEN_POOL_TOKENS
unset SGLANG_PD_HIDDEN_RECV_POOL_TOKENS

export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
# Keep Decode radix cache disabled until the experimental DSV4/DSpark path
# preserves the no-cache accuracy baseline.
export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0
export SGLANG_UVICORN_WORKER_HEALTHCHECK_TIMEOUT=120
export SGLANG_SET_CPU_AFFINITY=1
export SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_OPT_USE_FUSED_STORE_CACHE=false
export SGLANG_OPT_USE_FUSED_HASH_TOPK=true
export SGLANG_OPT_SWIGLU_CLAMP_FUSION=false
export SGLANG_TOPK_TRANSFORM_512_TORCH=false
export SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK=true
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export SGLANG_USE_AITER_AG=0
export ROCSHMEM_DISABLE_HDP_FLUSH=1
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288

export ROCSHMEM_HEAP_SIZE=1610612736
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=64

export SGLANG_ROCM_USE_AITER_MOE=1
export SGLANG_USE_OPT_CAT=1
export SGLANG_USE_FUSED_MLA_CAT=1
export SGLANG_USE_LIGHTOP_GROUP_FP8_QUANT=0
export SGLANG_USE_FUSED_DPSKV4_SILU_MUL_FP8_QUANT=0
export SGLANG_USE_LINEAR_BF16_FP32_USE_BLASLT=1
export SGLANG_APPLY_CONFIG_BACKUP=none
export SGLANG_ROCM_USE_AITER_TILELANG_MHC=1
export SGLANG_DSV4_SPLIT_PREFILL_DECODE_MLA=0
export SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=0
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_DPSKV4_LIGHTOP_QUANT_K_CACHE=1
export SGLANG_USE_DPSKV4_LIGHTOP_RMSNORM=1
export SGLANG_USE_FUSED_DPSKV4_QNORM_ROPE_KV_ROPE_QUANT=1
export SGLANG_USE_LIGHTOP_EP_MOE_ALIGN=1
export SGLANG_USE_LIGHTOP_EP_SCATTER=1
export SGLANG_USE_LIGHTOP_EP_GATHER=1
export SGLANG_USE_LIGHTOP_TOPK_IDS_POSTPROCESS=1
export SGLANG_OPT_FP8_WO_A_GEMM=0

export SGLANG_RAGGED_VERIFY_MODE=static
export SGLANG_DSPARK_CONFIDENCE_RELAY_LAG_STEPS=2
export SGLANG_DSPARK_OPT_MARKOV_W2_TP_SHARD=1
export SGLANG_DSPARK_ENABLE_MULTI_STREAM=1
export SGLANG_DSPARK_FAST_KERNEL=1
export SGLANG_DSPARK_FAST_SAMPLING=1
export DEEPEP_ENABLE_LL_LAYERED_OPT=1

export SGLANG_LIGHTOP_KVALLOC_KERNEL=1
export W8A8_SUPPORT_METHODS=3
export SGLANG_USE_W4A8_MASKED_HIPC=1
export SGLANG_USE_LIGHTOP_W4A8_MARLIN_MOE=false


sglang serve \
  --tp-size 8 \
  --dp 1 \
  --ep 8 \
  --watchdog-timeout 3600 \
  --port 31200 \
  --model-path /public/home/wanglong3/DeepSeek-V4-Pro-0813-INT4-Channel \
  --model-loader-extra-config '{"enable_multithread_load":"true","num_threads":8}' \
  --trust-remote-code \
  --chunked-prefill-size 8192 \
  --max-total-tokens 65172 \
  --disable-flashinfer-autotune \
  --cuda-graph-max-bs 32 \
  --mem-fraction-static 0.91 \
  --max-running-requests 32 \
  --enable-metrics \
  --quantization slimquant_marlin \
  --load-balance-method auto \
  --moe-a2a-backend deepep \
  --moe-runner-backend deep_gemm \
  --deepep-mode low_latency \
  --speculative-algorithm DSPARK \
  --speculative-num-steps 1 \
  --speculative-eagle-topk 1 \
  --speculative-moe-a2a-backend deepep \
  --speculative-moe-runner-backend deep_gemm \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-bootstrap-port "${BOOTSTRAP_PORT}" \
  --disaggregation-ib-device "${DISAGGREGATION_IB_DEVICE}" \
  2>&1 | tee ifb-p-tp8ep8.log
```

### Router

P、D 服务部署在不同节点，服务端口均沿用对应 IFB 脚本的 `31200`。

```Bash
#!/usr/bin/env bash
set -euo pipefail

PREFILL_IP="${PREFILL_IP:?请设置 P 服务 IP}"
DECODE_IP="${DECODE_IP:?请设置 D 服务 IP}"
ROUTER_PORT="${ROUTER_PORT:-30000}"

python3 -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill "http://${PREFILL_IP}:31200" \
  --decode "http://${DECODE_IP}:31200" \
  --policy round_robin \
  --host 0.0.0.0 \
  --port "${ROUTER_PORT}"
```

# W8A8\_FP8 风冷节点部署方案（BW1100）

## IFB部署

### P\-IFB-CP8EP8PP2

```Bash
#!/usr/bin/env bash

set -euo pipefail

# Usage:
#   bash ifb-p-doc-0902-chunkmasss--xiugai.sh 10.16.1.2 0
#   bash ifb-p-doc-0902-chunkmasss--xiugai.sh 10.16.1.2 1
#   bash ifb-p-doc-0902-chunk4096.sh 0
if [[ $# -eq 2 ]]; then
  MASTER_ADDR="*$1*"
  NODE_RANK="*$2*"
elif [[ $# -eq 1 ]]; then
  MASTER_ADDR="${MASTER_ADDR:-10.16.1.15}"
  NODE_RANK="*$1*"
else
  echo "Usage: *$0* [MASTER_ADDR] NODE_RANK" >&2
  exit 2
fi

if [[ "$NODE_RANK" != "0" && "$NODE_RANK" != "1" ]]; then
  echo "NODE_RANK must be 0 or 1, got: $NODE_RANK" >&2
  exit 2
fi

# 参数配置
export MASTER_ADDR="$MASTER_ADDR"
export MASTER_PORT=22000
export NNODES=2
export DIST_INIT_ADDR="${MASTER_ADDR}:${MASTER_PORT}"
export NODE_RANK="$NODE_RANK"
export HOST=0.0.0.0
export PORT=30100
export TP=8
export PP=2
export EP_SIZE=8
export CP_SIZE=8

# 模型权重路径配置
export MODEL_PATH=/llm-models/DeepSeek-V4-Pro-0813-FP8-Channel-v2
export TOKENIZER_PATH="$MODEL_PATH"
export DEEPEP_CONFIG=/public/home/mass/yangyn/package/deepep_config.json
export EPLP_PATH=/public/home/mass/yangyn/package/expert_distribution_recorder_1782542259.408945.pt

# 网卡配置
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ens19f0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-ens19f0}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_8:1,mlx5_9:1}"
export ROCSHMEM_ALLOWED_IBV_DEVICES="${ROCSHMEM_ALLOWED_IBV_DEVICES:-mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_8,mlx5_9}"
export MC_ALLOWED_IBV_DEVICES="${MC_ALLOWED_IBV_DEVICES:-mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_8,mlx5_9}"
export ROCSHMEM_IB_GID_INDEX="${ROCSHMEM_IB_GID_INDEX:-0}"

export ROCSHMEM_MAX_NUM_CONTEXTS=48 # 48 or 60
export ROCSHMEM_DISABLE_HDP_FLUSH=1
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export USE_DCU_CUSTOM_ALLREDUCE=0
export GPU_MAX_HW_QUEUES=2
export HIP_KERNEL_BATCH_CEILING=100
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export ROCSHMEM_HEAP_SIZE=3173741824

# SGLang环境变量配置
export SGLANG_DSV4_REQUEST_SCOPED_C128_STATE=true
export SGLANG_OPT_USE_ONLINE_COMPRESS=false
export SGLANG_TOPK_TRANSFORM_512_TORCH=false
export SGLANG_OPT_USE_FUSED_STORE_CACHE=false
export SGLANG_OPT_USE_FUSED_HASH_TOPK=true
export SGLANG_OPT_SWIGLU_CLAMP_FUSION=false
export SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK=true
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export SGLANG_DSV4_SPLIT_PREFILL_DECODE_MLA=1
export SGLANG_DSV4_SPLIT_HCA_NONSPARSE_MLA=false
export SGLANG_DSV4_SPARSE_PREFILL_SINGLE_CALL=false
export SGLANG_DSV4_SPARSE_PREFILL_TRITON_GATHER=false
export SGLANG_DISABLED_MODEL_ARCHS=midashenglm
export SGLANG_APPLY_CONFIG_BACKUP=none
export SGLANG_LIGHTOP_KVALLOC_KERNEL=1
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_LINEAR_BF16_FP32_USE_BLASLT=1
export SGLANG_USE_FP8_W8A8_MOE=1
export SGLANG_USE_DEEPGEMM_MOE=1
export SGLANG_ROCM_USE_AITER_MOE=1
export SGLANG_ROCM_USE_AITER_TILELANG_MHC=1
export SGLANG_USE_DPSKV4_LIGHTOP_RMSNORM=1
export SGLANG_USE_LIGHTOP_EP_SCATTER=true
export SGLANG_USE_LIGHTOP_EP_GATHER=true
export SGLANG_USE_LIGHTOP_EP_MOE_ALIGN=true
export SGLANG_USE_OPT_CAT=1
export SGLANG_USE_FUSED_MLA_CAT=1
export SGLANG_USE_LIGHTOP_GROUP_FP8_QUANT=1
export SGLANG_USE_DPSKV4_LIGHTOP_QUANT_K_CACHE=1
export SGLANG_RAGGED_VERIFY_MODE=static
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256
export SGLANG_USE_FUSED_DPSKV4_SILU_MUL_FP8_QUANT=1
export SGLANG_USE_FUSED_DPSKV4_QNORM_ROPE_KV_ROPE_QUANT=1
export SGLANG_OPT_FP8_WO_A_GEMM=0
export SGLANG_OPT_USE_TILELANG_MHC_PRE=1
export SGLANG_OPT_USE_TILELANG_MHC_POST=1
export SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=0
export SGLANG_USE_LIGHTOP_TOPK_IDS_POSTPROCESS=1

sglang serve \
  --trust-remote-code \
  --disable-radix-cache \
  --model-path "$MODEL_PATH" \
  --tokenizer-path "$TOKENIZER_PATH" \
  --nnodes "$NNODES" \
  --node-rank "$NODE_RANK" \
  --dist-init-addr "$DIST_INIT_ADDR" \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --chunked-prefill-size 16384 \
  --max-prefill-tokens 16384 \
  --context-length 1048576 \
  --max-running-requests 256 \
  --skip-server-warmup \
  --disable-flashinfer-autotune \
  --disable-cuda-graph \
  --host "$HOST" \
  --port "$PORT" \
  --mem-fraction-static 0.93 \
  --tp-size "$TP" \
  --pp-size "$PP" \
  --ep-size "$EP_SIZE" \
  --enable-nsa-prefill-context-parallel \
  --nsa-prefill-cp-mode round-robin-split \
  --attn-cp-size "$CP_SIZE" \
  --enable-dp-attention \
  --dp 1 \
  --enable-prefill-cp \
  --cp-strategy interleave \
  --init-expert-location "$EPLP_PATH" \
  --ep-dispatch-algorithm static \
  --ep-num-redundant-experts 8 \
  --eplb-algorithm deepseek \
  --deepep-config "$DEEPEP_CONFIG" \
  --moe-a2a-backend deepep \
  --deepep-mode auto \
  2>&1 | tee "./p-logs-mass/running_ifb-p_node—${NODE_RANK}_$(date +%Y%m%d_%H%M%S)_ifb-p-debug--0.9333333-32---1.log"
```

### D\-IFB-DP16EP16\+DSPARK

```Bash
#!/usr/bin/env bash

set -euo pipefail

# Usage:
#   node 0: bash ifb-d-dspark.sh 10.16.1.2 0
#   node 1: bash ifb-d-dspark.sh 10.16.1.2 1
# The rank-only form is also supported when MASTER_ADDR is exported.
if [[ $# -ge 2 ]]; then
  MASTER_ADDR="*${1}*"
  NODE_RANK="*${2}*"
elif [[ $# -eq 1 && ( "*$1*" == "0" || "*$1*" == "1" ) ]]; then
  MASTER_ADDR="${MASTER_ADDR:-10.16.1.2}"
  NODE_RANK="*${1}*"
else
  MASTER_ADDR="${MASTER_ADDR:-*${1*:-*10*.*16*.*1*.*2}*}"
  NODE_RANK="${NODE_RANK:-0}"
fi

if [[ "${NODE_RANK}" != "0" && "${NODE_RANK}" != "1" ]]; then
  echo "NODE_RANK must be 0 or 1, got: ${NODE_RANK}" >&2
  exit 2
fi

# 参数设置
export MASTER_ADDR="${MASTER_ADDR}"
export NODE_RANK="${NODE_RANK}"
export NNODES=2
export MASTER_PORT="${MASTER_PORT:-22000}"
export DIST_INIT_ADDR="${MASTER_ADDR}:${MASTER_PORT}"
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-30100}"
export TP=16
export PP=1
export EP_SIZE=16
export DP_SIZE=16
export MOE_DENSE_TP_SIZE=1

# 权重路径
export MODEL_PATH=/llm-models/DeepSeek-V4-Pro-0813-FP8-Channel-v2
export DRAFT_MODEL_PATH="$MODEL_PATH"
export TOKENIZER_PATH="$MODEL_PATH"
export EPLP_PATH=/public/home/mass/yangyn/package/expert_distribution_recorder_1782542259.408945.pt

# 网卡配置 需要根据实际情况修改
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ens14f0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-ens14f0}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_0:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1}"
export ROCSHMEM_ALLOWED_IBV_DEVICES="${ROCSHMEM_ALLOWED_IBV_DEVICES:-mlx5_0,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8}"
export MC_ALLOWED_IBV_DEVICES="${MC_ALLOWED_IBV_DEVICES:-mlx5_0,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8}"


# 网卡配置
export ROCSHMEM_DISABLE_HDP_FLUSH=1
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export ROCSHMEM_HEAP_SIZE=2000000000

# deepep配置
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=64

# sglang配置
unset SGLANG_ENABLE_UNIFIED_RADIX_TREE
unset SGLANG_EXPERIMENTAL_DSV4_DECODE_RADIX_CACHE
unset SGLANG_DEBUG_DSV4_DECODE_RADIX_TRANSFER
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0
export SGLANG_UVICORN_WORKER_HEALTHCHECK_TIMEOUT=120
export SGLANG_SET_CPU_AFFINITY=1
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_LIGHTOP_TOPK=1
export SGLANG_OPT_USE_FUSED_STORE_CACHE=false
export SGLANG_OPT_USE_FUSED_HASH_TOPK=true
export SGLANG_OPT_SWIGLU_CLAMP_FUSION=false
export SGLANG_TOPK_TRANSFORM_512_TORCH=false
export SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK=true
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export SGLANG_USE_AITER_AG=0
export SGLANG_USE_FP8_W8A8_MOE=1
export SGLANG_USE_DEEPGEMM_MOE=1
export SGLANG_ROCM_USE_AITER_MOE=1
export SGLANG_USE_LIGHTOP_EP_MOE_ALIGN=1
export SGLANG_USE_OPT_CAT=1
export SGLANG_USE_FUSED_MLA_CAT=1
export SGLANG_USE_LIGHTOP_GROUP_FP8_QUANT=1
export SGLANG_USE_FUSED_DPSKV4_SILU_MUL_FP8_QUANT=1
export SGLANG_USE_LINEAR_BF16_FP32_USE_BLASLT=1
export SGLANG_APPLY_CONFIG_BACKUP=none
export SGLANG_ROCM_USE_AITER_TILELANG_MHC=1
export SGLANG_DSV4_SPLIT_PREFILL_DECODE_MLA=1 # 1 or 0
export SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=1 # 1 or 0
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_DPSKV4_LIGHTOP_QUANT_K_CACHE=1
export SGLANG_USE_DPSKV4_LIGHTOP_RMSNORM=1
export SGLANG_USE_FUSED_DPSKV4_QNORM_ROPE_KV_ROPE_QUANT=1
export SGLANG_USE_LIGHTOP_EP_SCATTER=1
export SGLANG_USE_LIGHTOP_EP_GATHER=1
export SGLANG_USE_LIGHTOP_TOPK_IDS_POSTPROCESS=1
export SGLANG_OPT_FP8_WO_A_GEMM=0
export SGLANG_RAGGED_VERIFY_MODE=static
export SGLANG_DSPARK_CONFIDENCE_RELAY_LAG_STEPS=2
export SGLANG_DSPARK_OPT_MARKOV_W2_TP_SHARD=1
export SGLANG_DSPARK_ENABLE_MULTI_STREAM=1
export SGLANG_DSPARK_FAST_KERNEL=1
export SGLANG_DSPARK_FAST_SAMPLING=1
export DEEPEP_ENABLE_LL_LAYERED_OPT=1 #带确定
export SGLANG_DSV4_HCU_INT8_INDEX_K_CACHE=0

# v4pro特定的配置
export SGLANG_DSV4_REQUEST_SCOPED_C128_STATE=true
export SGLANG_OPT_USE_ONLINE_COMPRESS=false
export SGLANG_DSV4_SPLIT_HCA_NONSPARSE_MLA=false
export SGLANG_DSV4_SPARSE_PREFILL_SINGLE_CALL=false
export SGLANG_DSV4_SPARSE_PREFILL_TRITON_GATHER=false
export SGLANG_LIGHTOP_KVALLOC_KERNEL=1
export SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=false
export SGLANG_USE_LIGHTOP_TOPK_IDS_POSTPROCESS=1

# dspark配置
export SGLANG_RAGGED_VERIFY_MODE=static
export SGLANG_DSPARK_CONFIDENCE_RELAY_LAG_STEPS=2
export SGLANG_DSPARK_OPT_MARKOV_W2_TP_SHARD=1
export SGLANG_DSPARK_ENABLE_MULTI_STREAM=1
export SGLANG_DSPARK_FAST_KERNEL=1
export SGLANG_DSPARK_FAST_SAMPLING=1

sglang serve \
 --trust-remote-code \
 --disable-radix-cache \
 --model-path "${MODEL_PATH}" \
 --tokenizer-path "${TOKENIZER_PATH}" \
 --nnodes "${NNODES}" \
 --node-rank "${NODE_RANK}" \
 --dist-init-addr "${DIST_INIT_ADDR}" \
 --dist-timeout 10000 \
 --watchdog-timeout 3600 \
 --chunked-prefill-size 4096 \
 --max-prefill-tokens 4096 \
 --max-running-requests 32 \
 --disable-flashinfer-autotune \
 --disable-cuda-graph \
 --cuda-graph-max-bs 32 \
 --cuda-graph-max-bs-decode 32 \
 --max-total-tokens 32768 \
 --skip-server-warmup \
 --host "${HOST}" \
 --port "${PORT}" \
 --mem-fraction-static "${MEM_FRACTION_STATIC:-0.99}" \
 --tp-size 16 \
 --pp-size 1 \
 --ep-size 16 \
 --dp-size 16 \
 --moe-dense-tp-size 1 \
 --enable-dp-attention \
 --enable-dp-lm-head \
 --moe-a2a-backend deepep \
 --moe-runner-backend deep_gemm \
 --deepep-mode low_latency\
 --speculative-algorithm DSPARK \
 --speculative-num-steps 1 \
 --speculative-eagle-topk 1 \
 --speculative-moe-a2a-backend deepep \
 --speculative-moe-runner-backend deep_gemm \
 2>&1 | tee "running_ifb-d_node${NODE_RANK}.log"
```

## PD部署

PD 分离脚本完整继承上面的 Pro W8A8 P-IFB 和 D-IFB 配置，仅增加
Mooncake PD 分离所需参数。未使用 Flash 部署中的模型、并行度、token/batch
或算子配置。

### Prefill（P-IFB CP8EP8PP2）

```Bash
#!/usr/bin/env bash

set -euo pipefail

# Usage:
#   bash ifb-p-doc-0902-chunkmasss--xiugai.sh 10.16.1.2 0
#   bash ifb-p-doc-0902-chunkmasss--xiugai.sh 10.16.1.2 1
#   bash ifb-p-doc-0902-chunk4096.sh 0
if [[ $# -eq 2 ]]; then
  MASTER_ADDR="${1}"
  NODE_RANK="${2}"
elif [[ $# -eq 1 ]]; then
  MASTER_ADDR="${MASTER_ADDR:-10.16.1.15}"
  NODE_RANK="${1}"
else
  echo "Usage: $0 [MASTER_ADDR] NODE_RANK" >&2
  exit 2
fi

if [[ "$NODE_RANK" != "0" && "$NODE_RANK" != "1" ]]; then
  echo "NODE_RANK must be 0 or 1, got: $NODE_RANK" >&2
  exit 2
fi

# 参数配置
export MASTER_ADDR="$MASTER_ADDR"
export MASTER_PORT=22000
export NNODES=2
export DIST_INIT_ADDR="${MASTER_ADDR}:${MASTER_PORT}"
export NODE_RANK="$NODE_RANK"
export HOST=0.0.0.0
export PORT=30100
export TP=8
export PP=2
export EP_SIZE=8
export CP_SIZE=8

# 模型权重路径配置
export MODEL_PATH=/llm-models/DeepSeek-V4-Pro-0813-FP8-Channel-v2
export TOKENIZER_PATH="$MODEL_PATH"
export DEEPEP_CONFIG=/public/home/mass/yangyn/package/deepep_config.json
export EPLP_PATH=/public/home/mass/yangyn/package/expert_distribution_recorder_1782542259.408945.pt

# 网卡配置
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ens19f0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-ens19f0}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_8:1,mlx5_9:1}"
export ROCSHMEM_ALLOWED_IBV_DEVICES="${ROCSHMEM_ALLOWED_IBV_DEVICES:-mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_8,mlx5_9}"
export MC_ALLOWED_IBV_DEVICES="${MC_ALLOWED_IBV_DEVICES:-mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_8,mlx5_9}"
export BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-8998}"
export DISAGGREGATION_IB_DEVICE="${DISAGGREGATION_IB_DEVICE:-${MC_ALLOWED_IBV_DEVICES}}"
export ROCSHMEM_IB_GID_INDEX="${ROCSHMEM_IB_GID_INDEX:-0}"

export ROCSHMEM_MAX_NUM_CONTEXTS=48 # 48 or 60
export ROCSHMEM_DISABLE_HDP_FLUSH=1
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export USE_DCU_CUSTOM_ALLREDUCE=0
export GPU_MAX_HW_QUEUES=2
export HIP_KERNEL_BATCH_CEILING=100
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export ROCSHMEM_HEAP_SIZE=3173741824

# SGLang环境变量配置
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1
export SGLANG_DSV4_REQUEST_SCOPED_C128_STATE=true
export SGLANG_OPT_USE_ONLINE_COMPRESS=false
export SGLANG_TOPK_TRANSFORM_512_TORCH=false
export SGLANG_OPT_USE_FUSED_STORE_CACHE=false
export SGLANG_OPT_USE_FUSED_HASH_TOPK=true
export SGLANG_OPT_SWIGLU_CLAMP_FUSION=false
export SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK=true
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export SGLANG_DSV4_SPLIT_PREFILL_DECODE_MLA=1
export SGLANG_DSV4_SPLIT_HCA_NONSPARSE_MLA=false
export SGLANG_DSV4_SPARSE_PREFILL_SINGLE_CALL=false
export SGLANG_DSV4_SPARSE_PREFILL_TRITON_GATHER=false
export SGLANG_DISABLED_MODEL_ARCHS=midashenglm
export SGLANG_APPLY_CONFIG_BACKUP=none
export SGLANG_LIGHTOP_KVALLOC_KERNEL=1
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_LINEAR_BF16_FP32_USE_BLASLT=1
export SGLANG_USE_FP8_W8A8_MOE=1
export SGLANG_USE_DEEPGEMM_MOE=1
export SGLANG_ROCM_USE_AITER_MOE=1
export SGLANG_ROCM_USE_AITER_TILELANG_MHC=1
export SGLANG_USE_DPSKV4_LIGHTOP_RMSNORM=1
export SGLANG_USE_LIGHTOP_EP_SCATTER=true
export SGLANG_USE_LIGHTOP_EP_GATHER=true
export SGLANG_USE_LIGHTOP_EP_MOE_ALIGN=true
export SGLANG_USE_OPT_CAT=1
export SGLANG_USE_FUSED_MLA_CAT=1
export SGLANG_USE_LIGHTOP_GROUP_FP8_QUANT=1
export SGLANG_USE_DPSKV4_LIGHTOP_QUANT_K_CACHE=1
export SGLANG_RAGGED_VERIFY_MODE=static
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256
export SGLANG_USE_FUSED_DPSKV4_SILU_MUL_FP8_QUANT=1
export SGLANG_USE_FUSED_DPSKV4_QNORM_ROPE_KV_ROPE_QUANT=1
export SGLANG_OPT_FP8_WO_A_GEMM=0
export SGLANG_OPT_USE_TILELANG_MHC_PRE=1
export SGLANG_OPT_USE_TILELANG_MHC_POST=1
export SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=0
export SGLANG_USE_LIGHTOP_TOPK_IDS_POSTPROCESS=1

sglang serve \
  --trust-remote-code \
  --disable-radix-cache \
  --model-path "$MODEL_PATH" \
  --tokenizer-path "$TOKENIZER_PATH" \
  --nnodes "$NNODES" \
  --node-rank "$NODE_RANK" \
  --dist-init-addr "$DIST_INIT_ADDR" \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --chunked-prefill-size 16384 \
  --max-prefill-tokens 16384 \
  --context-length 1048576 \
  --max-running-requests 256 \
  --skip-server-warmup \
  --disable-flashinfer-autotune \
  --disable-cuda-graph \
  --host "$HOST" \
  --port "$PORT" \
  --mem-fraction-static 0.93 \
  --tp-size "$TP" \
  --pp-size "$PP" \
  --ep-size "$EP_SIZE" \
  --enable-nsa-prefill-context-parallel \
  --nsa-prefill-cp-mode round-robin-split \
  --attn-cp-size "$CP_SIZE" \
  --enable-dp-attention \
  --dp 1 \
  --enable-prefill-cp \
  --cp-strategy interleave \
  --init-expert-location "$EPLP_PATH" \
  --ep-dispatch-algorithm static \
  --ep-num-redundant-experts 8 \
  --eplb-algorithm deepseek \
  --deepep-config "$DEEPEP_CONFIG" \
  --moe-a2a-backend deepep \
  --deepep-mode auto \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-bootstrap-port "${BOOTSTRAP_PORT}" \
  --disaggregation-ib-device "${DISAGGREGATION_IB_DEVICE}" \
  2>&1 | tee "./p-logs-mass/running_ifb-p_node—${NODE_RANK}_$(date +%Y%m%d_%H%M%S)_ifb-p-debug--0.9333333-32---1.log"
```

### Decode（D-IFB DP16EP16+DSpark）

```Bash
#!/usr/bin/env bash

set -euo pipefail

# Usage:
#   node 0: bash ifb-d-dspark.sh 10.16.1.2 0
#   node 1: bash ifb-d-dspark.sh 10.16.1.2 1
# The rank-only form is also supported when MASTER_ADDR is exported.
if [[ $# -ge 2 ]]; then
  MASTER_ADDR="${1}"
  NODE_RANK="${2}"
elif [[ $# -eq 1 && ( "${1}" == "0" || "${1}" == "1" ) ]]; then
  MASTER_ADDR="${MASTER_ADDR:-10.16.1.2}"
  NODE_RANK="${1}"
else
  MASTER_ADDR="${MASTER_ADDR:-10.16.1.2}"
  NODE_RANK="${NODE_RANK:-0}"
fi

if [[ "${NODE_RANK}" != "0" && "${NODE_RANK}" != "1" ]]; then
  echo "NODE_RANK must be 0 or 1, got: ${NODE_RANK}" >&2
  exit 2
fi

# 参数设置
export MASTER_ADDR="${MASTER_ADDR}"
export NODE_RANK="${NODE_RANK}"
export NNODES=2
export MASTER_PORT="${MASTER_PORT:-22000}"
export DIST_INIT_ADDR="${MASTER_ADDR}:${MASTER_PORT}"
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-30100}"
export TP=16
export PP=1
export EP_SIZE=16
export DP_SIZE=16
export MOE_DENSE_TP_SIZE=1

# 权重路径
export MODEL_PATH=/llm-models/DeepSeek-V4-Pro-0813-FP8-Channel-v2
export DRAFT_MODEL_PATH="$MODEL_PATH"
export TOKENIZER_PATH="$MODEL_PATH"
export EPLP_PATH=/public/home/mass/yangyn/package/expert_distribution_recorder_1782542259.408945.pt

# 网卡配置 需要根据实际情况修改
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ens14f0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-ens14f0}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_0:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1}"
export ROCSHMEM_ALLOWED_IBV_DEVICES="${ROCSHMEM_ALLOWED_IBV_DEVICES:-mlx5_0,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8}"
export MC_ALLOWED_IBV_DEVICES="${MC_ALLOWED_IBV_DEVICES:-mlx5_0,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8}"
export BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-8998}"
export DISAGGREGATION_IB_DEVICE="${DISAGGREGATION_IB_DEVICE:-${MC_ALLOWED_IBV_DEVICES}}"


# 网卡配置
export ROCSHMEM_DISABLE_HDP_FLUSH=1
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export ROCSHMEM_HEAP_SIZE=2000000000

# deepep配置
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=64

# sglang配置
unset SGLANG_ENABLE_UNIFIED_RADIX_TREE
unset SGLANG_EXPERIMENTAL_DSV4_DECODE_RADIX_CACHE
unset SGLANG_DEBUG_DSV4_DECODE_RADIX_TRANSFER
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1
export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0
export SGLANG_UVICORN_WORKER_HEALTHCHECK_TIMEOUT=120
export SGLANG_SET_CPU_AFFINITY=1
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_LIGHTOP_TOPK=1
export SGLANG_OPT_USE_FUSED_STORE_CACHE=false
export SGLANG_OPT_USE_FUSED_HASH_TOPK=true
export SGLANG_OPT_SWIGLU_CLAMP_FUSION=false
export SGLANG_TOPK_TRANSFORM_512_TORCH=false
export SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK=true
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export SGLANG_USE_AITER_AG=0
export SGLANG_USE_FP8_W8A8_MOE=1
export SGLANG_USE_DEEPGEMM_MOE=1
export SGLANG_ROCM_USE_AITER_MOE=1
export SGLANG_USE_LIGHTOP_EP_MOE_ALIGN=1
export SGLANG_USE_OPT_CAT=1
export SGLANG_USE_FUSED_MLA_CAT=1
export SGLANG_USE_LIGHTOP_GROUP_FP8_QUANT=1
export SGLANG_USE_FUSED_DPSKV4_SILU_MUL_FP8_QUANT=1
export SGLANG_USE_LINEAR_BF16_FP32_USE_BLASLT=1
export SGLANG_APPLY_CONFIG_BACKUP=none
export SGLANG_ROCM_USE_AITER_TILELANG_MHC=1
export SGLANG_DSV4_SPLIT_PREFILL_DECODE_MLA=1 # 1 or 0
export SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=1 # 1 or 0
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_DPSKV4_LIGHTOP_QUANT_K_CACHE=1
export SGLANG_USE_DPSKV4_LIGHTOP_RMSNORM=1
export SGLANG_USE_FUSED_DPSKV4_QNORM_ROPE_KV_ROPE_QUANT=1
export SGLANG_USE_LIGHTOP_EP_SCATTER=1
export SGLANG_USE_LIGHTOP_EP_GATHER=1
export SGLANG_USE_LIGHTOP_TOPK_IDS_POSTPROCESS=1
export SGLANG_OPT_FP8_WO_A_GEMM=0
export SGLANG_RAGGED_VERIFY_MODE=static
export SGLANG_DSPARK_CONFIDENCE_RELAY_LAG_STEPS=2
export SGLANG_DSPARK_OPT_MARKOV_W2_TP_SHARD=1
export SGLANG_DSPARK_ENABLE_MULTI_STREAM=1
export SGLANG_DSPARK_FAST_KERNEL=1
export SGLANG_DSPARK_FAST_SAMPLING=1
export DEEPEP_ENABLE_LL_LAYERED_OPT=1 #带确定
export SGLANG_DSV4_HCU_INT8_INDEX_K_CACHE=0

# v4pro特定的配置
export SGLANG_DSV4_REQUEST_SCOPED_C128_STATE=true
export SGLANG_OPT_USE_ONLINE_COMPRESS=false
export SGLANG_DSV4_SPLIT_HCA_NONSPARSE_MLA=false
export SGLANG_DSV4_SPARSE_PREFILL_SINGLE_CALL=false
export SGLANG_DSV4_SPARSE_PREFILL_TRITON_GATHER=false
export SGLANG_LIGHTOP_KVALLOC_KERNEL=1
export SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=false
export SGLANG_USE_LIGHTOP_TOPK_IDS_POSTPROCESS=1

# dspark配置
export SGLANG_RAGGED_VERIFY_MODE=static
export SGLANG_DSPARK_CONFIDENCE_RELAY_LAG_STEPS=2
export SGLANG_DSPARK_OPT_MARKOV_W2_TP_SHARD=1
export SGLANG_DSPARK_ENABLE_MULTI_STREAM=1
export SGLANG_DSPARK_FAST_KERNEL=1
export SGLANG_DSPARK_FAST_SAMPLING=1

sglang serve \
 --trust-remote-code \
 --disable-radix-cache \
 --model-path "${MODEL_PATH}" \
 --tokenizer-path "${TOKENIZER_PATH}" \
 --nnodes "${NNODES}" \
 --node-rank "${NODE_RANK}" \
 --dist-init-addr "${DIST_INIT_ADDR}" \
 --dist-timeout 10000 \
 --watchdog-timeout 3600 \
 --chunked-prefill-size 4096 \
 --max-prefill-tokens 4096 \
 --max-running-requests 32 \
 --disable-flashinfer-autotune \
 --disable-cuda-graph \
 --cuda-graph-max-bs 32 \
 --cuda-graph-max-bs-decode 32 \
 --max-total-tokens 32768 \
 --skip-server-warmup \
 --host "${HOST}" \
 --port "${PORT}" \
 --mem-fraction-static "${MEM_FRACTION_STATIC:-0.99}" \
 --tp-size 16 \
 --pp-size 1 \
 --ep-size 16 \
 --dp-size 16 \
 --moe-dense-tp-size 1 \
 --enable-dp-attention \
 --enable-dp-lm-head \
 --moe-a2a-backend deepep \
 --moe-runner-backend deep_gemm \
 --deepep-mode low_latency \
 --speculative-algorithm DSPARK \
 --speculative-num-steps 1 \
 --speculative-eagle-topk 1 \
 --speculative-moe-a2a-backend deepep \
 --speculative-moe-runner-backend deep_gemm \
 --disaggregation-mode decode \
 --disaggregation-transfer-backend mooncake \
 --disaggregation-bootstrap-port "${BOOTSTRAP_PORT}" \
 --disaggregation-ib-device "${DISAGGREGATION_IB_DEVICE}" \
 2>&1 | tee "running_ifb-d_node${NODE_RANK}.log"
```

### Router

先启动 P 集群和 D 集群的所有节点，再启动 router。P、D 服务端口均沿用
对应 IFB 脚本的 `30100`，但位于不同节点组。

```Bash
#!/usr/bin/env bash
set -euo pipefail

PREFILL_IP="${PREFILL_IP:?请设置 P 集群服务 IP}"
DECODE_IP="${DECODE_IP:?请设置 D 集群服务 IP}"
ROUTER_PORT="${ROUTER_PORT:-30000}"

python3 -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill "http://${PREFILL_IP}:30100" \
  --decode "http://${DECODE_IP}:30100" \
  --policy round_robin \
  --host 0.0.0.0 \
  --port "${ROUTER_PORT}"
```
