# Step-5-Preview on vLLM

## 模型简介

本文提供 Step-5-Preview-Channel-FP8-w8a8 在 HCU 上的文本推理部署示例，采用 Channel FP8 W8A8 权重、FP8 KV cache 和 MTP3，包含单机 IFB、单机 Prefill / Decode，以及四节点超节点 Prefill / Decode。

Prefill 和 Decode 是分别侧重输入处理、输出生成的独立 API 服务配置。本文未配置跨实例 KV cache 传输和 PD Proxy。

## 模型列表

| 模型权重 | 量化方式 | vLLM 镜像 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | --------- | -------- | ---- | -------- | -------- |
| [hygon/Step-5-Preview-Channel-FP8-w8a8](https://www.modelscope.cn/models/hygon/Step-5-Preview-Channel-FP8-w8a8) | Channel FP8 W8A8 + FP8 KV cache | [0.28.1](https://github.com/HYGON-AI/inference-cookbook-das/blob/main/docs/model-deployment/docker_images.md) | BW1100 144GB（单机） | 8 | IFB | [**`>_`**](#single-ifb) |
| [hygon/Step-5-Preview-Channel-FP8-w8a8](https://www.modelscope.cn/models/hygon/Step-5-Preview-Channel-FP8-w8a8) | Channel FP8 W8A8 + FP8 KV cache | [0.28.1](https://github.com/HYGON-AI/inference-cookbook-das/blob/main/docs/model-deployment/docker_images.md) | BW1100 144GB（单机） | 8 | Prefill | [**`>_`**](#single-prefill) |
| [hygon/Step-5-Preview-Channel-FP8-w8a8](https://www.modelscope.cn/models/hygon/Step-5-Preview-Channel-FP8-w8a8) | Channel FP8 W8A8 + FP8 KV cache | [0.28.1](https://github.com/HYGON-AI/inference-cookbook-das/blob/main/docs/model-deployment/docker_images.md) | BW1100 144GB（单机） | 8 | Decode | [**`>_`**](#single-decode) |
| [hygon/Step-5-Preview-Channel-FP8-w8a8](https://www.modelscope.cn/models/hygon/Step-5-Preview-Channel-FP8-w8a8) | Channel FP8 W8A8 + FP8 KV cache | [0.28.1](https://github.com/HYGON-AI/inference-cookbook-das/blob/main/docs/model-deployment/docker_images.md) | gfx938 超节点（4 节点，每节点 4 卡） | 16 | Prefill | [**`>_`**](#super-prefill) |
| [hygon/Step-5-Preview-Channel-FP8-w8a8](https://www.modelscope.cn/models/hygon/Step-5-Preview-Channel-FP8-w8a8) | Channel FP8 W8A8 + FP8 KV cache | [0.28.1](https://github.com/HYGON-AI/inference-cookbook-das/blob/main/docs/model-deployment/docker_images.md) | gfx938 超节点（4 节点，每节点 4 卡） | 16 | Decode | [**`>_`**](#super-decode) |

## 启动命令

每套命令在独立容器或未加载其他配置环境变量的终端执行。同一组卡选择一套配置运行；各命令中的 IP、网卡、模型目录及配置文件路径按实际部署位置替换。



### Step-5-Preview-Channel-FP8-w8a8 IFB BW1100 8x vLLM 0.28.1

单机使用 8 张 BW1100，配置 TP4 / DP2 / EP8，采用 Triton MoE、eager 执行和 MTP3，上下文上限为 65536 token。

```bash
set -euo pipefail

export VLLM_USE_MODELSCOPE=1
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_HCU_USE_CUSTOM_OPS=1
export VLLM_USE_V2_MODEL_RUNNER=1
export NCCL_IB_DISABLE=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export GPU_MAX_HW_QUEUES=4
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_FLASH_ATTN_UNIFIED=1
export VLLM_USE_PD_SPLIT=1
export VLLM_HCU_USE_LIGHTOP_EP_SCATTER=0
export VLLM_ROCM_USE_AITER_MOE=1
export VLLM_USE_FLASH_ATTN_FP8=1
export VLLM_HCU_USE_LIGHTOP_MOE_ALIGN=1
export FA_DECODE_NUM_SPLITS=2
export VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=0
export VLLM_HCU_EP_FUSED_COMBINE=1
export VLLM_HCU_DP_CUSTOM_AG=1
export VLLM_HCU_STEP5_SWA_FUSED_QK=1
export VLLM_HCU_STEP5_FUSED_GATE=1
export VLLM_HCU_TRITON_TOKEN_FP8_QUANT=1
export VLLM_HCU_PIN_MEMORY=1
export HIP_FORCE_DEV_KERNARG=1

vllm serve hygon/Step-5-Preview-Channel-FP8-w8a8 \
  --served-model-name step5 \
  --host 0.0.0.0 \
  --reasoning-parser step3p5 \
  --tokenizer hygon/Step-5-Preview-Channel-FP8-w8a8 \
  --tensor-parallel-size 4 \
  --data-parallel-size 2 \
  --moe-backend triton \
  --enable-expert-parallel \
  --max-num-batched-tokens 32768 \
  --max-model-len 65536 \
  --max-num-seqs 64 \
  --no-enable-prefix-caching \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 3, "use_local_argmax_reduction": true}' \
  --enforce-eager \
  --compilation-config '{"mode":0,"cudagraph_mode":"NONE","custom_ops":["+swiglustep_and_mul"]}' \
  --kv-cache-dtype fp8_e4m3 \
  --limit-mm-per-prompt '{"image": 0}' \
  --gpu-memory-utilization 0.85
```



### Step-5-Preview-Channel-FP8-w8a8 Prefill BW1100 8x vLLM 0.28.1

单机使用 8 张 BW1100，配置 TP1 / DP8 / EP8，启用静态专家放置、MTP prefill tail、direct combine 和 DBO。调度 token 预算为 8192，上下文上限为 140000 token。

```bash
set -euo pipefail

export VLLM_USE_MODELSCOPE=1
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_HCU_USE_CUSTOM_OPS=1
export VLLM_USE_V2_MODEL_RUNNER=1
export NCCL_IB_DISABLE=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export GPU_MAX_HW_QUEUES=4
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_FLASH_ATTN_UNIFIED=1
export VLLM_USE_PD_SPLIT=1
export VLLM_ROCM_USE_AITER_MOE=1
export VLLM_USE_FLASH_ATTN_FP8=1
export VLLM_HCU_USE_LIGHTOP_MOE_ALIGN=1
export FA_DECODE_NUM_SPLITS=2
export VLLM_HCU_EP_FUSED_MOE=1
export VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=0
export VLLM_HCU_EP_FUSED_COMBINE=1
export VLLM_HCU_DP_CUSTOM_AG=1
export VLLM_HCU_STEP5_FP8_DENSE=1
export VLLM_HCU_STEP5_FP8_BF16_MOE=1
export VLLM_HCU_STEP5_SWA_FUSED_QK=1
export VLLM_HCU_STEP5_FUSED_GATE=1
export VLLM_HCU_TRITON_TOKEN_FP8_QUANT=1
export VLLM_HCU_STEP5_QKV_INDEXER=1
export VLLM_HCU_PIN_MEMORY=1
export HIP_FORCE_DEV_KERNARG=1
export VLLM_STEP4_DSA_GRAPH_REGIONS=0
export EP_FUSED_MOE_TABLE=/opt/step5-cookbook/assets/step5_ep8_moe_table.json
test -r "$EP_FUSED_MOE_TABLE"
export VLLM_STEP5_EXPERT_PLACEMENT=/opt/step5-cookbook/assets/expert_placement_prefill.pt
test -r "$VLLM_STEP5_EXPERT_PLACEMENT"
export HCU_DP_MOE_PIPE_MIN_TOKENS=16384
export NCCL_MAX_NCHANNELS=8
export HCU_MTP_PREFILL_TAIL=1536
export HCU_EP_DIRECT_SUM=16384
export HCU_DBO_MIN_TOKENS=8192
export HCU_SHARED_EXPERTS_STREAM_MIN_TOKENS=4096

vllm serve hygon/Step-5-Preview-Channel-FP8-w8a8 \
  --served-model-name step5 \
  --host 0.0.0.0 \
  --reasoning-parser step3p5 \
  --tokenizer hygon/Step-5-Preview-Channel-FP8-w8a8 \
  --port 8001 \
  --tensor-parallel-size 1 \
  --data-parallel-size 8 \
  --enable-expert-parallel \
  --max-num-batched-tokens 8192 \
  --max-model-len 140000 \
  --max-num-seqs 64 \
  --no-enable-prefix-caching \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 3, "use_local_argmax_reduction": true}' \
  --max-cudagraph-capture-size 64 \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["+swiglustep_and_mul"],"compile_sizes":[16,64]}' \
  --kv-cache-dtype fp8_e4m3 \
  --limit-mm-per-prompt '{"image": 0}' \
  --gpu-memory-utilization 0.875 \
  --api-server-count 1
```



### Step-5-Preview-Channel-FP8-w8a8 Decode BW1100 8x vLLM 0.28.1

单机使用 8 张 BW1100，配置 TP4 / DP2 / EP8，采用 MTP3，图捕获规模上限为 256，调度 token 预算为 32768，上下文上限为 140000 token。

```bash
set -euo pipefail

export VLLM_USE_MODELSCOPE=1
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_HCU_USE_CUSTOM_OPS=1
export VLLM_USE_V2_MODEL_RUNNER=1
export NCCL_IB_DISABLE=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export GPU_MAX_HW_QUEUES=4
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_FLASH_ATTN_UNIFIED=1
export VLLM_USE_PD_SPLIT=1
export VLLM_ROCM_USE_AITER_MOE=1
export VLLM_USE_FLASH_ATTN_FP8=1
export VLLM_HCU_USE_LIGHTOP_MOE_ALIGN=1
export FA_DECODE_NUM_SPLITS=2
export VLLM_HCU_EP_FUSED_MOE=1
export VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=0
export VLLM_HCU_EP_FUSED_COMBINE=1
export VLLM_HCU_DP_CUSTOM_AG=1
export VLLM_HCU_STEP5_FP8_DENSE=1
export VLLM_HCU_STEP5_FP8_BF16_MOE=1
export VLLM_HCU_STEP5_SWA_FUSED_QK=1
export VLLM_HCU_STEP5_FUSED_GATE=1
export VLLM_HCU_TRITON_TOKEN_FP8_QUANT=1
export VLLM_HCU_STEP5_QKV_INDEXER=1
export VLLM_HCU_PIN_MEMORY=1
export HIP_FORCE_DEV_KERNARG=1
export VLLM_STEP4_DSA_GRAPH_REGIONS=0
export EP_FUSED_MOE_TABLE=/opt/step5-cookbook/assets/step5_ep8_moe_table.json
test -r "$EP_FUSED_MOE_TABLE"
export VLLM_STEP5_EXPERT_PLACEMENT=""

vllm serve hygon/Step-5-Preview-Channel-FP8-w8a8 \
  --served-model-name step5 \
  --host 0.0.0.0 \
  --reasoning-parser step3p5 \
  --tokenizer hygon/Step-5-Preview-Channel-FP8-w8a8 \
  --port 8002 \
  --tensor-parallel-size 4 \
  --data-parallel-size 2 \
  --enable-expert-parallel \
  --max-num-batched-tokens 32768 \
  --max-model-len 140000 \
  --max-num-seqs 64 \
  --no-enable-prefix-caching \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 3, "use_local_argmax_reduction": true}' \
  --max-cudagraph-capture-size 256 \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["+swiglustep_and_mul"],"compile_sizes":[16,64]}' \
  --kv-cache-dtype fp8_e4m3 \
  --limit-mm-per-prompt '{"image": 0}' \
  --gpu-memory-utilization 0.85
```


### Step-5-Preview-Channel-FP8-w8a8 Prefill 超节点 4×4 卡 vLLM 0.28.1

四节点、每节点 4 张 gfx938，配置 TP1 / DP16 / EP16，采用 DeepEP HT MNVL、DeepGEMM/WASP 和固定 E23 专家映射。调度 token 预算为 8192，上下文上限为 70000 token。

四节点分别设置 `NODE_RANK=0/1/2/3`。node0 提供 API `8018`，其余节点以 headless 模式启动；DP start rank 分别为 `0/4/8/12`。该配置使用匹配的 MNVL 扩展、Dense 原生算子及算法配置。

```bash
set -euo pipefail

NODE_RANK=0                         # 四节点依次改为 0、1、2、3
NODE_IPS=(192.168.1.10 192.168.1.11 192.168.1.12 192.168.1.13)
HEAD_IP="${NODE_IPS[0]}"
NODE_IP="${NODE_IPS[$NODE_RANK]}"
IFACE=eth0                          # 改为承载节点间通信的网卡
MODEL_DIR=/data/model/Step5-Channel-FP8-w8a8
ASSETS_DIR=/opt/step5-cookbook/assets
TOKENIZER_DIR="$ASSETS_DIR/tokenizer-bytelevel"
STATE_DIR=/var/lib/step5-cookbook/super-prefill

export HIP_VISIBLE_DEVICES=0,1,2,3
export VLLM_HOST_IP="$NODE_IP"
export NCCL_SOCKET_IFNAME="$IFACE" GLOO_SOCKET_IFNAME="$IFACE"
export HSA_FORCE_FINE_GRAIN_PCIE=1
export OMP_NUM_THREADS=1
export GPU_MAX_HW_QUEUES=4
export VLLM_HCU_USE_CUSTOM_OPS=1
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM=1
export VLLM_HCU_USE_FUSE_MOE_GATE=1
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_FLASH_ATTN_UNIFIED=1
export VLLM_USE_PD_SPLIT=1
export VLLM_HCU_USE_LIGHTOP_EP_SCATTER=1
export VLLM_HCU_USE_CUSTOM_SILU_AND_MUL=1
export VLLM_HCU_USE_CUSTOM_GEMMA_RMS_NORM=1
export VLLM_ROCM_USE_AITER_MOE=1
export VLLM_USE_FLASH_ATTN_FP8=1
export VLLM_HCU_USE_LIGHTOP_MOE_ALIGN=1
export FA_DECODE_NUM_SPLITS=2
export VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=0
export VLLM_HCU_DP_CUSTOM_AG=1
export VLLM_HCU_STEP5_FP8_DENSE=1
export VLLM_HCU_STEP5_FP8_BF16_MOE=1
export VLLM_HCU_STEP5_SWA_FUSED_QK=1
export VLLM_HCU_STEP5_FUSED_GATE=1
export VLLM_HCU_TRITON_TOKEN_FP8_QUANT=1
export VLLM_HCU_STEP5_QKV_INDEXER=1
export VLLM_HCU_PIN_MEMORY=1
export HIP_FORCE_DEV_KERNARG=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_STEP4_DSA_FLASH_ATTN=prefill,decode
export VLLM_STEP4_DSA_GRAPH_REGIONS=0
export HCU_STEP5_CONTEXT_GRAPH_REGIONS=0

export VLLM_USE_DEEP_GEMM=1
export VLLM_MOE_USE_DEEP_GEMM=1
export DEEP_EP_NORMAL_MNVL=1
export VLLM_DEEPEP_BUFFER_SIZE_MB=1536
export NCCL_MAX_NCHANNELS=32
export HCU_DBO_MIN_TOKENS=4096
export HCU_MTP_PREFILL_TAIL=1536
export HCU_SHARED_EXPERTS_STREAM_MIN_TOKENS=4096
export HCU_DP_MOE_PIPE_MIN_TOKENS=0
export HCU_EP_DIRECT_SUM=0
export VLLM_HCU_EP_FUSED_MOE=0
export VLLM_HCU_EP_FUSED_COMBINE=1
export HCU_STEP5_DEEPEP_FUSED_SCATTER=1
export HCU_STEP5_DEEPEP_COMPACT_ROWS=1
export HCU_STEP5_DEEPEP_PROFILE_DBO_OUTPUT=1
export HCU_STEP5_DEEPEP_DBO_BUFFER_PAIR=1
# 要求已安装包含 Step5 Dense 原生算子的 wheel
export HCU_STEP5_DENSE_TUNING=1
export HCU_STEP5_PREFILL_ASM_G1=0
export HCU_STEP5_PREFILL_G2=0
export HCU_STEP5_EXPERT_GRID=0
export HCU_STEP5_FUSED_ROUTER_STREAM=0
export HCU_STEP5_BOUNDED_EP_WORKSPACE=0
export EP_FUSED_MOE_CHUNK=32768
export VLLM_HCU_SHARED_EXPERTS_EARLY_LAUNCH=0

export HCU_STEP5_TUNING_CONFIG="$ASSETS_DIR/supernode-prefill/tuning/step5-gfx938-image1021.json"
export VLLM_STEP5_EXPERT_PLACEMENT="$ASSETS_DIR/supernode-prefill/expert_placement_prefill_8192_ep16_20261020.pt"
export STEP5_E23_MAP="$ASSETS_DIR/supernode-prefill/expert_placement_prefill_e23_static_tokenhash_20261023.json"

mkdir -p "$STATE_DIR"
export HCU_STEP5_DEEPEP_RUNTIME_CONFIG="$STATE_DIR/communication.json"
cat > "$HCU_STEP5_DEEPEP_RUNTIME_CONFIG" <<'JSON'
{"version":1,"num_sms":16,"dispatch_send":8,"combine_send":8,"recv_tokens":512,"async_layout":true,"overlap":true}
JSON

ROLE_ARGS=(--api-server-count 1)
if [[ "$NODE_RANK" != 0 ]]; then
  ROLE_ARGS=(--headless --data-parallel-start-rank "$((NODE_RANK * 4))")
fi

vllm serve "$MODEL_DIR" \
  --tokenizer "$TOKENIZER_DIR" \
  --served-model-name step5 \
  --reasoning-parser step3p5 \
  --host 0.0.0.0 --port 8018 \
  --tensor-parallel-size 1 \
  --data-parallel-size 16 \
  --data-parallel-size-local 4 \
  --data-parallel-address "$HEAD_IP" \
  --data-parallel-rpc-port 18192 \
  --distributed-executor-backend mp \
  --enable-expert-parallel \
  --all2all-backend deepep_high_throughput \
  --moe-backend auto \
  --max-num-batched-tokens 8192 \
  --max-model-len 70000 \
  --max-num-seqs 64 \
  --long-prefill-token-threshold 0 \
  --no-enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"use_local_argmax_reduction":true}' \
  --max-cudagraph-capture-size 64 \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["+swiglustep_and_mul"],"compile_sizes":[16,64]}' \
  --kv-cache-dtype fp8_e4m3 \
  --limit-mm-per-prompt '{"image":0}' \
  --gpu-memory-utilization 0.875 \
  --enable-eplb \
  --eplb-config '{"num_redundant_experts":16,"use_async":false}' \
  --additional-config "{\"hcu\":{\"expert_map_path\":\"$STEP5_E23_MAP\",\"eplb_disable_rearrange\":true,\"eplb_static_dispatch_policy\":\"token_hash\"}}" \
  "${ROLE_ARGS[@]}"
```


### Step-5-Preview-Channel-FP8-w8a8 Decode 超节点 4×4 卡 vLLM 0.28.1

四节点、每节点 4 张 gfx938，配置 TP4 / DP4 / EP16，采用 E22 专家放置、EP fused MoE 和 DP Fabric 通信。调度 token 预算为 32768，上下文上限为 1048576 token。

四节点分别设置 `NODE_RANK=0/1/2/3`。node0 提供 API `18118`，其余节点以 headless 模式启动；DP start rank 分别为 `0/1/2/3`。Fabric 配合 packed AllGather 使用，`VLLM_HCU_DP_CUSTOM_AG` 设置为 `0`。

```bash
set -euo pipefail

NODE_RANK=0                         # 四节点依次改为 0、1、2、3
NODE_IPS=(192.168.1.10 192.168.1.11 192.168.1.12 192.168.1.13)
HEAD_IP="${NODE_IPS[0]}"
NODE_IP="${NODE_IPS[$NODE_RANK]}"
IFACE=eth0
MODEL_DIR=/data/model/Step5-Channel-FP8-w8a8
ASSETS_DIR=/opt/step5-cookbook/assets
TOKENIZER_DIR="$ASSETS_DIR/tokenizer-bytelevel"

export HIP_VISIBLE_DEVICES=0,1,2,3
export VLLM_HOST_IP="$NODE_IP"
export NCCL_SOCKET_IFNAME="$IFACE" GLOO_SOCKET_IFNAME="$IFACE"
export HSA_FORCE_FINE_GRAIN_PCIE=1
export OMP_NUM_THREADS=1
export GPU_MAX_HW_QUEUES=4
export VLLM_HCU_USE_CUSTOM_OPS=1
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM=1
export VLLM_HCU_USE_FUSE_MOE_GATE=1
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_FLASH_ATTN_UNIFIED=1
export VLLM_USE_PD_SPLIT=1
export VLLM_HCU_USE_LIGHTOP_EP_SCATTER=1
export VLLM_HCU_USE_CUSTOM_SILU_AND_MUL=1
export VLLM_HCU_USE_CUSTOM_GEMMA_RMS_NORM=1
export VLLM_ROCM_USE_AITER_MOE=1
export VLLM_USE_FLASH_ATTN_FP8=1
export VLLM_HCU_USE_LIGHTOP_MOE_ALIGN=1
export FA_DECODE_NUM_SPLITS=2
export VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=0
export VLLM_HCU_STEP5_FP8_DENSE=1
export VLLM_HCU_STEP5_FP8_BF16_MOE=1
export VLLM_HCU_STEP5_SWA_FUSED_QK=1
export VLLM_HCU_STEP5_FUSED_GATE=1
export VLLM_HCU_TRITON_TOKEN_FP8_QUANT=1
export VLLM_HCU_STEP5_QKV_INDEXER=1
export VLLM_HCU_PIN_MEMORY=1
export HIP_FORCE_DEV_KERNARG=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_STEP4_DSA_FLASH_ATTN=prefill,decode

export VLLM_HCU_EP_FUSED_MOE=1
export VLLM_HCU_EP_FUSED_COMBINE=1
export HCU_STEP5_EXPERT_GRID=1
export HCU_STEP5_BOUNDED_EP_WORKSPACE=1
export EP_FUSED_MOE_CHUNK=32768
export VLLM_HCU_DP_CUSTOM_AG=0
export VLLM_HCU_DP_PACKED_AG=1
export HCU_STEP5_DP_FABRIC=1
export HCU_STEP5_ROUTER_SHARED_MAX_M=64
export VLLM_HCU_HOST_SEQ_LENS=1
export VLLM_STEP4_DSA_GRAPH_REGIONS=8736
export HCU_STEP5_CONTEXT_GRAPH_REGIONS=8736
export HCU_MTP_PREFILL_TAIL=1536
export HCU_SHARED_EXPERTS_STREAM_MIN_TOKENS=64

export HCU_DBO_MIN_TOKENS=0
export HCU_DP_MOE_PIPE_MIN_TOKENS=0
export HCU_DP_MOE_PIPE_UNIFORM=0
export HCU_EP_DIRECT_SUM=0
export HCU_STEP5_DENSE_TUNING=0
export HCU_STEP5_PREFILL_ASM_G1=0
export HCU_STEP5_PREFILL_G2=0
export HCU_STEP5_FUSED_ROUTER_STREAM=0
export VLLM_HCU_SHARED_EXPERTS_EARLY_LAUNCH=0

export EP_FUSED_MOE_TABLE="$ASSETS_DIR/supernode-decode/ep_table.json"
export VLLM_STEP5_EXPERT_PLACEMENT="$ASSETS_DIR/supernode-decode/expert_placement_decode.pt"

ROLE_ARGS=(--api-server-count 1)
if [[ "$NODE_RANK" != 0 ]]; then
  ROLE_ARGS=(--headless --data-parallel-start-rank "$NODE_RANK")
fi

vllm serve "$MODEL_DIR" \
  --tokenizer "$TOKENIZER_DIR" \
  --served-model-name step5 \
  --reasoning-parser step3p5 \
  --host 0.0.0.0 --port 18118 \
  --tensor-parallel-size 4 \
  --data-parallel-size 4 \
  --data-parallel-size-local 1 \
  --data-parallel-address "$HEAD_IP" \
  --data-parallel-rpc-port 18192 \
  --distributed-executor-backend mp \
  --enable-expert-parallel \
  --all2all-backend allgather_reducescatter \
  --moe-backend auto \
  --max-num-batched-tokens 32768 \
  --max-model-len 1048576 \
  --max-num-seqs 64 \
  --no-enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"use_local_argmax_reduction":true}' \
  --max-cudagraph-capture-size 64 \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["+swiglustep_and_mul"],"compile_sizes":[16,64]}' \
  --kv-cache-dtype fp8_e4m3 \
  --limit-mm-per-prompt '{"image":0}' \
  --gpu-memory-utilization 0.85 \
  "${ROLE_ARGS[@]}"
```

## API 调用

以下以单机 IFB 为例，调用其他配置时将 `API_BASE` 替换为对应服务地址。

```bash
API_BASE=http://127.0.0.1:8000/v1
curl --fail "$API_BASE/models"
curl --fail "$API_BASE/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "step5",
    "messages": [{"role": "user", "content": "中国的首都是什么？"}],
    "temperature": 0,
    "max_tokens": 2048
  }'
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="step5",
    messages=[{"role": "user", "content": "中国的首都是什么？"}],
    temperature=0,
    max_tokens=2048,
)
print(response.choices[0].message.content)
```
