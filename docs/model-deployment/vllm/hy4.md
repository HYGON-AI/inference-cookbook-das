# Hy4 on vLLM

## 模型简介

Hy4-preview 是 Hy 系列模型，本文档提供其在 vLLM 上的部署示例。

## 模型列表

| 模型权重 | 量化方式 | vLLM 镜像 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | --------- | -------- | ---- | -------- | -------- |
| [hygon/Hy4-preview-Channel-FP8-w8a8](https://www.modelscope.cn/models/hygon/Hy4-preview-Channel-FP8-w8a8) | FP8 W8A8 | 0.28.1 | BW1100 | 8 | IFB | [**`>_`**](#hy4-preview-channel-fp8-w8a8-ifb-bw1100-8x-vllm-0281) |
|  | FP8 W8A8 | 0.25.1 | BW1301_LC | 16 | IFB-P | [**`>_`**](#hy4-preview-channel-fp8-w8a8-ifb-p-bw1301_lc-16x-vllm-0251) |
|  | FP8 W8A8 | 0.25.1 | BW1301_LC | 32 | IFB-D | [**`>_`**](#hy4-preview-channel-fp8-w8a8-ifb-d-bw1301_lc-32x-vllm-0251) |
|  | FP8 W8A8 | 0.25.1 | BW1100 | 8 | IFB | [**`>_`**](#hy4-preview-channel-fp8-w8a8-ifb-bw1100-8x-vllm-0251) |


### 0.28.1 平台支持说明

BW1000 不支持本次发版所列 FP8 权重模型；以下 FP8 权重仅提供 BW1100 启动命令：`hygon/Hy4-preview-Channel-FP8-w8a8`。

## 启动命令

### Hy4-preview-Channel-FP8-w8a8 IFB BW1100 8x vLLM 0.28.1

```bash

vllm serve hygon/Hy4-preview-Channel-FP8-w8a8 \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --attention-backend FLASHMLA_SPARSE \
  --moe-backend aiter \
  --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --enable-auto-tool-choice \
  --tool-call-parser hy_v4 \
  --reasoning-parser hy_v4 \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
```

### Hy4-preview-Channel-FP8-w8a8 IFB-P BW1301_LC 16x vLLM 0.25.1

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export VLLM_HCU_USE_LIGHTOP_PER_TOKEN_QUANT_FP8=1
export VLLM_HCU_USE_LIGHTOP_HY_V4_INDEXER=0
export NCCL_IB_GID_INDEX=3
export ROCSHMEM_GDR_DISABLE_XDP=1
export DEEP_EP_NORMAL_MNVL=1
export VLLM_DEEPEP_BUFFER_SIZE_MB=4096
export GPU_MAX_HW_QUEUES=4
export VLLM_MOE_ROUTING_SIMULATION_STRATEGY=uniform_random
export VLLM_RANDOMIZE_DP_DUMMY_INPUTS=1
export LIGHTOP_MQA_LOGITS_BETA=1
export VLLM_USE_V2_MODEL_RUNNER=1

vllm serve hygon/Hy4-preview-Channel-FP8-w8a8 \
  --tensor-parallel-size 1 \
  --prefill-context-parallel-size 16 \
  --enable-expert-parallel \
  --all2all-backend deepep_high_throughput \
  --moe-backend deep_gemm \
  --default-chat-template-kwargs '{"reasoning_effort":"no_think"}' \
  --tool-call-parser hy_v4 \
  --reasoning-parser hy_v4 \
  --hf-overrides '{"head_dtype": "float32"}' \
  --max-model-len 133120 \
  --gpu-memory-utilization 0.90 \
  --max-num-batched-tokens 32768 \
  --enable-auto-tool-choice \
  --enable-prefix-caching \
  --enforce-eager
```

### Hy4-preview-Channel-FP8-w8a8 IFB-D BW1301_LC 32x vLLM 0.25.1

#### Node 0

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export ROCSHMEM_IPC_MNVL=1
export VLLM_DEEPEP_BUFFER_SIZE_MB=0
export VLLM_HCU_USE_LIGHTOP_PER_TOKEN_QUANT_FP8=1
export GPU_MAX_HW_QUEUES=3
export VLLM_HCU_USE_LIGHTOP_HY_V4_INDEXER=0
export VLLM_MOE_ROUTING_SIMULATION_STRATEGY=hcu_eplb_balancedness
export VLLM_HCU_MOE_ROUTING_SIMULATION_BALANCEDNESS=0.6
export VLLM_RANDOMIZE_DP_DUMMY_INPUTS=1
export VLLM_HCU_USE_LIGHTOP_FAST_TOPK_TRANSFORM=1
export VLLM_HCU_USE_AITER_OPUS_PAGED_MQA_LOGITS=1
export VLLM_USE_V2_MODEL_RUNNER=1

vllm serve hygon/Hy4-preview-Channel-FP8-w8a8 \
  --tensor-parallel-size 1 \
  --data-parallel-size 32 \
  --enable-expert-parallel \
  --all2all-backend deepep_low_latency \
  --moe-backend deep_gemm \
  --no-enable-prefix-caching \
  --data-parallel-size-local 16 \
  --data-parallel-address "<node0_ip>" \
  --data-parallel-rpc-port 1127 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3, "use_local_argmax_reduction":true}' \
  --gpu-memory-utilization 0.9 \
  --max-model-len 133120 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 128 \
  --default-chat-template-kwargs '{"reasoning_effort":"no_think"}' \
  --reasoning-parser hy_v4 \
  --enable-auto-tool-choice \
  --tool-call-parser hy_v4 \
  --kv-cache-dtype fp8_e4m3 \
  --data-parallel-hybrid-lb
```

#### Node 1

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export ROCSHMEM_IPC_MNVL=1
export VLLM_DEEPEP_BUFFER_SIZE_MB=0
export VLLM_HCU_USE_LIGHTOP_PER_TOKEN_QUANT_FP8=1
export GPU_MAX_HW_QUEUES=3
export VLLM_HCU_USE_LIGHTOP_HY_V4_INDEXER=0
export VLLM_MOE_ROUTING_SIMULATION_STRATEGY=hcu_eplb_balancedness
export VLLM_HCU_MOE_ROUTING_SIMULATION_BALANCEDNESS=0.6
export VLLM_RANDOMIZE_DP_DUMMY_INPUTS=1
export VLLM_HCU_USE_LIGHTOP_FAST_TOPK_TRANSFORM=1
export VLLM_HCU_USE_AITER_OPUS_PAGED_MQA_LOGITS=1
export VLLM_USE_V2_MODEL_RUNNER=1

vllm serve hygon/Hy4-preview-Channel-FP8-w8a8 \
  --tensor-parallel-size 1 \
  --data-parallel-size 32 \
  --enable-expert-parallel \
  --all2all-backend deepep_low_latency \
  --moe-backend deep_gemm \
  --no-enable-prefix-caching \
  --data-parallel-size-local 16 \
  --data-parallel-address "<node0_ip>" \
  --data-parallel-rpc-port 1127 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3, "use_local_argmax_reduction":true}' \
  --gpu-memory-utilization 0.9 \
  --max-model-len 133120 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 128 \
  --default-chat-template-kwargs '{"reasoning_effort":"no_think"}' \
  --reasoning-parser hy_v4 \
  --enable-auto-tool-choice \
  --tool-call-parser hy_v4 \
  --kv-cache-dtype fp8_e4m3 \
  --data-parallel-hybrid-lb \
  --data-parallel-start-rank 16 \
  --headless
```

### Hy4-preview-Channel-FP8-w8a8 IFB BW1100 8x vLLM 0.25.1

#### TP 方式

```bash
export VLLM_USE_V2_MODEL_RUNNER=1

vllm serve hygon/Hy4-preview-Channel-FP8-w8a8 \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 1 \
  --no-enable-expert-parallel \
  --moe-backend aiter \
  --kv-cache-dtype fp8_e4m3 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 8192 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 8192 \
  --default-chat-template-kwargs '{"reasoning_effort":"no_think"}' \
  --reasoning-parser hy_v4 \
  --enable-auto-tool-choice \
  --tool-call-parser hy_v4 \
  --compilation-config '{"cudagraph_capture_sizes":[1,16]}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

#### PP 方式

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_PP_LAYER_PARTITION=9,8,12,8,12,8,12,9

vllm serve hygon/Hy4-preview-Channel-FP8-w8a8 \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 8 \
  --no-enable-expert-parallel \
  --moe-backend aiter \
  --kv-cache-dtype fp8_e4m3 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 8192 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 8192 \
  --default-chat-template-kwargs '{"reasoning_effort":"no_think"}' \
  --reasoning-parser hy_v4 \
  --enable-auto-tool-choice \
  --tool-call-parser hy_v4 \
  --compilation-config '{"cudagraph_capture_sizes":[1,16]}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

## API 调用

### IFB

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "hygon/Hy4-preview-Channel-FP8-w8a8", "messages": [{"role": "user", "content": "中国的首都是哪里？"}], "max_tokens": 128}'
```
