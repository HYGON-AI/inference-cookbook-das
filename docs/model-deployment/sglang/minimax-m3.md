# MiniMax-M3 on SGLang

## 模型简介

MiniMax-M3 是采用 MoE 和 MiniMax Sparse Attention（MSA）的长上下文模型。以下命令适用于单机 `8 × BW1100`。

## 模型列表

| 模型权重 | 量化方式 | SGLang 镜像 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | ----------- | -------- | ---- | -------- | -------- |
| [MiniMax-M3-BF16](https://www.modelscope.cn/models/MiniMax/MiniMax-M3) | BF16 | 0.5.18 | BW1100 | 8 | IFB | [**`>_`**](#minimax-m3-bf16-ifb-bw1100-8x-sglang-0518) |
| [MiniMax-M3-Channel-FP8-w8a8](https://www.modelscope.cn/models/hygon/MiniMax-M3-Channel-FP8-w8a8) | FP8 W8A8 + FP8 KV | 0.5.18 | BW1100 | 8 | IFB | [**`>_`**](#minimax-m3-channel-fp8-w8a8-ifb-bw1100-8x-sglang-0518) |
| [MiniMax-M3-Channel-FP8-w8a8](https://www.modelscope.cn/models/hygon/MiniMax-M3-Channel-FP8-w8a8) | FP8 W8A8 + BF16 KV | 0.5.18 | BW1100 | 8 | Prefill | [**`>_`**](#minimax-m3-channel-fp8-w8a8-prefill-bw1100-8x-sglang-0518) |
| [MiniMax-M3-Channel-FP8-w8a8](https://www.modelscope.cn/models/hygon/MiniMax-M3-Channel-FP8-w8a8) | FP8 W8A8 + FP8 KV | 0.5.18 | BW1100 | 8 | Decode | [**`>_`**](#minimax-m3-channel-fp8-w8a8-decode-bw1100-8x-sglang-0518) |

## DeepEP 配置

Prefill 使用以下 `minimax_m3_prefill_ep_config.json`。保存该文件后，将启动命令中的 `--deepep-config` 改为实际路径。IFB 和 Decode 使用 `auto` 模式的默认配置，不需要指定该文件。

```json
{
  "normal_dispatch": {
    "num_sms": 48,
    "num_max_nvl_chunked_send_tokens": 16,
    "num_max_nvl_chunked_recv_tokens": 256,
    "num_max_rdma_chunked_send_tokens": 6,
    "num_max_rdma_chunked_recv_tokens": 128
  },
  "normal_combine": {
    "num_sms": 48,
    "num_max_nvl_chunked_send_tokens": 2,
    "num_max_nvl_chunked_recv_tokens": 256,
    "num_max_rdma_chunked_send_tokens": 6,
    "num_max_rdma_chunked_recv_tokens": 128
  }
}
```

## DeepGEMM 配置

[`minimax-m3-tp8-cp8-ep8-r8-deepgemm-multibs.json`](./configs/minimax-m3/minimax-m3-tp8-cp8-ep8-r8-deepgemm-multibs.json) 是 Prefill winner 使用的 DeepGEMM 配置。部署时请下载该文件，并将启动命令中的路径替换为文件的实际绝对路径。

## EPLB 配置

Expert placement 按照 [EPLB](../../optimization/static-eplb-sglang.md) 中的 recorder 流程生成。Prefill 需要分别采集实际 BS2、BS4、BS8 流量，将三组负载归一化后等权合并，并生成 E128+R8 placement；IFB 和 Decode 使用代表性 Decode 流量生成 E128+R0 placement。生成后，将启动命令中的 `/xxx/*.pt` 替换为实际输出路径。

## 启动命令

### MiniMax-M3-BF16 IFB BW1100 8x SGLang 0.5.18

```bash
export SGLANG_USE_MODELSCOPE=1
export SGLANG_KV_LAYOUT_HCU_FA=0
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export GPU_MAX_HW_QUEUES=4

sglang serve \
  --model-path MiniMax/MiniMax-M3 \
  --trust-remote-code \
  --dtype bfloat16 \
  --kv-cache-dtype bfloat16 \
  --tp-size 8 \
  --dp-size 1 \
  --ep-size 1 \
  --moe-dp-size 1 \
  --moe-a2a-backend none \
  --attention-backend triton \
  --prefill-attention-backend triton \
  --decode-attention-backend triton \
  --moe-runner-backend triton \
  --page-size 1 \
  --disable-cuda-graph \
  --disable-custom-all-reduce \
  --disable-shared-experts-fusion \
  --context-length 32768 \
  --chunked-prefill-size 4096 \
  --max-prefill-tokens 8192 \
  --max-running-requests 8 \
  --mem-fraction-static 0.92 \
  --reasoning-parser minimax-m3 \
  --tool-call-parser minimax-m3 \
  --host 0.0.0.0
```

### MiniMax-M3-Channel-FP8-w8a8 IFB BW1100 8x SGLang 0.5.18

```bash
export SGLANG_USE_MODELSCOPE=1
export SGLANG_KV_LAYOUT_HCU_FA=0

export SGLANG_REQUIRE_KV_CACHE_SCALES=1
export SGLANG_ENABLE_M3_TRITON_FP8_ATTN_GEMM=1
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_LIGHTOP_CHANNEL_FP8=1

sglang serve \
  --model-path hygon/MiniMax-M3-Channel-FP8-w8a8 \
  --trust-remote-code \
  --dtype bfloat16 \
  --quantization compressed-tensors \
  --tp-size 8 \
  --pp-size 1 \
  --dp-size 8 \
  --load-balance-method round_robin \
  --enable-dp-attention \
  --enable-dp-attention-local-control-broadcast \
  --enable-dp-lm-head \
  --ep-size 8 \
  --moe-dp-size 1 \
  --moe-a2a-backend deepep \
  --deepep-mode auto \
  --deepep-dispatcher-output-dtype fp8 \
  --moe-runner-backend deep_gemm \
  --ep-num-redundant-experts 0 \
  --init-expert-location /xxx/minimax-m3-e128-r0-global.pt \
  --ep-dispatch-algorithm static \
  --disable-shared-experts-fusion \
  --attention-backend triton \
  --prefill-attention-backend triton \
  --decode-attention-backend triton \
  --attention-context-parallel-size 1 \
  --dcp-size 1 \
  --page-size 64 \
  --kv-cache-dtype fp8_e4m3 \
  --cuda-graph-backend-decode full \
  --cuda-graph-backend-prefill disabled \
  --cuda-graph-bs-decode 1 2 3 4 8 16 \
  --context-length 163840 \
  --chunked-prefill-size 16384 \
  --max-prefill-tokens 16384 \
  --max-running-requests 128 \
  --scheduler-recv-interval 1 \
  --mem-fraction-static 0.80 \
  --reasoning-parser minimax-m3 \
  --tool-call-parser minimax-m3 \
  --disable-custom-all-reduce \
  --host 0.0.0.0
```

### MiniMax-M3-Channel-FP8-w8a8 Prefill BW1100 8x SGLang 0.5.18

```bash
export SGLANG_USE_MODELSCOPE=1
export SGLANG_HCU_DEEPGEMM_CONTIG_TUNING_CONFIG=/xxx/minimax-m3-tp8-cp8-ep8-r8-deepgemm-multibs.json

export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_LIGHTOP_CHANNEL_FP8=1
export SGLANG_USE_LIGHTOP_GEMMA_RMSNORM=1
export SGLANG_USE_FUSED_TOPK_SOFTMAX=1
export SGLANG_ENABLE_CP_V2=1
export SGLANG_PREFILL_CP_MIN_TOKENS_PER_SEQUENCE=4096
export SGLANG_OPT_USE_MINIMAX_QUERY_SHARDED_CP=1
export SGLANG_OPT_USE_MINIMAX_FLASH_MLA_GFX938=1
export SGLANG_OPT_USE_MINIMAX_FLASH_MLA_GFX938_INDEXER=1
export SGLANG_MINIMAX_CACHE_PREFILL_INDEX_METADATA=1
export SGLANG_MINIMAX_PARALLEL_CP_SEGMENTS=1
export SGLANG_DEEPEP_ASYNC_FINISH=0
export SGLANG_DEEPEP_USE_MOE_EP_GROUP=1
export SGLANG_OPT_DG_HCU_FUSED_QUANT_SCATTER=1
export SGLANG_OPT_USE_MINIMAX_DEEPEP_SHARED_EXPERT_OVERLAP=1
export SGLANG_OPT_USE_MINIMAX_ROUTER_GEMV=1
export SGLANG_MINIMAX_PREFILL_EXACT_GROUPED_MAIN_Q=16
export SGLANG_MINIMAX_PREFILL_EXACT_GROUPED_MAIN_MAX=128
export SGLANG_MINIMAX_PREFILL_TOPK_BACKEND=aiter
export SGLANG_MINIMAX_PREFILL_SCORE_CONFIG=128,128,8,3
export SGLANG_KV_LAYOUT_HCU_FA=false

sglang serve \
  --model-path hygon/MiniMax-M3-Channel-FP8-w8a8 \
  --trust-remote-code \
  --dtype bfloat16 \
  --quantization compressed-tensors \
  --tp-size 8 \
  --pp-size 1 \
  --dp-size 1 \
  --ep-size 8 \
  --moe-dp-size 1 \
  --moe-a2a-backend deepep \
  --deepep-mode normal \
  --deepep-config /path/to/minimax_m3_prefill_ep_config.json \
  --deepep-dispatcher-output-dtype fp8 \
  --moe-runner-backend deep_gemm \
  --ep-num-redundant-experts 8 \
  --init-expert-location /xxx/minimax-m3-cp8-ep8-multibs-r8-placement.pt \
  --ep-dispatch-algorithm static \
  --disable-shared-experts-fusion \
  --attention-backend triton \
  --prefill-attention-backend fa3 \
  --decode-attention-backend triton \
  --attention-context-parallel-size 8 \
  --enable-prefill-cp \
  --cp-strategy zigzag \
  --dcp-size 1 \
  --page-size 128 \
  --kv-cache-dtype bf16 \
  --enable-prefill-delayer \
  --prefill-delayer-max-delay-passes 1000 \
  --enable-prefill-idle-coalescing \
  --prefill-idle-coalesce-max-delay-ms 50 \
  --prefill-idle-coalesce-settle-ms 50 \
  --prefill-idle-coalesce-burst-max-delay-ms 500 \
  --prefill-idle-coalesce-max-batch-size 32 \
  --cuda-graph-backend-decode disabled \
  --cuda-graph-backend-prefill disabled \
  --context-length 163840 \
  --chunked-prefill-size 131072 \
  --max-prefill-tokens 524288 \
  --max-running-requests 32 \
  --scheduler-recv-interval 1 \
  --mem-fraction-static 0.80 \
  --reasoning-parser minimax-m3 \
  --tool-call-parser minimax-m3 \
  --disable-custom-all-reduce \
  --host 0.0.0.0
```

### MiniMax-M3-Channel-FP8-w8a8 Decode BW1100 8x SGLang 0.5.18

```bash
export SGLANG_USE_MODELSCOPE=1
export SGLANG_KV_LAYOUT_HCU_FA=0

export SGLANG_REQUIRE_KV_CACHE_SCALES=1
export SGLANG_ENABLE_M3_TRITON_FP8_ATTN_GEMM=1
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_LIGHTOP_CHANNEL_FP8=1
export SGLANG_USE_LIGHTOP_GEMMA_RMSNORM=1
export SGLANG_USE_FUSED_TOPK_SOFTMAX=1
export SGLANG_DEEPEP_USE_MOE_EP_GROUP=1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=64
export SGLANG_OPT_DG_HCU_FUSED_QUANT_SCATTER=1
export SGLANG_OPT_USE_MINIMAX_ROUTER_GEMV=1
export SGLANG_MINIMAX_ROUTER_GEMV_BLOCK_K=1024
export SGLANG_MINIMAX_ROUTER_GEMV_NUM_WARPS=1
export SGLANG_OPT_USE_EAGLE3_LM_HEAD_TOP1=1
export SGLANG_OPT_USE_MINIMAX_MULTI_Q_VERIFY_SCORE=1
export SGLANG_MINIMAX_MTP_NUM_TOPK_CHUNKS=4

sglang serve \
  --model-path hygon/MiniMax-M3-Channel-FP8-w8a8 \
  --trust-remote-code \
  --dtype bfloat16 \
  --quantization compressed-tensors \
  --tp-size 8 \
  --pp-size 1 \
  --dp-size 8 \
  --load-balance-method round_robin \
  --enable-dp-attention \
  --enable-dp-attention-local-control-broadcast \
  --enable-dp-lm-head \
  --ep-size 8 \
  --moe-dp-size 1 \
  --moe-a2a-backend deepep \
  --deepep-mode auto \
  --deepep-dispatcher-output-dtype fp8 \
  --moe-runner-backend deep_gemm \
  --ep-num-redundant-experts 0 \
  --init-expert-location /xxx/minimax-m3-e128-r0-global.pt \
  --ep-dispatch-algorithm static \
  --disable-shared-experts-fusion \
  --attention-backend triton \
  --prefill-attention-backend triton \
  --decode-attention-backend triton \
  --attention-context-parallel-size 1 \
  --dcp-size 1 \
  --page-size 64 \
  --kv-cache-dtype fp8_e4m3 \
  --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path /path/to/MiniMax-M3-EAGLE3 \
  --speculative-draft-model-quantization unquant \
  --speculative-draft-kv-cache-dtype bf16 \
  --speculative-draft-attention-backend triton \
  --speculative-attention-mode prefill \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --speculative-draft-window-size 4096 \
  --cuda-graph-backend-decode full \
  --cuda-graph-backend-prefill disabled \
  --cuda-graph-bs-decode 1 2 3 4 8 16 \
  --context-length 163840 \
  --chunked-prefill-size 131072 \
  --max-prefill-tokens 16384 \
  --max-running-requests 128 \
  --scheduler-recv-interval 1 \
  --mem-fraction-static 0.80 \
  --reasoning-parser minimax-m3 \
  --tool-call-parser minimax-m3 \
  --disable-custom-all-reduce \
  --host 0.0.0.0
```

## API 调用

### IFB

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="hygon/MiniMax-M3-Channel-FP8-w8a8",  # 替换为实际使用的模型名
    messages=[
        {"role": "system", "content": "你是一个有帮助的 AI 助手。"},
        {"role": "user", "content": "请分析一下当前中国 AI 芯片产业的发展现状。"},
    ],
    temperature=0,
    max_tokens=2048,
)

print(response.choices[0].message.content)
```

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "hygon/MiniMax-M3-Channel-FP8-w8a8",
    "messages": [
      {"role": "system", "content": "你是一个有帮助的 AI 助手。"},
      {"role": "user", "content": "请简单介绍一下你自己。"}
    ],
    "temperature": 0,
    "max_tokens": 128
  }'
```
