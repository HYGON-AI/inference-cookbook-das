# MiniMax-M3.1 on SGLang

## 模型简介

[MiniMax-M3.1-Flash](https://www.modelscope.cn/models/MiniMax/MiniMax-M3.1-Flash) 使用 MoE、MiniMax Sparse Attention（MSA）和 DSpark。本文提供 BW1100 单机 8 卡的 Channel FP8 W8A8 与 Channel INT4 W4A8 部署配置。W4A8 使用 INT4 专家权重和 INT8 激活，保留 INT8 投影和 FP8 KV Cache。

## 模型列表

| 模型权重 | 量化方式 | SGLang 镜像 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | ----------- | -------- | ---- | -------- | -------- |
| [hygon/MiniMax-M3.1-Flash-Channel-W4A8-INT4-W8A8-INT8](https://www.modelscope.cn/models/hygon/MiniMax-M3.1-Flash-Channel-W4A8-INT4-W8A8-INT8) | INT4 W4A8 | 0.5.22 | BW1100 | 8 | IFB | [**`>_`**](#minimax-m31-flash-channel-w4a8-int4-w8a8-int8-ifb-bw1100-8x-sglang-0522) Prefill |
|  | INT4 W4A8 | 0.5.22 | BW1100 | 8 | IFB | [**`>_`**](#minimax-m31-flash-channel-w4a8-int4-w8a8-int8-ifb-bw1100-8x-sglang-0522-1) Decode |
| [hygon/MiniMax-M3.1-Flash-Channel-FP8-w8a8](https://www.modelscope.cn/models/hygon/MiniMax-M3.1-Flash-Channel-FP8-w8a8) | FP8 W8A8 | [0.5.18](../docker_images.md) | BW1100 | 8 | IFB | [**`>_`**](#minimax-m31-flash-channel-fp8-w8a8-ifb-bw1100-8x-sglang-0518) Prefill |
|  | FP8 W8A8 | [0.5.18](../docker_images.md) | BW1100 | 8 | IFB | [**`>_`**](#minimax-m31-flash-channel-fp8-w8a8-ifb-bw1100-8x-sglang-0518-1) Decode |

## 部署准备

- 在相应版本的完整镜像中运行。W4A8 使用带 MiniMax 插件及对应优化的 0.5.22 镜像。
- 使用本地权重时，将 `--model-path` 换成容器内的模型路径。W4A8 Decode 将 `dspark-channel-int8-smooth-20261107` 挂载到 `/models/MiniMax-M3.1-Flash-Channel-W4A8-INT4-W8A8-INT8/dspark-channel-int8-smooth-20261107`。FP8 Decode 将 `dspark` 挂载到 `/models/MiniMax-M3.1-Flash-Channel-FP8-w8a8/dspark`。
- W4A8 Prefill 和 Decode 的静态专家布局分别为 `./configs/m31_w4a8_cp8_ep8_rsc_shared_bs1_64_waves_replica8_20261109.pt` 和 `./configs/minimax-m31-bs128-tiles-tail.pt`。启动前按命令放置对应配置。命令参数采用部署目录相对路径。
- 命令以八个 RDMA 设备为例。按节点实际选用网卡。W4A8 Decode 的 NUMA 列表是验证节点的映射，迁移时按八张卡的 NUMA 归属调整。BS128 配方使用 BW1100 144 GB，需为 HiSparse 准备足够主机内存并允许锁页。
- 各命令独立运行，使用默认 API 端口 30000。W4A8 Decode 采用单机 Fake Prefill，支持固定接受长度下的性能复现。真实请求请使用完整 Prefill 服务。

## 启动命令

### MiniMax-M3.1-Flash-Channel-W4A8-INT4-W8A8-INT8 IFB BW1100 8x SGLang 0.5.22

**Prefill（CP8 / EP8）**。启用前缀缓存、静态专家布局和 DeepEP normal。

```bash
export HCU_ADAPTER_HOST=sglang
export SGLANG_USE_MODELSCOPE=1
export SGLANG_PLUGINS=hcu_minimax_m31
export HCU_PREFILL_CP=1
export SGLANG_MINIMAX_PREFILL_SHARED_OVERLAP_METHOD=split
export SGLANG_USE_LIGHTOP_MINIMAX_QK_ROPE=0
export HCU_PLUGIN_MINIMAX_M31_FLASHMLA=1
export HCU_PLUGIN_MINIMAX_M31_DEEPEP=1
export HCU_DEEPEP_BRIDGE=0
export HCU_AITER_AR=1
export PYTHONNOUSERSITE=1
export HIP_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export GPU_ARCHS=gfx938
export MSA128_FP8_PREFILL_KERNEL=v2
export SGLANG_MINIMAX_INDEX_FP8=dot
export SGLANG_MINIMAX_CACHE_PREFILL_INDEX_METADATA=1
export FLASH_MLA_STAGE1_PREFILL_ASYNC_H2D=1
export SGLANG_KV_LAYOUT_HCU_FA=0
export HCU_MINIMAX_CP_PACK_FP8=1
export LIGHTOP_PER_TOKEN_INT8_METHOD=native
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_LIGHTOP_OAI_INT8_QUANT=1
export SGLANG_USE_LIGHTOP_CHANNEL_FP8=1
export SGLANG_USE_LIGHTOP_GEMMA_RMSNORM=1
export SGLANG_USE_FP8_W8A8_MOE=0
export SGLANG_USE_INT4_W4A8=1
export SGLANG_W4A8_TPMOE_BACKEND=lightop
export SGLANG_USE_LIGHTOP_W4A8_MARLIN_MOE=0
export SGLANG_USE_W4A8_CONTIGUOUS_HIPC=1
export SGLANG_USE_W4A8_MASKED_HIPC=1
export SGLANG_GROUPGEMM=1
export W8A8_SUPPORT_METHODS=3
export SGLANG_USE_DEEPGEMM_MOE=0
export SGLANG_OPT_USE_MINIMAX_ROUTER_GEMV=1
export SGLANG_ROCM_USE_AITER_MOE=0
export SGLANG_USE_AITER_AG=0
export SGLANG_MINIMAX_PREFILL_SHARED_OAI_INT8=1
export ROCSHMEM_ALLOWED_IBV_DEVICES=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9

sglang serve \
  --model-path hygon/MiniMax-M3.1-Flash-Channel-W4A8-INT4-W8A8-INT8 \
  --quantization slimquant_w4a8_marlin \
  --trust-remote-code \
  --random-seed 42 \
  --tp-size 8 \
  --ep-size 8 \
  --moe-dense-tp-size 1 \
  --pp-size 1 \
  --attn-dp-size 1 \
  --attn-cp-size 8 \
  --enable-prefill-cp \
  --cp-strategy zigzag \
  --disable-overlap-schedule \
  --load-balance-method round_robin \
  --moe-a2a-backend deepep \
  --deepep-mode normal \
  --deepep-config '{"normal_dispatch":{"num_sms":64},"normal_combine":{"num_sms":64}}' \
  --deepep-dispatcher-output-dtype int8 \
  --moe-runner-backend deep_gemm \
  --disable-shared-experts-fusion \
  --init-expert-location ./configs/m31_w4a8_cp8_ep8_rsc_shared_bs1_64_waves_replica8_20261109.pt \
  --ep-num-redundant-experts 8 \
  --ep-dispatch-algorithm static \
  --ep-static-dispatch-policy locality_fair \
  --attention-backend fa3 \
  --prefill-attention-backend fa3 \
  --decode-attention-backend fa3 \
  --kv-cache-dtype fp8_e4m3 \
  --page-size 128 \
  --cuda-graph-backend-decode disabled \
  --cuda-graph-backend-prefill disabled \
  --context-length 1048576 \
  --chunked-prefill-size 65536 \
  --max-prefill-tokens 65536 \
  --max-running-requests 80 \
  --mem-fraction-static 0.86 \
  --reasoning-parser minimax-m3 \
  --tool-call-parser minimax-m3
```

### MiniMax-M3.1-Flash-Channel-W4A8-INT4-W8A8-INT8 IFB BW1100 8x SGLang 0.5.22

**Decode（DP8 / EP8，BS128）**。使用 Fake Prefill、HiSparse、DSpark 和 Verify8。完整形状为八个 DP rank 各 16 条请求。压测使用每条 147456 输入、1024 输出，模拟接受长度 3.6。

```bash
export PYTHONNOUSERSITE=1 HCU_ADAPTER_HOST=sglang SGLANG_PLUGINS=hcu_minimax_m31
export SGLANG_USE_MODELSCOPE=1
export HCU_PLUGIN_MINIMAX_M31_FLASHMLA=1 HCU_PLUGIN_MINIMAX_M31_DEEPEP=1 HCU_DEEPEP_BRIDGE=0
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 GPU_ARCHS=gfx938 GPU_MAX_HW_QUEUES=8
export HIP_GRAPH_ACCUMULATE_DISPATCH=0 ROCSHMEM_GDR_DISABLE_HDP=1
export FLASH_MLA_MSA128_FP8_SPLIT_POLICY=verify8
export M31_STAGE1_GRAPH_MAX_KV_LEN=148608
export SGLANG_MINIMAX_INDEX_FP8=dot SGLANG_M3_OPTIMIZE_SPARSE_ATTN_PREP=1
export SGLANG_DSPARK_FOLDED_PROPOSAL=1 SGLANG_DSPARK_USE_GREEDY_DRAFT=1 SGLANG_RAGGED_VERIFY_MODE=static
export SGLANG_SIMULATE_ACC_LEN=3.6 SGLANG_SIMULATE_ACC_METHOD=match-expected
export LIGHTOP_PER_TOKEN_INT8_METHOD=triton SGLANG_USE_LIGHTOP=1 W8A8_SUPPORT_METHODS=3
export SGLANG_USE_LIGHTOP_OAI_INT8_QUANT=1 SGLANG_USE_LIGHTOP_OAI_FP8_QUANT=1
export SGLANG_USE_LIGHTOP_CHANNEL_FP8=1 SGLANG_USE_LIGHTOP_GEMMA_RMSNORM=1
export SGLANG_USE_INT4_W4A8=1 SGLANG_USE_W4A8_MASKED_HIPC=1 SGLANG_GROUPGEMM=1
export SGLANG_USE_LIGHTOP_W4A8_MARLIN_MOE=0 SGLANG_W4A8_TPMOE_BACKEND=lightop
export SGLANG_USE_FP8_W8A8_MOE=0 SGLANG_USE_DEEPGEMM_MOE=0
export SGLANG_OPT_USE_MINIMAX_ROUTER_GEMV=1 SGLANG_MINIMAX_ROUTER_GEMV_BLOCK_K=1024
export SGLANG_MINIMAX_ROUTER_GEMV_NUM_WARPS=1
export DEEPEP_ENABLE_LL_LAYERED_OPT=1 SGLANG_DEEPEP_USE_MOE_EP_GROUP=1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export HCU_PLUGIN_MINIMAX_M31_HISPARSE=1 M31_HISPARSE_FAKE_VALIDATE=1
export SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB=4
export M31_HISPARSE_STABLE_SLOTS=1 M31_HISPARSE_HIP_STABLE_PLAN=1 M31_HISPARSE_FAST_PLAN=1
export M31_HISPARSE_AB_COMPACT_HOT=1 M31_HISPARSE_HEADWISE_SWAP=1
export M31_HISPARSE_DIRECT_VERIFY_INDICES=1 M31_HISPARSE_VERIFY_QUERY_CHUNK=8
export M31_HISPARSE_PIN_SLOT_INDICES=1 M31_HISPARSE_TILED_SWAP=256,4 M31_HISPARSE_BATCH_COMMIT=1
export HSA_USERPTR_FOR_PAGED_MEM=0 SGLANG_HISPARSE_HCU_PIN_HOST_KV=1
export M31_HISPARSE_FUSED_QK_CACHE=1 M31_HISPARSE_REUSE_VERIFY_METADATA=1
export M31_HISPARSE_SPLIT_ROUTER_M144=1
export M31_ALIGN_NATIVE_QK=1 M31_ALIGN_JOINT_INT8_QKV=1
export M31_ALIGN_RECT_INDEXER=1 M31_ALIGN_VERIFY_TOPK=1
export M31_ALIGN_COMPACT_PAGES=1 M31_ALIGN_DELTA_SWAP=1 M31_ALIGN_ATTN_OUT_INT8=1
export M31_ALIGN_MASKED_GEMM=1 M31_ALIGN_G1_PERSISTENT=5
export M31_ALIGN_DRAFT_SWA_WAVES1=1 M31_FOLLOW_DRAFT_SWA_ONLY=1 M31_FOLLOW_MARKOV=1
export SGLANG_MINIMAX_DECODE_SHARED_OVERLAP=1 M31_FOLLOW_SHARED_INT8=1
export ROCSHMEM_ALLOWED_IBV_DEVICES=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9

sglang serve \
  --model-path hygon/MiniMax-M3.1-Flash-Channel-W4A8-INT4-W8A8-INT8 \
  --quantization slimquant_w4a8_marlin \
  --trust-remote-code \
  --random-seed 42 \
  --tp-size 8 \
  --ep-size 8 \
  --moe-dense-tp-size 1 \
  --speculative-algorithm DSPARK \
  --speculative-draft-model-path /models/MiniMax-M3.1-Flash-Channel-W4A8-INT4-W8A8-INT8/dspark-channel-int8-smooth-20261107 \
  --speculative-draft-model-quantization compressed-tensors \
  --speculative-draft-attention-backend triton \
  --speculative-dspark-block-size 7 \
  --speculative-num-draft-tokens 8 \
  --attn-dp-size 8 \
  --enable-dp-lm-head \
  --enable-dp-attention-local-control-broadcast \
  --load-balance-method round_robin \
  --moe-a2a-backend deepep \
  --deepep-mode low_latency \
  --deepep-dispatcher-output-dtype int8 \
  --moe-runner-backend deep_gemm \
  --disable-shared-experts-fusion \
  --init-expert-location ./configs/minimax-m31-bs128-tiles-tail.pt \
  --ep-dispatch-algorithm static \
  --attention-backend triton \
  --prefill-attention-backend triton \
  --decode-attention-backend fa3 \
  --kv-cache-dtype fp8_e4m3 \
  --page-size 128 \
  --cuda-graph-backend-decode full \
  --cuda-graph-backend-prefill disabled \
  --cuda-graph-bs-decode 1 2 4 8 10 12 16 \
  --context-length 163840 \
  --chunked-prefill-size 8192 \
  --max-running-requests 128 \
  --mem-fraction-static 0.97 \
  --max-total-tokens 992256 \
  --reasoning-parser minimax-m3 \
  --tool-call-parser minimax-m3 \
  --skip-server-warmup \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend fake \
  --disable-radix-cache \
  --numa-node 3 1 2 0 7 5 6 4 \
  --enable-hisparse \
  --hisparse-config '{"device_buffer_size":30720,"host_to_device_ratio":2.3975}'
```

### MiniMax-M3.1-Flash-Channel-FP8-w8a8 IFB BW1100 8x SGLang 0.5.18

**Prefill（CP8 / EP8）**。启用前缀缓存，使用 DeepEP normal。

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONNOUSERSITE=1
export GPU_ARCHS=gfx938
export SGLANG_KV_LAYOUT_HCU_FA=0
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_LIGHTOP_OAI_FP8_QUANT=1
export SGLANG_USE_LIGHTOP_CHANNEL_FP8=1
export SGLANG_USE_LIGHTOP_GEMMA_RMSNORM=1
export SGLANG_ROCM_USE_AITER_MOE=0
export SGLANG_USE_AITER_AG=0
export SGLANG_OPT_USE_MINIMAX_QUERY_SHARDED_CP=1
export SGLANG_USE_DEEPGEMM_MOE=1
export SGLANG_MINIMAX_MOE_MODE=
export SGLANG_DEEPEP_USE_MOE_EP_GROUP=1
export SGLANG_OPT_DG_HCU_FUSED_QUANT_SCATTER=1
export SGLANG_HCU_DEEPEP_NO_M_INDICES_SYNC=1
export SGLANG_MINIMAX_PREFILL_SHARED_OVERLAP_METHOD=split
export SGLANG_OPT_USE_MINIMAX_ROUTER_GEMV=1
export SGLANG_OPT_USE_MINIMAX_FLASH_MLA_GFX938=1
export SGLANG_MINIMAX_INDEX_FP8=dot
export SGLANG_USE_LIGHTOP_MINIMAX_QK_ROPE=1
export SGLANG_MINIMAX_CP_FP8_KV_EXCHANGE=1
export SGLANG_MINIMAX_CACHE_PREFILL_INDEX_METADATA=1
export ROCSHMEM_ALLOWED_IBV_DEVICES=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
export SGLANG_USE_MODELSCOPE=1

sglang serve \
  --model-path hygon/MiniMax-M3.1-Flash-Channel-FP8-w8a8 \
  --trust-remote-code \
  --random-seed 42 \
  --tp-size 8 \
  --pp-size 1 \
  --attn-cp-size 8 \
  --enable-prefill-cp \
  --cp-strategy zigzag \
  --disable-overlap-schedule \
  --moe-dense-tp-size 1 \
  --dp-size 1 \
  --ep-size 8 \
  --moe-a2a-backend deepep \
  --moe-runner-backend deep_gemm \
  --deepep-mode normal \
  --deepep-config '{"normal_dispatch":{"num_sms":64},"normal_combine":{"num_sms":64}}' \
  --enable-dp-attention \
  --enable-dp-lm-head \
  --enable-dp-attention-local-control-broadcast \
  --load-balance-method round_robin \
  --deepep-dispatcher-output-dtype fp8 \
  --disable-shared-experts-fusion \
  --attention-backend fa3 \
  --prefill-attention-backend fa3 \
  --decode-attention-backend fa3 \
  --kv-cache-dtype fp8_e4m3 \
  --page-size 128 \
  --cuda-graph-backend-decode disabled \
  --cuda-graph-backend-prefill disabled \
  --custom-all-reduce-backend aiter \
  --context-length 1048576 \
  --chunked-prefill-size 65536 \
  --max-prefill-tokens 65536 \
  --max-running-requests 80 \
  --mem-fraction-static 0.85 \
  --reasoning-parser minimax-m3 \
  --tool-call-parser minimax-m3
```

### MiniMax-M3.1-Flash-Channel-FP8-w8a8 IFB BW1100 8x SGLang 0.5.18

**Decode（DP8 / EP8）**。启用 DSpark 与 Decode full graph。

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONNOUSERSITE=1
export GPU_ARCHS=gfx938
export DEEPEP_ENABLE_LL_LAYERED_OPT=1
export SGLANG_HCU_M3_NATIVE_DEQUANT=0
export SGLANG_USE_MODELSCOPE=1
export SGLANG_KV_LAYOUT_HCU_FA=0
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_LIGHTOP_OAI_FP8_QUANT=1
export SGLANG_USE_LIGHTOP_CHANNEL_FP8=1
export SGLANG_USE_LIGHTOP_GEMMA_RMSNORM=1
export SGLANG_USE_AITER=0
export SGLANG_ROCM_USE_AITER_MOE=0
export AITER_AR_TRANSPORT=ipc
export SGLANG_DP_USE_MAX_LEN=0
export SGLANG_USE_AITER_AG=0
export SGLANG_ROCM_USE_AITER_MOE_WITH_SHUFFLE=0
export SGLANG_USE_FP8_W8A8_MOE=1
export SGLANG_USE_DEEPGEMM_MOE=1
export SGLANG_DEEPEP_USE_MOE_EP_GROUP=1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export SGLANG_OPT_DG_HCU_FUSED_QUANT_SCATTER=1
export SGLANG_OPT_USE_MINIMAX_ROUTER_GEMV=1
export SGLANG_MINIMAX_ROUTER_GEMV_BLOCK_K=1024
export SGLANG_MINIMAX_ROUTER_GEMV_NUM_WARPS=1
export SGLANG_ENABLE_M3_TRITON_FP8_ATTN_GEMM=0
export SGLANG_OPT_USE_MINIMAX_FLASH_MLA_GFX938=1
export SGLANG_OPT_USE_MINIMAX_FLASH_MLA_GFX938_INDEXER=1
export SGLANG_OPT_USE_MINIMAX_FLASH_MLA_GFX938_DECODE=1
export SGLANG_DSPARK_FOLDED_PROPOSAL=0
export SGLANG_DSPARK_USE_GREEDY_DRAFT=0
export SGLANG_RAGGED_VERIFY_MODE=static
export SGLANG_SIMULATE_ACC_LEN=0
export SGLANG_LOG_DECODE_GRAPH_KEY=0
export SGLANG_MINIMAX_M31_DRAFT_SPLITKV=1
export SGLANG_MINIMAX_FLASH_MLA_PREFILL_LIVE_EXTENT=1
export MSA_STAGE1_KERNEL=v6_pair
export FLASH_MLA_MSA128_FP8_SPLIT_POLICY=verify8
export ROCSHMEM_ALLOWED_IBV_DEVICES=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9

sglang serve \
  --model-path hygon/MiniMax-M3.1-Flash-Channel-FP8-w8a8 \
  --trust-remote-code \
  --random-seed 42 \
  --speculative-algorithm DSPARK \
  --speculative-draft-model-path /models/MiniMax-M3.1-Flash-Channel-FP8-w8a8/dspark \
  --speculative-draft-attention-backend triton \
  --speculative-dspark-block-size 7 \
  --speculative-num-draft-tokens 8 \
  --tp-size 8 \
  --moe-dense-tp-size 1 \
  --dp-size 8 \
  --ep-size 8 \
  --enable-dp-attention \
  --enable-dp-lm-head \
  --enable-dp-attention-local-control-broadcast \
  --load-balance-method round_robin \
  --moe-a2a-backend deepep \
  --moe-runner-backend deep_gemm \
  --deepep-mode auto \
  --deepep-dispatcher-output-dtype fp8 \
  --disable-shared-experts-fusion \
  --attention-backend triton \
  --prefill-attention-backend triton \
  --decode-attention-backend fa3 \
  --kv-cache-dtype fp8_e4m3 \
  --page-size 128 \
  --cuda-graph-backend-decode full \
  --cuda-graph-backend-prefill disabled \
  --custom-all-reduce-backend aiter \
  --context-length 1048576 \
  --chunked-prefill-size 65536 \
  --max-running-requests 80 \
  --mem-fraction-static 0.85 \
  --reasoning-parser minimax-m3 \
  --tool-call-parser minimax-m3
```

## API 调用

### IFB

以下示例连接 FP8 单机服务。使用 W4A8 Prefill 服务时，`model` 填写 `hygon/MiniMax-M3.1-Flash-Channel-W4A8-INT4-W8A8-INT8`。使用本地模型路径启动时，API 的 `model` 同样填写该路径。

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="hygon/MiniMax-M3.1-Flash-Channel-FP8-w8a8",
    messages=[{"role": "user", "content": "请简单介绍一下你自己。"}],
    max_tokens=128,
)

print(response.choices[0].message.content)
```

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"hygon/MiniMax-M3.1-Flash-Channel-FP8-w8a8","messages":[{"role":"user","content":"请简单介绍一下你自己。"}],"max_tokens":128}'
```
