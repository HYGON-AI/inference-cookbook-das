# DeepSeek-V4.1 on SGLang

## 模型简介

DeepSeek-V4.1 是 DeepSeek 系列的混合专家（MoE）大语言模型。

## 模型列表

| 模型权重 | 量化方式 | SGLang 镜像 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | ----------- | -------- | ---- | -------- | -------- |
| hygon/DeepSeek-V4.1-Flash-1017-Channel-FP8-w8a8-wo_a-markov_head-lm_head | FP8 W8A8 | [0.5.18](../docker_images.md) | BW1100 | 16 | PD | [**`>_`**](#deepseek-v41-flash-1017-channel-fp8-w8a8-wo_a-markov_head-lm_head-pd-bw1100-16x-sglang-0518) |

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

## EPLB配置参考：[EPLB](../../optimization/static-eplb-sglang.md)。

## 启动命令

### DeepSeek-V4.1-Flash-1017-Channel-FP8-w8a8-wo_a-markov_head-lm_head PD BW1100 16x SGLang 0.5.18

网卡配置参考：[IB 网卡](../../troubleshooting/common-issues.md#ib网卡)。请按实际环境设置通信网卡和设备列表，并将 `<P_node0_ip>`、`<D_node0_ip>` 替换为两台节点之间可达的 IP。

#### P node 0

```bash
export NCCL_SOCKET_IFNAME=ens73f0np0
export GLOO_SOCKET_IFNAME=ens73f0np0
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
export ROCSHMEM_MAX_NUM_CONTEXTS=48
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_8,mlx5_9
export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_8:1,mlx5_9:1
export MC_ALLOWED_IBV_DEVICES=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_8,mlx5_9
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1
export SGLANG_HCU_MEGA_MOE_RUNTIME=deep_gemm
export SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=4096

sglang serve \
  --model-path hygon/DeepSeek-V4.1-Flash-1017-Channel-FP8-w8a8-wo_a-markov_head-lm_head \
  --tp-size 8 \
  --host "<P_node0_ip>" \
  --port 30000 \
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
  --disaggregation-ib-device mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_8,mlx5_9 \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-bootstrap-port 8998 \
  --init-expert-location "<path_to_eplb_trace_pt>" \
  --ep-num-redundant-experts 32 \
  --ep-dispatch-algorithm lp
```

#### D node 0

```bash
export NCCL_SOCKET_IFNAME=ens73f0np0
export GLOO_SOCKET_IFNAME=ens73f0np0
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
export ROCSHMEM_MAX_NUM_CONTEXTS=48
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_8,mlx5_9
export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_8:1,mlx5_9:1
export MC_ALLOWED_IBV_DEVICES=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_8,mlx5_9
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1
export SGLANG_USE_FP8_W8A8_MOE=1
export DEEPEP_ENABLE_LL_LAYERED_OPT=1
export ROCSHMEM_DISABLE_HDP_FLUSH=1
export SGLANG_HCU_FUSED_MHC_BOUNDARY=1
export SGLANG_SIMULATE_ACC_LEN=4
export SGLANG_SIMULATE_ACC_METHOD=match-expected

sglang serve \
  --model-path hygon/DeepSeek-V4.1-Flash-1017-Channel-FP8-w8a8-wo_a-markov_head-lm_head \
  --tp-size 8 \
  --host "<D_node0_ip>" \
  --port 30000 \
  --trust-remote-code \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --cuda-graph-backend-prefill disabled \
  --skip-server-warmup \
  --mem-fraction-static 0.85 \
  --max-running-requests 256 \
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
  --deepep-config /xxxxx/deepep_IntraConfig.json \
  --deepep-mode auto \
  --moe-runner-backend deep_gemm \
  --enable-single-batch-overlap \
  --speculative-moe-a2a-backend deepep \
  --speculative-moe-runner-backend deep_gemm \
  --disaggregation-ib-device mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_8,mlx5_9 \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-bootstrap-port 8998 \
  --disable-radix-cache
```

#### Router

```bash
python3 -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill "http://<P_node0_ip>:30000" \
  --decode "http://<D_node0_ip>:30000" \
  --host 0.0.0.0 \
  --port 30001
```

## API 调用

### PD 分离

请求发送至 Router 的 `30001` 端口，`model` 与 P、D 两端的 `--model-path` 一致。

```python
from openai import OpenAI

client = OpenAI(base_url="http://<router_ip>:30001/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="hygon/DeepSeek-V4.1-Flash-1017-Channel-FP8-w8a8-wo_a-markov_head-lm_head",
    messages=[{"role": "user", "content": "你好，请介绍一下你自己。"}],
    max_tokens=2048,
)
print(response.choices[0].message.content)
```

```bash
curl "http://<router_ip>:30001/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model": "hygon/DeepSeek-V4.1-Flash-1017-Channel-FP8-w8a8-wo_a-markov_head-lm_head", "messages": [{"role": "user", "content": "你好"}], "max_tokens": 128}'
```
