# Hy4 on vLLM

## 模型简介

Hy4-preview 是 Hy 系列模型，本文档提供其在 vLLM 上的部署示例。

## 模型列表

| 模型权重 | 量化方式 | vLLM 镜像 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | --------- | -------- | ---- | -------- | -------- |
| [hygon/Hy4-preview-Channel-FP8-w8a8](https://www.modelscope.cn/models/hygon/Hy4-preview-Channel-FP8-w8a8) | FP8 W8A8 | 0.25.1 | BW1100 | 8 | IFB | [**`>_`**](#hy4-preview-channel-fp8-w8a8-ifb-bw1100-8x-vllm-0251) |
| [hygon/Hy4-preview-Channel-INT4-w4a8](https://www.modelscope.cn/models/hygon/Hy4-preview-Channel-INT4-w4a8) | INT4 W4A8 | 0.25.1 | BW1100 | 32 | IFB | [**`>_`**](#hy4-preview-channel-int4-w4a8-ifb-ep32-ll-bw1100-32x-vllm-0251) |

## 启动命令

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

### Hy4-preview-Channel-INT4-w4a8 IFB EP32-LL BW1100 32x vLLM 0.25.1

4 节点 × 每节点 8 卡，共 32 卡。并行为 TP8 × DP4，开启 Expert Parallel（EP32），`--all2all-backend deepep_low_latency`。请将 `master-addr` 的 `10.x.x.x` 换成 Node 0 的实际 IP；各节点的 `VLLM_HOST_IP` 换成**本机** IP；将 `checkpoint_index` 换成模型目录下 `hy4-checkpoint.index.json` 的实际路径；`ROCSHMEM_TOPO_FILE_FORCE` / `ROCSHMEM_ALLOWED_IBV_DEVICES` / `GLOO_SOCKET_IFNAME` 按本机 IB 拓扑调整。

#### Node 0

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export GPU_MAX_HW_QUEUES=3
export VLLM_HCU_USE_LIGHTOP_HY_V4_INDEXER=0
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export VLLM_HOST_IP=10.x.x.x
export GLOO_SOCKET_IFNAME=ib0,ib1,ib2,ib3
export VLLM_ENGINE_READY_TIMEOUT_S=10800
export HSA_USE_SVM=0
export DEEPEP_ENABLE_LL_LAYERED_OPT=1
export ROCSHMEM_ALLOWED_IBV_DEVICES=shca_0,shca_1,shca_2,shca_3
export ROCSHMEM_TOPO_FILE_FORCE=./topo.config
export ROCSHMEM_DISABLE_AUTOTOPO=1

vllm serve hygon/Hy4-preview-Channel-INT4-w4a8 \
  --served-model-name hygon/Hy4-preview-Channel-INT4-w4a8 \
  --distributed-executor-backend mp \
  --nnodes 4 \
  --node-rank 0 \
  --master-addr 10.x.x.x \
  --master-port 29500 \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 1 \
  --data-parallel-size 4 \
  --enable-expert-parallel \
  --all2all-backend deepep_low_latency \
  --moe-backend deep_gemm \
  --no-async-scheduling \
  --hf-overrides '{"quantization_config":{"quant_method":"slimquant_w4a8","checkpoint_format":"hy4_w4a8_v1","checkpoint_index":"/path/to/Hy4-preview-Channel-INT4-w4a8/hy4-checkpoint.index.json"}}' \
  --kv-cache-dtype fp8_ds_mla \
  --gpu-memory-utilization 0.85 \
  --max-model-len 262144 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 512 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --compilation-config.max_cudagraph_capture_size 16 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --default-chat-template-kwargs '{"reasoning_effort":"high"}' \
  --reasoning-parser hy_v4 \
  --enable-auto-tool-choice \
  --tool-call-parser hy_v4
```

#### Node 1

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export GPU_MAX_HW_QUEUES=3
export VLLM_HCU_USE_LIGHTOP_HY_V4_INDEXER=0
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export VLLM_HOST_IP=10.x.x.x
export GLOO_SOCKET_IFNAME=ib0,ib1,ib2,ib3
export VLLM_ENGINE_READY_TIMEOUT_S=10800
export HSA_USE_SVM=0
export DEEPEP_ENABLE_LL_LAYERED_OPT=1
export ROCSHMEM_ALLOWED_IBV_DEVICES=shca_0,shca_1,shca_2,shca_3
export ROCSHMEM_TOPO_FILE_FORCE=./topo.config
export ROCSHMEM_DISABLE_AUTOTOPO=1

vllm serve hygon/Hy4-preview-Channel-INT4-w4a8 \
  --served-model-name hygon/Hy4-preview-Channel-INT4-w4a8 \
  --distributed-executor-backend mp \
  --nnodes 4 \
  --node-rank 1 \
  --master-addr 10.x.x.x \
  --master-port 29500 \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 1 \
  --data-parallel-size 4 \
  --enable-expert-parallel \
  --all2all-backend deepep_low_latency \
  --moe-backend deep_gemm \
  --no-async-scheduling \
  --headless \
  --hf-overrides '{"quantization_config":{"quant_method":"slimquant_w4a8","checkpoint_format":"hy4_w4a8_v1","checkpoint_index":"/path/to/Hy4-preview-Channel-INT4-w4a8/hy4-checkpoint.index.json"}}' \
  --kv-cache-dtype fp8_ds_mla \
  --gpu-memory-utilization 0.85 \
  --max-model-len 262144 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 512 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --compilation-config.max_cudagraph_capture_size 16 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --default-chat-template-kwargs '{"reasoning_effort":"high"}'
```

#### Node 2

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export GPU_MAX_HW_QUEUES=3
export VLLM_HCU_USE_LIGHTOP_HY_V4_INDEXER=0
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export VLLM_HOST_IP=10.x.x.x
export GLOO_SOCKET_IFNAME=ib0,ib1,ib2,ib3
export VLLM_ENGINE_READY_TIMEOUT_S=10800
export HSA_USE_SVM=0
export DEEPEP_ENABLE_LL_LAYERED_OPT=1
export ROCSHMEM_ALLOWED_IBV_DEVICES=shca_0,shca_1,shca_2,shca_3
export ROCSHMEM_TOPO_FILE_FORCE=./topo.config
export ROCSHMEM_DISABLE_AUTOTOPO=1

vllm serve hygon/Hy4-preview-Channel-INT4-w4a8 \
  --served-model-name hygon/Hy4-preview-Channel-INT4-w4a8 \
  --distributed-executor-backend mp \
  --nnodes 4 \
  --node-rank 2 \
  --master-addr 10.x.x.x \
  --master-port 29500 \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 1 \
  --data-parallel-size 4 \
  --enable-expert-parallel \
  --all2all-backend deepep_low_latency \
  --moe-backend deep_gemm \
  --no-async-scheduling \
  --headless \
  --hf-overrides '{"quantization_config":{"quant_method":"slimquant_w4a8","checkpoint_format":"hy4_w4a8_v1","checkpoint_index":"/path/to/Hy4-preview-Channel-INT4-w4a8/hy4-checkpoint.index.json"}}' \
  --kv-cache-dtype fp8_ds_mla \
  --gpu-memory-utilization 0.85 \
  --max-model-len 262144 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 512 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --compilation-config.max_cudagraph_capture_size 16 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --default-chat-template-kwargs '{"reasoning_effort":"high"}'
```

#### Node 3

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export GPU_MAX_HW_QUEUES=3
export VLLM_HCU_USE_LIGHTOP_HY_V4_INDEXER=0
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export VLLM_HOST_IP=10.x.x.x
export GLOO_SOCKET_IFNAME=ib0,ib1,ib2,ib3
export VLLM_ENGINE_READY_TIMEOUT_S=10800
export HSA_USE_SVM=0
export DEEPEP_ENABLE_LL_LAYERED_OPT=1
export ROCSHMEM_ALLOWED_IBV_DEVICES=shca_0,shca_1,shca_2,shca_3
export ROCSHMEM_TOPO_FILE_FORCE=./topo.config
export ROCSHMEM_DISABLE_AUTOTOPO=1

vllm serve hygon/Hy4-preview-Channel-INT4-w4a8 \
  --served-model-name hygon/Hy4-preview-Channel-INT4-w4a8 \
  --distributed-executor-backend mp \
  --nnodes 4 \
  --node-rank 3 \
  --master-addr 10.x.x.x \
  --master-port 29500 \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 1 \
  --data-parallel-size 4 \
  --enable-expert-parallel \
  --all2all-backend deepep_low_latency \
  --moe-backend deep_gemm \
  --no-async-scheduling \
  --headless \
  --hf-overrides '{"quantization_config":{"quant_method":"slimquant_w4a8","checkpoint_format":"hy4_w4a8_v1","checkpoint_index":"/path/to/Hy4-preview-Channel-INT4-w4a8/hy4-checkpoint.index.json"}}' \
  --kv-cache-dtype fp8_ds_mla \
  --gpu-memory-utilization 0.85 \
  --max-model-len 262144 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 512 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --compilation-config.max_cudagraph_capture_size 16 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --default-chat-template-kwargs '{"reasoning_effort":"high"}'
```

## API 调用

### IFB

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="hygon/Hy4-preview-Channel-INT4-w4a8",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=128,
)
print(response.choices[0].message.content)
```

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"hygon/Hy4-preview-Channel-INT4-w4a8","messages":[{"role":"user","content":"Hello"}],"max_tokens":128}'
```
