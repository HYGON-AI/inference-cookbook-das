# DeepSeek-V3 on vLLM

## 模型简介

DeepSeek-V3 是由深度求索推出的基于MoE架构的高性能开源大语言模型，以低成本训练实现接近甚至媲美顶级闭源模型的推理与编程能力。

## 模型列表

| 模型权重 | 量化方式 | vLLM 镜像 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | --------- | -------- | ---- | -------- | -------- |
| [hygon/DeepSeek-V3-0324-Channel-FP8-w8a8](https://www.modelscope.cn/models/hygon/DeepSeek-V3-0324-Channel-FP8-w8a8) | FP8 W8A8 | 0.28.1 | BW1100 | 8 | IFB | [**`>_`**](#deepseek-v3-0324-channel-fp8-w8a8-ifb-bw1100-8x-vllm-0281) |
|  | FP8 W8A8 | 0.21    | BW1100 | 8 | IFB | [**`>_`**](#deepseek-v3-0324-channel-fp8-w8a8-ifb-bw1100-8x-vllm-021) |
|                                                                                    | FP8 W8A8 | 0.18    | BW1100 | 8 | IFB | [**`>_`**](#deepseek-v3-0324-channel-fp8-w8a8-ifb-bw1100-8x-vllm-018) |
|                                                                                    | FP8 W8A8 | 0.15    | BW1100 | 8 | IFB | [**`>_`**](#deepseek-v3-0324-channel-fp8-w8a8-ifb-bw1100-8x-vllm-015) |
| [hygon/DeepSeek-V3-0324-Channel-INT8-w8a8](https://www.modelscope.cn/models/hygon/DeepSeek-V3-0324-Channel-INT8-w8a8) | INT8 W8A8 | 0.28.1 | BW1100 | 8 | IFB | [**`>_`**](#deepseek-v3-0324-channel-int8-w8a8-ifb-bw1100-8x-vllm-0281) |
|  | INT8 W8A8 | 0.28.1 | BW1000 | 16 | IFB | [**`>_`**](#deepseek-v3-0324-channel-int8-w8a8-ifb-bw1000-16x-vllm-0281) |
|  | INT8 W8A8 | 0.28.1 | K100_AI | 16 | IFB | [**`>_`**](#deepseek-v3-0324-channel-int8-w8a8-ifb-k100_ai-16x-vllm-0281) |
|  | INT8 W8A8 | [0.21](../docker_images.md) | BW1100 | 8 | IFB | [**`>_`**](#deepseek-v3-0324-channel-int8-w8a8-ifb-bw1100-8x-vllm-021) |
|  | INT8 W8A8 | [0.21](../docker_images.md) | BW1000 | 16 | IFB | [**`>_`**](#deepseek-v3-0324-channel-int8-w8a8-ifb-bw1000-16x-vllm-021) |

## 启动命令

### DeepSeek-V3-0324-Channel-FP8-w8a8 IFB BW1100 8x vLLM 0.28.1

```bash

vllm serve hygon/DeepSeek-V3-0324-Channel-FP8-w8a8 \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --dtype bfloat16 \
  --quantization slimquant_marlin \
  --attention-backend FLASHMLA \
  --moe-backend aiter \
  --kv-cache-dtype fp8_e4m3 \
  --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"quantization":"slimquant_marlin"}' \
  --enable-auto-tool-choice \
  --tool-call-parser deepseek_v3 \
  --reasoning-parser deepseek_v3 \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
```

### DeepSeek-V3-0324-Channel-FP8-w8a8 IFB BW1100 8x vLLM 0.21

```bash
vllm serve hygon/DeepSeek-V3-0324-Channel-FP8-w8a8 \
    --trust-remote-code \
    -q slimquant_marlin \
    -tp 8 \
    --dtype bfloat16 \
    --max-model-len 35000 \
    --gpu-memory-utilization 0.90 \
    --max-num-batched-tokens 16384 \
    --kv-cache-dtype fp8_e4m3 \
    --speculative_config '{
        "method": "deepseek_mtp",
        "num_speculative_tokens": 3,
        "quantization": "slimquant_marlin"
    }' \
    --attention-backend FLASHMLA
```

### DeepSeek-V3-0324-Channel-FP8-w8a8 IFB BW1100 8x vLLM 0.18

```bash
export VLLM_HCU_USE_FLASHMLA=1

vllm serve hygon/DeepSeek-V3-0324-Channel-FP8-w8a8 \
    --trust-remote-code \
    -q slimquant_marlin \
    -tp 8 \
    --dtype bfloat16 \
    --max-model-len 35000 \
    --gpu-memory-utilization 0.90 \
    --max-num-batched-tokens 16384 \
    --compilation-config '{
        "pass_config": {
            "fuse_act_quant": false
        }
    }' \
    --kv-cache-dtype fp8 \
    --speculative_config '{
        "method": "deepseek_mtp",
        "num_speculative_tokens": 3,
        "quantization": "slimquant_marlin"
    }'
```

### DeepSeek-V3-0324-Channel-FP8-w8a8 IFB BW1100 8x vLLM 0.15

```bash
rm -rf ~/.cache
rm -rf ~/.triton
export VLLM_USE_MODELSCOPE=1
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export Allgather_Base_STREAM_WITH_COMPUTE=1
export SENDRECV_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export VLLM_USE_CAT_MLA=1
export VLLM_SPEC_DECODE_EAGER=1
export VLLM_USE_GLOBAL_CACHE13=1
export VLLM_FUSED_MOE_CHUNK_SIZE=16384
export VLLM_USE_LIGHTOP=1
export VLLM_USE_FLASH_ATTN_FP8=1
vllm serve hygon/DeepSeek-V3-0324-Channel-INT8-w8a8  \
  --trust-remote-code   \
  -q slimquant_marlin \
  --dtype bfloat16  \
  -tp 8   \
  --max-model-len 65536  \
  --disable-log-requests  \
  --gpu-memory-utilization 0.90 \
  --max-num-batched-tokens 16384 \
  --compilation-config '{
      "pass_config": {
          "fuse_act_quant": false
      }
  }' \
  --kv-cache-dtype fp8 \
  --speculative_config '{
      "method": "deepseek_mtp",
      "num_speculative_tokens": 3,
      "quantization": "slimquant_marlin"
  }'
```

### DeepSeek-V3-0324-Channel-INT8-w8a8 IFB BW1100 8x vLLM 0.28.1

```bash

vllm serve hygon/DeepSeek-V3-0324-Channel-INT8-w8a8 \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --dtype bfloat16 \
  --quantization slimquant_marlin \
  --attention-backend FLASHMLA \
  --moe-backend aiter \
  --kv-cache-dtype fp8_e4m3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"quantization":"slimquant_marlin"}' \
  --enable-auto-tool-choice \
  --tool-call-parser deepseek_v3 \
  --reasoning-parser deepseek_v3 \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
```

### DeepSeek-V3-0324-Channel-INT8-w8a8 IFB BW1000 16x vLLM 0.28.1

总计 16 卡，2 个节点；各节点执行对应 rank 命令，MASTER_ADDR 和 MASTER_PORT 在所有节点设置为相同值。

#### rank 0

```bash

vllm serve hygon/DeepSeek-V3-0324-Channel-INT8-w8a8 \
  --trust-remote-code \
  --tensor-parallel-size 16 \
  --nnodes 2 \
  --node-rank 0 \
  --master-addr "${MASTER_ADDR}" \
  --master-port "${MASTER_PORT}" \
  --dtype bfloat16 \
  --quantization slimquant_marlin \
  --attention-backend FLASHMLA \
  --moe-backend aiter \
  --kv-cache-dtype fp8_e5m2 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"quantization":"slimquant_marlin"}' \
  --enable-auto-tool-choice \
  --tool-call-parser deepseek_v3 \
  --reasoning-parser deepseek_v3 \
  --no-async-scheduling \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
```

#### rank 1

```bash

vllm serve hygon/DeepSeek-V3-0324-Channel-INT8-w8a8 \
  --trust-remote-code \
  --tensor-parallel-size 16 \
  --nnodes 2 \
  --node-rank 1 \
  --master-addr "${MASTER_ADDR}" \
  --master-port "${MASTER_PORT}" \
  --dtype bfloat16 \
  --quantization slimquant_marlin \
  --attention-backend FLASHMLA \
  --moe-backend aiter \
  --kv-cache-dtype fp8_e5m2 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"quantization":"slimquant_marlin"}' \
  --enable-auto-tool-choice \
  --tool-call-parser deepseek_v3 \
  --reasoning-parser deepseek_v3 \
  --no-async-scheduling \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64 \
  --headless
```


### DeepSeek-V3-0324-Channel-INT8-w8a8 IFB K100_AI 16x vLLM 0.28.1

在各节点分别执行对应命令；`MASTER_ADDR` 为 rank 0 节点地址，各节点使用相同的 `MASTER_PORT`。数据并行地址和端口占位符如有使用，也应在各节点保持一致。

#### rank 0

```bash
export VLLM_HCU_USE_CUSTOM_OPS=0

vllm serve hygon/DeepSeek-V3-0324-Channel-INT8-w8a8 \
  --trust-remote-code \
  --tensor-parallel-size 16 \
  --nnodes 2 \
  --node-rank 0 \
  --master-addr "${MASTER_ADDR}" \
  --master-port "${MASTER_PORT}" \
  --dtype bfloat16 \
  --quantization slimquant_marlin \
  --attention-backend TRITON_MLA \
  --moe-backend triton \
  --no-enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"quantization":"slimquant_marlin"}' \
  --no-async-scheduling \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
```

#### rank 1

```bash
export VLLM_HCU_USE_CUSTOM_OPS=0

vllm serve hygon/DeepSeek-V3-0324-Channel-INT8-w8a8 \
  --trust-remote-code \
  --tensor-parallel-size 16 \
  --nnodes 2 \
  --node-rank 1 \
  --master-addr "${MASTER_ADDR}" \
  --master-port "${MASTER_PORT}" \
  --dtype bfloat16 \
  --quantization slimquant_marlin \
  --attention-backend TRITON_MLA \
  --moe-backend triton \
  --no-enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"quantization":"slimquant_marlin"}' \
  --no-async-scheduling \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64 \
  --headless
```

### DeepSeek-V3-0324-Channel-INT8-w8a8 IFB BW1100 8x vLLM 0.21

```bash
vllm serve hygon/DeepSeek-V3-0324-Channel-INT8-w8a8 \
  --trust-remote-code \
  -q slimquant_marlin \
  -tp 8 \
  --dtype bfloat16 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.90 \
  --max-num-batched-tokens 16384 \
  --speculative_config '{"method": "deepseek_mtp", "num_speculative_tokens": 2, "quantization": "slimquant_marlin"}' \
  --no-enable-prefix-caching \
  --kv-cache-dtype fp8_e4m3 \
  --attention-backend FLASHMLA
```

### DeepSeek-V3-0324-Channel-INT8-w8a8 IFB BW1000 16x vLLM 0.21

#### Node 0

```bash
vllm serve hygon/DeepSeek-V3-0324-Channel-INT8-w8a8 \
  --trust-remote-code \
  -q slimquant_marlin \
  -tp 16 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --no-enable-prefix-caching \
  --max-model-len 65536 \
  --max-num-seqs 256 \
  --max-num-batched-tokens 16384 \
  --speculative_config '{"method": "deepseek_mtp", "num_speculative_tokens": 3, "quantization": "slimquant_marlin"}' \
  --attention-backend FLASHMLA \
  --nnodes 2 \
  --node-rank 0 \
  --master-addr <master_node_ip> \
  --no-async-scheduling
```

#### Node 1

```bash
vllm serve hygon/DeepSeek-V3-0324-Channel-INT8-w8a8 \
  --trust-remote-code \
  -q slimquant_marlin \
  -tp 16 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --no-enable-prefix-caching \
  --max-model-len 65536 \
  --max-num-seqs 256 \
  --max-num-batched-tokens 16384 \
  --speculative_config '{"method": "deepseek_mtp", "num_speculative_tokens": 3, "quantization": "slimquant_marlin"}' \
  --attention-backend FLASHMLA \
  --nnodes 2 \
  --node-rank 1 \
  --master-addr <master_node_ip> \
  --no-async-scheduling \
  --headless
```

## API 调用

### IFB

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="hygon/DeepSeek-V3-0324-Channel-INT8-w8a8",  # 替换为实际使用的模型名
    messages=[
        {"role": "system", "content": "你是一个专业的编程助手。"},
        {"role": "user", "content": "用 Python 实现一个高效的 LRU Cache"},
    ],
    max_tokens=2048,
    temperature=0.7,
)
print(response.choices[0].message.content)
```

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
  "model": "hygon/DeepSeek-V3-0324-Channel-INT8-w8a8",
  "messages": [
    {"role": "system", "content": "你是一个专业的编程助手。"},
    {"role": "user", "content": "用 Python 实现一个高效的 LRU Cache"}
  ],
  "max_tokens": 128
  }'
```
