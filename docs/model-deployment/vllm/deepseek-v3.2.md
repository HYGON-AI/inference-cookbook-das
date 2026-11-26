# DeepSeek-V3.2 on vLLM

## 模型简介

DeepSeek V3.2 是深度求索公司于 2025 年底发布的大语言模型，基于创新的 DeepSeek Sparse Attention (DSA) 稀疏注意力机制，支持 128K 上下文。

## 模型列表

| 模型权重 | 量化方式 | vLLM 镜像 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | --------- | -------- | ---- | -------- | -------- |
| DeepSeek-V3.2-bf16（本地权重） | BF16 | 0.28.1 | BW1100（96GB） | 32（4 × 8） | IFB | [**`>_`**](#deepseek-v32-bf16-ifb-bw110096gb-32x-vllm-0281) |
|  | BF16 | 0.28.1 | BW1000 | 32（4 × 8） | IFB | [**`>_`**](#deepseek-v32-bf16-ifb-bw1000-32x-vllm-0281) |
| [hygon/DeepSeek-V3.2-Channel-FP8-w8a8](https://www.modelscope.cn/models/hygon/DeepSeek-V3.2-Channel-FP8-w8a8) | FP8 W8A8 | 0.28.1 | BW1100 | 8 | IFB | [**`>_`**](#deepseek-v32-channel-fp8-w8a8-ifb-bw1100-8x-vllm-0281) |
|  | FP8 W8A8 | [0.21](../docker_images.md) | BW1100 | 8 | IFB | [**`>_`**](#deepseek-v32-channel-fp8-w8a8-ifb-bw1100-8x-vllm-021) |
|  | FP8 W8A8 | [0.18](../docker_images.md) | BW1100 | 8 | IFB | [**`>_`**](#deepseek-v32-channel-fp8-w8a8-ifb-bw1100-8x-vllm-018) |
| [hygon/DeepSeek-V3.2-Channel-INT8-w8a8](https://www.modelscope.cn/models/hygon/DeepSeek-V3.2-Channel-INT8-w8a8) | INT8 W8A8 | 0.28.1 | BW1100 | 8 | IFB | [**`>_`**](#deepseek-v32-channel-int8-w8a8-ifb-bw1100-8x-vllm-0281) |
|  | INT8 W8A8 | 0.28.1 | BW1000 | 16 | IFB | [**`>_`**](#deepseek-v32-channel-int8-w8a8-ifb-bw1000-16x-vllm-0281) |
|  | INT8 W8A8 | 0.28.1 | K100_AI | 16 | IFB | [**`>_`**](#deepseek-v32-channel-int8-w8a8-ifb-k100_ai-16x-vllm-0281) |
|  | INT8 W8A8 | [0.21](../docker_images.md) | BW1100 | 8 | IFB | [**`>_`**](#deepseek-v32-channel-int8-w8a8-ifb-bw1100-8x-vllm-021) |
|  | INT8 W8A8 | [0.21](../docker_images.md) | BW1000 | 16 | IFB | [**`>_`**](#deepseek-v32-channel-int8-w8a8-ifb-bw1000-16x-vllm-021) |
|  | INT8 W8A8 | 0.18.1 | BW1100 | 8 | IFB | [**`>_`**](#deepseek-v32-channel-int8-w8a8-ifb-bw1100-8x-vllm-0181) |
|  | INT8 W8A8 | 0.15.1 | BW1100 | 8 | IFB | [**`>_`**](#deepseek-v32-channel-int8-w8a8-ifb-bw1100-8x-vllm-0151) |


### 0.28.1 平台支持说明

BW1000 不支持本次发版所列 FP8 权重模型；以下 FP8 权重仅提供 BW1100 启动命令：`hygon/DeepSeek-V3.2-Channel-FP8-w8a8`。

## 启动命令

### BF16 四机部署准备与参数说明

以下两套命令面向海光适配的 vLLM 0.28.1，均为 4 个节点、每节点 8 卡，总计 32 卡；BW1100 按每卡 96GB、BW1000 按每卡 64GB 编写。四台机器需使用相同的镜像及软件环境，并在容器内通过相同路径访问完整权重目录：

```text
/public/opendas/DL_DATA/llm-models/deepseek-v3.2/DeepSeek-V3.2-bf16
```

该目录的 `config.json` 为 `DeepseekV32ForCausalLM` / `deepseek_v32`，`torch_dtype` 为 `bfloat16`，且权重包含 MTP 层。命令使用 `--dtype bfloat16` 加载 BF16 权重；FP8 选项仅用于 KV cache。

两套配置综合参考了本页 V3.2 INT8 的稀疏注意力、tokenizer、MTP 和工具调用设置，[DeepSeek-R1 BF16 的 BW1100（96G）/ BW1000 四机配置](deepseek-r1.md)，以及 [GLM-5 BF16 的 BW1000 四机稀疏注意力配置](glm-5.md)。这些是待实机验证的启动配置，尚未在上述 BF16 权重和两种四机环境上完成启动、精度或性能验证。

| 参数 | BW1100（96GB）四机 | BW1000 四机 |
| ---- | ----------------- | ----------- |
| 并行方式 | TP32 + EP，`mp`，`--nnodes 4` | TP32 + EP，`mp`，`--nnodes 4` |
| Attention / MoE | `FLASHMLA_SPARSE` / `aiter` | `FLASHMLA_SPARSE` / `aiter` |
| KV cache / block size | `fp8_e4m3` / 64 | `fp8_ds_mla` / 64 |
| 上下文长度 | 131072（128K） | 32768（32K） |
| MTP | 3 个 speculative tokens | 3 个 speculative tokens |
| 批处理上限 | 16384 tokens / 64 sequences | 16384 tokens / 64 sequences |

BW1100 的 KV cache 设置沿用本页 0.28.1 示例；BW1000 使用本页 BW1000 示例的 `fp8_ds_mla` 稀疏 MLA 布局。BW1000 的 32K 上下文为初始配置，后续根据启动日志中的可用 KV cache 容量调整。

先在四个节点的启动终端中设置相同的通信地址和端口。将下面的 `192.168.1.10` 改为 rank 0 节点实际的内网 IP；`29500` 用于多机通信，需空闲且节点间可达。容器部署时需保证各节点间网络互通，例如使用 host 网络。

```bash
export MASTER_ADDR=192.168.1.10
export MASTER_PORT=29500
```

每台机器均需有 8 张空闲且对当前进程可见的 HCU。各节点执行对应 rank 的命令；rank 0 会等待其余节点加入，rank 1、2、3 使用 `--headless`。API 服务位于 rank 0 的默认端口 `8000`，请求中的模型名称统一为 `DeepSeek-V3.2-bf16`。

### DeepSeek-V3.2-bf16 IFB BW1100（96GB） 32x vLLM 0.28.1

按上节完成四个节点的准备后，分别执行以下命令。

#### rank 0

```bash
export NCCL_GRAPH_MIXING_SUPPORT=1

vllm serve /public/opendas/DL_DATA/llm-models/deepseek-v3.2/DeepSeek-V3.2-bf16 \
  --served-model-name DeepSeek-V3.2-bf16 \
  --trust-remote-code \
  --dtype bfloat16 \
  --tensor-parallel-size 32 \
  --enable-expert-parallel \
  --distributed-executor-backend mp \
  --nnodes 4 \
  --node-rank 0 \
  --master-addr "${MASTER_ADDR}" \
  --master-port "${MASTER_PORT}" \
  --tokenizer-mode deepseek_v32 \
  --attention-backend FLASHMLA_SPARSE \
  --moe-backend aiter \
  --all2all-backend allgather_reducescatter \
  --kv-cache-dtype fp8_e4m3 \
  --block-size 64 \
  --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --enable-auto-tool-choice \
  --tool-call-parser deepseek_v32 \
  --reasoning-parser deepseek_v3 \
  --no-async-scheduling \
  --gpu-memory-utilization 0.9 \
  --max-model-len 131072 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
```

#### rank 1

```bash
export NCCL_GRAPH_MIXING_SUPPORT=1

vllm serve /public/opendas/DL_DATA/llm-models/deepseek-v3.2/DeepSeek-V3.2-bf16 \
  --served-model-name DeepSeek-V3.2-bf16 \
  --trust-remote-code \
  --dtype bfloat16 \
  --tensor-parallel-size 32 \
  --enable-expert-parallel \
  --distributed-executor-backend mp \
  --nnodes 4 \
  --node-rank 1 \
  --master-addr "${MASTER_ADDR}" \
  --master-port "${MASTER_PORT}" \
  --headless \
  --tokenizer-mode deepseek_v32 \
  --attention-backend FLASHMLA_SPARSE \
  --moe-backend aiter \
  --all2all-backend allgather_reducescatter \
  --kv-cache-dtype fp8_e4m3 \
  --block-size 64 \
  --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --enable-auto-tool-choice \
  --tool-call-parser deepseek_v32 \
  --reasoning-parser deepseek_v3 \
  --no-async-scheduling \
  --gpu-memory-utilization 0.9 \
  --max-model-len 131072 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
```

#### rank 2

```bash
export NCCL_GRAPH_MIXING_SUPPORT=1

vllm serve /public/opendas/DL_DATA/llm-models/deepseek-v3.2/DeepSeek-V3.2-bf16 \
  --served-model-name DeepSeek-V3.2-bf16 \
  --trust-remote-code \
  --dtype bfloat16 \
  --tensor-parallel-size 32 \
  --enable-expert-parallel \
  --distributed-executor-backend mp \
  --nnodes 4 \
  --node-rank 2 \
  --master-addr "${MASTER_ADDR}" \
  --master-port "${MASTER_PORT}" \
  --headless \
  --tokenizer-mode deepseek_v32 \
  --attention-backend FLASHMLA_SPARSE \
  --moe-backend aiter \
  --all2all-backend allgather_reducescatter \
  --kv-cache-dtype fp8_e4m3 \
  --block-size 64 \
  --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --enable-auto-tool-choice \
  --tool-call-parser deepseek_v32 \
  --reasoning-parser deepseek_v3 \
  --no-async-scheduling \
  --gpu-memory-utilization 0.9 \
  --max-model-len 131072 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
```

#### rank 3

```bash
export NCCL_GRAPH_MIXING_SUPPORT=1

vllm serve /public/opendas/DL_DATA/llm-models/deepseek-v3.2/DeepSeek-V3.2-bf16 \
  --served-model-name DeepSeek-V3.2-bf16 \
  --trust-remote-code \
  --dtype bfloat16 \
  --tensor-parallel-size 32 \
  --enable-expert-parallel \
  --distributed-executor-backend mp \
  --nnodes 4 \
  --node-rank 3 \
  --master-addr "${MASTER_ADDR}" \
  --master-port "${MASTER_PORT}" \
  --headless \
  --tokenizer-mode deepseek_v32 \
  --attention-backend FLASHMLA_SPARSE \
  --moe-backend aiter \
  --all2all-backend allgather_reducescatter \
  --kv-cache-dtype fp8_e4m3 \
  --block-size 64 \
  --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --enable-auto-tool-choice \
  --tool-call-parser deepseek_v32 \
  --reasoning-parser deepseek_v3 \
  --no-async-scheduling \
  --gpu-memory-utilization 0.9 \
  --max-model-len 131072 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
```

### DeepSeek-V3.2-bf16 IFB BW1000 32x vLLM 0.28.1

按上节完成四个节点的准备后，分别执行以下命令。

#### rank 0

```bash
export NCCL_GRAPH_MIXING_SUPPORT=1

vllm serve /public/opendas/DL_DATA/llm-models/deepseek-v3.2/DeepSeek-V3.2-bf16 \
  --served-model-name DeepSeek-V3.2-bf16 \
  --trust-remote-code \
  --dtype bfloat16 \
  --tensor-parallel-size 32 \
  --enable-expert-parallel \
  --distributed-executor-backend mp \
  --nnodes 4 \
  --node-rank 0 \
  --master-addr "${MASTER_ADDR}" \
  --master-port "${MASTER_PORT}" \
  --tokenizer-mode deepseek_v32 \
  --attention-backend FLASHMLA_SPARSE \
  --moe-backend aiter \
  --all2all-backend allgather_reducescatter \
  --kv-cache-dtype fp8_ds_mla \
  --block-size 64 \
  --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --enable-auto-tool-choice \
  --tool-call-parser deepseek_v32 \
  --reasoning-parser deepseek_v3 \
  --no-async-scheduling \
  --gpu-memory-utilization 0.9 \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
```

#### rank 1

```bash
export NCCL_GRAPH_MIXING_SUPPORT=1

vllm serve /public/opendas/DL_DATA/llm-models/deepseek-v3.2/DeepSeek-V3.2-bf16 \
  --served-model-name DeepSeek-V3.2-bf16 \
  --trust-remote-code \
  --dtype bfloat16 \
  --tensor-parallel-size 32 \
  --enable-expert-parallel \
  --distributed-executor-backend mp \
  --nnodes 4 \
  --node-rank 1 \
  --master-addr "${MASTER_ADDR}" \
  --master-port "${MASTER_PORT}" \
  --headless \
  --tokenizer-mode deepseek_v32 \
  --attention-backend FLASHMLA_SPARSE \
  --moe-backend aiter \
  --all2all-backend allgather_reducescatter \
  --kv-cache-dtype fp8_ds_mla \
  --block-size 64 \
  --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --enable-auto-tool-choice \
  --tool-call-parser deepseek_v32 \
  --reasoning-parser deepseek_v3 \
  --no-async-scheduling \
  --gpu-memory-utilization 0.9 \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
```

#### rank 2

```bash
export NCCL_GRAPH_MIXING_SUPPORT=1

vllm serve /public/opendas/DL_DATA/llm-models/deepseek-v3.2/DeepSeek-V3.2-bf16 \
  --served-model-name DeepSeek-V3.2-bf16 \
  --trust-remote-code \
  --dtype bfloat16 \
  --tensor-parallel-size 32 \
  --enable-expert-parallel \
  --distributed-executor-backend mp \
  --nnodes 4 \
  --node-rank 2 \
  --master-addr "${MASTER_ADDR}" \
  --master-port "${MASTER_PORT}" \
  --headless \
  --tokenizer-mode deepseek_v32 \
  --attention-backend FLASHMLA_SPARSE \
  --moe-backend aiter \
  --all2all-backend allgather_reducescatter \
  --kv-cache-dtype fp8_ds_mla \
  --block-size 64 \
  --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --enable-auto-tool-choice \
  --tool-call-parser deepseek_v32 \
  --reasoning-parser deepseek_v3 \
  --no-async-scheduling \
  --gpu-memory-utilization 0.9 \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
```

#### rank 3

```bash
export NCCL_GRAPH_MIXING_SUPPORT=1

vllm serve /public/opendas/DL_DATA/llm-models/deepseek-v3.2/DeepSeek-V3.2-bf16 \
  --served-model-name DeepSeek-V3.2-bf16 \
  --trust-remote-code \
  --dtype bfloat16 \
  --tensor-parallel-size 32 \
  --enable-expert-parallel \
  --distributed-executor-backend mp \
  --nnodes 4 \
  --node-rank 3 \
  --master-addr "${MASTER_ADDR}" \
  --master-port "${MASTER_PORT}" \
  --headless \
  --tokenizer-mode deepseek_v32 \
  --attention-backend FLASHMLA_SPARSE \
  --moe-backend aiter \
  --all2all-backend allgather_reducescatter \
  --kv-cache-dtype fp8_ds_mla \
  --block-size 64 \
  --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --enable-auto-tool-choice \
  --tool-call-parser deepseek_v32 \
  --reasoning-parser deepseek_v3 \
  --no-async-scheduling \
  --gpu-memory-utilization 0.9 \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
```

### DeepSeek-V3.2-Channel-FP8-w8a8 IFB BW1100 8x vLLM 0.28.1

```bash

vllm serve hygon/DeepSeek-V3.2-Channel-FP8-w8a8 \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --tokenizer-mode deepseek_v32 \
  --attention-backend FLASHMLA_SPARSE \
  --moe-backend aiter \
  --kv-cache-dtype fp8_e4m3 \
  --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --enable-auto-tool-choice \
  --tool-call-parser deepseek_v32 \
  --reasoning-parser deepseek_v3 \
  --max-model-len 131072 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
```

### DeepSeek-V3.2-Channel-FP8-w8a8 IFB BW1100 8x vLLM 0.21

```bash
vllm serve hygon/DeepSeek-V3.2-Channel-FP8-w8a8 \
  -q slimquant_marlin \
  --trust-remote-code \
  --dtype bfloat16 \
  --max-model-len 65536 \
  --max-num-batched-tokens 16384 \
  -tp 8 \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 64 \
  --block-size 64 \
  --speculative_config '{
    "method":"deepseek_mtp",
    "num_speculative_tokens":2,
    "quantization":"slimquant_marlin"
  }' \
  --kv-cache-dtype fp8_ds_mla \
  --attention-backend FLASHMLA_SPARSE \
  --enable-auto-tool-choice \
  --tool-call-parser deepseek_v32 \
  --reasoning-parser deepseek_v3
```

### DeepSeek-V3.2-Channel-INT8-w8a8 IFB BW1100 8x vLLM 0.28.1

```bash

vllm serve hygon/DeepSeek-V3.2-Channel-INT8-w8a8 \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --tokenizer-mode deepseek_v32 \
  --attention-backend FLASHMLA_SPARSE \
  --moe-backend aiter \
  --kv-cache-dtype fp8_e4m3 \
  --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --enable-auto-tool-choice \
  --tool-call-parser deepseek_v32 \
  --reasoning-parser deepseek_v3 \
  --max-model-len 131072 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
```

### DeepSeek-V3.2-Channel-INT8-w8a8 IFB BW1000 16x vLLM 0.28.1

在各节点分别执行对应命令；`MASTER_ADDR` 为 rank 0 节点地址，各节点使用相同的 `MASTER_PORT`。数据并行地址和端口占位符如有使用，也应在各节点保持一致。

#### rank 0

```bash
export NCCL_GRAPH_MIXING_SUPPORT=1

vllm serve hygon/DeepSeek-V3.2-Channel-INT8-w8a8 \
  --trust-remote-code \
  --tensor-parallel-size 16 \
  --enable-expert-parallel \
  --distributed-executor-backend mp \
  --nnodes 2 \
  --node-rank 0 \
  --master-addr "${MASTER_ADDR}" \
  --master-port "${MASTER_PORT}" \
  --tokenizer-mode deepseek_v32 \
  --attention-backend FLASHMLA_SPARSE \
  --moe-backend aiter \
  --kv-cache-dtype fp8_ds_mla \
  --block-size 64 \
  --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --enable-auto-tool-choice \
  --tool-call-parser deepseek_v32 \
  --reasoning-parser deepseek_v3 \
  --no-async-scheduling \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
```

#### rank 1

```bash
export NCCL_GRAPH_MIXING_SUPPORT=1

vllm serve hygon/DeepSeek-V3.2-Channel-INT8-w8a8 \
  --trust-remote-code \
  --tensor-parallel-size 16 \
  --enable-expert-parallel \
  --distributed-executor-backend mp \
  --nnodes 2 \
  --node-rank 1 \
  --master-addr "${MASTER_ADDR}" \
  --master-port "${MASTER_PORT}" \
  --headless \
  --tokenizer-mode deepseek_v32 \
  --attention-backend FLASHMLA_SPARSE \
  --moe-backend aiter \
  --kv-cache-dtype fp8_ds_mla \
  --block-size 64 \
  --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --enable-auto-tool-choice \
  --tool-call-parser deepseek_v32 \
  --reasoning-parser deepseek_v3 \
  --no-async-scheduling \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
```

### DeepSeek-V3.2-Channel-INT8-w8a8 IFB K100_AI 16x vLLM 0.28.1

在各节点分别执行对应命令；`MASTER_ADDR` 为 rank 0 节点地址，各节点使用相同的 `MASTER_PORT`。数据并行地址和端口占位符如有使用，也应在各节点保持一致。

#### rank 0

```bash
export NCCL_GRAPH_MIXING_SUPPORT=1
export VLLM_HCU_USE_CUSTOM_OPS=0

vllm serve hygon/DeepSeek-V3.2-Channel-INT8-w8a8 \
  --trust-remote-code \
  --tensor-parallel-size 16 \
  --enable-expert-parallel \
  --distributed-executor-backend mp \
  --nnodes 2 \
  --node-rank 0 \
  --master-addr "${MASTER_ADDR}" \
  --master-port "${MASTER_PORT}" \
  --tokenizer-mode deepseek_v32 \
  --attention-backend FLASHMLA_SPARSE \
  --moe-backend triton \
  --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --enable-auto-tool-choice \
  --tool-call-parser deepseek_v32 \
  --reasoning-parser deepseek_v3 \
  --no-async-scheduling \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
```

#### rank 1

```bash
export NCCL_GRAPH_MIXING_SUPPORT=1
export VLLM_HCU_USE_CUSTOM_OPS=0

vllm serve hygon/DeepSeek-V3.2-Channel-INT8-w8a8 \
  --trust-remote-code \
  --tensor-parallel-size 16 \
  --enable-expert-parallel \
  --distributed-executor-backend mp \
  --nnodes 2 \
  --node-rank 1 \
  --master-addr "${MASTER_ADDR}" \
  --master-port "${MASTER_PORT}" \
  --tokenizer-mode deepseek_v32 \
  --attention-backend FLASHMLA_SPARSE \
  --moe-backend triton \
  --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --enable-auto-tool-choice \
  --tool-call-parser deepseek_v32 \
  --reasoning-parser deepseek_v3 \
  --no-async-scheduling \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64 \
  --headless
```

### DeepSeek-V3.2-Channel-INT8-w8a8 IFB BW1100 8x vLLM 0.21

```bash
vllm serve hygon/DeepSeek-V3.2-Channel-INT8-w8a8 \
    -q slimquant_marlin \
    --trust-remote-code \
    --dtype bfloat16 \
    --max-model-len 65536 \
    --max-num-batched-tokens 16384 \
    -tp 8 \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 64 \
    --block-size 64 \
    --speculative_config '{
        "method":"deepseek_mtp",
        "num_speculative_tokens":2,
        "quantization":"slimquant_marlin"
    }' \
    --kv-cache-dtype fp8_ds_mla \
    --attention-backend FLASHMLA_SPARSE \
    --enable-auto-tool-choice \
    --tool-call-parser deepseek_v32 \
    --reasoning-parser deepseek_v3
```

## 启动命令

### DeepSeek-V3.2-Channel-INT8-w8a8 IFB BW1000 16x vLLM 0.21

#### Node 0

```bash
vllm serve hygon/DeepSeek-V3.2-Channel-INT8-w8a8 \
  -q slimquant_marlin \
  --trust-remote-code \
  --dtype bfloat16 \
  --max-model-len 65536 \
  --max-num-batched-tokens 16384 \
  -tp 16 \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 64 \
  --block-size 64 \
  --speculative_config '{"method":"deepseek_mtp","num_speculative_tokens":2,"quantization":"slimquant_marlin"}' \
  --kv-cache-dtype fp8_ds_mla \
  --attention-backend FLASHMLA_SPARSE \
  --nnodes 2 \
  --node-rank 0 \
  --master-addr <master_node_ip> \
  --host 0.0.0.0
```

#### Node 1

```bash
vllm serve hygon/DeepSeek-V3.2-Channel-INT8-w8a8 \
  -q slimquant_marlin \
  --trust-remote-code \
  --dtype bfloat16 \
  --max-model-len 65536 \
  --max-num-batched-tokens 16384 \
  -tp 16 \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 64 \
  --block-size 64 \
  --speculative_config '{"method":"deepseek_mtp","num_speculative_tokens":2,"quantization":"slimquant_marlin"}' \
  --kv-cache-dtype fp8_ds_mla \
  --attention-backend FLASHMLA_SPARSE \
  --nnodes 2 \
  --node-rank 1 \
  --master-addr <master_node_ip> \
  --host 0.0.0.0 \
  --headless
```

### DeepSeek-V3.2-Channel-INT8-w8a8 IFB BW1100 8x vLLM 0.18.1

```bash
export VLLM_HCU_USE_FLASHMLA=1

vllm serve --model hygon/DeepSeek-V3.2-Channel-INT8 \
    --trust-remote-code \
    -q slimquant_marlin \
    --dtype bfloat16 \
    --max-model-len 28364 \
    -tp 8 \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs 256 \
    --block-size 64 \
    --kv-cache-dtype fp8_ds_mla \
    --speculative_config '{"method": "deepseek_mtp", "num_speculative_tokens": 3, "quantization": "slimquant_marlin"}'
```

### DeepSeek-V3.2-Channel-FP8-w8a8 IFB BW1100 8x vLLM 0.18

```bash
export VLLM_USE_MODELSCOPE=1
export VLLM_HCU_USE_FLASHMLA=1
export LMSLIM_USE_GLOBAL_MOE_CACHE=1

vllm serve hygon/DeepSeek-V3.2-Channel-FP8-w8a8 \
  --trust-remote-code \
  --dtype bfloat16 \
  -tp 8 \
  --gpu-memory-utilization 0.92 \
  --host 0.0.0.0 \
  --max-model-len 65536 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 256 \
  --block-size 64 \
  -q slimquant_marlin \
  --kv-cache-dtype fp8_ds_mla \
  --speculative_config '{"method": "deepseek_mtp", "num_speculative_tokens": 2, "quantization":"slimquant_marlin"}'
```

## API 调用

### BF16 四机服务

两种硬件配置均在 rank 0 节点执行以下命令；从其他机器访问时，将 `localhost` 替换为 rank 0 的 IP。

```bash
curl --fail http://localhost:8000/health

curl --fail http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "DeepSeek-V3.2-bf16",
    "messages": [{"role": "user", "content": "你好，请用一句话介绍你自己。"}],
    "temperature": 0,
    "max_tokens": 256
  }'
```

### IFB

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="hygon/DeepSeek-V3.2-Channel-INT8-w8a8",
    messages=[
        {"role": "user", "content": "请总结以下长文档的关键要点..."},
    ],
    max_tokens=4096,
    temperature=0,
)

print(response.choices[0].message.content)
```

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "hygon/DeepSeek-V3.2-Channel-INT8-w8a8",
        "messages": [{"role": "user", "content": "中国的首都是哪里？"}],
        "temperature": 0,
        "max_tokens": 100
    }'
```

### DeepSeek-V3.2-Channel-INT8-w8a8 IFB BW1100 8x vLLM 0.15.1

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7     
export ALLREDUCE_STREAM_WITH_COMPUTE=1           
export Allgather_Base_STREAM_WITH_COMPUTE=1
export SENDRECV_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1   
export VLLM_RPC_TIMEOUT=1800000  

export VLLM_SPEC_DECODE_EAGER=0
export VLLM_USE_GLOBAL_CACHE13=1
export VLLM_FUSED_MOE_CHUNK_SIZE=8192
export VLLM_REJECT_SAMPLE_OPT=0
export VLLM_USE_PIECEWISE=0
export USE_FUSED_RMS_QUANT=1
export USE_FUSED_SILU_MUL_QUANT=1
export VLLM_USE_OPT_CAT=1
export VLLM_USE_FUSED_FILL_RMS_CAT=1 
export VLLM_USE_LIGHTOP_MOE_SUM_MUL_ADD=0
export VLLM_USE_LIGHTOP_RMS_ROPE_CONCAT=0 
export VLLM_DISABLE_DSA=0
export VLLM_USE_V32_ENCODE=1
export VLLM_USE_FLASH_MLA=1
export VLLM_USE_FLASH_ATTN_FP8=1
export VLLM_USE_CAT_MLA=1

vllm serve hygon/DeepSeek-V3.2-Channel-INT8-w8a8 \
 --dtype bfloat16 \
 --trust-remote-code \
 --max-model-len 40960 \
 -tp 8 \
 --gpu-memory-utilization 0.92 \
 --max-num-seqs 256 \
 --disable-log-requests \
 --enable-chunked-prefill \
 --max-num-batched-tokens 16384 \
 --no-enable-prefix-caching \
 -cc '{"pass_config": {"fuse_act_quant": false}}' \
 --speculative_config '{"method": "mtp", "num_speculative_tokens": 2}' \
 --kv-cache-dtype fp8_ds_mla

```

## API 调用

### IFB

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="hygon/DeepSeek-V3.2-Channel-INT8-w8a8",
    messages=[
        {"role": "user", "content": "请总结以下长文档的关键要点..."},
    ],
    max_tokens=4096,
    temperature=0,

)
print(response.choices[0].message.content)
 
```

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json"  \
    -d '{
        "model": "hygon/DeepSeek-V3.2-Channel-INT8-w8a8", 
        "messages": [{"role": "user", "content": "中国的首都是什么？"}], 
        "temperature": 0, 
        "max_tokens": 100
    }'
```
