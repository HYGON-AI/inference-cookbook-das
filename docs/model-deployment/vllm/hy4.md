# Hy4 on vLLM

## 模型简介

Hy4-preview 是 Hy 系列模型，本文档提供其在 vLLM 上的部署示例。

## 模型列表

| 模型权重 | 量化方式 | vLLM 镜像 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | --------- | -------- | ---- | -------- | -------- |
| [hygon/Hy4-preview-Channel-FP8-w8a8](https://www.modelscope.cn/models/hygon/Hy4-preview-Channel-FP8-w8a8) | FP8 W8A8 | 0.28.1 | BW1100 | 8 | IFB | [**`>_`**](#hy4-preview-channel-fp8-w8a8-ifb-bw1100-8x-vllm-0281) |
|  | FP8 W8A8 | 0.25.1 | BW1100 | 8 | IFB | [**`>_`**](#hy4-preview-channel-fp8-w8a8-ifb-bw1100-8x-vllm-0251) |
| [hygon/Hy4-preview-Channel-INT4-w4a8](https://www.modelscope.cn/models/hygon/Hy4-preview-Channel-INT4-w4a8) | INT4 W4A8 | 0.25.1 | BW1100 | 32 | 1P1D | [**`>_`**](#hy4-preview-channel-int4-w4a8-1p1d-bw1100-32x-vllm-0251) |


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
  --max-model-len 131072 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
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

### Hy4-preview-Channel-INT4-w4a8 1P1D BW1100 32x vLLM 0.25.1

P、D 各 16 卡，离线 EPLB 冗余专家数为 16。将 `P_NODE_IP`、`D_NODE_IP` 替换为节点地址，以下命令在仓库根目录执行。RDMA 网卡配置参考：[IB 网卡](../../troubleshooting/common-issues.md#ib网卡)。

镜像：`42.228.13.241:5000/hcu/base_env/vllm:0.25.1-ubuntu22.04-dtk26041-py3.10-20261110-1933`。

离线映射使用 [r16.json](configs/hy4/r16.json)，P 启动时直接加载。

#### P node

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_USE_MODELSCOPE=1
export VLLM_HOST_IP="P_NODE_IP"
export VLLM_MOONCAKE_BOOTSTRAP_PORT=18998
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export ROCSHMEM_GDR_DISABLE_XDP=1
export DEEP_EP_NORMAL_MNVL=1
export VLLM_DEEPEP_BUFFER_SIZE_MB=4096

export GPU_MAX_HW_QUEUES=4
export VLLM_HCU_ENABLE_BYTELEVEL_DETOKENIZER_TAIL=1
export VLLM_HCU_ENABLE_PREFILL_MQA_GROUPING=1
export VLLM_HCU_ENABLE_TRITON_CP_GATHER=1
export VLLM_HCU_HYV4_MOE_CHUNK_SIZE=1024
export VLLM_HCU_SHARED_EXPERTS_STREAM_FORCE=1
export VLLM_HCU_STATIC_HT_W4A8_COUNT_SCRATCH=1
export HY4_EXPERIMENTAL_STATIC_R16=1
export LIGHTOP_MQA_LOGITS_BETA=1
export TOKENIZERS_PARALLELISM=true
export RAYON_NUM_THREADS=4

vllm serve hygon/Hy4-preview-Channel-INT4-w4a8 \
  --trust-remote-code \
  --quantization compressed-tensors \
  --host 0.0.0.0 \
  --tensor-parallel-size 1 \
  --enable-expert-parallel \
  --moe-backend deep_gemm \
  --enable-prefix-caching \
  --kv-cache-dtype fp8_e4m3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"use_local_argmax_reduction":true}' \
  --default-chat-template-kwargs '{"reasoning_effort":"no_think"}' \
  --reasoning-parser hy_v4 \
  --enable-auto-tool-choice \
  --tool-call-parser hy_v4 \
  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_producer","kv_connector_extra_config":{"mooncake_protocol":"rdma","num_workers":80}}' \
  --prefill-context-parallel-size 16 \
  --all2all-backend deepep_high_throughput \
  --hf-overrides '{"head_dtype":"float32"}' \
  --gpu-memory-utilization 0.90 \
  --max-num-batched-tokens 98304 \
  --enforce-eager \
  --max-model-len 88000 \
  --no-async-scheduling \
  --renderer-num-workers 4 \
  --long-prefill-token-threshold 65536 \
  --enable-eplb \
  --eplb-config '{"expert_map_path":"docs/model-deployment/vllm/configs/hy4/r16.json","num_redundant_experts":16,"static_dispatch_policy":"nearest"}'
```

#### D node

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_USE_MODELSCOPE=1
export VLLM_HOST_IP="D_NODE_IP"
export VLLM_MOONCAKE_BOOTSTRAP_PORT=18999
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export ROCSHMEM_IPC_MNVL=1
export VLLM_DEEPEP_BUFFER_SIZE_MB=0

export GPU_MAX_HW_QUEUES=3
export VLLM_HCU_USE_FUSED_RMS_QUANT=1
export VLLM_HCU_ENABLE_HYV4_ATTN_INPUT_OVERLAP=1
export VLLM_HCU_ENABLE_BYTELEVEL_DETOKENIZER_TAIL=1
export TOKENIZERS_PARALLELISM=false

vllm serve hygon/Hy4-preview-Channel-INT4-w4a8 \
  --trust-remote-code \
  --quantization compressed-tensors \
  --host 0.0.0.0 \
  --tensor-parallel-size 1 \
  --enable-expert-parallel \
  --moe-backend deep_gemm \
  --enable-prefix-caching \
  --kv-cache-dtype fp8_e4m3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"use_local_argmax_reduction":true}' \
  --default-chat-template-kwargs '{"reasoning_effort":"no_think"}' \
  --reasoning-parser hy_v4 \
  --enable-auto-tool-choice \
  --tool-call-parser hy_v4 \
  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"mooncake_protocol":"rdma"}}' \
  --data-parallel-size 16 \
  --data-parallel-size-local 16 \
  --data-parallel-start-rank 0 \
  --data-parallel-address 127.0.0.1 \
  --data-parallel-rpc-port 28927 \
  --data-parallel-hybrid-lb \
  --api-server-count 1 \
  --all2all-backend deepep_low_latency \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 32 \
  --max-model-len 89088
```

#### Proxy

两端服务就绪后，在 D 节点运行 [text_proxy.py](configs/hy4/text_proxy.py)：

```bash
python docs/model-deployment/vllm/configs/hy4/text_proxy.py \
  --host 0.0.0.0 \
  --port 8002 \
  --prefill http://P_NODE_IP:8000 18998 \
  --decode http://D_NODE_IP:8000
```

#### 精度测试

服务启动一次，先测 HumanEval，再执行下方两轮性能测试。

```bash
python -m pip install evalscope==1.9.1
evalscope eval \
  --model hygon/Hy4-preview-Channel-INT4-w4a8 \
  --eval-type openai_api \
  --api-url "http://D_NODE_IP:8002/v1" \
  --api-key EMPTY \
  --datasets humaneval \
  --repeats 1 \
  --eval-batch-size 16 \
  --seed 0 \
  --generation-config '{"temperature":0,"do_sample":false,"seed":0,"max_tokens":2048,"stream":true,"timeout":600,"extra_body":{"chat_template_kwargs":{"reasoning_effort":"no_think"}}}' \
  --work-dir results/hy4-humaneval \
  --no-timestamp
```

#### 性能测试

输入 85K、输出 1200，RPS 2.55。依次运行 146 和 2920。

```bash
export VLLM_USE_MODELSCOPE=1

vllm bench serve \
  --backend openai \
  --endpoint /v1/completions \
  --model hygon/Hy4-preview-Channel-INT4-w4a8 \
  --dataset-name prefix_repetition \
  --prefix-repetition-num-prefixes 1 \
  --prefix-repetition-prefix-len 79050 \
  --prefix-repetition-suffix-len 5950 \
  --prefix-repetition-output-len 1200 \
  --request-rate 2.55 \
  --burstiness inf \
  --temperature 0 \
  --disable-shuffle \
  --num-warmups 0 \
  --ready-check-timeout-sec 0 \
  --metric-percentiles 95,99 \
  --host "D_NODE_IP" \
  --port 8002 \
  --num-prompts 146 \
  --max-concurrency 146 \
  --ignore-eos \
  --trust-remote-code \
  --seed 0 \
  --save-result \
  --save-detailed \
  --result-dir results/hy4-146 \
  --result-filename result.json
```

```bash
export VLLM_USE_MODELSCOPE=1

vllm bench serve \
  --backend openai \
  --endpoint /v1/completions \
  --model hygon/Hy4-preview-Channel-INT4-w4a8 \
  --dataset-name prefix_repetition \
  --prefix-repetition-num-prefixes 1 \
  --prefix-repetition-prefix-len 79050 \
  --prefix-repetition-suffix-len 5950 \
  --prefix-repetition-output-len 1200 \
  --request-rate 2.55 \
  --burstiness inf \
  --temperature 0 \
  --disable-shuffle \
  --num-warmups 0 \
  --ready-check-timeout-sec 0 \
  --metric-percentiles 95,99 \
  --host "D_NODE_IP" \
  --port 8002 \
  --num-prompts 2920 \
  --max-concurrency 146 \
  --ignore-eos \
  --trust-remote-code \
  --seed 0 \
  --save-result \
  --save-detailed \
  --result-dir results/hy4-2920 \
  --result-filename result.json
```

## API 调用

### IFB

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "hygon/Hy4-preview-Channel-FP8-w8a8", "messages": [{"role": "user", "content": "中国的首都是哪里？"}], "max_tokens": 128}'
```

### PD 分离

```bash
curl http://D_NODE_IP:8002/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"hygon/Hy4-preview-Channel-INT4-w4a8","messages":[{"role":"user","content":"中国的首都是哪里？"}],"max_tokens":128,"temperature":0,"stream":true,"chat_template_kwargs":{"reasoning_effort":"no_think"}}'
```
