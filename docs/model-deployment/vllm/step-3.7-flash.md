# Step-3.7-Flash on vLLM

## 模型简介

Step-3.7-Flash 是 Step 系列大语言模型，本文档提供其在 vLLM 上的部署示例。

## 模型列表

| 模型权重 | 量化方式 | vLLM 镜像 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | --------- | -------- | ---- | -------- | -------- |
| [stepfun-ai/Step-3.7-Flash](https://www.modelscope.cn/models/stepfun-ai/Step-3.7-Flash) | BF16 | 0.28.1 | BW1100 | 8 | IFB | [**`>_`**](#step-37-flash-ifb-bw1100-8x-vllm-0281) |
|  | BF16 | 0.28.1 | BW1000 | 8 | IFB | [**`>_`**](#step-37-flash-ifb-bw1000-8x-vllm-0281) |
| [hygon/Step-3.7-Flash-FP8-Channel](https://www.modelscope.cn/models/hygon/Step-3.7-Flash-FP8-Channel) | FP8 Channel | 0.28.1 | BW1100 | 8 | IFB | [**`>_`**](#step-37-flash-fp8-channel-ifb-bw1100-8x-vllm-0281) |
|  | FP8 Channel | [0.21](../docker_images.md) | BW1100 | 8 | IFB | [**`>_`**](#step-37-flash-fp8-channel-ifb-bw1100-8x-vllm-021) |

## 启动命令

### Step-3.7-Flash IFB BW1100 8x vLLM 0.28.1

```bash
export VLLM_KV_CACHE_LAYOUT=HND

vllm serve stepfun-ai/Step-3.7-Flash \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --attention-backend FLASH_ATTN \
  --moe-backend triton \
  --kv-cache-dtype fp8_e4m3 \
  --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --enable-auto-tool-choice \
  --tool-call-parser step3p5 \
  --reasoning-parser step3p5 \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
```

### Step-3.7-Flash IFB BW1000 8x vLLM 0.28.1

```bash
export VLLM_KV_CACHE_LAYOUT=HND

vllm serve stepfun-ai/Step-3.7-Flash \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --attention-backend FLASH_ATTN \
  --moe-backend triton \
  --kv-cache-dtype fp8_e5m2 \
  --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --enable-auto-tool-choice \
  --tool-call-parser step3p5 \
  --reasoning-parser step3p5 \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
```

### Step-3.7-Flash-FP8-Channel IFB BW1100 8x vLLM 0.28.1

```bash
export VLLM_KV_CACHE_LAYOUT=HND
export GPU_MAX_HW_QUEUES=4

vllm serve hygon/Step-3.7-Flash-FP8-Channel \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --quantization slimquant_marlin \
  --attention-backend FLASH_ATTN \
  --moe-backend triton \
  --kv-cache-dtype fp8_e4m3 \
  --no-enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"enable_multi_layers_mtp":true,"quantization":"slimquant_marlin"}' \
  --enable-chunked-prefill \
  --reasoning-parser step3p5 \
  --limit-mm-per-prompt '{"image": 0}' \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
```


### Step-3.7-Flash-FP8-Channel IFB BW1100 8x vLLM 0.21

```bash
export GPU_MAX_HW_QUEUES=4

vllm serve hygon/Step-3.7-Flash-FP8-Channel \
  --tensor-parallel-size 8 \
  --limit-mm-per-prompt '{"image": 0}' \
  --trust-remote-code \
  -q slimquant_marlin \
  --no-enable-prefix-caching \
  --enable-chunked-prefill \
  --max-num-batched-tokens 65536 \
  --kv-cache-dtype fp8_e4m3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"enable_multi_layers_mtp":true,"quantization":"slimquant_marlin"}' \
  --gpu-memory-utilization 0.9 \
  --attention-backend FLASH_ATTN_CUTLASS \
  --reasoning-parser step3p5
```

## API 调用

### IFB

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "hygon/Step-3.7-Flash-FP8-Channel",
        "messages": [{"role": "user", "content": "中国的首都是什么？"}],
        "temperature": 0,
        "max_tokens": 400
    }'
```
