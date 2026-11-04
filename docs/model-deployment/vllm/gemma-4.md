# Gemma-4 on vLLM

## 模型简介

Gemma-4-31B-it 是 Gemma 系列指令模型，本文档提供其在 vLLM 上的部署示例。

## 模型列表

| 模型权重 | 量化方式 | vLLM 镜像 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | --------- | -------- | ---- | -------- | -------- |
| [google/gemma-4-31B-it](https://www.modelscope.cn/models/google/gemma-4-31B-it) | BF16 | 0.28.1 | BW1000 | 4 | IFB | [**`>_`**](#gemma-4-31b-it-ifb-bw1000-2x-vllm-0281) |
|  | BF16 | 0.25 | BW1000 | 2 | IFB | [**`>_`**](#gemma-4-31b-it-ifb-bw1000-2x-vllm-025) |
|  | BF16 | [0.21](../docker_images.md) | BW1000 | 1 | IFB | [**`>_`**](#gemma-4-31b-it-ifb-bw1000-1x-vllm-021) |
[google/gemma-4-31B-it-assistant](https://www.modelscope.cn/models/google/gemma-4-31B-it-assistant) | BF16 | - | - | - | - | -

## 启动命令

### gemma-4-31B-it IFB BW1000 2x vLLM 0.28.1

```bash

vllm serve google/gemma-4-31B-it \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --attention-backend TRITON_ATTN \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --reasoning-parser gemma4 \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
```
### gemma-4-31B-it IFB BW1000 2x vLLM 0.28.1 + MTP
参照此[PR](https://github.com/HYGON-AI/vllm-plugin-das/pull/221/)修改vllm_dcu
```bash
export VLLM_HCU_GEMMA4_USE_FA=1

vllm serve google/gemma-4-31B-it \
  --served-model-name gemma4 \
  --tensor-parallel-size 2 \
  --trust-remote-code \
  --max-model-len 19968 \
  --data-parallel-size 1 \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --reasoning-parser gemma4 \
  --block-size 32 \
  --compilation-config '{"cudagraph_capture_sizes": [1,2,4,8,16,24,32,40,48,56,64,80,96,112,120,125,128]}' \
  --speculative-config '{"method":"mtp","model":"{PATH-TO-ASSISTANT-WEIGHT}","num_speculative_tokens":4}'
```

### Gemma-4-31B-it IFB BW1000 2x vLLM 0.25

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_KV_CACHE_LAYOUT=HND

vllm serve \
  --model google/gemma-4-31B-it \
  --tensor-parallel-size 2 \
  --max-model-len 32768 \
  --attention-backend TRITON_ATTN \
  --hf-overrides '{"text_config":{"allow_global_per_layer_attribute_access":true,"global_head_dim":512,"num_global_key_value_heads":4,"use_bidirectional_attention":null}}' \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --reasoning-parser gemma4
```

### Gemma-4-31B-it IFB BW1000 1x vLLM 0.21

```bash
export VLLM_HCU_USE_CUSTOM_OPS=0

vllm serve google/gemma-4-31B-it \
    -tp 2 \
    --enable-auto-tool-choice \
    --tool-call-parser gemma4 \
    --reasoning-parser gemma4 \
    --chat-template tool_chat_template_gemma4.jinja \
    --attention-backend TRITON_ATTN
```

## API 调用

### IFB

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "google/gemma-4-31B-it",
        "messages": [{"role": "user", "content": "中国的首都是什么？"}],
        "temperature": 0,
        "max_tokens": 400
    }'
```
