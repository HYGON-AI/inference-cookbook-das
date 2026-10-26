# Step-5-Preview on vLLM

## 模型简介

本文提供 [Step-5-Preview-Channel-FP8-w8a8](https://www.modelscope.cn/models/hygon/Step-5-Preview-Channel-FP8-w8a8) 在 BW1100 八卡上的 vLLM 文本推理部署示例，采用 Channel FP8 W8A8 权重、FP8 KV cache 和 MTP3。

## 模型列表

| 模型权重 | 量化方式 | vLLM 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | --------- | -------- | ---- | -------- | -------- |
| [hygon/Step-5-Preview-Channel-FP8-w8a8](https://www.modelscope.cn/models/hygon/Step-5-Preview-Channel-FP8-w8a8) | FP8 W8A8（Channel） | [0.28.1](../docker_images.md) | BW1100 144GB | 8 | IFB | [**`>_`**](#step-5-preview-channel-fp8-w8a8-ifb-bw1100-8x-vllm-0281) |

## 启动命令

### Step-5-Preview-Channel-FP8-w8a8 IFB BW1100 8x vLLM 0.28.1

单机使用 8 张 BW1100，配置 TP4、DP2、EP8，采用 Triton MoE、eager 执行和正常 MTP3。上下文上限为 65,536 token。

```bash
export VLLM_USE_MODELSCOPE=1
export NCCL_IB_DISABLE=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export GPU_MAX_HW_QUEUES=4
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_FLASH_ATTN_UNIFIED=1
export VLLM_USE_PD_SPLIT=1
export VLLM_HCU_USE_LIGHTOP_EP_SCATTER=0
export VLLM_ROCM_USE_AITER_MOE=1
export VLLM_USE_FLASH_ATTN_FP8=1
export VLLM_HCU_USE_LIGHTOP_MOE_ALIGN=1
export FA_DECODE_NUM_SPLITS=2
export VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=0
export VLLM_HCU_EP_FUSED_COMBINE=1
export VLLM_HCU_DP_CUSTOM_AG=1
export VLLM_HCU_STEP5_SWA_FUSED_QK=1
export VLLM_HCU_STEP5_FUSED_GATE=1
export VLLM_HCU_TRITON_TOKEN_FP8_QUANT=1
export VLLM_HCU_PIN_MEMORY=1
export HIP_FORCE_DEV_KERNARG=1

vllm serve hygon/Step-5-Preview-Channel-FP8-w8a8 \
  --served-model-name step5 \
  --reasoning-parser step3p5 \
  --tokenizer hygon/Step-5-Preview-Channel-FP8-w8a8 \
  --tensor-parallel-size 4 \
  --data-parallel-size 2 \
  --moe-backend triton \
  --enable-expert-parallel \
  --max-num-batched-tokens 32768 \
  --max-model-len 65536 \
  --max-num-seqs 64 \
  --no-enable-prefix-caching \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 3, "use_local_argmax_reduction": true}' \
  --enforce-eager \
  --compilation-config '{"mode":0,"cudagraph_mode":"NONE","custom_ops":["+swiglustep_and_mul"]}' \
  --kv-cache-dtype fp8_e4m3 \
  --limit-mm-per-prompt '{"image": 0}' \
  --gpu-memory-utilization 0.85
```

## API 调用

### IFB

服务默认监听 8000 端口，API 模型名与 `--served-model-name step5` 一致。模型加载完成后检查服务：

```bash
curl --fail http://localhost:8000/v1/models
```

```bash
curl --fail http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "step5",
    "messages": [{"role": "user", "content": "中国的首都是什么？"}],
    "temperature": 0,
    "max_tokens": 2048
  }'
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="step5",
    messages=[{"role": "user", "content": "中国的首都是什么？"}],
    temperature=0,
    max_tokens=2048,
)
print(response.choices[0].message.content)
```
