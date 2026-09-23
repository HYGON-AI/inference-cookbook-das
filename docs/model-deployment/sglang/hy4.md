# hy4

# Hy4

## 模型列表

|模型权重|量化方式|SGLang 镜像|推荐硬件|卡数|部署方式|启动命令|
|---|---|---|---|---|---|---|
|[hygon/Hy4\-preview\-Channel\-FP8\-w8a8](https://www.modelscope.cn/models/hygon/Hy4-preview-Channel-FP8-w8a8)|FP8 W8A8|0\.5\.18|BW1100|8|IFB \(tp8dp8\)|**`>_`**|

## 启动命令

### Hy4\-preview\-Channel\-FP8\-w8a8 IFB BW1100 8x SGLang 0\.5\.18 \(tp8dp8\)

```Bash
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export SGLANG_USE_FP8_W8A8_MOE=1
export SGLANG_USE_DEEPGEMM_MOE=1
export SGLANG_OPT_USE_TOPK_V2=false
export SGLANG_ROCM_USE_AITER_MOE=1
export SGLANG_OPT_HY4_IHC_TILELANG=1
# export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/amdsmi:$LD_LIBRARY_PATH #按照实际需求

python -m sglang.launch_server \
  --model-path hygon/Hy4-preview-Channel-FP8-w8a8 \
  --trust-remote-code \
  --dtype bfloat16 \
  --tp-size 8 \
  --dp-size 8 \
  --enable-dp-attention \
  --enable-dp-lm-head \
  --moe-a2a-backend deepep \
  --deepep-mode auto \
  --moe-dense-tp-size=1 \
  --disable-shared-experts-fusion \
  --attention-backend dsa \
  --dsa-prefill-backend flashmla_sparse \
  --dsa-decode-backend flashmla_kv \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 2 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 3 \
  --page-size 64 \
  --chunked-prefill-size 8192 \
  --context-length 65536 \
  --max-running-requests 64 \
  --numa-node 0 0 0 0 1 1 1 1 \
  --kv-cache-dtype fp8_e5m2 \
  --disable-radix-cache \
  --disable-cuda-graph \
  --mem-fraction-static 0.95 \
  --host 0.0.0.0 \
  --port 30000
```

## API 调用

### IFB

```Python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="hygon/Hy4-preview-Channel-FP8-w8a8",
    messages=[{"role": "user", "content": "中国的首都是哪里？"}],
    max_tokens=128,
)
```

```Bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "hygon/Hy4-preview-Channel-FP8-w8a8", "messages": [{"role": "user", "content": "中国的首都是哪里？"}], "max_tokens": 128}'
```
