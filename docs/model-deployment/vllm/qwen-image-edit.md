# Qwen-Image-Edit-2511 on vLLM-Omni

本文提供 Qwen-Image-Edit-2511 在 BW1000 和 BW1100 上的单卡部署，以及单图、双图编辑 API 示例。使用原始 BF16 权重，不加载外置 LoRA。

## 模型列表

| 模型权重 | 量化方式 | vLLM 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| --- | --- | --- | --- | --- | --- | --- |
| [Qwen/Qwen-Image-Edit-2511](https://www.modelscope.cn/models/Qwen/Qwen-Image-Edit-2511) | BF16 | 0.21.0 | BW1100 | 1 | Online | [**`>_`**](#qwen-image-edit-2511-online-1x-vllm-omni-0210) |
|  | BF16 | 0.21.0 | BW1000 | 1 | Online | [**`>_`**](#qwen-image-edit-2511-online-1x-vllm-omni-0210) |

## 模型与场景

| 场景 | Pipeline | 输入 | 输出 |
| --- | --- | --- | --- |
| 单图编辑 | `QwenImageEditPlusPipeline` | 文本 Prompt + 一张参考图片 | 一张 PNG 图片 |
| 双图编辑 | `QwenImageEditPlusPipeline` | 文本 Prompt + 两张参考图片 | 一张 PNG 图片 |

## 启动命令

### Qwen-Image-Edit-2511 Online 1x vLLM-Omni 0.21.0

```bash
export DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN

vllm serve Qwen/Qwen-Image-Edit-2511 \
  --omni \
  --host 127.0.0.1 \
  --tensor-parallel-size 1 \
  --dtype bfloat16
```

## 服务检查

```bash
curl --noproxy '*' -fsS -o /dev/null -w 'HTTP %{http_code}\n' \
  http://127.0.0.1:8000/health
curl --noproxy '*' -fsS http://127.0.0.1:8000/v1/models | python -m json.tool
```

## API 调用

替换图片路径和 `prompt` 中的编辑指令后发送请求。

### 单图编辑

```bash
curl --noproxy '*' --fail-with-body -sS \
  http://127.0.0.1:8000/v1/images/edits \
  -F 'image=@/path/to/input.jpg' \
  --form-string 'model=Qwen/Qwen-Image-Edit-2511' \
  --form-string 'prompt=请填写编辑指令' \
  --form-string 'negative_prompt= ' \
  --form-string 'size=1024x768' \
  --form-string 'n=1' \
  --form-string 'num_inference_steps=40' \
  --form-string 'true_cfg_scale=4.0' \
  --form-string 'guidance_scale=1.0' \
  --form-string 'seed=0' \
  --form-string 'output_format=png' \
  --form-string 'response_format=b64_json' \
  -o single_image_response.json
```

### 双图编辑

```bash
curl --noproxy '*' --fail-with-body -sS \
  http://127.0.0.1:8000/v1/images/edits \
  -F 'image=@/path/to/input1.jpg' \
  -F 'image=@/path/to/input2.jpg' \
  --form-string 'model=Qwen/Qwen-Image-Edit-2511' \
  --form-string 'prompt=请填写编辑指令' \
  --form-string 'negative_prompt= ' \
  --form-string 'size=1024x768' \
  --form-string 'n=1' \
  --form-string 'num_inference_steps=40' \
  --form-string 'true_cfg_scale=4.0' \
  --form-string 'guidance_scale=1.0' \
  --form-string 'seed=0' \
  --form-string 'output_format=png' \
  --form-string 'response_format=b64_json' \
  -o two_image_response.json
```

## 结果查询

接口同步返回结果，无需查询任务 ID。单图和双图响应分别保存在 `single_image_response.json`、`two_image_response.json`，图片数据位于 `data[].b64_json`。

```bash
python3 - <<'PY'
import json
from pathlib import Path

for name in ("single_image_response.json", "two_image_response.json"):
    path = Path(name)
    if path.exists():
        result = json.loads(path.read_text())
        images = result["data"]
        print(name, "图片数:", len(images),
              "非空 Base64 数:", sum(bool(item.get("b64_json")) for item in images))
PY
```

## 日志检查

启动和请求完成后，确认日志中包含以下关键内容：

```text
Resolved diffusion attention backend 'FLASH_ATTN' for role='self'
Application startup complete.
40/40
"POST /v1/images/edits HTTP/1.1" 200 OK
```

## 可选：开启 SLA

需使用支持 SLA 的 HCU FlashAttention 包。启动服务前设置以下变量，再执行上面的启动命令：

```bash
export ENABLE_SPARSE_ATTN=1
export SPARSE_ATTN_TOPK=0.2
export SPARSE_ATTN_FEATURE_MAP=softmax
export SPARSE_ATTN_USE_BF16=true
export SPARSE_ATTN_USE_FP8=false
```

SLA 为近似稀疏注意力，可能影响生成效果。关闭时将 `ENABLE_SPARSE_ATTN` 改为 `0` 并重启服务。
