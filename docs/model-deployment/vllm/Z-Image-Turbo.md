# Z-Image

## 模型简介

Z-Image-Turbo 面向文生图场景，支持 vLLM-Omni 部署。

## 模型列表


| 模型权重                                                                                  | 量化方式 | vLLM 版本 | 推荐硬件   | 卡数  | 部署方式 | 启动命令 |
| ------------------------------------------------------------------------------------- | ---- | ------- | ------ | --- | ---- | ---- |
| [Tongyi-MAI/Z-Image-Turbo](https://www.modelscope.cn/models/Tongyi-MAI/Z-Image-Turbo) | BF16 | 0.21    | BW1100 | 1   | IFB  | [**`>_`**](#z-image-turbo-ifb-bw1100-1x-vllm-021) |


## 启动命令

### Z-Image-Turbo IFB BW1100 1x vLLM 0.21

#### 离线推理(Offline Inference)

```bash
cd vllm-omni/examples/offline_inference/text_to_image

python text_to_image.py \
  --model Tongyi-MAI/Z-Image-Turbo \
  --prompt "a cup of coffee on the table" \
  --seed 42 \
  --guidance-scale 0.0 \
  --num-images-per-prompt 1 \
  --num-inference-steps 9 \
  --height 1024 \
  --width 1024 \
  --output outputs/coffee.png
```



#### 在线推理(Online Inference)



##### 启动 vllm server

```bash
vllm serve Tongyi-MAI/Z-Image-Turbo \
  --omni \
  --port 8190
```



##### API Calls

```bash
curl -s http://127.0.0.1:8190/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Tongyi-MAI/Z-Image-Turbo",
    "prompt": "a cup of coffee on the table",
    "size": "1024x1024",
    "response_format": "b64_json"
  }' | python3 -c "import sys,json,base64; open('smoke.png','wb').write(base64.b64decode(json.load(sys.stdin)['data'][0]['b64_json']))"
```

