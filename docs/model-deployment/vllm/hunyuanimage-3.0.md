# HunyuanImage-3.0 on vLLM-Omni

## 模型简介

HunyuanImage-3.0 是腾讯混元推出的原生多模态文生图（Text-to-Image）模型，采用 MoE 架构，总参数量 80B（每 token 激活 13B），在统一的自回归框架内同时支持图像理解与图像生成。vLLM-Omni 为其提供 DiT 单阶段部署形态，可进行离线推理与在线服务部署，并提供 OpenAI 兼容的 `POST /v1/images/generations` 接口。

## 模型列表

| 模型权重 | 量化方式 | vLLM-Omni 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | --------- | -------- | ---- | -------- | -------- |
| [Tencent-Hunyuan/HunyuanImage-3.0](https://www.modelscope.cn/models/Tencent-Hunyuan/HunyuanImage-3.0) | BF16 | 0.21 | BW1100 | 4 | IFB | [**`>_`**](#hunyuanimage-30-ifb-bw1100-4x-vllm-021) |

## 启动命令

### HunyuanImage-3.0 IFB BW1100 4x vLLM 0.21

本章节的离线推理与在线推理均使用 DiT 单阶段部署配置 `hunyuan_image3_dit.yaml`（tensor_parallel_size=4，共 4 卡），完整内容见 [YAML 配置](#yaml-配置)，请先将其保存到启动命令的工作目录。

默认使用 YAML 中 `devices: "0,1,2,3"` 指定的前 4 张卡。如需使用其它物理卡（例如后 4 张卡），在启动前设置 `export HIP_VISIBLE_DEVICES="4,5,6,7"`，此时 YAML 中的 `devices` 为可见设备的逻辑编号，无需修改。运行前建议用 `hy-smi` 确认目标卡处于空闲状态。

#### 离线推理(Offline Inference)

离线推理使用 vllm-omni 仓库自带的 HunyuanImage-3 示例脚本：

```bash
cd vllm-omni/examples/offline_inference/hunyuan_image3

export VLLM_HCU_USE_CUSTOM_OPS=1
export VLLM_ROCM_USE_AITER_MOE=1
export VLLM_ROCM_USE_AITER=1

python end2end.py \
  --model Tencent-Hunyuan/HunyuanImage-3.0 \
  --modality text2img \
  --deploy-config ./hunyuan_image3_dit.yaml \
  --height 1024 --width 1024 \
  --prompts "A cinematic portrait of an astronaut in a greenhouse"
```

生成的图片默认保存到当前目录（可用 `--output` 指定输出目录），文件名形如 `output_0_0.png`。常用可选参数：`--steps`（去噪步数，默认 50）、`--guidance-scale`（CFG 系数，默认 5.0）、`--seed`（随机种子，默认 42）。

#### 在线推理(Online Inference)

##### 启动 vllm server

```bash
export VLLM_HCU_USE_CUSTOM_OPS=1
export VLLM_ROCM_USE_AITER_MOE=1
export VLLM_ROCM_USE_AITER=1

vllm serve Tencent-Hunyuan/HunyuanImage-3.0 \
  --omni \
  --port 8098 \
  --deploy-config ./hunyuan_image3_dit.yaml \
  --trust-remote-code
```

服务启动完成后可检查：

```bash
curl http://localhost:8098/health
```

##### API Calls

通过 OpenAI 兼容的图像生成接口发起文生图请求。接口为同步接口，响应 JSON 的 `data[0].b64_json` 字段中包含 base64 编码的 PNG 图片，下面命令在收到响应后直接解码保存为 `hunyuanimage3_output.png`：

```bash
curl -s -X POST http://localhost:8098/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A cute cat sitting on a windowsill watching the sunset",
    "size": "1024x1024",
    "num_inference_steps": 50,
    "guidance_scale": 5.0,
    "seed": 42
  }' \
  | python3 -c "import sys, json, base64; sys.stdout.buffer.write(base64.b64decode(json.load(sys.stdin)['data'][0]['b64_json']))" \
  > hunyuanimage3_output.png
```

## YAML 配置

将以下内容保存为 `hunyuan_image3_dit.yaml`：

```yaml
# HunyuanImage-3.0 DiT-only deploy.
pipeline: hunyuan_image3_dit
async_chunk: false

stages:
  - stage_id: 0
    max_num_seqs: 1
    gpu_memory_utilization: 0.9
    enforce_eager: true
    devices: "0,1,2,3"
    vae_use_slicing: false
    vae_use_tiling: false
    cache_backend:
    cache_config:
    enable_cache_dit_summary: false
    parallel_config:
      pipeline_parallel_size: 1
      data_parallel_size: 1
      tensor_parallel_size: 4
      enable_expert_parallel: false
      sequence_parallel_size: 1
      ulysses_degree: 1
      ring_degree: 1
      cfg_parallel_size: 1
      vae_patch_parallel_size: 1
      use_hsdp: false
      hsdp_shard_size: -1
      hsdp_replicate_size: 1
    default_sampling_params:
      seed: 42
```

## 注意事项

- 本文档覆盖 DiT 单阶段（文生图）部署形态；请求中的 `prompt` 直接作为 DiT 阶段的生成条件。
- `/v1/images/generations` 为同步接口，响应 JSON 中图片在 `data[0].b64_json` 字段以 base64 PNG 编码返回；上文 API Calls 命令已通过管道直接解码保存为图片文件。
- 模型权重约 158GB（BF16），4 卡 TP=4 部署时单卡显存占用约 90%（`gpu_memory_utilization: 0.9`），请勿与其它服务共用同一组 HCU。
