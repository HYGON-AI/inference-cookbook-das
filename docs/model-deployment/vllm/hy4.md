# Hy4 on vLLM

## 模型简介

Hy4-preview 是 Hy 系列模型，本文档提供其在 vLLM 上的部署示例。

## 模型列表

| 模型权重 | 量化方式 | vLLM 镜像 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | --------- | -------- | ---- | -------- | -------- |
| [hygon/Hy4-preview-Channel-FP8-w8a8](https://www.modelscope.cn/models/hygon/Hy4-preview-Channel-FP8-w8a8) | FP8 W8A8 | 0.28.1 | BW1100 | 8 | IFB | [**`>_`**](#hy4-preview-channel-fp8-w8a8-ifb-bw1100-8x-vllm-0281) |
|  | FP8 W8A8 | 0.25.1 | BW1100 | 8 | IFB | [**`>_`**](#hy4-preview-channel-fp8-w8a8-ifb-bw1100-8x-vllm-0251) |
| [hygon/Hy4-preview-Channel-INT4-w4a8](https://www.modelscope.cn/models/hygon/Hy4-preview-Channel-INT4-w4a8) | INT4 W4A8 | 0.25.1 | BW1100(超节点) | 32 | 1P1D | [**`>_`**](#hy4-preview-channel-int4-w4a8-1p1d-bw1100-32x-vllm-0251) |


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

P、D 各使用一个 16 卡 BW1100 超节点，离线 EPLB 冗余专家数为 16。

镜像：`42.228.13.241:5000/jenkins/model_test_env/vllm:0.25.1-ubuntu22.04-dtk26041-py3.10-20261112-0837`。

离线映射使用 [r16.json](configs/hy4/r16.json)，P 启动时直接加载。

#### 设置环境变量

在两台宿主机的本仓库根目录分别执行，替换节点地址及各自的模型目录绝对路径：

```bash
export P_NODE_IP="<P_IP>"
export D_NODE_IP="<D_IP>"
export MODEL_DIR="<MODEL_DIR>"
```

镜像拉取失败时，在对应宿主机的 `/etc/docker/daemon.json` 中，将 `42.228.13.241:5000` 追加到 `insecure-registries` 数组，保留已有配置：

```json
{
  "insecure-registries": ["42.228.13.241:5000"]
}
```

修改后、创建容器前重启 Docker；重启会影响该宿主机已有容器。

```bash
sudo systemctl restart docker
```

#### P 机器：创建容器

在 P 宿主机执行，最后一条命令进入 P 容器：

```bash
docker pull 42.228.13.241:5000/jenkins/model_test_env/vllm:0.25.1-ubuntu22.04-dtk26041-py3.10-20261112-0837
docker run -d \
  --name hy4-p \
  --hostname hy4-p \
  --network host \
  --shm-size 64g \
  --cap-add IPC_LOCK \
  --group-add video \
  --security-opt seccomp=unconfined \
  --ulimit memlock=-1:-1 \
  --ulimit nofile=1048576:1048576 \
  --device /dev/kfd \
  --device /dev/mkfd \
  --device /dev/dri \
  --device /dev/infiniband \
  -v /usr/local/hyhal:/usr/local/hyhal:ro \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v "$PWD":/workspace \
  -v "$MODEL_DIR":/models/hy4:ro \
  -e P_NODE_IP="$P_NODE_IP" \
  -e D_NODE_IP="$D_NODE_IP" \
  -e VLLM_HOST_IP="$P_NODE_IP" \
  --workdir /workspace \
  --entrypoint /usr/bin/sleep \
  42.228.13.241:5000/jenkins/model_test_env/vllm:0.25.1-ubuntu22.04-dtk26041-py3.10-20261112-0837 infinity
docker exec -it hy4-p bash
```

#### D 机器：创建容器

在 D 宿主机执行，最后一条命令进入 D 容器：

```bash
docker pull 42.228.13.241:5000/jenkins/model_test_env/vllm:0.25.1-ubuntu22.04-dtk26041-py3.10-20261112-0837
docker run -d \
  --name hy4-d \
  --hostname hy4-d \
  --network host \
  --shm-size 64g \
  --cap-add IPC_LOCK \
  --group-add video \
  --security-opt seccomp=unconfined \
  --ulimit memlock=-1:-1 \
  --ulimit nofile=1048576:1048576 \
  --device /dev/kfd \
  --device /dev/mkfd \
  --device /dev/dri \
  --device /dev/infiniband \
  -v /usr/local/hyhal:/usr/local/hyhal:ro \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v "$PWD":/workspace \
  -v "$MODEL_DIR":/models/hy4:ro \
  -e P_NODE_IP="$P_NODE_IP" \
  -e D_NODE_IP="$D_NODE_IP" \
  -e VLLM_HOST_IP="$D_NODE_IP" \
  --workdir /workspace \
  --entrypoint /usr/bin/sleep \
  42.228.13.241:5000/jenkins/model_test_env/vllm:0.25.1-ubuntu22.04-dtk26041-py3.10-20261112-0837 infinity
docker exec -it hy4-d bash
```

#### P node

在 P 容器执行：

```bash
cp docs/model-deployment/vllm/configs/hy4/r16.json r16.json
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_HOST_IP="$P_NODE_IP"
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

vllm serve /models/hy4 \
  --served-model-name hygon/Hy4-preview-Channel-INT4-w4a8 \
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
  --eplb-config '{"expert_map_path":"r16.json","num_redundant_experts":16,"static_dispatch_policy":"nearest"}'
```

#### D node

在 D 容器执行：

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_HOST_IP="$D_NODE_IP"
export VLLM_MOONCAKE_BOOTSTRAP_PORT=18999
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export ROCSHMEM_IPC_MNVL=1
export VLLM_DEEPEP_BUFFER_SIZE_MB=0
export GPU_MAX_HW_QUEUES=3
export VLLM_HCU_USE_FUSED_RMS_QUANT=1
export VLLM_HCU_ENABLE_HYV4_ATTN_INPUT_OVERLAP=1
export VLLM_HCU_ENABLE_BYTELEVEL_DETOKENIZER_TAIL=1
export TOKENIZERS_PARALLELISM=false

vllm serve /models/hy4 \
  --served-model-name hygon/Hy4-preview-Channel-INT4-w4a8 \
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

两端服务就绪后，在 D 宿主机另开终端，进入 D 容器运行 [proxy.py](configs/hy4/proxy.py)：

```bash
docker exec -it hy4-d bash
```

```bash
cp docs/model-deployment/vllm/configs/hy4/proxy.py proxy.py
python proxy.py \
  --host 0.0.0.0 \
  --port 8002 \
  --prefill "http://${P_NODE_IP}:8000" 18998 \
  --decode "http://${D_NODE_IP}:8000"
```

#### 精度测试

在 D 宿主机另开终端进入 D 容器，先测 HumanEval，再执行下方两轮性能测试：

```bash
docker exec -it hy4-d bash
```

```bash
python -m pip install evalscope==1.9.1
evalscope eval \
  --model hygon/Hy4-preview-Channel-INT4-w4a8 \
  --eval-type openai_api \
  --api-url "http://${D_NODE_IP}:8002/v1" \
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

输入 85K、输出 1200，RPS 2.55。

##### 预热：146 请求

```bash
vllm bench serve \
  --backend openai \
  --endpoint /v1/completions \
  --model hygon/Hy4-preview-Channel-INT4-w4a8 \
  --tokenizer /models/hy4 \
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
  --host "$D_NODE_IP" \
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

##### 实跑：2920 请求

```bash
vllm bench serve \
  --backend openai \
  --endpoint /v1/completions \
  --model hygon/Hy4-preview-Channel-INT4-w4a8 \
  --tokenizer /models/hy4 \
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
  --host "$D_NODE_IP" \
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
