# Hy4 on vLLM

## 模型简介

Hy4-preview 是 Hy 系列模型，本文档提供其在 vLLM 上的部署示例。

## 模型列表

| 模型权重 | 量化方式 | vLLM 镜像 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | --------- | -------- | ---- | -------- | -------- |
| [hygon/Hy4-preview-Channel-FP8-w8a8](https://www.modelscope.cn/models/hygon/Hy4-preview-Channel-FP8-w8a8) | FP8 W8A8 | 0.28.1 | BW1100 | 8 | IFB | [**`>_`**](#hy4-preview-channel-fp8-w8a8-ifb-bw1100-8x-vllm-0281) |
|  | FP8 W8A8 | 0.25.1 | BW1100 | 8 | IFB | [**`>_`**](#hy4-preview-channel-fp8-w8a8-ifb-bw1100-8x-vllm-0251) |
| [hygon/Hy4-preview-Channel-FP8-w8a8-v2](https://www.modelscope.cn/models/hygon/Hy4-preview-Channel-FP8-w8a8-v2) | FP8 W8A8 | 0.25.1 | BW1100(超节点) | 16 | 单 P | [**`>_`**](#hy4-preview-channel-fp8-w8a8-v2-单-p-bw1100-16x-vllm-0251) |
|  | FP8 W8A8 | 0.25.1 | BW1100(超节点) | 32 | 单 D（fake prefill） | [**`>_`**](#hy4-preview-channel-fp8-w8a8-v2-单-d-bw1100-32x-vllm-0251) |
| [hygon/Hy4-preview-Channel-INT4-w4a8](https://www.modelscope.cn/models/hygon/Hy4-preview-Channel-INT4-w4a8) | INT4 W4A8 | 0.25.1 | BW1100(超节点) | 32 | 1P1D | [**`>_`**](#hy4-preview-channel-int4-w4a8-1p1d-bw1100-32x-vllm-0251) |
|  | INT4 W4A8 | 0.25.1 | BW1100(超节点) | 16 | 单 P | [**`>_`**](#hy4-preview-channel-int4-w4a8-单-p-bw1100-16x-vllm-0251) |
|  | INT4 W4A8 | 0.25.1 | BW1100(超节点) | 16 | 单 D（fake prefill） | [**`>_`**](#hy4-preview-channel-int4-w4a8-单-d-bw1100-16x-vllm-0251) |


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

### Hy4-preview-Channel-FP8-w8a8-v2 单 P BW1100 16x vLLM 0.25.1

一个 16 卡 BW1100 超节点。容器创建及插件安装见下方 [P 环境准备](#hy4-preview-channel-int4-w4a8-单-p-bw1100-16x-vllm-0251)。

#### 启动服务

```bash
export P_NODE_IP="<P_IP>"
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_USE_MODELSCOPE=1
export VLLM_HCU_USE_LIGHTOP_PER_TOKEN_QUANT_FP8=1
export VLLM_HCU_USE_LIGHTOP_HY_V4_INDEXER=0
export NCCL_SOCKET_IFNAME="<SOCKET_IFNAME>"
export GLOO_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME"
export VLLM_HOST_IP="$P_NODE_IP"
export HEAD_IP="$P_NODE_IP"
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1,mlx5_10:1,mlx5_11:1,mlx5_12:1,mlx5_13:1,mlx5_14:1,mlx5_15:1
export NCCL_IB_GID_INDEX=3
export ROCSHMEM_GDR_DISABLE_XDP=1
export DEEP_EP_NORMAL_MNVL=1
export VLLM_DEEPEP_BUFFER_SIZE_MB=4096
export GPU_MAX_HW_QUEUES=4
export VLLM_MOE_ROUTING_SIMULATION_STRATEGY=uniform_random
export VLLM_RANDOMIZE_DP_DUMMY_INPUTS=1
export LIGHTOP_MQA_LOGITS_BETA=1

vllm serve hygon/Hy4-preview-Channel-FP8-w8a8-v2 \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --prefill-context-parallel-size 16 \
  --enable-expert-parallel \
  --all2all-backend deepep_high_throughput \
  --moe-backend deep_gemm \
  --default-chat-template-kwargs '{"reasoning_effort":"no_think"}' \
  --tool-call-parser hy_v4 \
  --reasoning-parser hy_v4 \
  --hf-overrides '{"head_dtype":"float32"}' \
  --max-model-len 86000 \
  --gpu-memory-utilization 0.90 \
  --max-num-batched-tokens 32768 \
  --enable-auto-tool-choice \
  --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"use_local_argmax_reduction":true}' \
  --enforce-eager
```

#### 客户端

在 P 容器另开终端执行；两轮不重启服务。

##### 预热：146 请求

```bash
export VLLM_USE_MODELSCOPE=1

vllm bench serve \
  --backend openai \
  --endpoint /v1/completions \
  --model hygon/Hy4-preview-Channel-FP8-w8a8-v2 \
  --dataset-name prefix_repetition \
  --prefix-repetition-num-prefixes 1 \
  --prefix-repetition-prefix-len 79050 \
  --prefix-repetition-suffix-len 5950 \
  --prefix-repetition-output-len 1 \
  --request-rate 2.6 \
  --burstiness inf \
  --temperature 0 \
  --disable-shuffle \
  --num-warmups 0 \
  --ready-check-timeout-sec 0 \
  --metric-percentiles 95,99 \
  --host 127.0.0.1 \
  --port 8000 \
  --num-prompts 146 \
  --max-concurrency 146 \
  --ignore-eos \
  --trust-remote-code \
  --seed 0
```

##### 实跑：2920 请求

```bash
export VLLM_USE_MODELSCOPE=1

vllm bench serve \
  --backend openai \
  --endpoint /v1/completions \
  --model hygon/Hy4-preview-Channel-FP8-w8a8-v2 \
  --dataset-name prefix_repetition \
  --prefix-repetition-num-prefixes 1 \
  --prefix-repetition-prefix-len 79050 \
  --prefix-repetition-suffix-len 5950 \
  --prefix-repetition-output-len 1 \
  --request-rate 2.6 \
  --burstiness inf \
  --temperature 0 \
  --disable-shuffle \
  --num-warmups 0 \
  --ready-check-timeout-sec 0 \
  --metric-percentiles 95,99 \
  --host 127.0.0.1 \
  --port 8000 \
  --num-prompts 2920 \
  --max-concurrency 146 \
  --ignore-eos \
  --trust-remote-code \
  --seed 0
```

### Hy4-preview-Channel-FP8-w8a8-v2 单 D BW1100 32x vLLM 0.25.1

两个 16 卡 BW1100 超节点，DP32＋EP32、fake prefill、MTP3 固定接受长度 2.885。两节点分别按下方 [D 环境准备](#hy4-preview-channel-int4-w4a8-单-d-bw1100-16x-vllm-0251)创建容器并安装插件。

#### 两节点环境变量

在两个 D 容器分别执行：

```bash
export D_NODE_IP="<D_NODE_0_IP>"
export D_PEER_IP="<D_NODE_1_IP>"
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_USE_MODELSCOPE=1
export ROCSHMEM_IPC_MNVL=1
export VLLM_DEEPEP_BUFFER_SIZE_MB=0
export VLLM_HCU_USE_LIGHTOP_PER_TOKEN_QUANT_FP8=1
export GPU_MAX_HW_QUEUES=3
export VLLM_HCU_USE_LIGHTOP_HY_V4_INDEXER=0
export VLLM_MOE_ROUTING_SIMULATION_STRATEGY=hcu_eplb_balancedness
export VLLM_HCU_MOE_ROUTING_SIMULATION_BALANCEDNESS=0.6
export VLLM_RANDOMIZE_DP_DUMMY_INPUTS=1
export VLLM_HCU_USE_LIGHTOP_FAST_TOPK_TRANSFORM=1
export VLLM_HCU_USE_AITER_OPUS_PAGED_MQA_LOGITS=1
```

#### D 节点 0：DP rank 0–15

```bash
export VLLM_HOST_IP="$D_NODE_IP"

vllm serve hygon/Hy4-preview-Channel-FP8-w8a8-v2 \
  --trust-remote-code \
  --host 0.0.0.0 \
  --tensor-parallel-size 1 \
  --data-parallel-size 32 \
  --data-parallel-size-local 16 \
  --data-parallel-start-rank 0 \
  --data-parallel-address "$D_NODE_IP" \
  --data-parallel-rpc-port 1127 \
  --data-parallel-hybrid-lb \
  --api-server-count 1 \
  --enable-expert-parallel \
  --all2all-backend deepep_low_latency \
  --moe-backend deep_gemm \
  --no-enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"use_local_argmax_reduction":true,"rejection_sample_method":"synthetic","synthetic_acceptance_length":2.885}' \
  --gpu-memory-utilization 0.85 \
  --max-model-len 89088 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 32 \
  --default-chat-template-kwargs '{"reasoning_effort":"no_think"}' \
  --reasoning-parser hy_v4 \
  --enable-auto-tool-choice \
  --tool-call-parser hy_v4 \
  --kv-transfer-config '{"kv_connector":"DecodeBenchConnector","kv_role":"kv_both","kv_connector_extra_config":{"fill_mean":0.015,"fill_std":0.0}}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --kv-cache-dtype fp8_e4m3
```

#### D 节点 1：DP rank 16–31

```bash
export VLLM_HOST_IP="$D_PEER_IP"

vllm serve hygon/Hy4-preview-Channel-FP8-w8a8-v2 \
  --trust-remote-code \
  --host 0.0.0.0 \
  --tensor-parallel-size 1 \
  --data-parallel-size 32 \
  --data-parallel-size-local 16 \
  --data-parallel-start-rank 16 \
  --data-parallel-address "$D_NODE_IP" \
  --data-parallel-rpc-port 1127 \
  --data-parallel-hybrid-lb \
  --api-server-count 1 \
  --enable-expert-parallel \
  --all2all-backend deepep_low_latency \
  --moe-backend deep_gemm \
  --no-enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"use_local_argmax_reduction":true,"rejection_sample_method":"synthetic","synthetic_acceptance_length":2.885}' \
  --gpu-memory-utilization 0.85 \
  --max-model-len 89088 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 32 \
  --default-chat-template-kwargs '{"reasoning_effort":"no_think"}' \
  --reasoning-parser hy_v4 \
  --enable-auto-tool-choice \
  --tool-call-parser hy_v4 \
  --kv-transfer-config '{"kv_connector":"DecodeBenchConnector","kv_role":"kv_both","kv_connector_extra_config":{"fill_mean":0.015,"fill_std":0.0}}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --kv-cache-dtype fp8_e4m3
```

#### Router

在 D 节点 0 容器另开终端执行：

```bash
export D_NODE_IP="<D_NODE_0_IP>"
export D_PEER_IP="<D_NODE_1_IP>"

curl -fsS "http://${D_NODE_IP}:8000/health"
curl -fsS "http://${D_PEER_IP}:8000/health"
pip install vllm-router
vllm-router \
  --host 0.0.0.0 \
  --port 30000 \
  --worker-urls "http://${D_NODE_IP}:8000" "http://${D_PEER_IP}:8000" \
  --policy round_robin
```

#### 客户端

在 D 节点 0 容器另开终端执行：

```bash
export VLLM_USE_MODELSCOPE=1
export D_NODE_IP="<D_NODE_0_IP>"

vllm bench serve \
  --model hygon/Hy4-preview-Channel-FP8-w8a8-v2 \
  --base-url "http://${D_NODE_IP}:30000" \
  --dataset-name random \
  --random-input-len 87040 \
  --random-output-len 1228 \
  --random-range-ratio 0 \
  --num-prompts 512 \
  --max-concurrency 256 \
  --request-rate inf \
  --temperature 0 \
  --ignore-eos \
  --trust-remote-code \
  --metric-percentiles 95,99 \
  --percentile-metrics ttft,tpot,itl,e2el
```

### Hy4-preview-Channel-INT4-w4a8 1P1D BW1100 32x vLLM 0.25.1

P、D 各使用一个 16 卡 BW1100 超节点，离线 EPLB 冗余专家数为 16。

镜像：`42.228.13.241:5000/jenkins/model_test_env/vllm:0.25.1-ubuntu22.04-dtk26041-py3.10-20261111-0523`。

P 容器当前目录需有 [r16.json（raw 下载）](https://raw.githubusercontent.com/HYGON-AI/inference-cookbook-das/b873be9a4a3dc7af08227e3b07ef57aafddcee58/docs/model-deployment/vllm/configs/hy4/r16.json)，D 容器当前目录需有 [proxy.py（raw 下载）](https://raw.githubusercontent.com/HYGON-AI/inference-cookbook-das/926d8f20c58734aa2f8cf090870bf1f189db32cf/docs/model-deployment/vllm/configs/hy4/proxy.py)。

`/opt/hyhal` 是宿主机驱动目录，两端容器均需只读挂载；若驱动安装在其他目录，将挂载源改为实际目录，容器内目标仍为 `/opt/hyhal`。

#### 设置环境变量

在两台宿主机分别执行，替换为实际节点地址：

```bash
export P_NODE_IP="<P_IP>"
export D_NODE_IP="<D_IP>"
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
docker pull 42.228.13.241:5000/jenkins/model_test_env/vllm:0.25.1-ubuntu22.04-dtk26041-py3.10-20261111-0523
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
  -v /opt/hyhal:/opt/hyhal:ro \
  -e P_NODE_IP="$P_NODE_IP" \
  -e D_NODE_IP="$D_NODE_IP" \
  -e VLLM_HOST_IP="$P_NODE_IP" \
  --workdir /workspace \
  --entrypoint /usr/bin/sleep \
  42.228.13.241:5000/jenkins/model_test_env/vllm:0.25.1-ubuntu22.04-dtk26041-py3.10-20261111-0523 infinity
docker exec -it hy4-p bash
```

#### D 机器：创建容器

在 D 宿主机执行，最后一条命令进入 D 容器：

```bash
docker pull 42.228.13.241:5000/jenkins/model_test_env/vllm:0.25.1-ubuntu22.04-dtk26041-py3.10-20261111-0523
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
  -v /opt/hyhal:/opt/hyhal:ro \
  -e P_NODE_IP="$P_NODE_IP" \
  -e D_NODE_IP="$D_NODE_IP" \
  -e VLLM_HOST_IP="$D_NODE_IP" \
  --workdir /workspace \
  --entrypoint /usr/bin/sleep \
  42.228.13.241:5000/jenkins/model_test_env/vllm:0.25.1-ubuntu22.04-dtk26041-py3.10-20261111-0523 infinity
docker exec -it hy4-d bash
```

#### 重新安装插件

使用此镜像后，P、D 两端均需在容器内安装以下版本插件，再启动服务：

```bash
pip install vllm-hcu==0.25.1+dtk2604.torch2110.2611121359.g55130a -i https://pypi.sourcefind.cn/nightly/dtk/
```

#### P node

在 P 容器执行：

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_USE_MODELSCOPE=1
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
  --eplb-config '{"expert_map_path":"r16.json","num_redundant_experts":16,"static_dispatch_policy":"nearest"}'
```

#### D node

在 D 容器执行：

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_USE_MODELSCOPE=1
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

两端服务就绪后，在 D 宿主机另开终端，进入 D 容器运行 [proxy.py（raw 下载）](https://raw.githubusercontent.com/HYGON-AI/inference-cookbook-das/926d8f20c58734aa2f8cf090870bf1f189db32cf/docs/model-deployment/vllm/configs/hy4/proxy.py)：

```bash
docker exec -it hy4-d bash
```

```bash
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

在 D 宿主机另开终端进入 D 容器：

```bash
docker exec -it hy4-d bash
```

输入 85K、输出 1200。

##### 预热：146 请求

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

##### 实测结果：2920 请求

本次客户端输出如下：

```text
============ Serving Benchmark Result ============
Successful requests:                     2920
Failed requests:                         0
Maximum request concurrency:             146
Request rate configured (RPS):           2.55
Benchmark duration (s):                  1182.40
Total input tokens:                      248200000
Total generated tokens:                  3504000
Request throughput (req/s):              2.47
Output token throughput (tok/s):         2963.46
Peak output token throughput (tok/s):    1160.00
Peak concurrent requests:                149.00
Total token throughput (tok/s):          212875.44
---------------Time to First Token----------------
Mean TTFT (ms):                          7880.24
Median TTFT (ms):                        4687.29
P95 TTFT (ms):                           18757.51
P99 TTFT (ms):                           21082.39
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          30.30
Median TPOT (ms):                        30.13
P95 TPOT (ms):                           30.92
P99 TPOT (ms):                           45.67
---------------Inter-token Latency----------------
Mean ITL (ms):                           118.80
Median ITL (ms):                         117.05
P95 ITL (ms):                            181.52
P99 ITL (ms):                            308.37
---------------Speculative Decoding---------------
Acceptance rate (%):                     96.12
Acceptance length:                       3.88
Drafts:                                  903352
Draft tokens:                            2710056
Accepted tokens:                         2604839
Per-position acceptance (%):
  Position 0:                            97.17
  Position 1:                            95.91
  Position 2:                            95.27
==================================================
```


### Hy4-preview-Channel-INT4-w4a8 单 P BW1100 16x vLLM 0.25.1

P 使用一个 16 卡 BW1100 超节点，离线 EPLB 冗余专家数为 16。

镜像：`42.228.13.241:5000/jenkins/model_test_env/vllm:0.25.1-ubuntu22.04-dtk26041-py3.10-20261111-0523`。

P 容器当前目录需有 [r16.json（raw 下载）](https://raw.githubusercontent.com/HYGON-AI/inference-cookbook-das/926d8f20c58734aa2f8cf090870bf1f189db32cf/docs/model-deployment/vllm/configs/hy4/r16.json)。

`/opt/hyhal` 是宿主机驱动目录，容器需只读挂载；若驱动安装在其他目录，将挂载源改为实际目录，容器内目标仍为 `/opt/hyhal`。

#### 设置环境变量

在 P 宿主机执行，替换为实际节点地址：

```bash
export P_NODE_IP="<P_IP>"
```

镜像拉取失败时，在宿主机 `/etc/docker/daemon.json` 的 `insecure-registries` 数组中追加 `42.228.13.241:5000`，保留已有配置：

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
docker pull 42.228.13.241:5000/jenkins/model_test_env/vllm:0.25.1-ubuntu22.04-dtk26041-py3.10-20261111-0523
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
  -v /opt/hyhal:/opt/hyhal:ro \
  -e P_NODE_IP="$P_NODE_IP" \
  -e VLLM_HOST_IP="$P_NODE_IP" \
  --workdir /workspace \
  --entrypoint /usr/bin/sleep \
  42.228.13.241:5000/jenkins/model_test_env/vllm:0.25.1-ubuntu22.04-dtk26041-py3.10-20261111-0523 infinity
docker exec -it hy4-p bash
```

#### 重新安装插件

使用此镜像后，在 P 容器内安装以下版本插件，再启动服务：

```bash
pip install vllm-hcu==0.25.1+dtk2604.torch2110.2611121359.g55130a -i https://pypi.sourcefind.cn/nightly/dtk/
```

#### 启动服务

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_USE_MODELSCOPE=1
export VLLM_HOST_IP="$P_NODE_IP"
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

#### 客户端

在 P 服务所在容器另开终端执行。输入 79050 共享前缀 + 5950 随机后缀 = 85000 token，输出 1 token。先预热 146 条，再实跑 2920 条，两轮不重启服务。两条命令保持相同 seed；共享前缀占输入的 93%，实际命中率以服务端统计为准。

##### 预热：146 请求

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
  --prefix-repetition-output-len 1 \
  --request-rate 2.55 \
  --burstiness inf \
  --temperature 0 \
  --disable-shuffle \
  --num-warmups 0 \
  --ready-check-timeout-sec 0 \
  --metric-percentiles 95,99 \
  --host 127.0.0.1 \
  --port 8000 \
  --num-prompts 146 \
  --max-concurrency 146 \
  --ignore-eos \
  --trust-remote-code \
  --seed 0 \
  --save-result \
  --save-detailed \
  --result-dir results/hy4-p-146 \
  --result-filename result.json
```

##### 实跑：2920 请求

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
  --prefix-repetition-output-len 1 \
  --request-rate 2.55 \
  --burstiness inf \
  --temperature 0 \
  --disable-shuffle \
  --num-warmups 0 \
  --ready-check-timeout-sec 0 \
  --metric-percentiles 95,99 \
  --host 127.0.0.1 \
  --port 8000 \
  --num-prompts 2920 \
  --max-concurrency 146 \
  --ignore-eos \
  --trust-remote-code \
  --seed 0 \
  --save-result \
  --save-detailed \
  --result-dir results/hy4-p-2920 \
  --result-filename result.json
```


### Hy4-preview-Channel-INT4-w4a8 单 D BW1100 16x vLLM 0.25.1

D 使用一个 16 卡 BW1100 超节点。

镜像：`42.228.13.241:5000/jenkins/model_test_env/vllm:0.25.1-ubuntu22.04-dtk26041-py3.10-20261111-0523`。

`/opt/hyhal` 是宿主机驱动目录，容器需只读挂载；若驱动安装在其他目录，将挂载源改为实际目录，容器内目标仍为 `/opt/hyhal`。

#### 设置环境变量

在 D 宿主机执行，替换为实际节点地址：

```bash
export D_NODE_IP="<D_IP>"
```

镜像拉取失败时，在宿主机 `/etc/docker/daemon.json` 的 `insecure-registries` 数组中追加 `42.228.13.241:5000`，保留已有配置：

```json
{
  "insecure-registries": ["42.228.13.241:5000"]
}
```

修改后、创建容器前重启 Docker；重启会影响该宿主机已有容器。

```bash
sudo systemctl restart docker
```

#### D 机器：创建容器

在 D 宿主机执行，最后一条命令进入 D 容器：

```bash
docker pull 42.228.13.241:5000/jenkins/model_test_env/vllm:0.25.1-ubuntu22.04-dtk26041-py3.10-20261111-0523
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
  -v /opt/hyhal:/opt/hyhal:ro \
  -e D_NODE_IP="$D_NODE_IP" \
  -e VLLM_HOST_IP="$D_NODE_IP" \
  --workdir /workspace \
  --entrypoint /usr/bin/sleep \
  42.228.13.241:5000/jenkins/model_test_env/vllm:0.25.1-ubuntu22.04-dtk26041-py3.10-20261111-0523 infinity
docker exec -it hy4-d bash
```

#### 重新安装插件

使用此镜像后，在 D 容器内安装以下版本插件，再启动服务：

```bash
pip install vllm-hcu==0.25.1+dtk2604.torch2110.2611121359.g55130a -i https://pypi.sourcefind.cn/nightly/dtk/
```

#### 启动服务

`DecodeBenchConnector` 填充模拟 KV；MTP3 平均接受长度设为 2.95（含 bonus，对应约 65% draft token 接受率）。

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_USE_MODELSCOPE=1
export VLLM_HOST_IP="$D_NODE_IP"
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
  --no-enable-prefix-caching \
  --kv-cache-dtype fp8_e4m3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"use_local_argmax_reduction":true,"rejection_sample_method":"synthetic","synthetic_acceptance_length":2.95}' \
  --default-chat-template-kwargs '{"reasoning_effort":"no_think"}' \
  --reasoning-parser hy_v4 \
  --enable-auto-tool-choice \
  --tool-call-parser hy_v4 \
  --kv-transfer-config '{"kv_connector":"DecodeBenchConnector","kv_role":"kv_both","kv_connector_extra_config":{"fill_mean":0.015,"fill_std":0.0}}' \
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

#### 客户端

在 D 服务所在容器另开终端执行。使用 random 数据集，输入 85000 token、输出 1200 token；fake prefill 按输入长度分配模拟 KV，不需要构造共享前缀。

##### 预热：146 请求

```bash
export VLLM_USE_MODELSCOPE=1

vllm bench serve \
  --backend openai \
  --endpoint /v1/completions \
  --model hygon/Hy4-preview-Channel-INT4-w4a8 \
  --dataset-name random \
  --random-input-len 85000 \
  --random-output-len 1200 \
  --random-prefix-len 0 \
  --random-range-ratio 0 \
  --request-rate 2.55 \
  --burstiness inf \
  --temperature 0 \
  --disable-shuffle \
  --num-warmups 0 \
  --ready-check-timeout-sec 0 \
  --metric-percentiles 95,99 \
  --host 127.0.0.1 \
  --port 8000 \
  --num-prompts 146 \
  --max-concurrency 146 \
  --ignore-eos \
  --trust-remote-code \
  --seed 0 \
  --save-result \
  --save-detailed \
  --result-dir results/hy4-d-146 \
  --result-filename result.json
```

##### 实跑：2920 请求

```bash
export VLLM_USE_MODELSCOPE=1

vllm bench serve \
  --backend openai \
  --endpoint /v1/completions \
  --model hygon/Hy4-preview-Channel-INT4-w4a8 \
  --dataset-name random \
  --random-input-len 85000 \
  --random-output-len 1200 \
  --random-prefix-len 0 \
  --random-range-ratio 0 \
  --request-rate 2.55 \
  --burstiness inf \
  --temperature 0 \
  --disable-shuffle \
  --num-warmups 0 \
  --ready-check-timeout-sec 0 \
  --metric-percentiles 95,99 \
  --host 127.0.0.1 \
  --port 8000 \
  --num-prompts 2920 \
  --max-concurrency 146 \
  --ignore-eos \
  --trust-remote-code \
  --seed 0 \
  --save-result \
  --save-detailed \
  --result-dir results/hy4-d-2920 \
  --result-filename result.json
```
