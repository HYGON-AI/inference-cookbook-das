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

本节为纯文本 HY4 W4A8 的 **1P1D + P PCP16/EP16 + 离线 EPLB R16 + D DP16/EP16 + 自然 MTP3** 部署方案。P、D 各使用 16 卡，共 32 卡。P 使用真实模型路由，未开启 uniform EPLB；D 不设置 `synthetic_acceptance_length`。R16 是整个 EP 组每层增加 16 个冗余物理专家，不是每卡增加 16 个：256 个逻辑专家对应 272 个物理槽，每卡 17 个。

以下命令中的 `P_NODE_IP`、`D_NODE_IP`、`PROXY_IP` 是地址占位符，执行时替换为实际地址；本方案 proxy 与 D 同节点，`PROXY_IP` 填 D 地址。P/D HTTP 使用默认端口 `8000`；proxy 使用 `8002`，P/D Mooncake bootstrap 分别使用 `18998`/`18999`。各脚本通过位置参数接收地址，不绑定测试节点。网卡配置参考：[IB 网卡](../../troubleshooting/common-issues.md#ib网卡)，其中 GID index `3` 是本次环境的取值。

#### 环境和配套文件

| 项目 | 已验证配置 |
| --- | --- |
| 模型 | [hygon/Hy4-preview-Channel-INT4-w4a8](https://www.modelscope.cn/models/hygon/Hy4-preview-Channel-INT4-w4a8)，compressed-tensors INT4 W4A8 |
| 硬件 | 两节点各 16 张 BW1100；需要 HCU 和 RDMA 设备、host 网络、无限制 memlock、足够的共享内存 |
| 镜像 | `42.228.13.241:5000/hcu/base_env/vllm:0.25.1-ubuntu22.04-dtk26041-py3.10-20261110-1933` |
| vLLM core | `0.25.1+das185.dtk2604.torch2110.2608171710.g7b108a` |
| 插件 | `67c376c0aaf400425ef2e0d42410ee51b18385b8`，包含 [MR389 空 chunk 修复](https://github.com/HYGON-AI/vllm-plugin-das/pull/389) |
| flash_mla | `1.2.0+dtk2604.torch2110.2611101151.gb832af` |
| aiter / deep-ep | `0.1.8+dtk2604.torch2110.2611101900.g8900be` / `1.1.1+dtk2604.torch2110.2611101610.gac3937` |
| P | TP1 / DP1 / PCP16 / EP16，DeepEP HT，`deep_gemm`，eager，同步调度，R16 nearest |
| D | TP1 / DP16 / EP16，DeepEP LL，`deep_gemm`，自然 MTP3 |
| KV | FP8 E4M3，开启 prefix caching；跨节点通过 Mooncake RDMA 传输 |

配套脚本和映射位于 [configs/hy4](configs/hy4/)。需连同 `proxy/` 的全部文件一起复制，不能只取 `text_proxy.py`。插件和这些库须使用匹配的 DTK、Python、PyTorch ABI；安装插件不会自动替换 vLLM core 或提供模型权重。此部署依赖 core 的 PCP/EPLB 校验支持；已有镜像已通过启动验证，其他 core 版本先核对[插件的依赖说明](https://github.com/HYGON-AI/vllm-plugin-das/blob/67c376c0aaf400425ef2e0d42410ee51b18385b8/docs/hy4_w4a8_static_eplb.md)，不要直接改安装目录绕过校验。

在两端容器中获取本仓库并准备工作目录：

```bash
git clone https://github.com/HYGON-AI/inference-cookbook-das.git /workspace/inference-cookbook-das
mkdir -p /workspace/hy4-pd-eplb
cp -a /workspace/inference-cookbook-das/docs/model-deployment/vllm/configs/hy4/. /workspace/hy4-pd-eplb/
```

首次使用新镜像时，在两端分别本地安装已验证插件版本；已有源码目录应先核实提交，不要覆盖其他工作目录：

```bash
git clone --branch v0.25.1 https://github.com/HYGON-AI/vllm-plugin-das.git /workspace/vllm-plugin-das-hy4-pd
git -C /workspace/vllm-plugin-das-hy4-pd checkout 67c376c0aaf400425ef2e0d42410ee51b18385b8
cd /workspace/vllm-plugin-das-hy4-pd
env -u PYTHONPATH MAX_JOBS=16 python -m pip install --no-deps --no-build-isolation -e .
env -u PYTHONPATH python -c 'import torch; import vllm_hcu.hcu_ops; import importlib.metadata as m; print(m.version("vllm")); print(m.version("vllm-hcu"))'
python -m pip install modelscope
```

下面命令通过 `VLLM_USE_MODELSCOPE=1` 读取官方模型 ID，需有 ModelScope 下载权限和足够的模型缓存空间；上面的安装步骤补齐镜像未预装的 ModelScope 依赖。历史测试使用本地 `/data_hy4_w4a8/Hy4-preview-Channel-INT4-w4a8` checkpoint；复现实验时需核对权重版本、config 和 tokenizer，模型 ID 本身不保证远程后续 revision 与历史文件相同。`PYTHONPATH` 不用于覆盖安装包。P 的 ByteLevel 尾部解码、MQA grouping、CP gather、共享专家 stream、HT count scratch、4 线程分词等有效开关保留；D 保留 fused RMS quant 和 attention input overlap。它们属于部署组合，不能把组合测得的收益归因于离线 EPLB 单项。

#### 离线 JSON：复用与生成

最直接的复现方式是使用随文提供的 [r16.json](configs/hy4/r16.json)，它与本次服务加载文件逐字节相同：

```bash
cd /workspace/hy4-pd-eplb
printf '%s\n' '9b99af7c6aa4824cfeca3e008365cf8a8bdaff071f458ddec9039f62b568a94b  r16.json' | sha256sum -c -
```

该文件是 version 2 格式，同时包含 `HYV4ForCausalLM` 的 77 行和 `HYV4MTP` 的 1 行；每行 272 个物理槽位，值为逻辑专家 ID，每个 0..255 的逻辑专家至少出现一次。模型、MoE 层数、逻辑专家数、EP 拓扑、冗余数或 MTP 配置变化后，需重新校验和采集，不能把此文本映射套到多模态 checkpoint。

现有 JSON 的生成流程如下，供更换流量后重新校准：

1. 使用真实路由采集全部 16 个源 rank 的逐层逻辑专家选中计数，主模型和 MTP 分开统计。原校准输入为 32 条 85K 请求，79,050 共享前缀、输出 8、并发 4；前 24 条拟合，后 8 条独立留出。
2. 将拟合集的同一模型、同一层计数按逻辑专家 ID 累加，得到主模型 `[77, 256]` 和 MTP `[1, 256]` 的非负整数直方图。计数包括实际执行的 PCP/MTP padding；invalid ID 不计入。留出集不能参与拟合。
3. 使用校准时固定的 vLLM `DefaultEplbPolicy.rebalance_experts_hierarchical(counts, 272, 1, 1, 16)` 计算映射，分别得到 `[77, 272]` 和 `[1, 272]`。原 policy 源文件 SHA256 为 `3b54229e6240d050480390f6b29d4744fd38437d23719f78a6decf7002296a29`；更换 policy 后需记录新来源，不能声称复现了原布局。
4. 将结果写入 version 2 的 `model_maps`，每个 entry 保存 `model_class`、`num_logical_experts=256`、`num_moe_layers`、`num_physical_experts=272`、`num_redundant_experts=16`、`physical_to_logical_map`；逐行校验 ID 范围和所有逻辑专家均有副本。
5. 用新服务加载映射，再回放独立留出集，记录**实际派发后的物理专家负载**及精度。不能把 CPU 上均匀分摊副本的估算当作 nearest 策略下的真实均衡度或提速。

这里提供的是可直接加载的原 JSON 和校准流程；原始校准请求、逐 rank captures 不随本文发布，不能仅凭上述文字重新得到相同 JSON。需要重新校准时，应先准备代表性流量和受控的路由采集工具，采集关闭后再测性能。通用的[静态 EPLB 说明](../../optimization/static-eplb-vllm.md)可用于理解文件格式，但 HY4 W4A8 此路径是**启动时直接加载并固定映射**，不支持动态专家权重迁移；不能照搬其他模型的动态 `expert_map_record_path` 生成方式。P 加载参数见下文，D 不加载该 R16 JSON。

#### P node

对应文件：[P-server.sh](configs/hy4/P-server.sh)。不显式限制 P 的 `max-num-seqs`；上下文上限保留实测的 88,000。

```bash
#!/bin/bash
set -euo pipefail

unset PYTHONPATH HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_USE_MODELSCOPE=1
export VLLM_HOST_IP="${1:?usage: P-server.sh P_NODE_IP}"
export VLLM_MOONCAKE_BOOTSTRAP_PORT=18998
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export MC_GID_INDEX=3
export NCCL_IB_GID_INDEX=3
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

exec vllm serve hygon/Hy4-preview-Channel-INT4-w4a8 \
  --trust-remote-code \
  --quantization compressed-tensors \
  --host 0.0.0.0 \
  --port 8000 \
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
  --eplb-config '{"expert_map_path":"/workspace/hy4-pd-eplb/r16.json","num_redundant_experts":16,"static_dispatch_policy":"nearest"}'
```

#### D node

对应文件：[D-server.sh](configs/hy4/D-server.sh)。D 保留实测的每 DP rank `max-num-seqs=8`；MTP 是自然接受，不固定接受率。

```bash
#!/bin/bash
set -euo pipefail

unset PYTHONPATH HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_USE_MODELSCOPE=1
export VLLM_HOST_IP="${1:?usage: D-server.sh D_NODE_IP}"
export VLLM_MOONCAKE_BOOTSTRAP_PORT=18999
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export MC_GID_INDEX=3
export NCCL_IB_GID_INDEX=3
export ROCSHMEM_IPC_MNVL=1
export VLLM_DEEPEP_BUFFER_SIZE_MB=0

export GPU_MAX_HW_QUEUES=3
export VLLM_HCU_USE_FUSED_RMS_QUANT=1
export VLLM_HCU_ENABLE_HYV4_ATTN_INPUT_OVERLAP=1
export VLLM_HCU_ENABLE_BYTELEVEL_DETOKENIZER_TAIL=1
export TOKENIZERS_PARALLELISM=false

exec vllm serve hygon/Hy4-preview-Channel-INT4-w4a8 \
  --trust-remote-code \
  --quantization compressed-tensors \
  --host 0.0.0.0 \
  --port 8000 \
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

在 D 节点运行 [proxy.sh](configs/hy4/proxy.sh)：

```bash
#!/bin/bash
set -euo pipefail

export VLLM_PLUGINS=__disabled__
export HY4_PROXY_RUN_DIR=/workspace/hy4-pd-eplb/proxy_run
unset PYTHONPATH HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
mkdir -p "$HY4_PROXY_RUN_DIR"

exec /usr/bin/python -u \
  /workspace/hy4-pd-eplb/proxy/text_proxy.py \
  --host 0.0.0.0 \
  --port 8002 \
  --prefill "http://${1:?usage: proxy.sh P_NODE_IP D_NODE_IP}:8000" 18998 \
  --decode "http://${2:?usage: proxy.sh P_NODE_IP D_NODE_IP}:8000"
```

[text_proxy.py](configs/hy4/proxy/text_proxy.py) 使用同目录的 `managed_proxy_ka1.py`、`mooncake_connector_proxy_ka1.py` 和 `proxy_task_lifecycle.py`。P 请求和 D 请求并发发起，由 Mooncake 完成 KV 就绪协调；proxy 管理失败/断连时的任务取消和回收。客户端连接 keepalive 为 60 秒，上游连接空闲保留 1 秒，不重试模型请求。P 进程更换后需要重启 proxy，以重新获取 producer engine ID。

本方案使用独立 Mooncake proxy，没有使用旧 DuSwift 模板中的 P 进程内置代理。客户端统一访问 D 上的 `8002`，不要直接请求 D 的 `8000` 来代替完整 PD 链路。

#### 启动与顺序测试

在 P、D 两端各开一个容器终端，分别执行以下对应命令；它们在前台运行，日志保存到工作目录：

```bash
bash /workspace/hy4-pd-eplb/P-server.sh P_NODE_IP > /workspace/hy4-pd-eplb/P-server.log 2>&1
```

```bash
bash /workspace/hy4-pd-eplb/D-server.sh D_NODE_IP > /workspace/hy4-pd-eplb/D-server.log 2>&1
```

待模型加载完成后，在 D 的另一个终端检查两端健康并启动 proxy：

```bash
curl --noproxy '*' --fail "http://P_NODE_IP:8000/health"
curl --noproxy '*' --fail "http://D_NODE_IP:8000/health"
bash /workspace/hy4-pd-eplb/proxy.sh P_NODE_IP D_NODE_IP > /workspace/hy4-pd-eplb/proxy.log 2>&1
```

正式测试用新的服务进程，保证不继承前一次服务的 KV。只做 `/health` 检查，不先发模型预热请求。P、D、proxy 都就绪后，下面顺序入口只运行一次，三阶段共用同一组服务：

```bash
curl --noproxy '*' --fail "http://PROXY_IP:8002/health"
bash /workspace/hy4-pd-eplb/client-humaneval.sh PROXY_IP &&
bash /workspace/hy4-pd-eplb/client-146.sh PROXY_IP &&
bash /workspace/hy4-pd-eplb/client-2920.sh PROXY_IP
```

HumanEval 客户端需要 EvalScope `1.9.1` 及完整评测依赖，可以在独立 CPU 工具容器中运行；将同一脚本包放在该工具容器的相同路径并保证能访问 proxy。准备 EvalScope 格式的全量 164 道题到 `/workspace/humaneval/dataset/test.jsonl`，以下脚本会核对题号和已使用数据的 SHA256。数据由评测工具环境提供，不在此包内分发。

```bash
#!/bin/bash
set -euo pipefail

mkdir -p /workspace/hy4-pd-eplb/results

export VLLM_PLUGINS=__disabled__
export HIP_VISIBLE_DEVICES=
export CUDA_VISIBLE_DEVICES=
export ROCR_VISIBLE_DEVICES=
unset PYTHONPATH HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

/usr/bin/python - <<'PY'
from pathlib import Path
import hashlib, json
p = Path('/workspace/humaneval/dataset/test.jsonl')
assert hashlib.sha256(p.read_bytes()).hexdigest() == 'b2adeeae8b383b6f0c615746a397f25c8abe44da4afb29cd10e6c4bbf0463fd3'
rows = [json.loads(line) for line in p.read_text().splitlines()]
assert len(rows) == 164
assert {row['task_id'] for row in rows} == {f'HumanEval/{i}' for i in range(164)}
print('HumanEval dataset verified: 164 tasks')
PY

exec evalscope eval \
  --model hygon/Hy4-preview-Channel-INT4-w4a8 \
  --eval-type openai_api \
  --api-url "http://${1:?usage: client-humaneval.sh PROXY_IP}:8002/v1" \
  --api-key EMPTY \
  --datasets humaneval \
  --dataset-args '{"humaneval":{"local_path":"/workspace/humaneval/dataset","subset_list":["default"]}}' \
  --repeats 1 \
  --eval-batch-size 16 \
  --seed 0 \
  --generation-config '{"temperature":0,"do_sample":false,"seed":0,"max_tokens":2048,"stream":true,"timeout":600,"extra_body":{"chat_template_kwargs":{"reasoning_effort":"no_think"}}}' \
  --work-dir /workspace/hy4-pd-eplb/results/humaneval \
  --no-timestamp
```

性能负载为 `prefix_repetition`：1 个共享前缀 79,050 + 后缀 5,950 = 85,000 输入 token，输出 1,200，固定到达率 2.55 请求/秒，并发上限 146，temperature 0、seed 0。146 和 2920 使用相同 seed、前缀配置和 cache salt，**中间不重启、不清 KV**，允许复用 146 留下的 KV；2920 轮内也允许命中。这是共享长前缀场景，不是每请求冷计算。`num-warmups=0` 只表示不发送额外 warmup，并不关闭 prefix caching 或消除跨轮复用。

第一轮：[client-146.sh](configs/hy4/client-146.sh)。

```bash
#!/bin/bash
set -euo pipefail

mkdir -p /workspace/hy4-pd-eplb/results

export VLLM_PLUGINS=__disabled__
export VLLM_USE_MODELSCOPE=1
unset PYTHONPATH HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

exec vllm bench serve \
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
  --host "${1:?usage: client-146.sh PROXY_IP}" \
  --port 8002 \
  --num-prompts 146 \
  --max-concurrency 146 \
  --ignore-eos \
  --trust-remote-code \
  --seed 0 \
  --save-result \
  --save-detailed \
  --result-dir /workspace/hy4-pd-eplb/results/round146-greedy \
  --result-filename result.json \
  --header X-Hy4-Bench-Case=hy4-v0251-latest-minenv-146 \
  --extra-body '{"cache_salt":"hy4-pd-eplb-run1"}'
```

第二轮：[client-2920.sh](configs/hy4/client-2920.sh) 与第一轮仅请求数、结果目录及用例标记不同：`--num-prompts 2920`，结果写到 `/workspace/hy4-pd-eplb/results/round2920-greedy/result.json`。命令已保存在配套脚本中。重复独立测试前，保存原结果、重新启动整组服务；测试期间不要混入其他请求。

#### 已有结果与口径

下表来自 **9c92afe 插件版本**的历史运行，服务参数与本节同一部署组合。实际顺序是 HumanEval 164 → 12 条精度诊断 → 146 → 2920。表中数字不是改成模型 ID 和新目录后重新跑出的结果，也不是 MR389 的独立收益。

| 指标 | 146 | 2920（保留 146 KV） |
| --- | ---: | ---: |
| 成功 / 失败 | 146 / 0 | 2920 / 0 |
| 完成 RPS | 1.37190 | 2.45934 |
| 输出 tokens/s | 1,646.28 | 2,951.21 |
| 系统输出 TPM | 98,776.89 | 177,072.84 |
| 按 32 卡平均输出 TPM/卡 | 3,086.78 | 5,533.53 |
| 客户端平均 TTFT | 16,249.49 ms | 2,752.53 ms |
| 客户端 P95 / P99 TTFT | 32,304.88 / 34,446.80 ms | 3,361.07 / 3,598.68 ms |
| 平均 TPOT | 29.74 ms/token | 30.17 ms/token |
| P 引擎平均 TTFT | 3.662 s | 1.927 s |
| D 引擎平均 TTFT | 16.111 s | 2.590 s |
| P 本地 prefix 命中率 | 92.2766% | 92.9439% |
| D 本地 prefix 命中率 | 61.0934% | 92.9393% |
| 自然 MTP 平均接受长度（含 bonus） | 3.90556 | 3.89122 |

配置 RPS 是请求到达率；完成 RPS 是成功请求数除以全轮耗时，含末尾 drain。TPM/卡按 P+D 总共 32 卡分摊，不是单卡独立解码能力。缓存命中率取 P/D `/metrics` 的前后 counter delta，不用客户端缺失明细时的 `cached=0` 判断。

该历史 HumanEval 全量为 **155/164（94.51%）**，上一轮为 159/164（96.95%），差异尚未定位，不能声称精度等价通过。greedy 合成长输入的自然 MTP 接受长度约 3.9，不代表真实对话接受率。

加入 MR389 后，另起新服务、无功能预热的真实会话长度筛选子集 C16 回放为 **100/100 成功**，平均 TTFT **9.751 s**、中位 **5.652 s**、P95 **33.622 s**、输出 **182.79 tokens/s**，P prefix 命中率 **74.25%**；自然 MTP 平均接受长度 3.1711。数据为 25 个 session、平均输入约 70.8K、自然输出上限 2048，既不是固定 RPS 2.55，也不是 85K/1200 的高命中压测，不能与上表直接比较吞吐。[MR389 记录](https://github.com/HYGON-AI/vllm-plugin-das/pull/389#issuecomment-6813985231)包含该轮完整命令、结果和验证限制；100 条 API 成功不等于工具语义正确或 HumanEval 通过。

当前方案不是单项性能 A/B。MR389 修复空 chunk 异常及请求不能完成的问题；不能据此声明 TTFT 或吞吐提升了某个百分比。本文未将未执行的 85K/100 条新合成数据测试记为通过。

#### 统计口径与退出

离线映射只固定专家放置和副本选择，并不保证任意流量都均衡。正式计时轮关闭路由采集和均衡度日志，避免额外通信影响；历史上述两轮没有记录实际专家 balancedness，不能将 92.94% 的缓存命中率称为专家均衡度。

本次不启用均衡度打印或路由采集，不附额外采集步骤。结束时先让请求 drain，再向已核实属于本轮的 P/D/proxy 主进程发送 SIGTERM，等待退出和显存释放。


## API 调用

### IFB

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "hygon/Hy4-preview-Channel-FP8-w8a8", "messages": [{"role": "user", "content": "中国的首都是哪里？"}], "max_tokens": 128}'
```

### PD 分离

验证功能时请求 proxy；如果要测冷 KV 性能，下面的模型调用示例不要在正式测试前执行：

```bash
curl --noproxy '*' --fail "http://PROXY_IP:8002/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"hygon/Hy4-preview-Channel-INT4-w4a8","messages":[{"role":"user","content":"中国的首都是哪里？"}],"max_tokens":128,"temperature":0,"stream":true,"chat_template_kwargs":{"reasoning_effort":"no_think"}}'
```
