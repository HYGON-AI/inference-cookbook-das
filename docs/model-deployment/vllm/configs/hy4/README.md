# HY4 W4A8 1P1D / offline R16

完整环境、参数、操作流程和验证边界见 [HY4 部署文档](../../hy4.md#hy4-preview-channel-int4-w4a8-1p1d-bw1100-32x-vllm-0251)。

将本目录完整复制到两端容器的 `/workspace/hy4-pd-eplb`。P 启动 `P-server.sh`；D 启动 `D-server.sh` 和 `proxy.sh`。P/D HTTP 8000，proxy 8002。地址通过脚本位置参数传入：P-server 接收 P 地址，D-server 接收 D 地址，proxy 接收 P/D 地址，三个客户端脚本接收 proxy 地址。`P_NODE_IP`、`D_NODE_IP`、`PROXY_IP` 是占位符，执行时替换为实际地址；GID index 也需符合本机网络。权重和 EvalScope 数据不在本目录分发。

| 文件 | 用途 |
| --- | --- |
| `P-server.sh` / `D-server.sh` | PCP16 + 离线 R16 / DP16；自然 MTP3 |
| `proxy.sh` / `proxy/` | 实测独立 Mooncake proxy 完整配套 |
| `r16.json` | 主模型 77 行 + MTP 1 行；256 logical / 272 physical |
| `client-humaneval.sh` | EvalScope 1.9.1，164 道题，核对本地数据 |
| `client-146.sh` / `client-2920.sh` | 85K/1200，RPS 2.55，跨轮保留 KV |
| `SHA256SUMS.json` | 文件 SHA256 清单 |

先用 `/health` 确认服务就绪，再运行：

```bash
bash /workspace/hy4-pd-eplb/client-humaneval.sh PROXY_IP &&
bash /workspace/hy4-pd-eplb/client-146.sh PROXY_IP &&
bash /workspace/hy4-pd-eplb/client-2920.sh PROXY_IP
```

若 HumanEval 使用独立工具容器，先在工具容器运行精度脚本，成功后再回到 D 服务容器依次运行两轮性能脚本；整个过程中不重启模型服务。

`r16.json` SHA256：`9b99af7c6aa4824cfeca3e008365cf8a8bdaff071f458ddec9039f62b568a94b`。它复用原 fit24 / heldout8 的纯文本校准布局，不是针对新流量重新拟合。校准 policy 源文件 SHA256：`3b54229e6240d050480390f6b29d4744fd38437d23719f78a6decf7002296a29`。完整校准流程与缺失的外部输入在主文档中说明。

此包将历史命令规范为官方模型 ID、`vllm serve` / `vllm bench serve` 和固定工作目录，显式写出 checkpoint 已声明的 `compressed-tensors` 量化方法。P/D 和性能客户端通过 `VLLM_USE_MODELSCOPE=1` 读取模型，需要 ModelScope 依赖与模型缓存空间；主文档给出依赖安装命令。算法、并行与优化开关保留；目录/命令规范化后仅做静态和 CPU 检查，未宣称重跑 HCU。历史精度差异、性能口径及不同插件版本的结果分别披露。本次不启用均衡度打印或路由采集。

`proxy/managed_proxy_ka1.py`、`proxy/mooncake_connector_proxy_ka1.py`、`proxy/proxy_task_lifecycle.py` 逐字节来自插件提交 `67c376c0aaf400425ef2e0d42410ee51b18385b8` 的 `tools/hy_v4/vl_pd_validation/`。`text_proxy.py` 为该代理的实测纯文本入口，设置前端 keepalive 并记录传输配置；它不改变推理计算或增加模型请求重试。Apache-2.0 授权及出处见该目录 `LICENSE`、`NOTICE` 和仓库 [第三方声明](../../../../../THIRD_PARTY_NOTICES.md)。
