# Third-Party Notices

## HY4 Mooncake proxy snapshot

The files under `docs/model-deployment/vllm/configs/hy4/proxy/` are distributed
under Apache-2.0; the repository's MIT license does not replace that license.
The Apache-2.0 license and the source project's NOTICE are retained there.

| Local file | Source / fixed revision | Copyright / license | Modifications |
| --- | --- | --- | --- |
| `managed_proxy_ka1.py`, `proxy_task_lifecycle.py` | [HYGON-AI/vllm-plugin-das](https://github.com/HYGON-AI/vllm-plugin-das/tree/67c376c0aaf400425ef2e0d42410ee51b18385b8/tools/hy_v4/vl_pd_validation), commit `67c376c0aaf400425ef2e0d42410ee51b18385b8` | Copyright (c) 2026 Hygon Information Technology Co., Ltd.; Apache-2.0 | Copied byte for byte; no runtime changes. |
| `mooncake_connector_proxy_ka1.py` | Same pinned plugin path; adapted from [vLLM](https://github.com/vllm-project/vllm), upstream baseline `752a3a504485790a2e8491cacbb35c137339ad34` | Copyright contributors to the vLLM project; Apache-2.0 | HCU snapshot uses keepalive expiry 1 second and zero transport retries; no further cookbook modifications. |
| `text_proxy.py` | HYGON HY4 pure-text deployment entry used on 2026-11-11/12 | Copyright (c) 2026 Hygon Information Technology Co., Ltd.; Apache-2.0 | Starts the managed proxy with frontend keepalive 60 seconds and records transport metadata. |

The documentation references third-party models, frameworks, tools, services, and their public documentation. Those external projects are not redistributed by this repository and remain subject to their respective licenses and terms.

If third-party source code or copied/adapted examples are added later, record the project name, repository URL, fixed commit or version, copyright, license, local path, and HYGON modifications in this file before release.
