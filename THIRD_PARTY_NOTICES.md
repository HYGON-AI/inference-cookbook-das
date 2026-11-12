# Third-Party Notices

## HY4 Mooncake proxy snapshot

`docs/model-deployment/vllm/configs/hy4/proxy.py` is distributed under
Apache-2.0; the repository's MIT license does not replace that license.
The Apache-2.0 license and the source project's NOTICE are retained beside it.

| Local file | Source / fixed revision | Copyright / license | Modifications |
| --- | --- | --- | --- |
| `proxy.py` | [HYGON-AI/vllm-plugin-das](https://github.com/HYGON-AI/vllm-plugin-das/tree/67c376c0aaf400425ef2e0d42410ee51b18385b8/tools/hy_v4/vl_pd_validation), commit `67c376c0aaf400425ef2e0d42410ee51b18385b8`; proxy originally adapted from [vLLM](https://github.com/vllm-project/vllm), baseline `752a3a504485790a2e8491cacbb35c137339ad34` | Copyright contributors to the vLLM project; Copyright (c) 2026 Hygon Information Technology Co., Ltd.; Apache-2.0 | Combines Mooncake request forwarding, managed startup and owned stream cleanup into one file. Retains frontend keepalive 60 seconds, upstream expiry 1 second and zero inference retries; removes transport-metadata file output. |

The documentation references third-party models, frameworks, tools, services, and their public documentation. Those external projects are not redistributed by this repository and remain subject to their respective licenses and terms.

If third-party source code or copied/adapted examples are added later, record the project name, repository URL, fixed commit or version, copyright, license, local path, and HYGON modifications in this file before release.
