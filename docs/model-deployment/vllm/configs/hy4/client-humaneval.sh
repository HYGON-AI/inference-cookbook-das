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
