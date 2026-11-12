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
