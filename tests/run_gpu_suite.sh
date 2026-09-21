#!/usr/bin/env bash
set -euo pipefail
export HF_HUB_DISABLE_PROGRESS_BARS=1
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
python_bin="${KALM_PYTHON:-python}"
model_root="${KALM_MODEL_ROOT:-./models}"
export KALM_MODEL_ROOT="$model_root"
for size in Nano Small Large; do
    if [ ! -f "$model_root/KaLM-Reranker-V1-$size-R2/config.json" ]; then
        printf 'Missing %s model; set KALM_MODEL_ROOT to the directory containing all three R2 models.\n' "$size" >&2
        exit 1
    fi
done
for size in Nano Small Large; do
    alias="kalm-jev-${size,,}"
    model_path="$model_root/KaLM-Reranker-V1-$size-R2"
    for dtype in float32 bfloat16; do
        "$python_bin" tests/validate_models.py --model "$alias" --model-path "$model_path" --dtype "$dtype"
    done
    "$python_bin" tests/evaluate_semantics.py --model "$alias" --model-path "$model_path"
done
"$python_bin" benchmarks/cache_benchmark.py --model-path "$model_root/KaLM-Reranker-V1-Nano-R2" --samples 30
"$python_bin" tests/http_smoke.py
"$python_bin" tests/summarize_results.py
