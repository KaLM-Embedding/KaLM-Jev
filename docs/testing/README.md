# Tests and benchmarks

Run these commands from the repository root:

```bash
python -m pytest -q
KALM_MODEL_ROOT=./models bash tests/run_gpu_suite.sh
# Run the Nano cache benchmark: 30 samples per workload and mode
python benchmarks/cache_benchmark.py \
  --model-path ./models/KaLM-Reranker-V1-Nano-R2 \
  --samples 30
```

The GPU suite loads and releases the three models sequentially. It runs official Choice/Score/Noul examples and a mixed request, comparing raw margins captured from the native `_predict_batch` with the adapter. It checks cache off/cold/warm paths, actual encoder hook counts, batch sizes, padding, state changes, and document changes. FP32 uses `atol=1e-5, rtol=1e-4`. BF16 reports native batching variation, margin/probability differences, and choice flips; near-tie invariance is not guaranteed.

`tests/semantic_cases.json` contains 36 hand-reviewed English/Chinese smoke cases, 12 per primitive, with additional repeat-contact judgments for Noul and a threshold of 0.5. Coverage includes negation, topic mentions without a true condition, similar categories, explicit “other” options, adjacent levels, and multiple true Noul questions. These are smoke tests, not a leaderboard or calibration dataset. Bilingual test inputs and recorded outputs retain their original language.

The cache benchmark covers Choice with 3/32/128 candidates, Score with 5 levels, and Noul with 8 rules. It fixes model, dtype, batch size, and rules, warms up computation, then measures off/cold/warm paths. Warm requests use different states from the cache-filling request. Results include p50/p95, throughput, directly measured encoding/scoring time, actual encoding counts, cache bytes, peak allocated CUDA memory, token lengths, and raw latency samples. Cold mode clears the cache before each request. Warm mode asserts zero encoder calls for fixed Documents. Noul without criteria uses each new state as a Document, so each distinct-state request requires one encoding shared by its eight questions; an additional check verifies that adding one candidate encodes only one new document.

The full GPU suite requires all three model directories. Scripts default to the active environment's `python` and `./models`; override these with `KALM_PYTHON` and `KALM_MODEL_ROOT`. Scripts do not automatically set NVIDIA driver environment variables.

Sanitized measurements and the test report are included in [results/](../../results/README.md) and can be committed to GitHub. Test scripts also write to `results/`, and the summary generator updates `results/README.md`; rerunning the suite replaces the corresponding recorded files, so review the changes before committing. Public reports use `public_identity` to retain repository identifiers, file sizes, and source/configuration/tokenizer hashes. Internal cache identities remain complete. Raw logs and JUnit XML are Git-ignored. Official example provenance is documented in [examples/README.md](../../examples/README.md).

[Back to the main README](../../README.md)
