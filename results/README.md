# KaLM-Jev v0.1 Test Report

Historical measurement date: 2026-09-21. Model alignment, semantic evaluation, performance, and CLI HTTP results were obtained with real R2 weights. Mocks were used only for unit tests that do not require weights.

Environment: H100 80GB MIG 3g.40gb (approximately 40GB available), PyTorch 2.8.0+cu129, and Transformers 5.3.0. Models were loaded sequentially, one per process. Precision checks used batch_size=4 and default pooling=4. See [requirements.txt](../requirements.txt) for dependency versions.

## Native inference and cache consistency

Four input groups were tested: official Jev Choice, Score, and Noul examples, plus this project's mixed request. The native reference directly executed the checkpoint's `_predict_batch`, capturing yes/no logits before softmax.

| Model | dtype | Max margin difference | Max probability difference | Max native batch variation | Warm encoder calls | Batch choice flips |
|---|---|---:|---:|---:|---:|---:|
| nano | float32 | 2.8610229e-06 | 2.5686642e-07 | 4.2915344e-06 | 0 | 0 |
| nano | bfloat16 | 0.0625 | 0.015605962 | 0.0625 | 0 | 0 |
| small | float32 | 4.2915344e-06 | 1.233561e-06 | 8.5830688e-06 | 0 | 0 |
| small | bfloat16 | 0.0625 | 0.0080576994 | 0.046875 | 0 | 0 |
| large | float32 | 9.059906e-06 | 1.4723277e-06 | 1.3828278e-05 | 0 | 0 |
| large | bfloat16 | 0.0625 | 0.014571463 | 0.0625 | 0 | 0 |

For each group, cache off/cold/warm returned identical final answers and usage, with warm `encoded_documents=0`. Additional checks covered different document lengths and padding, batch_size=1, changed states that reuse encoding but rerun scoring, instruction changes, reversed Score levels, and encoding exactly one newly added document.

FP32 acceptance targets were `atol=1e-5, rtol=1e-4`. BF16 is affected by logit quantization and batch shapes. The comparison bound explicitly used native batching variation plus a 0.0625 quantization step; actual differences are retained in JSON. FP32 independently checked mask and position consistency. The BF16 tolerance does not imply that every near-tie choice is invariant.

## Official example outputs (BF16)

| Model | Choice | P(returns) | Safari Score | Human-agent Noul | Repeat-contact Noul |
|---|---|---:|---:|---:|---:|
| nano | returns | 0.772172 | 1.441192 | 0.930458 | 0.963780 |
| small | returns | 0.929612 | 1.125890 | 0.936285 | 0.880797 |
| large | returns | 0.992038 | 1.080540 | 0.949669 | 0.962673 |

These are measured KaLM outputs, not a reproduction of Jev's weights or documented probabilities. Sources: [Choice](https://docs.typesafe.ai/primitives/choice), [Score](https://docs.typesafe.ai/primitives/score), and [Noul](https://docs.typesafe.ai/primitives/noul).

## Bilingual semantic smoke evaluation

There are 36 hand-reviewed cases: 12 per primitive, plus 12 additional repeat-contact judgments for Noul. The Noul threshold is 0.5, with no calibration. Score MAE uses manual levels 0/1/2 as references, not fitted Jev probabilities.

| Model | Choice accuracy | Score MAE | Human-agent Noul accuracy | Repeat-contact accuracy |
|---|---:|---:|---:|---:|
| nano | 66.67% | 0.598001 | 50.00% | 16.67% |
| small | 91.67% | 0.495509 | 50.00% | 41.67% |
| large | 83.33% | 0.248574 | 83.33% | 50.00% |

The fixed v1 adapters have clear semantic limitations. Noul in particular can return affirmative scores for negation, missing evidence, or mere topic mentions. Engineering alignment does not establish Jev-level judgment quality. Templates and thresholds were not tuned to improve these small-sample results; incorrect predictions remain in the records. This sample does not support general capability rankings or calibration claims.

## Cache performance (Nano BF16)

Configuration: batch_size=8, pooling=4, and a 256 MiB cache. Each workload/mode had 30 measured requests, totaling 450 timing samples. Computation was warmed up first, CUDA was explicitly synchronized, warm requests used distinct queries, and model loading was excluded.

| workload | off p50/p95 ms | cold p50/p95 ms | warm p50/p95 ms | speedup |
|---|---:|---:|---:|---:|
| choice-3 | 53.40/57.27 | 53.86/56.45 | 30.54/33.36 | 1.748x |
| choice-32 | 215.83/222.32 | 219.27/227.04 | 131.89/137.08 | 1.636x |
| choice-128 | 872.58/915.08 | 859.07/923.13 | 525.06/536.87 | 1.662x |
| score-5 | 53.29/56.44 | 53.30/54.34 | 31.68/32.18 | 1.682x |
| noul-8 | 55.83/58.62 | 55.37/56.84 | 33.82/35.80 | 1.651x |

All warm requests had actual `encoded_documents=0` and `encoder_calls=0`. JSON records also contain directly measured component timings, throughput, cache bytes, peak allocated CUDA memory, per-request token lengths, and raw samples. Encoder time was not inferred by subtracting total times. For short documents, encoding is only part of the workload; query/decoder computation still runs for every candidate.

## Other checks and scope

The original run passed 32 unit/protocol/HTTP tests, including strict JSON parsing, candidate limits, LRU/disabled/invalidation behavior, mixed-answer mapping, stable aggregation, and HTTP 400/422/429/500/503. Starlette emitted one TestClient/httpx deprecation warning without affecting results. CLI help, Python compilation, and shell syntax checks also passed. Subsequent release checks passed 35 tests, wheel installation, and real Nano/CLI HTTP regression.

The real CLI launched Nano on a temporary loopback port. Actual HTTP calls for the official examples and mixed request passed, with zero document encoding on warm requests. The server was stopped after the tests.

Full records are the `*-validation.json`, `*-semantics.json`, `*-benchmark.json`, and `http-smoke.json` files in this directory. Public metadata retains repository identifiers, file sizes, tokenizer/configuration/source SHA256 hashes, devices, and versions. Host directories and local modification times have been removed. Weight files have size metadata only; no full weight-content hash is claimed. Numerical measurements and bilingual evaluation text are unchanged. Raw logs, hostname-bearing JUnit XML, bytecode, and model weights are excluded.

Unverified areas include other GPUs, the complete CPU path, FP16, contexts longer than those tested, full Small/Large cache-performance matrices, and broad evaluations across pooling ratios. No remote repository was published or pushed, and no public-facing service was left running.
