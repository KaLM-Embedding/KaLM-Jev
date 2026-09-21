# KaLM-Jev Test Results

Values are read from the selected results directory; device and dtype come from run records. Host paths are omitted.

Current template: `v3-noul-state-document`. Original Choice/Score adapters and MEP 4x are retained. Noul with criteria scores two separate Documents; omitted criteria uses state as Query and one Document with a direct yes/no sigmoid. No calibration was fitted.

## Native Inference Alignment

| Model | dtype | Device | Max margin difference | Max probability difference | Warm encoder calls |
|---|---|---|---:|---:|---:|
| nano | float32 | NVIDIA H100 80GB HBM3 MIG 3g.40gb | 2.3841858e-06 | 4.653257e-07 | 0 |
| nano | bfloat16 | NVIDIA H100 80GB HBM3 MIG 3g.40gb | 0.0625 | 0.0061048123 | 0 |
| small | float32 | NVIDIA H100 80GB HBM3 MIG 3g.40gb | 1.001358e-05 | 2.4877323e-06 | 0 |
| small | bfloat16 | NVIDIA H100 80GB HBM3 MIG 3g.40gb | 0.0625 | 0.0034876829 | 0 |
| large | float32 | NVIDIA H100 80GB HBM3 MIG 3g.40gb | 1.0490417e-05 | 1.4723277e-06 | 0 |
| large | bfloat16 | NVIDIA H100 80GB HBM3 MIG 3g.40gb | 0.0625 | 0.0066669971 | 0 |

## Example Outputs (BF16)

| Model | Choice | P(returns) | Safari Score | Human-agent Noul | Repeat-contact Noul |
|---|---|---:|---:|---:|---:|
| nano | returns | 0.772172 | 1.441192 | 0.997973 | 0.373876 |
| small | returns | 0.929612 | 1.125890 | 0.998590 | 0.877477 |
| large | returns | 0.992038 | 1.080540 | 0.998830 | 0.974043 |

## Semantic Evaluation

36 hand-reviewed bilingual smoke cases: 12 per primitive plus 12 repeat-contact judgments. Noul threshold is 0.5. This is a small smoke set, not the JevBench leaderboard.

| Model | Choice accuracy | Score MAE | Human-agent Noul accuracy | Repeat-contact accuracy |
|---|---:|---:|---:|---:|
| nano | 66.67% | 0.598001 | 50.00% | 83.33% |
| small | 91.67% | 0.495509 | 50.00% | 58.33% |
| large | 83.33% | 0.248574 | 50.00% | 33.33% |

### Omitted-criteria Noul Breakdown

| Model | True positives | False positives | True negatives | False negatives |
|---|---:|---:|---:|---:|
| nano | 6 | 6 | 0 | 0 |
| small | 6 | 6 | 0 | 0 |
| large | 6 | 6 | 0 | 0 |

A high value on one affirmative example does not establish discrimination: inspect false positives on explicit negation and unrelated statements. Raw responses are retained in the semantics JSON files.

## Nano Cache Benchmark

Samples per mode: 30; batch_size=8.

| workload | off p50/p95 ms | cold p50/p95 ms | warm p50/p95 ms | speedup |
|---|---:|---:|---:|---:|
| choice-3 | 51.77/56.58 | 51.15/55.60 | 30.26/34.67 | 1.711x |
| choice-32 | 213.89/219.92 | 211.25/233.19 | 131.44/135.38 | 1.627x |
| choice-128 | 854.33/872.44 | 850.85/1020.02 | 500.91/512.17 | 1.706x |
| score-5 | 52.49/57.63 | 75.90/89.25 | 31.00/32.80 | 1.693x |
| noul-8 | 50.64/55.34 | 51.01/53.77 | 50.83/52.74 | 0.996x |

Warm requests have distinct states. Fixed criterion Documents are reused; for Noul without criteria each new state needs one encoder call, shared by the eight questions. Its warm mode does not mean an encoder cache hit.

These are measured results; runs with different devices, query lengths, or batch settings are not directly comparable.

Real CLI HTTP smoke passed. Run unit tests separately with python -m pytest -q.

Results do not establish Jev-equivalent quality or calibrated probabilities. Public identities omit host directories and modification times.

## Public JevBench Regression

Nano: 122/231 correct (52.81%); 0 prediction changes against the selected candidate-adapter baseline; maximum probability difference 0. All 74 Noul items supply both criteria, so this checks the supplied-criteria path, not the new missing-criteria behavior. Query limit 8192, decoder limit 9216, BF16, MEP 4x.
