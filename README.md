# KaLM-Jev v0.1

A local Choice / Score / Noul service built on KaLM-Reranker R2. It uses PyTorch and Transformers, runs one model per process, and returns structured judgments without generating answer text.

The underlying reranker is described in [KaLM-Reranker-V1: Fast but Not Late Interaction for Compressed Document Reranking](https://arxiv.org/abs/2606.22807). See [Citation](#citation) for the BibTeX entry.

## Installation and quick start

Requires Python 3.10+. Run the following commands from the repository root. Tested dependency versions are listed in [requirements.txt](requirements.txt). GPU users should install a PyTorch build compatible with their device and driver. No specific Conda environment or CUDA library path is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
kalm-jev serve --model kalm-jev-nano \
  --device cuda --dtype bfloat16 --batch-size 4 --cache-max-mib 256
```

MIG instances work as ordinary CUDA devices. For CPU execution, use `--device cpu --dtype float32`. Recorded validation primarily used an H100 MIG instance; a full CPU performance evaluation has not been completed. The server binds to `127.0.0.1:8000`, uses one worker, and has no built-in authentication.

| Model alias | Checkpoint directory / Hugging Face repository suffix |
|---|---|
| kalm-jev-nano | KaLM-Reranker-V1-Nano-R2 |
| kalm-jev-small | KaLM-Reranker-V1-Small-R2 |
| kalm-jev-large | KaLM-Reranker-V1-Large-R2 |

All three repositories belong to `KaLM-Embedding`. Without `--model-path`, startup downloads a snapshot from the corresponding repository; the first download requires network access. Use `--revision` to select a revision. If you already have the complete model repository, run offline with a relative path:

```bash
kalm-jev serve --model kalm-jev-nano \
  --model-path ./models/KaLM-Reranker-V1-Nano-R2 --device cuda
```

Prepare `models/` separately; weights are not included in this repository. Each model directory must contain the weights, tokenizer, configuration, and original `kalm_reranker.py` and `kalm_reranker_utils.py` files.

Omitting `model` in a request uses the loaded alias. Requesting another model returns HTTP 400; requests never trigger model switching or downloads.

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' --data-binary @examples/mixed.json
```

The Python API returns a dictionary and shares the same engine as HTTP:

```python
from pathlib import Path
from kalm_jev import Engine

engine = Engine(
    model="kalm-jev-nano",
    model_path="./models/KaLM-Reranker-V1-Nano-R2",
    device="cuda", dtype="bfloat16",
)
request = Path("examples/choice.json").read_text()
print(engine.evaluate(request))  # Cold: 3 misses, 3 encoded documents
print(engine.evaluate(request))  # Warm: 3 hits, 0 encoded documents
```

## Example results

The following three examples show successful judgments from the recorded `kalm-jev-large` BF16 run with default, uncalibrated aggregation. Inputs are the official Jev examples preserved in [examples/](examples/README.md); outputs are KaLM-Jev measurements, not hosted Jev responses. The JSON outputs below show only `answers`, rounded to six decimal places. Full-precision outputs and execution metadata are available in the [recorded validation results](results/kalm-jev-large-bfloat16-validation.json).

### Choice: route a size-exchange request

Request ([examples/choice.json](examples/choice.json)):

```json
{
  "state": "My running shoes arrived in the wrong size. Can I swap them for a size 10?",
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": {
        "returns": "Exchanges, wrong or damaged items",
        "shipping": "Delivery status, delays, lost packages",
        "billing": "Charges, invoices, payment problems"
      }
    }
  }
}
```

Measured answer:

```json
{
  "department": {
    "type": "choice",
    "probabilities": {
      "returns": 0.992038,
      "shipping": 0.005541,
      "billing": 0.002421
    },
    "confidence": 0.953301,
    "choice": "returns"
  }
}
```

The model selects `returns`, matching the request to exchange an incorrectly sized item.

### Score: assess a broken feature with a workaround

Request ([examples/score.json](examples/score.json)):

```json
{
  "state": "The export button crashes the settings page in Safari. It works in Chrome, but a few of our customers only use Safari.",
  "questions": {
    "bug_severity": {
      "type": "score",
      "instructions": "How severe is the reported issue?",
      "criteria": [
        "Cosmetic; no impact to functionality",
        "Broken or degraded feature, but workaround exists",
        "Blocking issue; no workaround exists"
      ]
    }
  }
}
```

Measured answer:

```json
{
  "bug_severity": {
    "type": "score",
    "probabilities": {
      "0": 0.124152,
      "1": 0.671157,
      "2": 0.204691
    },
    "confidence": 0.225087,
    "score": 1.080540,
    "legend": {
      "0": "Cosmetic; no impact to functionality",
      "1": "Broken or degraded feature, but workaround exists",
      "2": "Blocking issue; no workaround exists"
    }
  }
}
```

Level `1` receives the most probability, and the expected score is close to `1`: the feature fails in Safari but works in Chrome. This is a sensible result, though the distribution is not sharply concentrated; its entropy-based confidence is only `0.225087`. Score is a continuous expected level index, not a hard class label.

### Noul: detect human escalation and repeat contact independently

Request ([examples/noul.json](examples/noul.json)):

```json
{
  "state": "I have asked three times now. Can I please just talk to a real person?",
  "questions": {
    "is_human_escalation": {
      "type": "noul",
      "instructions": "Is the customer asking for a human agent?"
    },
    "is_repeat_contact": {
      "type": "noul",
      "instructions": "Has the customer contacted support about this before?",
      "criteria": {
        "true": "Mentions a prior attempt, ticket, or that they have asked before",
        "false": "No sign of any previous contact"
      }
    }
  }
}
```

Measured answers:

```json
{
  "is_human_escalation": {
    "type": "noul",
    "noul": 0.949669
  },
  "is_repeat_contact": {
    "type": "noul",
    "noul": 0.962673
  }
}
```

Both judgments agree with explicit evidence in the message: “a real person” and “asked three times.” Each value is independent; they do not sum to one. Both exceed the `0.5` threshold used in the semantic smoke tests, but the API returns numeric values rather than Boolean decisions.

These are selected positive examples, not an estimate of overall accuracy. Outputs are not calibrated correctness probabilities. For all three model sizes, broader semantic results, and known limitations such as negation errors and Noul false positives, see the [test report](results/README.md).

## Inputs and model execution

Requests support `state`, `questions`, and optional `model`. An array-valued state is one complete business state, not a batch of independent samples. Content strings retain their whitespace. Objects and arrays use compact, Unicode-preserving JSON with sorted object keys. Question and candidate order are preserved; question IDs are not sent to the model.

| Primitive | Document | Task-specific instruction | Query |
|---|---|---|---|
| Choice | `option_id: description`, or only the ID for a null description | Original instructions + fixed Choice adapter | Serialized state |
| Score | The level description, without its index | Original instructions + fixed Score adapter | Serialized state |
| Noul | `Question: ...` plus optional True/False criteria | Original instructions + fixed Noul adapter | Serialized state |

Unknown fields, duplicate JSON keys, NaN/Infinity, implicit type coercion, empty IDs, and invalid candidate counts are rejected. If Noul criteria are supplied, they must contain at least one of `true` or `false`; null values are not accepted. Errors use `{"error":{"type":"...","message":"...","field":"..."}}`, with `field` omitted when not applicable.

The adapter loads the original `kalm_reranker.py` and `kalm_reranker_utils.py` from the checkpoint directory. These are executable Python files, so use trusted model repositories. Weights load through Transformers' built-in `T5Gemma2ForConditionalGeneration`; AutoModel does not require `trust_remote_code`. The two original scripts have matching SHA256 hashes across the three tested checkpoints.

The original `<Document>: ` prefix, system instruction, `<bos>/<start_of_turn>/<end_of_turn>` tokens, query tokenization/decoding, and yes/no readout at the final non-padding decoder token are preserved. The native implementation verifies that yes and no each occupy one token. The model runs in `.eval()` and `torch.inference_mode()`. By default, encoder states are mean-pooled in groups of four tokens using the attention mask.

The adapter supplies `encoder_outputs=BaseModelOutput(...)` to bypass document encoding on cache hits. Decoder and language-model-head execution follow the original model, with unused decoder KV caching disabled. It does not call `generate()` or replace interaction scoring with embedding similarity. Each task gets its own instruction, including mixed-question requests.

## Document cache and limits

The cache stores actual pooled hidden states and valid masks on the model device. Tensors are cloned to their effective length to avoid retaining padded batch storage. Scoring builds fresh padded tensors without modifying cached representations.

SHA256 cache keys include the checkpoint path and revision, local file size/mtime fingerprint, configuration/tokenizer/source hashes, dtype, Transformers/PyTorch versions, document template version, length and pooling settings, and document text. Replacing weights or changing encoder settings in a running engine is unsupported; create a new Engine instead.

The default LRU tensor budget is 256 MiB. Use `--cache-max-mib 0` to disable caching. A document larger than the budget remains usable for the current request but is not cached. The budget covers hidden-state and mask tensor bytes, not allocator-reserved memory, Python object overhead, or inference working memory. References held by an active request may exceed the cache budget, so request limits also matter.

Documents are deduplicated within each request. `unique_documents = hit_documents + miss_documents`; first-time duplicates are not counted as cross-request hits. `encoded_documents` counts actual encoding events. Changing state or Choice instructions reuses the document representation but reruns decoder scoring. Reordering Score levels reuses representations and aggregates against the new indices. Changing Noul instructions or criteria changes the document and causes re-encoding.

Requests default to at most 32 questions and 1,024 point-wise pairs; configure these with `--max-questions` and `--max-pairs`. `--max-pending 8` bounds requests running or waiting inside the engine; excess requests receive HTTP 429. A lock serializes inference. HTTP bodies default to a 2 MiB limit, configurable with `--max-body-bytes`.

Native defaults are 512 query tokens and 1,024 document tokens including the document prefix. The complete decoder template is separately limited to 2,048 tokens, including padding to a multiple of eight, to bound instruction length. Configure these with `--query-max-length`, `--document-max-length`, and `--decoder-max-length`. Overlong requests receive HTTP 422 before encoding; inputs are not silently truncated. A checkpoint's positional capacity does not imply that this project has validated that context length. Use `--chunk-size 4` for default pooling or `--chunk-size none` to disable it.

Validation failures return 422, model mismatches 400, unavailable models or device memory exhaustion 503, and other inference failures 500. HTTP responses omit stack traces. A model-loading failure at startup exits the process.

## Aggregation and compatibility

Each task produces float32 `z = yes_logit - no_logit`. Aggregation uses Python double-precision floats:

- Choice: `softmax(z / T_choice)`. Ties select the first candidate in request order. A single candidate receives probability and confidence 1 by protocol convention.
- Score: `softmax(z / T_score)`, followed by the expected level index in `[0, K-1]`. The legend contains the original rendered level descriptions.
- Noul: independent `sigmoid(a*z+b)`. Noul questions are not normalized against each other and do not return confidence.
- Choice/Score confidence: `1 - H(p)/ln(K)`. This measures distribution concentration, not correctness, and does not reproduce Jev's unpublished formula.

Defaults are `T_choice=T_score=a=1,b=0`, with `calibration=none`. CLI temperature and Noul parameters are configurable; non-default settings are labeled `manual`. Version 0.1 does not load fitted calibration files or fit parameters automatically.

Softmax creates a distribution within the supplied candidate set, so it still selects a winner when every candidate is unsuitable. The service does not automatically add an “other” option, abstain, or choose business thresholds. Define explicit criteria/options and downstream policies as needed. For Noul, whether missing evidence should count as false belongs in the task criteria.

`usage.input_tokens` sums the actual non-padding encoder and decoder token counts for every logical pair. Repeated document references still count on cache hits, so enabling caching does not change usage. `output_tokens=0` because no answer text is generated. This is a measure of local logical work, not hosted Jev billing. The `kalm` extension holds template, calibration, and cache metadata.

The service supports the core fields and three answer types. It does not promise full compatibility with official SDKs, weights, model quality, probability calibration, or hosted-service behavior. KaLM-Jev is based on KaLM-Reranker and is not officially affiliated with TypeSafe. The three tested model cards declare Apache-2.0; consult the original publishers for applicable checkpoint and T5Gemma/Gemma terms. This repository does not redistribute weights, tokenizers, or native model source files.

## Tests and benchmarks

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

The cache benchmark covers Choice with 3/32/128 candidates, Score with 5 levels, and Noul with 8 rules. It fixes model, dtype, batch size, and rules, warms up computation, then measures off/cold/warm paths. Warm requests use different states from the cache-filling request. Results include p50/p95, throughput, directly measured encoding/scoring time, actual encoding counts, cache bytes, peak allocated CUDA memory, token lengths, and raw latency samples. Cold mode clears the cache before each request. Warm mode asserts zero encoder calls; an additional check verifies that adding one candidate encodes only one new document.

The full GPU suite requires all three model directories. Scripts default to the active environment's `python` and `./models`; override these with `KALM_PYTHON` and `KALM_MODEL_ROOT`. Scripts do not automatically set NVIDIA driver environment variables.

Sanitized measurements and the test report are included in [results/](results/README.md) and can be committed to GitHub. Numerical results are unchanged; only host paths and file timestamps were removed. Test scripts also write to `results/`, and the summary generator updates `results/README.md`; rerunning the suite replaces the corresponding recorded files, so review the changes before committing. Public reports use `public_identity` to retain repository identifiers, file sizes, and source/configuration/tokenizer hashes. Internal cache identities remain complete. Raw logs and JUnit XML are Git-ignored. Official example provenance is documented in [examples/README.md](examples/README.md).


## Citation

If you use KaLM-Jev or the underlying KaLM-Reranker models in your research, please cite:

```bibtex
@misc{zhao2026kalmrerankerv1,
      title={KaLM-Reranker-V1: Fast but Not Late Interaction for Compressed Document Reranking},
      author={Xinping Zhao and Jiaxin Xu and Ziqi Dai and Xin Zhang and Shouzheng Huang and Danyu Tang and Xinshuo Hu and Meishan Zhang and Baotian Hu and Min Zhang},
      year={2026},
      eprint={2606.22807},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2606.22807},
}
```
