# KaLM-Jev

A local Choice / Score / Noul service built on KaLM-Reranker-V1-R2. It uses PyTorch and Transformers, runs one model per process, and returns structured judgments without generating answer text.

The underlying reranker is described in [KaLM-Reranker-V1: Fast but Not Late Interaction for Compressed Document Reranking](https://arxiv.org/abs/2606.22807).

- **Demo:** Try KaLM-Jev in the [Hugging Face Space](https://huggingface.co/spaces/Yuki131/KaLM-Jev).
- **Models:** Explore the available checkpoints in the [Lychee KaLM Reranker collection](https://huggingface.co/collections/KaLM-Embedding/lychee-kalm-reranker).

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

The following examples use `kalm-jev-large` BF16 with default, uncalibrated aggregation. Inputs are the official Jev examples preserved in [examples/](examples/README.md); outputs are KaLM-Jev measurements, not hosted Jev responses. The JSON outputs below show only `answers`, rounded to six decimal places. Full-precision outputs and execution metadata are available in the [Choice/Score validation results](results/kalm-jev-large-bfloat16-validation.json) and the [Noul example result](results/kalm-jev-large-noul-candidates-example.json).

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
    "score": 1.08054,
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

Noul questions with criteria score true and false as separate Documents and normalize their matching scores. Without criteria, serialized state is the sole Document and its native yes/no margin is converted directly with sigmoid.

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
    "noul": 0.99883
  },
  "is_repeat_contact": {
    "type": "noul",
    "noul": 0.974043
  }
}
```

The repeat-contact result agrees with “asked three times.” The human-escalation question omits criteria and scores the state as its sole Document. Its value reflects the native yes/no margin for that input. Each question's value is independent; the two values do not sum to one. The API returns numeric values rather than Boolean decisions.

These examples illustrate the API and its measured behavior, not overall accuracy. Outputs are not calibrated correctness probabilities. Evaluation details are available in the [test report](results/README.md).

## Inputs and model execution

Requests support `state`, `questions`, and optional `model`. An array-valued state is one complete business state, not a batch of independent samples. Content strings retain their whitespace. Objects and arrays use compact, Unicode-preserving JSON with sorted object keys. Question and candidate order are preserved; question IDs are not sent to the model.

| Primitive | Document | Task-specific instruction | Query |
|---|---|---|---|
| Choice | `option_id: description`, or only the ID for a null description | Original instructions + fixed Choice adapter | Serialized state |
| Score | The level description, without its index | Original instructions + fixed Score adapter | Serialized state |
| Noul | Separate true/false criteria when supplied; serialized state as one Document when omitted | Original instructions + candidate-criterion Noul adapter | Serialized state |

Unknown fields, duplicate JSON keys, NaN/Infinity, implicit type coercion, empty IDs, and invalid candidate counts are rejected. If Noul criteria are supplied, they must contain at least one of `true` or `false`; null values are not accepted. Errors use `{"error":{"type":"...","message":"...","field":"..."}}`, with `field` omitted when not applicable.

When Noul criteria are supplied, two candidates are scored in true/false order.
Supplied criteria retain their original text. A missing `true` side uses
`The answer to the question is yes.`; a missing `false` side uses
`The answer to the question is no.`. When criteria are omitted entirely,
serialized state is used as both Query and the sole Document. The native yes/no
margin is then converted directly with sigmoid. The same task-specific adapter
is used in both paths. Template version: `v3-noul-state-document`.

The task-specific instruction is the original question instructions, two newlines,
and this adapter:

```text
Given the Query, evaluate whether the candidate criterion in the Document correctly describes the answer to the question above. Answer yes if this candidate criterion is satisfied, otherwise no.
```

For supplied criteria, the native `yes` readout means that the current candidate is satisfied. A
stronger match for the false candidate lowers the returned business `noul` value.
Choice and Score adapters and the default 4x MEP compression remain unchanged.

The adapter loads the original `kalm_reranker.py` and `kalm_reranker_utils.py` from the checkpoint directory. These are executable Python files, so use trusted model repositories. Weights load through Transformers' built-in `T5Gemma2ForConditionalGeneration`; AutoModel does not require `trust_remote_code`. The two original scripts have matching SHA256 hashes across the three tested checkpoints.

The original `<Document>: ` prefix, system instruction, `<bos>/<start_of_turn>/<end_of_turn>` tokens, query tokenization/decoding, and yes/no readout at the final non-padding decoder token are preserved. The native implementation verifies that yes and no each occupy one token. The model runs in `.eval()` and `torch.inference_mode()`. By default, encoder states are mean-pooled in groups of four tokens using the attention mask.

The adapter supplies `encoder_outputs=BaseModelOutput(...)` to bypass document encoding on cache hits. Decoder and language-model-head execution follow the original model, with unused decoder KV caching disabled. It does not call `generate()` or replace interaction scoring with embedding similarity. Each task gets its own instruction, including mixed-question requests.

## Document cache and limits

A 256 MiB LRU cache reuses encoded Documents. Noul without criteria uses state as its Document, so a new state requires encoding. Request, token, and cache limits are configurable.

See [cache behavior and limits](docs/cache/README.md).

## Aggregation and compatibility

Choice uses softmax; Score returns the expected level index. Noul uses the difference between true/false candidate margins when criteria are supplied, or a single state margin when omitted. Probabilities are uncalibrated by default.

See [aggregation formulas and compatibility](docs/aggregation/README.md).

## Tests and benchmarks

Run `python -m pytest -q` for unit tests. GPU checks cover native inference alignment, bilingual semantic cases, cache performance, and HTTP behavior.

See [test commands and methodology](docs/testing/README.md) and [measured results](results/README.md).

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
