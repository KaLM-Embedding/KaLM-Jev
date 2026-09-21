# Document cache and limits

The cache stores actual pooled hidden states and valid masks on the model device. Tensors are cloned to their effective length to avoid retaining padded batch storage. Scoring builds fresh padded tensors without modifying cached representations.

SHA256 cache keys include the checkpoint path and revision, local file size/mtime fingerprint, configuration/tokenizer/source hashes, dtype, Transformers/PyTorch versions, document template version, length and pooling settings, and document text. Replacing weights or changing encoder settings in a running engine is unsupported; create a new Engine instead.

The default LRU tensor budget is 256 MiB. Use `--cache-max-mib 0` to disable caching. A document larger than the budget remains usable for the current request but is not cached. The budget covers hidden-state and mask tensor bytes, not allocator-reserved memory, Python object overhead, or inference working memory. References held by an active request may exceed the cache budget, so request limits also matter.

Documents are deduplicated within each request. `unique_documents = hit_documents + miss_documents`; first-time duplicates are not counted as cross-request hits. `encoded_documents` counts actual encoding events. Changing state reuses fixed candidate Documents but reruns decoder scoring. For Noul without criteria, changing state changes the Document and requires encoding it unless already cached. Changing Choice instructions reuses the document representation. Reordering Score levels reuses representations and aggregates against the new indices. Changing Noul instructions reuses criterion Documents but reruns scoring; changing one criterion encodes only the new Document. Identical criterion texts can share encoder cache entries across questions.

Requests default to at most 32 questions and 1,024 point-wise pairs; configure these with `--max-questions` and `--max-pairs`. `--max-pending 8` bounds requests running or waiting inside the engine; excess requests receive HTTP 429. A lock serializes inference. HTTP bodies default to a 2 MiB limit, configurable with `--max-body-bytes`.

Noul consumes two point-wise pairs with criteria and one without criteria, even when
Documents are cached or shared. All pairs contribute to pair-limit accounting and logical input-token usage.

Native defaults are 512 query tokens and 1,024 document tokens including the document prefix. The complete decoder template is separately limited to 2,048 tokens, including padding to a multiple of eight, to bound instruction length. Configure these with `--query-max-length`, `--document-max-length`, and `--decoder-max-length`. Overlong requests receive HTTP 422 before encoding; inputs are not silently truncated. A checkpoint's positional capacity does not imply that this project has validated that context length. Use `--chunk-size 4` for default pooling or `--chunk-size none` to disable it.

Validation failures return 422, model mismatches 400, unavailable models or device memory exhaustion 503, and other inference failures 500. HTTP responses omit stack traces. A model-loading failure at startup exits the process.

[Back to the main README](../../README.md)
