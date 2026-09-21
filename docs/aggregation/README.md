# Aggregation and compatibility

Each task produces float32 `z = yes_logit - no_logit`. Aggregation uses Python double-precision floats:

- Choice: `softmax(z / T_choice)`. Ties select the first candidate in request order. A single candidate receives probability and confidence 1 by protocol convention.
- Score: `softmax(z / T_score)`, followed by the expected level index in `[0, K-1]`. The legend contains the original rendered level descriptions.
- Noul with criteria: `sigmoid(a*(z_true-z_false)+b)`. With default `a=1,b=0`, this equals `softmax([z_true,z_false])[0]`. Each question is normalized over its own true/false candidates; different Noul questions are independent and do not return confidence. Without criteria, Noul uses `sigmoid(a*z_state+b)` for its single state Document.
- Choice/Score confidence: `1 - H(p)/ln(K)`. This measures distribution concentration, not correctness, and does not reproduce Jev's unpublished formula.

Defaults are `T_choice=T_score=a=1,b=0`, with `calibration=none`. CLI temperature and Noul parameters are configurable; non-default settings are labeled `manual`. Version 0.1 does not load fitted calibration files or fit parameters automatically.

Softmax creates a distribution within the supplied candidate set, so it still selects a winner when every candidate is unsuitable. The service does not automatically add an “other” option, abstain, or choose business thresholds. Define explicit criteria/options and downstream policies as needed. For Noul, whether missing evidence should count as false belongs in the task criteria.

`usage.input_tokens` sums the actual non-padding encoder and decoder token counts for every logical pair. Repeated document references still count on cache hits, so enabling caching does not change usage. `output_tokens=0` because no answer text is generated. This is a measure of local logical work, not hosted Jev billing. The `kalm` extension holds template, calibration, and cache metadata.

The service supports the core fields and three answer types. It does not promise full compatibility with official SDKs, weights, model quality, probability calibration, or hosted-service behavior. KaLM-Jev is based on KaLM-Reranker and is not officially affiliated with TypeSafe. The three tested model cards declare Apache-2.0; consult the original publishers for applicable checkpoint and T5Gemma/Gemma terms. This repository does not redistribute weights, tokenizers, or native model source files.

[Back to the main README](../../README.md)
