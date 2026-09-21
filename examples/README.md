# Example Provenance

Sources checked on 2026-09-21:

- `choice.json`: the shoe-size exchange example from the [official Jev Choice documentation](https://docs.typesafe.ai/primitives/choice).
- `score.json`: the Safari export failure example from the [official Jev Score documentation](https://docs.typesafe.ai/primitives/score).
- `noul.json`: the human-agent and repeat-contact example from the [official Jev Noul documentation](https://docs.typesafe.ai/primitives/noul), covering questions with and without criteria.
- `mixed.json`: a combined example created for this project. Its business state is newly written, not a fourth official example.

The first three preserve the official state, instructions, criteria, and question IDs. The hosted service's `jev-latest` model field is omitted so the currently loaded KaLM alias is used. They are real-model smoke tests and do not require reproducing Jev's documented probabilities.

Measured outputs are in `../results/*-validation.json`; new runs update the corresponding files in `../results/`. Official response values were not copied into this project's measured results.
