"""Summarize a completed run without inserting host-specific metadata."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.results_dir / "README.md"

    def read(name):
        return json.loads((args.results_dir / name).read_text())

    lines = ["# KaLM-Jev Test Results", "",
             "Values are read from the selected results directory; device and dtype come from run records. Host paths are omitted.",
             "", "Current template: `v3-noul-state-document`. Original Choice/Score adapters and MEP 4x are retained. Noul with criteria scores two separate Documents; omitted criteria uses state as Query and one Document with a direct yes/no sigmoid. No calibration was fitted.",
             "", "## Native Inference Alignment", "",
             "| Model | dtype | Device | Max margin difference | Max probability difference | Warm encoder calls |",
             "|---|---|---|---:|---:|---:|"]
    for size in ("nano", "small", "large"):
        for dtype in ("float32", "bfloat16"):
            report = read(f"kalm-jev-{size}-{dtype}-validation.json")
            assert report["passed"]
            cases = list(report["cases"].values())
            lines.append(f'| {size} | {dtype} | {report["device_name"]} | '
                         f'{max(x["native_backend_max_margin_diff"] for x in cases):.8g} | '
                         f'{max(x["native_backend_max_probability_diff"] for x in cases):.8g} | '
                         f'{sum(x["warm_actual_encoder_calls"] for x in cases)} |')
    lines += ["", "## Example Outputs (BF16)", "",
              "| Model | Choice | P(returns) | Safari Score | Human-agent Noul | Repeat-contact Noul |",
              "|---|---|---:|---:|---:|---:|"]
    for size in ("nano", "small", "large"):
        cases = read(f"kalm-jev-{size}-bfloat16-validation.json")["cases"]
        choice = next(iter(cases["choice"]["off"]["answers"].values()))
        score = next(iter(cases["score"]["off"]["answers"].values()))
        noul = cases["noul"]["off"]["answers"]
        lines.append(f'| {size} | {choice["choice"]} | {choice["probabilities"]["returns"]:.6f} | '
                     f'{score["score"]:.6f} | {noul["is_human_escalation"]["noul"]:.6f} | '
                     f'{noul["is_repeat_contact"]["noul"]:.6f} |')
    large = read("kalm-jev-large-bfloat16-validation.json")
    example = {"model": large["model"], "identity": large["identity"],
               "configuration": {"dtype": "bfloat16", "batch_size": 4, "chunk_size": 4,
                                 "query_max_length": 512, "document_max_length": 1024, "decoder_max_length": 2048},
               "request": json.loads((ROOT / "examples/noul.json").read_text()),
               "response": large["cases"]["noul"]["off"],
               "margins": large["cases"]["noul"]["backend_margins"],
               "source": "kalm-jev-large-bfloat16-validation.json: cases.noul.off"}
    (args.results_dir / "kalm-jev-large-noul-candidates-example.json").write_text(json.dumps(example, ensure_ascii=False, indent=2) + "\n")
    lines += ["", "## Semantic Evaluation", "",
              "36 hand-reviewed bilingual smoke cases: 12 per primitive plus 12 repeat-contact judgments. Noul threshold is 0.5. This is a small smoke set, not the JevBench leaderboard.", "",
              "| Model | Choice accuracy | Score MAE | Human-agent Noul accuracy | Repeat-contact accuracy |",
              "|---|---:|---:|---:|---:|"]
    for size in ("nano", "small", "large"):
        m = read(f"kalm-jev-{size}-bfloat16-semantics.json")["metrics"]
        lines.append(f'| {size} | {m["choice_accuracy"]:.2%} | {m["score_mae"]:.6f} | '
                     f'{m["noul_accuracy"]:.2%} | {m["repeat_contact_accuracy"]:.2%} |')
    lines += ["", "### Omitted-criteria Noul Breakdown", "",
              "| Model | True positives | False positives | True negatives | False negatives |",
              "|---|---:|---:|---:|---:|"]
    for size in ("nano", "small", "large"):
        cases = read(f"kalm-jev-{size}-bfloat16-semantics.json")["cases"]
        counts = {(True, True): 0, (False, True): 0, (False, False): 0, (True, False): 0}
        for case in cases:
            if case["id"].startswith("noul-"):
                pred = case["response"]["answers"]["is_human_escalation"]["noul"] >= .5
                counts[bool(case["expected"]), pred] += 1
        lines.append(f'| {size} | ' + ' | '.join(str(v) for v in counts.values()) + ' |')
    lines += ["", "A high value on one affirmative example does not establish discrimination: inspect false positives on explicit negation and unrelated statements. Raw responses are retained in the semantics JSON files."]
    benchmark = read("kalm-jev-nano-bfloat16-benchmark.json")
    lines += ["", "## Nano Cache Benchmark", "",
              f'Samples per mode: {benchmark["samples_per_mode"]}; batch_size={benchmark["batch_size"]}.',
              "", (args.results_dir / "kalm-jev-nano-bfloat16-benchmark.md").read_text().strip(),
              "", "Warm requests have distinct states. Fixed criterion Documents are reused; for Noul without criteria each new state needs one encoder call, shared by the eight questions. Its warm mode does not mean an encoder cache hit.",
              "", "These are measured results; runs with different devices, query lengths, or batch settings are not directly comparable."]
    assert read("http-smoke.json")["passed"]
    lines += ["", "Real CLI HTTP smoke passed. Run unit tests separately with python -m pytest -q.",
              "", "Results do not establish Jev-equivalent quality or calibrated probabilities. Public identities omit host directories and modification times."]
    regression_path = args.results_dir / "noul-v3-nano-regression.json"
    if regression_path.exists():
        regression = read(regression_path.name)
        lines += ["", "## Public JevBench Regression", "",
                  f'Nano: {regression["correct"]}/{regression["n"]} correct ({regression["correct"]/regression["n"]:.2%}); '
                  f'{regression["prediction_mismatches"]} prediction changes against the selected candidate-adapter baseline; '
                  f'maximum probability difference {regression["max_probability_diff"]:.8g}. '
                  'All 74 Noul items supply both criteria, so this checks the supplied-criteria path, not the new missing-criteria behavior. Query limit 8192, decoder limit 9216, BF16, MEP 4x.']
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    print("Generated report:", output.name)


if __name__ == "__main__":
    main()
