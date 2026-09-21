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
    lines += ["", "## Semantic Evaluation", "",
              "| Model | Choice accuracy | Score MAE | Human-agent Noul accuracy | Repeat-contact accuracy |",
              "|---|---:|---:|---:|---:|"]
    for size in ("nano", "small", "large"):
        m = read(f"kalm-jev-{size}-bfloat16-semantics.json")["metrics"]
        lines.append(f'| {size} | {m["choice_accuracy"]:.2%} | {m["score_mae"]:.6f} | '
                     f'{m["noul_accuracy"]:.2%} | {m["repeat_contact_accuracy"]:.2%} |')
    benchmark = read("kalm-jev-nano-bfloat16-benchmark.json")
    lines += ["", "## Nano Cache Benchmark", "",
              f'Samples per mode: {benchmark["samples_per_mode"]}; batch_size={benchmark["batch_size"]}.',
              "", (args.results_dir / "kalm-jev-nano-bfloat16-benchmark.md").read_text().strip(),
              "", "These are measured results; runs with different devices, query lengths, or batch settings are not directly comparable."]
    assert read("http-smoke.json")["passed"]
    lines += ["", "Real CLI HTTP smoke passed. Run unit tests separately with python -m pytest -q.",
              "", "Results do not establish Jev-equivalent quality or calibrated probabilities. Public identities omit host directories and modification times."]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    print("Generated report:", output.name)


if __name__ == "__main__":
    main()
