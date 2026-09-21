"""36 hand-authored bilingual smoke cases, not a calibration/leaderboard dataset."""
import argparse
import json
from pathlib import Path

import torch

from kalm_jev import Engine

ROOT = Path(__file__).resolve().parents[1]


def run(args):
    torch.set_num_threads(4)
    engine = Engine(model=args.model, model_path=args.model_path, device=args.device, dtype=args.dtype)
    cases = json.loads((ROOT / "tests/semantic_cases.json").read_text())
    records, choice, score, noul, repeat = [], [], [], [], []
    for case in cases:
        result = engine.evaluate(case["request"])
        answer = result["answers"][case["question_id"]]
        if case["kind"] == "choice":
            choice.append(int(answer["choice"] == case["expected"]))
        elif case["kind"] == "score":
            score.append(abs(answer["score"] - case["expected"]))
        else:
            noul.append(int((answer["noul"] >= .5) == bool(case["expected"])))
            repeat.append(int((result["answers"]["is_repeat_contact"]["noul"] >= .5) == bool(case["additional_expected"]["is_repeat_contact"])))
        records.append({"id": case["id"], "expected": case["expected"], "response": result})
    metrics = {"choice_accuracy": sum(choice) / len(choice), "choice_n": len(choice),
               "score_mae": sum(score) / len(score), "score_n": len(score),
               "noul_accuracy": sum(noul) / len(noul), "noul_n": len(noul),
               "repeat_contact_accuracy": sum(repeat) / len(repeat), "repeat_contact_n": len(repeat),
               "noul_threshold": .5}
    report = {"model": args.model, "identity": engine.backend.public_identity, "metrics": metrics, "cases": records,
              "label_policy": "Hand-reviewed task expectations; absent prior contact is false per explicit criteria. Human-agent requests require affirmative evidence. No fitting performed."}
    output = ROOT / "results" / f"{args.model}-{args.dtype}-semantics.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(args.model, metrics, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    run(parser.parse_args())
