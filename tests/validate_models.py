"""Real checkpoints only: native margin parity, official examples and cache events."""
import argparse
from collections import defaultdict
import copy
import json
from pathlib import Path
import platform

import torch
import transformers

from kalm_jev import Engine, Request
from kalm_jev.aggregation import Calibration, aggregate
from kalm_jev.backend import load_native
from kalm_jev.compiler import compile_request
from kalm_jev.schemas import JevError

ROOT = Path(__file__).resolve().parents[1]


def native_margins(backend, tasks, batch_size):
    # Execute the original _predict_batch, capturing raw logits before its softmax.
    native, _ = load_native(Path(backend.identity["path"]))
    original = native.extract_yes_no_logits
    captured = []

    def capture(*args, **kwargs):
        values = original(*args, **kwargs)
        captured.extend((values[:, 0] - values[:, 1]).cpu().tolist())
        return values

    groups = defaultdict(list)
    for i, task in enumerate(tasks):
        groups[task.instruction].append((i, task))
    result = [0.] * len(tasks)
    native.extract_yes_no_logits = capture
    try:
        for instruction, group in groups.items():
            for start in range(0, len(group), batch_size):
                batch = group[start:start + batch_size]
                captured.clear()
                backend.reranker._predict_batch([(t.query, t.document) for _, t in batch], instruction)
                for (i, _), value in zip(batch, captured):
                    result[i] = value
    finally:
        native.extract_yes_no_logits = original
    return result


def delta(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


def probabilities(response):
    result = []
    for answer in response["answers"].values():
        result.extend(answer["probabilities"].values() if "probabilities" in answer else [answer["noul"]])
    return result


def run(args):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    engine = Engine(model=args.model, model_path=args.model_path, device=args.device,
                    dtype=args.dtype, batch_size=args.batch_size)
    backend = engine.backend
    report = {"model": args.model, "identity": backend.public_identity, "dtype": args.dtype,
              "device": str(backend.device), "device_name": torch.cuda.get_device_name() if args.device == "cuda" else platform.processor(),
              "torch": torch.__version__, "transformers": transformers.__version__, "cases": {}}
    encoder_events = []
    hook = backend.utils.get_encoder(backend.model).register_forward_hook(lambda *x: encoder_events.append(1))
    try:
        for name in ("choice", "score", "noul", "mixed"):
            req = json.loads((ROOT / "examples" / f"{name}.json").read_text())
            tasks = compile_request(Request.model_validate(req))
            reference = native_margins(backend, tasks, args.batch_size)
            single_reference = native_margins(backend, tasks, 1)
            # Same native model's own batch variation is recorded separately.
            native_variation = delta(reference, single_reference)
            engine.cache.max_bytes = 0
            off, off_detail = engine.evaluate_detailed(req)
            engine.cache.max_bytes = 256 * 1024**2
            engine.clear_cache()
            cold, cold_detail = engine.evaluate_detailed(req)
            events_before = len(encoder_events)
            warm, warm_detail = engine.evaluate_detailed(req)
            actual_warm_calls = len(encoder_events) - events_before
            assert actual_warm_calls == warm_detail["encoder_calls"] == 0
            assert warm["kalm"]["cache"]["encoded_documents"] == 0
            assert off["usage"] == cold["usage"] == warm["usage"]
            assert off["answers"] == cold["answers"] == warm["answers"]
            assert off_detail["margins"] == cold_detail["margins"] == warm_detail["margins"]
            previous_batch = backend.batch_size
            backend.batch_size = 1
            single, single_detail = engine.evaluate_detailed(req)
            backend.batch_size = previous_batch
            ref_response = {"answers": aggregate(Request.model_validate(req), tasks, reference, Calibration())}
            diff = delta(reference, off_detail["margins"])
            # FP32 acceptance is checked element-wise. BF16 bound follows measured
            # native batch variation plus one logit quantization step (reported).
            if args.dtype == "float32":
                torch.testing.assert_close(torch.tensor(reference), torch.tensor(off_detail["margins"]), atol=1e-5, rtol=1e-4)
            else:
                assert diff <= max(0.0625, native_variation * 2 + 0.0625), (name, diff, native_variation)
            changed = copy.deepcopy(req)
            changed["state"] = "No problem occurred. This is my first message. I do not need a human."
            events_before = len(encoder_events)
            changed_result, changed_detail = engine.evaluate_detailed(changed)
            changed_documents = {t.document for t in compile_request(Request.model_validate(changed))}
            new_documents = changed_documents - {t.document for t in tasks}
            assert changed_result["kalm"]["cache"]["encoded_documents"] == len(new_documents)
            assert len(encoder_events) - events_before == changed_detail["encoder_calls"]
            assert bool(changed_detail["encoder_calls"]) == bool(new_documents)
            assert changed_detail["margins"] != warm_detail["margins"]
            report["cases"][name] = {
                "native_margins": reference, "native_single_margins": single_reference,
                "native_batch_variation": native_variation, "backend_margins": off_detail["margins"],
                "native_backend_max_margin_diff": diff,
                "native_backend_max_probability_diff": delta(probabilities(ref_response), probabilities(off)),
                "off_cold_warm_max_margin_diff": max(delta(off_detail["margins"], cold_detail["margins"]), delta(cold_detail["margins"], warm_detail["margins"])),
                "batch1_backend_margin_diff": delta(single_detail["margins"], warm_detail["margins"]),
                "batch1_backend_probability_diff": delta(probabilities(single), probabilities(warm)),
                "choice_flips": [key for key, answer in warm["answers"].items() if answer["type"] == "choice" and answer["choice"] != single["answers"][key]["choice"]],
                "off": off, "cold": cold, "warm": warm, "warm_actual_encoder_calls": actual_warm_calls,
                "changed_state": changed_result, "changed_state_margins": changed_detail["margins"],
            }
            print(args.model, name, "margin difference", diff, "native batch variation", native_variation, flush=True)
        # Varied document lengths and right padding, including partial pooling chunks.
        req = {"state": "A short message", "questions": {"lengths": {"type": "score", "instructions": "How long is the message?",
               "criteria": ["Short", "A somewhat longer message with several words", "A long message. " * 50]}}}
        tasks = compile_request(Request.model_validate(req))
        ref = native_margins(backend, tasks, args.batch_size)
        result, detail = engine.evaluate_detailed(req)
        if args.dtype == "float32":
            torch.testing.assert_close(torch.tensor(ref), torch.tensor(detail["margins"]), atol=1e-5, rtol=1e-4)
        report["padding_max_margin_diff"] = delta(ref, detail["margins"])
        # Reordering levels must reuse representations and reflect the new indices.
        req["questions"]["lengths"]["criteria"].reverse()
        reversed_result, reversed_detail = engine.evaluate_detailed(req)
        assert reversed_detail["encoder_calls"] == 0
        assert abs(reversed_result["answers"]["lengths"]["score"] - (2 - result["answers"]["lengths"]["score"])) < .03
        req["questions"]["lengths"]["instructions"] = "Does the description fit this message?"
        assert engine.evaluate(req)["kalm"]["cache"]["encoded_documents"] == 0
        req["questions"]["lengths"]["criteria"][0] = "A new unique document"
        assert engine.evaluate(req)["kalm"]["cache"]["encoded_documents"] == 1
        too_long = copy.deepcopy(req)
        too_long["state"] = "hello " * 600
        before = len(encoder_events)
        try:
            engine.evaluate(too_long)
            raise AssertionError("Overlong input accepted")
        except JevError as exc:
            assert exc.status == 422 and len(encoder_events) == before
        report["passed"] = True
    finally:
        hook.remove()
    output = ROOT / "results" / f"{args.model}-{args.dtype}-validation.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("Saved", output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--batch-size", type=int, default=4)
    run(parser.parse_args())
