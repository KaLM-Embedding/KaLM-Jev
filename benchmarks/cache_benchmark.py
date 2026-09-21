"""Cross-request document reuse; warm requests always use different state values."""
import argparse
import copy
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from kalm_jev import Engine, Request
from kalm_jev.compiler import compile_request

ROOT = Path(__file__).resolve().parents[1]


def workloads():
    cases = {}
    for size in (3, 32, 128):
        cases[f"choice-{size}"] = {"type": "choice", "instructions": "Which service queue matches the requested queue number?",
            "criteria": {f"queue_{i:03}": f"Handle support requests explicitly addressed to service queue {i:03}." for i in range(size)}}
    cases["score-5"] = {"type": "score", "instructions": "How severe is this incident?", "criteria": [
        "No incident; everything works normally", "Visual defect only; all features work",
        "One feature fails but a workaround exists", "A core function fails without a workaround",
        "The whole service is unavailable to every user"]}
    cases["noul-8"] = {f"rule_{i}": {"type": "noul", "instructions": f"Does the customer explicitly request service queue {i:03}?"} for i in range(8)}
    for name, q in cases.items():
        yield name, {"state": "", "questions": q if name == "noul-8" else {"decision": q}}


def run(args):
    torch.set_num_threads(4)
    engine = Engine(model=args.model, model_path=args.model_path, device=args.device,
                    dtype=args.dtype, batch_size=args.batch_size)
    report = {"model": args.model, "identity": engine.backend.public_identity, "batch_size": args.batch_size,
              "samples_per_mode": args.samples, "device_name": torch.cuda.get_device_name() if args.device == "cuda" else "CPU",
              "workloads": {}}
    for name, template in workloads():
        requests = []
        for i in range(args.samples + 2):
            req = copy.deepcopy(template)
            req["state"] = f"Ticket {i:03}: Please send my request to service queue {i % 3:03}. One feature fails but there is a workaround."
            requests.append(req)
        lengths = []
        for req in requests:
            docs, decoders, _ = engine.backend.prepare(compile_request(Request.model_validate(req)))
            lengths.append({"encoder": list(map(len, docs.values())), "decoder": list(map(len, decoders))})
        # Warm up kernels, allocator and model, independently of the measured cache mode.
        engine.evaluate(requests[-1])
        engine.evaluate(requests[-2])
        modes = {}
        for mode in ("off", "cold", "warm"):
            engine.clear_cache()
            engine.cache.max_bytes = 0 if mode == "off" else 256 * 1024**2
            if mode == "warm":
                engine.evaluate(requests[-1])
            if args.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            timings, enc_times, score_times, encoded, calls = [], [], [], [], []
            for req in requests[:args.samples]:
                if mode == "cold":
                    engine.clear_cache()
                engine.backend.synchronize()
                start = time.perf_counter()
                response, detail = engine.evaluate_detailed(req)
                engine.backend.synchronize()
                timings.append(time.perf_counter() - start)
                enc_times.append(detail["encoder_seconds"])
                score_times.append(detail["score_seconds"])
                encoded.append(response["kalm"]["cache"]["encoded_documents"])
                calls.append(detail["encoder_calls"])
                if mode == "warm":
                    assert calls[-1] == encoded[-1] == 0
            modes[mode] = {"n": len(timings), "p50_ms": statistics.median(timings) * 1000,
                "p95_ms": float(np.percentile(timings, 95)) * 1000,
                "requests_per_second": len(timings) / sum(timings),
                "encoder_p50_ms": statistics.median(enc_times) * 1000,
                "score_p50_ms": statistics.median(score_times) * 1000,
                "encoded_documents": sum(encoded), "encoder_calls": sum(calls),
                "cache_bytes": engine.cache.bytes,
                "peak_device_allocated_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
                "raw_seconds": timings}
            print(name, mode, "p50_ms", modes[mode]["p50_ms"], flush=True)
        report["workloads"][name] = {"modes": modes, "speedup": modes["off"]["p50_ms"] / modes["warm"]["p50_ms"],
                                      "token_lengths": lengths, "queries_are_distinct": len({r["state"] for r in requests}) == len(requests)}
        if name.startswith("choice"):
            extra = copy.deepcopy(requests[0])
            extra["questions"]["decision"]["criteria"]["extra"] = "A new queue for a new topic"
            assert engine.evaluate(extra)["kalm"]["cache"]["encoded_documents"] == 1
    output = ROOT / "results" / f"{args.model}-{args.dtype}-benchmark.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    lines = ["| workload | off p50/p95 ms | cold p50/p95 ms | warm p50/p95 ms | speedup |", "|---|---:|---:|---:|---:|"]
    for name, record in report["workloads"].items():
        cells = [f'{record["modes"][mode]["p50_ms"]:.2f}/{record["modes"][mode]["p95_ms"]:.2f}' for mode in ("off", "cold", "warm")]
        lines.append(f'| {name} | ' + " | ".join(cells) + f' | {record["speedup"]:.3f}x |')
    output.with_suffix(".md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="kalm-jev-nano")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--samples", type=int, default=30)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("samples must be positive")
    run(args)
