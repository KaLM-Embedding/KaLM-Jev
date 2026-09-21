"""Start the actual CLI on loopback, exercise official requests, then stop it."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import httpx

ROOT = Path(__file__).resolve().parents[1]


def run():
    model_root = Path(os.environ.get("KALM_MODEL_ROOT", ROOT / "models"))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    log_path = ROOT / "results/http-smoke.log"
    log_path.parent.mkdir(exist_ok=True)
    with log_path.open("w") as log:
        process = subprocess.Popen([sys.executable, "-m", "kalm_jev.server", "serve",
            "--model-path", str(model_root / "KaLM-Reranker-V1-Nano-R2"), "--device", "cuda",
            "--dtype", "bfloat16", "--port", str(port)], stdout=log, stderr=subprocess.STDOUT)
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=60, trust_env=False) as client:
                deadline = time.monotonic() + 180
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f"Server exited; see {log_path}")
                    try:
                        health = client.get("/health")
                        if health.status_code == 200:
                            break
                    except httpx.ConnectError:
                        pass
                    if time.monotonic() > deadline:
                        raise TimeoutError("Server startup timed out")
                    time.sleep(.5)
                results = {"health": health.json(), "requests": {}}
                for name in ("choice", "score", "noul", "mixed"):
                    body = json.loads((ROOT / "examples" / f"{name}.json").read_text())
                    cold = client.post("/v1/systemone", json=body)
                    warm = client.post("/v1/systemone", json=body)
                    assert cold.status_code == warm.status_code == 200
                    cold, warm = cold.json(), warm.json()
                    assert cold["answers"] == warm["answers"]
                    assert warm["kalm"]["cache"]["encoded_documents"] == 0
                    results["requests"][name] = warm
                assert client.post("/v1/systemone", content='{"state":1,"state":2}').status_code == 422
                results["passed"] = True
                (ROOT / "results/http-smoke.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
                print("Real CLI HTTP smoke passed", flush=True)
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    run()
