import argparse
import logging

from fastapi import FastAPI, Request as HTTPRequest
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException

from .aggregation import Calibration
from .engine import Engine
from .schemas import JevError, strict_loads


def create_app(engine, max_body_bytes=2 * 1024**2):
    app = FastAPI(title="KaLM-Jev", version="0.1.0")

    @app.exception_handler(JevError)
    async def jev_error(request, exc):
        return JSONResponse(exc.body(), status_code=exc.status)

    @app.exception_handler(HTTPException)
    async def http_error(request, exc):
        return JSONResponse(JevError(exc.status_code, str(exc.detail), kind="http_error").body(), status_code=exc.status_code)

    @app.exception_handler(Exception)
    async def internal_error(request, exc):
        logging.getLogger(__name__).exception("Inference failed", exc_info=exc)
        return JSONResponse(JevError(500, "Internal server error", kind="internal_error").body(), status_code=500)

    @app.get("/health")
    def health():
        if engine is None:
            raise JevError(503, "Model unavailable", kind="model_unavailable")
        return {"status": "ok", "model": engine.model, "backend": "transformers"}

    @app.post("/v1/systemone")
    async def evaluate(request: HTTPRequest):
        if engine is None:
            raise JevError(503, "Model unavailable", kind="model_unavailable")
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > max_body_bytes:
                raise JevError(422, "Request body too large")
        return await run_in_threadpool(engine.evaluate, strict_loads(body))

    return app


def main():
    parser = argparse.ArgumentParser(description="Single-process KaLM-Jev server")
    parser.add_argument("command", choices=["serve"])
    parser.add_argument("--model", default="kalm-jev-nano", choices=["kalm-jev-nano", "kalm-jev-small", "kalm-jev-large"])
    parser.add_argument("--model-path")
    parser.add_argument("--revision")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--query-max-length", type=int, default=512)
    parser.add_argument("--document-max-length", type=int, default=1024)
    parser.add_argument("--decoder-max-length", type=int, default=2048)
    parser.add_argument("--chunk-size", type=lambda x: None if x == "none" else int(x), default=4)
    parser.add_argument("--cache-max-mib", type=float, default=256)
    parser.add_argument("--max-questions", type=int, default=32)
    parser.add_argument("--max-pairs", type=int, default=1024)
    parser.add_argument("--max-pending", type=int, default=8)
    parser.add_argument("--max-body-bytes", type=int, default=2 * 1024**2)
    parser.add_argument("--choice-temperature", type=float, default=1)
    parser.add_argument("--score-temperature", type=float, default=1)
    parser.add_argument("--noul-a", type=float, default=1)
    parser.add_argument("--noul-b", type=float, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    options = vars(parser.parse_args())
    options.pop("command")
    host, port, max_body = options.pop("host"), options.pop("port"), options.pop("max_body_bytes")
    calibration = Calibration(**{key: options.pop(key) for key in ("choice_temperature", "score_temperature", "noul_a", "noul_b")})
    engine = Engine(**options, calibration=calibration)
    import uvicorn
    uvicorn.run(create_app(engine, max_body), host=host, port=port, workers=1)


if __name__ == "__main__":
    main()
