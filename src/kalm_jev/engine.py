import threading
import time

from pydantic import ValidationError

from .aggregation import Calibration, aggregate
from .cache import DocumentCache, document_key
from .compiler import compile_request
from .schemas import JevError, Request, strict_loads
from .templates import TEMPLATE_VERSION


class Engine:
    def __init__(self, model="kalm-jev-nano", *, backend=None, cache_max_mib=256,
                 max_questions=32, max_pairs=1024, max_pending=8, calibration=None, **backend_options):
        if min(max_questions, max_pairs, max_pending) <= 0:
            raise ValueError("Request limits must be positive")
        if backend is None:
            from .backend import TransformersBackend
            backend = TransformersBackend(model=model, **backend_options)
        self.backend, self.model = backend, model
        self.cache = DocumentCache(int(cache_max_mib * 1024**2))
        self.max_questions, self.max_pairs = max_questions, max_pairs
        self.calibration = calibration or Calibration()
        self._lock = threading.Lock()
        self._slots = threading.BoundedSemaphore(max_pending)

    def evaluate(self, request):
        return self.evaluate_detailed(request)[0]

    def evaluate_detailed(self, request):
        """Returns (response, diagnostics); diagnostics are for local validation/benchmarks."""
        if isinstance(request, (str, bytes)):
            request = strict_loads(request)
        try:
            if isinstance(request, Request):
                request = request.model_dump(exclude_unset=True)
            request = Request.model_validate(request)
        except ValidationError as exc:
            error = exc.errors()[0]
            raise JevError(422, error["msg"], ".".join(map(str, error["loc"])) or None) from exc
        if request.model is not None and request.model != self.model:
            raise JevError(400, f"Only {self.model} is loaded", "model", "model_error")
        if len(request.questions) > self.max_questions:
            raise JevError(422, "Too many questions", "questions")
        tasks = compile_request(request)
        if len(tasks) > self.max_pairs:
            raise JevError(422, "Too many point-wise tasks", "questions")
        if not self._slots.acquire(blocking=False):
            raise JevError(429, "Inference queue is full", kind="queue_full")
        try:
            with self._lock:
                return self._evaluate(request, tasks)
        except MemoryError as exc:
            raise JevError(503, "Device or host memory exhausted", kind="model_unavailable") from exc
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise JevError(503, "Device memory exhausted", kind="model_unavailable") from exc
            raise
        finally:
            self._slots.release()

    def clear_cache(self):
        with self._lock:
            self.cache.clear()

    def _evaluate(self, request, tasks):
        backend = self.backend
        doc_rows, decoder_rows, usage = backend.prepare(tasks)
        keys = {doc: document_key(backend.identity, doc) for doc in doc_rows}
        representations, misses = {}, []
        for doc, key in keys.items():
            value = self.cache.get(key)
            if value is None:
                misses.append(doc)
            else:
                representations[doc] = value
        encoded_before, calls_before = backend.encoded_documents, backend.encoder_calls
        backend.synchronize()
        start = time.perf_counter()
        for offset in range(0, len(misses), backend.batch_size):
            docs = misses[offset:offset + backend.batch_size]
            encoded = backend.encode_documents([doc_rows[doc] for doc in docs])
            for doc, value in zip(docs, encoded):
                representations[doc] = value
                self.cache.put(keys[doc], value)
        backend.synchronize()
        encoder_seconds = time.perf_counter() - start
        start = time.perf_counter()
        margins = []
        for offset in range(0, len(tasks), backend.batch_size):
            batch = tasks[offset:offset + backend.batch_size]
            margins.extend(backend.score_encoded(decoder_rows[offset:offset + backend.batch_size],
                                                 [representations[task.document] for task in batch]))
        backend.synchronize()
        score_seconds = time.perf_counter() - start
        stats = {"enabled": self.cache.max_bytes > 0, "unique_documents": len(keys),
                 "hit_documents": len(keys) - len(misses), "miss_documents": len(misses),
                 "encoded_documents": backend.encoded_documents - encoded_before}
        response = {"model": self.model, "answers": aggregate(request, tasks, margins, self.calibration),
                    "usage": {"input_tokens": usage, "output_tokens": 0},
                    "kalm": {"backend": "transformers", "template_version": TEMPLATE_VERSION,
                             "confidence_method": "normalized_entropy_v1", "calibration": self.calibration.status,
                             "cache": stats}}
        diagnostics = {"margins": margins, "encoder_seconds": encoder_seconds,
                       "score_seconds": score_seconds, "encoder_calls": backend.encoder_calls - calls_before,
                       "cache_bytes": self.cache.bytes}
        return response, diagnostics
