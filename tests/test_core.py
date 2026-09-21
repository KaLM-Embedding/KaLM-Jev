import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from kalm_jev import Engine, Request
from kalm_jev.aggregation import Calibration, aggregate, confidence, sigmoid, softmax
from kalm_jev.cache import DocumentCache, document_key
from kalm_jev.compiler import compile_request
from kalm_jev.schemas import JevError, strict_loads
from kalm_jev.server import create_app
from kalm_jev.templates import CHOICE_ADAPTER, SCORE_ADAPTER, NOUL_ADAPTER, render_content

ROOT = Path(__file__).resolve().parents[1]


def example(name="mixed"):
    return json.loads((ROOT / "examples" / f"{name}.json").read_text())


class FakeBackend:
    """Protocol-only test double. Never used in production or model evaluation."""
    identity = {"test": "fake"}
    batch_size = 4
    encoded_documents = 0
    encoder_calls = 0

    def synchronize(self):
        pass

    def prepare(self, tasks):
        return dict.fromkeys((t.document for t in tasks), [1]), list(range(len(tasks))), len(tasks) * 2

    def encode_documents(self, rows):
        self.encoder_calls += 1
        self.encoded_documents += len(rows)
        return [SimpleNamespace(nbytes=8) for _ in rows]

    def score_encoded(self, rows, docs):
        return [float(row) for row in rows]


def test_compiler_exact():
    request = Request.model_validate({"state": {"z": [2, "中文"], "a": 1}, "questions": {
        "secret-id": {"type": "choice", "instructions": " Q \n", "criteria": {"b": None, "a": {"y": 2, "x": 1}}},
        "score": {"type": "score", "instructions": ["等级"], "criteria": [" high ", {"b": 1, "a": 2}]},
        "n1": {"type": "noul", "instructions": "Yes?"},
        "n2": {"type": "noul", "instructions": "Again?", "criteria": {"false": "no", "true": ["yes"]}}}})
    tasks = compile_request(request)
    assert [t.document for t in tasks] == ["b", 'a: {"x":1,"y":2}', " high ", '{"a":2,"b":1}',
        "Question: Yes?", 'Question: Again?\nTrue criteria: ["yes"]\nFalse criteria: no']
    assert tasks[0].instruction == " Q \n\n\n" + CHOICE_ADAPTER
    assert tasks[2].instruction == '["等级"]\n\n' + SCORE_ADAPTER
    assert tasks[4].instruction == "Yes?\n\n" + NOUL_ADAPTER
    assert all(t.query == '{"a":1,"z":[2,"中文"]}' for t in tasks)
    assert all("secret-id" not in t.query + t.document + t.instruction for t in tasks)
    assert render_content(" \n") == " \n"


def test_aggregation():
    assert softmax([10000, 10000]) == [0.5, 0.5]
    assert sigmoid(-10000) == 0 and sigmoid(10000) == 1
    assert confidence([1]) == 1 and confidence([0.5, 0.5]) == 0
    assert confidence([0, .57, .43]) == pytest.approx(.378, abs=.001)
    req = Request.model_validate(example("score"))
    answer = aggregate(req, compile_request(req), [-10000, math.log(.57), math.log(.43)], Calibration())["bug_severity"]
    assert answer["score"] == pytest.approx(1.43)
    req = Request.model_validate({"state": "x", "questions": {"q": {"type": "choice", "instructions": "?", "criteria": {"first": None, "second": None}}}})
    assert aggregate(req, compile_request(req), [0, 0], Calibration())["q"]["choice"] == "first"
    req = Request.model_validate(example("noul"))
    answers = aggregate(req, compile_request(req), [10, 10], Calibration())
    assert all(a["noul"] > .99 and "confidence" not in a for a in answers.values())
    for params in ({"noul_a": 0}, {"score_temperature": -1}, {"noul_b": math.nan}):
        with pytest.raises(ValueError):
            Calibration(**params)


def test_cache_lru_and_namespace():
    cache = DocumentCache(16)
    small, big = SimpleNamespace(nbytes=8), SimpleNamespace(nbytes=17)
    cache.put("a", small)
    cache.put("b", small)
    cache.get("a")
    cache.put("c", small)
    assert list(cache.items) == ["a", "c"] and cache.bytes == 16
    cache.put("big", big)
    assert "big" not in cache.items
    cache.clear()
    assert cache.bytes == 0
    disabled = DocumentCache(0)
    disabled.put("a", small)
    assert disabled.get("a") is None
    cache.put("a", small)
    cache.max_bytes = 0
    assert cache.get("a") is None
    assert document_key({"model": "a"}, "x") != document_key({"model": "b"}, "x")
    assert document_key({"a": 1, "b": 2}, "x") == document_key({"b": 2, "a": 1}, "x")


def test_engine_cache_changes():
    engine = Engine(backend=FakeBackend())
    req = example("score")
    cold = engine.evaluate(req)
    warm, detail = engine.evaluate_detailed(req)
    assert cold["answers"] == warm["answers"]
    assert cold["usage"] == warm["usage"]
    assert cold["kalm"]["cache"]["encoded_documents"] == 3
    assert warm["kalm"]["cache"]["hit_documents"] == 3 and detail["encoder_calls"] == 0
    req["state"] = "different"
    req["questions"]["bug_severity"]["instructions"] = "New question"
    req["questions"]["bug_severity"]["criteria"].reverse()
    assert engine.evaluate(req)["kalm"]["cache"]["hit_documents"] == 3
    req["questions"]["bug_severity"]["criteria"][0] = "new"
    assert engine.evaluate(req)["kalm"]["cache"]["encoded_documents"] == 1
    req["questions"]["bug_severity"]["criteria"] = ["same", "same", "same"]
    stats = engine.evaluate(req)["kalm"]["cache"]
    assert stats["unique_documents"] == stats["miss_documents"] == stats["encoded_documents"] == 1
    assert stats["hit_documents"] == 0
    engine = Engine(backend=FakeBackend(), cache_max_mib=0)
    assert engine.evaluate(req)["kalm"]["cache"]["encoded_documents"] == 1
    assert engine.evaluate(req)["kalm"]["cache"]["encoded_documents"] == 1


@pytest.mark.parametrize("raw", ['{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":1e999}', '{'])
def test_strict_json(raw):
    with pytest.raises(JevError):
        strict_loads(raw)


@pytest.mark.parametrize("change", [
    {"state": 1}, {"state": True}, {"state": None}, {"state": {"x": float("nan")}},
    {"questions": {}}, {"unknown": 1}, {"model": 1}, {"model": None},
    {"questions": {"": {"type": "noul", "instructions": "?"}}},
    {"questions": {"q": {"type": "noul", "instructions": "?", "criteria": {}}}},
    {"questions": {"q": {"type": "noul", "instructions": "?", "criteria": {"true": None}}}},
    {"questions": {"q": {"type": "noul", "instructions": "?", "criteria": {"other": "x"}}}},
    {"questions": {"q": {"type": "score", "instructions": "?", "criteria": ["one"]}}},
    {"questions": {"q": {"type": "choice", "instructions": "?", "criteria": {"": None}}}},
])
def test_invalid_requests(change):
    request = example()
    request.update(change)
    with pytest.raises(ValidationError):
        Request.model_validate(request)


def test_http():
    engine = Engine(backend=FakeBackend())
    client = TestClient(create_app(engine), raise_server_exceptions=False)
    assert client.get("/health").status_code == 200
    assert client.post("/v1/systemone", json=example()).status_code == 200
    response = client.post("/v1/systemone", content='{"state":"a","state":"b"}')
    assert response.status_code == 422 and "error" in response.json()
    request = example()
    request["model"] = "kalm-jev-large"
    assert client.post("/v1/systemone", json=request).status_code == 400
    assert TestClient(create_app(None)).get("/health").status_code == 503
    assert TestClient(create_app(engine, 1)).post("/v1/systemone", json=example()).status_code == 422
    limited = Engine(backend=FakeBackend(), max_pairs=1)
    with pytest.raises(JevError, match="Too many"):
        limited.evaluate(example())
    engine._slots = SimpleNamespace(acquire=lambda **kw: False)
    assert client.post("/v1/systemone", json=example()).status_code == 429


def test_mixed_mapping():
    req = example()
    result = Engine(backend=FakeBackend()).evaluate(req)
    assert list(result["answers"]) == list(req["questions"])
    assert result["answers"]["department"]["choice"] == "billing"
    assert list(result["answers"]["bug_severity"]["legend"]) == ["0", "1", "2"]


@pytest.mark.parametrize("kind,count,valid", [("choice", 1, True), ("choice", 255, True),
    ("choice", 256, False), ("score", 10, True), ("score", 11, False)])
def test_candidate_boundaries(kind, count, valid):
    criteria = {str(i): None for i in range(count)} if kind == "choice" else ["level"] * count
    req = {"state": "x", "questions": {"q": {"type": kind, "instructions": "?", "criteria": criteria}}}
    if valid:
        result = Engine(backend=FakeBackend()).evaluate(req)
        if count == 1:
            assert result["answers"]["q"]["confidence"] == 1
            assert result["answers"]["q"]["probabilities"] == {"0": 1}
    else:
        with pytest.raises(ValidationError):
            Request.model_validate(req)


def test_noul_cache_invalidation():
    req = example("noul")
    engine = Engine(backend=FakeBackend())
    engine.evaluate(req)
    req["questions"]["is_human_escalation"]["instructions"] = "Does the customer want a human?"
    assert engine.evaluate(req)["kalm"]["cache"]["encoded_documents"] == 1
    req["questions"]["is_repeat_contact"]["criteria"]["false"] = "No explicit previous contact"
    assert engine.evaluate(req)["kalm"]["cache"]["encoded_documents"] == 1


def test_http_internal_errors_are_sanitized():
    backend = FakeBackend()
    engine = Engine(backend=backend)
    def fail(*args):
        raise RuntimeError("private model path and sensitive stack")
    backend.score_encoded = fail
    client = TestClient(create_app(engine), raise_server_exceptions=False)
    response = client.post("/v1/systemone", json=example())
    assert response.status_code == 500
    assert response.json() == {"error": {"type": "internal_error", "message": "Internal server error"}}
    def oom(*args):
        raise RuntimeError("CUDA out of memory")
    backend.score_encoded = oom
    assert client.post("/v1/systemone", json=example()).status_code == 503
