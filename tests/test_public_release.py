import copy
import importlib.util
import json
from pathlib import Path

from kalm_jev.backend import TransformersBackend

ROOT = Path(__file__).resolve().parents[1]


def test_public_identity_does_not_mutate_cache_identity(tmp_path):
    backend = object.__new__(TransformersBackend)
    backend.identity = {
        "model": "kalm-jev-nano", "path": str(tmp_path), "tokenizer": str(tmp_path),
        "revision": None, "dtype": "torch.float32",
        "local_fingerprint": {"config.json": [100, 123456789, "a" * 64], "model.safetensors": [200, 123456789]},
    }
    original = copy.deepcopy(backend.identity)
    public = backend.public_identity
    assert backend.identity == original
    assert str(tmp_path) not in json.dumps(public)
    assert "123456789" not in json.dumps(public)
    assert public["repository"] == "KaLM-Embedding/KaLM-Reranker-V1-Nano-R2"
    assert public["files"]["config.json"] == {"size_bytes": 100, "sha256": "a" * 64}
    assert public["files"]["model.safetensors"] == {"size_bytes": 200}


def test_published_tree_and_metadata():
    spec = importlib.util.spec_from_file_location("public_check", ROOT / "scripts/check_public_tree.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    count, issues = module.scan(ROOT)
    assert count > 30
    assert issues == []
    result_files = list((ROOT / "results").glob("*.json"))
    assert len(result_files) >= 11
    for path in result_files:
        data = json.loads(path.read_text())
        if "identity" in data:
            assert not {"path", "tokenizer", "local_fingerprint"} & data["identity"].keys()


def test_scripts_use_portable_defaults():
    script = (ROOT / "tests/run_gpu_suite.sh").read_text()
    assert 'python_bin="${KALM_PYTHON:-python}"' in script
    assert 'model_root="${KALM_MODEL_ROOT:-./models}"' in script
    assert "export NVIDIA_DRIVER_CAPABILITIES" not in script
    assert "export LD_LIBRARY_PATH" not in script
