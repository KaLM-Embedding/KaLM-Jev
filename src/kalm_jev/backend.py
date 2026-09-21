"""Thin adapter around the checkpoint's original R2 implementation."""
from dataclasses import dataclass
import hashlib
import importlib.util
from pathlib import Path
import sys
import types

import torch
import transformers
from transformers.modeling_outputs import BaseModelOutput

from .schemas import JevError

MODELS = {f"kalm-jev-{size.lower()}": f"KaLM-Embedding/KaLM-Reranker-V1-{size}-R2"
          for size in ("Nano", "Small", "Large")}


def load_native(path):
    # Isolated package names prevent mixing source code from different checkpoints.
    name = "_kalm_r2_" + hashlib.sha256(str(path).encode()).hexdigest()[:16]
    if name + ".kalm_reranker" in sys.modules:
        return sys.modules[name + ".kalm_reranker"], sys.modules[name + ".kalm_reranker_utils"]
    package = types.ModuleType(name)
    package.__path__ = [str(path)]
    sys.modules[name] = package
    for module in ("kalm_reranker_utils", "kalm_reranker"):
        spec = importlib.util.spec_from_file_location(name + "." + module, path / (module + ".py"))
        loaded = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = loaded
        spec.loader.exec_module(loaded)
    return sys.modules[name + ".kalm_reranker"], sys.modules[name + ".kalm_reranker_utils"]


@dataclass(frozen=True)
class EncodedDocument:
    hidden: torch.Tensor
    mask: torch.Tensor

    @property
    def nbytes(self):
        return self.hidden.numel() * self.hidden.element_size() + self.mask.numel() * self.mask.element_size()


class TransformersBackend:
    def __init__(self, model="kalm-jev-nano", model_path=None, revision=None,
                 device=None, dtype=None, batch_size=4, query_max_length=512,
                 document_max_length=1024, decoder_max_length=2048, chunk_size=4):
        if model not in MODELS:
            raise JevError(400, "Unknown model", "model", "model_error")
        if min(batch_size, query_max_length, document_max_length, decoder_max_length) <= 0:
            raise ValueError("Batch size and token limits must be positive")
        if model_path is None:
            from huggingface_hub import snapshot_download
            model_path = snapshot_download(MODELS[model], revision=revision)
        path = Path(model_path).resolve()
        native, self.utils = load_native(path)
        self.reranker = native.KaLMReranker(
            str(path), device=device, dtype=dtype, batch_size=batch_size,
            query_max_length=query_max_length, max_length=document_max_length,
            chunk_size=chunk_size, local_files_only=True,
        )
        self.model, self.tokenizer = self.reranker.model, self.reranker.tokenizer
        self.device = self.reranker.device
        self.batch_size = batch_size
        self.query_max_length, self.document_max_length = query_max_length, document_max_length
        self.decoder_max_length = decoder_max_length
        encoder_limit = self.model.config.encoder.text_config.max_position_embeddings
        decoder_limit = self.model.config.decoder.max_position_embeddings
        if document_max_length > encoder_limit or decoder_max_length > decoder_limit or query_max_length > decoder_limit:
            raise ValueError("Configured token limit exceeds the checkpoint position limit")
        # Metadata fingerprint avoids re-hashing GBs of weights at each startup.
        # Replacing local files while running is unsupported.
        files = {}
        for file in sorted(path.iterdir()):
            if file.suffix in (".json", ".py", ".safetensors", ".bin"):
                stat = file.stat()
                files[file.name] = [stat.st_size, stat.st_mtime_ns]
                if file.suffix in (".py", ".json"):
                    files[file.name].append(hashlib.sha256(file.read_bytes()).hexdigest())
        self.identity = {
            "model": model, "path": str(path), "revision": revision,
            "local_fingerprint": files, "tokenizer": str(path),
            "encoder_implementation": "native-r2-v1", "document_template": "Document-v1",
            "document_max_length": document_max_length, "dtype": str(self.reranker.dtype),
            "chunk_size": chunk_size, "transformers": transformers.__version__,
            "torch": torch.__version__, "device": str(self.device),
        }
        self.encoded_documents = 0
        self.encoder_calls = 0

    @property
    def public_identity(self):
        """Shareable metadata; the private cache identity remains unchanged."""
        identity = {key: value for key, value in self.identity.items()
                    if key not in ("path", "tokenizer", "local_fingerprint")}
        identity["repository"] = MODELS[self.identity["model"]]
        identity["files"] = {
            name: {"size_bytes": entry[0], **({"sha256": entry[2]} if len(entry) > 2 else {})}
            for name, entry in self.identity["local_fingerprint"].items()
        }
        return identity

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def prepare(self, tasks):
        documents, decoders = {}, []
        queries = set()
        for task in tasks:
            if task.query not in queries:
                ids = self.tokenizer(task.query, add_special_tokens=False)["input_ids"]
                if len(ids) > self.query_max_length:
                    raise JevError(422, f"state exceeds {self.query_max_length} query tokens", "state")
                queries.add(task.query)
            if task.document not in documents:
                ids = self.tokenizer("<Document>: " + task.document, add_special_tokens=False)["input_ids"]
                if len(ids) > self.document_max_length:
                    raise JevError(422, f"Document exceeds {self.document_max_length} tokens", f"questions.{task.question_id}.criteria")
                documents[task.document] = ids
            text = self.reranker._decoder_text(task.query, task.instruction)
            ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            if (len(ids) + 7) // 8 * 8 > self.decoder_max_length:
                raise JevError(422, f"Full decoder template exceeds {self.decoder_max_length} tokens", f"questions.{task.question_id}.instructions")
            decoders.append(ids)
        usage = sum(len(documents[task.document]) + len(ids) for task, ids in zip(tasks, decoders))
        return documents, decoders, usage

    def _pad(self, rows, decoder=False):
        return self.tokenizer.pad({"input_ids": rows}, padding=True,
                                  pad_to_multiple_of=8 if decoder else None,
                                  return_tensors="pt").to(self.device)

    @torch.inference_mode()
    def encode_documents(self, rows):
        batch = self._pad(rows)
        hidden = self.utils.get_encoder(self.model)(**batch, return_dict=True).last_hidden_state
        self.encoder_calls += 1
        self.encoded_documents += len(rows)
        mask = batch["attention_mask"]
        if self.reranker.chunk_size is not None:
            hidden, mask = self.utils.pool_encoder_chunks(hidden, mask, self.reranker.chunk_size)
        result = []
        for i in range(len(rows)):
            length = int(mask[i].sum())
            result.append(EncodedDocument(hidden[i, :length].detach().clone(), mask[i, :length].detach().clone()))
        return result

    @torch.inference_mode()
    def score_encoded(self, decoder_rows, documents):
        batch = self._pad(decoder_rows, decoder=True)
        hidden = torch.nn.utils.rnn.pad_sequence([d.hidden for d in documents], batch_first=True)
        mask = torch.nn.utils.rnn.pad_sequence([d.mask for d in documents], batch_first=True)
        outputs = self.model(
            encoder_outputs=BaseModelOutput(last_hidden_state=hidden), attention_mask=mask,
            decoder_input_ids=batch["input_ids"], decoder_attention_mask=batch["attention_mask"],
            use_cache=False, return_dict=True,
        )
        logits = self.utils.extract_yes_no_logits(outputs.logits, batch["attention_mask"],
                                                 self.reranker.yes_token_id, self.reranker.no_token_id)
        return (logits[:, 0] - logits[:, 1]).cpu().tolist()
