from collections import OrderedDict
import hashlib

from .templates import render_content


def document_key(identity, document):
    return hashlib.sha256(render_content({"identity": identity, "document": document}).encode()).hexdigest()


class DocumentCache:
    def __init__(self, max_bytes=256 * 1024**2):
        if max_bytes < 0:
            raise ValueError("Cache budget must be non-negative")
        self.max_bytes = max_bytes
        self.bytes = 0
        self.items = OrderedDict()

    def get(self, key):
        if not self.max_bytes:
            return None
        value = self.items.get(key)
        if value is not None:
            self.items.move_to_end(key)
        return value

    def put(self, key, value):
        if not self.max_bytes or value.nbytes > self.max_bytes:
            return
        if key in self.items:
            self.bytes -= self.items.pop(key).nbytes
        while self.bytes + value.nbytes > self.max_bytes:
            self.bytes -= self.items.popitem(last=False)[1].nbytes
        self.items[key] = value
        self.bytes += value.nbytes

    def clear(self):
        self.items.clear()
        self.bytes = 0
