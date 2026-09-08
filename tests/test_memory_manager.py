from __future__ import annotations

import sys
from types import SimpleNamespace

from fireviewer_vision_runtime.memory_manager import MemoryManager


class FakeAdapter:
    def __init__(self) -> None:
        self.unloads = 0

    def unload(self) -> None:
        self.unloads += 1


def test_cuda_ipc_collection_runs_once_at_the_end_of_the_job(monkeypatch) -> None:
    calls = {"empty": 0, "ipc": 0}
    cuda = SimpleNamespace(
        is_available=lambda: True,
        empty_cache=lambda: calls.__setitem__("empty", calls["empty"] + 1),
        ipc_collect=lambda: calls.__setitem__("ipc", calls["ipc"] + 1),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    manager = MemoryManager()
    adapters = [FakeAdapter(), FakeAdapter()]

    for adapter in adapters:
        manager.release(adapter)

    assert [adapter.unloads for adapter in adapters] == [1, 1]
    assert calls == {"empty": 2, "ipc": 0}

    manager.finalize_job()

    assert calls == {"empty": 2, "ipc": 1}
