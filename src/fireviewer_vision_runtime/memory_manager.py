from __future__ import annotations

import gc
from typing import Protocol


class Releasable(Protocol):
    def unload(self) -> None: ...


class MemoryManager:
    """Owns the explicit release boundary between two GPU model families."""

    def peak_vram_bytes(self) -> int | None:
        try:
            import torch
        except ImportError:
            return None
        if not torch.cuda.is_available():
            return None
        return int(torch.cuda.max_memory_allocated())

    def reset_peak(self) -> None:
        try:
            import torch
        except ImportError:
            return
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def release(self, adapter: Releasable) -> None:
        adapter.unload()
        release_cuda_memory()

    def finalize_job(self) -> None:
        """Release CUDA IPC allocations once after the whole sequential model session."""

        try:
            import torch
        except ImportError:
            return
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()  # type: ignore[no-untyped-call]


def synchronize_cuda() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def release_cuda_memory() -> None:
    """Collect Python objects and release unused CUDA allocations between models."""

    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
