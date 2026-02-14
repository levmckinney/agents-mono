"""CUDA memory debugging utilities. Import in pdb: import debug_cuda as dc"""

from __future__ import annotations

import gc
from typing import Any, TypedDict

import torch
from torch import Tensor
from torch.autograd.graph import Node


class MemorySnapshot(TypedDict):
    allocated: int
    reserved: int
    tensors: int


def stats() -> None:
    """Print GPU memory statistics."""
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    for i in range(torch.cuda.device_count()):
        print(f"=== GPU {i}: {torch.cuda.get_device_name(i)} ===")
        alloc = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        max_alloc = torch.cuda.max_memory_allocated(i) / 1e9
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  Allocated: {alloc:.2f} GB")
        print(f"  Reserved:  {reserved:.2f} GB")
        print(f"  Peak:      {max_alloc:.2f} GB")
        print(f"  Total:     {total:.2f} GB")
        print(f"  Free:      {total - reserved:.2f} GB (approx)")


def tensors(n: int = 20, device: int | None = None) -> None:
    """List top N largest CUDA tensors in memory."""
    cuda_tensors: list[Tensor] = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                if device is None or obj.device.index == device:
                    cuda_tensors.append(obj)
        except Exception:
            pass

    cuda_tensors.sort(key=lambda x: x.nelement(), reverse=True)

    total_mb: float = 0
    print(f"{'Size':<30} {'MB':>10} {'Grad':>6} {'Leaf':>6} {'Names'}")
    print("-" * 80)
    for t in cuda_tensors[:n]:
        mb = t.element_size() * t.nelement() / 1e6
        total_mb += mb
        refs = gc.get_referrers(t)
        names = [k for r in refs if isinstance(r, dict) for k, v in r.items() if v is t][:3]
        print(f"{str(list(t.size())):<30} {mb:>10.1f} {str(t.requires_grad):>6} {str(t.is_leaf):>6} {names}")

    print("-" * 80)
    print(f"Top {n} total: {total_mb:.1f} MB | All CUDA tensors: {len(cuda_tensors)}")


def trace(t: Tensor) -> None:
    """Show computation graph that created a tensor."""
    if t.grad_fn is None:
        print("Leaf tensor (no grad_fn)")
        return

    def _trace(fn: Node, depth: int = 0) -> None:
        print("  " * depth + str(type(fn).__name__))
        if hasattr(fn, "next_functions"):
            for f, _ in fn.next_functions:
                if f is not None:
                    _trace(f, depth + 1)

    _trace(t.grad_fn)


def refs(t: Tensor, max_depth: int = 2) -> None:
    """Show what objects reference a tensor."""
    seen: set[int] = set()

    def _refs(obj: Any, depth: int = 0) -> None:
        if depth > max_depth or id(obj) in seen:
            return
        seen.add(id(obj))

        for ref in gc.get_referrers(obj):
            ref_type = type(ref).__name__
            if ref_type in ("frame", "list", "tuple") and depth == 0:
                continue  # Skip noise

            if isinstance(ref, dict):
                names = [k for k, v in ref.items() if v is obj]
                if names:
                    print("  " * depth + f"dict key: {names}")
            elif hasattr(ref, "__class__"):
                print("  " * depth + f"{ref_type}")
                if depth < max_depth:
                    _refs(ref, depth + 1)

    _refs(t)


def clear() -> None:
    """Clear CUDA cache and run garbage collection."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print("Cleared cache and reset peak stats")
