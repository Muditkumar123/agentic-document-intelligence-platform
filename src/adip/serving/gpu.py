"""GPU memory helpers for serving and LLMOps metrics."""

from __future__ import annotations

from typing import Any


def torch_gpu_memory_snapshot(device: str = "cuda:0") -> dict[str, Any] | None:
    try:
        import torch
    except ImportError:
        return None

    if not torch.cuda.is_available() or not device.startswith("cuda"):
        return None

    device_index = torch.device(device).index
    if device_index is None:
        device_index = torch.cuda.current_device()

    return {
        "device": f"cuda:{device_index}",
        "device_name": torch.cuda.get_device_name(device_index),
        "allocated_mb": bytes_to_mb(torch.cuda.memory_allocated(device_index)),
        "reserved_mb": bytes_to_mb(torch.cuda.memory_reserved(device_index)),
        "max_allocated_mb": bytes_to_mb(torch.cuda.max_memory_allocated(device_index)),
        "max_reserved_mb": bytes_to_mb(torch.cuda.max_memory_reserved(device_index)),
    }


def reset_torch_peak_memory(device: str = "cuda:0") -> None:
    try:
        import torch
    except ImportError:
        return

    if not torch.cuda.is_available() or not device.startswith("cuda"):
        return
    device_index = torch.device(device).index
    if device_index is None:
        device_index = torch.cuda.current_device()
    torch.cuda.reset_peak_memory_stats(device_index)


def bytes_to_mb(value: int) -> float:
    return float(value) / (1024 * 1024)
