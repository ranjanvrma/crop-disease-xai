"""Model size and desktop latency (on-device numbers come from the Android app)."""

import io
import platform
import time

import numpy as np
import torch
from torch import nn


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def state_dict_size_mb(model: nn.Module) -> float:
    """Size of the serialised weights, as they would be saved to disk."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / 1e6


def device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return platform.processor() or platform.machine()


@torch.no_grad()
def measure_latency(
    model: nn.Module, image_size: int, device: torch.device, warmup: int = 20, iters: int = 100
) -> dict[str, float | str]:
    """Batch-1 forward latency in ms: median and p95 after warm-up."""
    model.eval().to(device)
    x = torch.randn(1, 3, image_size, image_size, device=device)
    for _ in range(warmup):
        model(x)
    times = []
    for _ in range(iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return {
        "median_ms": float(np.median(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "device": device_name(device),
        "threads": torch.get_num_threads(),
    }
