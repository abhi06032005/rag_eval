"""
Systems Telemetry and Hardware Profiling.
Records per-stage latency percentiles (p50, p95, p99), process memory, and host metadata.
"""

import platform
from collections.abc import Sequence

import numpy as np
import psutil
from pydantic import BaseModel


class StageTimings(BaseModel):
    embedding_ms: float = 0.0
    retrieval_ms: float = 0.0
    context_assembly_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0


class HardwareMetadata(BaseModel):
    platform: str
    machine: str
    processor: str
    cpu_count_logical: int
    cpu_count_physical: int
    ram_bytes: int
    python_version: str
    docker_version: str = "unknown"
    gpu_model: str = "none"


def get_hardware_metadata() -> HardwareMetadata:
    """Capture snapshot of host system specifications."""
    mem = psutil.virtual_memory()
    return HardwareMetadata(
        platform=platform.platform(),
        machine=platform.machine(),
        processor=platform.processor() or "Unknown CPU",
        cpu_count_logical=psutil.cpu_count(logical=True) or 1,
        cpu_count_physical=psutil.cpu_count(logical=False) or 1,
        ram_bytes=mem.total,
        python_version=platform.python_version(),
    )


def calculate_latency_percentiles(latencies_ms: Sequence[float]) -> dict[str, float]:
    """Calculate p50, p95, and p99 from a series of durations in milliseconds."""
    if not latencies_ms:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "mean_ms": 0.0}
    arr = np.array(latencies_ms)
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "mean_ms": float(np.mean(arr)),
    }
