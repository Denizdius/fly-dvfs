"""Turn one hardware window into a compute-bound fraction.

SM throughput is active SM cycles over elapsed SM cycles. DRAM throughput is
bytes moved over this GPU's peak DRAM bandwidth. Both are fractions of their
own peaks. The high-level CUPTI throughput metrics need many passes, which
PM sampling rejects, so these one-pass counters are the readings the
controller uses.

A window where both are near zero is idle. The caller leaves the clock alone.
"""

from __future__ import annotations

IDLE_THRESHOLD = 0.02


def dram_peak_bytes_per_s(mem_clock_mhz: int, bus_width_bits: int) -> float:
    """Peak DRAM bandwidth.

    NVML reports the memory clock. HBM and GDDR move data on both clock edges,
    so the peak is twice clock times bus width.
    """
    if mem_clock_mhz <= 0 or bus_width_bits <= 0:
        raise ValueError("memory clock and bus width must be positive")
    return mem_clock_mhz * 1_000_000 * (bus_width_bits / 8) * 2


def window_throughputs(
    sm_active: float,
    sm_elapsed: float,
    dram_bytes: float,
    window_s: float,
    dram_peak_bytes_per_s: float,
) -> tuple[float, float] | None:
    """Return (sm_fraction, dram_fraction), each clamped to 0..1."""
    if sm_elapsed <= 0 or window_s <= 0 or dram_peak_bytes_per_s <= 0:
        return None
    sm = sm_active / sm_elapsed
    dram = dram_bytes / (window_s * dram_peak_bytes_per_s)
    return max(0.0, min(1.0, sm)), max(0.0, min(1.0, dram))


def compute_fraction(sm: float, dram: float, idle_threshold: float = IDLE_THRESHOLD) -> float | None:
    """Compute-bound fraction, or None when the window is idle."""
    if sm < idle_threshold and dram < idle_threshold:
        return None
    total = sm + dram
    if total <= 0:
        return None
    return sm / total
