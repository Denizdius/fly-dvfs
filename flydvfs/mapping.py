"""Map a compute-bound fraction onto clocks a GPU actually holds.

A GEMM and a memory pass on a laptop GPU never sat on the catalog minimum
or the catalog maximum. While the GPU was busy, the SM clock stayed in a
coarse window: about one third of the way up the catalog span at the low
end, and about three quarters of the way up at the high end. Those two
fractions are the general rule. Each GPU still uses its own catalog span,
so the same level is a different MHz on a different card.

At or above 0.90 the clock is left unlocked.
"""

# (lowest compute fraction in the level, position from catalog min toward catalog max)
# None means do not lock the clock.
_LEVELS = (
    (0.90, None),
    (0.60, 0.75),
    (0.30, 0.50),
)
_LOWEST = 1.0 / 3.0


def span_position(compute_fraction: float) -> float | None:
    """Where to place this fraction along the catalog span.

    None means leave the SM clock unlocked. A number is the fraction of
    the way from the catalog minimum to the catalog maximum.
    """
    fraction = max(0.0, min(1.0, float(compute_fraction)))
    for low, position in _LEVELS:
        if fraction >= low:
            return position
    return _LOWEST


def map_compute_fraction_to_sm_mhz(
    compute_fraction: float,
    sm_min_mhz: int,
    sm_max_mhz: int,
    supported_sm_mhz: list[int] | None = None,
) -> int | None:
    """Pick an SM clock, or None when this fraction should not be capped.

    Locked levels use this GPU's own catalog span. The result is snapped
    to the nearest supported SM clock when that list is available.
    """
    if sm_min_mhz <= 0 or sm_max_mhz <= 0:
        raise ValueError("SM clock bounds must be positive")
    if sm_min_mhz > sm_max_mhz:
        raise ValueError(f"sm_min_mhz {sm_min_mhz} is above sm_max_mhz {sm_max_mhz}")

    position = span_position(compute_fraction)
    if position is None:
        return None

    span = sm_max_mhz - sm_min_mhz
    mhz = int(round(sm_min_mhz + position * span))
    mhz = max(sm_min_mhz, min(sm_max_mhz, mhz))
    if not supported_sm_mhz:
        return mhz

    allowed = [clock for clock in supported_sm_mhz if sm_min_mhz <= clock <= sm_max_mhz]
    if not allowed:
        return mhz
    return min(allowed, key=lambda clock: (abs(clock - mhz), clock))
