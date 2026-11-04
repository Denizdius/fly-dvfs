"""Map a compute fraction to a point on this GPU's catalog span.

The fraction is sm / (sm + dram). A higher value is a more compute-bound
window. Each locked level is a cut down from the top of the catalog span,
so the same level is a different MHz on another card.

    fraction          cut of the span    position
    0.80 and above    unlocked           —
    0.70 to 0.80      10%                0.90
    0.50 to 0.70      25%                0.75
    0.30 to 0.50      50%                0.50
    below 0.30        70%                0.30

On a 210..3150 MHz catalog those locks fall at 2856, 2415, 1680 and 1092 MHz
before snapping to a supported clock. On a 210..1410 MHz catalog they fall
at 1290, 1110, 810 and 570 MHz.
"""

# (lowest compute fraction in the level, position from catalog min toward catalog max)
# None means do not lock the clock. The first matching level wins.
_LEVELS = (
    (0.80, None),
    (0.70, 0.90),
    (0.50, 0.75),
    (0.30, 0.50),
)
_LOWEST = 0.30


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
