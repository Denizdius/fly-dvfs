"""Write the 100 ms trace and the power and frequency figures."""

from __future__ import annotations

import csv
from pathlib import Path


TRACE_FIELDS = ("t_s", "power_w", "sm_mhz", "mem_mhz", "cap_mhz")


def write_trace(path: Path, rows: list[tuple[float, float, int, int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(TRACE_FIELDS)
        writer.writerows(rows)


def plot_run(output_dir: Path, rows: list[tuple[float, float, int, int, int]]) -> tuple[Path, Path]:
    """Save power.png and frequency.png. The cap line is the SM limit in force."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    time_s = [row[0] for row in rows]
    power_w = [row[1] for row in rows]
    sm_mhz = [row[2] for row in rows]
    mem_mhz = [row[3] for row in rows]
    cap_mhz = [row[4] for row in rows]

    power_path = output_dir / "power.png"
    figure, axis = plt.subplots(figsize=(10, 4))
    axis.plot(time_s, power_w, color="#c0392b", linewidth=1.2)
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Power (W)")
    axis.set_title("GPU power")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(power_path, dpi=150)
    plt.close(figure)

    frequency_path = output_dir / "frequency.png"
    figure, axis = plt.subplots(figsize=(10, 4))
    axis.plot(time_s, sm_mhz, color="#1f4e79", linewidth=1.2, label="SM actual")
    axis.step(time_s, cap_mhz, where="post", color="#e67e22", linewidth=1.4, label="SM cap")
    axis.plot(time_s, mem_mhz, color="#1e8449", linewidth=1.0, linestyle=":", label="Memory")
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Frequency (MHz)")
    axis.set_title("GPU frequency")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(frequency_path, dpi=150)
    plt.close(figure)
    return power_path, frequency_path
