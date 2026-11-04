# fly-dvfs

Online GPU clock control from a live compute-bound fraction. This project is separate from DEPO and split-dev. The aim, the laptop measurements, and the control loop are in [DESIGN.md](DESIGN.md).

The controller reads a one-pass CUPTI window (SM active cycles, SM elapsed cycles, and DRAM bytes), turns that into a compute-bound fraction, and locks this GPU's SM clock. A fraction at or above 0.80 leaves the clock unlocked.

```bash
python -m flydvfs --delay 60 --cap-window 1 -- /data/fly-dvfs/TokenPowerBench/trial.sh
```

Without a command, the same program only prints the clock range and where a fraction would land.

While the application runs, power, SM frequency, and memory frequency are recorded every 100 ms. After it exits, the run directory contains `trace.csv`, `power.png`, and `frequency.png`. The frequency figure draws the measured SM clock, the SM cap that was in force, and the memory clock. An unlocked cap sits at the catalog maximum.


Each lock is a cut down from the top of that GPU's catalog span. The same cut is a different MHz on another card. The chosen clock is then snapped to a supported step. The catalog minimum is not used as a lock.

- `0.80` and above: no cap. The clock stays unlocked.
- `0.70` to `0.80`: 10% off the top of the span (position 0.90).
- `0.50` to `0.70`: 25% (position 0.75).
- `0.30` to `0.50`: 50% (position 0.50).
- Below `0.30`: 70% (position 0.30). That is the lowest lock.

On a 210–1410 MHz catalog those locks are 1290, 1110, 810, and 570 MHz. On a 210–3150 MHz catalog they are 2856, 2415, 1680, and 1092 MHz before snapping.

The earlier two-level map is kept in `mapping_snapshots/two_level_0.30_0.90`. It unlocked at 0.90, locked 0.30–0.90 at three quarters of the span, and used one third of the span below 0.30.

```bash
python -m flydvfs
python -m flydvfs --compute-fraction 0.50
python -m flydvfs --gpu 0 --compute-fraction 0.90
python -m unittest flydvfs.test_mapping
```

Reading supported clocks uses NVML. If the driver does not expose them, the minimum is reported equal to the maximum and no lower clock is invented.
