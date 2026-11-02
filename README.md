# fly-dvfs

Online GPU clock control from a live compute-bound fraction. This project is separate from DEPO and split-dev. The aim, the laptop measurements, and the control loop are in [DESIGN.md](DESIGN.md).

The controller reads a one-pass CUPTI window (SM active cycles, SM elapsed cycles, and DRAM bytes), turns that into a compute-bound fraction, and locks this GPU's SM clock. A fraction at or above 0.90 leaves the clock unlocked.

```bash
python -m flydvfs --delay 60 --cap-window 1 -- /data/fly-dvfs/TokenPowerBench/trial.sh
```

Without a command, the same program only prints the clock range and where a fraction would land.

While the application runs, power, SM frequency, and memory frequency are recorded every 100 ms. After it exits, the run directory contains `trace.csv`, `power.png`, and `frequency.png`. The frequency figure draws the measured SM clock, the SM cap that was in force, and the memory clock. An unlocked cap sits at the catalog maximum.


On the laptop run, catalog SM clocks were 210–3105 MHz, but GEMM and memory traffic stayed near 1140–2370 MHz. That is about one third of the catalog span at the low end and about three quarters at the high end. The catalog minimum appeared at idle, and the catalog maximum was not held. Those two fractions are the rule for every GPU; the MHz still come from that GPU's own span.

- `0.90` and above: no cap. The clock stays unlocked.
- `0.30` to `0.90`: three quarters of the way up the catalog span. On the 210–1410 MHz A100 that lock is 1110 MHz.
- Below `0.30`: one third of the way up. That is the lowest lock. The catalog minimum is not used.

On the 210–3105 MHz laptop GPU those locks are about 1175 and 2380 MHz before snapping to a supported clock.

```bash
python -m flydvfs
python -m flydvfs --compute-fraction 0.50
python -m flydvfs --gpu 0 --compute-fraction 0.90
python -m unittest flydvfs.test_mapping
```

Reading supported clocks uses NVML. If the driver does not expose them, the minimum is reported equal to the maximum and no lower clock is invented.
