# fly-dvfs

Online GPU clock control from a live compute-bound fraction. This project is separate from DEPO and split-dev. The aim, the laptop measurements, and the control loop are in [DESIGN.md](DESIGN.md).

What is implemented so far learns each GPU's own SM clock range and maps a compute-bound fraction into the window a busy GPU actually holds. The live sampler runs on the desktop GPU, not on this WSL laptop.

On the laptop run, catalog SM clocks were 210–3105 MHz, but GEMM and memory traffic stayed near 1140–2370 MHz. That is about one third of the catalog span at the low end and about three quarters at the high end. The catalog minimum appeared at idle, and the catalog maximum was not held. Those two fractions are the rule for every GPU; the MHz still come from that GPU's own span.

- `0.90` and above: no cap. The clock stays unlocked.
- `0.60` to `0.90`: three quarters of the way up the catalog span.
- `0.30` to `0.60`: halfway up the catalog span.
- Below `0.30`: one third of the way up. That is the lowest lock. The catalog minimum is not used.

On the 210–3105 MHz laptop GPU those three locks are about 1175, 1660, and 2380 MHz before snapping to a supported clock.

```bash
python -m flydvfs
python -m flydvfs --compute-fraction 0.50
python -m flydvfs --gpu 0 --compute-fraction 0.90
python -m unittest flydvfs.test_mapping
```

Reading supported clocks uses NVML. If the driver does not expose them, the minimum is reported equal to the maximum and no lower clock is invented.
