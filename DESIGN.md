# fly-dvfs

Online SM-clock control for a live GPU application. The controller reads how compute-bound the GPU is right now and sets an SM clock from that. It is a separate project from DEPO and split-dev. It does not assign SMs, and it does not use an offline model of the application.

The real runs will be on a desktop RTX 5090. This laptop is WSL, and the hardware sampler we need is not available here. The mapping rule and the control loop below are what that machine should run.

## Aim

While an application runs, every control step does three things:

1. Over a short window, about 50–100 ms, read SM throughput and DRAM bytes.
2. Turn those two readings into one compute-bound fraction from 0 to 1.
3. Map that fraction onto this GPU's own SM clock range, and keep the chosen clock until the next decision.

A fraction at or above 0.90 leaves the clock unlocked. Lower fractions lock a lower clock. The same fraction is a different MHz on a different GPU, because the MHz come from that GPU's catalog span.

The fraction is how close the GPU is to being limited by SM throughput rather than by DRAM. A kernel sitting on the memory roof is a low fraction and should be capped. A kernel sitting on the compute roof is a high fraction and should be left alone. Both pipes busy at once lands near the middle.

```
compute_fraction = sm_throughput / (sm_throughput + dram_throughput)
```

`sm_throughput` and `dram_throughput` are fractions of their own peaks over that window. DRAM bytes are recorded as well, so the log shows how much memory moved. If both readings are near zero, the window is idle and the controller does not change the clock.

There is no per-kernel model and no SM mask. The only device facts are the catalog SM range and the peaks the hardware already uses to report those two throughputs.

## What the laptop run showed

GPU: NVIDIA GeForce RTX 4080 Laptop. Catalog SM range 210–3105 MHz. Memory clock maximum 9001 MHz.

A float32 GEMM and a memory copy were sampled about 20 times a second. The catalog maximum was never held. The catalog minimum appeared at idle. While the GPU was busy the SM clock stayed about 1140–2370 MHz.

| Phase | SM clock | Memory clock |
|---|---|---|
| Idle | 210–1380 MHz, median 210 | median 405 |
| GEMM | 1140–2055 MHz, median 1425 | median 7001 |
| Memory copy | 2055–2370 MHz, median 2370 | median 9001 |
| GEMM then copy | 1140–2370 MHz, median 1290 | held at 9001 |

1140 MHz is about one third of the way up the catalog span. 2370 MHz is about three quarters of the way up. Those two fractions are the general rule. On this GPU the mapper snaps to:

| Compute fraction | Lock |
|---|---|
| 0.90 and above | unlocked |
| 0.60 to 0.90 | 2385 MHz, three quarters of the span |
| 0.30 to 0.60 | 1665 MHz, halfway up |
| below 0.30 | 1170 MHz, one third of the way up |

The catalog minimum is not a lock. Clocks are chosen and not yet applied.

## Sampler

The score has to come from hardware counters, not from NVML's coarse utilization. On this laptop, NVML's datacenter performance counters report the device as unsupported. The sampler to use on the 5090 is CUPTI PM sampling: periodic hardware samples, one pass, no kernel replay. The readings are SM throughput as a percent of peak, DRAM throughput as a percent of peak, and DRAM bytes over the window.

CUPTI has to be initialized with a subscriber before the CUDA context exists. PM sampling is checked with `cuptiProfilerDeviceSupported`. WSL can refuse that API, which is why this laptop stops at the mapping rule. The 5090 run is where the same GEMM and memory copy should show a high fraction on the GEMM and a low fraction on the copy, and where the 1/3 and 3/4 span points are checked against the clocks that GPU actually holds.

## How a run works

```bash
python -m flydvfs ./the_real_llm_application --delay 60 --cap-window 1
```

`--delay` is how long to leave the clock unlocked after the application starts. Model load is not the work we want to cap.

`--cap-window` is how long the chosen clock stays before the next decision. It is a hold, not a duty cycle.

After the delay, the loop is:

1. Read one 50–100 ms window and compute the fraction.
2. Map it. At or above 0.90, leave the clock unlocked. Otherwise lock the level from the table.
3. Keep that choice for `--cap-window` seconds.
4. Read again and choose the next clock from the new fraction.

The end of a cap window does not unlock the GPU. The clock changes only because the new fraction maps somewhere else. Unlocking happens when the fraction is at or above 0.90.

## Why the hold stays short

The sample is cheap, so the controller can decide often. It does not need a separate uncapped test stage to see the fraction. The counters are read while the current clock is still in force.

A schedule that measures for 10 seconds with the clock free, then caps for 10 seconds, spends half the timeline at full clocks. The energy saved by a cap is the energy during the memory-bound stretch. A forced free half gives that stretch back to the GPU. Ten seconds is also longer than a prefill and longer than a short decode, so one decision gets applied to both phases.

A short hold, on the order of a second, is there so the clock is not rewritten on every noisy sample. It is still short enough to follow a phase change. If a lock is too low for a compute phase, SM throughput rises against DRAM, the fraction climbs, and the next decision unlocks. The correction does not require dropping the cap first in order to look.

`--cap-window` should start at about 1 second. The 10 second free-then-cap split is the pattern to avoid.
