"""Watch SM and memory clocks during a compute GEMM and a memory pass.

This does not set a clock. It records what the GPU actually does so the
lowest useful cap can be chosen from real frequencies.
"""

from __future__ import annotations

import argparse
import statistics
import threading
import time

from flydvfs.nvml import Nvml


def _summarize(name: str, samples: list[tuple[int, int]]) -> None:
    if not samples:
        print(f"{name}: no samples")
        return
    sm = [item[0] for item in samples]
    mem = [item[1] for item in samples]
    print(
        f"{name}: samples={len(samples)}  "
        f"SM {min(sm)}..{max(sm)} MHz (median {int(statistics.median(sm))})  "
        f"MEM {min(mem)}..{max(mem)} MHz (median {int(statistics.median(mem))})"
    )


def main() -> int:
    import torch

    parser = argparse.ArgumentParser(description="Record GPU clocks under GEMM and memory traffic.")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gemm-n", type=int, default=8192)
    parser.add_argument("--memory-mb", type=int, default=1024)
    parser.add_argument("--seconds", type=float, default=3.0)
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    bucket: dict[str, list[tuple[int, int]]] = {}
    phase = {"name": "idle"}
    stop = threading.Event()

    with Nvml() as nvml:
        bounds = nvml.read_device(args.gpu)
        print(
            f"GPU {bounds.index}: {bounds.name}  "
            f"supported SM {bounds.sm_min_mhz}..{bounds.sm_max_mhz} MHz  "
            f"MEM max {bounds.mem_max_mhz} MHz"
        )
        def run_sampler():
            while not stop.is_set():
                handle = nvml._handle(args.gpu)
                sm = nvml._current_clock(handle, 1)
                mem = nvml._current_clock(handle, 2)
                bucket.setdefault(phase["name"], []).append((sm, mem))
                stop.wait(0.05)

        worker = threading.Thread(target=run_sampler, daemon=True)
        worker.start()

        time.sleep(1.0)

        n = args.gemm_n
        a = torch.randn(n, n, device="cuda")
        b = torch.randn(n, n, device="cuda")
        for _ in range(3):
            c = a @ b
        torch.cuda.synchronize()
        phase["name"] = "gemm"
        deadline = time.perf_counter() + args.seconds
        while time.perf_counter() < deadline:
            c = a @ b
        torch.cuda.synchronize()
        print(f"gemm: {tuple(c.shape)} float32, {args.seconds:.1f}s")

        count = args.memory_mb * (1 << 20) // 4
        x = torch.empty(count, device="cuda")
        y = torch.empty(count, device="cuda")
        x.fill_(1)
        for _ in range(3):
            y.copy_(x)
        torch.cuda.synchronize()
        phase["name"] = "memory"
        deadline = time.perf_counter() + args.seconds
        while time.perf_counter() < deadline:
            y.copy_(x)
            x.add_(y)
        torch.cuda.synchronize()
        moved = args.memory_mb * 3
        print(f"memory: about {args.memory_mb} MiB buffers, copy+add, {args.seconds:.1f}s")

        phase["name"] = "mixed"
        deadline = time.perf_counter() + args.seconds
        while time.perf_counter() < deadline:
            c = a @ b
            y.copy_(x)
        torch.cuda.synchronize()
        print(f"mixed: gemm then copy, {args.seconds:.1f}s (bytes moved per gemm step ~{moved} MiB scale)")

        stop.set()
        worker.join(timeout=1.0)

    for name in ("idle", "gemm", "memory", "mixed"):
        _summarize(name, bucket.get(name, []))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
