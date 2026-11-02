"""Run an application and hold an SM clock chosen from the live fraction."""

from __future__ import annotations

import fcntl
import json
import os
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from flydvfs.fraction import compute_fraction, dram_peak_bytes_per_s, window_throughputs
from flydvfs.nvml import Nvml, NvmlError
from flydvfs.plot import plot_run, write_trace

TRACE_INTERVAL_S = 0.1

_NATIVE = Path(__file__).resolve().parent / "native"
_SOURCE = _NATIVE / "sampler.cc"
_LIBRARY = _NATIVE / "libflydvfs_sampler.so"


def ensure_sampler() -> Path:
    """Build the CUPTI preload library if the source is newer."""
    if _LIBRARY.exists() and _LIBRARY.stat().st_mtime >= _SOURCE.stat().st_mtime:
        return _LIBRARY
    cuda = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    command = [
        "g++", "-shared", "-fPIC", "-O2", "-std=c++17",
        f"-I{cuda}/include",
        str(_SOURCE), "-o", str(_LIBRARY),
        f"-L{cuda}/lib64", "-lcuda", "-lcupti", "-ldl", "-pthread",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "failed to build the PM sampler")
    return _LIBRARY


def memory_bus_width_bits(index: int) -> int:
    import ctypes
    from ctypes import c_int, c_uint, byref, POINTER

    cuda = ctypes.CDLL("libcuda.so.1")
    cuda.cuInit.argtypes = [c_uint]
    cuda.cuInit.restype = c_int
    cuda.cuDeviceGet.argtypes = [POINTER(c_int), c_int]
    cuda.cuDeviceGet.restype = c_int
    cuda.cuDeviceGetAttribute.argtypes = [POINTER(c_int), c_int, c_int]
    cuda.cuDeviceGetAttribute.restype = c_int
    if cuda.cuInit(0) != 0:
        raise NvmlError("cuInit failed while reading the memory bus width")
    device = c_int()
    if cuda.cuDeviceGet(byref(device), index) != 0:
        raise NvmlError(f"CUDA device {index} was not found")
    width = c_int()
    # CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH
    if cuda.cuDeviceGetAttribute(byref(width), 37, device) != 0:
        raise NvmlError("failed to read the memory bus width")
    if width.value <= 0:
        raise NvmlError("memory bus width is not positive")
    return width.value


class _Samples:
    def __init__(self, fifo: str) -> None:
        self._fd = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)
        flags = fcntl.fcntl(self._fd, fcntl.F_GETFL)
        fcntl.fcntl(self._fd, fcntl.F_SETFL, flags & ~os.O_NONBLOCK)
        self._latest: dict | None = None
        self._seq = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self) -> None:
        pending = b""
        while not self._stop.is_set():
            try:
                chunk = os.read(self._fd, 4096)
            except BlockingIOError:
                time.sleep(0.01)
                continue
            except OSError as exc:
                print(f"[flydvfs] sample read failed: {exc}", flush=True)
                break
            if not chunk:
                time.sleep(0.05)
                continue
            pending += chunk
            while b"\n" in pending:
                line, pending = pending.split(b"\n", 1)
                if not line.strip():
                    continue
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError:
                    continue
                with self._lock:
                    self._latest = sample
                    self._seq += 1

    def latest(self) -> tuple[dict | None, int]:
        with self._lock:
            return (dict(self._latest) if self._latest is not None else None, self._seq)

    def close(self) -> None:
        self._stop.set()
        os.close(self._fd)


class _Cap:
    """SM limit currently in force. Unlocked means the catalog maximum."""

    def __init__(self, unlocked_mhz: int) -> None:
        self._lock = threading.Lock()
        self._mhz: int | None = None
        self.unlocked_mhz = unlocked_mhz

    def set_lock(self, mhz: int | None) -> None:
        with self._lock:
            self._mhz = mhz

    def cap_mhz(self) -> int:
        with self._lock:
            return self.unlocked_mhz if self._mhz is None else self._mhz


class _Trace:
    def __init__(self, nvml: Nvml, gpu: int, cap: _Cap, nvml_lock: threading.Lock) -> None:
        self._nvml = nvml
        self._gpu = gpu
        self._cap = cap
        self._nvml_lock = nvml_lock
        self._stop = threading.Event()
        self._rows: list[tuple[float, float, int, int, int]] = []
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._sample()
        self._thread.start()

    def stop(self) -> list[tuple[float, float, int, int, int]]:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._sample()
        return list(self._rows)

    def _sample(self) -> None:
        try:
            with self._nvml_lock:
                power = self._nvml.power_watts(self._gpu)
                sm = self._nvml.current_sm_mhz(self._gpu)
                mem = self._nvml.current_mem_mhz(self._gpu)
        except NvmlError as exc:
            print(f"[flydvfs] trace sample failed: {exc}", flush=True)
            return
        self._rows.append((time.perf_counter() - self._t0, power, sm, mem, self._cap.cap_mhz()))

    def _run(self) -> None:
        next_tick = time.perf_counter() + TRACE_INTERVAL_S
        while not self._stop.is_set():
            remaining = next_tick - time.perf_counter()
            if remaining > 0 and self._stop.wait(remaining):
                break
            self._sample()
            next_tick += TRACE_INTERVAL_S
            if next_tick < time.perf_counter():
                next_tick = time.perf_counter() + TRACE_INTERVAL_S


def _describe(fraction: float | None, sm: float | None, dram: float | None, mhz: int | None) -> str:
    if fraction is None or sm is None or dram is None:
        return "idle, clock unchanged"
    if mhz is None:
        return f"fraction {fraction:.2f}  sm {sm:.2f}  dram {dram:.2f}  -> unlock"
    return f"fraction {fraction:.2f}  sm {sm:.2f}  dram {dram:.2f}  -> lock {mhz} MHz"


def run_application(
    command: list[str],
    gpu: int,
    delay_s: float,
    cap_window_s: float,
    sample_ms: float,
    output_dir: Path | None = None,
) -> int:
    if delay_s < 0 or cap_window_s <= 0 or sample_ms <= 0:
        raise ValueError("delay must be >= 0 and cap-window and sample-ms must be positive")
    library = ensure_sampler()
    with Nvml() as nvml:
        device = nvml.read_device(gpu)
        if not device.supported_clocks_available:
            raise NvmlError("supported SM clocks are unavailable, so a lock would be invented")
        bus_bits = memory_bus_width_bits(gpu)
        peak = dram_peak_bytes_per_s(device.mem_max_mhz, bus_bits)
        # GA100 PM sampling counts SYSCLK cycles. Size one window from the catalog max.
        interval = max(1, int(device.sm_max_mhz * 1000 * sample_ms))
        print(
            f"GPU {device.index}: {device.name}  "
            f"SM {device.sm_min_mhz}..{device.sm_max_mhz} MHz  "
            f"DRAM peak {peak / 1e9:.0f} GB/s  "
            f"sample {sample_ms:.0f} ms"
        )
        if output_dir is None:
            output_dir = Path("flydvfs-results") / time.strftime("%Y%m%d_%H%M%S")
        output_dir.mkdir(parents=True, exist_ok=True)
        cap = _Cap(device.sm_max_mhz)
        nvml_lock = threading.Lock()
        trace = _Trace(nvml, gpu, cap, nvml_lock)
        fifo_dir = tempfile.mkdtemp(prefix="flydvfs-")
        fifo = os.path.join(fifo_dir, "samples")
        os.mkfifo(fifo)
        samples = _Samples(fifo)
        env = os.environ.copy()
        env["CUDA_INJECTION64_PATH"] = str(library)
        env["FLYDVFS_SAMPLE_PATH"] = fifo
        env["FLYDVFS_GPU"] = str(gpu)
        env["FLYDVFS_INTERVAL_CYCLES"] = str(interval)
        process = None
        locked: int | None = None
        seen = 0
        hold_until = time.perf_counter() + delay_s
        trace.start()
        try:
            process = subprocess.Popen(command, env=env)
            while process.poll() is None:
                if time.perf_counter() < hold_until:
                    time.sleep(0.05)
                    continue
                sample, seq = samples.latest()
                if sample is None or seq <= seen or "sm_elapsed" not in sample:
                    time.sleep(0.05)
                    continue
                seen = seq
                window_s = (sample["end_ns"] - sample["start_ns"]) * 1e-9
                parts = window_throughputs(
                    sample["sm_active"], sample["sm_elapsed"], sample["dram_bytes"], window_s, peak
                )
                if parts is None:
                    print("[flydvfs] sample could not be converted, clock unchanged")
                    hold_until = time.perf_counter() + cap_window_s
                    continue
                sm, dram = parts
                fraction = compute_fraction(sm, dram)
                target = None if fraction is None else device.target_sm_mhz(fraction)
                print("[flydvfs] " + _describe(fraction, sm, dram, target))
                if fraction is not None and target != locked:
                    with nvml_lock:
                        if target is None:
                            nvml.unlock_sm_clock(gpu)
                        else:
                            nvml.lock_sm_clock(gpu, target)
                    cap.set_lock(target)
                    locked = target
                hold_until = time.perf_counter() + cap_window_s
        finally:
            rows = trace.stop()
            trace_path = output_dir / "trace.csv"
            try:
                write_trace(trace_path, rows)
                print(f"[flydvfs] trace {trace_path} ({len(rows)} samples)")
                if len(rows) >= 2:
                    power_path, frequency_path = plot_run(output_dir, rows)
                    print(f"[flydvfs] power {power_path}")
                    print(f"[flydvfs] frequency {frequency_path}")
            except Exception as exc:
                print(f"[flydvfs] could not write the trace figures: {exc}")
            try:
                with nvml_lock:
                    nvml.unlock_sm_clock(gpu)
            except NvmlError as exc:
                print(f"[flydvfs] unlock failed: {exc}")
            if process is not None and process.poll() is None:
                process.send_signal(signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            samples.close()
            os.unlink(fifo)
            os.rmdir(fifo_dir)
        if process is None or process.returncode is None:
            return 1
        return process.returncode
