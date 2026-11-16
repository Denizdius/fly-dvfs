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
from flydvfs.mapping import LevelHold, position_to_sm_mhz
from flydvfs.nvml import Nvml, NvmlError
from flydvfs.plot import plot_run, write_trace
from flydvfs.validate import SampleValidator

TRACE_INTERVAL_S = 0.1
FAILSAFE_S = 3.0

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


class _Util:
    """Latest NVML GPU utilization, written by the trace thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._percent: float | None = None

    def set(self, percent: float | None) -> None:
        with self._lock:
            self._percent = percent

    def get(self) -> float | None:
        with self._lock:
            return self._percent


class _Samples:
    def __init__(self, fifo: str, validator: SampleValidator, windows_path: Path, util: _Util) -> None:
        self._validator = validator
        self._windows = windows_path.open("w")
        self._util = util
        self.received = 0
        self.valid = 0
        self.reject_reasons: dict[str, int] = {}
        self.restart_reasons: dict[str, int] = {}
        self.max_restart_ms: float | None = None
        self._fd = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)
        flags = fcntl.fcntl(self._fd, fcntl.F_GETFL)
        fcntl.fcntl(self._fd, fcntl.F_SETFL, flags & ~os.O_NONBLOCK)
        self._latest: dict | None = None
        self._seq = 0
        self._last_valid_at: float | None = None
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
                if not self._stop.is_set():
                    print(f"[flydvfs] sample read failed: {exc}", flush=True)
                break
            if not chunk:
                time.sleep(0.05)
                continue
            pending += chunk
            while b"\n" in pending:
                raw, pending = pending.split(b"\n", 1)
                if not raw.strip():
                    continue
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict) and "event" in payload:
                    self._event(payload)
                else:
                    self._window(raw, payload)

    def _event(self, payload: dict) -> None:
        kind = payload.get("event")
        if kind == "config":
            interval = payload.get("interval")
            self._validator.set_trigger(str(payload.get("trigger", "")), interval)
            print(
                f"[flydvfs] sampler chip {payload.get('chip')} trigger {payload.get('trigger')} "
                f"interval {interval} restart_samples {payload.get('restart_samples')} "
                f"restart_seconds {payload.get('restart_seconds')}",
                flush=True,
            )
        elif kind == "restart":
            self._validator.session_restarted()
            reason = str(payload.get("reason", "restart"))
            self.restart_reasons[reason] = self.restart_reasons.get(reason, 0) + 1
            ms = payload.get("ms")
            if isinstance(ms, (int, float)) and not isinstance(ms, bool):
                self.max_restart_ms = ms if self.max_restart_ms is None else max(self.max_restart_ms, ms)
            print(
                f"[flydvfs] restart {reason} ms {ms} session_samples {payload.get('session_samples')}",
                flush=True,
            )
        elif kind == "stopped":
            print(f"[flydvfs] sampler stopped {payload.get('reason')}", flush=True)

    def _window(self, raw: bytes, payload: dict) -> None:
        self.received += 1
        self._windows.write(raw.decode("utf-8", "replace") + "\n")
        self._windows.flush()
        if not isinstance(payload, dict):
            self._reject("nonfinite")
            return
        reason = self._validator.check(payload, self._util.get())
        if reason is not None:
            self._reject(reason)
            return
        self.valid += 1
        with self._lock:
            self._latest = payload
            self._seq += 1
            self._last_valid_at = time.perf_counter()

    def _reject(self, reason: str) -> None:
        first = reason not in self.reject_reasons
        self.reject_reasons[reason] = self.reject_reasons.get(reason, 0) + 1
        if first:
            print(f"[flydvfs] rejected window {reason}", flush=True)

    def latest(self) -> tuple[dict | None, int]:
        with self._lock:
            return (dict(self._latest) if self._latest is not None else None, self._seq)

    def last_valid_at(self) -> float | None:
        with self._lock:
            return self._last_valid_at

    def close(self) -> None:
        self._stop.set()
        os.close(self._fd)
        self._thread.join(timeout=2.0)
        self._windows.close()


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
    def __init__(self, nvml: Nvml, gpu: int, cap: _Cap, nvml_lock: threading.Lock, util: _Util) -> None:
        self._nvml = nvml
        self._gpu = gpu
        self._cap = cap
        self._nvml_lock = nvml_lock
        self._util = util
        self._util_failed = False
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
                try:
                    util = self._nvml.gpu_utilization_percent(self._gpu)
                except NvmlError as exc:
                    if not self._util_failed:
                        print(f"[flydvfs] utilization sample failed: {exc}", flush=True)
                        self._util_failed = True
                    util = None
        except NvmlError as exc:
            print(f"[flydvfs] trace sample failed: {exc}", flush=True)
            return
        self._util.set(util)
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
    return f"fraction {fraction:.2f}  sm {sm:.2f}  dram {dram:.2f}  -> cap {mhz} MHz"


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
        # GPU-time windows are nanoseconds. The cycle count remains for the SYSCLK fallback.
        interval_ns = max(1, int(round(sample_ms * 1_000_000)))
        interval_cycles = max(1, int(device.sm_max_mhz * 1000 * sample_ms))
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
        util = _Util()
        validator = SampleValidator(sample_ms / 1000.0, peak)
        trace = _Trace(nvml, gpu, cap, nvml_lock, util)
        fifo_dir = tempfile.mkdtemp(prefix="flydvfs-")
        fifo = os.path.join(fifo_dir, "samples")
        os.mkfifo(fifo)
        samples = _Samples(fifo, validator, output_dir / "windows.jsonl", util)
        env = os.environ.copy()
        env["CUDA_INJECTION64_PATH"] = str(library)
        env["FLYDVFS_SAMPLE_PATH"] = fifo
        env["FLYDVFS_GPU"] = str(gpu)
        env["FLYDVFS_INTERVAL_NS"] = str(interval_ns)
        env["FLYDVFS_INTERVAL_CYCLES"] = str(interval_cycles)
        process = None
        locked: int | None = None
        level_hold = LevelHold()
        seen = 0
        failsafe_unlocks = 0
        hold_until = time.perf_counter() + delay_s
        trace.start()
        try:
            process = subprocess.Popen(command, env=env)
            while process.poll() is None:
                if locked is not None:
                    last_valid = samples.last_valid_at()
                    stale = last_valid is None or time.perf_counter() - last_valid >= FAILSAFE_S
                    if stale:
                        try:
                            with nvml_lock:
                                nvml.unlock_sm_clock(gpu)
                        except NvmlError as exc:
                            print(f"[flydvfs] fail-safe unlock failed: {exc}", flush=True)
                        else:
                            cap.set_lock(None)
                            locked = None
                            level_hold.reset()
                            failsafe_unlocks += 1
                            print("[flydvfs] no valid window for 3 s, unlocked", flush=True)
                if time.perf_counter() < hold_until:
                    time.sleep(0.05)
                    continue
                sample, seq = samples.latest()
                if sample is None or seq <= seen or "sm_elapsed" not in sample:
                    time.sleep(0.05)
                    continue
                last_valid = samples.last_valid_at()
                if last_valid is None or time.perf_counter() - last_valid >= FAILSAFE_S:
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
                if fraction is None:
                    print("[flydvfs] " + _describe(fraction, sm, dram, None))
                    hold_until = time.perf_counter() + cap_window_s
                    continue
                position = level_hold.update(fraction)
                target = position_to_sm_mhz(
                    position,
                    device.sm_min_mhz,
                    device.sm_max_mhz,
                    device.supported_sm_mhz or None,
                )
                print("[flydvfs] " + _describe(fraction, sm, dram, target))
                if target != locked:
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
            try:
                samples.close()
                health = {
                    "windows_received": samples.received,
                    "windows_valid": samples.valid,
                    "reject_reasons": samples.reject_reasons,
                    "restart_reasons": samples.restart_reasons,
                    "max_restart_ms": samples.max_restart_ms,
                    "failsafe_unlocks": failsafe_unlocks,
                }
                health_path = output_dir / "sampler_health.json"
                health_path.write_text(json.dumps(health, indent=2) + "\n")
                print(f"[flydvfs] sampler health {health_path} {json.dumps(health)}", flush=True)
            except Exception as exc:
                print(f"[flydvfs] could not write sampler health: {exc}", flush=True)
            os.unlink(fifo)
            os.rmdir(fifo_dir)
        if process is None or process.returncode is None:
            return 1
        return process.returncode
