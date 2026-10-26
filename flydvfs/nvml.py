"""Read per-GPU minimum and maximum clocks from NVML."""

from __future__ import annotations

import ctypes
from ctypes import POINTER, byref, c_char_p, c_int, c_uint, c_void_p
from dataclasses import dataclass, field


NVML_SUCCESS = 0
NVML_ERROR_INSUFFICIENT_SIZE = 7
NVML_CLOCK_SM = 1
NVML_CLOCK_MEM = 2

_LIB_NAMES = ("libnvidia-ml.so.1", "libnvidia-ml.so")


class NvmlError(RuntimeError):
    pass


@dataclass
class DeviceClocks:
    index: int
    name: str
    sm_min_mhz: int
    sm_max_mhz: int
    mem_max_mhz: int
    sm_current_mhz: int
    supported_sm_mhz: list[int] = field(default_factory=list)
    supported_clocks_available: bool = False

    def target_sm_mhz(self, compute_fraction: float) -> int | None:
        from flydvfs.mapping import map_compute_fraction_to_sm_mhz

        return map_compute_fraction_to_sm_mhz(
            compute_fraction,
            self.sm_min_mhz,
            self.sm_max_mhz,
            self.supported_sm_mhz or None,
        )


def _load_nvml():
    last_error = None
    for name in _LIB_NAMES:
        try:
            return ctypes.CDLL(name)
        except OSError as exc:
            last_error = exc
    raise NvmlError(f"NVML library not found ({last_error})")


class Nvml:
    def __init__(self) -> None:
        self.lib = _load_nvml()
        self.lib.nvmlErrorString.argtypes = [c_int]
        self.lib.nvmlErrorString.restype = c_char_p
        self.lib.nvmlInit_v2.restype = c_int
        self.lib.nvmlShutdown.restype = c_int
        self.lib.nvmlDeviceGetCount_v2.argtypes = [POINTER(c_uint)]
        self.lib.nvmlDeviceGetCount_v2.restype = c_int
        self.lib.nvmlDeviceGetHandleByIndex_v2.argtypes = [c_uint, POINTER(c_void_p)]
        self.lib.nvmlDeviceGetHandleByIndex_v2.restype = c_int
        self.lib.nvmlDeviceGetName.argtypes = [c_void_p, c_char_p, c_uint]
        self.lib.nvmlDeviceGetName.restype = c_int
        self.lib.nvmlDeviceGetMaxClockInfo.argtypes = [c_void_p, c_int, POINTER(c_uint)]
        self.lib.nvmlDeviceGetMaxClockInfo.restype = c_int
        self.lib.nvmlDeviceGetClockInfo.argtypes = [c_void_p, c_int, POINTER(c_uint)]
        self.lib.nvmlDeviceGetClockInfo.restype = c_int
        self.lib.nvmlDeviceGetSupportedMemoryClocks.argtypes = [
            c_void_p,
            POINTER(c_uint),
            POINTER(c_uint),
        ]
        self.lib.nvmlDeviceGetSupportedMemoryClocks.restype = c_int
        self.lib.nvmlDeviceGetSupportedGraphicsClocks.argtypes = [
            c_void_p,
            c_uint,
            POINTER(c_uint),
            POINTER(c_uint),
        ]
        self.lib.nvmlDeviceGetSupportedGraphicsClocks.restype = c_int
        self._check(self.lib.nvmlInit_v2(), "nvmlInit_v2")

    def close(self) -> None:
        self.lib.nvmlShutdown()

    def __enter__(self) -> "Nvml":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def _check(self, rc: int, what: str) -> None:
        if rc != NVML_SUCCESS:
            message = self.lib.nvmlErrorString(rc)
            text = message.decode() if message else str(rc)
            raise NvmlError(f"{what}: {text}")

    def _clock_list(self, query) -> list[int] | None:
        count = c_uint(0)
        rc = query(byref(count), None)
        if rc not in (NVML_SUCCESS, NVML_ERROR_INSUFFICIENT_SIZE):
            return None
        if count.value == 0:
            return []
        clocks = (c_uint * count.value)()
        rc = query(byref(count), clocks)
        if rc == NVML_ERROR_INSUFFICIENT_SIZE:
            clocks = (c_uint * count.value)()
            rc = query(byref(count), clocks)
        if rc != NVML_SUCCESS:
            return None
        return [clocks[i] for i in range(count.value)]

    def device_count(self) -> int:
        count = c_uint()
        self._check(self.lib.nvmlDeviceGetCount_v2(byref(count)), "nvmlDeviceGetCount_v2")
        return count.value

    def _handle(self, index: int):
        handle = c_void_p()
        self._check(
            self.lib.nvmlDeviceGetHandleByIndex_v2(index, byref(handle)),
            "nvmlDeviceGetHandleByIndex_v2",
        )
        return handle

    def _name(self, handle) -> str:
        name = ctypes.create_string_buffer(96)
        self._check(self.lib.nvmlDeviceGetName(handle, name, 96), "nvmlDeviceGetName")
        return name.value.decode()

    def _max_clock(self, handle, clock_type: int) -> int:
        clock = c_uint()
        self._check(
            self.lib.nvmlDeviceGetMaxClockInfo(handle, clock_type, byref(clock)),
            "nvmlDeviceGetMaxClockInfo",
        )
        return clock.value

    def _current_clock(self, handle, clock_type: int) -> int:
        clock = c_uint()
        self._check(
            self.lib.nvmlDeviceGetClockInfo(handle, clock_type, byref(clock)),
            "nvmlDeviceGetClockInfo",
        )
        return clock.value

    def _memory_clocks(self, handle) -> list[int] | None:
        def query(count_ref, clocks):
            return self.lib.nvmlDeviceGetSupportedMemoryClocks(handle, count_ref, clocks)

        return self._clock_list(query)

    def _graphics_clocks(self, handle, memory_mhz: int) -> list[int] | None:
        def query(count_ref, clocks):
            return self.lib.nvmlDeviceGetSupportedGraphicsClocks(
                handle, memory_mhz, count_ref, clocks
            )

        return self._clock_list(query)

    def read_device(self, index: int) -> DeviceClocks:
        handle = self._handle(index)
        sm_max = self._max_clock(handle, NVML_CLOCK_SM)
        mem_max = self._max_clock(handle, NVML_CLOCK_MEM)
        sm_current = self._current_clock(handle, NVML_CLOCK_SM)

        supported: list[int] = []
        for memory_mhz in self._memory_clocks(handle) or []:
            supported.extend(self._graphics_clocks(handle, memory_mhz) or [])
        supported = sorted(set(supported))

        if supported:
            sm_min = min(supported)
            sm_max = max(sm_max, max(supported))
            clocks_ok = True
        else:
            # Supported-clock queries can be blocked. Keep the known maximum
            # and do not invent a minimum below it.
            sm_min = sm_max
            clocks_ok = False

        return DeviceClocks(
            index=index,
            name=self._name(handle),
            sm_min_mhz=sm_min,
            sm_max_mhz=sm_max,
            mem_max_mhz=mem_max,
            sm_current_mhz=sm_current,
            supported_sm_mhz=supported,
            supported_clocks_available=clocks_ok,
        )

    def read_all(self) -> list[DeviceClocks]:
        return [self.read_device(index) for index in range(self.device_count())]
