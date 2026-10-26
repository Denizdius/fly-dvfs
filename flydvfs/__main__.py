"""Probe device clock bounds and map a compute-bound fraction onto them."""

import argparse
import sys

from flydvfs.nvml import Nvml, NvmlError


def _print_device(device, compute_fraction: float | None) -> None:
    print(f"GPU {device.index}: {device.name}")
    print(f"  SM current: {device.sm_current_mhz} MHz")
    print(f"  SM range:   {device.sm_min_mhz} .. {device.sm_max_mhz} MHz")
    print(f"  MEM max:    {device.mem_max_mhz} MHz")
    if device.supported_clocks_available:
        print(f"  Supported SM clocks: {len(device.supported_sm_mhz)}")
    else:
        print("  Supported SM clocks: unavailable; minimum is set equal to maximum")
    if compute_fraction is not None:
        target = device.target_sm_mhz(compute_fraction)
        if target is None:
            print(f"  compute_fraction {compute_fraction:.2f} -> uncapped")
        else:
            print(f"  compute_fraction {compute_fraction:.2f} -> {target} MHz")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Learn each GPU's SM clock range and map a compute-bound fraction into it."
    )
    parser.add_argument("--gpu", type=int, default=None, help="Single GPU index. Default: all.")
    parser.add_argument(
        "--compute-fraction",
        type=float,
        default=None,
        help="0..1 compute-bound score. 0.90 and above leaves the SM clock uncapped.",
    )
    args = parser.parse_args(argv)

    try:
        with Nvml() as nvml:
            devices = nvml.read_all()
    except NvmlError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.gpu is not None:
        devices = [device for device in devices if device.index == args.gpu]
        if not devices:
            print(f"error: GPU {args.gpu} not found", file=sys.stderr)
            return 1

    for device in devices:
        _print_device(device, args.compute_fraction)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
