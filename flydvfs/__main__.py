"""Probe clock bounds, or run an application under the live clock controller."""

import argparse
import sys
from pathlib import Path

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
        description="Map a live compute-bound fraction onto this GPU's SM clocks."
    )
    parser.add_argument("--gpu", type=int, default=None, help="GPU index. Default with a command: 0. Default without one: every GPU.")
    parser.add_argument(
        "--compute-fraction",
        type=float,
        default=None,
        help="0..1 compute-bound score to print. 0.90 and above leaves the SM clock uncapped.",
    )
    parser.add_argument("--delay", type=float, default=60, help="Seconds to leave the clock unlocked after the application starts.")
    parser.add_argument("--cap-window", type=float, default=1, help="Seconds to hold a clock choice before reading again.")
    parser.add_argument("--sample-ms", type=float, default=80, help="Hardware sample window in milliseconds.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for the 100 ms trace and figures.")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Application to run. Put -- before its own flags.")
    args = parser.parse_args(argv)
    command = [item for item in args.command if item != "--"]

    if command:
        from flydvfs.control import run_application

        try:
            return run_application(
                command, 0 if args.gpu is None else args.gpu, args.delay, args.cap_window, args.sample_ms, args.output_dir,
            )
        except (NvmlError, RuntimeError, ValueError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

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
