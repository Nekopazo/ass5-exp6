#!/usr/bin/env python3
"""Reserve a fixed amount of GPU memory on a selected CUDA device.

Example:
    python vram.py --device 0 --gb 8
    python vram.py --device 1 --mb 12288 --hold-seconds 600
"""

import argparse
import time
from typing import List

import torch


def format_bytes(num_bytes: int) -> str:
    gb = num_bytes / (1024 ** 3)
    return f"{gb:.2f} GiB"


def current_used_bytes(device: torch.device) -> int:
    free, total = torch.cuda.mem_get_info(device)
    return total - free


def current_free_bytes(device: torch.device) -> int:
    free, _ = torch.cuda.mem_get_info(device)
    return free


def allocate_target_bytes(
    device: torch.device,
    target_bytes: int,
    chunk_mb: int = 256,
) -> List[torch.Tensor]:
    tensors: List[torch.Tensor] = []
    chunk_bytes = max(1, chunk_mb) * 1024 * 1024

    while True:
        used = current_used_bytes(device)
        remaining = target_bytes - used
        if remaining <= 0:
            break

        alloc_bytes = min(chunk_bytes, remaining)
        success = False

        # Reduce allocation size on OOM until it can fit or becomes too small.
        while alloc_bytes >= 1 * 1024 * 1024:
            try:
                t = torch.empty((alloc_bytes,), dtype=torch.uint8, device=device)
                t.fill_(1)
                tensors.append(t)
                success = True
                break
            except RuntimeError as err:
                if "out of memory" in str(err).lower():
                    alloc_bytes //= 2
                    torch.cuda.empty_cache()
                    continue
                raise

        if not success:
            break

    return tensors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Occupy a fixed amount of VRAM on a selected GPU using PyTorch."
    )
    parser.add_argument("--device", type=int, required=True, help="CUDA device index")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--gb", type=float, help="Target occupied memory in GiB")
    group.add_argument("--mb", type=int, help="Target occupied memory in MiB")

    parser.add_argument(
        "--chunk-mb",
        type=int,
        default=256,
        help="Allocation chunk size in MiB (default: 256)",
    )
    parser.add_argument(
        "--hold-seconds",
        type=int,
        default=0,
        help="How long to hold memory. 0 means hold until Ctrl+C.",
    )
    parser.add_argument(
        "--report-interval",
        type=int,
        default=5,
        help="Status print interval in seconds while holding (default: 5)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your PyTorch/CUDA setup.")

    device_count = torch.cuda.device_count()
    if args.device < 0 or args.device >= device_count:
        raise ValueError(f"Invalid device {args.device}, available range: 0..{device_count - 1}")

    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)

    prop = torch.cuda.get_device_properties(device)
    total_bytes = prop.total_memory
    target_bytes = int(args.gb * (1024 ** 3)) if args.gb is not None else int(args.mb * (1024 ** 2))

    print(f"[INFO] Device: cuda:{args.device} ({prop.name})")
    print(f"[INFO] Total VRAM: {format_bytes(total_bytes)}")
    print(f"[INFO] Target occupied VRAM: {format_bytes(target_bytes)}")

    if target_bytes <= 0:
        raise ValueError("Target memory must be > 0")

    if target_bytes > total_bytes:
        print("[WARN] Target exceeds physical VRAM. Will allocate as much as possible.")

    tensors = allocate_target_bytes(device=device, target_bytes=target_bytes, chunk_mb=args.chunk_mb)
    torch.cuda.synchronize(device)

    used = current_used_bytes(device)
    free = current_free_bytes(device)
    print(f"[INFO] Allocated chunks: {len(tensors)}")
    print(f"[INFO] Current used VRAM: {format_bytes(used)}")
    print(f"[INFO] Probe free VRAM: {format_bytes(free)}")

    interval = max(1, args.report_interval)
    start = time.time()

    try:
        if args.hold_seconds > 0:
            while True:
                elapsed = time.time() - start
                if elapsed >= args.hold_seconds:
                    break
                used = current_used_bytes(device)
                free = current_free_bytes(device)
                print(
                    f"[HOLD] elapsed={int(elapsed)}s used={format_bytes(used)} "
                    f"probe_free={format_bytes(free)}"
                )
                time.sleep(interval)
        else:
            print("[INFO] Holding memory. Press Ctrl+C to release.")
            while True:
                elapsed = time.time() - start
                used = current_used_bytes(device)
                free = current_free_bytes(device)
                print(
                    f"[HOLD] elapsed={int(elapsed)}s used={format_bytes(used)} "
                    f"probe_free={format_bytes(free)}"
                )
                time.sleep(interval)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        del tensors
        torch.cuda.empty_cache()
        time.sleep(0.2)
        used = current_used_bytes(device)
        free = current_free_bytes(device)
        print(
            f"[INFO] Released. Current used VRAM: {format_bytes(used)} "
            f"probe_free={format_bytes(free)}"
        )


if __name__ == "__main__":
    main()
