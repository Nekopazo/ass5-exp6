#!/usr/bin/env python3
"""Reserve a fixed amount of GPU memory on a selected CUDA device.

Example:
    python vram.py --device 0 --gb 8
    python vram.py --device 1 --mb 12288 --hold-seconds 600
    python vram.py --device 1 --gb 20 --daemon --target-mode self
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from typing import List

import torch


def default_pid_file(device_index: int) -> str:
    return f"/tmp/vram-reserve-cuda{device_index}.pid"


def default_log_file(device_index: int) -> str:
    return f"/tmp/vram-reserve-cuda{device_index}.log"


def is_pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def read_pid_file(pid_file: str) -> int:
    with open(pid_file, "r", encoding="utf-8") as f:
        return int(f.read().strip())


def ensure_pid_file_available(pid_file: str) -> None:
    if os.path.exists(pid_file):
        try:
            existing_pid = read_pid_file(pid_file)
            if is_pid_running(existing_pid):
                raise RuntimeError(
                    f"Daemon already running with PID {existing_pid}. "
                    f"Use --stop --pid-file {pid_file} to stop it first."
                )
            print(f"[WARN] Removing stale PID file: {pid_file}")
            os.remove(pid_file)
        except (ValueError, OSError):
            print(f"[WARN] Invalid PID file detected. Removing: {pid_file}")
            os.remove(pid_file)


def start_daemon_subprocess(pid_file: str, log_file: str) -> None:
    ensure_pid_file_available(pid_file)

    script_path = os.path.abspath(__file__)
    child_cmd = [sys.executable, script_path, *sys.argv[1:], "--_daemon-child"]

    with open(log_file, "a", encoding="utf-8") as logf:
        child = subprocess.Popen(
            child_cmd,
            stdin=subprocess.DEVNULL,
            stdout=logf,
            stderr=logf,
            cwd="/",
            start_new_session=True,
            close_fds=True,
        )

    with open(pid_file, "w", encoding="utf-8") as f:
        f.write(str(child.pid))

    print(f"[INFO] Daemon started. PID={child.pid} log={log_file} pid_file={pid_file}")


def daemonize_process(pid_file: str, log_file: str) -> None:
    start_daemon_subprocess(pid_file=pid_file, log_file=log_file)


def stop_daemon(pid_file: str) -> None:
    if not os.path.exists(pid_file):
        print(f"[INFO] No daemon PID file found: {pid_file}")
        return

    try:
        pid = read_pid_file(pid_file)
    except (ValueError, OSError):
        print(f"[WARN] PID file is invalid, removing: {pid_file}")
        os.remove(pid_file)
        return

    if not is_pid_running(pid):
        print(f"[INFO] Process {pid} is not running, removing stale PID file.")
        os.remove(pid_file)
        return

    os.kill(pid, signal.SIGTERM)
    print(f"[INFO] Sent SIGTERM to daemon PID {pid}")

    for _ in range(50):
        if not is_pid_running(pid):
            break
        time.sleep(0.1)

    if is_pid_running(pid):
        print(f"[WARN] Daemon PID {pid} is still running.")
    else:
        print(f"[INFO] Daemon PID {pid} stopped.")
        if os.path.exists(pid_file):
            os.remove(pid_file)


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
    target_mode: str = "self",
) -> List[torch.Tensor]:
    tensors: List[torch.Tensor] = []
    chunk_bytes = max(1, chunk_mb) * 1024 * 1024

    while True:
        if target_mode == "total":
            used = current_used_bytes(device)
        else:
            used = sum(t.numel() * t.element_size() for t in tensors)
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

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--gb", type=float, help="Target occupied memory in GiB")
    group.add_argument("--mb", type=int, help="Target occupied memory in MiB")

    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run as a Unix daemon process in background.",
    )
    parser.add_argument(
        "--_daemon-child",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Stop daemon process by PID file and exit.",
    )
    parser.add_argument(
        "--pid-file",
        type=str,
        default=None,
        help="Path to daemon PID file (default: /tmp/vram-reserve-cuda<device>.pid)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to daemon log file (default: /tmp/vram-reserve-cuda<device>.log)",
    )

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
    parser.add_argument(
        "--target-mode",
        choices=["self", "total"],
        default="self",
        help=(
            "How to interpret target size: 'self' means this script allocates the requested "
            "size; 'total' means GPU total used memory reaches target (default: self)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pid_file = args.pid_file or default_pid_file(args.device)
    log_file = args.log_file or default_log_file(args.device)

    if args.stop:
        stop_daemon(pid_file)
        return

    if args.gb is None and args.mb is None:
        raise ValueError("Either --gb or --mb must be provided unless using --stop.")

    if args.daemon and args.hold_seconds == 0:
        print("[WARN] --daemon with hold-seconds=0 means running until explicitly stopped.")

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
    print(f"[INFO] Target mode: {args.target_mode}")

    daemon_pid_file: str = ""
    stop_flag = {"stop": False}

    if args.daemon and not args._daemon_child:
        daemonize_process(pid_file=pid_file, log_file=log_file)
        return
    if args._daemon_child:
        daemon_pid_file = pid_file

    if target_bytes <= 0:
        raise ValueError("Target memory must be > 0")

    if target_bytes > total_bytes:
        print("[WARN] Target exceeds physical VRAM. Will allocate as much as possible.")

    tensors = allocate_target_bytes(
        device=device,
        target_bytes=target_bytes,
        chunk_mb=args.chunk_mb,
        target_mode=args.target_mode,
    )
    torch.cuda.synchronize(device)

    self_allocated = sum(t.numel() * t.element_size() for t in tensors)
    used = current_used_bytes(device)
    free = current_free_bytes(device)
    print(f"[INFO] Allocated chunks: {len(tensors)}")
    print(f"[INFO] Self allocated VRAM: {format_bytes(self_allocated)}")
    print(f"[INFO] Current used VRAM: {format_bytes(used)}")
    print(f"[INFO] Probe free VRAM: {format_bytes(free)}")

    interval = max(1, args.report_interval)
    start = time.time()

    def _on_term(signum: int, _frame: object) -> None:
        del signum
        stop_flag["stop"] = True

    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT, _on_term)

    try:
        if args.hold_seconds > 0:
            while True:
                elapsed = time.time() - start
                if stop_flag["stop"]:
                    print("[INFO] Received termination signal, releasing memory.")
                    break
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
                if stop_flag["stop"]:
                    print("[INFO] Received termination signal, releasing memory.")
                    break
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
        if daemon_pid_file and os.path.exists(daemon_pid_file):
            os.remove(daemon_pid_file)


if __name__ == "__main__":
    main()
