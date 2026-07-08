from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


STEP_RE = re.compile(
    r"\[Step\s+(?P<step>\d+)/(?P<step_total>\d+)\] reward=(?P<reward>[-\d.]+), "
    r"visual_std=(?P<visual_std>[\d.]+), raw_std=(?P<raw_std>[\d.]+), "
    r"std_ratio=(?P<std_ratio>[\d.]+), action_abs_mean=(?P<action_abs_mean>[\d.]+), "
    r"effective_action_abs_mean=(?P<effective_action_abs_mean>[\d.]+)"
)
BASELINE_RE = re.compile(
    r"Episode (?P<episode>\d+) baseline at gray (?P<gray>\d+): mean=(?P<mean>[\d.]+) visual_std=(?P<visual_std>[\d.]+)"
)
DONE_RE = re.compile(
    r"Episode (?P<episode>\d+) done\. gray=(?P<gray>\d+) weight=(?P<weight>[\d.]+) std=(?P<std>[\d.]+) reward=(?P<reward>[-\d.]+)"
)
BEST_RE = re.compile(r"New per-gray best for G(?P<gray>\d+): (?P<std>[\d.]+)")


def latest_log(log_dir: Path) -> Optional[Path]:
    files = sorted(log_dir.glob("train_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def process_running(pid: Optional[int]) -> bool:
    if pid is None:
        return True
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def stop_process(pid: int) -> None:
    if sys.platform.startswith("win"):
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        os.kill(pid, signal.SIGTERM)


def monitor(log_path: Path, pid: Optional[int], poll_seconds: float, degrade_limit: float, degrade_patience: int, improve_target: Optional[float]) -> int:
    last_size = 0
    degrade_count = 0
    best_std = float("inf")
    last_baseline_std = None
    current_episode = None

    print(f"Monitoring: {log_path}")
    while True:
        if log_path.exists():
            with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
                handle.seek(last_size)
                chunk = handle.read()
                last_size = handle.tell()

            for raw_line in chunk.splitlines():
                line = raw_line.strip()
                baseline_match = BASELINE_RE.search(line)
                if baseline_match:
                    current_episode = int(baseline_match.group("episode"))
                    last_baseline_std = float(baseline_match.group("visual_std"))
                    print(f"[ep{current_episode}] baseline_std={last_baseline_std:.4f}")
                    continue

                step_match = STEP_RE.search(line)
                if step_match:
                    current_episode = current_episode or 0
                    visual_std = float(step_match.group("visual_std"))
                    std_ratio = float(step_match.group("std_ratio"))
                    effective = float(step_match.group("effective_action_abs_mean"))
                    print(
                        f"[ep{current_episode} step {step_match.group('step')}/{step_match.group('step_total')}] "
                        f"std={visual_std:.4f} ratio={std_ratio:.3f} eff={effective:.4f}"
                    )
                    if std_ratio >= degrade_limit:
                        degrade_count += 1
                    else:
                        degrade_count = 0
                    if pid is not None and degrade_count >= degrade_patience:
                        print(
                            f"Stopping training: std_ratio exceeded {degrade_limit:.3f} for {degrade_count} consecutive steps."
                        )
                        stop_process(pid)
                        return 2
                    continue

                best_match = BEST_RE.search(line)
                if best_match:
                    best_std = min(best_std, float(best_match.group("std")))
                    print(f"[best] gray={best_match.group('gray')} std={float(best_match.group('std')):.4f}")
                    if improve_target is not None and best_std <= improve_target:
                        print(f"Improvement target reached: best_std={best_std:.4f} <= {improve_target:.4f}")
                    continue

                done_match = DONE_RE.search(line)
                if done_match:
                    episode = int(done_match.group("episode"))
                    std = float(done_match.group("std"))
                    reward = float(done_match.group("reward"))
                    baseline_text = f"{last_baseline_std:.4f}" if last_baseline_std is not None else "nan"
                    print(f"[ep{episode} done] std={std:.4f} reward={reward:.3f} baseline={baseline_text}")
                    continue

        if pid is not None and not process_running(pid):
            print("Observed training process exit.")
            return 0

        time.sleep(poll_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor real-world training logs and optionally stop a bad run.")
    parser.add_argument("--log-dir", type=Path, default=Path("TrainLogs"))
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--pid", type=int, default=None)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--degrade-limit", type=float, default=1.25)
    parser.add_argument("--degrade-patience", type=int, default=2)
    parser.add_argument("--improve-target", type=float, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log_path = args.log_file or latest_log(args.log_dir)
    if log_path is None or not log_path.exists():
        print("No training log found to monitor.", file=sys.stderr)
        return 1
    return monitor(
        log_path=log_path,
        pid=args.pid,
        poll_seconds=args.poll_seconds,
        degrade_limit=args.degrade_limit,
        degrade_patience=args.degrade_patience,
        improve_target=args.improve_target,
    )


if __name__ == "__main__":
    raise SystemExit(main())
