from __future__ import annotations

import argparse
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional


PREPARED_RUN_RE = re.compile(r"Prepared run directory:\s+(?P<run_dir>.+?)\s+\|")
BASELINE_RE = re.compile(
    r"Episode (?P<episode>\d+) baseline at gray (?P<gray>\d+): "
    r"(?:mean|target_nit)=(?P<baseline_ref>[\d.]+) "
    r"(?:visual_std|norm_std)=(?P<visual_std>[\d.]+)"
)
STEP_RE = re.compile(
    r"\[Step\s+(?P<step>\d+)/(?P<step_total>\d+)\] reward=(?P<reward>[-\d.]+), "
    r"visual_std=(?P<visual_std>[\d.]+), raw_std=(?P<raw_std>[\d.]+), "
    r"std_ratio=(?P<std_ratio>[\d.]+), action_abs_mean=(?P<action_abs_mean>[\d.]+), "
    r"effective_action_abs_mean=(?P<effective_action_abs_mean>[\d.]+)"
)
DONE_RE = re.compile(
    r"Episode (?P<episode>\d+) done\. gray=(?P<gray>\d+) weight=(?P<weight>[\d.]+) std=(?P<std>[\d.]+) reward=(?P<reward>[-\d.]+)"
)
BEST_RE = re.compile(r"New per-gray best for G(?P<gray>\d+): (?P<std>[\d.]+)")
STOP_RE = re.compile(r"stopped early because std_ratio=(?P<std_ratio>[\d.]+)")


@dataclass
class CycleConfig:
    prior_steps: int
    residual_action_clip: float
    baseline_deviation_weight: float
    skip_replay_std_ratio: float
    learn_start_size: int
    actor_update_every: int
    actor_learn_start_updates: int
    min_action_abs_mean: float
    max_action_scale_boost: float
    max_std_ratio: float
    policy_action_lowpass_kernel: int
    policy_action_lowpass_passes: int
    random_warmup_episodes: int


@dataclass
class CycleResult:
    cycle_index: int
    run_dir: str
    return_code: int
    baseline_std: Optional[float]
    best_std: Optional[float]
    best_effective_action_abs_mean: float
    worst_std_ratio: float
    early_stop_count: int
    improvement_hit: bool
    stop_reason: str


def default_cycle_config() -> CycleConfig:
    return CycleConfig(
        prior_steps=8,
        residual_action_clip=0.50,
        baseline_deviation_weight=0.0,
        skip_replay_std_ratio=1.15,
        learn_start_size=2,
        actor_update_every=1,
        actor_learn_start_updates=4,
        min_action_abs_mean=0.05,
        max_action_scale_boost=4.0,
        max_std_ratio=1.20,
        policy_action_lowpass_kernel=31,
        policy_action_lowpass_passes=2,
        random_warmup_episodes=1,
    )


def write_status(status_path: Path, payload: Dict[str, object]) -> None:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_status_text(status_text_path: Path, payload: Dict[str, object]) -> None:
    lines = [
        f"phase: {payload.get('phase')}",
        f"cycle: {payload.get('cycle_index')}/{payload.get('max_cycles')}",
        f"run_dir: {payload.get('run_dir', '')}",
        f"latest_episode: {payload.get('latest_episode', '')}",
        f"latest_step: {payload.get('latest_step', '')}",
        f"baseline_std: {payload.get('baseline_std', '')}",
        f"best_std: {payload.get('best_std', '')}",
        f"current_visual_std: {payload.get('current_visual_std', '')}",
        f"current_std_ratio: {payload.get('current_std_ratio', '')}",
        f"current_effective_action_abs_mean: {payload.get('current_effective_action_abs_mean', '')}",
        f"worst_std_ratio: {payload.get('worst_std_ratio', '')}",
        f"early_stop_count: {payload.get('early_stop_count', '')}",
        f"stop_reason: {payload.get('stop_reason', '')}",
        f"next_plan: {payload.get('next_plan', '')}",
        f"updated_at: {payload.get('updated_at', '')}",
    ]
    status_text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def relay_stream(stream, sink: queue.Queue[str]) -> None:
    try:
        for line in iter(stream.readline, ""):
            if not line:
                break
            sink.put(line.rstrip("\n"))
    finally:
        stream.close()


def adjust_cycle_config(config: CycleConfig, result: CycleResult) -> CycleConfig:
    new_config = CycleConfig(**asdict(config))
    if result.best_std is not None and result.baseline_std is not None:
        improvement_ratio = result.best_std / max(result.baseline_std, 1e-6)
        if improvement_ratio <= 0.70:
            new_config.skip_replay_std_ratio = min(1.30, max(new_config.skip_replay_std_ratio, 1.18))
            new_config.actor_learn_start_updates = min(40, max(new_config.actor_learn_start_updates, 24))
            return new_config
        if improvement_ratio <= 0.90:
            new_config.min_action_abs_mean = max(0.03, new_config.min_action_abs_mean * 0.95)
            return new_config

    if result.best_effective_action_abs_mean > 0.10 or result.worst_std_ratio >= new_config.max_std_ratio:
        new_config.prior_steps = min(12, new_config.prior_steps + 1)
        new_config.residual_action_clip = max(0.18, new_config.residual_action_clip * 0.75)
        new_config.skip_replay_std_ratio = max(1.03, new_config.skip_replay_std_ratio * 0.98)
        new_config.actor_learn_start_updates = min(120, new_config.actor_learn_start_updates + 10)
        new_config.min_action_abs_mean = max(0.01, new_config.min_action_abs_mean * 0.8)
        new_config.max_action_scale_boost = max(1.5, new_config.max_action_scale_boost * 0.8)
    elif result.best_effective_action_abs_mean < 0.005:
        new_config.prior_steps = max(3, new_config.prior_steps - 1)
        new_config.residual_action_clip = min(0.75, max(0.50, new_config.residual_action_clip * 1.15))
        new_config.min_action_abs_mean = min(0.10, new_config.min_action_abs_mean * 1.25)
        new_config.max_action_scale_boost = min(6.0, new_config.max_action_scale_boost * 1.25)
        new_config.learn_start_size = min(8, max(2, new_config.learn_start_size))

    return new_config


def build_train_command(args: argparse.Namespace, cycle_config: CycleConfig) -> List[str]:
    return [
        args.python,
        "train_real.py",
        "--gray",
        str(args.gray),
        "--episodes",
        str(args.episodes),
        "--steps",
        str(args.steps),
        "--prior-steps",
        str(cycle_config.prior_steps),
        "--no-pretrained",
        "--no-wait-for-enter",
        "--no-confirm",
        "--policy-action-lowpass-kernel",
        str(cycle_config.policy_action_lowpass_kernel),
        "--policy-action-lowpass-passes",
        str(cycle_config.policy_action_lowpass_passes),
        "--residual-action-clip",
        str(cycle_config.residual_action_clip),
        "--skip-replay-std-ratio",
        str(cycle_config.skip_replay_std_ratio),
        "--learn-start-size",
        str(cycle_config.learn_start_size),
        "--actor-update-every",
        str(cycle_config.actor_update_every),
        "--actor-learn-start-updates",
        str(cycle_config.actor_learn_start_updates),
        "--discard-initial-episodes",
        str(args.discard_initial_episodes),
        "--max-std-ratio",
        str(cycle_config.max_std_ratio),
        "--min-action-abs-mean",
        str(cycle_config.min_action_abs_mean),
        "--max-action-scale-boost",
        str(cycle_config.max_action_scale_boost),
        "--random-warmup-episodes",
        str(cycle_config.random_warmup_episodes),
    ]


def run_cycle(
    args: argparse.Namespace,
    cycle_index: int,
    cycle_config: CycleConfig,
    status_path: Path,
    status_text_path: Path,
) -> CycleResult:
    command = build_train_command(args, cycle_config)
    process = subprocess.Popen(
        command,
        cwd=str(args.workdir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="ignore",
        bufsize=1,
    )
    output_queue: queue.Queue[str] = queue.Queue()
    relay_thread = threading.Thread(target=relay_stream, args=(process.stdout, output_queue), daemon=True)
    relay_thread.start()

    run_dir = ""
    baseline_std = None
    best_std = None
    latest_episode = 0
    latest_step = 0
    current_visual_std = None
    current_std_ratio = 0.0
    current_effective = 0.0
    worst_std_ratio = 0.0
    best_effective = 0.0
    early_stop_count = 0
    stop_reason = ""
    improvement_hit = False

    while True:
        try:
            line = output_queue.get(timeout=0.5)
            print(line, flush=True)

            prepared_match = PREPARED_RUN_RE.search(line)
            if prepared_match:
                run_dir = prepared_match.group("run_dir").strip()

            baseline_match = BASELINE_RE.search(line)
            if baseline_match:
                latest_episode = int(baseline_match.group("episode"))
                baseline_std = float(baseline_match.group("visual_std"))

            step_match = STEP_RE.search(line)
            if step_match:
                latest_step = int(step_match.group("step"))
                current_visual_std = float(step_match.group("visual_std"))
                current_std_ratio = float(step_match.group("std_ratio"))
                current_effective = float(step_match.group("effective_action_abs_mean"))
                worst_std_ratio = max(worst_std_ratio, current_std_ratio)
                best_effective = max(best_effective, current_effective)
                if latest_episode > args.discard_initial_episodes:
                    if current_std_ratio >= args.degrade_limit:
                        early_stop_count += 1
                    else:
                        early_stop_count = 0
                    if early_stop_count >= args.degrade_patience:
                        stop_reason = (
                            f"std_ratio {current_std_ratio:.3f} exceeded limit {args.degrade_limit:.3f} "
                            f"for {early_stop_count} consecutive monitored steps"
                        )
                        process.terminate()

            best_match = BEST_RE.search(line)
            if best_match:
                candidate_best = float(best_match.group("std"))
                best_std = candidate_best if best_std is None else min(best_std, candidate_best)
                if args.improve_target is not None and best_std <= args.improve_target:
                    improvement_hit = True

            done_match = DONE_RE.search(line)
            if done_match and best_std is None:
                done_episode = int(done_match.group("episode"))
                if done_episode > args.discard_initial_episodes:
                    best_std = float(done_match.group("std"))

            stop_match = STOP_RE.search(line)
            if stop_match:
                worst_std_ratio = max(worst_std_ratio, float(stop_match.group("std_ratio")))

            payload = {
                "phase": "running",
                "cycle_index": cycle_index,
                "max_cycles": args.max_cycles,
                "pid": process.pid,
                "run_dir": run_dir,
                "latest_episode": latest_episode,
                "latest_step": latest_step,
                "baseline_std": baseline_std,
                "best_std": best_std,
                "current_visual_std": current_visual_std,
                "current_std_ratio": current_std_ratio,
                "current_effective_action_abs_mean": current_effective,
                "worst_std_ratio": worst_std_ratio,
                "early_stop_count": early_stop_count,
                "stop_reason": stop_reason,
                "next_plan": "",
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "cycle_config": asdict(cycle_config),
            }
            write_status(status_path, payload)
            write_status_text(status_text_path, payload)
        except queue.Empty:
            if process.poll() is not None:
                break

    return_code = process.wait()
    if not stop_reason:
        stop_reason = "training process exited normally" if return_code == 0 else f"training process exited with code {return_code}"

    payload = {
        "phase": "cycle_completed",
        "cycle_index": cycle_index,
        "max_cycles": args.max_cycles,
        "pid": process.pid,
        "run_dir": run_dir,
        "latest_episode": latest_episode,
        "latest_step": latest_step,
        "baseline_std": baseline_std,
        "best_std": best_std,
        "current_visual_std": current_visual_std,
        "current_std_ratio": current_std_ratio,
        "current_effective_action_abs_mean": current_effective,
        "worst_std_ratio": worst_std_ratio,
        "early_stop_count": early_stop_count,
        "stop_reason": stop_reason,
        "next_plan": "",
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cycle_config": asdict(cycle_config),
        "return_code": return_code,
        "improvement_hit": improvement_hit,
    }
    write_status(status_path, payload)
    write_status_text(status_text_path, payload)

    return CycleResult(
        cycle_index=cycle_index,
        run_dir=run_dir,
        return_code=return_code,
        baseline_std=baseline_std,
        best_std=best_std,
        best_effective_action_abs_mean=best_effective,
        worst_std_ratio=worst_std_ratio,
        early_stop_count=early_stop_count,
        improvement_hit=improvement_hit,
        stop_reason=stop_reason,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Iterative launcher for conservative real-world demura training.")
    parser.add_argument("--python", required=True, help="Python executable for the training environment.")
    parser.add_argument("--workdir", type=Path, default=Path.cwd())
    parser.add_argument("--gray", type=int, default=16)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--traditional-bmp", default="")
    parser.add_argument("--discard-initial-episodes", type=int, default=1)
    parser.add_argument("--max-cycles", type=int, default=3)
    parser.add_argument("--degrade-limit", type=float, default=1.22)
    parser.add_argument("--degrade-patience", type=int, default=2)
    parser.add_argument("--improve-target", type=float, default=0.1100)
    parser.add_argument("--status-path", type=Path, default=Path("RealWorld_Train") / "live_status.json")
    parser.add_argument("--status-text-path", type=Path, default=Path("RealWorld_Train") / "live_status.txt")
    parser.add_argument("--history-path", type=Path, default=Path("RealWorld_Train") / "live_history.json")
    parser.add_argument("--archive-history-path", type=Path, default=Path("RealWorld_Train") / "optimization_history.jsonl")
    parser.add_argument("--initial-prior-steps", type=int, default=0)
    parser.add_argument("--initial-min-action-abs-mean", type=float, default=0.0)
    parser.add_argument("--initial-skip-replay-std-ratio", type=float, default=0.0)
    parser.add_argument("--initial-residual-action-clip", type=float, default=0.0)
    parser.add_argument("--initial-learn-start-size", type=int, default=0)
    parser.add_argument("--initial-actor-update-every", type=int, default=0)
    parser.add_argument("--initial-actor-learn-start-updates", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cycle_config = default_cycle_config()
    if args.initial_prior_steps > 0:
        cycle_config.prior_steps = args.initial_prior_steps
    if args.initial_min_action_abs_mean > 0:
        cycle_config.min_action_abs_mean = args.initial_min_action_abs_mean
    if args.initial_skip_replay_std_ratio > 0:
        cycle_config.skip_replay_std_ratio = args.initial_skip_replay_std_ratio
    if args.initial_residual_action_clip > 0:
        cycle_config.residual_action_clip = args.initial_residual_action_clip
    if args.initial_learn_start_size > 0:
        cycle_config.learn_start_size = args.initial_learn_start_size
    if args.initial_actor_update_every > 0:
        cycle_config.actor_update_every = args.initial_actor_update_every
    if args.initial_actor_learn_start_updates > 0:
        cycle_config.actor_learn_start_updates = args.initial_actor_learn_start_updates
    history: List[Dict[str, object]] = []

    initial_payload = {
        "phase": "starting",
        "cycle_index": 0,
        "max_cycles": args.max_cycles,
        "run_dir": "",
        "latest_episode": 0,
        "latest_step": 0,
        "baseline_std": None,
        "best_std": None,
        "current_visual_std": None,
        "current_std_ratio": None,
        "current_effective_action_abs_mean": None,
        "worst_std_ratio": None,
        "early_stop_count": 0,
        "stop_reason": "",
        "next_plan": "launch first conservative residual cycle",
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cycle_config": asdict(cycle_config),
    }
    write_status(args.status_path, initial_payload)
    write_status_text(args.status_text_path, initial_payload)

    for cycle_index in range(1, args.max_cycles + 1):
        print(f"=== Iteration cycle {cycle_index}/{args.max_cycles} ===", flush=True)
        print(f"Cycle config: {json.dumps(asdict(cycle_config), ensure_ascii=False)}", flush=True)
        result = run_cycle(args, cycle_index, cycle_config, args.status_path, args.status_text_path)
        history.append(
            {
                "cycle_index": cycle_index,
                "config": asdict(cycle_config),
                "result": asdict(result),
            }
        )
        args.history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        with args.archive_history_path.open("a", encoding="utf-8") as archive_handle:
            archive_handle.write(
                json.dumps(
                    {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "workdir": str(args.workdir),
                        "gray": args.gray,
                        "episodes": args.episodes,
                        "steps": args.steps,
                        "cycle_index": cycle_index,
                        "config": asdict(cycle_config),
                        "result": asdict(result),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

        if result.improvement_hit:
            final_payload = {
                "phase": "completed",
                "cycle_index": cycle_index,
                "max_cycles": args.max_cycles,
                "run_dir": result.run_dir,
                "latest_episode": "",
                "latest_step": "",
                "baseline_std": result.baseline_std,
                "best_std": result.best_std,
                "current_visual_std": "",
                "current_std_ratio": "",
                "current_effective_action_abs_mean": result.best_effective_action_abs_mean,
                "worst_std_ratio": result.worst_std_ratio,
                "early_stop_count": result.early_stop_count,
                "stop_reason": "improvement target reached",
                "next_plan": "hold current config and inspect saved best table",
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "cycle_config": asdict(cycle_config),
            }
            write_status(args.status_path, final_payload)
            write_status_text(args.status_text_path, final_payload)
            return 0

        if cycle_index < args.max_cycles:
            next_config = adjust_cycle_config(cycle_config, result)
            transition_payload = {
                "phase": "reconfiguring",
                "cycle_index": cycle_index,
                "max_cycles": args.max_cycles,
                "run_dir": result.run_dir,
                "latest_episode": "",
                "latest_step": "",
                "baseline_std": result.baseline_std,
                "best_std": result.best_std,
                "current_visual_std": "",
                "current_std_ratio": "",
                "current_effective_action_abs_mean": result.best_effective_action_abs_mean,
                "worst_std_ratio": result.worst_std_ratio,
                "early_stop_count": result.early_stop_count,
                "stop_reason": result.stop_reason,
                "next_plan": f"next config -> {json.dumps(asdict(next_config), ensure_ascii=False)}",
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "cycle_config": asdict(next_config),
            }
            write_status(args.status_path, transition_payload)
            write_status_text(args.status_text_path, transition_payload)
            cycle_config = next_config

    final_result = history[-1]["result"] if history else {}
    final_payload = {
        "phase": "completed",
        "cycle_index": len(history),
        "max_cycles": args.max_cycles,
        "run_dir": final_result.get("run_dir", ""),
        "latest_episode": "",
        "latest_step": "",
        "baseline_std": final_result.get("baseline_std"),
        "best_std": final_result.get("best_std"),
        "current_visual_std": "",
        "current_std_ratio": "",
        "current_effective_action_abs_mean": final_result.get("best_effective_action_abs_mean"),
        "worst_std_ratio": final_result.get("worst_std_ratio"),
        "early_stop_count": final_result.get("early_stop_count"),
        "stop_reason": final_result.get("stop_reason", "iteration budget exhausted"),
        "next_plan": "inspect live_history.json and choose a tighter residual schedule if needed",
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cycle_config": asdict(cycle_config),
    }
    write_status(args.status_path, final_payload)
    write_status_text(args.status_text_path, final_payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
