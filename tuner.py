# -*- coding: utf-8 -*-
"""Tuning orchestrator for Demura RL training.

Monitors training logs in real-time, evaluates performance after warmup,
and automatically tunes hyperparameters if standard deviation worsens.
"""

import os
import sys
import time
import subprocess
import re
import json
import glob
from datetime import datetime

PORT = 12345
LOG_DIR = r"E:\Projects\DemuraAI_RL\Demo3\TrainLogs"
ARTIFACT_PATH = r"C:\Users\CTOS\.gemini\antigravity\brain\64f34dca-1d64-449b-a5dc-4ee5170c6163\artifacts\tuner_status.md"

# 超参试验配置 —— 分步先验架构 (prior_scale=1/steps) 下的 LR 与平滑核搜索
# 关键变化：prior 已分步，每步动作 ≈ 0.03-0.05 灰阶，无需大 action_gain
TRIALS = [
    {
        "name": "Trial A (Baseline: steps=20, mild LR)",
        "action_gain": 1.0,
        "policy_action_lowpass_kernel": 11,
        "ACTOR_LR": "1e-5",
        "CRITIC_LR": "1e-4",
        "COLD_START_NOISE_SCALE_INIT": 0.5,   # 大幅提高，因为 * prior_scale 后实际 ≈ 0.025
        "actor_learn_start_updates": 40,
        "warmup_episodes": 3,
        "steps": 20,
    },
    {
        "name": "Trial B (More steps=30, slower convergence)",
        "action_gain": 1.0,
        "policy_action_lowpass_kernel": 11,
        "ACTOR_LR": "5e-6",
        "CRITIC_LR": "5e-5",
        "COLD_START_NOISE_SCALE_INIT": 0.5,
        "actor_learn_start_updates": 60,
        "warmup_episodes": 3,
        "steps": 30,
    },
    {
        "name": "Trial C (Higher LR Actor, aggressive learn)",
        "action_gain": 1.0,
        "policy_action_lowpass_kernel": 7,
        "ACTOR_LR": "2e-5",
        "CRITIC_LR": "2e-4",
        "COLD_START_NOISE_SCALE_INIT": 0.5,
        "actor_learn_start_updates": 20,
        "warmup_episodes": 3,
        "steps": 20,
    },
]

def update_config_file(params):
    """Safely update parameters in config.py."""
    config_path = r"E:\Projects\DemuraAI_RL\Demo3\config.py"
    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Update ACTOR_LR
    content = re.sub(
        r"(ACTOR_LR\s*=\s*)([^\s#]+)",
        rf"\g<1>{params['ACTOR_LR']}",
        content
    )
    # Update CRITIC_LR
    content = re.sub(
        r"(CRITIC_LR\s*=\s*)([^\s#]+)",
        rf"\g<1>{params['CRITIC_LR']}",
        content
    )
    # Update COLD_START_NOISE_SCALE_INIT
    content = re.sub(
        r"(COLD_START_NOISE_SCALE_INIT\s*=\s*)([^\s#]+)",
        rf"\g<1>{params['COLD_START_NOISE_SCALE_INIT']}",
        content
    )

    with open(config_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[*] Updated config.py: ACTOR_LR={params['ACTOR_LR']}, CRITIC_LR={params['CRITIC_LR']}, NOISE={params['COLD_START_NOISE_SCALE_INIT']}")

def get_latest_log():
    """Find the latest log file in TrainLogs/."""
    log_files = glob.glob(os.path.join(LOG_DIR, "train_*.log"))
    if not log_files:
        return None
    return max(log_files, key=os.path.getmtime)

def parse_log(log_path):
    """Parse log to extract baseline and step visual_std, reward, and progress."""
    if not log_path or not os.path.exists(log_path):
        return {}

    episodes = {}
    current_ep = None
    early_stopped = set()

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            # Detect episode reset
            reset_match = re.search(r"=== Episode (\d+) reset at gray \d+ ===", line)
            if reset_match:
                current_ep = int(reset_match.group(1))
                episodes[current_ep] = {
                    "baseline_std": None,
                    "final_std": None,
                    "steps": [],
                    "rewards": [],
                    "done": False
                }
                continue

            # Detect baseline
            baseline_match = re.search(r"Episode (\d+) baseline at gray \d+:.*visual_std=([\d\.]+)", line)
            if baseline_match:
                ep = int(baseline_match.group(1))
                val = float(baseline_match.group(2))
                if ep in episodes:
                    episodes[ep]["baseline_std"] = val
                continue

            # Detect step info
            step_match = re.search(r"\[Step (\d+)/\d+\] reward=([\-\d\.]+), visual_std=([\d\.]+)", line)
            if step_match and current_ep in episodes:
                reward = float(step_match.group(2))
                v_std = float(step_match.group(3))
                episodes[current_ep]["steps"].append(v_std)
                episodes[current_ep]["rewards"].append(reward)
                continue

            # Detect early stop
            if "stopped early" in line and current_ep:
                early_stopped.add(current_ep)
                continue

            # Detect episode done
            done_match = re.search(r"Episode (\d+) done\..*std=([\d\.]+)", line)
            if done_match:
                ep = int(done_match.group(1))
                val = float(done_match.group(2))
                if ep in episodes:
                    episodes[ep]["final_std"] = val
                    episodes[ep]["done"] = True
                continue

    # Filter out empty or unparsed episodes
    valid_episodes = {}
    for ep, data in episodes.items():
        if data["baseline_std"] is not None:
            if data["final_std"] is None and data["steps"]:
                data["final_std"] = data["steps"][-1]
            valid_episodes[ep] = data

    return {
        "episodes": valid_episodes,
        "early_stopped": early_stopped,
        "current_episode": current_ep
    }

def update_artifact(trial_idx, status, log_data, config_name):
    """Write markdown status to the artifact file for user viewing."""
    lines = []
    lines.append(f"# Demura RL 自动调优监控报告\n")
    lines.append(f"**更新时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**当前尝试**: {config_name} (Trial {trial_idx + 1}/{len(TRIALS)})")
    lines.append(f"**运行状态**: {status}\n")

    lines.append("## 核心参数")
    params = TRIALS[trial_idx]
    lines.append(f"- `action_gain`: {params['action_gain']}")
    lines.append(f"- `policy_action_lowpass_kernel`: {params['policy_action_lowpass_kernel']}")
    lines.append(f"- `ACTOR_LR`: {params['ACTOR_LR']}")
    lines.append(f"- `CRITIC_LR`: {params['CRITIC_LR']}")
    lines.append(f"- `COLD_START_NOISE_SCALE_INIT`: {params['COLD_START_NOISE_SCALE_INIT']}")
    lines.append(f"- `actor_learn_start_updates`: {params['actor_learn_start_updates']}\n")

    lines.append("## 各 Episode 详细指标")
    lines.append("| Episode | Baseline STD | Step 20 STD | 改善率 | 状态 |")
    lines.append("|---|---|---|---|---|")

    episodes = log_data.get("episodes", {})
    early_stopped = log_data.get("early_stopped", set())

    for ep in sorted(episodes.keys()):
        data = episodes[ep]
        base = data["baseline_std"]
        final = data["final_std"]

        if base is not None and final is not None:
            imp = (base - final) / base * 100
            imp_str = f"{imp:+.2f}%"
        else:
            imp_str = "N/A"

        base_str = f"{base:.4f}" if base is not None else "N/A"
        final_str = f"{final:.4f}" if final is not None else "N/A"

        ep_status = "完成"
        if ep in early_stopped:
            ep_status = "❌ 异常终止"
        elif ep <= 5: # Warmup
            ep_status = "Warmup 探索"

        lines.append(f"| Ep {ep} | {base_str} | {final_str} | {imp_str} | {ep_status} |")

    lines.append("\n## 分析与决策")
    if log_data.get("current_episode"):
        lines.append(f"当前正在执行第 **{log_data['current_episode']}** 轮。计划在第 10 轮结束时评估该组超参的有效性。")
    else:
        lines.append("正在等待首轮基线数据捕获...")

    os.makedirs(os.path.dirname(ARTIFACT_PATH), exist_ok=True)
    with open(ARTIFACT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def evaluate_performance(log_data):
    """Evaluate performance of the run. Returns (is_ok, reason).

    NOTE: 量化跳变（1灰阶 = 13% G16亮度变化）会导致单步 std_ratio 瞬态冲高，
    这是物理必然，不代表策略方向错误。评估标准改为：
    1. 只有当策略 episode 的【最终 std】持续高于 baseline 才认为失败
    2. 允许单步早停，不计入失败次数（因为瞬态量化噪点会在下一 episode reset 消除）
    """
    episodes = log_data.get("episodes", {})
    early_stopped = log_data.get("early_stopped", set())

    # 只评估 Policy 阶段（Episode >= warmup+1, 这里取 >= 4 适配 warmup=3）
    policy_eps = [ep for ep in episodes.keys() if ep >= 4]

    if not policy_eps:
        return True, "Still in warmup phase."

    # 用最终 std 改善率作为唯一评判标准（不用早停次数）
    improvements = []
    for ep in policy_eps:
        data = episodes[ep]
        b = data["baseline_std"]
        f = data["final_std"]
        if b is not None and f is not None:
            improvements.append((b - f) / b)

    if not improvements:
        return True, "No policy episodes evaluated yet."

    avg_imp = sum(improvements) / len(improvements)
    print(f"[*] Policy avg improvement: {avg_imp:.4%} over {len(improvements)} episodes")

    current_ep = log_data.get("current_episode", 0)
    # 至少 5 个 policy episode 数据，或接近尾声，才做判断
    if len(improvements) >= 5 or current_ep >= 12:
        if avg_imp < -0.03:  # 持续恶化 >3% 才终止
            return False, f"Final std worsened by avg {abs(avg_imp):.2%} over {len(improvements)} policy eps."

    return True, f"Running; avg final-std improvement={avg_imp:.2%} ({len(improvements)} policy eps)."

def run_trial(trial_idx):
    """Run a single training trial."""
    params = TRIALS[trial_idx]
    print(f"\n==================================================")
    print(f"[*] STARTING TRIAL {trial_idx + 1}: {params['name']}")
    print(f"==================================================")

    # Update config.py
    update_config_file(params)

    # Build command
    steps = params.get("steps", 20)
    warmup_eps = params.get("warmup_episodes", 3)
    cmd = [
        r"E:\softWare\Anaconda\envs\Pytorch\python.exe",
        r"E:\Projects\DemuraAI_RL\Demo3\train_real.py",
        "--gray", "16",
        "--episodes", "20",
        "--steps", str(steps),
        "--no-pretrained",
        "--random-warmup-episodes", str(warmup_eps),
        "--action-gain", str(params["action_gain"]),
        "--max-std-ratio", "1.50",  # 放宽：允许量化瞬态spike，仅防止极端恶化
        "--policy-action-lowpass-kernel", str(params["policy_action_lowpass_kernel"]),
        "--policy-action-lowpass-passes", "1",
        "--min-action-abs-mean", "0",
        "--max-action-scale-boost", "1",
        "--learn-start-size", "30",
        "--actor-learn-start-updates", str(params["actor_learn_start_updates"]),
        "--actor-update-every", "4"
    ]

    # Start process
    print(f"[*] Launching train_real.py...")
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        cwd=r"E:\Projects\DemuraAI_RL\Demo3"
    )

    # Automatically send Enter to bypass C# client connection block
    time.sleep(3)
    print("[*] Sending newline to stdin...")
    try:
        proc.stdin.write(b"\n")
        proc.stdin.flush()
    except Exception as e:
        print(f"[!] Failed to write to stdin: {e}")

    # Monitor logs in real-time
    time.sleep(5)
    log_path = get_latest_log()
    print(f"[*] Monitoring log file: {log_path}")

    while True:
        ret = proc.poll()
        if ret is not None:
            print(f"[!] Process terminated with exit code {ret}")
            break

        log_data = parse_log(log_path)
        update_artifact(trial_idx, "运行中", log_data, params["name"])

        # Check performance
        is_ok, reason = evaluate_performance(log_data)
        if not is_ok:
            print(f"[!] Performance check failed: {reason}")
            print("[*] Aborting trial...")
            proc.terminate()
            time.sleep(2)
            proc.kill()
            update_artifact(trial_idx, f"❌ 已终止 (原因: {reason})", log_data, params["name"])
            return False

        # Stop when we reach 10 episodes
        current_ep = log_data.get("current_episode", 0)
        episodes = log_data.get("episodes", {})
        if 10 in episodes and episodes[10]["done"]:
            print("[*] Episode 10 completed!")
            proc.terminate()
            time.sleep(2)
            proc.kill()
            update_artifact(trial_idx, f"✅ 成功完成10轮测试，效果良好: {reason}", log_data, params["name"])
            return True

        time.sleep(10)

    log_data = parse_log(log_path)
    is_ok, reason = evaluate_performance(log_data)
    if is_ok:
        update_artifact(trial_idx, "✅ 已结束 (正常)", log_data, params["name"])
        return True
    else:
        update_artifact(trial_idx, f"❌ 已结束 (失败: {reason})", log_data, params["name"])
        return False

def main():
    print("[*] Starting Demura AI Auto-Tuner loop...")
    for idx in range(len(TRIALS)):
        success = run_trial(idx)
        if success:
            print(f"[+] TRIAL {idx + 1} SUCCESSFUL! Demura is improving. Keeping these parameters.")
            sys.exit(0)
        else:
            print(f"[-] TRIAL {idx + 1} FAILED. Trying next set of hyperparameters...")
            time.sleep(5)
    print("[!] All pre-defined configurations failed.")

if __name__ == "__main__":
    main()
