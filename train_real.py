# -*- coding: utf-8 -*-
"""Real hardware training aligned with Simulation_Train/trainOneGray.py."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import shutil
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
import tifffile as tiff

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import GrayConfig, PanelConfig, PathConfig, TrainConfig, get_capture_params
from models import Actor, Critic, device
from real_world_env import RealWorldEnv, compute_uniformity_reward_from_rel_error, logger
from rl_training_utils import (
    compute_reference_normalized_metrics,
    compute_step_quality_metrics,
    crop_center_region,
    select_better_step_snapshot,
    should_store_transition_for_replay,
)


FIXED_EFFECTIVE_STEPS = 4


def parse_args():
    parser = argparse.ArgumentParser(
        description="Single-gray real hardware DDPG training aligned with trainOneGray.py"
    )
    parser.add_argument("--gray", type=int, default=GrayConfig.DEFAULT_SINGLE_GRAY)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=FIXED_EFFECTIVE_STEPS)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-capacity", type=int, default=256)
    parser.add_argument("--learn-crop-size", type=int, default=200)
    parser.add_argument("--crops-per-transition", type=int, default=2)
    parser.add_argument("--prior-episodes", type=int, default=5)
    parser.add_argument("--prior-only-episodes", type=int, default=3)
    parser.add_argument("--prior-bootstrap-episodes", type=int, default=8)
    parser.add_argument("--critic-only-episodes", type=int, default=12)
    parser.add_argument("--bootstrap-prior-gains", type=str, default="0.5,1.0")
    parser.add_argument("--bootstrap-min-positive", type=int, default=2)
    parser.add_argument("--bootstrap-min-negative", type=int, default=4)
    parser.add_argument("--bootstrap-positive-reward-threshold", type=float, default=0.0005)
    parser.add_argument("--bootstrap-negative-reward-threshold", type=float, default=-0.005)
    parser.add_argument("--critic-warmup-updates", type=int, default=50)
    parser.add_argument("--actor-warmup-episodes", type=int, default=20,
                        help="Episodes where actor generates actions but does NOT update weights, letting critic adapt.")
    parser.add_argument("--actor-update-every", type=int, default=3)
    parser.add_argument("--critic-updates-per-step", type=int, default=2)
    parser.add_argument("--prior-gain", type=float, default=0.1)
    parser.add_argument("--prior-gamma", type=float, default=2.2)
    parser.add_argument("--residual-clip", type=float, default=0.05)
    parser.add_argument("--gain-limit-init", type=float, default=0.05)
    parser.add_argument("--gain-limit-final", type=float, default=0.30)
    parser.add_argument("--gain-limit-ramp-episodes", type=int, default=100)
    parser.add_argument("--gain-limit-ramp-delay", type=int, default=20,
                        help="Actor episodes to stay at gain_limit_init before ramp starts.")
    parser.add_argument("--gain-abs-weight", type=float, default=0.005)
    parser.add_argument("--gain-tv-weight", type=float, default=0.02)
    parser.add_argument("--gain-laplacian-weight", type=float, default=0.03)
    parser.add_argument("--step-gain-decay", type=float, default=0.5,
                        help="Per-step multiplicative decay for prior gain (step N gains = base * decay^N)")
    parser.add_argument("--residual-noise-init", type=float, default=0.005)
    parser.add_argument("--residual-noise-min", type=float, default=0.001)
    parser.add_argument("--noise-grid-rows", type=int, default=20)
    parser.add_argument("--noise-grid-cols", type=int, default=40)
    parser.add_argument("--rebound-negative-reward", type=float, default=0.5)
    parser.add_argument("--slice-grid-rows", type=int, default=10)
    parser.add_argument("--slice-grid-cols", type=int, default=5)
    parser.add_argument("--patches-per-transition", type=int, default=16)
    parser.add_argument("--random-crop-ratio", type=float, default=0.5)
    parser.add_argument("--large-crop-size", type=int, default=400)
    parser.add_argument("--large-crop-update-ratio", type=float, default=0.25)
    parser.add_argument("--large-crop-batch-size", type=int, default=2)
    parser.add_argument("--large-crops-per-transition", type=int, default=1)
    parser.add_argument("--freeze-target-per-run", action="store_true", default=True)
    parser.add_argument("--max-std-ratio-for-replay", type=float, default=1.08)
    parser.add_argument("--max-std-ratio-for-episode", type=float, default=1.15)
    parser.add_argument("--max-step-quality-rebound-ratio", type=float, default=1.05)
    parser.add_argument("--max-step-quality-rebound-abs", type=float, default=0.01)
    parser.add_argument("--quality-std-weight", type=float, default=1.0)
    parser.add_argument("--quality-mean-weight", type=float, default=1.2)
    parser.add_argument("--quality-p95-weight", type=float, default=0.4)
    parser.add_argument("--quality-low-weight", type=float, default=1.5)
    parser.add_argument("--quality-mid-weight", type=float, default=2.5)
    parser.add_argument("--quality-high-weight", type=float, default=1.2)
    parser.add_argument("--quality-tail-weight", type=float, default=1.0)
    parser.add_argument("--quality-profile-weight", type=float, default=0.8)
    parser.add_argument("--quality-grad-weight", type=float, default=0.4)
    parser.add_argument(
        "--reference-model-path",
        type=str,
        default="",
        help="Traditional reference TIFF. Empty means refData/W<gray>_model.tiff.",
    )
    parser.add_argument(
        "--reference-metric-gate",
        type=float,
        default=1.0,
        help="A reference-normalized metric passes when current/reference is at or below this value.",
    )
    parser.add_argument("--quality-delta-tv-weight", type=float, default=0.08)
    parser.add_argument("--quality-delta-abs-weight", type=float, default=0.03)
    parser.add_argument("--quality-clip-weight", type=float, default=2.0)
    parser.add_argument("--action-smoothness-weight", type=float, default=0.03)
    parser.add_argument("--action-laplacian-weight", type=float, default=0.05)
    parser.add_argument("--noise-init", type=float, default=0.05)
    parser.add_argument("--noise-min", type=float, default=0.005)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--lr-step-size", type=int, default=300)
    parser.add_argument("--lr-gamma", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reward-roi-size", type=int, default=0,
                        help="Center square used for reward/reference metrics. 0 defaults to train-roi-size; both 0 means full ROI.")
    parser.add_argument(
        "--min-replay-effective-action-abs-mean",
        type=float,
        default=1e-6,
        help="Skip replay insertion when the screen-visible effective action mean is below this threshold.",
    )
    parser.add_argument(
        "--min-replay-quantized-step-changed-ratio",
        type=float,
        default=1e-6,
        help="Skip replay insertion when too few pixels changed after 8-bit quantization.",
    )
    parser.add_argument("--no-quantization-carryover", action="store_true",
                        help="Disable per-step quantization error carry-over for cleaner multi-step results.")
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument(
        "--best-patience",
        type=int,
        default=80,
        help="Stop after this many post-best episodes without a new best quality. Use 0 to disable.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use a synthetic luma source for loop validation without hardware.",
    )
    parser.add_argument(
        "--hardware-preflight-only",
        action="store_true",
        help="Connect hardware, refresh static references, capture one reset frame, then exit without training.",
    )
    parser.add_argument(
        "--optimization-archive-path",
        type=str,
        default=os.path.join(PathConfig.SAVE_DIR, "optimization_history.jsonl"),
        help="JSONL archive for per-run optimization summaries.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def all_finite_values(*values) -> bool:
    for value in values:
        try:
            if not math.isfinite(float(value)):
                return False
        except (TypeError, ValueError):
            return False
    return True


def soft_update(target, source, tau: float) -> None:
    for source_param, target_param in zip(source.parameters(), target.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)


def action_for_replay(info, fallback_action):
    effective_action = info.get("effective_action_tensor")
    if isinstance(effective_action, torch.Tensor):
        return effective_action.detach()
    return fallback_action.detach()


def quantize_action_from_state(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    displayed_gray = torch.round(torch.clamp(state[:, 1:2] * 255.0, 0.0, 255.0))
    proposed_gray = torch.clamp(displayed_gray + action, 0.0, 255.0)
    quantized_proposed = torch.round(proposed_gray)
    return quantized_proposed - displayed_gray


def quantize_action_straight_through(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    quantized_action = quantize_action_from_state(state, action)
    return action + (quantized_action - action).detach()


def apply_gain_map_to_prior(prior_action: torch.Tensor, gain_map: torch.Tensor) -> torch.Tensor:
    scale = torch.clamp(1.0 + gain_map, 0.5, 3.0)
    return prior_action * scale


def bound_gain_map(raw_gain_map: torch.Tensor, limit: float) -> torch.Tensor:
    return torch.tanh(raw_gain_map) * max(float(limit), 0.0)


def make_structured_noise(
    reference: torch.Tensor,
    scale: float,
    grid_rows: int,
    grid_cols: int,
) -> torch.Tensor:
    if scale <= 0.0:
        return torch.zeros_like(reference)
    rows = min(max(1, int(grid_rows)), int(reference.shape[-2]))
    cols = min(max(1, int(grid_cols)), int(reference.shape[-1]))
    low_resolution = torch.randn(
        (reference.shape[0], reference.shape[1], rows, cols),
        device=reference.device,
        dtype=reference.dtype,
    )
    smooth = F.interpolate(
        low_resolution,
        size=reference.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    max_abs = torch.amax(torch.abs(smooth), dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    return smooth / max_abs * float(scale)


def parse_float_candidates(value: str) -> tuple[float, ...]:
    candidates = tuple(float(item.strip()) for item in str(value).split(",") if item.strip())
    if not candidates:
        raise ValueError("At least one bootstrap prior gain is required.")
    return candidates


def build_bootstrap_prior_action(
    base_prior_action: torch.Tensor,
    gain_candidates,
    attempt_index: int,
) -> torch.Tensor:
    candidates = tuple(float(value) for value in gain_candidates)
    if not candidates:
        raise ValueError("gain_candidates must not be empty")
    gain = candidates[int(attempt_index) % len(candidates)]
    return base_prior_action * gain


def select_bootstrap_prior_gain(
    gain_candidates,
    gain_quality_scores: Dict[float, List[float]],
    fallback_gain: float,
) -> float:
    scored = []
    for gain in tuple(float(value) for value in gain_candidates):
        scores = [float(score) for score in gain_quality_scores.get(gain, []) if math.isfinite(float(score))]
        if scores:
            scored.append((sum(scores) / len(scores), gain))
    if not scored:
        return float(fallback_gain)
    return min(scored, key=lambda item: item[0])[1]


def compute_improvement_reward(
    previous_rel_error: torch.Tensor,
    next_rel_error: torch.Tensor,
) -> torch.Tensor:
    previous_reward, _ = compute_uniformity_reward_from_rel_error(previous_rel_error)
    next_reward, _ = compute_uniformity_reward_from_rel_error(next_rel_error)
    denominator = torch.abs(previous_reward).clamp_min(1e-4)
    return torch.clamp((next_reward - previous_reward) / denominator, -1.0, 1.0)


def compute_reference_reward_info_from_luma(reference_luma: np.ndarray) -> Dict[str, float]:
    crop = np.asarray(reference_luma, dtype=np.float32)
    finite_crop = np.where(np.isfinite(crop), crop, np.nan)
    target = float(np.nanmean(finite_crop))
    if not math.isfinite(target) or abs(target) < 1e-6:
        raise ValueError("Reference luma crop has invalid or near-zero mean.")
    clean_crop = np.nan_to_num(finite_crop, nan=target, posinf=target, neginf=target)
    tensor = torch.from_numpy(clean_crop).float().unsqueeze(0).unsqueeze(0).to(device)
    rel_error = (tensor - target) / (target + 1e-6)
    _reward, reward_info = compute_uniformity_reward_from_rel_error(rel_error)
    return {key: float(value) for key, value in reward_info.items()}


def load_reference_reward_info(reference_path: str, crop_h: int, crop_w: int) -> Dict[str, float]:
    reference_image = tiff.imread(reference_path)
    reference_crop = crop_center_region(reference_image, height=crop_h, width=crop_w)
    return compute_reference_reward_info_from_luma(reference_crop)


def resolve_reference_model_path(gray: int, explicit_path: str) -> str:
    if str(explicit_path).strip():
        return os.path.abspath(str(explicit_path))
    return os.path.join(PathConfig.CUR_DIR, "refData", f"W{int(gray)}_model.tiff")


def resolve_gain_limit(args, episode: int) -> float:
    delay = max(0, int(args.gain_limit_ramp_delay))
    if int(episode) < delay:
        return float(args.gain_limit_init)
    effective_ep = int(episode) - delay
    ramp_episodes = max(1, int(args.gain_limit_ramp_episodes))
    progress = min(1.0, max(0, effective_ep) / ramp_episodes)
    return float(args.gain_limit_init) + progress * (
        float(args.gain_limit_final) - float(args.gain_limit_init)
    )


def resolve_training_phase(buffer, critic_update_count: int, actor_warmup_count: int, args) -> str:
    if (
        buffer.count_rewards_at_least(args.bootstrap_positive_reward_threshold)
        < max(0, int(args.bootstrap_min_positive))
        or buffer.count_rewards_at_most(args.bootstrap_negative_reward_threshold)
        < max(0, int(args.bootstrap_min_negative))
    ):
        return "bootstrap"
    if int(critic_update_count) < max(0, int(args.critic_warmup_updates)):
        return "critic_only"
    if int(actor_warmup_count) < max(0, int(args.actor_warmup_episodes)):
        return "actor_warmup"
    return "actor"


def compute_action_smoothness_loss(action: torch.Tensor) -> torch.Tensor:
    if action.shape[-1] <= 1 and action.shape[-2] <= 1:
        return action.new_tensor(0.0)
    loss = action.new_tensor(0.0)
    if action.shape[-1] > 1:
        loss = loss + torch.mean(torch.abs(action[..., :, 1:] - action[..., :, :-1]))
    if action.shape[-2] > 1:
        loss = loss + torch.mean(torch.abs(action[..., 1:, :] - action[..., :-1, :]))
    return loss


def compute_action_laplacian_loss(action: torch.Tensor) -> torch.Tensor:
    if action.shape[-1] < 3 or action.shape[-2] < 3:
        return action.new_tensor(0.0)
    center = action[..., 1:-1, 1:-1]
    laplacian = (
        -4.0 * center
        + action[..., :-2, 1:-1]
        + action[..., 2:, 1:-1]
        + action[..., 1:-1, :-2]
        + action[..., 1:-1, 2:]
    )
    return torch.mean(torch.abs(laplacian))


def build_traditional_prior_action(env: RealWorldEnv, gamma: float, gain: float) -> torch.Tensor:
    luma = torch.clamp(env.current_luma_map, min=1e-6)
    gray = float(env.current_gray_int)
    center_mean = float(env.target_mean_nit) if env.target_mean_nit is not None else float(torch.mean(luma).item())
    gray_ratio = max(gray / 255.0, 1e-6)
    scale = center_mean / (gray_ratio ** max(float(gamma), 1e-6))
    estimated_gray = torch.pow(torch.clamp(luma / max(scale, 1e-6), min=1e-6), 1.0 / max(float(gamma), 1e-6)) * 255.0
    target_gray_map = 2.0 * gray - estimated_gray
    current_gray_map = env.displayed_gray_map if env.displayed_gray_map is not None else env.current_gray_map
    if current_gray_map is None:
        current_gray_map = torch.full_like(target_gray_map, gray)
    elif current_gray_map.shape != target_gray_map.shape:
        current_gray_map = env._crop_to_train_roi(current_gray_map)
        if current_gray_map.shape != target_gray_map.shape:
            current_gray_map = torch.full_like(target_gray_map, gray)
    return (target_gray_map - current_gray_map) * float(gain)


def build_prior_action_from_state(state: torch.Tensor, gray: float, gamma: float, gain: float) -> torch.Tensor:
    norm_luma = torch.clamp(state[:, 0:1] + 1.0, min=1e-6)
    current_gray = torch.clamp(state[:, 1:2] * 255.0, 0.0, 255.0)
    gamma_value = max(float(gamma), 1e-6)
    estimated_gray = float(gray) * torch.pow(norm_luma, 1.0 / gamma_value)
    target_gray_map = 2.0 * float(gray) - estimated_gray
    return (target_gray_map - current_gray) * float(gain)


def _slice_transition_grid(
    state: torch.Tensor,
    action: torch.Tensor,
    next_state: torch.Tensor,
    patch_h: int = 200,
    patch_w: int = 200,
    rows: int = 10,
    cols: int = 5,
):
    height = state.shape[-2]
    width = state.shape[-1]
    patch_h = min(max(1, int(patch_h)), height)
    patch_w = min(max(1, int(patch_w)), width)
    row_positions = [min(index * patch_h, height - patch_h) for index in range(max(1, int(rows)))]
    col_positions = [min(index * patch_w, width - patch_w) for index in range(max(1, int(cols)))]
    patches = []
    for top in row_positions:
        for left in col_positions:
            patches.append(
                (
                    state[..., top : top + patch_h, left : left + patch_w].contiguous(),
                    action[..., top : top + patch_h, left : left + patch_w].contiguous(),
                    next_state[..., top : top + patch_h, left : left + patch_w].contiguous(),
                )
            )
    return patches


def _crop_transition_random(
    state: torch.Tensor,
    action: torch.Tensor,
    next_state: torch.Tensor,
    patch_h: int,
    patch_w: int,
):
    height = int(state.shape[-2])
    width = int(state.shape[-1])
    patch_h = min(max(1, int(patch_h)), height)
    patch_w = min(max(1, int(patch_w)), width)
    top = random.randint(0, height - patch_h) if height > patch_h else 0
    left = random.randint(0, width - patch_w) if width > patch_w else 0
    return (
        state[..., top : top + patch_h, left : left + patch_w].contiguous(),
        action[..., top : top + patch_h, left : left + patch_w].contiguous(),
        next_state[..., top : top + patch_h, left : left + patch_w].contiguous(),
    )


class RealReplayBuffer:
    """CPU replay buffer for full-resolution hardware frames.

    Full ROI transitions are stored in float16 on CPU. Sampling crops before
    moving tensors to GPU, so training memory follows the simulation-scale crop.
    """

    def __init__(self, capacity: int = 256):
        self.buf = deque(maxlen=capacity)
        self.patch_h = 200
        self.patch_w = 200
        self.grid_rows = 10
        self.grid_cols = 5

    def push(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        next_state: torch.Tensor,
        done: bool = False,
    ) -> None:
        self.buf.append(
            (
                state.detach().cpu().half(),
                action.detach().cpu().half(),
                float(reward),
                next_state.detach().cpu().half(),
                bool(done),
            )
        )

    def sample(
        self,
        batch_size: int,
        patches_per_transition: int,
        crop_h: Optional[int] = None,
        crop_w: Optional[int] = None,
        random_crop_ratio: float = 0.5,
    ):
        batch = random.sample(self.buf, batch_size)
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        patch_h = self.patch_h if crop_h is None else int(crop_h)
        patch_w = self.patch_w if crop_w is None else int(crop_w)
        random_crop_ratio = min(max(float(random_crop_ratio), 0.0), 1.0)
        for state, action, global_reward, next_state, done in batch:
            patches = _slice_transition_grid(
                state,
                action,
                next_state,
                patch_h=patch_h,
                patch_w=patch_w,
                rows=self.grid_rows,
                cols=self.grid_cols,
            )
            patches_per_transition = min(max(1, int(patches_per_transition)), len(patches))
            selected_patches = []
            for _ in range(patches_per_transition):
                if random.random() < random_crop_ratio:
                    selected_patches.append(
                        _crop_transition_random(
                            state,
                            action,
                            next_state,
                            patch_h=patch_h,
                            patch_w=patch_w,
                        )
                    )
                else:
                    selected_patches.append(random.choice(patches))
            for state_patch, action_patch, next_state_patch in selected_patches:
                state_patch = state_patch.float()
                action_patch = action_patch.float()
                next_state_patch = next_state_patch.float()
                reward_value = compute_improvement_reward(
                    state_patch[:, 0:1],
                    next_state_patch[:, 0:1],
                )
                reward_scalar = float(reward_value.detach().cpu().item())
                if done:
                    reward_scalar = min(reward_scalar, float(global_reward))
                states.append(state_patch)
                actions.append(action_patch)
                rewards.append(reward_scalar)
                next_states.append(next_state_patch)
                dones.append(float(done))

        return (
            torch.cat(states).float().to(device),
            torch.cat(actions).float().to(device),
            torch.tensor(rewards).float().unsqueeze(1).to(device),
            torch.cat(next_states).float().to(device),
            torch.tensor(dones).float().unsqueeze(1).to(device),
        )

    def __len__(self) -> int:
        return len(self.buf)

    @property
    def positive_count(self) -> int:
        return sum(1 for _state, _action, reward, _next_state, _done in self.buf if reward > 0.0)

    @property
    def negative_count(self) -> int:
        return sum(1 for _state, _action, reward, _next_state, _done in self.buf if reward < 0.0)

    def count_rewards_at_least(self, threshold: float) -> int:
        return sum(
            1
            for _state, _action, reward, _next_state, _done in self.buf
            if reward >= float(threshold)
        )

    def count_rewards_at_most(self, threshold: float) -> int:
        return sum(
            1
            for _state, _action, reward, _next_state, _done in self.buf
            if reward <= float(threshold)
        )


def _clear_directory_contents(dir_path: str) -> None:
    os.makedirs(dir_path, exist_ok=True)
    for entry in os.scandir(dir_path):
        if entry.is_dir():
            shutil.rmtree(entry.path)
        else:
            os.remove(entry.path)


def prepare_run_directory(gray: int) -> str:
    os.makedirs(PathConfig.SAVE_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = os.path.join(PathConfig.SAVE_DIR, f"run_{timestamp}_G{int(gray)}")
    os.makedirs(run_dir, exist_ok=True)
    PathConfig.ACTIVE_SAVE_DIR = run_dir

    for dir_path in (
        PathConfig.CAMERA_IMAGE_DIR,
        PathConfig.RESULT_DIR,
        PathConfig.TRANSFER_BMP_DIR,
    ):
        _clear_directory_contents(dir_path)

    return run_dir


def append_startup_trace(save_dir: str, message: str) -> None:
    trace_path = os.path.join(save_dir, "startup_trace.log")
    with open(trace_path, "a", encoding="utf-8") as trace_file:
        trace_file.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")


def attach_run_file_logger(save_dir: str) -> logging.FileHandler:
    file_handler = logging.FileHandler(
        os.path.join(save_dir, "train_real_detailed.log"),
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)
    return file_handler


def detach_run_file_logger(file_handler: logging.FileHandler) -> None:
    logger.removeHandler(file_handler)
    file_handler.close()


def classify_failure_reason(failure_reason: Optional[str]) -> Tuple[str, Optional[str]]:
    if not failure_reason:
        return "failed", None
    text = str(failure_reason)
    lower = text.lower()
    if "process fail" in lower or "定位图" in text or "瀹氫綅鍥" in text:
        return "hardware_locator_failure", "locator"
    if "predemura returned" in lower or "predemura timed out" in lower or "failed to execute predemura" in lower:
        return "hardware_predemura_failure", "predemura"
    if "no result tiff found" in lower:
        return "hardware_result_tiff_failure", "result_tiff"
    if "socket" in lower or "automation client" in lower or "capture failed for gray" in lower:
        return "hardware_socket_failure", "socket"
    if "failed to capture initial luma image after retries" in lower:
        return "hardware_not_ready", "reset_luma"
    return "failed", None


def append_optimization_archive(
    archive_path: str,
    *,
    args: argparse.Namespace,
    save_dir: str,
    best_std: float,
    best_quality_score: float,
    actor_updates: int,
    history: List[Dict[str, float]],
    status: str = "completed",
    failure_reason: Optional[str] = None,
) -> None:
    def _json_float(value: float) -> Optional[float]:
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None

    archive_dir = os.path.dirname(archive_path)
    if archive_dir:
        os.makedirs(archive_dir, exist_ok=True)

    final_row = history[-1] if history else {}
    best_row = min(history, key=lambda row: float(row.get("std", float("inf")))) if history else {}
    payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "direct_train_real",
        "dry_run": bool(args.dry_run),
        "gray": int(args.gray),
        "episodes": int(args.episodes),
        "requested_steps": int(args.steps),
        "effective_steps": int(FIXED_EFFECTIVE_STEPS),
        "batch_size": int(args.batch_size),
        "effective_batch_size": int(args.batch_size * max(1, args.patches_per_transition)),
        "learn_crop_size": int(args.learn_crop_size),
        "patches_per_transition": int(args.patches_per_transition),
        "slice_grid_rows": int(args.slice_grid_rows),
        "slice_grid_cols": int(args.slice_grid_cols),
        "buffer_capacity": int(args.buffer_capacity),
        "random_crop_ratio": float(args.random_crop_ratio),
        "large_crop_size": int(args.large_crop_size),
        "large_crop_update_ratio": float(args.large_crop_update_ratio),
        "bootstrap_prior_gains": str(args.bootstrap_prior_gains),
        "critic_warmup_updates": int(args.critic_warmup_updates),
        "actor_update_every": int(args.actor_update_every),
        "residual_clip": float(args.residual_clip),
        "residual_noise_init": float(args.residual_noise_init),
        "residual_noise_min": float(args.residual_noise_min),
        "reference_model_path": str(getattr(args, "reference_model_path", "")),
        "reference_metric_gate": float(getattr(args, "reference_metric_gate", 1.0)),
        "save_dir": save_dir,
        "status": str(status),
        "failure_stage": classify_failure_reason(failure_reason)[1] if failure_reason else None,
        "failure_reason": failure_reason,
        "history_rows": len(history),
        "actor_updates": int(actor_updates),
        "best_std": _json_float(best_std),
        "best_quality_score": _json_float(best_quality_score),
        "best_episode": int(best_row["episode"]) if best_row else None,
        "final_std": _json_float(final_row["std"]) if final_row else None,
        "final_quality_score": _json_float(final_row.get("quality_score", float("nan"))) if final_row else None,
    }
    with open(archive_path, "a", encoding="utf-8") as archive_file:
        archive_file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def save_history_csv(save_dir: str, history: List[Dict[str, float]]) -> None:
    if not history:
        return

    columns = [
        "episode",
        "gray",
        "std",
        "reward",
        "r_mse",
        "r_std",
        "r_low",
        "r_mid",
        "r_high",
        "r_tail",
        "r_profile",
        "r_grad",
        "r_mean",
        "quality_score",
        "quality_mean_loss",
        "quality_p95_abs_error",
        "quality_low_std",
        "quality_mid_abs_p99",
        "quality_high_abs_p99",
        "quality_tail_abs_p99",
        "quality_profile_loss",
        "quality_grad_loss",
        "quality_delta_tv",
        "quality_delta_abs_mean",
        "quality_clip_ratio",
        "reference_ratio_max",
        "reference_ratio_mean",
        "reference_all_pass",
        "reference_std_norm_ratio",
        "reference_low_std_ratio",
        "reference_mid_abs_p99_ratio",
        "reference_high_abs_p99_ratio",
        "reference_tail_abs_p99_ratio",
        "reference_profile_loss_ratio",
        "reference_grad_loss_ratio",
        "prior_abs_mean",
        "residual_abs_mean",
        "actual_action_abs_mean",
        "action_abs_mean",
        "effective_action_abs_mean",
        "quantized_delta_abs_mean",
        "quantized_step_changed_ratio",
        "effective_action_train_roi_abs_mean",
        "quantized_step_train_roi_changed_ratio",
        "quantized_step_train_roi_delta_abs_mean",
        "effective_action_reward_roi_abs_mean",
        "quantized_step_reward_roi_changed_ratio",
        "quantized_step_reward_roi_delta_abs_mean",
    ]
    with open(os.path.join(save_dir, "training_history.csv"), "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        for row in history:
            writer.writerow({column: row.get(column, "") for column in columns})


def save_training_summary(save_dir: str, history: List[Dict[str, float]]) -> None:
    if not history:
        return

    episodes = [item["episode"] for item in history]
    stds = [item["std"] for item in history]
    rewards = [item["reward"] for item in history]
    quality_scores = [item.get("quality_score", float("nan")) for item in history]
    r_mse = [item["r_mse"] for item in history]
    r_std = [item["r_std"] for item in history]
    r_low = [item.get("r_low", float("nan")) for item in history]
    r_mid = [item.get("r_mid", float("nan")) for item in history]
    r_high = [item.get("r_high", float("nan")) for item in history]
    r_tail = [item.get("r_tail", float("nan")) for item in history]
    r_grad = [item["r_grad"] for item in history]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(episodes, stds, color="tab:blue")
    axes[0, 0].set_title("Std over Episodes")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Std")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(episodes, rewards, color="tab:red")
    axes[0, 1].plot(episodes, quality_scores, color="tab:purple", label="quality_score")
    axes[0, 1].set_title("Reward / Quality over Episodes")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Reward / Quality")
    axes[0, 1].legend(fontsize="small")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(episodes, r_mse, label="r_mse")
    axes[1, 0].plot(episodes, r_std, label="r_std")
    axes[1, 0].plot(episodes, r_low, label="r_low")
    axes[1, 0].plot(episodes, r_mid, label="r_mid")
    axes[1, 0].plot(episodes, r_high, label="r_high")
    axes[1, 0].plot(episodes, r_tail, label="r_tail")
    axes[1, 0].plot(episodes, r_grad, label="r_grad")
    axes[1, 0].set_title("Reward Components")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].legend(fontsize="small")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(
        episodes,
        [item["quantized_delta_abs_mean"] for item in history],
        label="quantized_delta_abs_mean",
    )
    axes[1, 1].plot(
        episodes,
        [item["quantized_step_changed_ratio"] for item in history],
        label="quantized_step_changed_ratio",
    )
    axes[1, 1].set_title("8-bit Quantization")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].legend(fontsize="small")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_summary.png"), dpi=150)
    plt.close(fig)


def _mock_reset(env: RealWorldEnv, gray: int) -> torch.Tensor:
    env.set_target_gray(gray)
    env.episode_count += 1
    env.step_count = 0
    env.current_gray_map = torch.full(
        (1, 1, env.roi_h, env.roi_w),
        env.target_gray,
        device=device,
    )
    env.total_delta = torch.zeros_like(env.current_gray_map)
    env.displayed_gray_map = torch.round(env.current_gray_map.detach()).to(device)

    exposure, _gain = get_capture_params(gray)
    mock_std = max(0.5, exposure / 12000.0)
    mock_luma = np.random.normal(
        100.0 + gray * 0.5,
        mock_std,
        (env.roi_h, env.roi_w),
    ).astype(np.float32)
    env.current_luma_map = torch.from_numpy(mock_luma).unsqueeze(0).unsqueeze(0).to(device)
    if env.run_target_mean_nit is None or not env.freeze_target_per_run:
        env.run_target_mean_nit = float(torch.mean(env.current_luma_map).item())
        env.run_target_luma_map = torch.full_like(env.current_luma_map, env.run_target_mean_nit)
    env.target_mean_nit = env.run_target_mean_nit
    env.target_luma_map = env.run_target_luma_map
    env._update_baseline_std_metrics()
    return env._get_observation()


def _mock_step(env: RealWorldEnv, action: torch.Tensor):
    env.step_count += 1
    if env._train_roi is not None and action.shape[-2:] != (env.roi_h, env.roi_w):
        top, left, h, w = env._train_roi
        full_action = torch.zeros((1, 1, env.roi_h, env.roi_w), device=action.device, dtype=action.dtype)
        full_action[..., top:top + h, left:left + w] = action
        action = full_action
    safe_action = env._safety_check(action)
    previous_displayed = env.displayed_gray_map
    proposed_gray = torch.clamp(env.current_gray_map + safe_action, 0, 255)
    displayed_gray = torch.round(proposed_gray)
    effective_action = displayed_gray - previous_displayed

    env.current_gray_map = displayed_gray
    env.displayed_gray_map = displayed_gray
    env.total_delta = env.current_gray_map - env.target_gray

    base_std = max(env.initial_std * (1.0 - 0.05 * env.step_count), 0.5)
    mock_luma = np.random.normal(
        env.target_mean_nit,
        base_std,
        (env.roi_h, env.roi_w),
    ).astype(np.float32)
    env.current_luma_map = torch.from_numpy(mock_luma).unsqueeze(0).unsqueeze(0).to(device)

    reward, reward_info = env._compute_reward()
    info = {
        "step": env.step_count,
        "gray": env.current_gray_int,
        "std": float(reward_info.get("visual_std", float("nan"))),
        "raw_std": float(reward_info.get("real_std", float("nan"))),
        "std_ratio": float(reward_info.get("visual_std", float("nan"))) / max(float(env.run_target_norm_std or 0.0), 1e-6),
        "reward_info": reward_info,
        "action_abs_mean": float(torch.mean(torch.abs(safe_action)).item()),
        "effective_action_tensor": effective_action.detach(),
        "effective_action_abs_mean": float(torch.mean(torch.abs(effective_action)).item()),
        "quantized_delta_abs_mean": float(torch.mean(torch.abs(env.total_delta)).item()),
        "quantized_step_changed_ratio": float(torch.mean((effective_action != 0).float()).item()),
        "effective_action_train_roi_abs_mean": float(torch.mean(torch.abs(env._crop_to_train_roi(effective_action))).item()),
        "quantized_step_train_roi_changed_ratio": float(torch.mean((env._crop_to_train_roi(effective_action) != 0).float()).item()),
        "quantized_step_train_roi_delta_abs_mean": float(torch.mean(torch.abs(env._crop_to_train_roi(effective_action))).item()),
        "effective_action_reward_roi_abs_mean": float(torch.mean(torch.abs(env._crop_to_reward_roi(effective_action))).item()),
        "quantized_step_reward_roi_changed_ratio": float(torch.mean((env._crop_to_reward_roi(effective_action) != 0).float()).item()),
        "quantized_step_reward_roi_delta_abs_mean": float(torch.mean(torch.abs(env._crop_to_reward_roi(effective_action))).item()),
        "error": False,
        "capture_used_retry": False,
    }
    return env._get_observation(), reward, info


def run_real_training(args) -> None:
    set_seed(args.seed)
    save_dir = prepare_run_directory(args.gray)
    append_startup_trace(save_dir, "prepare_run_directory done")
    run_file_handler = attach_run_file_logger(save_dir)
    append_startup_trace(save_dir, "attach_run_file_logger done")

    append_startup_trace(save_dir, "construct RealWorldEnv start")
    env = RealWorldEnv(gray_candidates=[args.gray], freeze_target_per_run=args.freeze_target_per_run,
                       no_quantization_carryover=args.no_quantization_carryover)
    append_startup_trace(save_dir, "construct RealWorldEnv done")
    actor = Actor().to(device)
    target_actor = Actor().to(device)
    target_actor.load_state_dict(actor.state_dict())

    critic = Critic().to(device)
    target_critic = Critic().to(device)
    target_critic.load_state_dict(critic.state_dict())

    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    actor_scheduler = torch.optim.lr_scheduler.StepLR(
        actor_opt,
        step_size=args.lr_step_size,
        gamma=args.lr_gamma,
    )
    critic_scheduler = torch.optim.lr_scheduler.StepLR(
        critic_opt,
        step_size=args.lr_step_size,
        gamma=args.lr_gamma,
    )
    buffer = RealReplayBuffer(args.buffer_capacity)
    buffer.patch_h = args.learn_crop_size
    buffer.patch_w = args.learn_crop_size
    buffer.grid_rows = args.slice_grid_rows
    buffer.grid_cols = args.slice_grid_cols
    effective_steps = FIXED_EFFECTIVE_STEPS
    reference_model_path = resolve_reference_model_path(args.gray, args.reference_model_path)
    resolved_reward_roi_size = int(args.reward_roi_size) if int(args.reward_roi_size) > 0 else 0
    reference_crop_h = resolved_reward_roi_size if resolved_reward_roi_size > 0 else PanelConfig.ROI_HEIGHT
    reference_crop_w = resolved_reward_roi_size if resolved_reward_roi_size > 0 else PanelConfig.ROI_WIDTH
    reference_reward_info: Optional[Dict[str, float]] = None
    if os.path.exists(reference_model_path):
        reference_reward_info = load_reference_reward_info(
            reference_model_path,
            crop_h=reference_crop_h,
            crop_w=reference_crop_w,
        )
    else:
        logger.warning("Reference model TIFF not found; reference-normalized metrics disabled: %s", reference_model_path)

    history: List[Dict[str, float]] = []
    best_std = float("inf")
    best_quality_score = float("inf")
    best_actor_quality_score = float("inf")
    archive_status = "completed"
    archive_failure_reason: Optional[str] = None
    no_improve_counter = 0
    no_improve_after_best_counter = 0
    critic_update_count = 0
    total_actor_update_count = 0
    bootstrap_attempt_count = 0
    actor_phase_episode_count = 0
    actor_warmup_episode_count = 0
    bootstrap_gain_candidates = parse_float_candidates(args.bootstrap_prior_gains)
    bootstrap_gain_quality_scores = {gain: [] for gain in bootstrap_gain_candidates}
    selected_prior_gain = bootstrap_gain_candidates[0]
    abort_training_for_numeric_instability = False

    logger.info(
        "Start real training: gray=%s episodes=%s requested_steps=%s effective_steps=%s batch=%s effective_batch=%s patch=%sx%s patches_per_transition=%s grid=%sx%s random_crop_ratio=%.2f large_crop=%s large_update_ratio=%.2f buffer_capacity=%s bootstrap_gains=%s bootstrap_balance=%s/%s reward_thresholds=%+.3f/%+.3f critic_warmup_updates=%s actor_update_every=%s gain_limit=%.3f->%.3f step_gain_decay=%.3f residual_noise_init=%.4f residual_noise_min=%.4f reference=%s reference_gate=%.3f min_replay_effective=%.6f min_replay_changed=%.6f device=%s save_dir=%s",
        args.gray,
        args.episodes,
        args.steps,
        effective_steps,
        args.batch_size,
        args.batch_size * max(1, args.patches_per_transition),
        args.learn_crop_size,
        args.learn_crop_size,
        args.patches_per_transition,
        args.slice_grid_rows,
        args.slice_grid_cols,
        args.random_crop_ratio,
        args.large_crop_size,
        args.large_crop_update_ratio,
        args.buffer_capacity,
        bootstrap_gain_candidates,
        args.bootstrap_min_positive,
        args.bootstrap_min_negative,
        args.bootstrap_positive_reward_threshold,
        args.bootstrap_negative_reward_threshold,
        args.critic_warmup_updates,
        args.actor_update_every,
        args.gain_limit_init,
        args.gain_limit_final,
        args.step_gain_decay,
        args.residual_noise_init,
        args.residual_noise_min,
        reference_model_path if reference_reward_info else "disabled",
        args.reference_metric_gate,
        args.min_replay_effective_action_abs_mean,
        args.min_replay_quantized_step_changed_ratio,
        device,
        save_dir,
    )
    append_startup_trace(save_dir, "first logger.info emitted")

    try:
        if not args.dry_run:
            append_startup_trace(save_dir, "env.connect start")
            env.connect()
            append_startup_trace(save_dir, "env.connect done")
            append_startup_trace(save_dir, "ensure_static_references start")
            env.ensure_static_references(force_refresh=True)
            append_startup_trace(save_dir, "ensure_static_references done")
            if args.hardware_preflight_only:
                append_startup_trace(save_dir, "hardware_preflight reset start")
                env.reset(args.gray)
                append_startup_trace(save_dir, "hardware_preflight reset done")
                logger.info("Hardware preflight completed successfully for gray %s.", args.gray)
                return
        else:
            logger.info("Dry-run mode enabled. Hardware interaction is skipped.")

        for episode in range(args.episodes):
            state = _mock_reset(env, args.gray) if args.dry_run else env.reset(args.gray)
            ep_reward = 0.0
            ep_std = float("inf")
            last_info: Dict[str, object] = {"reward_info": {}}
            best_step_snapshot = None
            phase_name = resolve_training_phase(buffer, critic_update_count, actor_warmup_episode_count, args)
            bootstrap_phase = phase_name == "bootstrap"
            critic_only_phase = phase_name == "critic_only"
            actor_warmup_phase = phase_name == "actor_warmup"
            actor_enabled = phase_name in ("actor_warmup", "actor")
            gain_limit = resolve_gain_limit(args, actor_phase_episode_count)
            noise_scale = max(args.residual_noise_min, args.residual_noise_init * max(0.0, 1.0 - episode / max(1, args.episodes)))
            if actor_enabled and not critic_only_phase:
                if actor_warmup_phase:
                    actor_warmup_episode_count += 1
                else:
                    actor_phase_episode_count += 1
            did_actor_update_this_episode = False
            did_critic_update_this_episode = False
            episode_actor_updates = 0
            episode_critic_updates = 0
            numeric_instability_detected = False
            logger.info(
                "Ep %04d phase=%s replay=%d positive=%d negative=%d critic_updates_total=%d selected_prior_gain=%+.3f gain_limit=%.4f noise_scale=%.5f",
                episode + 1,
                phase_name,
                len(buffer),
                buffer.count_rewards_at_least(args.bootstrap_positive_reward_threshold),
                buffer.count_rewards_at_most(args.bootstrap_negative_reward_threshold),
                critic_update_count,
                selected_prior_gain,
                gain_limit,
                noise_scale,
            )

            for step_index in range(effective_steps):
                with torch.no_grad():
                    step_prior_gain = selected_prior_gain * (args.step_gain_decay ** step_index)
                    if bootstrap_phase:
                        base_prior_action = build_traditional_prior_action(
                            env,
                            args.prior_gamma,
                            1.0,
                        )
                        base_prior_action = env._crop_to_train_roi(base_prior_action)
                        active_prior_gain = bootstrap_gain_candidates[
                            bootstrap_attempt_count % len(bootstrap_gain_candidates)
                        ]
                        action = build_bootstrap_prior_action(
                            base_prior_action,
                            bootstrap_gain_candidates,
                            bootstrap_attempt_count,
                        )
                        bootstrap_attempt_count += 1
                        prior_action = action
                        gain_map = torch.full_like(base_prior_action, active_prior_gain)
                        noise = torch.zeros_like(prior_action)
                    elif critic_only_phase:
                        active_prior_gain = step_prior_gain
                        prior_action = build_traditional_prior_action(
                            env,
                            args.prior_gamma,
                            step_prior_gain,
                        )
                        prior_action = env._crop_to_train_roi(prior_action)
                        gain_map = torch.zeros_like(prior_action)
                        noise = make_structured_noise(
                            prior_action,
                            scale=noise_scale,
                            grid_rows=args.noise_grid_rows,
                            grid_cols=args.noise_grid_cols,
                        )
                        action = prior_action + noise
                    else:
                        active_prior_gain = step_prior_gain
                        prior_action = build_traditional_prior_action(
                            env,
                            args.prior_gamma,
                            step_prior_gain,
                        )
                        prior_action = env._crop_to_train_roi(prior_action)
                        residual = actor(state) * gain_limit
                        noise = make_structured_noise(
                            residual,
                            scale=noise_scale,
                            grid_rows=args.noise_grid_rows,
                            grid_cols=args.noise_grid_cols,
                        )
                        action = prior_action + residual + noise
                        gain_map = residual
                prior_abs_mean = float(torch.mean(torch.abs(prior_action)).item())
                residual_abs_mean = float(torch.mean(torch.abs(gain_map)).item())
                actual_action_abs_mean = float(torch.mean(torch.abs(action)).item())

                if args.dry_run:
                    next_state, reward, info = _mock_step(env, action)
                else:
                    next_state, reward, info = env.step(action)

                if info.get("error"):
                    logger.error(
                        "Episode %s step %s capture failed; skipping this transition.",
                        episode + 1,
                        step_index + 1,
                    )
                    continue

                if info.get("capture_used_retry"):
                    logger.warning(
                        "Episode %s step %s used capture retry; stopping episode before replay insertion.",
                        episode + 1,
                        step_index + 1,
                    )
                    last_info = info
                    break

                ep_std = float(info.get("std", float("inf")))
                baseline_norm_std = env.run_target_norm_std
                std_ratio = ep_std / max(baseline_norm_std, 1e-6)
                gray_snapshot = env.get_current_gray_snapshot()
                quantized_gray_snapshot = env.get_displayed_gray_snapshot()
                luma_snapshot = (
                    env.current_luma_map.detach().cpu().numpy()[0, 0]
                    if env.current_luma_map is not None
                    else None
                )
                quality_metrics = compute_step_quality_metrics(
                    info.get("reward_info", {}),
                    gray_map=gray_snapshot,
                    target_gray=float(args.gray),
                    quantized_gray_map=quantized_gray_snapshot,
                    std_weight=args.quality_std_weight,
                    mean_weight=args.quality_mean_weight,
                    p95_weight=args.quality_p95_weight,
                    low_weight=args.quality_low_weight,
                    mid_weight=args.quality_mid_weight,
                    high_weight=args.quality_high_weight,
                    tail_weight=args.quality_tail_weight,
                    profile_weight=args.quality_profile_weight,
                    grad_weight=args.quality_grad_weight,
                    delta_tv_weight=args.quality_delta_tv_weight,
                    delta_abs_weight=args.quality_delta_abs_weight,
                    clip_weight=args.quality_clip_weight,
                )
                reference_metrics = compute_reference_normalized_metrics(
                    info.get("reward_info", {}),
                    reference_reward_info,
                    gate=args.reference_metric_gate,
                )
                quality_metrics.update(reference_metrics)
                current_quality_score = float(quality_metrics["quality_score"])
                raw_std = float(info.get("raw_std", float("nan")))
                effective_action_abs_mean = float(info.get("effective_action_abs_mean", float("nan")))
                quantized_delta_abs_mean = float(info.get("quantized_delta_abs_mean", float("nan")))
                quantized_step_changed_ratio = float(info.get("quantized_step_changed_ratio", float("nan")))
                numeric_values_ok = all_finite_values(
                    reward,
                    ep_std,
                    raw_std,
                    std_ratio,
                    current_quality_score,
                    prior_abs_mean,
                    residual_abs_mean,
                    actual_action_abs_mean,
                    float(info.get("action_abs_mean", float("nan"))),
                    effective_action_abs_mean,
                    quantized_delta_abs_mean,
                    quantized_step_changed_ratio,
                )
                if not numeric_values_ok:
                    numeric_instability_detected = True
                    abort_training_for_numeric_instability = True
                    last_info = info
                    last_info["prior_abs_mean"] = prior_abs_mean
                    last_info["residual_abs_mean"] = residual_abs_mean
                    last_info["actual_action_abs_mean"] = actual_action_abs_mean
                    last_info["std_ratio"] = std_ratio
                    last_info["learning_reward"] = float("nan")
                    last_info["active_prior_gain"] = active_prior_gain
                    last_info.update(quality_metrics)
                    ep_std = float("inf")
                    logger.error(
                        "Numeric instability detected at episode %s step %s: reward=%s std=%s raw_std=%s quality=%s action_abs=%s effective_action_abs=%s quant_delta_abs=%s",
                        episode + 1,
                        step_index + 1,
                        reward,
                        ep_std,
                        raw_std,
                        current_quality_score,
                        actual_action_abs_mean,
                        effective_action_abs_mean,
                        quantized_delta_abs_mean,
                    )
                    break
                best_step_quality = (
                    float(best_step_snapshot["quality_score"])
                    if best_step_snapshot is not None
                    else current_quality_score
                )
                quality_rebound_ratio = current_quality_score / max(best_step_quality, 1e-6)
                quality_rebound_abs = current_quality_score - best_step_quality
                rebound_stop = False
                if best_step_snapshot is not None:
                    if quality_rebound_ratio > args.max_step_quality_rebound_ratio or quality_rebound_abs > args.max_step_quality_rebound_abs:
                        rebound_stop = True
                learning_reward = float(
                    compute_improvement_reward(
                        state[:, 0:1],
                        next_state[:, 0:1],
                    ).detach().cpu().item()
                )
                if rebound_stop:
                    learning_reward = min(learning_reward, -abs(args.rebound_negative_reward))
                unsafe_transition = std_ratio > args.max_std_ratio_for_replay
                if unsafe_transition:
                    learning_reward = min(learning_reward, -abs(args.rebound_negative_reward))
                transition_done = rebound_stop or unsafe_transition or step_index == effective_steps - 1
                last_info = info
                last_info["prior_abs_mean"] = prior_abs_mean
                last_info["residual_abs_mean"] = residual_abs_mean
                last_info["actual_action_abs_mean"] = actual_action_abs_mean
                last_info["std_ratio"] = std_ratio
                last_info["learning_reward"] = learning_reward
                last_info["active_prior_gain"] = active_prior_gain
                last_info.update(quality_metrics)
                replay_action = action_for_replay(info, action)
                store_transition = should_store_transition_for_replay(
                    info,
                    min_effective_action_abs_mean=args.min_replay_effective_action_abs_mean,
                    min_quantized_step_changed_ratio=args.min_replay_quantized_step_changed_ratio,
                ) or unsafe_transition or rebound_stop
                if store_transition:
                    if bootstrap_phase:
                        bootstrap_gain_quality_scores[active_prior_gain].append(current_quality_score)
                        selected_prior_gain = select_bootstrap_prior_gain(
                            bootstrap_gain_candidates,
                            bootstrap_gain_quality_scores,
                            fallback_gain=selected_prior_gain,
                        )
                    buffer.push(state, replay_action, learning_reward, next_state, done=transition_done)
                    ep_reward += learning_reward
                else:
                    logger.info(
                        "Episode %s step %s skipped replay no-op: effective_action_abs=%.6f quant_step_changed=%.6f thresholds=%.6f/%.6f",
                        episode + 1,
                        step_index + 1,
                        effective_action_abs_mean,
                        quantized_step_changed_ratio,
                        args.min_replay_effective_action_abs_mean,
                        args.min_replay_quantized_step_changed_ratio,
                    )
                state = next_state
                if store_transition:
                    best_step_snapshot = select_better_step_snapshot(
                        best_step_snapshot,
                        step=int(info.get("step", step_index + 1)),
                        std=ep_std,
                        reward=learning_reward,
                        gray_map=gray_snapshot,
                        luma_map=luma_snapshot,
                        quantized_gray_map=quantized_gray_snapshot,
                        quality_score=quality_metrics["quality_score"],
                        quality_metrics=quality_metrics,
                    )

                logger.info(
                    (
                        "Ep %04d step %02d/%02d learning_reward=%+.4f absolute_reward=%.3f std=%.6f raw_std=%.6f "
                        "quality=%.6f low=%.6f mid99=%.6f high99=%.6f tail99=%.6f grad=%.6f "
                        "prior_gain=%+.3f prior_abs_mean=%.6f residual_abs_mean=%.6f actual_action_abs_mean=%.6f "
                        "action_abs=%.6f effective_action_abs=%.6f quant_delta_abs=%.6f "
                        "quant_step_changed=%.6f reward_roi_effective=%.6f reward_roi_changed=%.6f "
                        "train_roi_effective=%.6f train_roi_changed=%.6f ref_ratio_max=%.6f ref_pass=%.0f store_replay=%s"
                    ),
                    episode + 1,
                    step_index + 1,
                    effective_steps,
                    learning_reward,
                    reward,
                    ep_std,
                    raw_std,
                    quality_metrics["quality_score"],
                    quality_metrics["quality_low_std"],
                    quality_metrics["quality_mid_abs_p99"],
                    quality_metrics["quality_high_abs_p99"],
                    quality_metrics["quality_tail_abs_p99"],
                    quality_metrics["quality_grad_loss"],
                    active_prior_gain,
                    prior_abs_mean,
                    residual_abs_mean,
                    actual_action_abs_mean,
                    float(info.get("action_abs_mean", float("nan"))),
                    effective_action_abs_mean,
                    quantized_delta_abs_mean,
                    quantized_step_changed_ratio,
                    float(info.get("effective_action_reward_roi_abs_mean", float("nan"))),
                    float(info.get("quantized_step_reward_roi_changed_ratio", float("nan"))),
                    float(info.get("effective_action_train_roi_abs_mean", float("nan"))),
                    float(info.get("quantized_step_train_roi_changed_ratio", float("nan"))),
                    float(quality_metrics.get("reference_ratio_max", float("nan"))),
                    float(quality_metrics.get("reference_all_pass", float("nan"))),
                    store_transition,
                )
                if unsafe_transition:
                    logger.warning(
                        "Episode %s step %s stored as terminal negative sample due to std_ratio=%.4f > %.4f",
                        episode + 1,
                        step_index + 1,
                        std_ratio,
                        args.max_std_ratio_for_replay,
                    )
                if rebound_stop:
                    logger.warning(
                        "Episode %s step %s terminated early due to quality rebound: current_quality=%.6f best_step_quality=%.6f",
                        episode + 1,
                        step_index + 1,
                        current_quality_score,
                        best_step_quality,
                    )
                    break
                if unsafe_transition:
                    break
                if bootstrap_phase:
                    logger.info("Bootstrap calibration uses one action per reset.")
                    break

                use_large_crop = (
                    args.large_crop_size > args.learn_crop_size
                    and random.random() < args.large_crop_update_ratio
                )
                sample_batch_size = args.large_crop_batch_size if use_large_crop else args.batch_size
                sample_crop_size = args.large_crop_size if use_large_crop else args.learn_crop_size
                sample_patches = (
                    args.large_crops_per_transition
                    if use_large_crop
                    else args.patches_per_transition
                )
                if not bootstrap_phase and len(buffer) > sample_batch_size:
                    for _critic_update in range(max(1, args.critic_updates_per_step)):
                        sb, ab, rb, nsb, db = buffer.sample(
                            sample_batch_size,
                            sample_patches,
                            crop_h=sample_crop_size,
                            crop_w=sample_crop_size,
                            random_crop_ratio=args.random_crop_ratio,
                        )
                        with torch.no_grad():
                            prior_nsb = build_prior_action_from_state(
                                nsb,
                                args.gray,
                                args.prior_gamma,
                                selected_prior_gain,
                            )
                            if actor_enabled:
                                target_residual = target_actor(nsb) * gain_limit
                            else:
                                target_residual = torch.zeros_like(prior_nsb)
                            tgt_a = quantize_action_from_state(
                                nsb,
                                prior_nsb + target_residual,
                            )
                            target_q = rb + args.gamma * (1.0 - db) * target_critic(nsb, tgt_a)

                        curr_q = critic(sb, ab)
                        loss_c = F.mse_loss(curr_q, target_q)
                        critic_opt.zero_grad()
                        loss_c.backward()
                        critic_opt.step()
                        critic_update_count += 1
                        episode_critic_updates += 1
                        did_critic_update_this_episode = True
                        soft_update(target_critic, critic, args.tau)

                        if actor_enabled and not actor_warmup_phase and critic_update_count % args.actor_update_every == 0:
                            prior_sb = build_prior_action_from_state(
                                sb,
                                args.gray,
                                args.prior_gamma,
                                selected_prior_gain,
                            )
                            residual = actor(sb) * gain_limit
                            pred_a = quantize_action_straight_through(
                                sb,
                                prior_sb + residual,
                            )
                            gain_map = residual
                            for critic_param in critic.parameters():
                                critic_param.requires_grad_(False)
                            policy_loss = -critic(sb, pred_a).mean()
                            gain_abs_loss = torch.mean(torch.abs(gain_map))
                            gain_tv_loss = compute_action_smoothness_loss(gain_map)
                            gain_laplacian_loss = compute_action_laplacian_loss(gain_map)
                            smoothness_loss = compute_action_smoothness_loss(pred_a)
                            laplacian_loss = compute_action_laplacian_loss(pred_a)
                            loss_a = (
                                policy_loss
                                + args.gain_abs_weight * gain_abs_loss
                                + args.gain_tv_weight * gain_tv_loss
                                + args.gain_laplacian_weight * gain_laplacian_loss
                                + args.action_smoothness_weight * smoothness_loss
                                + args.action_laplacian_weight * laplacian_loss
                            )
                            actor_opt.zero_grad()
                            loss_a.backward()
                            actor_opt.step()
                            for critic_param in critic.parameters():
                                critic_param.requires_grad_(True)
                            did_actor_update_this_episode = True
                            episode_actor_updates += 1
                            total_actor_update_count += 1
                            soft_update(target_actor, actor, args.tau)

                if std_ratio > args.max_std_ratio_for_episode:
                    logger.warning(
                        "Episode %s step %s terminated early due to std_ratio=%.4f > %.4f",
                        episode + 1,
                        step_index + 1,
                        std_ratio,
                        args.max_std_ratio_for_episode,
                    )
                    break

            if numeric_instability_detected:
                logger.error("Abort training after episode %s due to numeric instability.", episode + 1)

            if did_actor_update_this_episode:
                actor_scheduler.step()
            if did_critic_update_this_episode:
                critic_scheduler.step()

            reward_info = last_info.get("reward_info", {})
            history.append(
                {
                    "episode": float(episode + 1),
                    "gray": float(args.gray),
                    "std": ep_std,
                    "reward": ep_reward,
                    "r_mse": float(reward_info.get("r_mse", float("nan"))),
                    "r_std": float(reward_info.get("r_std", float("nan"))),
                    "r_low": float(reward_info.get("r_low", float("nan"))),
                    "r_mid": float(reward_info.get("r_mid", float("nan"))),
                    "r_high": float(reward_info.get("r_high", float("nan"))),
                    "r_tail": float(reward_info.get("r_tail", float("nan"))),
                    "r_profile": float(reward_info.get("r_profile", float("nan"))),
                    "r_grad": float(reward_info.get("r_grad", float("nan"))),
                    "r_mean": float(reward_info.get("r_mean", float("nan"))),
                    "quality_score": float(last_info.get("quality_score", float("nan"))),
                    "quality_mean_loss": float(last_info.get("quality_mean_loss", float("nan"))),
                    "quality_p95_abs_error": float(last_info.get("quality_p95_abs_error", float("nan"))),
                    "quality_low_std": float(last_info.get("quality_low_std", float("nan"))),
                    "quality_mid_abs_p99": float(last_info.get("quality_mid_abs_p99", float("nan"))),
                    "quality_high_abs_p99": float(last_info.get("quality_high_abs_p99", float("nan"))),
                    "quality_tail_abs_p99": float(last_info.get("quality_tail_abs_p99", float("nan"))),
                    "quality_profile_loss": float(last_info.get("quality_profile_loss", float("nan"))),
                    "quality_grad_loss": float(last_info.get("quality_grad_loss", float("nan"))),
                    "quality_delta_tv": float(last_info.get("quality_delta_tv", float("nan"))),
                    "quality_delta_abs_mean": float(last_info.get("quality_delta_abs_mean", float("nan"))),
                    "quality_clip_ratio": float(last_info.get("quality_clip_ratio", float("nan"))),
                    "reference_ratio_max": float(last_info.get("reference_ratio_max", float("nan"))),
                    "reference_ratio_mean": float(last_info.get("reference_ratio_mean", float("nan"))),
                    "reference_all_pass": float(last_info.get("reference_all_pass", float("nan"))),
                    "reference_std_norm_ratio": float(last_info.get("reference_std_norm_ratio", float("nan"))),
                    "reference_low_std_ratio": float(last_info.get("reference_low_std_ratio", float("nan"))),
                    "reference_mid_abs_p99_ratio": float(last_info.get("reference_mid_abs_p99_ratio", float("nan"))),
                    "reference_high_abs_p99_ratio": float(last_info.get("reference_high_abs_p99_ratio", float("nan"))),
                    "reference_tail_abs_p99_ratio": float(last_info.get("reference_tail_abs_p99_ratio", float("nan"))),
                    "reference_profile_loss_ratio": float(last_info.get("reference_profile_loss_ratio", float("nan"))),
                    "reference_grad_loss_ratio": float(last_info.get("reference_grad_loss_ratio", float("nan"))),
                    "prior_abs_mean": float(last_info.get("prior_abs_mean", float("nan"))),
                    "residual_abs_mean": float(last_info.get("residual_abs_mean", float("nan"))),
                    "actual_action_abs_mean": float(last_info.get("actual_action_abs_mean", float("nan"))),
                    "action_abs_mean": float(last_info.get("action_abs_mean", float("nan"))),
                    "effective_action_abs_mean": float(last_info.get("effective_action_abs_mean", float("nan"))),
                    "quantized_delta_abs_mean": float(last_info.get("quantized_delta_abs_mean", float("nan"))),
                    "quantized_step_changed_ratio": float(last_info.get("quantized_step_changed_ratio", float("nan"))),
                    "effective_action_train_roi_abs_mean": float(last_info.get("effective_action_train_roi_abs_mean", float("nan"))),
                    "quantized_step_train_roi_changed_ratio": float(last_info.get("quantized_step_train_roi_changed_ratio", float("nan"))),
                    "quantized_step_train_roi_delta_abs_mean": float(last_info.get("quantized_step_train_roi_delta_abs_mean", float("nan"))),
                    "effective_action_reward_roi_abs_mean": float(last_info.get("effective_action_reward_roi_abs_mean", float("nan"))),
                    "quantized_step_reward_roi_changed_ratio": float(last_info.get("quantized_step_reward_roi_changed_ratio", float("nan"))),
                    "quantized_step_reward_roi_delta_abs_mean": float(last_info.get("quantized_step_reward_roi_delta_abs_mean", float("nan"))),
                }
            )
            save_history_csv(save_dir, history)
            save_training_summary(save_dir, history)

            logger.info(
                (
                    "Ep %04d done: phase=%s critic_updates=%d actor_updates=%d std=%.6f quality=%.6f reward=%.3f "
                    "r_std=%.3f r_low=%.3f r_mid=%.3f r_high=%.3f r_tail=%.3f "
                    "r_profile=%.3f r_grad=%.3f r_mean=%.3f ref_ratio_max=%.6f ref_pass=%.0f"
                ),
                episode + 1,
                phase_name,
                episode_critic_updates,
                episode_actor_updates,
                ep_std,
                float(last_info.get("quality_score", float("nan"))),
                ep_reward,
                float(reward_info.get("r_std", float("nan"))),
                float(reward_info.get("r_low", float("nan"))),
                float(reward_info.get("r_mid", float("nan"))),
                float(reward_info.get("r_high", float("nan"))),
                float(reward_info.get("r_tail", float("nan"))),
                float(reward_info.get("r_profile", float("nan"))),
                float(reward_info.get("r_grad", float("nan"))),
                float(reward_info.get("r_mean", float("nan"))),
                float(last_info.get("reference_ratio_max", float("nan"))),
                float(last_info.get("reference_all_pass", float("nan"))),
            )

            best_step_std = float("inf")
            best_step_quality_score = float("inf")
            if best_step_snapshot is not None:
                best_step_std = float(best_step_snapshot["std"])
                best_step_quality_score = float(best_step_snapshot.get("quality_score", float("inf")))
            if best_step_snapshot is not None and best_step_quality_score < best_quality_score:
                best_std = best_step_std
                best_quality_score = best_step_quality_score
                no_improve_counter = 0
                no_improve_after_best_counter = 0
                best_table = best_step_snapshot["gray_map"] - float(args.gray)
                tiff.imwrite(
                    os.path.join(save_dir, f"Best_DemuraTable_Gray{args.gray}_Std{best_std:.3f}.tiff"),
                    best_table.astype(np.float32),
                )
                quantized_gray_map = best_step_snapshot.get("quantized_gray_map")
                if quantized_gray_map is not None:
                    bmp_path = os.path.join(save_dir, f"Best_Panel_Gray{args.gray}_Std{best_std:.3f}.bmp")
                    env.image_processor.render_panel_bmp(
                        np.asarray(quantized_gray_map, dtype=np.float32),
                        f"Best_G{args.gray}_S{best_std:.3f}",
                    )
                    if os.path.exists(bmp_path):
                        pass  # render_panel_bmp writes to TransferBmp/display_name.bmp
                    # Copy from TransferBmp to save_dir for archival
                    src_bmp = os.path.join(PathConfig.TRANSFER_BMP_DIR, f"Best_G{args.gray}_S{best_std:.3f}.bmp")
                    if os.path.exists(src_bmp):
                        shutil.copy2(src_bmp, bmp_path)
                        logger.info("Saved best panel BMP: %s", bmp_path)
                best_luma = best_step_snapshot.get("luma_map")
                if best_luma is None and env.current_luma_map is not None:
                    best_luma = env.current_luma_map.detach().cpu().numpy()[0, 0]
                if best_luma is not None:
                    tiff.imwrite(
                        os.path.join(save_dir, f"Best_Luma_Gray{args.gray}_Std{best_std:.3f}.tiff"),
                        np.asarray(best_luma, dtype=np.float32),
                    )
                logger.info(
                    "New best quality: %.6f std=%.6f low=%.6f mid99=%.6f high99=%.6f tail99=%.6f grad=%.6f delta_tv=%.6f ref_ratio_max=%.6f ref_pass=%.0f",
                    best_quality_score,
                    best_std,
                    float(best_step_snapshot.get("quality_low_std", float("nan"))),
                    float(best_step_snapshot.get("quality_mid_abs_p99", float("nan"))),
                    float(best_step_snapshot.get("quality_high_abs_p99", float("nan"))),
                    float(best_step_snapshot.get("quality_tail_abs_p99", float("nan"))),
                    float(best_step_snapshot.get("quality_grad_loss", float("nan"))),
                    float(best_step_snapshot.get("quality_delta_tv", float("nan"))),
                    float(best_step_snapshot.get("reference_ratio_max", float("nan"))),
                    float(best_step_snapshot.get("reference_all_pass", float("nan"))),
                )
            elif best_quality_score < float("inf"):
                no_improve_counter += 1
                no_improve_after_best_counter += 1

            if (
                actor_enabled
                and episode_actor_updates > 0
                and best_step_snapshot is not None
                and best_step_quality_score < best_actor_quality_score
            ):
                best_actor_quality_score = best_step_quality_score
                torch.save(actor.state_dict(), os.path.join(save_dir, "best_actor_real.pth"))
                logger.info(
                    "New best actor checkpoint: quality=%.6f phase=%s ref_ratio_max=%.6f ref_pass=%.0f",
                    best_actor_quality_score,
                    phase_name,
                    float(best_step_snapshot.get("reference_ratio_max", float("nan"))),
                    float(best_step_snapshot.get("reference_all_pass", float("nan"))),
                )

            if args.patience > 0 and no_improve_counter >= args.patience:
                logger.info("Early stop: no quality improvement for %s episodes.", args.patience)
                break
            if args.best_patience > 0 and no_improve_after_best_counter >= args.best_patience:
                logger.info(
                    "Early stop: no post-best quality improvement for %s episodes.",
                    args.best_patience,
                )
                break
            if abort_training_for_numeric_instability:
                logger.error("Early stop: numeric instability detected.")
                break

    except KeyboardInterrupt:
        archive_status = "interrupted"
        archive_failure_reason = "keyboard_interrupt"
        logger.info("Training interrupted by user.")
    except Exception as exc:
        archive_failure_reason = str(exc)
        failure_text = archive_failure_reason.lower()
        archive_status, _ = classify_failure_reason(archive_failure_reason)
        if "process fail" in failure_text or "定位图" in archive_failure_reason:
            archive_status = "hardware_locator_failure"
        elif "Failed to capture initial luma image after retries" in archive_failure_reason:
            archive_status = "hardware_not_ready"
        archive_status, _ = classify_failure_reason(archive_failure_reason)
        logger.error("Training failed: %s", exc, exc_info=True)
        raise
    finally:
        if total_actor_update_count > 0:
            torch.save(actor.state_dict(), os.path.join(save_dir, "final_actor_real.pth"))
        else:
            logger.warning("Actor received no optimizer updates; no actor checkpoint was saved.")
        save_history_csv(save_dir, history)
        save_training_summary(save_dir, history)
        append_optimization_archive(
            args.optimization_archive_path,
            args=args,
            save_dir=save_dir,
            best_std=best_std,
            best_quality_score=best_quality_score,
            actor_updates=total_actor_update_count,
            history=history,
            status=archive_status,
            failure_reason=archive_failure_reason,
        )
        logger.info(
            "Training finished. Actor updates: %d Best quality: %.6f Best std: %.6f",
            total_actor_update_count,
            best_quality_score,
            best_std,
        )
        if not args.dry_run:
            env.close()
        detach_run_file_logger(run_file_handler)


def train(args) -> None:
    run_real_training(args)


if __name__ == "__main__":
    train(parse_args())
