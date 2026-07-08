from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
from PIL import Image

from config import PanelConfig


REFERENCE_METRIC_KEYS = (
    "std_norm",
    "low_std",
    "mid_abs_p99",
    "high_abs_p99",
    "tail_abs_p99",
    "profile_loss",
    "grad_loss",
)


def quantize_effective_delta(displayed_gray_map: np.ndarray, proposed_gray_map: np.ndarray) -> np.ndarray:
    displayed = np.clip(np.round(displayed_gray_map.astype(np.float32)), 0.0, 255.0)
    proposed = np.clip(np.round(proposed_gray_map.astype(np.float32)), 0.0, 255.0)
    return proposed - displayed


def load_traditional_delta_map(reference_bmp_path: str, target_gray: float) -> np.ndarray:
    with Image.open(reference_bmp_path) as image:
        gray = np.array(image.convert("L"), dtype=np.float32)

    y0 = PanelConfig.ROI_START_Y
    x0 = PanelConfig.ROI_START_X
    y1 = y0 + PanelConfig.ROI_HEIGHT
    x1 = x0 + PanelConfig.ROI_WIDTH
    roi = gray[y0:y1, x0:x1]
    expected_shape = (PanelConfig.ROI_HEIGHT, PanelConfig.ROI_WIDTH)
    if roi.shape != expected_shape:
        raise ValueError(f"Traditional ROI shape mismatch: got {roi.shape}, expected {expected_shape}")
    return roi - float(target_gray)


def crop_center_region(image: np.ndarray, height: int, width: int) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim < 2:
        raise ValueError(f"Reference image must be at least 2D, got shape {arr.shape}")
    if arr.ndim > 2:
        arr = np.squeeze(arr)
        if arr.ndim > 2:
            arr = arr[..., 0]
    crop_h = min(max(1, int(height)), int(arr.shape[-2]))
    crop_w = min(max(1, int(width)), int(arr.shape[-1]))
    top = (int(arr.shape[-2]) - crop_h) // 2
    left = (int(arr.shape[-1]) - crop_w) // 2
    return arr[top : top + crop_h, left : left + crop_w].astype(np.float32, copy=True)


def compute_reference_normalized_metrics(
    current_info: Dict[str, float],
    reference_info: Optional[Dict[str, float]],
    *,
    gate: float = 1.0,
) -> Dict[str, float]:
    if not reference_info:
        return {}

    metrics: Dict[str, float] = {}
    ratios = []
    for key in REFERENCE_METRIC_KEYS:
        current = float(current_info.get(key, float("nan")))
        reference = float(reference_info.get(key, float("nan")))
        if not math.isfinite(current) or not math.isfinite(reference) or abs(reference) < 1e-12:
            continue
        ratio = current / max(reference, 1e-12)
        metrics[f"reference_{key}_ratio"] = float(ratio)
        ratios.append(float(ratio))

    if ratios:
        ratio_max = max(ratios)
        ratio_mean = sum(ratios) / len(ratios)
        metrics["reference_ratio_max"] = float(ratio_max)
        metrics["reference_ratio_mean"] = float(ratio_mean)
        metrics["reference_all_pass"] = 1.0 if ratio_max <= float(gate) else 0.0
    return metrics


def should_store_transition_for_replay(
    info: Dict[str, object],
    *,
    min_effective_action_abs_mean: float,
    min_quantized_step_changed_ratio: float,
) -> bool:
    effective_abs = float(info.get("effective_action_abs_mean", 0.0) or 0.0)
    changed_ratio = float(info.get("quantized_step_changed_ratio", 0.0) or 0.0)
    return (
        effective_abs >= float(min_effective_action_abs_mean)
        and changed_ratio >= float(min_quantized_step_changed_ratio)
    )


def _mean_abs_spatial_diff(arr: np.ndarray) -> float:
    arr = arr.astype(np.float32, copy=False)
    parts = []
    if arr.shape[0] > 1:
        parts.append(np.mean(np.abs(np.diff(arr, axis=0))))
    if arr.shape[1] > 1:
        parts.append(np.mean(np.abs(np.diff(arr, axis=1))))
    if not parts:
        return 0.0
    return float(np.mean(parts))


def compute_step_quality_metrics(
    reward_info: Dict[str, float],
    *,
    gray_map: np.ndarray,
    target_gray: float,
    quantized_gray_map: Optional[np.ndarray] = None,
    std_weight: float = 1.0,
    mean_weight: float = 0.7,
    p95_weight: float = 0.0,
    low_weight: float = 1.5,
    mid_weight: float = 2.5,
    high_weight: float = 1.2,
    tail_weight: float = 0.6,
    profile_weight: float = 0.4,
    grad_weight: float = 0.4,
    delta_tv_weight: float = 0.08,
    delta_abs_weight: float = 0.03,
    clip_weight: float = 2.0,
) -> Dict[str, float]:
    """Compute a lower-is-better visual quality score for selecting saved results.

    The training reward can remain exploration-oriented; this score is stricter for
    checkpoint selection, so a lower std cannot hide mean drift, local outliers, or
    rough compensation maps.
    """
    std_norm = float(reward_info.get("std_norm", reward_info.get("visual_std", float("inf"))))
    mean_loss = float(reward_info.get("mean_loss", 0.0))
    p95_abs_error = float(reward_info.get("p95_abs_error", mean_loss + std_norm))
    low_std = float(reward_info.get("low_std", std_norm))
    mid_abs_p99 = float(reward_info.get("mid_abs_p99", 0.0))
    high_abs_p99 = float(reward_info.get("high_abs_p99", 0.0))
    tail_abs_p99 = float(reward_info.get("tail_abs_p99", p95_abs_error))
    profile_loss = float(reward_info.get("profile_loss", 0.0))
    grad_loss = float(reward_info.get("grad_loss", 0.0))

    selected_gray = quantized_gray_map if quantized_gray_map is not None else gray_map
    selected_gray = np.asarray(selected_gray, dtype=np.float32)
    delta_map = selected_gray - float(target_gray)
    delta_tv = _mean_abs_spatial_diff(delta_map) / 255.0
    delta_abs_mean = float(np.mean(np.abs(delta_map))) / 255.0
    clip_ratio = float(np.mean((selected_gray <= 0.5) | (selected_gray >= 254.5)))

    quality_score = (
        std_weight * std_norm
        + mean_weight * mean_loss
        + p95_weight * p95_abs_error
        + low_weight * low_std
        + mid_weight * mid_abs_p99
        + high_weight * high_abs_p99
        + tail_weight * tail_abs_p99
        + profile_weight * profile_loss
        + grad_weight * grad_loss
        + delta_tv_weight * delta_tv
        + delta_abs_weight * delta_abs_mean
        + clip_weight * clip_ratio
    )
    return {
        "quality_score": float(quality_score),
        "quality_std_norm": std_norm,
        "quality_mean_loss": mean_loss,
        "quality_p95_abs_error": p95_abs_error,
        "quality_low_std": low_std,
        "quality_mid_abs_p99": mid_abs_p99,
        "quality_high_abs_p99": high_abs_p99,
        "quality_tail_abs_p99": tail_abs_p99,
        "quality_profile_loss": profile_loss,
        "quality_grad_loss": grad_loss,
        "quality_delta_tv": float(delta_tv),
        "quality_delta_abs_mean": float(delta_abs_mean),
        "quality_clip_ratio": clip_ratio,
    }


def select_better_step_snapshot(
    best_snapshot: Optional[Dict[str, object]],
    *,
    step: int,
    std: float,
    reward: float,
    gray_map: np.ndarray,
    luma_map: Optional[np.ndarray] = None,
    quantized_gray_map: Optional[np.ndarray] = None,
    quality_score: Optional[float] = None,
    quality_metrics: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    if best_snapshot is not None:
        if quality_score is not None and "quality_score" in best_snapshot:
            if float(best_snapshot["quality_score"]) <= float(quality_score):
                return best_snapshot
        elif float(best_snapshot["std"]) <= float(std):
            return best_snapshot

    snapshot = {
        "step": int(step),
        "std": float(std),
        "reward": float(reward),
        "gray_map": gray_map.astype(np.float32, copy=True),
    }
    if quality_score is not None:
        snapshot["quality_score"] = float(quality_score)
    if quality_metrics is not None:
        snapshot.update({key: float(value) for key, value in quality_metrics.items()})
    if luma_map is not None:
        snapshot["luma_map"] = luma_map.astype(np.float32, copy=True)
    if quantized_gray_map is not None:
        snapshot["quantized_gray_map"] = quantized_gray_map.astype(np.float32, copy=True)
    return snapshot
