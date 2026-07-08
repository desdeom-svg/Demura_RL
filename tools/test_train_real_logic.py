import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rl_training_utils import (
    compute_reference_normalized_metrics,
    compute_step_quality_metrics,
    crop_center_region,
    load_traditional_delta_map,
    quantize_effective_delta,
    select_better_step_snapshot,
    should_store_transition_for_replay,
)

try:
    import torch
    from models import Actor
    from real_world_env import compute_uniformity_reward_from_rel_error
    from train_real import (
        RealReplayBuffer,
        apply_gain_map_to_prior,
        bound_gain_map,
        build_bootstrap_prior_action,
        build_prior_action_from_state,
        build_traditional_prior_action,
        compute_action_laplacian_loss,
        compute_action_smoothness_loss,
        compute_improvement_reward,
        make_structured_noise,
        quantize_action_from_state as quantize_tensor_action_from_state,
        quantize_action_straight_through,
        resolve_training_phase,
        select_bootstrap_prior_gain,
    )
except ModuleNotFoundError:
    torch = None
    Actor = None
    RealReplayBuffer = None
    apply_gain_map_to_prior = None
    bound_gain_map = None
    build_bootstrap_prior_action = None
    build_prior_action_from_state = None
    build_traditional_prior_action = None
    compute_action_laplacian_loss = None
    compute_action_smoothness_loss = None
    compute_improvement_reward = None
    compute_uniformity_reward_from_rel_error = None
    make_structured_noise = None
    quantize_tensor_action_from_state = None
    quantize_action_straight_through = None
    resolve_training_phase = None
    select_bootstrap_prior_gain = None


def test_reward_roi_can_be_smaller_than_action_roi_if_torch_available():
    if torch is None:
        return
    from real_world_env import RealWorldEnv

    env = RealWorldEnv(gray_candidates=[16], train_roi_size=0, reward_roi_size=2)
    env.roi_h = 6
    env.roi_w = 4
    env._train_roi = None
    env._reward_roi = env._compute_center_roi(2, "Test reward ROI")

    tensor = torch.arange(24, dtype=torch.float32).view(1, 1, 6, 4)

    assert env._crop_to_train_roi(tensor).shape[-2:] == (6, 4)
    reward_crop = env._crop_to_reward_roi(tensor)
    assert reward_crop.shape[-2:] == (2, 2)
    assert torch.equal(reward_crop.cpu(), tensor[..., 2:4, 1:3].cpu())


def test_luma_crop_expands_into_reward_roi_for_larger_action_roi_if_torch_available():
    if torch is None:
        return
    from real_world_env import RealWorldEnv

    env = RealWorldEnv(gray_candidates=[16], train_roi_size=4, reward_roi_size=2)
    env.roi_h = 6
    env.roi_w = 4
    env._train_roi = env._compute_center_roi(4, "Test train ROI")
    env._reward_roi = env._compute_center_roi(2, "Test reward ROI")

    crop = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    expanded = env._expand_luma_to_full_roi_if_needed(crop, fill_value=9.0)

    assert expanded.shape == (6, 4)
    assert np.array_equal(expanded[2:4, 1:3], crop)
    assert float(expanded[0, 0]) == 9.0


def test_reward_roi_baseline_ignores_constant_expanded_padding_if_torch_available():
    if torch is None:
        return
    from real_world_env import RealWorldEnv

    env = RealWorldEnv(gray_candidates=[16], train_roi_size=4, reward_roi_size=2)
    env.roi_h = 6
    env.roi_w = 4
    env._train_roi = env._compute_center_roi(4, "Test train ROI")
    env._reward_roi = env._compute_center_roi(2, "Test reward ROI")

    expanded = torch.full((1, 1, 6, 4), 10.0)
    expanded[..., 2:4, 1:3] = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    target = torch.full_like(expanded, 2.5)

    env.current_luma_map = expanded
    env.target_luma_map = target
    env.run_target_luma_map = target
    env._update_baseline_std_metrics()

    reward_crop = expanded[..., 2:4, 1:3]
    expected_real_std = float(torch.std(reward_crop).item())
    expected_norm_std = float(torch.std(reward_crop / (target[..., 2:4, 1:3] + 1e-6)).item())

    assert abs(env.initial_std - expected_real_std) < 1e-6
    assert abs(env.run_target_norm_std - expected_norm_std) < 1e-6


def test_train_real_reference_metrics_use_reward_roi_size():
    source = (ROOT_DIR / "train_real.py").read_text(encoding="utf-8")

    assert 'parser.add_argument("--reward-roi-size"' in source
    assert "reward_roi_size=args.reward_roi_size" in source
    assert "crop_h=reference_crop_h" in source
    assert "crop_w=reference_crop_w" in source


def test_bootstrap_prior_action_respects_train_roi_source():
    source = (ROOT_DIR / "train_real.py").read_text(encoding="utf-8")

    assert "base_prior_action = env._crop_to_train_roi(base_prior_action)" in source


def test_quantize_action_from_state():
    displayed_gray = np.full((2, 2), 16.0, dtype=np.float32)
    proposed_gray = displayed_gray + 0.49
    tiny_effective = quantize_effective_delta(displayed_gray, proposed_gray)
    assert np.count_nonzero(tiny_effective) == 0

    proposed_gray = displayed_gray + 0.51
    visible_effective = quantize_effective_delta(displayed_gray, proposed_gray)
    assert np.all(visible_effective == 1.0)


def test_select_better_step_snapshot():
    best = None
    best = select_better_step_snapshot(best, step=1, std=0.0500, reward=-10.0, gray_map=np.array([[1.0]], dtype=np.float32))
    best = select_better_step_snapshot(best, step=2, std=0.0400, reward=-20.0, gray_map=np.array([[2.0]], dtype=np.float32))
    best = select_better_step_snapshot(best, step=3, std=0.0450, reward=-5.0, gray_map=np.array([[3.0]], dtype=np.float32))

    assert best["step"] == 2
    assert abs(best["std"] - 0.0400) < 1e-6
    assert float(best["gray_map"][0, 0]) == 2.0


def test_select_better_step_snapshot_prefers_quality_over_std():
    best = None
    best = select_better_step_snapshot(
        best,
        step=1,
        std=0.0400,
        reward=-10.0,
        gray_map=np.array([[16.0]], dtype=np.float32),
        quality_score=0.0500,
        quality_metrics={"quality_score": 0.0500, "quality_mean_loss": 0.0},
    )
    best = select_better_step_snapshot(
        best,
        step=2,
        std=0.0350,
        reward=-8.0,
        gray_map=np.array([[30.0]], dtype=np.float32),
        quality_score=0.0800,
        quality_metrics={"quality_score": 0.0800, "quality_mean_loss": 0.04},
    )

    assert best["step"] == 1
    assert abs(best["std"] - 0.0400) < 1e-6
    assert abs(best["quality_score"] - 0.0500) < 1e-6


def test_iterative_training_launcher_uses_bootstrap_escape_defaults():
    script = (ROOT_DIR / "tools" / "start_iterative_training.ps1").read_text(encoding="utf-8")
    train_real_source = (ROOT_DIR / "train_real.py").read_text(encoding="utf-8")

    assert "[double]$BootstrapPositiveRewardThreshold = 0.0005" in script
    assert "[double]$BootstrapNegativeRewardThreshold = -0.005" in script
    assert '[string]$BootstrapPriorGains = "0.5,1.0"' in script
    assert "[int]$BootstrapMinPositive = 2" in script
    assert "[int]$BestPatience = 80" in script
    assert "[double]$GainLimitInit = 0.05" in script
    assert "[double]$GainLimitFinal = 0.30" in script
    assert "[int]$CriticWarmupUpdates = 120" in script
    assert "[double]$QualityMeanWeight = 1.2" in script
    assert "[double]$QualityTailWeight = 1.0" in script
    assert "[string]$ReferenceModelPath = \"\"" in script
    assert "[double]$ReferenceMetricGate = 1.0" in script
    assert "[double]$MinReplayEffectiveActionAbsMean = 0.000001" in script
    assert '"--bootstrap-positive-reward-threshold", $BootstrapPositiveRewardThreshold' in script
    assert '"--bootstrap-negative-reward-threshold", $BootstrapNegativeRewardThreshold' in script
    assert '"--bootstrap-min-positive", $BootstrapMinPositive' in script
    assert '"--bootstrap-min-negative", $BootstrapMinNegative' in script
    assert '"--reference-metric-gate", $ReferenceMetricGate' in script
    assert '"--min-replay-effective-action-abs-mean", $MinReplayEffectiveActionAbsMean' in script
    assert 'parser.add_argument("--bootstrap-prior-gains", type=str, default="0.5,1.0")' in train_real_source
    assert 'parser.add_argument("--bootstrap-min-positive", type=int, default=2)' in train_real_source
    assert 'parser.add_argument("--bootstrap-min-negative", type=int, default=4)' in train_real_source
    assert 'parser.add_argument("--bootstrap-positive-reward-threshold", type=float, default=0.0005)' in train_real_source
    assert 'parser.add_argument("--bootstrap-negative-reward-threshold", type=float, default=-0.005)' in train_real_source
    assert 'parser.add_argument("--gain-limit-init", type=float, default=0.05)' in train_real_source
    assert 'parser.add_argument("--gain-limit-final", type=float, default=0.30)' in train_real_source
    assert 'parser.add_argument("--gain-abs-weight", type=float, default=0.005)' in train_real_source
    assert 'parser.add_argument("--quality-mean-weight", type=float, default=1.2)' in train_real_source
    assert 'parser.add_argument("--quality-tail-weight", type=float, default=1.0)' in train_real_source
    assert 'parser.add_argument(\n        "--reference-model-path",' in train_real_source
    assert 'parser.add_argument(\n        "--reference-metric-gate",' in train_real_source
    assert "default=80,\n        help=\"Stop after this many post-best episodes without a new best quality." in train_real_source


def test_compute_step_quality_metrics_penalizes_bad_mean_and_outliers():
    good = compute_step_quality_metrics(
        {"std_norm": 0.030, "mean_loss": 0.002, "p95_abs_error": 0.060, "grad_loss": 0.010},
        gray_map=np.full((4, 4), 16.0, dtype=np.float32),
        target_gray=16.0,
    )
    bad = compute_step_quality_metrics(
        {"std_norm": 0.028, "mean_loss": 0.040, "p95_abs_error": 0.120, "grad_loss": 0.030},
        gray_map=np.array(
            [
                [16.0, 32.0, 16.0, 32.0],
                [32.0, 16.0, 32.0, 16.0],
                [16.0, 32.0, 16.0, 32.0],
                [32.0, 16.0, 32.0, 16.0],
            ],
            dtype=np.float32,
        ),
        target_gray=16.0,
    )

    assert bad["quality_score"] > good["quality_score"]


def test_compute_step_quality_metrics_penalizes_mid_high_frequency_artifacts():
    common = {
        "std_norm": 0.025,
        "mean_loss": 0.001,
        "p95_abs_error": 0.050,
        "grad_loss": 0.020,
        "low_std": 0.010,
        "tail_abs_p99": 0.070,
        "profile_loss": 0.010,
    }
    smooth = compute_step_quality_metrics(
        {**common, "mid_abs_p99": 0.006, "high_abs_p99": 0.043},
        gray_map=np.full((4, 4), 16.0, dtype=np.float32),
        target_gray=16.0,
    )
    blotchy = compute_step_quality_metrics(
        {**common, "mid_abs_p99": 0.019, "high_abs_p99": 0.055},
        gray_map=np.full((4, 4), 16.0, dtype=np.float32),
        target_gray=16.0,
    )

    assert blotchy["quality_score"] > smooth["quality_score"]
    assert blotchy["quality_mid_abs_p99"] == 0.019
    assert blotchy["quality_high_abs_p99"] == 0.055


def test_crop_center_region_uses_reference_center_area():
    assert crop_center_region is not None

    image = np.arange(10 * 8, dtype=np.float32).reshape(10, 8)
    crop = crop_center_region(image, height=4, width=2)

    assert crop.shape == (4, 2)
    assert np.array_equal(crop, image[3:7, 3:5])


def test_reference_normalized_metrics_report_pass_fail_against_traditional_reference():
    assert compute_reference_normalized_metrics is not None

    current = {
        "std_norm": 0.040,
        "low_std": 0.012,
        "mid_abs_p99": 0.030,
        "high_abs_p99": 0.020,
        "tail_abs_p99": 0.080,
        "profile_loss": 0.010,
        "grad_loss": 0.006,
    }
    reference = {
        "std_norm": 0.050,
        "low_std": 0.015,
        "mid_abs_p99": 0.040,
        "high_abs_p99": 0.025,
        "tail_abs_p99": 0.100,
        "profile_loss": 0.020,
        "grad_loss": 0.010,
    }

    metrics = compute_reference_normalized_metrics(current, reference)

    assert metrics["reference_ratio_max"] < 1.0
    assert metrics["reference_all_pass"] == 1.0
    assert abs(metrics["reference_std_norm_ratio"] - 0.8) < 1e-6
    assert abs(metrics["reference_profile_loss_ratio"] - 0.5) < 1e-6


def test_reference_normalized_metrics_fails_when_any_visual_metric_is_worse():
    assert compute_reference_normalized_metrics is not None

    current = {
        "std_norm": 0.040,
        "low_std": 0.012,
        "mid_abs_p99": 0.041,
        "high_abs_p99": 0.020,
        "tail_abs_p99": 0.080,
        "profile_loss": 0.010,
        "grad_loss": 0.006,
    }
    reference = {
        "std_norm": 0.050,
        "low_std": 0.015,
        "mid_abs_p99": 0.040,
        "high_abs_p99": 0.025,
        "tail_abs_p99": 0.100,
        "profile_loss": 0.020,
        "grad_loss": 0.010,
    }

    metrics = compute_reference_normalized_metrics(current, reference)

    assert metrics["reference_ratio_max"] > 1.0
    assert metrics["reference_all_pass"] == 0.0


def test_reference_metrics_are_observation_only_for_quality_selection():
    train_real_source = (ROOT_DIR / "train_real.py").read_text(encoding="utf-8")
    launcher_source = (ROOT_DIR / "tools" / "start_iterative_training.ps1").read_text(encoding="utf-8")

    assert "quality_reference_penalty" not in train_real_source
    assert "quality-reference-ratio-weight" not in train_real_source
    assert "QualityReferenceRatioWeight" not in launcher_source
    assert "quality-reference-ratio-weight" not in launcher_source


def test_should_skip_noop_transition_before_replay():
    assert should_store_transition_for_replay is not None

    assert not should_store_transition_for_replay(
        {"effective_action_abs_mean": 0.0, "quantized_step_changed_ratio": 0.0},
        min_effective_action_abs_mean=1e-6,
        min_quantized_step_changed_ratio=1e-6,
    )
    assert should_store_transition_for_replay(
        {"effective_action_abs_mean": 0.002, "quantized_step_changed_ratio": 0.0002},
        min_effective_action_abs_mean=1e-6,
        min_quantized_step_changed_ratio=1e-6,
    )


def test_load_traditional_delta_map():
    panel_h = 2652
    panel_w = 1200
    image = np.zeros((panel_h, panel_w), dtype=np.uint8)
    image[200:2200, 100:1100] = 18

    with tempfile.TemporaryDirectory() as tmp_dir:
        bmp_path = os.path.join(tmp_dir, "traditional.bmp")
        Image.fromarray(image, mode="L").save(bmp_path)

        delta = load_traditional_delta_map(bmp_path, target_gray=16.0)
        assert delta.shape == (2000, 1000)
        assert abs(float(delta.mean()) - 2.0) < 1e-6


def test_tensor_action_quantization_if_torch_available():
    if torch is None:
        return

    state = torch.zeros((1, 2, 1, 3), dtype=torch.float32)
    state[:, 1:2] = torch.tensor([[[[16.0, 16.0, 254.0]]]]) / 255.0
    action = torch.tensor([[[[0.49, 0.51, 3.0]]]], dtype=torch.float32)

    quantized = quantize_tensor_action_from_state(state, action)
    expected = torch.tensor([[[[0.0, 1.0, 1.0]]]], dtype=torch.float32)
    assert torch.equal(quantized.cpu(), expected)

    straight_through = quantize_action_straight_through(state, action)
    assert torch.equal(straight_through.detach().cpu(), expected)


def test_real_replay_buffer_multi_crop_if_torch_available():
    if torch is None:
        return

    buffer = RealReplayBuffer(capacity=4)
    buffer.patch_h = 2
    buffer.patch_w = 2
    buffer.grid_rows = 3
    buffer.grid_cols = 4
    state = torch.arange(2 * 6 * 8, dtype=torch.float32).view(1, 2, 6, 8)
    action = torch.ones((1, 1, 6, 8), dtype=torch.float32)
    next_state = state + 1000.0
    for _ in range(2):
        buffer.push(state, action, -1.0, next_state)

    state_batch, action_batch, reward_batch, next_state_batch, done_batch = buffer.sample(
        batch_size=2,
        patches_per_transition=2,
    )

    assert state_batch.shape == (4, 2, 2, 2)
    assert action_batch.shape == (4, 1, 2, 2)
    assert next_state_batch.shape == (4, 2, 2, 2)
    assert reward_batch.shape == (4, 1)
    assert done_batch.shape == (4, 1)
    assert state_batch.dtype == torch.float32


def test_real_replay_buffer_supports_random_overlap_and_large_crops_if_torch_available():
    if torch is None:
        return

    buffer = RealReplayBuffer(capacity=2)
    state = torch.arange(2 * 8 * 8, dtype=torch.float32).view(1, 2, 8, 8)
    action = torch.ones((1, 1, 8, 8), dtype=torch.float32)
    next_state = state + 1.0
    for _ in range(2):
        buffer.push(state, action, -1.0, next_state)

    state_batch, action_batch, reward_batch, next_state_batch, done_batch = buffer.sample(
        batch_size=2,
        patches_per_transition=1,
        crop_h=4,
        crop_w=4,
        random_crop_ratio=1.0,
    )

    assert state_batch.shape == (2, 2, 4, 4)
    assert action_batch.shape == (2, 1, 4, 4)
    assert reward_batch.shape == (2, 1)
    assert next_state_batch.shape == (2, 2, 4, 4)
    assert done_batch.shape == (2, 1)


def test_uniformity_reward_prefers_low_std_over_zero_mean_if_torch_available():
    if torch is None:
        return

    uniform_offset = torch.full((1, 1, 4, 4), 0.05, dtype=torch.float32)
    uneven_zero_mean = torch.tensor(
        [[[[0.20, -0.20, 0.20, -0.20],
           [0.20, -0.20, 0.20, -0.20],
           [0.20, -0.20, 0.20, -0.20],
           [0.20, -0.20, 0.20, -0.20]]]],
        dtype=torch.float32,
    )

    uniform_reward, _ = compute_uniformity_reward_from_rel_error(uniform_offset)
    uneven_reward, _ = compute_uniformity_reward_from_rel_error(uneven_zero_mean)
    assert uniform_reward.item() > uneven_reward.item()


def test_uniformity_reward_penalizes_mid_high_frequency_blobs_if_torch_available():
    if torch is None:
        return

    yy, xx = torch.meshgrid(torch.arange(64), torch.arange(64), indexing="ij")
    smooth = 0.04 * torch.sin(2.0 * torch.pi * yy.float() / 64.0)
    checker = 0.03 * torch.sign(torch.sin(2.0 * torch.pi * xx.float() / 8.0))
    checker += 0.03 * torch.sign(torch.sin(2.0 * torch.pi * yy.float() / 8.0))
    smooth = smooth.view(1, 1, 64, 64)
    checker = checker.view(1, 1, 64, 64)

    smooth_reward, smooth_info = compute_uniformity_reward_from_rel_error(smooth)
    checker_reward, checker_info = compute_uniformity_reward_from_rel_error(checker)

    assert "r_low" in smooth_info
    assert "r_mid" in smooth_info
    assert "r_high" in smooth_info
    assert "r_tail" in smooth_info
    assert checker_info["high_abs_p99"] > smooth_info["high_abs_p99"]
    assert checker_reward.item() < smooth_reward.item()


def test_improvement_reward_is_positive_for_better_next_state_if_torch_available():
    if torch is None:
        return

    state = torch.zeros((1, 2, 8, 8), dtype=torch.float32)
    state[:, 0:1] = 0.10
    next_state = torch.zeros((1, 2, 4, 4), dtype=torch.float32)
    next_state = torch.zeros_like(state)

    improvement = compute_improvement_reward(state[:, 0:1], next_state[:, 0:1])
    regression = compute_improvement_reward(next_state[:, 0:1], state[:, 0:1])

    assert improvement.item() > 0.0
    assert regression.item() < 0.0


def test_real_replay_buffer_recomputes_improvement_and_keeps_done_if_torch_available():
    if torch is None:
        return

    buffer = RealReplayBuffer(capacity=1)
    state = torch.zeros((1, 2, 4, 4), dtype=torch.float32)
    state[:, 0:1] = 0.10
    action = torch.zeros((1, 1, 4, 4), dtype=torch.float32)
    next_state = torch.zeros_like(state)
    buffer.push(state, action, reward=-0.75, next_state=next_state, done=True)

    _state_batch, _action_batch, reward_batch, _next_state_batch, done_batch = buffer.sample(
        batch_size=1,
        patches_per_transition=1,
    )
    assert done_batch[0, 0].item() == 1.0
    assert reward_batch[0, 0].item() <= -0.75


def test_build_prior_action_from_state_moves_bright_pixels_down_if_torch_available():
    if torch is None:
        return

    state = torch.zeros((1, 2, 2, 2), dtype=torch.float32)
    state[:, 0:1] = torch.tensor([[[[0.20, -0.20], [-0.10, 0.10]]]], dtype=torch.float32)
    state[:, 1:2] = 16.0 / 255.0
    action = build_prior_action_from_state(state, gray=16.0, gamma=2.2, gain=1.0)
    assert action[0, 0, 0, 0].item() < 0.0
    assert action[0, 0, 0, 1].item() > 0.0


def test_apply_gain_map_to_prior_scales_prior_if_torch_available():
    if torch is None:
        return

    prior = torch.tensor([[[[2.0, -4.0], [8.0, -10.0]]]], dtype=torch.float32)
    gain_map = torch.tensor([[[[0.5, -0.25], [0.0, 1.0]]]], dtype=torch.float32)
    action = apply_gain_map_to_prior(prior, gain_map)
    expected = torch.tensor([[[[3.0, -3.0], [8.0, -20.0]]]], dtype=torch.float32)
    assert torch.allclose(action, expected)


def test_apply_gain_map_to_prior_clamps_scale_if_torch_available():
    if torch is None:
        return

    prior = torch.tensor([[[[5.0, 5.0]]]], dtype=torch.float32)
    gain_map = torch.tensor([[[[-5.0, 5.0]]]], dtype=torch.float32)
    action = apply_gain_map_to_prior(prior, gain_map)
    expected = torch.tensor([[[[2.5, 15.0]]]], dtype=torch.float32)
    assert torch.allclose(action, expected)


def test_bound_gain_map_uses_tanh_limit_if_torch_available():
    if torch is None:
        return

    raw = torch.tensor([[[[-100.0, 0.0, 100.0]]]], dtype=torch.float32)
    bounded = bound_gain_map(raw, limit=0.5)
    assert torch.all(bounded <= 0.5)
    assert torch.all(bounded >= -0.5)
    assert abs(float(bounded[0, 0, 0, 1])) < 1e-6


def test_structured_noise_is_smooth_and_has_requested_shape_if_torch_available():
    if torch is None:
        return

    reference = torch.zeros((1, 1, 40, 80), dtype=torch.float32)
    noise = make_structured_noise(reference, scale=0.2, grid_rows=4, grid_cols=8)
    assert noise.shape == reference.shape
    assert torch.max(torch.abs(noise)).item() <= 0.200001
    horizontal_delta = torch.mean(torch.abs(noise[..., 1:] - noise[..., :-1])).item()
    assert horizontal_delta < 0.08


def test_bootstrap_prior_action_cycles_signed_absolute_gains_if_torch_available():
    if torch is None:
        return

    base_prior = torch.full((1, 1, 2, 2), 2.0, dtype=torch.float32)
    gains = (0.5, 1.0, -0.5, -1.0)
    actions = [
        build_bootstrap_prior_action(base_prior, gains, attempt_index=index)
        for index in range(5)
    ]

    assert torch.allclose(actions[0], torch.full_like(base_prior, 1.0))
    assert torch.allclose(actions[1], torch.full_like(base_prior, 2.0))
    assert torch.allclose(actions[2], torch.full_like(base_prior, -1.0))
    assert torch.allclose(actions[3], torch.full_like(base_prior, -2.0))
    assert torch.allclose(actions[4], actions[0])


def test_training_phase_depends_on_sample_balance_and_critic_updates_if_torch_available():
    if torch is None:
        return

    args = SimpleNamespace(
        bootstrap_min_positive=2,
        bootstrap_min_negative=2,
        bootstrap_positive_reward_threshold=0.01,
        bootstrap_negative_reward_threshold=-0.01,
        critic_warmup_updates=5,
        actor_warmup_episodes=0,
    )
    buffer = RealReplayBuffer(capacity=8)
    state = torch.zeros((1, 2, 2, 2), dtype=torch.float32)
    action = torch.zeros((1, 1, 2, 2), dtype=torch.float32)

    assert resolve_training_phase(buffer, critic_update_count=0, actor_warmup_count=0, args=args) == "bootstrap"
    for reward in (0.001, -0.001):
        buffer.push(state, action, reward, state, done=False)
    assert resolve_training_phase(buffer, critic_update_count=0, actor_warmup_count=0, args=args) == "bootstrap"

    for reward in (0.2, 0.1, -0.2, -0.3):
        buffer.push(state, action, reward, state, done=reward < 0)

    assert buffer.positive_count == 3
    assert buffer.negative_count == 3
    assert resolve_training_phase(buffer, critic_update_count=4, actor_warmup_count=0, args=args) == "critic_only"
    assert resolve_training_phase(buffer, critic_update_count=5, actor_warmup_count=0, args=args) == "actor"


def test_bootstrap_prior_gain_selection_prefers_lower_quality_if_torch_available():
    if torch is None:
        return

    gain_quality_scores = {
        0.5: [0.480, 0.472],
        1.0: [0.491, 0.489],
        -0.5: [0.505, 0.470],
        -1.0: [0.512],
    }

    selected = select_bootstrap_prior_gain(
        (0.5, 1.0, -0.5, -1.0),
        gain_quality_scores,
        fallback_gain=1.0,
    )

    assert selected == 0.5


def test_actor_is_single_gain_head_if_torch_available():
    if torch is None:
        return

    actor = Actor()
    state = torch.zeros((1, 2, 8, 8), dtype=torch.float32)
    gain_map = actor(state)

    assert gain_map.shape == (1, 1, 8, 8)
    assert not hasattr(actor, "direct_residual_head")


def test_action_smoothness_loss_penalizes_checkerboard_if_torch_available():
    if torch is None:
        return

    smooth = torch.zeros((1, 1, 4, 4), dtype=torch.float32)
    checker = torch.tensor(
        [[[[0.0, 2.0, 0.0, 2.0],
           [2.0, 0.0, 2.0, 0.0],
           [0.0, 2.0, 0.0, 2.0],
           [2.0, 0.0, 2.0, 0.0]]]],
        dtype=torch.float32,
    )

    assert compute_action_smoothness_loss(checker).item() > compute_action_smoothness_loss(smooth).item()


def test_action_laplacian_loss_penalizes_local_bumps_if_torch_available():
    if torch is None:
        return

    flat = torch.zeros((1, 1, 5, 5), dtype=torch.float32)
    bump = flat.clone()
    bump[:, :, 2, 2] = 2.0

    assert compute_action_laplacian_loss(bump).item() > compute_action_laplacian_loss(flat).item()


class _FakeEnv:
    def __init__(self):
        self.current_gray_int = 16
        self.target_gray = 16.0
        self.current_gray_map = None
        self.displayed_gray_map = None
        self.current_luma_map = torch.tensor([[[[2.0, 1.0], [1.0, 2.0]]]], dtype=torch.float32)
        self.target_mean_nit = 1.5


def test_traditional_prior_action_moves_bright_pixels_down_if_torch_available():
    if torch is None:
        return

    action = build_traditional_prior_action(_FakeEnv(), gamma=2.2, gain=1.0)
    assert action[0, 0, 0, 0].item() < 0.0
    assert action[0, 0, 0, 1].item() > 0.0


if __name__ == "__main__":
    test_quantize_action_from_state()
    test_select_better_step_snapshot()
    test_select_better_step_snapshot_prefers_quality_over_std()
    test_compute_step_quality_metrics_penalizes_bad_mean_and_outliers()
    test_compute_step_quality_metrics_penalizes_mid_high_frequency_artifacts()
    test_crop_center_region_uses_reference_center_area()
    test_reward_roi_can_be_smaller_than_action_roi_if_torch_available()
    test_luma_crop_expands_into_reward_roi_for_larger_action_roi_if_torch_available()
    test_reward_roi_baseline_ignores_constant_expanded_padding_if_torch_available()
    test_train_real_reference_metrics_use_reward_roi_size()
    test_bootstrap_prior_action_respects_train_roi_source()
    test_reference_normalized_metrics_report_pass_fail_against_traditional_reference()
    test_reference_normalized_metrics_fails_when_any_visual_metric_is_worse()
    test_reference_metrics_are_observation_only_for_quality_selection()
    test_should_skip_noop_transition_before_replay()
    test_load_traditional_delta_map()
    test_tensor_action_quantization_if_torch_available()
    test_real_replay_buffer_multi_crop_if_torch_available()
    test_real_replay_buffer_supports_random_overlap_and_large_crops_if_torch_available()
    test_uniformity_reward_prefers_low_std_over_zero_mean_if_torch_available()
    test_uniformity_reward_penalizes_mid_high_frequency_blobs_if_torch_available()
    test_improvement_reward_is_positive_for_better_next_state_if_torch_available()
    test_real_replay_buffer_recomputes_improvement_and_keeps_done_if_torch_available()
    test_build_prior_action_from_state_moves_bright_pixels_down_if_torch_available()
    test_apply_gain_map_to_prior_scales_prior_if_torch_available()
    test_apply_gain_map_to_prior_clamps_scale_if_torch_available()
    test_bound_gain_map_uses_tanh_limit_if_torch_available()
    test_structured_noise_is_smooth_and_has_requested_shape_if_torch_available()
    test_bootstrap_prior_action_cycles_signed_absolute_gains_if_torch_available()
    test_training_phase_depends_on_sample_balance_and_critic_updates_if_torch_available()
    test_bootstrap_prior_gain_selection_prefers_lower_quality_if_torch_available()
    test_actor_is_single_gain_head_if_torch_available()
    test_action_smoothness_loss_penalizes_checkerboard_if_torch_available()
    test_action_laplacian_loss_penalizes_local_bumps_if_torch_available()
    test_traditional_prior_action_moves_bright_pixels_down_if_torch_available()
    print("ok")
