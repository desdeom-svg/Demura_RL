import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def source_for(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


class RealTrainingAlignmentTests(unittest.TestCase):
    def test_real_env_target_is_center_mean_plane(self):
        source = source_for("real_world_env.py")
        self.assertIn("calc_center_mean(self.current_luma_map", source)
        self.assertIn("torch.full_like(self.current_luma_map", source)

        reset_source = source[
            source.index("    def reset(") : source.index("    def step(")
        ]
        self.assertNotIn("apply_multi_pass_lowpass", reset_source)
        self.assertIn("self.run_target_mean_nit", source)
        self.assertIn("self.run_target_luma_map", source)
        self.assertIn("if self.run_target_mean_nit is None or not self.freeze_target_per_run:", reset_source)

    def test_actor_forward_is_network_only(self):
        tree = ast.parse(source_for("models.py"))
        actor_class = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Actor"
        )
        forward = next(
            node
            for node in actor_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "forward"
        )
        model_source = source_for("models.py")
        lines = model_source.splitlines()
        end_lineno = getattr(forward, "end_lineno", None)
        if end_lineno is None:
            end_lineno = next(
                index
                for index in range(forward.lineno, len(lines))
                if lines[index].startswith("class Critic")
            )
        forward_source = "\n".join(lines[forward.lineno - 1 : end_lineno])
        self.assertNotIn("prior_out", forward_source)
        self.assertNotIn("torch.pow(luma_ratio", forward_source)
        self.assertIn("network_out = self.exit(feat)", forward_source)
        self.assertIn("return self.output_smoother(network_out)", forward_source)
        self.assertNotIn("forward_heads", model_source)
        self.assertNotIn("direct_residual_head", model_source)

    def test_train_entrypoint_delegates_to_simulation_aligned_training_flow(self):
        source = source_for("train_real.py")
        self.assertIn("def run_real_training(args) -> None:", source)
        self.assertIn("def train(args) -> None:\n    run_real_training(args)", source)
        self.assertIn('if __name__ == "__main__":\n    train(parse_args())', source)
        self.assertNotIn("DistillSimulationEnv", source)
        self.assertNotIn("run_single_teacher_distillation", source)
        self.assertNotIn("run_multi_teacher_distillation", source)
        self.assertNotIn("BestStepTeacherBuffer", source)
        self.assertNotIn("teacher_buffer", source)
        self.assertNotIn("distill_loss", source)
        self.assertNotIn("pretrained", source)
        self.assertNotIn("prior_scale", source)
        self.assertNotIn("resolve_training_grays", source)

    def test_real_training_uses_staged_critic_then_delayed_actor_updates(self):
        source = source_for("train_real.py")
        self.assertIn("noise_scale = max(args.residual_noise_min, args.residual_noise_init * (1 - episode / 200))", source)
        self.assertIn('parser.add_argument("--bootstrap-prior-gains", type=str, default="0.5,1.0,-0.5,-1.0")', source)
        self.assertIn('parser.add_argument("--bootstrap-min-positive", type=int, default=4)', source)
        self.assertIn('parser.add_argument("--bootstrap-min-negative", type=int, default=4)', source)
        self.assertIn('parser.add_argument("--bootstrap-positive-reward-threshold", type=float, default=0.01)', source)
        self.assertIn('parser.add_argument("--bootstrap-negative-reward-threshold", type=float, default=-0.01)', source)
        self.assertIn('parser.add_argument("--critic-warmup-updates", type=int, default=50)', source)
        self.assertIn('parser.add_argument("--actor-update-every", type=int, default=3)', source)
        self.assertIn("phase_name = resolve_training_phase(buffer, critic_update_count, args)", source)
        self.assertIn('bootstrap_phase = phase_name == "bootstrap"', source)
        self.assertIn('actor_enabled = phase_name == "actor"', source)
        self.assertIn("sb, ab, rb, nsb, db = buffer.sample(", source)
        self.assertIn("target_q = rb + args.gamma * (1.0 - db) * target_critic(nsb, tgt_a)", source)
        self.assertIn("policy_loss = -critic(sb, pred_a).mean()", source)
        self.assertIn("critic_update_count % args.actor_update_every == 0", source)
        self.assertIn("gain_abs_loss = torch.mean(torch.abs(gain_map))", source)
        self.assertIn("gain_tv_loss = compute_action_smoothness_loss(gain_map)", source)
        self.assertIn("gain_laplacian_loss = compute_action_laplacian_loss(gain_map)", source)
        self.assertIn("soft_update(target_actor, actor, args.tau)", source)
        self.assertIn("did_actor_update_this_episode", source)
        self.assertIn("did_critic_update_this_episode", source)
        self.assertNotIn("actor(state, prior_scale=", source)
        self.assertNotIn("target_actor(next_state_batch, prior_scale=", source)
        self.assertNotIn("actor(state_batch, prior_scale=", source)

    def test_real_training_uses_quantized_effective_actions_for_replay(self):
        source = source_for("train_real.py")
        self.assertIn("def action_for_replay(info, fallback_action):", source)
        self.assertIn("def quantize_action_from_state(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:", source)
        self.assertIn("def quantize_action_straight_through(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:", source)
        self.assertIn("class RealReplayBuffer:", source)
        self.assertIn("buffer = RealReplayBuffer(args.buffer_capacity)", source)
        self.assertIn('info.get("effective_action_tensor")', source)
        self.assertIn("transition_done = rebound_stop or unsafe_transition or step_index == effective_steps - 1", source)
        self.assertIn("buffer.push(state, replay_action, learning_reward, next_state, done=transition_done)", source)
        self.assertIn("def apply_gain_map_to_prior(prior_action: torch.Tensor, gain_map: torch.Tensor) -> torch.Tensor:", source)
        self.assertNotIn("def apply_dual_head_action(", source)
        self.assertNotIn("direct_residual", source)
        self.assertIn("scale = torch.clamp(1.0 + gain_map, 0.2, 4.0)", source)
        self.assertIn("def bound_gain_map(raw_gain_map: torch.Tensor, limit: float) -> torch.Tensor:", source)
        self.assertIn("prior_nsb = build_prior_action_from_state(", source)
        self.assertIn("prior_sb = build_prior_action_from_state(", source)
        self.assertIn("target_gain_map = bound_gain_map(target_actor(nsb), gain_limit)", source)
        self.assertIn("tgt_a = quantize_action_from_state(", source)
        self.assertIn("apply_gain_map_to_prior(prior_nsb, target_gain_map)", source)
        self.assertIn("gain_map = bound_gain_map(actor(sb), gain_limit)", source)
        self.assertIn("pred_a = quantize_action_straight_through(", source)
        self.assertIn("apply_gain_map_to_prior(prior_sb, gain_map)", source)

    def test_real_training_uses_fixed_four_effective_steps(self):
        source = source_for("train_real.py")
        self.assertIn("FIXED_EFFECTIVE_STEPS = 4", source)
        self.assertIn("effective_steps = FIXED_EFFECTIVE_STEPS", source)
        self.assertIn("for step_index in range(effective_steps):", source)
        self.assertNotIn("for step_index in range(args.steps):", source)

    def test_real_training_samples_learning_crops_before_gpu_transfer(self):
        source = source_for("train_real.py")
        self.assertIn('parser.add_argument("--slice-grid-rows", type=int, default=10)', source)
        self.assertIn('parser.add_argument("--slice-grid-cols", type=int, default=5)', source)
        self.assertIn('parser.add_argument("--patches-per-transition", type=int, default=16)', source)
        self.assertIn("def _slice_transition_grid(", source)
        self.assertIn("def _crop_transition_random(", source)
        self.assertIn("patches_per_transition = min(max(1, int(patches_per_transition)), len(patches))", source)
        self.assertIn("random_crop_ratio", source)
        self.assertIn("large_crop_update_ratio", source)
        self.assertIn("torch.cat(states).float().to(device)", source)

    def test_real_training_noise_is_parameterized_for_hardware(self):
        source = source_for("train_real.py")
        self.assertIn('parser.add_argument("--residual-noise-init", type=float, default=0.005)', source)
        self.assertIn('parser.add_argument("--residual-noise-min", type=float, default=0.001)', source)
        self.assertIn('parser.add_argument("--noise-grid-rows", type=int, default=20)', source)
        self.assertIn('parser.add_argument("--noise-grid-cols", type=int, default=40)', source)
        self.assertIn("noise_scale = max(args.residual_noise_min, args.residual_noise_init * (1 - episode / 200))", source)
        self.assertIn("make_structured_noise(", source)
        self.assertNotIn("torch.randn_like(gain_map) * noise_scale", source)
        self.assertNotIn("0.2 * (1 - episode / 200)", source)

    def test_real_reward_prioritizes_uniformity_over_pixel_mse(self):
        env_source = source_for("real_world_env.py")
        train_source = source_for("train_real.py")
        self.assertIn("def compute_uniformity_reward_from_rel_error(", env_source)
        self.assertIn("r_low = -low_std * 120.0", env_source)
        self.assertIn("r_mid = -mid_abs_p99 * 420.0", env_source)
        self.assertIn("r_high = -high_abs_p99 * 220.0", env_source)
        self.assertIn("r_tail = -tail_abs_p99 * 110.0", env_source)
        self.assertIn("r_grad = -grad_loss * 55.0", env_source)
        self.assertIn("r_mean = -mean_loss * 20.0", env_source)
        self.assertNotIn("r_mse = -mse_raw * 300.0", env_source)
        self.assertIn("compute_uniformity_reward_from_rel_error", train_source)

    def test_best_step_teacher_distillation_is_removed(self):
        source = source_for("train_real.py")
        self.assertIn('parser.add_argument("--action-smoothness-weight", type=float, default=0.03)', source)
        self.assertIn('parser.add_argument("--action-laplacian-weight", type=float, default=0.05)', source)
        self.assertNotIn("--teacher-buffer-capacity", source)
        self.assertNotIn("--teacher-distill", source)
        self.assertNotIn("class BestStepTeacherBuffer:", source)
        self.assertNotIn("teacher_buffer", source)
        self.assertNotIn("distill_loss", source)
        self.assertIn("+ args.action_smoothness_weight * smoothness_loss", source)

    def test_replay_recomputes_patch_improvement_and_preserves_terminal(self):
        source = source_for("train_real.py")
        self.assertIn("reward_value = compute_improvement_reward(", source)
        self.assertIn("dones.append(float(done))", source)
        self.assertIn("torch.tensor(dones).float().unsqueeze(1).to(device)", source)
        self.assertNotIn("rewards.append(reward)", source)

    def test_traditional_physics_prior_can_collect_positive_samples(self):
        source = source_for("train_real.py")
        self.assertIn('parser.add_argument("--prior-only-episodes", type=int, default=3)', source)
        self.assertIn('parser.add_argument("--prior-gain", type=float, default=0.1)', source)
        self.assertIn('parser.add_argument("--prior-gamma", type=float, default=2.2)', source)
        self.assertIn("def build_traditional_prior_action(", source)
        self.assertIn("def build_prior_action_from_state(", source)
        self.assertIn("def apply_gain_map_to_prior(prior_action: torch.Tensor, gain_map: torch.Tensor) -> torch.Tensor:", source)
        self.assertNotIn("def apply_dual_head_action(", source)
        self.assertIn("def build_bootstrap_prior_action(", source)
        self.assertIn("bootstrap_attempt_count", source)
        self.assertIn("build_bootstrap_prior_action(", source)
        self.assertIn('logger.info("Bootstrap calibration uses one action per reset.")', source)
        self.assertIn("action = apply_gain_map_to_prior(prior_action, gain_map + noise)", source)

    def test_real_training_keeps_detailed_logs_and_intermediate_outputs(self):
        source = source_for("train_real.py")
        self.assertIn("logging.FileHandler", source)
        self.assertIn("def save_history_csv(", source)
        self.assertIn("def save_training_summary(", source)
        self.assertIn("Best_DemuraTable_Gray", source)
        self.assertIn("Best_Luma_Gray", source)
        self.assertIn("prior_abs_mean", source)
        self.assertIn("residual_abs_mean", source)
        self.assertIn("actual_action_abs_mean", source)

    def test_real_training_is_fully_automatic_and_refreshes_locators(self):
        source = source_for("train_real.py")
        self.assertNotIn("input()", source)
        self.assertNotIn("no_wait_for_enter", source)
        self.assertNotIn("force_refresh_locators", source)
        self.assertIn("env.connect()", source)
        self.assertIn("env.ensure_static_references(force_refresh=True)", source)

    def test_real_training_clears_hardware_output_dirs_at_startup(self):
        source = source_for("train_real.py")
        prepare_source = source[
            source.index("def prepare_run_directory(") : source.index("def attach_run_file_logger(")
        ]
        self.assertIn("PathConfig.CAMERA_IMAGE_DIR", prepare_source)
        self.assertIn("PathConfig.RESULT_DIR", prepare_source)
        self.assertIn("PathConfig.TRANSFER_BMP_DIR", prepare_source)
        self.assertIn("_clear_directory_contents(dir_path)", prepare_source)

    def test_real_env_applies_safety_check_without_double_counting_total_delta(self):
        source = source_for("real_world_env.py")
        step_source = source[source.index("    def step(") : source.index("    def _safety_check(")]
        self.assertIn("action_gray_diff = self._safety_check(action_gray_diff)", step_source)
        self.assertIn("self.total_delta = self.current_gray_map - self.target_gray", step_source)
        self.assertNotIn("self.total_delta += action_gray_diff", step_source)

    def test_real_training_skips_bad_transitions_and_stops_bad_episodes(self):
        source = source_for("train_real.py")
        self.assertIn('parser.add_argument("--freeze-target-per-run", action="store_true", default=True)', source)
        self.assertIn('parser.add_argument("--max-std-ratio-for-replay", type=float, default=1.08)', source)
        self.assertIn('parser.add_argument("--max-std-ratio-for-episode", type=float, default=1.15)', source)
        self.assertIn("baseline_norm_std = env.run_target_norm_std", source)
        self.assertIn("std_ratio = ep_std / max(baseline_norm_std, 1e-6)", source)
        self.assertIn("unsafe_transition = std_ratio > args.max_std_ratio_for_replay", source)
        self.assertIn("transition_done = rebound_stop or unsafe_transition or step_index == effective_steps - 1", source)
        self.assertIn("buffer.push(state, replay_action, learning_reward, next_state, done=transition_done)", source)
        self.assertNotIn("if not skip_replay:", source)
        self.assertIn("learning_reward = min(learning_reward, -abs(args.rebound_negative_reward))", source)
        self.assertIn("if std_ratio > args.max_std_ratio_for_episode:", source)
        self.assertIn('parser.add_argument("--max-step-quality-rebound-ratio", type=float, default=1.05)', source)
        self.assertIn('parser.add_argument("--max-step-quality-rebound-abs", type=float, default=0.01)', source)
        self.assertIn('float(best_step_snapshot["quality_score"])', source)
        self.assertIn("quality_rebound_ratio = current_quality_score / max(best_step_quality, 1e-6)", source)
        self.assertIn("if quality_rebound_ratio > args.max_step_quality_rebound_ratio or quality_rebound_abs > args.max_step_quality_rebound_abs:", source)

    def test_real_training_saves_best_from_best_step_snapshot(self):
        source = source_for("train_real.py")
        self.assertIn("best_step_std = float(best_step_snapshot[\"std\"])", source)
        self.assertIn("best_step_quality_score = float(best_step_snapshot.get(\"quality_score\", float(\"inf\")))", source)
        self.assertIn("if best_step_snapshot is not None and best_step_quality_score < best_quality_score:", source)
        self.assertIn("best_std = best_step_std", source)
        self.assertIn("best_quality_score = best_step_quality_score", source)
        self.assertIn("episode_actor_updates > 0", source)
        self.assertIn("if total_actor_update_count > 0:", source)

    def test_visual_quality_score_uses_frequency_bands(self):
        source = source_for("rl_training_utils.py")
        self.assertIn("low_std = float(reward_info.get(\"low_std\"", source)
        self.assertIn("mid_abs_p99 = float(reward_info.get(\"mid_abs_p99\"", source)
        self.assertIn("high_abs_p99 = float(reward_info.get(\"high_abs_p99\"", source)
        self.assertIn("tail_abs_p99 = float(reward_info.get(\"tail_abs_p99\"", source)
        self.assertIn("profile_loss = float(reward_info.get(\"profile_loss\"", source)
        self.assertIn("low_weight * low_std", source)
        self.assertIn("mid_weight * mid_abs_p99", source)
        self.assertIn("high_weight * high_abs_p99", source)


if __name__ == "__main__":
    unittest.main()
