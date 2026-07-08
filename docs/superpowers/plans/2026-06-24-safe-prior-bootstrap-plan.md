# Safe Prior Bootstrap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ensure real-hardware RL collects useful positive and negative experience before enabling Critic and Actor learning.

**Architecture:** Keep the existing Actor, Critic, physical prior, four-step episodes, and improvement reward. Replace episode-number phase gates with replay/update-count gates, collect a small signed prior-gain calibration set, preserve unsafe transitions as terminal negative samples, and save Actor checkpoints only after actual Actor updates.

**Tech Stack:** Python, PyTorch, unittest, real-hardware DDPG training.

---

### Task 1: Prior Calibration And Phase State

**Files:**
- Modify: `train_real.py`
- Test: `tools/test_train_real_logic.py`
- Test: `tests/test_real_training_alignment.py`

- [x] Add failing tests for signed absolute bootstrap gains and phase selection based on replay counts and Critic updates.
- [x] Run both test suites and confirm the new tests fail.
- [x] Add `build_bootstrap_prior_action()` and `resolve_training_phase()` helpers.
- [x] Run both test suites and confirm they pass.

### Task 2: Preserve Unsafe Negative Experience

**Files:**
- Modify: `train_real.py`
- Test: `tests/test_real_training_alignment.py`

- [x] Add a failing source-alignment test requiring unsafe transitions to be inserted with `done=True`.
- [x] Run the test and confirm it fails.
- [x] Separate `unsafe_transition` from replay insertion and force a bounded negative reward for unsafe samples.
- [x] Run the test and confirm it passes.

### Task 3: Gate Learning And Checkpoint Saving

**Files:**
- Modify: `train_real.py`
- Test: `tests/test_real_training_alignment.py`

- [x] Add failing tests requiring Critic learning to wait for bootstrap sample balance and Actor learning to wait for a Critic-update threshold.
- [x] Add a failing test requiring `episode_actor_updates > 0` before saving `best_actor_real.pth`.
- [x] Implement the phase gates and checkpoint condition.
- [x] Run syntax checks, unit tests, and a short dry-run phase smoke test.
