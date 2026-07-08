# Two-Layer ROI Training Design

## Goal

Keep the reward and reference comparison focused on the center 200x200 area while allowing a larger action/support ROI to correct low-frequency and surrounding-field effects.

## Problem

The 2026-06-25 run reached `std=0.025446` before actor learning, mainly through physical-prior actions over the full 1000x2000 ROI. The current 200x200 run limits the controllable action area to the same small area as the reward, which removes the wider spatial degrees of freedom needed to correct the center patch.

Full-ROI action-effect logs are also misleading for 200x200 control because the center patch is only 2% of the full ROI.

## Design

Add a separate `reward_roi_size` parameter:

- `train_roi_size` continues to define the state/action support ROI. `0` means full 1000x2000 ROI.
- `reward_roi_size` defines the center crop used for reward and reference-normalized metrics.
- If `reward_roi_size` is not supplied, it defaults to `train_roi_size` for backward compatibility.
- Step diagnostics report both full-ROI and reward-ROI quantized action effect metrics.

## Recommended Training Shape

Use `reward_roi_size=200` and test larger action supports:

- Diagnostic: `train_roi_size=400`, then `600`, then `0` if 400/600 cannot beat the current result.
- Reward/reference remains center 200x200 in all cases.

## Success Criteria

- Center 200x200 `std` improves below `0.04` in physical-prior or critic-only phase.
- Reward-ROI changed ratio is visible in logs and comparable across action support sizes.
- Reference metrics remain comparison-only and do not drive best checkpoint or early stop.
