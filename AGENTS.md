# Workspace Guidance

This project contains large generated outputs, binaries, and training artifacts that can slow down interactive Codex sessions.

Default focus areas:
- `config.py`
- `models.py`
- `real_world_env.py`
- `train_real.py`
- `tools/`

Avoid scanning these paths unless the task explicitly needs them:
- `build/`
- `dist/`
- `Log/`
- `TrainLogs/`
- `ResultData/`
- `TransferBmp/`
- `images/`
- `imageCompre/00_tools/`
- `opencv_world410.dll`
- `PreDemura.exe`

When searching, prefer targeted `rg` queries over recursive broad scans from the repository root.
