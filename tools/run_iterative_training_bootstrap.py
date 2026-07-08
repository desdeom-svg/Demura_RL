from __future__ import annotations

import os
import sys
import traceback
import importlib
from datetime import datetime


def append_line(path: str, message: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")


def main() -> int:
    if len(sys.argv) < 2:
        raise RuntimeError("bootstrap log path is required as argv[1]")

    bootstrap_log = sys.argv[1]
    train_args = sys.argv[2:]
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    append_line(bootstrap_log, f"Python bootstrap pid={os.getpid()} argv_count={len(train_args)}")
    append_line(bootstrap_log, f"Python bootstrap args={' '.join(train_args)}")
    append_line(bootstrap_log, f"Python bootstrap repo_root={repo_root}")

    try:
        probe_modules = [
            "matplotlib",
            "numpy",
            "torch",
            "tifffile",
            "matplotlib.pyplot",
            "config",
            "models",
            "real_world_env",
            "rl_training_utils",
        ]
        for module_name in probe_modules:
            append_line(bootstrap_log, f"Import probe start {module_name}")
            importlib.import_module(module_name)
            append_line(bootstrap_log, f"Import probe done {module_name}")

        append_line(bootstrap_log, "Importing train_real")
        import train_real

        append_line(bootstrap_log, "Imported train_real successfully")
        normalized_args = train_args[1:] if train_args and train_args[0].endswith(".py") else train_args
        sys.argv = ["train_real.py", *normalized_args]
        append_line(bootstrap_log, f"Normalized argv={' '.join(sys.argv)}")
        append_line(bootstrap_log, "Parsing args")
        parsed_args = train_real.parse_args()
        append_line(bootstrap_log, f"Parsed args gray={parsed_args.gray} episodes={parsed_args.episodes} steps={parsed_args.steps}")
        append_line(bootstrap_log, "Calling train()")
        train_real.train(parsed_args)
        append_line(bootstrap_log, "train() returned normally")
        return 0
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        append_line(bootstrap_log, f"SystemExit raised with code={code}")
        return code
    except Exception:
        append_line(bootstrap_log, "Unhandled exception in bootstrap:")
        append_line(bootstrap_log, traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
