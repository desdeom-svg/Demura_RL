from __future__ import annotations

import os
import sys
import time
from datetime import datetime


def append_line(path: str, message: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")


def main() -> int:
    if len(sys.argv) < 2:
        raise RuntimeError("log path is required")

    log_path = sys.argv[1]
    append_line(log_path, f"probe pid={os.getpid()} start")
    append_line(log_path, f"cwd={os.getcwd()}")
    append_line(log_path, f"python={sys.executable}")
    append_line(log_path, f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    import_start = time.time()
    append_line(log_path, "import torch start")
    import torch  # noqa: F401
    append_line(log_path, f"import torch done elapsed={time.time() - import_start:.3f}s")

    cuda_start = time.time()
    append_line(log_path, "torch.cuda.is_available start")
    import torch
    available = torch.cuda.is_available()
    append_line(
        log_path,
        f"torch.cuda.is_available done value={available} elapsed={time.time() - cuda_start:.3f}s",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
