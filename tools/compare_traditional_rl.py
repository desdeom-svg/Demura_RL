"""Compare a traditional W16 compensation map with RL-generated delta maps.

This script is for R&D calibration only. It helps answer whether RL differs
from a traditional best compensation map mainly in:
1. amplitude,
2. low-frequency structure,
3. row/column profile,
4. high-frequency noise.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff

matplotlib.use("Agg")

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import PanelConfig


STEP_PATTERN = re.compile(
    r"ep(?P<episode>\d+)_g(?P<gray>\d+)_(?P<stage>init|step(?P<step>\d+))_delta_gray\.tiff$",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a traditional compensation BMP against RL delta TIFFs."
    )
    parser.add_argument(
        "--reference-bmp",
        type=Path,
        required=True,
        help="Absolute path to the traditional best compensation BMP.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Training run directory that contains ep*_delta_gray.tiff outputs.",
    )
    parser.add_argument(
        "--target-gray",
        type=float,
        default=16.0,
        help="Base gray used to convert absolute BMP into a delta map.",
    )
    parser.add_argument(
        "--lowpass-kernel",
        type=int,
        default=31,
        help="Odd Gaussian kernel size for low-frequency analysis.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many top candidates to list in the report.",
    )
    parser.add_argument(
        "--include-init",
        action="store_true",
        help="Include ep*_init_delta_gray.tiff in the comparison set.",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="*_delta_gray.tiff",
        help="Optional filename glob inside the run dir.",
    )
    return parser.parse_args()


def ensure_odd(value: int) -> int:
    return value if value % 2 == 1 else value + 1


def safe_std(values: np.ndarray) -> float:
    return float(np.std(values.astype(np.float64)))


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    if a.size != b.size:
        raise ValueError(f"Correlation shape mismatch: {a.shape} vs {b.shape}")
    a_std = float(np.std(a))
    b_std = float(np.std(b))
    if a_std < 1e-12 or b_std < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def l2_projection_scale(candidate: np.ndarray, reference: np.ndarray) -> float:
    ref = reference.astype(np.float64).ravel()
    cand = candidate.astype(np.float64).ravel()
    denom = float(np.dot(ref, ref))
    if denom < 1e-12:
        return float("nan")
    return float(np.dot(cand, ref) / denom)


def gaussian_lowpass(image: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel_size = ensure_odd(kernel_size)
    return cv2.GaussianBlur(image.astype(np.float32), (kernel_size, kernel_size), 0)


def row_profile(image: np.ndarray) -> np.ndarray:
    return image.mean(axis=1)


def col_profile(image: np.ndarray) -> np.ndarray:
    return image.mean(axis=0)


def load_reference_delta(path: Path, target_gray: float) -> Tuple[np.ndarray, np.ndarray]:
    ref = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if ref is None:
        raise FileNotFoundError(f"Failed to read reference BMP: {path}")

    y0 = PanelConfig.ROI_START_Y
    x0 = PanelConfig.ROI_START_X
    y1 = y0 + PanelConfig.ROI_HEIGHT
    x1 = x0 + PanelConfig.ROI_WIDTH
    roi = ref[y0:y1, x0:x1].astype(np.float32)
    if roi.shape != (PanelConfig.ROI_HEIGHT, PanelConfig.ROI_WIDTH):
        raise ValueError(
            f"Reference ROI shape mismatch: got {roi.shape}, "
            f"expected {(PanelConfig.ROI_HEIGHT, PanelConfig.ROI_WIDTH)}"
        )
    return roi - float(target_gray), roi


def load_delta_tiff(path: Path) -> np.ndarray:
    arr = tiff.imread(path).astype(np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D delta TIFF, got {arr.shape} from {path}")
    return arr


def parse_candidate_metadata(path: Path) -> Dict[str, Optional[int]]:
    match = STEP_PATTERN.match(path.name)
    if not match:
        return {"episode": None, "gray": None, "step": None, "is_init": None}
    step_text = match.group("step")
    return {
        "episode": int(match.group("episode")),
        "gray": int(match.group("gray")),
        "step": int(step_text) if step_text is not None else None,
        "is_init": 1 if match.group("stage") == "init" else 0,
    }


def calc_metrics(
    candidate: np.ndarray,
    reference: np.ndarray,
    lowpass_kernel: int,
) -> Dict[str, float]:
    candidate_low = gaussian_lowpass(candidate, lowpass_kernel)
    reference_low = gaussian_lowpass(reference, lowpass_kernel)
    candidate_high = candidate - candidate_low
    reference_high = reference - reference_low

    candidate_row = row_profile(candidate)
    reference_row = row_profile(reference)
    candidate_col = col_profile(candidate)
    reference_col = col_profile(reference)

    ref_abs_mean = float(np.mean(np.abs(reference)))
    cand_abs_mean = float(np.mean(np.abs(candidate)))

    diff = candidate - reference
    diff_low = candidate_low - reference_low

    return {
        "mean": float(np.mean(candidate)),
        "std": safe_std(candidate),
        "abs_mean": cand_abs_mean,
        "min": float(np.min(candidate)),
        "max": float(np.max(candidate)),
        "lowpass_std": safe_std(candidate_low),
        "highpass_std": safe_std(candidate_high),
        "row_std": safe_std(candidate_row),
        "col_std": safe_std(candidate_col),
        "corr_full": safe_corr(candidate, reference),
        "corr_lowpass": safe_corr(candidate_low, reference_low),
        "corr_highpass": safe_corr(candidate_high, reference_high),
        "corr_row": safe_corr(candidate_row, reference_row),
        "corr_col": safe_corr(candidate_col, reference_col),
        "rmse_full": float(np.sqrt(np.mean(diff ** 2))),
        "rmse_lowpass": float(np.sqrt(np.mean(diff_low ** 2))),
        "mae_full": float(np.mean(np.abs(diff))),
        "scale_vs_ref": cand_abs_mean / ref_abs_mean if ref_abs_mean > 1e-12 else float("nan"),
        "proj_scale_full": l2_projection_scale(candidate, reference),
        "proj_scale_lowpass": l2_projection_scale(candidate_low, reference_low),
    }


def collect_candidates(run_dir: Path, glob_pattern: str, include_init: bool) -> List[Path]:
    candidates: List[Path] = []
    for path in sorted(run_dir.glob(glob_pattern)):
        if not path.is_file():
            continue
        meta = parse_candidate_metadata(path)
        if meta["episode"] is None:
            continue
        if not include_init and meta["is_init"] == 1:
            continue
        candidates.append(path)
    return candidates


def save_summary_csv(rows: Sequence[Dict[str, object]], output_path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_reference_overview(
    reference_delta: np.ndarray,
    reference_abs: np.ndarray,
    output_path: Path,
    target_gray: float,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    im0 = axes[0].imshow(reference_abs, cmap="gray")
    axes[0].set_title(f"Traditional BMP ROI\nabsolute gray, target={target_gray:g}")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    vmax = float(np.percentile(np.abs(reference_delta), 99))
    vmax = max(vmax, 1e-3)
    im1 = axes[1].imshow(reference_delta, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    axes[1].set_title("Traditional delta ROI")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].hist(reference_delta.ravel(), bins=120, color="black")
    axes[2].set_title("Traditional delta histogram")
    axes[2].set_xlabel("delta gray")
    axes[2].set_ylabel("count")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_summary_scatter(rows: Sequence[Dict[str, object]], output_path: Path) -> None:
    if not rows:
        return
    x = np.array([float(row["corr_lowpass"]) for row in rows], dtype=np.float64)
    y = np.array([float(row["scale_vs_ref"]) for row in rows], dtype=np.float64)
    color = np.array([float(row["rmse_lowpass"]) for row in rows], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    scatter = ax.scatter(x, y, c=color, cmap="viridis", s=45)
    ax.set_xlabel("Low-pass correlation vs traditional delta")
    ax.set_ylabel("Amplitude ratio vs traditional |delta| mean")
    ax.set_title("RL delta structure vs traditional compensation")
    ax.axhline(1.0, color="tab:red", linestyle="--", linewidth=1)
    ax.grid(True, alpha=0.25)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Low-pass RMSE")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_candidate_comparison(
    reference: np.ndarray,
    candidate: np.ndarray,
    candidate_name: str,
    output_path: Path,
) -> None:
    diff = candidate - reference
    ref_row = row_profile(reference)
    cand_row = row_profile(candidate)
    ref_col = col_profile(reference)
    cand_col = col_profile(candidate)

    vmax = float(np.percentile(np.abs(np.concatenate([reference.ravel(), candidate.ravel()])), 99))
    vmax = max(vmax, 1e-3)
    diff_vmax = float(np.percentile(np.abs(diff), 99))
    diff_vmax = max(diff_vmax, 1e-3)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    im0 = axes[0, 0].imshow(reference, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    axes[0, 0].set_title("Traditional delta")
    axes[0, 0].axis("off")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(candidate, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    axes[0, 1].set_title(candidate_name)
    axes[0, 1].axis("off")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im2 = axes[0, 2].imshow(diff, cmap="coolwarm", vmin=-diff_vmax, vmax=diff_vmax)
    axes[0, 2].set_title("Candidate - Traditional")
    axes[0, 2].axis("off")
    fig.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

    axes[1, 0].plot(ref_row, label="Traditional", linewidth=1.5)
    axes[1, 0].plot(cand_row, label="RL", linewidth=1.1)
    axes[1, 0].set_title("Row mean profile")
    axes[1, 0].grid(True, alpha=0.25)
    axes[1, 0].legend()

    axes[1, 1].plot(ref_col, label="Traditional", linewidth=1.5)
    axes[1, 1].plot(cand_col, label="RL", linewidth=1.1)
    axes[1, 1].set_title("Column mean profile")
    axes[1, 1].grid(True, alpha=0.25)
    axes[1, 1].legend()

    axes[1, 2].hist(reference.ravel(), bins=120, alpha=0.65, label="Traditional")
    axes[1, 2].hist(candidate.ravel(), bins=120, alpha=0.65, label="RL")
    axes[1, 2].set_title("Delta histogram")
    axes[1, 2].grid(True, alpha=0.25)
    axes[1, 2].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def float_text(value: object) -> str:
    if isinstance(value, (float, np.floating)):
        if math.isnan(float(value)):
            return "nan"
        return f"{float(value):.4f}"
    return str(value)


def build_report(
    report_path: Path,
    reference_path: Path,
    run_dir: Path,
    rows: Sequence[Dict[str, object]],
    top_k: int,
) -> None:
    top_lowpass = sorted(
        rows,
        key=lambda row: (
            -float(row["corr_lowpass"]) if not math.isnan(float(row["corr_lowpass"])) else float("-inf"),
            float(row["rmse_lowpass"]),
        ),
    )[:top_k]
    top_rmse = sorted(rows, key=lambda row: float(row["rmse_lowpass"]))[:top_k]

    lines: List[str] = []
    lines.append("# Traditional vs RL delta analysis")
    lines.append("")
    lines.append(f"- Reference BMP: `{reference_path}`")
    lines.append(f"- Run dir: `{run_dir}`")
    lines.append(f"- Compared candidates: `{len(rows)}`")
    lines.append("")
    lines.append("## Top candidates by low-pass correlation")
    lines.append("")
    lines.append("| file | corr_lowpass | proj_scale_lowpass | rmse_lowpass | corr_row | corr_col |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for row in top_lowpass:
        lines.append(
            "| {file} | {corr_lowpass} | {proj_scale_lowpass} | {rmse_lowpass} | {corr_row} | {corr_col} |".format(
                file=row["file"],
                corr_lowpass=float_text(row["corr_lowpass"]),
                proj_scale_lowpass=float_text(row["proj_scale_lowpass"]),
                rmse_lowpass=float_text(row["rmse_lowpass"]),
                corr_row=float_text(row["corr_row"]),
                corr_col=float_text(row["corr_col"]),
            )
        )
    lines.append("")
    lines.append("## Top candidates by low-pass RMSE")
    lines.append("")
    lines.append("| file | rmse_lowpass | corr_lowpass | scale_vs_ref | highpass_std |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for row in top_rmse:
        lines.append(
            "| {file} | {rmse_lowpass} | {corr_lowpass} | {scale_vs_ref} | {highpass_std} |".format(
                file=row["file"],
                rmse_lowpass=float_text(row["rmse_lowpass"]),
                corr_lowpass=float_text(row["corr_lowpass"]),
                scale_vs_ref=float_text(row["scale_vs_ref"]),
                highpass_std=float_text(row["highpass_std"]),
            )
        )
    lines.append("")
    lines.append("## How to interpret")
    lines.append("")
    lines.append("- `corr_lowpass` high: RL is matching the large-scale mura structure.")
    lines.append("- `proj_scale_lowpass` near `1.0`: RL low-frequency amplitude is close to traditional compensation.")
    lines.append("- `scale_vs_ref < 1`: RL is too weak overall.")
    lines.append("- `scale_vs_ref > 1`: RL is stronger than the traditional map.")
    lines.append("- `highpass_std` rising while `corr_lowpass` is flat: RL is adding fine texture/noise instead of fixing main mura modes.")
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    reference_path = args.reference_bmp.resolve()
    output_dir = run_dir / "analysis_traditional"
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_delta, reference_abs = load_reference_delta(reference_path, args.target_gray)
    candidates = collect_candidates(run_dir, args.glob, args.include_init)
    if not candidates:
        raise FileNotFoundError(f"No candidate TIFFs found in {run_dir} using glob {args.glob!r}")

    rows: List[Dict[str, object]] = []
    candidate_cache: Dict[str, np.ndarray] = {}
    for path in candidates:
        candidate = load_delta_tiff(path)
        if candidate.shape != reference_delta.shape:
            raise ValueError(
                f"Candidate shape mismatch for {path.name}: got {candidate.shape}, "
                f"expected {reference_delta.shape}"
            )
        metrics = calc_metrics(candidate, reference_delta, args.lowpass_kernel)
        meta = parse_candidate_metadata(path)
        row: Dict[str, object] = {
            "file": path.name,
            "episode": meta["episode"],
            "step": meta["step"],
            "is_init": meta["is_init"],
        }
        row.update(metrics)
        rows.append(row)
        candidate_cache[path.name] = candidate

    rows.sort(key=lambda row: (int(row["episode"] or 0), int(row["step"] or -1), int(row["is_init"] or 0)))
    save_summary_csv(rows, output_dir / "summary.csv")
    plot_reference_overview(
        reference_delta,
        reference_abs,
        output_dir / "reference_overview.png",
        args.target_gray,
    )
    plot_summary_scatter(rows, output_dir / "summary_scatter.png")
    build_report(output_dir / "report.md", reference_path, run_dir, rows, args.top_k)

    best_corr = max(
        rows,
        key=lambda row: float(row["corr_lowpass"]) if not math.isnan(float(row["corr_lowpass"])) else float("-inf"),
    )
    best_rmse = min(rows, key=lambda row: float(row["rmse_lowpass"]))
    final_row = max(
        rows,
        key=lambda row: (int(row["episode"] or 0), int(row["step"] or -1), int(row["is_init"] or 0)),
    )

    selected = {
        "best_lowpass_corr": best_corr["file"],
        "best_lowpass_rmse": best_rmse["file"],
        "final_candidate": final_row["file"],
    }
    for tag, filename in selected.items():
        plot_candidate_comparison(
            reference_delta,
            candidate_cache[str(filename)],
            str(filename),
            output_dir / f"{tag}_{Path(str(filename)).stem}.png",
        )

    print(f"Saved analysis to: {output_dir}")
    print("Top by low-pass correlation:")
    for row in sorted(
        rows,
        key=lambda item: float(item["corr_lowpass"]) if not math.isnan(float(item["corr_lowpass"])) else float("-inf"),
        reverse=True,
    )[: args.top_k]:
        print(
            f"  {row['file']}: corr_lowpass={float_text(row['corr_lowpass'])}, "
            f"proj_scale_lowpass={float_text(row['proj_scale_lowpass'])}, "
            f"rmse_lowpass={float_text(row['rmse_lowpass'])}, "
            f"scale_vs_ref={float_text(row['scale_vs_ref'])}"
        )
    print("Top by low-pass RMSE:")
    for row in sorted(rows, key=lambda item: float(item["rmse_lowpass"]))[: args.top_k]:
        print(
            f"  {row['file']}: rmse_lowpass={float_text(row['rmse_lowpass'])}, "
            f"corr_lowpass={float_text(row['corr_lowpass'])}, "
            f"highpass_std={float_text(row['highpass_std'])}"
        )


if __name__ == "__main__":
    main()
