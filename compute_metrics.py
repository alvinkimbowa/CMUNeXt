import argparse
import csv
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from monai.metrics import (
    compute_average_surface_distance,
    compute_dice,
    compute_hausdorff_distance,
)
from monai.networks.utils import one_hot

join = os.path.join


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def _load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _is_supported_seg_file(path: str) -> bool:
    path = path.lower()
    return path.endswith(".png") or path.endswith(".tif") or path.endswith(".tiff")


def _read_seg_image(path: str) -> np.ndarray:
    # Expect integer 2D label image.
    arr = np.array(Image.open(path))
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D label image at {path}, got shape {arr.shape}")
    if not np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.int64)
    return arr


def _stem(path: str) -> str:
    base = os.path.basename(path)
    lower = base.lower()
    if lower.endswith(".png") or lower.endswith(".tif"):
        return base[:-4]
    if lower.endswith(".tiff"):
        return base[:-5]
    return os.path.splitext(base)[0]


def _index_dir_by_stem(dir_path: str) -> Dict[str, str]:
    files = []
    for name in os.listdir(dir_path):
        full = join(dir_path, name)
        if not os.path.isfile(full):
            continue
        if _is_supported_seg_file(name):
            files.append(full)
    return {_stem(p): p for p in sorted(files)}


@dataclass(frozen=True)
class CaseMetrics:
    image_id: str
    dice: float
    hd95: float
    masd: float


def _nanmean(x: Sequence[float]) -> float:
    arr = np.array(list(x), dtype=np.float64)
    if np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def compute_case_metrics(
    pred_lbl: np.ndarray,
    gt_lbl: np.ndarray,
    num_classes: int,
    device: torch.device,
    ignore_empty: bool = True,
) -> Tuple[float, float, float]:
    """
    Returns (dice_mean, hd95_mean, masd_mean) aggregated across foreground classes.
    """
    if pred_lbl.shape != gt_lbl.shape:
        raise ValueError(
            f"Pred/GT shape mismatch: {pred_lbl.shape} vs {gt_lbl.shape}"
        )
    if num_classes < 2:
        raise ValueError(f"num_classes must be >= 2, got {num_classes}")

    # MONAI one_hot expects shape (B, 1, ...), so we add batch and channel dimensions.
    pred_t = torch.as_tensor(pred_lbl, dtype=torch.long, device=device).unsqueeze(0).unsqueeze(0)
    gt_t = torch.as_tensor(gt_lbl, dtype=torch.long, device=device).unsqueeze(0).unsqueeze(0)

    pred_oh = one_hot(pred_t, num_classes=num_classes)  # (1, C, ...)
    gt_oh = one_hot(gt_t, num_classes=num_classes)

    pred_oh = pred_oh.to(torch.float32)
    gt_oh = gt_oh.to(torch.float32)

    # Metrics return per-class values (background excluded via include_background=False).
    dice_pc = compute_dice(
        pred_oh, gt_oh, include_background=False, ignore_empty=ignore_empty
    )
    hd95_pc = compute_hausdorff_distance(
        pred_oh,
        gt_oh,
        include_background=False,
        percentile=95,
        directed=False,
    )
    asd_pc = compute_average_surface_distance(
        pred_oh,
        gt_oh,
        include_background=False,
        symmetric=True,
    )

    dice_vals = dice_pc.detach().cpu().numpy().reshape(-1)
    hd95_vals = hd95_pc.detach().cpu().numpy().reshape(-1)
    asd_vals = asd_pc.detach().cpu().numpy().reshape(-1)

    return _nanmean(dice_vals), _nanmean(hd95_vals), _nanmean(asd_vals)


def save_csv(
    rows: List[CaseMetrics],
    out_csv_path: str,
    add_aggregate_rows: bool = True,
) -> None:
    os.makedirs(os.path.dirname(out_csv_path) or ".", exist_ok=True)
    fieldnames = ["image_id", "dice", "hd95", "masd"]

    def _fmt(x: float) -> str:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return ""
        return f"{x:.6f}"

    with open(out_csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "image_id": r.image_id,
                    "dice": _fmt(r.dice),
                    "hd95": _fmt(r.hd95),
                    "masd": _fmt(r.masd),
                }
            )

        if add_aggregate_rows and rows:
            dice_vals = [r.dice for r in rows]
            hd95_vals = [r.hd95 for r in rows]
            masd_vals = [r.masd for r in rows]
            w.writerow(
                {
                    "image_id": "MEAN",
                    "dice": _fmt(_nanmean(dice_vals)),
                    "hd95": _fmt(_nanmean(hd95_vals)),
                    "masd": _fmt(_nanmean(masd_vals)),
                }
            )


def save_violin_plot(rows: List[CaseMetrics], out_path: str, title: str) -> None:
    # Import here to avoid requiring matplotlib for metric-only workflows.
    import matplotlib.pyplot as plt

    dice = np.array([r.dice for r in rows], dtype=np.float64)
    hd95 = np.array([r.hd95 for r in rows], dtype=np.float64)
    masd = np.array([r.masd for r in rows], dtype=np.float64)

    data = [dice, hd95, masd]
    labels = ["Dice", "HD95", "MASD"]

    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, values, label in zip(axes, data, labels):
        values = values[np.isfinite(values)]
        if len(values) > 0:
            parts = axis.violinplot(values, showmeans=True, showextrema=True)
        else:
            parts = {}
            axis.text(0.5, 0.5, "No finite values", ha="center", va="center", transform=axis.transAxes)
        for pc in parts.get("bodies", []):
            pc.set_alpha(0.6)
        axis.set_xticks([])
        axis.set_title(label)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle(title)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    figure.tight_layout()
    figure.savefig(out_path, dpi=200)
    plt.close(figure)


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute segmentation metrics from a prediction folder and a "
                    "ground-truth folder. Both are compared as-is: no resizing happens "
                    "here, so the predictions must already be in the space of the labels."
    )

    # Deliberately path-only. Every caller has its own results layout, so working out
    # where the predictions live is the calling script's job, not this one's.
    p.add_argument("--pred_dir", type=str, required=True, help="Directory of predicted label maps.")
    p.add_argument("--gt_dir", type=str, required=True, help="Directory of ground-truth label maps.")
    p.add_argument("--results_csv", type=str, required=True, help="Where to write the per-image CSV.")
    p.add_argument("--plot_path", type=str, default=None,
                   help="Where to write the violin plot. Defaults to metrics_violin.png "
                        "next to --results_csv; pass an empty string to skip the plot.")

    p.add_argument("--num_classes", type=int, default=None,
                   help="Number of classes including background. Read from --dataset_json "
                        "when omitted.")
    p.add_argument("--dataset_json", type=str, default=None,
                   help="dataset.json to take the class count from when --num_classes is "
                        "not given.")

    p.add_argument("--ignore_empty", type=str2bool, default=True, help="Ignore empty ground-truth classes.")
    p.add_argument("--title", type=str, default=None, help="Title for the violin plot.")
    return p.parse_args()


def resolve_num_classes(args) -> int:
    if args.num_classes is not None:
        return args.num_classes
    if args.dataset_json is None:
        raise ValueError("pass either --num_classes or --dataset_json")
    dataset_json = _load_json(args.dataset_json)
    # dataset.json "labels" can be dict label_id -> name or name -> id depending on source
    if not isinstance(dataset_json.get("labels"), dict):
        raise ValueError(f"Unsupported dataset.json labels format in {args.dataset_json}")
    return len(dataset_json["labels"])


def main():
    args = parse_args()

    device = torch.device("cpu")
    num_classes = resolve_num_classes(args)

    pred_dir, gt_dir = args.pred_dir, args.gt_dir
    if not os.path.isdir(pred_dir):
        raise FileNotFoundError(f"pred_dir not found: {pred_dir}")
    if not os.path.isdir(gt_dir):
        raise FileNotFoundError(f"gt_dir not found: {gt_dir}")

    pred_index = _index_dir_by_stem(pred_dir)
    gt_index = _index_dir_by_stem(gt_dir)
    common_ids = sorted(set(pred_index.keys()).intersection(gt_index.keys()))
    if not common_ids:
        raise RuntimeError(
            f"No matching PNG stems between pred_dir and gt_dir.\n"
            f"pred_dir={pred_dir}\n"
            f"gt_dir={gt_dir}"
        )

    rows: List[CaseMetrics] = []
    for image_id in common_ids:
        pred_lbl = _read_seg_image(pred_index[image_id])
        gt_lbl = _read_seg_image(gt_index[image_id])

        dice, hd95, masd = compute_case_metrics(
            pred_lbl=pred_lbl,
            gt_lbl=gt_lbl,
            num_classes=num_classes,
            device=device,
            ignore_empty=args.ignore_empty,
        )
        rows.append(CaseMetrics(image_id=image_id, dice=dice, hd95=hd95, masd=masd))

    out_csv_path = args.results_csv
    save_csv(rows, out_csv_path, add_aggregate_rows=True)

    plot_path = args.plot_path
    if plot_path is None:
        plot_path = join(os.path.dirname(out_csv_path) or ".", "metrics_violin.png")
    if plot_path:
        save_violin_plot(
            rows,
            plot_path,
            title=args.title or os.path.basename(os.path.dirname(out_csv_path) or "metrics"),
        )

    mean_dice = _nanmean([r.dice for r in rows])
    mean_hd95 = _nanmean([r.hd95 for r in rows])
    mean_masd = _nanmean([r.masd for r in rows])
    print(f"Wrote per-image + MEAN CSV: {out_csv_path} ({len(rows)} images)")
    if plot_path:
        print(f"Wrote violin plot: {plot_path}")
    print(f"MEAN Dice={mean_dice:.4f}, HD95={mean_hd95:.4f}, MASD={mean_masd:.4f}")


if __name__ == "__main__":
    main()
