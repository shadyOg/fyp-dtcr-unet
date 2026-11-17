"""Evaluate several trained checkpoints alone and as an ensemble.

Each model runs at its own input size; its lesion probabilities are resized back
to the resolution the data was preprocessed at, so all members and the ensemble
(the mean of the members' probabilities) are scored against the same
full-resolution masks.

Thresholds are chosen on the VALIDATION patients (same conservative rule as
training, `utils.choose_threshold`) and then applied once to the TEST patients.

Example:
    python ensemble.py --data /kaggle/working/data/processed \
        --checkpoints ckpts/resnet34_512/best_model.pth ckpts/dtcr_paper_net_256/best_model.pth \
        --out outputs/ensemble
"""

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from data.datasets import get_dataloaders
from utils import choose_threshold, load_model_from_checkpoint, predict_probs

THRESHOLDS = tuple(round(0.10 + 0.05 * i, 2) for i in range(17))   # 0.10 ... 0.90


def _metrics(c: np.ndarray) -> Dict[str, float]:
    tp, fp, fn, tn = c
    return {
        "dice": 100.0 * 2 * tp / max(2 * tp + fp + fn, 1.0),
        "sensitivity": 100.0 * tp / max(tp + fn, 1.0),
        "specificity": 100.0 * tn / max(tn + fp, 1.0),
        "accuracy": 100.0 * (tp + tn) / max(tp + tn + fp + fn, 1.0),
    }


@torch.no_grad()
def collect_counts(members: List[Tuple[str, torch.nn.Module, dict]], loader, device, use_amp: bool = True):
    """Per-patient confusion counts at every threshold, for each member and the ensemble.

    Returns:
        counts: {name: {patient_id: array (n_thresholds, 4) of [tp, fp, fn, tn]}}
        datasets: {patient_id: source dataset}
    """
    names = [n for n, _, _ in members] + (["ensemble"] if len(members) > 1 else [])
    counts = {n: {} for n in names}
    datasets: Dict[str, str] = {}
    amp_on = use_amp and device.type == "cuda"

    for batch in tqdm(loader, desc="[Ensemble]", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        target = batch["mask"].to(device) > 0.5
        size = images.shape[-2:]
        probs = {}
        for name, model, info in members:
            x = images
            if info["image_size"] and tuple(size) != (info["image_size"],) * 2:
                x = F.interpolate(images, size=(info["image_size"],) * 2, mode="area")
            with torch.amp.autocast(device_type=device.type, enabled=amp_on):
                _, p = predict_probs(model, x, tta=info["tta"])
            if p.shape[-2:] != size:
                p = F.interpolate(p, size=size, mode="bilinear", align_corners=False)
            probs[name] = p.float()
        if len(members) > 1:
            probs["ensemble"] = torch.stack(list(probs.values()), dim=0).mean(dim=0)

        dims = (1, 2, 3)
        for name, p in probs.items():
            per_t = []
            for t in THRESHOLDS:
                pt = p > t
                per_t.append(torch.stack([(pt & target).sum(dims), (pt & ~target).sum(dims),
                                          (~pt & target).sum(dims), (~pt & ~target).sum(dims)], dim=1))
            per_t = torch.stack(per_t, dim=1).double().cpu().numpy()     # (B, n_thresholds, 4)
            for i, pid in enumerate(batch["patient_id"]):
                if pid not in counts[name]:
                    counts[name][pid] = np.zeros((len(THRESHOLDS), 4))
                counts[name][pid] += per_t[i]
        for i, pid in enumerate(batch["patient_id"]):
            datasets[pid] = batch["dataset"][i]
    return counts, datasets


def summarise(per_patient_counts: Dict[str, np.ndarray], datasets: Dict[str, str], t_index: int) -> Dict:
    per_patient = {pid: _metrics(c[t_index]) for pid, c in per_patient_counts.items()}
    out = {}
    for k in ("dice", "sensitivity", "specificity", "accuracy"):
        vals = [m[k] for m in per_patient.values()]
        out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    out["by_dataset"] = {
        ds: float(np.mean([per_patient[p]["dice"] for p in per_patient if datasets[p] == ds]))
        for ds in sorted(set(datasets.values()))
    }
    out["patients"] = {pid: round(m["dice"], 2) for pid, m in per_patient.items()}
    return out


def sweep(per_patient_counts: Dict[str, np.ndarray]) -> Dict[str, float]:
    return {
        f"{t:.2f}": float(np.mean([_metrics(c[j])["dice"] for c in per_patient_counts.values()]))
        for j, t in enumerate(THRESHOLDS)
    }


def run_ensemble(checkpoints: List[str], data_dir: str, out_dir: str, batch_size: int = 8,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu", num_workers: int = 2) -> Dict:
    dev = torch.device(device)
    os.makedirs(out_dir, exist_ok=True)

    members = []
    for path in checkpoints:
        model, ckpt = load_model_from_checkpoint(path, dev)
        cfg = ckpt.get("config", {})
        name = f"{os.path.basename(os.path.dirname(path))}/{os.path.splitext(os.path.basename(path))[0]}"
        members.append((name, model, {"image_size": cfg.get("image_size"), "tta": bool(ckpt.get("tta", True))}))
        print(f"Member {name}: {cfg.get('arch', 'dtcr')} @ {cfg.get('image_size')} | TTA {ckpt.get('tta', True)}")

    # Native resolution loaders (image_size=None): every prediction is scored on full-resolution masks.
    _, val_loader, test_loader = get_dataloaders(data_dir, batch_size=batch_size, num_workers=num_workers,
                                                 image_size=None)
    val_counts, val_ds = collect_counts(members, val_loader, dev)
    test_counts, test_ds = collect_counts(members, test_loader, dev)

    results = {}
    print("\n" + "=" * 100)
    print(f"{'model':<40} {'thr':>5} | {'VAL dice':>8} | {'TEST dice':>16} {'SE':>6} {'SP':>6} | per dataset")
    print("=" * 100)
    for name in val_counts:
        val_sweep = sweep(val_counts[name])
        thr = choose_threshold(val_sweep)
        t_index = THRESHOLDS.index(float(thr))
        test = summarise(test_counts[name], test_ds, t_index)
        results[name] = {"threshold": float(thr), "val_dice": val_sweep[thr], "val_sweep": val_sweep,
                         "test": test, "test_sweep": sweep(test_counts[name])}
        per_ds = " ".join(f"{k}={v:.1f}" for k, v in test["by_dataset"].items())
        print(f"{name:<40} {thr:>5} | {val_sweep[thr]:8.2f} | {test['dice']['mean']:7.2f} +- {test['dice']['std']:5.2f} "
              f"{test['sensitivity']['mean']:6.1f} {test['specificity']['mean']:6.2f} | {per_ds}")
    print("=" * 100)
    print("All scores: patient-level, at the full preprocessed resolution; thresholds chosen on validation.")

    with open(os.path.join(out_dir, "ensemble_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoints alone and as an ensemble")
    parser.add_argument("--checkpoints", nargs="+", required=True, help="Checkpoint files to evaluate / combine")
    parser.add_argument("--data", required=True, help="Preprocessed data directory")
    parser.add_argument("--out", default="outputs/ensemble", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()
    run_ensemble(args.checkpoints, args.data, args.out, batch_size=args.batch_size, num_workers=args.num_workers)


if __name__ == "__main__":
    main()
