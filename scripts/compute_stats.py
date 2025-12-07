import argparse
from pathlib import Path
import numpy as np
import glob
import csv
import sys

try:
    from sklearn.metrics import r2_score
except Exception:
    r2_score = None


def read_gt_csv(path):
    mapping = {}
    with open(path, newline="", encoding="utf-8") as f:
        rdr = csv.reader(f)
        for row in rdr:
            if len(row) < 2:
                continue
            img, cnt = row[0].strip(), row[1].strip()
            try:
                mapping[img] = float(cnt)
            except Exception:
                continue
    return mapping


def guess_gt_csv():
    candidates = [
        Path("data/dataset/annotations_counts.csv"),
        Path("data/dataset/annotations.csv"),
        Path("data/dataset/gt_counts.csv"),
        Path("data/dataset/counts.csv"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def load_density_sums_from_dir(density_dir, suffixes=("_density.npy", ".npy")):
    density_dir = Path(density_dir)
    files = list(density_dir.glob("**/*.npy"))
    mapping = {}
    for p in files:
        name = p.stem
        # normalize name: remove _density suffix if present
        for s in suffixes:
            if name.endswith(s.replace('.npy','')):
                name = name[: -len(s.replace('.npy',''))]
        try:
            arr = np.load(p)
            mapping[name] = float(arr.sum())
        except Exception:
            continue
    return mapping


def read_labelme_json_counts(folder):
    """Read LabelMe-style JSON point annotations and return mapping image_id -> count."""
    folder = Path(folder)
    mapping = {}
    for p in folder.glob("*.json"):
        try:
            import json
            j = json.loads(p.read_text(encoding='utf-8'))
            shapes = j.get('shapes', [])
            cnt = 0
            for s in shapes:
                if s.get('shape_type') == 'point' or s.get('label'):
                    # count points labeled as chicken (or all points)
                    if 'label' in s:
                        if s.get('label') == 'chicken' or s.get('label').lower() == 'chicken':
                            cnt += 1
                        else:
                            # if not labeled, still consider points
                            if s.get('shape_type') == 'point':
                                cnt += 1
                    else:
                        if s.get('shape_type') == 'point':
                            cnt += 1
            name = p.stem
            mapping[name] = float(cnt)
        except Exception:
            continue
    return mapping


def load_pred_sums(pred_dir):
    pred_dir = Path(pred_dir)
    files = sorted(pred_dir.glob("*_density.npy"))
    if not files:
        files = sorted(pred_dir.glob("*.npy"))
    mapping = {}
    for p in files:
        name = p.stem.replace("_density", "")
        try:
            arr = np.load(p)
            mapping[name] = float(arr.sum())
        except Exception:
            continue
    return mapping


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_dir", default="outputs", help="Directory with predicted density .npy files")
    p.add_argument("--gt_csv", default=None, help="CSV file with image_id,count (no header)")
    p.add_argument("--gt_density_dir", default=None, help="Directory with ground-truth density .npy files (optional)")
    p.add_argument("--save_csv", default="stats_per_image.csv", help="Save per-image stats to CSV")
    p.add_argument("--top_k", type=int, default=20, help="Show top-K worst predictions")
    args = p.parse_args()

    pred_map = load_pred_sums(args.pred_dir)
    if not pred_map:
        print(f"No prediction files found in {args.pred_dir}")
        sys.exit(1)

    gt_map = {}
    if args.gt_csv:
        gt_map = read_gt_csv(args.gt_csv)
    else:
        guessed = guess_gt_csv()
        if guessed:
            print(f"Using guessed GT csv: {guessed}")
            gt_map = read_gt_csv(guessed)

    if args.gt_density_dir:
        gt_density_map = load_density_sums_from_dir(args.gt_density_dir)
        gt_map.update(gt_density_map)

    # If GT is still empty but there are files named similarly in data/dataset, try to find counts in folder names
    if not gt_map:
        print("No GT CSV provided and no guessed CSV found. Trying to find GT files in data/dataset (npy/.json)...")
        possible = Path("data/dataset")
        if possible.exists():
            gt_density_map = load_density_sums_from_dir(possible)
            if gt_density_map:
                print("Found GT density maps inside data/dataset")
                gt_map.update(gt_density_map)
            # try LabelMe JSON point annotations
            gt_json_map = read_labelme_json_counts(possible)
            if gt_json_map:
                print("Found LabelMe .json annotations inside data/dataset")
                # json counts likely keyed by image id (filename without extension)
                gt_map.update(gt_json_map)

    # Align predictions and GT
    common = sorted([k for k in gt_map.keys() if k in pred_map])
    if not common:
        # also try numeric ids from prediction filenames
        common = sorted([k for k in pred_map.keys() if k in gt_map])

    if not common:
        print("No overlapping image ids between GT and predictions.\nProvide --gt_csv or --gt_density_dir to help match files.")
        # Still allow reporting aggregate pred mean
        preds = np.array(list(pred_map.values()))
        print(f"Predictions: count={preds.size} mean={preds.mean():.3f} median={np.median(preds):.3f}")
        sys.exit(1)

    gt_vals = np.array([gt_map[k] for k in common])
    pred_vals = np.array([pred_map[k] for k in common])

    diffs = pred_vals - gt_vals
    absdiff = np.abs(diffs)

    mae = float(np.mean(absdiff))
    rmse = float(np.sqrt(np.mean(diffs ** 2)))
    mean_gt = float(np.mean(gt_vals))
    median_gt = float(np.median(gt_vals))
    relative_mae = mae / mean_gt if mean_gt > 0 else float('inf')
    mape = float(np.mean(np.abs(diffs / (gt_vals + 1e-8)))) * 100.0

    print("Summary:")
    print(f"  images matched: {len(common)}")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  Mean GT count: {mean_gt:.4f}")
    print(f"  Median GT count: {median_gt:.4f}")
    print(f"  Relative MAE (MAE / mean_gt): {relative_mae:.4f}")
    print(f"  MAPE (%): {mape:.2f}%")

    if r2_score is not None:
        try:
            r2 = r2_score(gt_vals, pred_vals)
            print(f"  R^2: {r2:.4f}")
        except Exception:
            pass

    # Save per-image CSV
    with open(args.save_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["image_id","gt_count","pred_count","abs_error","signed_error"])
        for k, g, pr, ae, se in zip(common, gt_vals, pred_vals, absdiff, diffs):
            writer.writerow([k, f"{g:.6f}", f"{pr:.6f}", f"{ae:.6f}", f"{se:.6f}"])
    print(f"Per-image stats saved to {args.save_csv}")

    # Print top-k worst
    idx = np.argsort(-absdiff)
    print(f"\nTop {args.top_k} worst predictions (image_id, gt, pred, abs_error):")
    for i in idx[: args.top_k]:
        print(f"  {common[i]}  GT={gt_vals[i]:.2f}  PRED={pred_vals[i]:.2f}  ERR={absdiff[i]:.2f}")

    # Optional: plot scatter if matplotlib available
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6,6))
        mn = min(gt_vals.min(), pred_vals.min())
        mx = max(gt_vals.max(), pred_vals.max())
        plt.scatter(gt_vals, pred_vals, s=8, alpha=0.6)
        plt.plot([mn, mx], [mn, mx], 'r--', linewidth=1)
        plt.xlabel('GT count')
        plt.ylabel('Predicted count')
        plt.title('GT vs Predicted')
        plt.tight_layout()
        outpng = 'gt_vs_pred.png'
        plt.savefig(outpng, dpi=150)
        print(f"Saved scatter plot to {outpng}")
    except Exception:
        pass

if __name__ == '__main__':
    main()
