"""Simple inference script to run the chicken counting model on the dataset.

Usage: python src/inference.py --weights checkpoints/model.pth --data_dir data/dataset --out_dir outputs
"""
import argparse
import sys
from pathlib import Path
import json
import numpy as np
from PIL import Image

import torch
from torchvision import transforms

# make project root importable when running this script directly
PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ_ROOT))

from src.data_treatment.dataset_loader import DatasetLoader
from src.model.chicken_model import ChickenCountingModel
from src.model.checkpoint import load_checkpoint


def pil_to_tensor(img: Image.Image, size=(256, 256)):
    tf = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return tf(img)


def run_inference(weights: str, data_dir: str, out_dir: str, device: str = "auto"):
    # allow "auto" to pick cuda when available, otherwise cpu
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    model = ChickenCountingModel(pretrained=False).to(device)
    if weights:
        load_checkpoint(weights, model, map_location=device.type)

    loader = DatasetLoader(data_dir, load_images=False)
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    model.eval()
    results = []
    with torch.no_grad():
        for i, sample in enumerate(loader):
            img_path = sample.get("image_path")
            if img_path is None:
                continue
            img = Image.open(img_path).convert("RGB")
            inp = pil_to_tensor(img).unsqueeze(0).to(device)
            den = model(inp)
            den_np = den.squeeze(0).squeeze(0).cpu().numpy()
            count = float(den_np.sum())

            # save density map as numpy and a simple heatmap png
            base = img_path.stem
            np.save(out_root / f"{base}_density.npy", den_np)

            # save heatmap using matplotlib (optional dependency)
            try:
                import matplotlib.pyplot as plt

                plt.imsave(out_root / f"{base}_density.png", den_np, cmap="jet")
            except Exception:
                pass

            results.append({"image": str(img_path), "count": count})

    # write summary
    with open(out_root / "results.json", "w") as fh:
        json.dump(results, fh, indent=2)

    print(f"Wrote {len(results)} results to {out_root}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=None)
    parser.add_argument("--data_dir", default="data/dataset")
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--device", default="auto", help="device to run on, e.g. 'cpu', 'cuda', or 'cuda:0'. Use 'auto' to select CUDA when available")
    args = parser.parse_args()
    run_inference(args.weights, args.data_dir, args.out_dir, args.device)


if __name__ == "__main__":
    main()
