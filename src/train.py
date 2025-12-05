"""Training script for the Chicken Counting model.

Features:
- Builds a simple dataset from `DatasetLoader` (uses the existing LabelMe-style JSONs)
- Generates Gaussian density maps from annotated points
- Training loop with AdamW, CurriculumLoss, validation (MAE/RMSE)
- Checkpoint saving to `checkpoints/`

Note: This script intentionally does not modify your augmentation code; it uses
images as provided (you can pre-run augmentation pipeline to create `data/augmented`).
"""
import argparse
import sys
from pathlib import Path
import math
import json
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# ensure repo `src` is importable when running script
PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ_ROOT))

from src.data_treatment.dataset_loader import DatasetLoader
from src.model.chicken_model import ChickenCountingModel
from src.model.losses import CurriculumLoss


def generate_density_map(points: List[Tuple[float, float]], H: int, W: int, sigma: float = 4.0) -> np.ndarray:
    """Create a density map of shape (H,W) from a list of (x,y) points (image coordinates).

    Uses a simple Gaussian kernel placed at each point and then returns the map.
    """
    import scipy.ndimage

    den = np.zeros((H, W), dtype=np.float32)
    for (x, y) in points:
        ix = int(round(x))
        iy = int(round(y))
        if 0 <= iy < H and 0 <= ix < W:
            den[iy, ix] += 1.0
    if sigma > 0:
        den = scipy.ndimage.gaussian_filter(den, sigma=sigma)
    return den


class ChickenDataset(Dataset):
    def __init__(self, samples: List[dict], crop_size: int = 256, transform=None):
        self.samples = samples
        self.crop_size = crop_size
        self.transform = transform or transforms.Compose([
            transforms.Resize((crop_size, crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img_path = s.get("image_path")
        points = s.get("points", [])
        if img_path is None:
            raise RuntimeError(f"Sample {idx} missing image_path")

        img = Image.open(img_path).convert("RGB")
        # resize image and scale points accordingly
        orig_w, orig_h = img.size
        img_resized = img.resize((self.crop_size, self.crop_size), resample=Image.BILINEAR)

        # scale points
        scale_x = self.crop_size / float(orig_w)
        scale_y = self.crop_size / float(orig_h)
        scaled_points = [(x * scale_x, y * scale_y) for (x, y) in points]

        img_t = self.transform(img_resized)

        den = generate_density_map(scaled_points, self.crop_size, self.crop_size, sigma=4.0)
        den_t = torch.from_numpy(den).unsqueeze(0)

        return img_t, den_t, len(points)


def collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    dens = torch.stack([b[1] for b in batch], dim=0)
    counts = torch.tensor([b[2] for b in batch], dtype=torch.float32)
    return imgs, dens, counts


def evaluate_counts(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    preds = []
    gts = []
    with torch.no_grad():
        for imgs, dens, counts in loader:
            imgs = imgs.to(device)
            out = model(imgs)
            out_np = out.detach().cpu().numpy()
            # sum per-sample
            for i in range(out_np.shape[0]):
                preds.append(float(out_np[i].sum()))
            gts.extend([float(c) for c in counts.tolist()])

    preds = np.array(preds)
    gts = np.array(gts)
    mae = float(np.mean(np.abs(preds - gts)))
    rmse = float(np.sqrt(np.mean((preds - gts) ** 2)))
    return mae, rmse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/dataset")
    parser.add_argument("--out_dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--device", default="auto", help="device to run on, e.g. 'cpu', 'cuda', or 'cuda:0'. Use 'auto' to select CUDA when available")
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # allow the user to request automatic device selection
    if args.device == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = args.device
    device = torch.device(device_str)

    loader_full = DatasetLoader(args.data_dir, load_images=False)
    samples = loader_full.as_list()

    # deterministic split
    rng = np.random.RandomState(args.seed)
    idxs = np.arange(len(samples))
    rng.shuffle(idxs)
    n_train = int(len(idxs) * args.train_frac)
    train_idxs = idxs[:n_train]
    val_idxs = idxs[n_train:]

    train_samples = [samples[i] for i in train_idxs]
    val_samples = [samples[i] for i in val_idxs]

    train_ds = ChickenDataset(train_samples, crop_size=256)
    val_ds = ChickenDataset(val_samples, crop_size=256)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = ChickenCountingModel(pretrained=False).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = None

    criterion = CurriculumLoss(lambda_ot=1.0, lambda_tv=1.0)

    best_val = math.inf
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for imgs, dens, counts in train_loader:
            imgs = imgs.to(device)
            dens = dens.to(device)

            pred = model(imgs)
            loss = criterion(pred, dens, epoch, args.epochs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += float(loss.detach().cpu().item())
            n_batches += 1

        avg_loss = running_loss / max(1, n_batches)

        val_mae, val_rmse = evaluate_counts(model, val_loader, device)

        print(f"Epoch {epoch}/{args.epochs} - train_loss: {avg_loss:.4f} - val_mae: {val_mae:.2f} - val_rmse: {val_rmse:.2f}")

        # checkpoint
        ckpt_path = out_dir / f"model_epoch_{epoch}.pth"
        torch.save({"epoch": epoch, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}, ckpt_path)

        if val_mae < best_val:
            best_val = val_mae
            torch.save(model.state_dict(), out_dir / "model_best.pth")


if __name__ == "__main__":
    main()
