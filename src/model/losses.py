import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def counting_loss(pred_density: torch.Tensor, gt_density: torch.Tensor) -> torch.Tensor:
    """L1 loss between predicted and ground-truth counts (sums of density maps).

    pred_density, gt_density: tensors shaped [B,1,H,W]
    """
    pred_counts = pred_density.view(pred_density.size(0), -1).sum(dim=1)
    gt_counts = gt_density.view(gt_density.size(0), -1).sum(dim=1)
    return F.l1_loss(pred_counts, gt_counts, reduction="mean")


def total_variation_loss(density: torch.Tensor) -> torch.Tensor:
    """Isotropic total variation loss for a batch of density maps.

    density: [B,1,H,W]
    """
    dx = torch.abs(density[:, :, :, 1:] - density[:, :, :, :-1])
    dy = torch.abs(density[:, :, 1:, :] - density[:, :, :-1, :])
    loss = dx.mean() + dy.mean()
    return loss


def _sinkhorn(a: torch.Tensor, b: torch.Tensor, M: torch.Tensor, reg: float = 0.1, iters: int = 50) -> torch.Tensor:
    """Simple PyTorch Sinkhorn for entropic-regularized OT.

    a, b: probability vectors (sum to 1) of shape [n]
    M: cost matrix shape [n,n]
    returns transport cost (scalar)
    NOTE: this is not optimized for very large n. Use downsampled maps.
    """
    # u and v scaling vectors
    K = torch.exp(-M / reg)
    Kp = K / a.sum()
    u = torch.ones_like(a)
    v = torch.ones_like(b)
    for _ in range(iters):
        u = a / (K @ v)
        v = b / (K.t() @ u)
    # transport plan
    T = torch.diag(u) @ K @ torch.diag(v)
    cost = torch.sum(T * M)
    return cost


def optimal_transport_loss(pred: torch.Tensor, gt: torch.Tensor, downsample: int = 4, reg: float = 0.05) -> torch.Tensor:
    """Compute an approximate OT loss between predicted and gt density maps.

    - Downsamples maps by factor `downsample` to reduce complexity.
    - Flattens and normalizes to distributions.
    """
    # pred, gt: [B,1,H,W]
    B, C, H, W = pred.shape
    assert C == 1
    if downsample > 1:
        pred_ds = F.avg_pool2d(pred, kernel_size=downsample)
        gt_ds = F.avg_pool2d(gt, kernel_size=downsample)
    else:
        pred_ds = pred
        gt_ds = gt

    _, _, h, w = pred_ds.shape
    n = h * w
    device = pred.device

    # precompute coordinates
    ys = torch.arange(h, device=device, dtype=torch.float32)
    xs = torch.arange(w, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    coords = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=1)  # [n,2]

    # compute pairwise squared euclidean distances
    M = torch.cdist(coords, coords, p=2.0) ** 2  # [n,n]

    losses = []
    for i in range(B):
        a = pred_ds[i, 0].reshape(-1)
        b = gt_ds[i, 0].reshape(-1)
        # ensure non-negative
        a = F.relu(a)
        b = F.relu(b)
        # normalize to sum=1 (if empty, make uniform to avoid division by zero)
        suma = a.sum()
        sumb = b.sum()
        if suma.item() == 0:
            a = torch.ones_like(a) / float(n)
        else:
            a = a / suma
        if sumb.item() == 0:
            b = torch.ones_like(b) / float(n)
        else:
            b = b / sumb

        cost = _sinkhorn(a, b, M, reg=reg, iters=50)
        losses.append(cost)

    return torch.stack(losses).mean()


class CurriculumLoss(nn.Module):
    """Combine losses with curriculum scheduling of component weights.

    We implement a simple schedule: start with high weight on counting loss and
    progressively increase weight on OT and TV losses.
    """

    def __init__(self, lambda_ot: float = 1.0, lambda_tv: float = 1.0, lambda_cl: float = 1.0):
        super().__init__()
        self.lambda_ot = lambda_ot
        self.lambda_tv = lambda_tv
        self.lambda_cl = lambda_cl

    def forward(self, pred: torch.Tensor, gt: torch.Tensor, epoch: int, max_epoch: int) -> torch.Tensor:
        # base counting loss
        l_count = counting_loss(pred, gt)

        # schedule factor in [0,1]
        t = float(epoch) / float(max(1, max_epoch))
        # OT and TV ramp up with t
        ot_w = self.lambda_ot * t
        tv_w = self.lambda_tv * t

        l_ot = optimal_transport_loss(pred, gt) if ot_w > 0 else torch.tensor(0.0, device=pred.device)
        l_tv = total_variation_loss(pred) if tv_w > 0 else torch.tensor(0.0, device=pred.device)

        loss = l_count + ot_w * l_ot + tv_w * l_tv
        return loss


__all__ = [
    "counting_loss",
    "total_variation_loss",
    "optimal_transport_loss",
    "CurriculumLoss",
]
