"""Common hyperspectral image quality metrics."""

from __future__ import annotations

import math

import torch


def batch_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    data_range: float = 1.0,
    scale_factor: int = 4,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    dims = (1, 2, 3)
    error = prediction - target
    mae = error.abs().mean(dim=dims)
    rmse = error.square().mean(dim=dims).sqrt()
    psnr = 20.0 * torch.log10(
        torch.as_tensor(data_range, device=prediction.device) / (rmse + eps)
    )

    dot = (prediction * target).sum(dim=1)
    pred_norm = prediction.square().sum(dim=1).sqrt()
    target_norm = target.square().sum(dim=1).sqrt()
    cosine = (dot / (pred_norm * target_norm + eps)).clamp(-1.0, 1.0)
    sam = torch.rad2deg(torch.acos(cosine)).mean(dim=(1, 2))

    band_rmse = error.square().mean(dim=(2, 3)).sqrt()
    band_mean = target.mean(dim=(2, 3)).abs()
    ergas = (100.0 / scale_factor) * torch.sqrt(
        ((band_rmse / (band_mean + eps)) ** 2).mean(dim=1)
    )
    return {"MAE": mae, "RMSE": rmse, "PSNR": psnr, "SAM": sam, "ERGAS": ergas}
