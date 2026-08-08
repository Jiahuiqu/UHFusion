"""Model evaluation and optional prediction export."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy.io import savemat
from tqdm import tqdm

from .metrics import batch_metrics


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    device,
    *,
    data_range: float = 1.0,
    scale_factor: int = 4,
    save_dir: str | Path | None = None,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    sample_count = 0
    output_dir = Path(save_dir) if save_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    for aux, lr_hs, gt_hs, task_id in tqdm(dataloader, desc="Testing", leave=False):
        aux = aux.float().to(device, non_blocking=True)
        lr_hs = lr_hs.float().to(device, non_blocking=True)
        gt_hs = gt_hs.float().to(device, non_blocking=True)
        task_id = task_id.long().to(device, non_blocking=True)
        prediction = model(aux, lr_hs, task_id)

        values = batch_metrics(
            prediction,
            gt_hs,
            data_range=data_range,
            scale_factor=scale_factor,
        )
        batch_size = prediction.shape[0]
        sample_count += batch_size
        for name, tensor in values.items():
            totals[name] = totals.get(name, 0.0) + tensor.sum().item()

        if output_dir:
            pred_np = prediction.detach().cpu().numpy()
            gt_np = gt_hs.detach().cpu().numpy()
            lr_np = lr_hs.detach().cpu().numpy()
            first_index = sample_count - batch_size
            for offset in range(batch_size):
                savemat(
                    output_dir / f"{first_index + offset:05d}.mat",
                    {
                        "prediction": pred_np[offset],
                        "ground_truth": gt_np[offset],
                        "lr_hsi": lr_np[offset],
                        "task_id": np.array([int(task_id[offset].item())]),
                    },
                )

    if sample_count == 0:
        raise ValueError("Test DataLoader is empty")
    return {name: value / sample_count for name, value in totals.items()}
