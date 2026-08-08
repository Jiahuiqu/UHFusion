"""DataLoader construction."""

from __future__ import annotations

from functools import partial

from torch.utils.data import DataLoader

from .collate import collate_balanced
from .datasets import BaseHSIDataset, MultiTaskDataset


def build_loader(
    data_cfg: dict,
    split: str,
    batch_size: int,
    num_workers: int,
    *,
    shuffle: bool,
    task_mode: str,
    fixed_task: int | None = None,
    balance_tasks: bool = False,
    pin_memory: bool = True,
) -> DataLoader:
    base = BaseHSIDataset(
        root=data_cfg["root"],
        mode=split,
        size=data_cfg.get("source_size", 160),
        output_size=data_cfg.get("output_size", 128),
        hsi_channels=data_cfg.get("hsi_channels", 102),
        scale_factor=data_cfg.get("scale_factor", 4),
        mat_key=data_cfg.get("mat_key", "da"),
    )
    dataset = MultiTaskDataset(
        base,
        srf_path=data_cfg.get("srf_path"),
        srf_key=data_cfg.get("srf_key", "F_h"),
        task_mode=task_mode,
        fixed_task=fixed_task,
        selected_bands=tuple(data_cfg.get("selected_bands", [59, 39, 19, 19])),
        n_tasks=data_cfg.get("n_tasks", 4),
    )
    collate_fn = (
        partial(collate_balanced, n_tasks=data_cfg.get("n_tasks", 4))
        if balance_tasks
        else None
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        collate_fn=collate_fn,
    )
