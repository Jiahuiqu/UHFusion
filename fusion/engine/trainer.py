"""Training and validation loops."""

from __future__ import annotations

import torch
from tqdm import tqdm


def _move_batch(batch, device: torch.device):
    aux, lr_hs, gt_hs, task_id = batch
    return (
        aux.float().to(device, non_blocking=True),
        lr_hs.float().to(device, non_blocking=True),
        gt_hs.float().to(device, non_blocking=True),
        task_id.long().to(device, non_blocking=True),
    )


def train_one_epoch(model, dataloader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Training", leave=False):
        aux, lr_hs, gt_hs, task_id = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(aux, lr_hs, task_id)
        loss = criterion(prediction, gt_hs)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if len(dataloader) == 0:
        raise ValueError("Training DataLoader is empty")
    return total_loss / len(dataloader)


@torch.no_grad()
def validate(model, dataloader, criterion, device) -> float:
    model.eval()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Validation", leave=False):
        aux, lr_hs, gt_hs, task_id = _move_batch(batch, device)
        prediction = model(aux, lr_hs, task_id)
        total_loss += criterion(prediction, gt_hs).item()
    if len(dataloader) == 0:
        raise ValueError("Validation DataLoader is empty")
    return total_loss / len(dataloader)
