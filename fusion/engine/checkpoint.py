"""Checkpoint save/load helpers supporting raw and full checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def _torch_load(path: str | Path, device: torch.device) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _extract_state_dict(payload: Any) -> dict:
    if isinstance(payload, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in payload and isinstance(payload[key], dict):
                return payload[key]
    return payload


def _strip_module_prefix(state_dict: dict) -> dict:
    if state_dict and all(key.startswith("module.") for key in state_dict):
        return {key[7:]: value for key, value in state_dict.items()}
    return state_dict


def load_model_checkpoint(model, path, device, strict: bool = True) -> Any:
    payload = _torch_load(path, device)
    state_dict = _strip_module_prefix(_extract_state_dict(payload))
    model.load_state_dict(state_dict, strict=strict)
    return payload


def resume_training(model, optimizer, scheduler, path, device) -> tuple[int, float]:
    payload = load_model_checkpoint(model, path, device)
    if not isinstance(payload, dict) or "optimizer" not in payload:
        return 0, float("inf")
    optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and "scheduler" in payload:
        scheduler.load_state_dict(payload["scheduler"])
    return int(payload.get("epoch", -1)) + 1, float(payload.get("best_loss", float("inf")))


def save_training_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch: int,
    best_loss: float,
    config: dict,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "best_loss": best_loss,
            "config": config,
        },
        target,
    )
