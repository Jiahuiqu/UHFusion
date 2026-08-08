"""JSON configuration and runtime helpers."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> dict:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    for key in ("root", "srf_path"):
        value = config["data"].get(key)
        if value:
            candidate = Path(os.path.expandvars(value)).expanduser()
            if not candidate.is_absolute():
                candidate = PROJECT_ROOT / candidate
            config["data"][key] = str(candidate.resolve())
    for section, key in (("training", "output_dir"), ("test", "checkpoint"), ("test", "output_dir")):
        value = config.get(section, {}).get(key)
        if value:
            candidate = Path(os.path.expandvars(value)).expanduser()
            if not candidate.is_absolute():
                candidate = PROJECT_ROOT / candidate
            config[section][key] = str(candidate.resolve())
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(spec: str = "auto") -> torch.device:
    if spec == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {spec}")
    return device
