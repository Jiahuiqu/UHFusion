"""Pavia hyperspectral fusion datasets and task construction."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from scipy.io import loadmat
from sklearn.decomposition import PCA
from skimage.transform import resize
from torch.utils.data import Dataset

TaskMode = Literal["random", "cyclic", "fixed"]


def _numeric_stem(path: Path) -> int:
    try:
        return int(path.stem)
    except ValueError as exc:
        raise ValueError(f"Expected a numeric MAT filename, got: {path.name}") from exc


def get_pc1(hsi: np.ndarray) -> np.ndarray:
    """Return the first principal component with shape ``(1, H, W)``."""
    channels, height, width = hsi.shape
    pixels = hsi.transpose(1, 2, 0).reshape(-1, channels)
    pc1 = PCA(n_components=1).fit_transform(pixels)
    return pc1.reshape(height, width, 1).transpose(2, 0, 1)


class BaseHSIDataset(Dataset):
    """Load paired GT-HSI, LR-HSI and PAN MAT files.

    Expected layout::

        root/
          train/{gtHS,LRHS,PAN}/*.mat
          val/{gtHS,LRHS,PAN}/*.mat
          test/{gtHS,LRHS,PAN}/*.mat
    """

    def __init__(
        self,
        root: str | Path,
        mode: str,
        size: int = 160,
        output_size: int = 128,
        hsi_channels: int = 102,
        scale_factor: int = 4,
        mat_key: str = "da",
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.mode = mode
        self.size = int(size)
        self.output_size = int(output_size)
        self.hsi_channels = int(hsi_channels)
        self.scale_factor = int(scale_factor)
        self.mat_key = mat_key

        split_root = self.root / mode
        self.gt_hs = self._list_mat_files(split_root / "gtHS")
        self.lr_hs = self._list_mat_files(split_root / "LRHS")
        self.pan = self._list_mat_files(split_root / "PAN")

        counts = (len(self.gt_hs), len(self.lr_hs), len(self.pan))
        if len(set(counts)) != 1:
            raise ValueError(
                f"Unmatched sample counts in {split_root}: "
                f"gtHS={counts[0]}, LRHS={counts[1]}, PAN={counts[2]}"
            )
        if not self.gt_hs:
            raise ValueError(f"No MAT files found under {split_root}")

    @staticmethod
    def _list_mat_files(directory: Path) -> list[Path]:
        if not directory.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {directory}")
        return sorted(directory.glob("*.mat"), key=_numeric_stem)

    def __len__(self) -> int:
        return len(self.gt_hs)

    def _load_array(self, path: Path) -> np.ndarray:
        payload = loadmat(path)
        if self.mat_key not in payload:
            raise KeyError(f"MAT key '{self.mat_key}' not found in {path}")
        return payload[self.mat_key]

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        gt_hs_data = resize(
            self._load_array(self.gt_hs[index]).reshape(
                self.hsi_channels, self.size, self.size
            ),
            (self.hsi_channels, self.output_size, self.output_size),
            mode="reflect",
            anti_aliasing=True,
            preserve_range=True,
        )
        lr_size = self.size // self.scale_factor
        lr_hs_data = resize(
            self._load_array(self.lr_hs[index]).reshape(
                self.hsi_channels, lr_size, lr_size
            ),
            (self.hsi_channels, self.output_size, self.output_size),
            mode="reflect",
            anti_aliasing=True,
            preserve_range=True,
        )
        pan_data = resize(
            self._load_array(self.pan[index]).reshape(1, self.size, self.size),
            (1, self.output_size, self.output_size),
            mode="reflect",
            anti_aliasing=True,
            preserve_range=True,
        )
        return pan_data, gt_hs_data, lr_hs_data


class MultiTaskDataset(Dataset):
    """Construct one of four auxiliary-input tasks for each HSI sample.

    Task IDs retain the original experiment definition:
    0 = PAN, 1 = selected HSI bands, 2 = SRF-projected MSI, 3 = missing/PC1.
    """

    def __init__(
        self,
        base_dataset: BaseHSIDataset,
        srf_path: str | Path | None,
        srf_key: str = "F_h",
        task_mode: TaskMode = "random",
        fixed_task: int | None = None,
        selected_bands: tuple[int, ...] = (59, 39, 19, 19),
        n_tasks: int = 4,
    ) -> None:
        super().__init__()
        if task_mode not in {"random", "cyclic", "fixed"}:
            raise ValueError(f"Unsupported task_mode: {task_mode}")
        if task_mode == "fixed" and fixed_task is None:
            raise ValueError("fixed_task is required when task_mode='fixed'")
        if fixed_task is not None and not 0 <= fixed_task < n_tasks:
            raise ValueError(f"fixed_task must be in [0, {n_tasks - 1}]")

        self.base_dataset = base_dataset
        self.task_mode = task_mode
        self.fixed_task = fixed_task
        self.selected_bands = selected_bands
        self.n_tasks = n_tasks
        self.srf: np.ndarray | None = None

        if task_mode != "fixed" or fixed_task == 2:
            if srf_path is None:
                raise ValueError("srf_path is required for random/cyclic mode and task 2")
            srf_file = Path(srf_path)
            payload = loadmat(srf_file)
            if srf_key not in payload:
                raise KeyError(f"MAT key '{srf_key}' not found in {srf_file}")
            self.srf = payload[srf_key].T

    def __len__(self) -> int:
        return len(self.base_dataset)

    def _task_for_index(self, index: int) -> int:
        if self.task_mode == "fixed":
            return int(self.fixed_task)
        if self.task_mode == "cyclic":
            return index % self.n_tasks
        return random.randrange(self.n_tasks)

    def __getitem__(self, index: int):
        pan_data, gt_hs_data, lr_hs_data = self.base_dataset[index]
        task_id = self._task_for_index(index)

        if task_id == 0:
            aux = np.repeat(pan_data, 4, axis=0)
        elif task_id == 1:
            aux = gt_hs_data[list(self.selected_bands), ...]
        elif task_id == 2:
            if self.srf is None:
                raise RuntimeError("SRF was not loaded for task 2")
            aux = np.tensordot(self.srf, gt_hs_data, axes=([1], [0]))
        else:
            aux = np.repeat(get_pc1(lr_hs_data), 4, axis=0)

        aux = torch.from_numpy(np.ascontiguousarray(aux, dtype=np.float32))
        lr_hs = torch.from_numpy(np.ascontiguousarray(lr_hs_data, dtype=np.float32))
        gt_hs = torch.from_numpy(np.ascontiguousarray(gt_hs_data, dtype=np.float32))
        return aux, lr_hs, gt_hs, task_id
