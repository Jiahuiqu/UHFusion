"""Batch collation utilities."""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as F


def collate_balanced(batch, n_tasks: int = 4):
    """Balance represented tasks by under/over-sampling within a batch.

    This retains the training behavior of the original script. A task that
    is absent from the incoming batch cannot be synthesized safely, so the
    fallback reuses available samples without relabeling their modality.
    """
    batch = [sample for sample in batch if sample is not None]
    if not batch:
        raise ValueError("collate_balanced received an empty batch")

    groups = {task_id: [] for task_id in range(n_tasks)}
    for aux, lr_hs, gt_hs, task_id in batch:
        task_id = int(task_id)
        groups.setdefault(task_id, []).append((aux, lr_hs, gt_hs, task_id))

    per_task = max(1, len(batch) // n_tasks)
    balanced = []
    for task_id in range(n_tasks):
        items = groups.get(task_id, [])
        if len(items) >= per_task:
            balanced.extend(random.sample(items, per_task))
        elif items:
            balanced.extend(items)
            indices = np.random.randint(0, len(items), size=per_task - len(items))
            balanced.extend(items[int(index)] for index in indices)
        else:
            available = [sample for values in groups.values() for sample in values]
            indices = np.random.randint(0, len(available), size=per_task)
            balanced.extend(available[int(index)] for index in indices)

    aux_items, lr_items, gt_items, task_ids = zip(*balanced)
    max_channels = max(item.shape[0] for item in aux_items)
    aux_items = [
        F.pad(item, (0, 0, 0, 0, 0, max_channels - item.shape[0]))
        if item.shape[0] < max_channels
        else item
        for item in aux_items
    ]
    return (
        torch.stack(aux_items),
        torch.stack(lr_items),
        torch.stack(gt_items),
        torch.tensor(task_ids, dtype=torch.long),
    )
