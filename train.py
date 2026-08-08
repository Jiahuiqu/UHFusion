"""Train the task-guided HSI fusion model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from tgrs_fusion.config import get_device, load_config, set_seed
from tgrs_fusion.data import build_loader
from tgrs_fusion.engine.checkpoint import resume_training, save_training_checkpoint
from tgrs_fusion.engine.trainer import train_one_epoch, validate
from tgrs_fusion.models import HSI_Fusion_DualBranch_MoE


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pavia.json")
    parser.add_argument("--data-root", help="Override data.root")
    parser.add_argument("--srf-path", help="Override data.srf_path")
    parser.add_argument("--resume", help="Resume from a full or raw checkpoint")
    parser.add_argument("--device", help="Override runtime.device, e.g. cuda:0 or cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.data_root:
        config["data"]["root"] = str(Path(args.data_root).resolve())
    if args.srf_path:
        config["data"]["srf_path"] = str(Path(args.srf_path).resolve())

    runtime = config["runtime"]
    training = config["training"]
    device = get_device(args.device or runtime.get("device", "auto"))
    set_seed(int(runtime.get("seed", 3407)))

    train_loader = build_loader(
        config["data"],
        split=config["data"].get("train_split", "train"),
        batch_size=int(training["batch_size"]),
        num_workers=int(runtime.get("num_workers", 4)),
        shuffle=True,
        task_mode=config["data"].get("train_task_mode", "random"),
        balance_tasks=bool(training.get("balance_tasks", True)),
        pin_memory=device.type == "cuda",
    )
    val_loader = build_loader(
        config["data"],
        split=config["data"].get("val_split", "val"),
        batch_size=int(training.get("val_batch_size", training["batch_size"])),
        num_workers=int(runtime.get("num_workers", 4)),
        shuffle=False,
        task_mode=config["data"].get("val_task_mode", "cyclic"),
        balance_tasks=False,
        pin_memory=device.type == "cuda",
    )

    model = HSI_Fusion_DualBranch_MoE(**config.get("model", {})).to(device)
    criterion_name = training.get("criterion", "L1").upper()
    criterion = nn.L1Loss() if criterion_name == "L1" else nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(training.get("learning_rate", 1e-3)),
        weight_decay=float(training.get("weight_decay", 1e-6)),
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=list(training.get("milestones", [250, 550, 750])),
        gamma=float(training.get("gamma", 0.1)),
    )

    output_dir = Path(training["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 0
    best_loss = float("inf")
    if args.resume:
        start_epoch, best_loss = resume_training(
            model, optimizer, scheduler, args.resume, device
        )
        print(f"Resumed at epoch {start_epoch}; best validation loss={best_loss:.6f}")

    epochs = int(training.get("epochs", 1000))
    for epoch in range(start_epoch, epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step()
        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"train={train_loss:.6f} val={val_loss:.6f}"
        )

        improved = val_loss < best_loss
        if improved:
            best_loss = val_loss
        save_training_checkpoint(
            output_dir / "latest.pth",
            model,
            optimizer,
            scheduler,
            epoch,
            best_loss,
            config,
        )
        if improved and epoch + 1 >= int(training.get("save_best_after", 100)):
            torch.save(model.state_dict(), output_dir / "best.pth")
            print(f"Saved new best model: val={best_loss:.6f}")


if __name__ == "__main__":
    main()
