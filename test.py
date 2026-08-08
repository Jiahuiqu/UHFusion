"""Evaluate a trained model on each configured fusion task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tgrs_fusion.config import get_device, load_config, set_seed
from tgrs_fusion.data import build_loader
from tgrs_fusion.engine.checkpoint import load_model_checkpoint
from tgrs_fusion.engine.evaluator import evaluate
from tgrs_fusion.models import HSI_Fusion_DualBranch_MoE


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pavia.json")
    parser.add_argument("--checkpoint", help="Override test.checkpoint")
    parser.add_argument("--data-root", help="Override data.root")
    parser.add_argument("--srf-path", help="Override data.srf_path")
    parser.add_argument("--split", help="Override test.split")
    parser.add_argument("--device", help="Override runtime.device")
    parser.add_argument("--save-predictions", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.data_root:
        config["data"]["root"] = str(Path(args.data_root).resolve())
    if args.srf_path:
        config["data"]["srf_path"] = str(Path(args.srf_path).resolve())

    runtime = config["runtime"]
    test_cfg = config["test"]
    device = get_device(args.device or runtime.get("device", "auto"))
    set_seed(int(runtime.get("seed", 3407)))
    checkpoint = args.checkpoint or test_cfg["checkpoint"]

    model = HSI_Fusion_DualBranch_MoE(**config.get("model", {})).to(device)
    load_model_checkpoint(model, checkpoint, device, strict=True)
    print(f"Loaded checkpoint: {checkpoint}")

    output_dir = Path(test_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}
    for task_id in test_cfg.get("task_ids", [0, 1, 2, 3]):
        loader = build_loader(
            config["data"],
            split=args.split or test_cfg.get("split", "test"),
            batch_size=int(test_cfg.get("batch_size", 1)),
            num_workers=int(runtime.get("num_workers", 4)),
            shuffle=False,
            task_mode="fixed",
            fixed_task=int(task_id),
            balance_tasks=False,
            pin_memory=device.type == "cuda",
        )
        prediction_dir = (
            output_dir / "predictions" / f"task_{task_id}"
            if args.save_predictions or test_cfg.get("save_predictions", False)
            else None
        )
        metrics = evaluate(
            model,
            loader,
            device,
            data_range=float(test_cfg.get("data_range", 1.0)),
            scale_factor=int(config["data"].get("scale_factor", 4)),
            save_dir=prediction_dir,
        )
        all_results[f"task_{task_id}"] = metrics
        rendered = " | ".join(f"{name}={value:.6f}" for name, value in metrics.items())
        print(f"Task {task_id}: {rendered}")

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
