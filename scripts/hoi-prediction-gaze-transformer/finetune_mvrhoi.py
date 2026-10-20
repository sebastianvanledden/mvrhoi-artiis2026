#!/usr/bin/env python3
"""Fine-tune STTran-Gaze on MVRHOI using a VidHOI checkpoint."""

import sys
from pathlib import Path

sys.path.insert(0, "modules/object_tracking/yolov5")


import argparse

import torch
import re

import train_vidhoi as trainer
from modules.sthoip_transformer import sttran_gaze as sttran_module
import modules.object_tracking as tracking_module
from modules.object_tracking import feature_backbone as backbone_module
from modules.object_tracking import FeatureExtractionResNet101 as OriginalBackbone


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    # Base training options copied from train_vidhoi
    parser.add_argument("--cfg", type=str, default="configs/mvrhoi_finetune_rtx3060.yaml", help="Path to hyperparameter config")
    parser.add_argument("--weights", type=str, default="weights", help="Root folder for pretrained assets")
    parser.add_argument("--finetune-backbone", action="store_true", help="Finetune the ResNet backbone")
    parser.add_argument("--data", type=str, default="D:/MVRHOI", help="Dataset root path")
    parser.add_argument("--subset-train", type=int, default=-1, help="Subset length for debugging")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="Image size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup epochs")
    parser.add_argument("--project", default="runs/mvrhoi_finetune", help="Project folder")
    parser.add_argument("--name", default="sttran_mvrhoi", help="Run name")
    parser.add_argument("--save-period", type=int, default=1, help="Checkpoint save period")
    parser.add_argument("--disable-wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument("--gaze", type=str, default="cross", help="Gaze usage mode")
    parser.add_argument("--global-token", action="store_true", help="Enable global token (cross-attention only)")

    parser.add_argument("--checkpoint", type=str, default="weights/last.pt", help="VidHOI model checkpoint")
    parser.add_argument("--backbone-checkpoint", type=str, default=None, help="Optional backbone checkpoint")
    parser.add_argument("--strict-load", action="store_true", help="Use strict=True when loading checkpoint")
    return parser


def load_state(path: Path, device: torch.device):
    if path is None:
        return None
    if not path.exists():
        print(f"Warning: checkpoint {path} not found")
        return None
    state = torch.load(path, map_location=device)
    print(f"Loaded checkpoint from {path}")
    return state


def patch_models(model_state, backbone_state, strict=False):
    original_sttran = sttran_module.STTranGaze
    original_cross = sttran_module.STTranGazeCrossAttention
    original_backbone = OriginalBackbone

    applied = {"sttran": False}

    def load_partial(module, state_dict, label):
        if strict:
            try:
                module.load_state_dict(state_dict, strict=True)
            except RuntimeError as e:
                msg = str(e)
                # Provide a targeted hint for the most common mismatch we see:
                # transformer FFN dim differs (e.g., checkpoint 2048 vs config 1280).
                if "size mismatch" in msg and "linear1.weight" in msg:
                    m = re.search(r"shape torch\.Size\(\[(\d+),\s*(\d+)\]\) from checkpoint.*shape in current model is torch\.Size\(\[(\d+),\s*(\d+)\]\)", msg)
                    if m:
                        ck_ffn = m.group(1)
                        cur_ffn = m.group(3)
                        print(
                            f"ERROR: Strict checkpoint load failed due to architecture mismatch. "
                            f"This checkpoint appears to use dim_transformer_ffn={ck_ffn}, "
                            f"but the current model uses dim_transformer_ffn={cur_ffn}.\n"
                            f"Fix: set 'dim_transformer_ffn: {ck_ffn}' in your cfg to fully load this checkpoint, "
                            f"or re-run without --strict-load to allow partial loading."
                        )
                raise
            print(f"Loaded {label} checkpoint with strict=True")
            return [], [], []

        current_state = module.state_dict()
        filtered = {}
        skipped = []
        for key, value in state_dict.items():
            if key in current_state and current_state[key].shape == value.shape:
                filtered[key] = value
            else:
                skipped.append(key)
        load_result = module.load_state_dict(filtered, strict=False)
        if skipped:
            print(f"Skipped {len(skipped)} {label} keys due to shape mismatch")
        return load_result.missing_keys, load_result.unexpected_keys, skipped

    class PatchedSTTran(original_sttran):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if model_state is not None and not applied["sttran"]:
                missing, unexpected, skipped = load_partial(self, model_state, "STTran")
                print(
                    "Loaded STTran checkpoint ("
                    f"missing={len(missing)}, unexpected={len(unexpected)}, skipped={len(skipped)})"
                )
                applied["sttran"] = True

    class PatchedCross(original_cross):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if model_state is not None and not applied["sttran"]:
                missing, unexpected, skipped = load_partial(self, model_state, "STTran")
                print(
                    "Loaded STTran checkpoint ("
                    f"missing={len(missing)}, unexpected={len(unexpected)}, skipped={len(skipped)})"
                )
                applied["sttran"] = True

    class PatchedBackbone(original_backbone):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if backbone_state is not None:
                missing, unexpected, skipped = load_partial(self, backbone_state, "backbone")
                print(
                    "Loaded backbone checkpoint ("
                    f"missing={len(missing)}, unexpected={len(unexpected)}, skipped={len(skipped)})"
                )

    sttran_module.STTranGaze = PatchedSTTran
    sttran_module.STTranGazeCrossAttention = PatchedCross
    trainer.STTranGaze = PatchedSTTran
    trainer.STTranGazeCrossAttention = PatchedCross
    trainer.FeatureExtractionResNet101 = PatchedBackbone
    tracking_module.FeatureExtractionResNet101 = PatchedBackbone
    backbone_module.FeatureExtractionResNet101 = PatchedBackbone

    def restore():
        sttran_module.STTranGaze = original_sttran
        sttran_module.STTranGazeCrossAttention = original_cross
        trainer.STTranGaze = original_sttran
        trainer.STTranGazeCrossAttention = original_cross
        trainer.FeatureExtractionResNet101 = original_backbone
        tracking_module.FeatureExtractionResNet101 = original_backbone
        backbone_module.FeatureExtractionResNet101 = original_backbone

    return restore


def main():
    parser = build_parser()
    opt = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_state = load_state(Path(opt.checkpoint), device)
    backbone_state = load_state(Path(opt.backbone_checkpoint), device) if opt.backbone_checkpoint else None

    restore = patch_models(model_state, backbone_state, strict=opt.strict_load)
    try:
        trainer.train_gt_bbox(opt, str(device))
    finally:
        restore()


if __name__ == "__main__":
    main()
