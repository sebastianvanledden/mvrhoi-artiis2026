#!/usr/bin/env python3
"""
Generate YOLOv5+DeepSORT detections and gaze features for MVRHOI.
Outputs match the VidHOI detection-mode layout so eval_inference_vidhoi.py can
load them with --detection <name>.
"""

import sys
from pathlib import Path

sys.path.insert(0, "modules/object_tracking/yolov5")
sys.path.insert(0, ".")

import argparse
import json
import shelve

from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from modules.object_tracking import ObjectTracking, HeadDetection
from modules.gaze_following import GazeFollowing
from modules.gaze_following.head_association import assign_human_head_video
from common.vidhoi_dataset import VidHOIDataset
from common.data_io import FrameDatasetLoader
from common.transforms import YOLOv5Transform
from common.image_processing import convert_annotation_frame_to_video


def load_object_label_map(map_path: Path) -> Dict[int, int]:
    with map_path.open("r") as f:
        data = json.load(f)

    mapping: Dict[int, int] = {}

    def _coerce(key_obj, val_obj):
        try:
            key_int = int(key_obj)
            val_int = int(val_obj)
        except (TypeError, ValueError):
            return None
        return key_int, val_int

    if isinstance(data, dict):
        if all(isinstance(v, (int, str)) or v is None for v in data.values()):
            for raw_id, mapped_id in data.items():
                pair = _coerce(raw_id, mapped_id)
                if pair is not None:
                    mapping[pair[0]] = pair[1]
        if not mapping and "vidor_to_config_mapping" in data:
            nested = data["vidor_to_config_mapping"]
            if isinstance(nested, dict):
                for raw_id, info in nested.items():
                    if not isinstance(info, dict):
                        continue
                    mapped = info.get("config_id")
                    if mapped is None:
                        mapped = info.get("suggested_config_id")
                    pair = _coerce(raw_id, mapped)
                    if pair is not None:
                        mapping[pair[0]] = pair[1]
        if not mapping:
            for key in ("mapping", "object_map", "vidor_to_vidhoi"):
                nested = data.get(key)
                if isinstance(nested, dict):
                    for raw_id, mapped_id in nested.items():
                        pair = _coerce(raw_id, mapped_id)
                        if pair is not None:
                            mapping[pair[0]] = pair[1]
        if not mapping and isinstance(data.get("labels"), list):
            for entry in data["labels"]:
                if not isinstance(entry, dict):
                    continue
                src = entry.get("source_id", entry.get("src_id"))
                dst = entry.get("target_id", entry.get("dst_id"))
                pair = _coerce(src, dst)
                if pair is not None:
                    mapping[pair[0]] = pair[1]

    if not mapping:
        raise ValueError(
            f"No usable entries found in object mapping file {map_path}. "
            "Expected raw-id -> vidhoi-id pairs or a nested 'vidor_to_config_mapping'."
        )

    return mapping


def load_split_annotations(annotation_file: Path):
    with annotation_file.open("r") as f:
        annotations = json.load(f)
    if not annotations:
        raise ValueError(f"No annotations found in {annotation_file}")
    videos = {}
    current_video = None
    frame_ids = []
    last_frame_id = None
    for ann in annotations:
        video_key = f"{ann['video_folder']}/{ann['video_id']}"
        frame_id = ann["frame_id"]
        if video_key != current_video:
            if current_video is not None:
                videos[current_video] = list(frame_ids)
            current_video = video_key
            frame_ids = []
            last_frame_id = None
        if frame_id != last_frame_id:
            frame_ids.append(frame_id)
            last_frame_id = frame_id
    if current_video is not None:
        videos[current_video] = list(frame_ids)
    return videos


def ensure_serializable(arr):
    if isinstance(arr, np.ndarray):
        return arr.tolist()
    if torch.is_tensor(arr):
        return arr.detach().cpu().tolist()
    if isinstance(arr, (list, tuple)):
        return [ensure_serializable(x) for x in arr]
    return arr


def run_detection(args):
    device_str = args.device if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    dataset_root = Path(args.mvrhoi_root)
    images_dir = dataset_root / "images"

    annotations_dir = dataset_root / "VidHOI_annotation"
    if not annotations_dir.exists():
        annotations_dir = dataset_root / "Vidhoi_annotation"
    annotation_file = annotations_dir / f"{args.split}_frame_annots.json"
    if not annotation_file.exists():
        raise FileNotFoundError(f"Missing annotation file: {annotation_file}")

    output_dir = dataset_root / "VidHOI_detection"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Detection + Gaze Extraction (MVRHOI) ===")
    print(f"Split: {args.split}")
    print(f"Dataset root: {dataset_root}")
    print(f"Saving detections to: {output_dir}")
    print(f"Device: {device_str}")

    if args.object_map is not None:
        object_label_map = load_object_label_map(Path(args.object_map))
        print(f"Loaded {len(object_label_map)} object label mappings from {args.object_map}")
    else:
        object_label_map = {}

    # Build list of key frames per video
    split_keyframes = load_split_annotations(annotation_file)

    # Object tracker
    print("Initializing object tracker...")
    tracker = ObjectTracking(
        yolo_weights_path=args.yolo_weights,
        deep_sort_model_dir="weights/deep_sort/",
        config_path="configs/object_tracking.yaml",
        device=device_str,
    )
    img_size = 640
    stride = tracker.yolov5_stride

    # Iterate over frames
    print("Running object detection and tracking...")
    detections = {}
    for video_name, frame_ids in tqdm(split_keyframes.items(), desc="Tracking videos"):
        video_path = images_dir / video_name
        loader = FrameDatasetLoader(video_path, YOLOv5Transform(img_size, stride))
        clip_detections = {"bboxes": [], "ids": [], "labels": [], "confidences": [], "frame_ids": []}
        tracker.clear()
        key_frame_set = set(frame_ids)
        for frame_idx, (frame_tensor, frame_np, _, _, meta) in enumerate(loader):
            frame_gpu = frame_tensor.to(device)
            if frame_idx == 0:
                tracker.warmup(frame_gpu, frame_np)
            frame_id = Path(meta["frame_path"]).stem.split("_")[-1]
            bboxes, ids, labels, _, confidences, _ = tracker.track_one(
                frame_gpu, frame_np, draw=False
            )
            if frame_id in key_frame_set:
                boxes_serial = [ensure_serializable(bbox) for bbox in bboxes]
                ids_serial = [int(i) for i in ensure_serializable(ids)]
                raw_labels_serial = ensure_serializable(labels)
                if object_label_map:
                    labels_serial = [
                        int(object_label_map.get(int(lbl), args.object_default_label)) for lbl in raw_labels_serial
                    ]
                else:
                    labels_serial = [int(lbl) for lbl in raw_labels_serial]
                conf_serial = [float(c) for c in ensure_serializable(confidences)]
                clip_detections["bboxes"].append(boxes_serial)
                clip_detections["ids"].append(ids_serial)
                clip_detections["labels"].append(labels_serial)
                clip_detections["confidences"].append(conf_serial)
                clip_detections["frame_ids"].append(frame_id)
                clip_detections.setdefault("raw_labels", []).append([int(lbl) for lbl in raw_labels_serial])
        detections[video_name] = clip_detections
        if len(clip_detections["frame_ids"]) != len(frame_ids):
            missing = [fid for fid in frame_ids if fid not in clip_detections["frame_ids"]]
            if missing:
                print(f"Warning: missing detections for {video_name} frames: {missing[:5]}" + ("..." if len(missing) > 5 else ""))

    trace_name = f"{args.split}_trace_{args.detection_tag}.json"
    trace_path = output_dir / trace_name
    with trace_path.open("w") as f:
        json.dump(detections, f)
    print(f"Detections saved to {trace_path}")

    # Prepare dataset loader for gaze extraction (clip mode)
    dataset = VidHOIDataset(
        annotations_file=annotation_file,
        frames_dir=images_dir,
        transform=YOLOv5Transform(img_size, stride),
        annotation_mode="clip",
        train_ratio=0.0,
    )
    dataset.eval()
    dataloader = DataLoader(dataset, batch_size=None, shuffle=False)

    head_detector = HeadDetection(
        crowd_human_weight_path="weights/yolov5/crowdhuman_yolov5m.pt",
        config_path="configs/object_tracking.yaml",
        device=device_str,
    )
    gaze_model = GazeFollowing(
        weight_path="weights/detecting_attended/model_videoatttarget.pt",
        config_path="configs/gaze_following.yaml",
        device=device_str,
    )
    head_thresh = 0.5

    head_path = output_dir / f"{args.split}_frame_heads_{args.detection_tag}"
    gaze_path = output_dir / f"{args.split}_frame_gazes_{args.detection_tag}"
    inout_path = output_dir / f"{args.split}_frame_inout_{args.detection_tag}"
    head_db = shelve.open(str(head_path))
    gaze_db = shelve.open(str(gaze_path))
    inout_db = shelve.open(str(inout_path))

    try:
        for frames, _, meta in tqdm(dataloader, desc="Gaze extraction"):
            video_name = meta["video_name"]
            original_frames = meta["original_frames"]
            clip = detections[video_name]
            bboxes, ids, labels, _ = convert_annotation_frame_to_video(
                clip["bboxes"], clip["ids"], clip["labels"], clip["confidences"]
            )
            with torch.no_grad():
                head_list = assign_human_head_video(
                    frames,
                    original_frames,
                    bboxes,
                    ids,
                    labels,
                    head_detector,
                    head_thresh,
                    device_str,
                    method="hungarian",
                )
            head_db[video_name] = head_list
            head_db.sync()

            gaze_list = []
            inout_list = []
            hx_memory = {}
            for frame_idx, (head_map, frame0) in enumerate(zip(head_list, original_frames)):
                frame_np = frame0.numpy() if torch.is_tensor(frame0) else frame0
                gaze_dict = {}
                inout_dict = {}
                for human_id, head_bbox in head_map.items():
                    if len(head_bbox) == 0:
                        gaze_dict[human_id] = []
                        inout_dict[human_id] = []
                        continue
                    hidden_state = hx_memory.get(human_id)
                    with torch.no_grad():
                        heatmap, inout, hx, _, _, _ = gaze_model.detect_one(
                            frame_np,
                            np.asarray(head_bbox, dtype=np.float32),
                            hidden_state,
                            draw=False,
                        )
                    hx_memory[human_id] = (hx[0].detach(), hx[1].detach())
                    inout_value = 1 / (1 + np.exp(-inout))
                    if isinstance(heatmap, np.ndarray):
                        heatmap_np = heatmap
                    elif torch.is_tensor(heatmap):
                        heatmap_np = heatmap.detach().cpu().numpy()
                    else:
                        heatmap_np = np.asarray(heatmap)
                    gaze_dict[human_id] = heatmap_np
                    inout_dict[human_id] = float(np.squeeze(inout_value))
                gaze_list.append(gaze_dict)
                inout_list.append(inout_dict)
            gaze_db[video_name] = gaze_list
            gaze_db.sync()
            inout_db[video_name] = inout_list
            inout_db.sync()
    finally:
        head_db.close()
        gaze_db.close()
        inout_db.close()

    print(f"Heads saved to {head_path}")
    print(f"Gaze heatmaps saved to {gaze_path}")
    print(f"Gaze in/out saved to {inout_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate detection-mode inputs for MVRHOI")
    parser.add_argument("--mvrhoi_root", type=str, default="D:/MVRHOI", help="MVRHOI dataset root")
    parser.add_argument("--split", type=str, choices=["train", "val"], default="val", help="Dataset split")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device")
    parser.add_argument(
        "--yolo_weights",
        type=str,
        default="weights/yolov5/vidor_yolov5l.pt",
        help="YOLOv5 weights for object detector",
    )
    parser.add_argument(
        "--detection_tag",
        type=str,
        default="yolov5l_deepsort",
        help="Suffix used for detection files (match --detection argument)",
    )
    parser.add_argument(
        "--object-map",
        type=str,
        default=None,
        help="Optional JSON file mapping raw detector labels to VidHOI object ids.",
    )
    parser.add_argument(
        "--object-default-label",
        type=int,
        default=0,
        help="Fallback VidHOI object id when a raw detector label is missing from the mapping.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_detection(parse_args())
