#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

sys.path.insert(0, "modules/object_tracking/yolov5")

import argparse
import numpy as np
import cv2
import torch
from pathlib import Path
import json
from tqdm import tqdm
from collections import deque, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from common.config_parser import get_config
from common.data_io import VideoDatasetLoader
from common.transforms import STTranTransform, YOLOv5Transform
from common.inference_utils import fill_sttran_entry_inference
from common.image_processing import convert_annotation_frame_to_video
from common.model_utils import (
    bbox_pair_generation,
    concat_separated_head,
    construct_sliding_window,
    generate_sliding_window_mask,
)
from common.plot import draw_bboxes
from common.metrics_utils import generate_triplets_scores

from modules.object_tracking import HeadTracking, ObjectTracking
from modules.gaze_following import GazeFollowing
from modules.gaze_following.head_association import assign_human_head_frame
from modules.sthoip_transformer.sttran_gaze import STTranGazeCrossAttention
from modules.object_tracking import FeatureExtractionResNet101


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="", help="path to source")
    parser.add_argument("--future", type=int, default=0, help="seconds in future")
    parser.add_argument("--cfg-root", type=str, default="configs", help="root folder for config files")
    parser.add_argument("--eval-cfg", type=str, default="eval_mvrhoi_f0.yaml", help="evaluation config file")
    parser.add_argument("--weights-root", type=str, default="weights", help="root folder for detector and gaze weights")
    parser.add_argument("--model", type=str, default="weights/mvrhoi/last.pt", help="path to STTran checkpoint finetuned on MVRHOI")
    parser.add_argument("--backbone", type=str, default="weights/backbone", help="path to ResNet backbone weights or directory")
    parser.add_argument(
        "--backbone-finetuned",
        action="store_true",
        help="Treat --backbone as a fine-tuned ResNet state dict instead of downloading the default weights.",
    )
    parser.add_argument("--semantic", type=str, default="weights/semantic", help="path to word vector directory")
    parser.add_argument("--imgsz", type=int, default=640, help="train, val image size (pixels)")
    parser.add_argument("--hoi-thres", type=float, default=0.25, help="threshold for HOI score")
    parser.add_argument("--out", type=str, default="output", help="output folder")
    parser.add_argument("--print", action="store_true", help="print HOIs")
    parser.add_argument("--video", action="store_true", help="write output video")
    parser.add_argument(
        "--oracle-ann",
        type=str,
        default=None,
        help="Optional JSON annotations to run in oracle mode for this video",
    )
    parser.add_argument(
        "--human-label",
        type=int,
        default=0,
        help="Label index representing humans in annotations (default: 0)",
    )
    parser.add_argument(
        "--oracle-iou-thres",
        type=float,
        default=0.3,
        help="IoU threshold to map detector heads to oracle humans",
    )
    parser.add_argument(
        "--object-map",
        type=str,
        default=None,
        help="Optional JSON mapping from source dataset object IDs to VidHOI IDs; used to display original class names.",
    )
    opt = parser.parse_args()
    return opt


def _parse_bbox_coords(bbox_data: Any, frame_idx: Optional[int] = None) -> Optional[List[float]]:
    if bbox_data is None:
        return None
    # list or tuple [x1, y1, x2, y2]
    if isinstance(bbox_data, (list, tuple)) and len(bbox_data) >= 4:
        values = list(bbox_data)
        if len(values) > 4 and frame_idx is not None:
            try:
                first_val = int(values[0])
            except (TypeError, ValueError):
                first_val = None
            if first_val is not None and abs(first_val - frame_idx) <= 1:
                values = values[1:]
        try:
            return [float(values[i]) for i in range(4)]
        except (TypeError, ValueError, IndexError):
            return None
    # dict with known keys
    if isinstance(bbox_data, dict):
        key_sets = [
            {"xmin", "ymin", "xmax", "ymax"},
            {"x1", "y1", "x2", "y2"},
            {"left", "top", "right", "bottom"},
        ]
        for keys in key_sets:
            if keys.issubset(bbox_data.keys()):
                try:
                    mapping = {
                        "xmin": "xmin",
                        "ymin": "ymin",
                        "xmax": "xmax",
                        "ymax": "ymax",
                        "x1": "x1",
                        "y1": "y1",
                        "x2": "x2",
                        "y2": "y2",
                        "left": "left",
                        "top": "top",
                        "right": "right",
                        "bottom": "bottom",
                    }
                    names = list(keys)
                    vals = [float(bbox_data[mapping[n]]) for n in names]
                except (TypeError, ValueError, KeyError):
                    continue
                # ensure order xmin,ymin,xmax,ymax
                if "xmin" in keys:
                    return [vals[names.index("xmin")], vals[names.index("ymin")], vals[names.index("xmax")], vals[names.index("ymax")]]
                if "x1" in keys:
                    return [vals[names.index("x1")], vals[names.index("y1")], vals[names.index("x2")], vals[names.index("y2")]]
                if "left" in keys:
                    return [vals[names.index("left")], vals[names.index("top")], vals[names.index("right")], vals[names.index("bottom")]]
        # nested under bbox field
        for nested_key in ("bbox", "box", "bbox_xyxy", "xyxy"):
            if nested_key in bbox_data:
                return _parse_bbox_coords(bbox_data[nested_key], frame_idx)
    return None


def _parse_object_entry(obj: Dict[str, Any], frame_idx: int, fallback_counter: int) -> Optional[Tuple[int, int, List[float]]]:
    label = (
        obj.get("label")
        or obj.get("class_id")
        or obj.get("category_id")
        or obj.get("object_class")
        or obj.get("obj_class")
    )
    if label is None:
        return None
    try:
        label_int = int(label)
    except (TypeError, ValueError):
        return None

    bbox = obj.get("bbox") or obj.get("bbox_xyxy") or obj.get("box")
    bbox_coords = _parse_bbox_coords(bbox, frame_idx) or _parse_bbox_coords(obj, frame_idx)
    if bbox_coords is None:
        return None

    obj_id = (
        obj.get("id")
        or obj.get("object_id")
        or obj.get("track_id")
        or obj.get("tid")
        or obj.get("instance_id")
    )
    if obj_id is None:
        obj_id = frame_idx * 1000 + fallback_counter
    try:
        obj_id_int = int(obj_id)
    except (TypeError, ValueError):
        obj_id_int = frame_idx * 1000 + fallback_counter

    return obj_id_int, label_int, bbox_coords


def load_oracle_annotations(path: Path, target_video: Optional[str] = None) -> Dict[int, Dict[str, List]]:
    with path.open("r") as f:
        data = json.load(f)

    if isinstance(data, dict):
        if "frames" in data and isinstance(data["frames"], list):
            frame_entries = data["frames"]
        elif "annotations" in data and isinstance(data["annotations"], list):
            frame_entries = data["annotations"]
        else:
            frame_entries = [data]
    elif isinstance(data, list):
        frame_entries = data
    else:
        raise ValueError(f"Unsupported oracle annotation format in {path}")

    frames: Dict[int, Dict[str, List]] = {}

    for entry in frame_entries:
        if not isinstance(entry, dict):
            continue
        entry_video = (
            entry.get("video_name")
            or entry.get("video")
            or entry.get("video_path")
            or entry.get("name")
        )
        if target_video and entry_video:
            if Path(str(entry_video)).stem != Path(str(target_video)).stem:
                continue
        frame_id = entry.get("frame_id") or entry.get("frame") or entry.get("frameIndex")
        if frame_id is None:
            continue
        try:
            frame_idx = int(frame_id)
        except (TypeError, ValueError):
            continue

        if "bboxes_gt" in entry and "labels_gt" in entry:
            frame_store = frames.setdefault(frame_idx, {"ids": [], "labels": [], "bboxes": []})
            ids_gt = entry.get("ids_gt") or []
            bboxes_gt = entry.get("bboxes_gt") or []
            labels_gt = entry.get("labels_gt") or []
            for idx, (bbox_gt, label_gt) in enumerate(zip(bboxes_gt, labels_gt)):
                bbox_coords = _parse_bbox_coords(bbox_gt, frame_idx)
                if bbox_coords is None:
                    continue
                try:
                    label_int = int(label_gt)
                except (TypeError, ValueError):
                    continue
                if idx < len(ids_gt):
                    obj_id_val = ids_gt[idx]
                else:
                    obj_id_val = frame_idx * 1000 + idx
                try:
                    obj_id_int = int(obj_id_val)
                except (TypeError, ValueError):
                    obj_id_int = frame_idx * 1000 + idx
                frame_store["ids"].append(obj_id_int)
                frame_store["labels"].append(label_int)
                frame_store["bboxes"].append(bbox_coords)
            # allow concurrent object format parsing as well if present
            if entry.get("objects") is None:
                continue

        objects = (
            entry.get("objects")
            or entry.get("detections")
            or entry.get("annotations")
            or entry.get("instances")
        )
        if not isinstance(objects, list):
            # Support flattened format where entry itself represents an object
            objects = [entry]

        frame_store = frames.setdefault(frame_idx, {"ids": [], "labels": [], "bboxes": []})
        for idx, obj in enumerate(objects):
            if not isinstance(obj, dict):
                continue
            parsed = _parse_object_entry(obj, frame_idx, idx)
            if parsed is None:
                continue
            obj_id_int, label_int, bbox_coords = parsed
            frame_store["ids"].append(obj_id_int)
            frame_store["labels"].append(label_int)
            frame_store["bboxes"].append(bbox_coords)

    # ensure lists are aligned
    for frame_idx, frame_store in frames.items():
        counts = {len(frame_store["ids"]), len(frame_store["labels"]), len(frame_store["bboxes"])}
        if len(counts) != 1:
            raise ValueError(
                f"Oracle annotation mismatch in frame {frame_idx}: ids/labels/bboxes length differ"
            )
    return frames


def lookup_oracle_frame(annotations: Optional[Dict[int, Dict[str, List]]], frame_idx: int) -> Optional[Dict[str, List]]:
    if not annotations:
        return None
    if frame_idx in annotations:
        return annotations[frame_idx]
    # try zero-based fallback
    return annotations.get(frame_idx - 1)


def bbox_iou(box_a: List[float], box_b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter_area
    if denom <= 0:
        return 0.0
    return inter_area / denom


def remap_gaze_to_oracle(
    detection_gaze: Dict[int, Any],
    detection_ids: np.ndarray,
    detection_bboxes: np.ndarray,
    oracle_frame: Optional[Dict[str, List]],
    human_label: int,
    iou_thres: float,
) -> Dict[int, List[List[float]]]:
    zero_heatmap = np.zeros((64, 64), dtype=np.float32)

    # Prepare detection entries (id, bbox, heatmap)
    detection_entries: List[Tuple[int, List[float], np.ndarray]] = []
    for det_id, det_bbox in zip(detection_ids, detection_bboxes):
        try:
            det_id_int = int(det_id)
        except (TypeError, ValueError):
            continue
        heatmap_raw = detection_gaze.get(det_id_int, [])
        if isinstance(heatmap_raw, np.ndarray):
            heatmap = heatmap_raw.astype(np.float32)
        else:
            try:
                heatmap = np.asarray(heatmap_raw, dtype=np.float32)
            except ValueError:
                heatmap = zero_heatmap
        if heatmap.shape != (64, 64):
            heatmap = zero_heatmap
        detection_entries.append((det_id_int, det_bbox.tolist(), heatmap))

    if oracle_frame is None or not oracle_frame.get("ids"):
        # fallback to detection ids
        fallback = {}
        for det_id, _, heatmap in detection_entries:
            fallback[det_id] = heatmap.tolist()
        return fallback

    oracle_ids = oracle_frame["ids"]
    oracle_labels = oracle_frame["labels"]
    oracle_bboxes = oracle_frame["bboxes"]

    assigned_det_ids = set()
    result: Dict[int, List[List[float]]] = {}

    for obj_id, label, bbox in zip(oracle_ids, oracle_labels, oracle_bboxes):
        if label != human_label:
            continue
        best_entry: Optional[Tuple[int, List[float], np.ndarray]] = None
        best_iou = 0.0
        for det_id, det_bbox, heatmap in detection_entries:
            if det_id in assigned_det_ids:
                continue
            iou = bbox_iou(bbox, det_bbox)
            if iou > best_iou:
                best_iou = iou
                best_entry = (det_id, det_bbox, heatmap)
        if best_entry is not None and best_iou >= iou_thres:
            assigned_det_ids.add(best_entry[0])
            result[int(obj_id)] = best_entry[2].tolist()
        else:
            result[int(obj_id)] = zero_heatmap.tolist()

    return result


def load_reverse_object_map(path: Path) -> Dict[str, Dict[str, List]]:
    with path.open("r") as f:
        data = json.load(f)

    entries: List[Dict[str, Any]] = []

    if isinstance(data, dict):
        if isinstance(data.get("labels"), list):
            entries.extend(data["labels"])
        elif all(isinstance(v, (int, str)) for v in data.values()):
            for raw_id, target in data.items():
                entries.append({"source_name": raw_id, "target_name": target})
        else:
            entries.append(data)
    elif isinstance(data, list):
        entries.extend(data)
    else:
        raise ValueError(f"Unsupported object mapping format in {path}")

    reverse_store: Dict[str, Dict[str, Any]] = {}

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        
        target_name = (
            entry.get("target_name")
            or entry.get("target_label")
            or entry.get("target")
        )
        
        if not target_name:
            continue
            
        target_key = str(target_name)

        source_id = entry.get("source_id")
        try:
            source_id = int(source_id) if source_id is not None else None
        except (ValueError, TypeError):
            source_id = None

        source_name = (
            entry.get("source_name")
            or entry.get("source_label")
            or entry.get("source")
            or entry.get("notes")
            or entry.get("raw_name")
        )

        store = reverse_store.setdefault(target_key, {
            "source_ids": set(),
            "source_names": set(),
            "pairs_by_id": defaultdict(set),
            "pairs_no_id": set()
        })

        if source_id is not None:
            store["source_ids"].add(source_id)
            if source_name:
                store["source_names"].add(source_name)
                store["pairs_by_id"][source_id].add(source_name)
            else:
                if source_id not in store["pairs_by_id"]:
                    store["pairs_by_id"][source_id] = set()
        elif source_name:
            store["source_names"].add(source_name)
            store["pairs_no_id"].add(source_name)

    reverse_map: Dict[str, Dict[str, List]] = {}
    for key, val in reverse_store.items():
        source_pairs = []
        for sid, names in val["pairs_by_id"].items():
            source_pairs.append({
                "source_id": sid,
                "source_names": sorted(list(names))
            })
        
        if val["pairs_no_id"]:
             source_pairs.append({
                "source_id": None,
                "source_names": sorted(list(val["pairs_no_id"]))
            })

        reverse_map[key] = {
            "source_ids": sorted(list(val["source_ids"])),
            "source_names": sorted(list(val["source_names"])),
            "source_pairs": source_pairs,
            "name_only": sorted(list(val["pairs_no_id"]))
        }

    return reverse_map


def build_object_display_name(base_name: str, mapping_entry: Optional[Dict[str, List]]) -> str:
    if not mapping_entry:
        return base_name
    candidate_names: Set[str] = set()
    for collection_key in ("source_names", "name_only"):
        for val in mapping_entry.get(collection_key, []) or []:
            if val:
                candidate_names.add(str(val))
    for pair in mapping_entry.get("source_pairs", []) or []:
        for val in pair.get("source_names", []) or []:
            if val:
                candidate_names.add(str(val))
    candidate_names = {n for n in candidate_names if n and n.lower() != base_name.lower()}
    if not candidate_names:
        return base_name
    extras = ", ".join(sorted(candidate_names))
    return f"{base_name} [{extras}]"


@torch.no_grad()
def main(opt):
    source = Path(opt.source)
    if not source.exists():
        print(f"{source} does not exist, exit")
        return -1
    video_name = source.stem
    future = opt.future
    hoi_thres = opt.hoi_thres
    print_hois = opt.print
    video_output = opt.video
    human_label = opt.human_label
    oracle_annotations = None
    if opt.oracle_ann:
        oracle_path = Path(opt.oracle_ann)
        if oracle_path.exists():
            try:
                oracle_annotations = load_oracle_annotations(oracle_path, target_video=video_name)
                if oracle_annotations:
                    print(
                        f"Loaded {len(oracle_annotations)} oracle annotation frames from {oracle_path}"
                    )
                else:
                    print(
                        f"⚠️  Oracle annotations found in {oracle_path} but none matched video '{video_name}'."
                    )
                    oracle_annotations = None
            except Exception as exc:
                print(f"⚠️  Failed to load oracle annotations from {oracle_path}: {exc}")
                oracle_annotations = None
        else:
            print(f"⚠️  Oracle annotation file {oracle_path} not found, continuing without oracle mode")
    use_oracle = oracle_annotations is not None
    oracle_missing_warned = False
    oracle_iou_thres = opt.oracle_iou_thres
    if use_oracle:
        print("✅ Oracle mode enabled: STTran will use provided annotations for detections.")

    cfg_root = Path(opt.cfg_root)
    eval_cfg_path = Path(opt.eval_cfg)
    if not eval_cfg_path.is_absolute():
        eval_cfg_path = (cfg_root / eval_cfg_path).resolve()
    weight_root = Path(opt.weights_root)
    yolo_weight_path = weight_root / "yolov5"
    deepsort_weight_path = weight_root / "deep_sort"
    gaze_following_weight_path = weight_root / "detecting_attended" / "model_videoatttarget.pt"
    backbone_model_path = Path(opt.backbone)
    sttran_word_vector_dir = Path(opt.semantic)
    model_path = Path(opt.model)

    out = opt.out
    output_folder = Path(out) / video_name
    output_folder.mkdir(parents=True, exist_ok=True)
    trace_file = output_folder / f"{video_name}_trace.json"
    gaze_file = output_folder / f"{video_name}_gaze.json"
    hoi_json_file = output_folder / f"{video_name}_hoi.json"
    result_file = output_folder / f"{video_name}_result.json"
    result_video_file = output_folder / f"{video_name}_result.mp4"

    imgsz = opt.imgsz
    gaze = "cross"
    global_token = True
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg = get_config(str(eval_cfg_path))
    sampling_mode = cfg["sampling_mode"]
    dim_gaze_heatmap = cfg["dim_gaze_heatmap"]
    dim_transformer_ffn = cfg["dim_transformer_ffn"]
    sttran_enc_layer_num = cfg["sttran_enc_layer_num"]
    sttran_dec_layer_num = cfg["sttran_dec_layer_num"]
    sttran_sliding_window = cfg["sttran_sliding_window"]
    separate_head = cfg["separate_head"]
    loss_type = cfg["loss_type"]
    mlp_projection = cfg["mlp_projection"]
    sinusoidal_encoding = cfg["sinusoidal_encoding"]

    print(f"======================================")
    object_tracking_module = ObjectTracking(
        yolo_weights_path=str(yolo_weight_path / "mvrhoi_yolov5l.pt"),
        deep_sort_model_dir=str(deepsort_weight_path),
        config_path=str(cfg_root / "object_tracking.yaml"),
        device=device,
    )
    yolov5_stride = object_tracking_module.yolov5_stride
    print(f"======================================")
    head_tracking_module = HeadTracking(
        crowd_human_weight_path=str(yolo_weight_path / "crowdhuman_yolov5m.pt"),
        deep_sort_model_dir=str(deepsort_weight_path),
        config_path=str(cfg_root / "object_tracking.yaml"),
        device=device,
    )
    print(f"======================================")
    gaze_following_module = GazeFollowing(
        weight_path=str(gaze_following_weight_path),
        config_path=str(cfg_root / "gaze_following.yaml"),
        device=device,
    )
    matching_iou_thres = 0.7
    matching_method = "hungarian"

    print(f"======================================")
    if backbone_model_path.is_dir():
        feature_backbone = FeatureExtractionResNet101(str(backbone_model_path), download=True, finetune=False).to(device)
        print(f"Feature backbone directory provided; downloading default ResNet weights into {backbone_model_path}.")
    else:
        if opt.backbone_finetuned:
            feature_backbone = FeatureExtractionResNet101(str(backbone_model_path), download=False, finetune=True).to(device)
            print(f"Feature backbone loaded from fine-tuned ResNet state dict {backbone_model_path}.")
        else:
            feature_backbone = FeatureExtractionResNet101(str(backbone_model_path), download=False, finetune=False).to(device)
            print(f"Feature backbone loaded from serialized FeatureExtraction checkpoint {backbone_model_path}.")
    feature_backbone.requires_grad_(False)
    feature_backbone.eval()
    print(f"Feature backbone loaded from {backbone_model_path}")

    with Path("vidhoi_related/obj_categories.json").open("r") as f:
        object_classes = json.load(f)
    with Path("vidhoi_related/pred_categories.json").open("r") as f:
        interaction_classes = json.load(f)
    with Path("vidhoi_related/pred_split_categories.json").open("r") as f:
        temp_dict = json.load(f)
        spatial_class_idxes = temp_dict["spatial"]
        action_class_idxes = temp_dict["action"]
    num_object_classes = len(object_classes)
    num_interaction_classes = len(interaction_classes)
    num_spatial_classes = len(spatial_class_idxes)
    num_action_classes = len(action_class_idxes)
    num_interaction_classes_loss = num_interaction_classes

    reverse_object_map: Dict[str, Dict[str, List]] = {}
    if opt.object_map:
        object_map_path = Path(opt.object_map)
        if not object_map_path.exists():
            print(
                f"⚠️  Object mapping file {object_map_path} not found; continuing without reverse name mapping."
            )
        else:
            try:
                reverse_object_map = load_reverse_object_map(object_map_path)
                if reverse_object_map:
                    print(
                        f"Loaded reverse object mapping for {len(reverse_object_map)} VidHOI classes from {object_map_path}"
                    )
                else:
                    print(
                        f"⚠️  Object mapping {object_map_path} did not yield any entries; continuing without reverse mapping."
                    )
            except Exception as exc:
                print(f"⚠️  Failed to load object mapping {object_map_path}: {exc}")
                reverse_object_map = {}

    def base_object_name(label_idx: int) -> str:
        if 0 <= label_idx < len(object_classes):
            return object_classes[label_idx]
        return str(label_idx)

    def display_object_name(label_idx: int) -> str:
        name = base_object_name(label_idx)
        return build_object_display_name(name, reverse_object_map.get(name))


    print(f"Transformer configs: {cfg}")
    loss_type_dict = {"spatial_head": "bce", "action_head": "bce"}
    separate_head_num = [num_spatial_classes, -1]
    separate_head_name = ["spatial_head", "action_head"]
    class_idxes_dict = {"spatial_head": spatial_class_idxes, "action_head": action_class_idxes}
    loss_gt_dict = {"spatial_head": "spatial_gt", "action_head": "action_gt"}
    sttran_gaze_model = STTranGazeCrossAttention(
        num_interaction_classes=num_interaction_classes_loss,
        obj_class_names=object_classes,
        spatial_layer_num=sttran_enc_layer_num,
        cross_layer_num=1,
        temporal_layer_num=sttran_dec_layer_num - 1,
        dim_transformer_ffn=dim_transformer_ffn,
        d_gaze=512,
        cross_sa=True,
        cross_ffn=False,
        global_token=global_token,
        mlp_projection=mlp_projection,
        sinusoidal_encoding=sinusoidal_encoding,
        dropout=0,
        word_vector_dir=sttran_word_vector_dir,
        sliding_window=sttran_sliding_window,
        separate_head=separate_head_num,
        separate_head_name=separate_head_name,
    )
    sttran_gaze_model = sttran_gaze_model.to(device)
    incompatibles = sttran_gaze_model.load_state_dict(torch.load(model_path, map_location=device))
    sttran_gaze_model.eval()
    print(f"STTranGaze loaded. Incompatible keys {incompatibles}")

    dataset = VideoDatasetLoader(
        source, transform=YOLOv5Transform(imgsz, yolov5_stride), additional_transform=STTranTransform(img_size=imgsz)
    )
    frame_num = dataset.frame_num
    print(f"Video {source} loaded with {frame_num} frames")
    object_tracking_module.clear()
    frame, frame0, _, _, _ = next(iter(dataset))
    object_tracking_module.warmup(frame.to(device), frame0)
    print(f"Object tracker warmup finished.")

    fourcc = "mp4v"
    fps, w, h = round(dataset.fps), frame0.shape[1], frame0.shape[0]
    if video_output:
        video_writer = cv2.VideoWriter(str(result_video_file), cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))

    detection_dict = defaultdict(list)
    gaze_list = []
    hoi_json_list = []
    result_list = []
    frames_queue = deque(maxlen=sttran_sliding_window)
    frame_ids_queue = deque(maxlen=sttran_sliding_window)
    hx_memory = {}
    print(f"============ Inference Start... ============")
    t = tqdm(enumerate(iter(dataset)), total=frame_num)
    for idx, batch in t:
        frame, frame0, _, _, meta_info = batch
        meta_info["original_shape"] = frame0.shape
        bboxes, ids, labels, names, confs, _ = object_tracking_module.track_one(frame.to(device), frame0, draw=False)
        if len(ids) > 0:
            frame_annotated = draw_bboxes(frame0.copy(), bboxes, ids, labels, names, confs)
        else:
            frame_annotated = frame0.copy()

        bboxes_array = np.asarray(bboxes, dtype=np.float32) if len(bboxes) > 0 else np.zeros((0, 4), dtype=np.float32)
        labels_array = np.asarray(labels, dtype=np.int32) if len(labels) > 0 else np.zeros((0,), dtype=np.int32)
        ids_array = np.asarray(ids, dtype=np.int64) if len(ids) > 0 else np.zeros((0,), dtype=np.int64)

        human_idxes = labels_array == human_label
        human_bboxes = bboxes_array[human_idxes] if human_idxes.any() else np.zeros((0, 4), dtype=np.float32)
        human_ids = ids_array[human_idxes] if human_idxes.any() else np.zeros((0,), dtype=np.int64)
        h_bboxes, _, _, _, h_confs, _ = head_tracking_module.track_one(frame.to(device), frame0, draw=False)
        head_bbox_dict = assign_human_head_frame(
            h_bboxes, h_confs, human_bboxes, human_ids, matching_iou_thres, matching_method
        )
        frame_gaze_dict = {}
        for human_id, head_bbox in head_bbox_dict.items():
            if len(head_bbox) == 0:
                frame_gaze_dict[int(human_id)] = []
                continue
            hidden_state = hx_memory.get(human_id)
            heatmap, _, hx, _, _, frame_annotated = gaze_following_module.detect_one(
                frame0, head_bbox, hidden_state, draw=True, frame_to_draw=frame_annotated, id=human_id, arrow=True
            )
            hx_memory[human_id] = (hx[0].detach(), hx[1].detach())
            frame_gaze_dict[int(human_id)] = heatmap.tolist()

        oracle_frame = lookup_oracle_frame(oracle_annotations, meta_info["frame_count"]) if use_oracle else None
        if use_oracle:
            detection_gaze = {int(k): v for k, v in frame_gaze_dict.items()}
            frame_gaze_dict = remap_gaze_to_oracle(
                detection_gaze,
                human_ids,
                human_bboxes,
                oracle_frame,
                human_label,
                oracle_iou_thres,
            )

        if idx % fps == 0:
            detection_dict["frame_ids"].append(meta_info["frame_count"])
            if use_oracle and oracle_frame is not None and oracle_frame.get("ids"):
                detection_dict["bboxes"].append(
                    [[float(coord) for coord in bbox] for bbox in oracle_frame["bboxes"]]
                )
                detection_dict["ids"].append([int(obj_id) for obj_id in oracle_frame["ids"]])
                detection_dict["labels"].append([int(label) for label in oracle_frame["labels"]])
                detection_dict["confidences"].append([1.0] * len(oracle_frame["ids"]))
            else:
                if use_oracle and (oracle_frame is None or not oracle_frame.get("ids")) and not oracle_missing_warned:
                    print(
                        "⚠️  Oracle annotations missing for some frames; falling back to detector outputs."
                    )
                    oracle_missing_warned = True
                detection_dict["bboxes"].append(
                    [[float(coord) for coord in bbox] for bbox in bboxes]
                )
                detection_dict["ids"].append([int(obj_id) for obj_id in ids])
                detection_dict["labels"].append([int(label) for label in labels])
                detection_dict["confidences"].append([float(score) for score in confs])
            gaze_list.append(frame_gaze_dict)
        if video_output:
            video_writer.write(frame_annotated)

        if idx % fps == 0:
            frames_queue.append(meta_info["additional"])
            frame_ids_queue.append(meta_info["frame_count"])
            sttran_frames = torch.cat(list(frames_queue)).to(device)
            det_bboxes = detection_dict["bboxes"][-sttran_sliding_window:]
            det_ids = detection_dict["ids"][-sttran_sliding_window:]
            det_labels = detection_dict["labels"][-sttran_sliding_window:]
            det_confidences = detection_dict["confidences"][-sttran_sliding_window:]
            bboxes, ids, pred_labels, confidences = convert_annotation_frame_to_video(
                det_bboxes, det_ids, det_labels, det_confidences
            )
            if len(bboxes) == 0:
                pair_idxes = []
                im_idxes = []
            else:
                pair_idxes, im_idxes = bbox_pair_generation(bboxes, pred_labels, human_label)
            detected = {
                "bboxes": bboxes,
                "pred_labels": pred_labels,
                "ids": ids,
                "confidences": confidences,
                "pair_idxes": pair_idxes,
                "im_idxes": im_idxes,
            }
            entry = fill_sttran_entry_inference(
                sttran_frames,
                detected,
                gaze_list[-sttran_sliding_window:],
                feature_backbone,
                meta_info,
                loss_type_dict,
                class_idxes_dict,
                loss_gt_dict,
                device,
                annotations=None,
                human_label=human_label,
            )

            windows = construct_sliding_window(entry, sampling_mode, sttran_sliding_window, 0, None, gt=False)
            entry, windows, windows_out, out_im_idxes, _ = generate_sliding_window_mask(entry, windows, None, "pair")

            if len(windows) > 0:
                entry["pair_idxes"] = entry["pair_idxes"].to(device)
                for i in range(len(entry["full_heatmaps"])):
                    entry["full_heatmaps"][i] = entry["full_heatmaps"][i].to(device)
                entry["pred_labels"] = entry["pred_labels"].to(device)
                entry["windows"] = windows.to(device)
                entry["windows_out"] = windows_out.to(device)

                entry = sttran_gaze_model(entry)
                for head_name in loss_type_dict.keys():
                    if loss_type_dict[head_name] == "ce":
                        entry[head_name] = torch.softmax(entry[head_name], dim=-1)
                    else:
                        entry[head_name] = torch.sigmoid(entry[head_name])
                len_preds = len(entry[list(loss_type_dict.keys())[0]])
                interaction_distribution = concat_separated_head(
                    entry, len_preds, loss_type_dict, class_idxes_dict, device, True
                )

            frame_ids = [int(fid) for fid in list(frame_ids_queue)]
            out_im_idx = len(frame_ids) - 1
            window_start_frame = frame_ids[0]
            window_end_frame = frame_ids[-1]
            window_anno = {
                "video_name": video_name,
                "frame_id": frame_ids[out_im_idx],
                "window_start_frame": window_start_frame,
                "window_end_frame": window_end_frame,
                "window_frame_count": len(frame_ids),
                "window_frame_ids": frame_ids,
                "sampling_iteration": int(idx),
            }
            if sampling_mode == "anticipation":
                if idx + future >= frame_num:
                    window_anno["future_frame_id"] = ""
                else:
                    window_anno["future_frame_id"] = f"{idx + fps * future:06d}"

            window_prediction = {
                "bboxes": [],
                "pred_labels": [],
                "confidences": [],
                "pair_idxes": [],
                "interaction_distribution": [],
            }
            if len(entry["bboxes"]) != 0:
                det_out_idxes = entry["bboxes"][:, 0] == out_im_idx
                if det_out_idxes.any():
                    det_idx_offset = det_out_idxes.nonzero(as_tuple=True)[0][0]
                    window_prediction["bboxes"] = entry["bboxes"][det_out_idxes, 1:].numpy().tolist()
                    window_prediction["pred_labels"] = entry["pred_labels"][det_out_idxes].cpu().numpy().tolist()
                    window_prediction["confidences"] = entry["confidences"][det_out_idxes].numpy().tolist()
                    window_prediction["ids"] = torch.LongTensor(entry["ids"])[det_out_idxes].numpy().tolist()

                    pair_out_idxes = entry["im_idxes"] == out_im_idx
                    if pair_out_idxes.any():
                        pair_idxes = entry["pair_idxes"][pair_out_idxes] - det_idx_offset
                        window_prediction["pair_idxes"] = pair_idxes.cpu().numpy().tolist()
                        window_prediction["interaction_distribution"] = interaction_distribution.cpu().numpy().tolist()

            window_result = {**window_anno, **window_prediction}
            result_list.append(window_result)
            triplets_scores = generate_triplets_scores(
                window_result["pair_idxes"],
                [1.0] * len(window_result.get("confidences", [])),
                window_result["interaction_distribution"],
                multiply=True,
                top_k=100,
                thres=hoi_thres,
            )
            s_hois = "-------------------------------\n"
            s_hois += f"Frame {idx}/{frame_num}:\n"
            for score, idx_pair, interaction_pred in triplets_scores:
                subj_idx = window_result["pair_idxes"][idx_pair][0]
                subj_cls = int(window_result["pred_labels"][subj_idx])
                subj_name = base_object_name(subj_cls)
                subj_display_name = display_object_name(subj_cls)
                subj_mapping_entry = reverse_object_map.get(subj_name) or {}
                subj_source_ids = [int(sid) for sid in subj_mapping_entry.get("source_ids", [])]
                subj_source_names = list(subj_mapping_entry.get("source_names", []))
                subj_source_pairs = []
                for pair in subj_mapping_entry.get("source_pairs", []) or []:
                    pair_id = pair.get("source_id")
                    try:
                        pair_id = int(pair_id) if pair_id is not None else None
                    except (TypeError, ValueError):
                        pass
                    subj_source_pairs.append(
                        {
                            "source_id": pair_id,
                            "source_names": list(pair.get("source_names", [])),
                        }
                    )
                subj_name_only = list(subj_mapping_entry.get("name_only", []))
                subj_id = window_result["ids"][subj_idx]
                subj_bbox = window_result["bboxes"][subj_idx]
                subj_weight = window_result["confidences"][subj_idx]

                obj_idx = window_result["pair_idxes"][idx_pair][1]
                obj_cls = int(window_result["pred_labels"][obj_idx])
                obj_name = base_object_name(obj_cls)
                obj_display_name = display_object_name(obj_cls)
                obj_mapping_entry = reverse_object_map.get(obj_name) or {}
                obj_source_ids = [int(sid) for sid in obj_mapping_entry.get("source_ids", [])]
                obj_source_names = list(obj_mapping_entry.get("source_names", []))
                obj_source_pairs = []
                for pair in obj_mapping_entry.get("source_pairs", []) or []:
                    pair_id = pair.get("source_id")
                    try:
                        pair_id = int(pair_id) if pair_id is not None else None
                    except (TypeError, ValueError):
                        pass
                    obj_source_pairs.append(
                        {
                            "source_id": pair_id,
                            "source_names": list(pair.get("source_names", [])),
                        }
                    )
                obj_name_only = list(obj_mapping_entry.get("name_only", []))
                obj_id = window_result["ids"][obj_idx]
                obj_bbox = window_result["bboxes"][obj_idx]
                obj_weight = window_result["confidences"][obj_idx]

                interaction_name = interaction_classes[interaction_pred]

                s_hois_dict = {
                    "frame": {
                        "index": int(window_result["frame_id"]),
                        "total": frame_num,
                        "window_start": int(window_result["window_start_frame"]),
                        "window_end": int(window_result["window_end_frame"]),
                        "window_size": int(window_result["window_frame_count"]),
                        "window_frames": window_result["window_frame_ids"],
                        "sampling_iteration": int(window_result.get("sampling_iteration", idx)),
                    },
                    "subject": {
                        "name": subj_name,
                        "display_name": subj_display_name,
                        "id": subj_id,
                        "bbox": subj_bbox,
                        "weight": subj_weight,
                        "mapped_source_ids": subj_source_ids,
                        "mapped_source_names": subj_source_names,
                        "mapped_source_pairs": subj_source_pairs,
                        "mapped_name_only": subj_name_only,
                    },
                    "interaction": {
                        "name": interaction_name,
                        "pair_indices": [int(subj_idx), int(obj_idx)]
                    },
                    "object": {
                        "name": obj_name,
                        "display_name": obj_display_name,
                        "id": obj_id,
                        "bbox": obj_bbox,
                        "weight": obj_weight,
                        "mapped_source_ids": obj_source_ids,
                        "mapped_source_names": obj_source_names,
                        "mapped_source_pairs": obj_source_pairs,
                        "mapped_name_only": obj_name_only,
                    },
                    "score": score
                }
                hoi_json_list.append(s_hois_dict)
                s_hois += json.dumps(s_hois_dict, ensure_ascii=False) + "\n"

            if print_hois:
                print(s_hois)

    if video_output:
        video_writer.release()
    with trace_file.open("w") as f:
        json.dump(detection_dict, f)
    with gaze_file.open("w") as f:
        json.dump(gaze_list, f)
    with result_file.open("w") as f:
        json.dump(result_list, f)
    with hoi_json_file.open("w") as f:
        json.dump(hoi_json_list, f)

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
    sys.exit(0)
