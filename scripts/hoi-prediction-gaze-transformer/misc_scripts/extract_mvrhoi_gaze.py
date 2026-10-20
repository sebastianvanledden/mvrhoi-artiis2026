#!/usr/bin/env python3
"""
Extract gaze features for MVRHOI dataset

Adapted from vidhoi_gt_gaze_extraction.py to work with MVRHOI annotation format
"""

import sys
from pathlib import Path

sys.path.insert(0, "modules/object_tracking/yolov5")
sys.path.insert(0, ".")

import numpy as np
import shelve
import json
import cv2
from tqdm import tqdm
import torch

from modules.object_tracking import HeadDetection
from modules.gaze_following import GazeFollowing
from common.transforms import YOLOv5Transform

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def calculate_iou(bbox1, bbox2):
    """Calculate Intersection over Union (IoU) of two bounding boxes."""
    # Extract coordinates
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Calculate intersection area
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0.0


def initialize_modules():
    """Initialize head detection and gaze following modules."""
    # Load Head Tracking and Gaze Following modules
    head_detection_module = HeadDetection(
        crowd_human_weight_path="weights/yolov5/crowdhuman_yolov5m.pt",
        config_path="configs/object_tracking.yaml",
        device=device,
    )
    gaze_following_module = GazeFollowing(
        weight_path="weights/detecting_attended/model_videoatttarget.pt",
        config_path="configs/gaze_following.yaml",
        device=device,
    )

    img_size = 640
    yolov5_stride = head_detection_module.yolov5_stride
    # NOTE adjust this tolerance and method
    head_matching_iou_thres = 0.5
    return (
        head_detection_module,
        gaze_following_module,
        img_size,
        yolov5_stride,
        head_matching_iou_thres,
    )


def load_mvrhoi_annotations(mvrhoi_root, annotation_filename):
    """Load MVRHOI annotations and organize them per video in frame order."""

    annotations_dir = Path(mvrhoi_root) / "VidHOI_annotation"
    if not annotations_dir.exists():
        annotations_dir = Path(mvrhoi_root) / "Vidhoi_annotation"

    annotations_file = annotations_dir / annotation_filename
    if not annotations_file.exists():
        raise FileNotFoundError(f"Annotation file not found: {annotations_file}")

    with open(annotations_file, "r") as f:
        annotations = json.load(f)

    if len(annotations) == 0:
        raise ValueError(f"No annotations found in {annotations_file}")

    videos_data = {}
    current_video = None
    frame_ids = []
    frame_person_bboxes = []
    last_frame_id = None

    for ann in annotations:
        video_key = f"{ann['video_folder']}/{ann['video_id']}"
        frame_id = ann["frame_id"]

        if video_key != current_video:
            if current_video is not None:
                videos_data[current_video] = {
                    "frame_ids": frame_ids,
                    "person_bboxes": frame_person_bboxes,
                }
            current_video = video_key
            frame_ids = []
            frame_person_bboxes = []
            last_frame_id = None

        if frame_id != last_frame_id:
            frame_ids.append(frame_id)
            frame_person_bboxes.append({})
            last_frame_id = frame_id

        frame_person_bboxes[-1][ann["person_id"]] = [
            ann["person_box"]["xmin"],
            ann["person_box"]["ymin"],
            ann["person_box"]["xmax"],
            ann["person_box"]["ymax"],
        ]

    if current_video is not None:
        videos_data[current_video] = {
            "frame_ids": frame_ids,
            "person_bboxes": frame_person_bboxes,
        }

    return videos_data


def process_mvrhoi_video(video_name, video_data, images_root, head_detection_module, gaze_following_module, img_size, yolov5_stride, head_matching_iou_thres):
    """Process a single MVRHOI video and extract gaze features in VidHOI-compatible format."""

    print(f"Processing video: {video_name}")

    frame_ids = video_data["frame_ids"]
    person_bboxes_per_frame = video_data["person_bboxes"]

    yolov5_transform = YOLOv5Transform(img_size, yolov5_stride)
    video_head_list = []
    video_gaze_list = []
    video_inout_list = []
    hx_memory = {}

    video_folder, video_id = video_name.split("/")

    for frame_idx, frame_id in enumerate(tqdm(frame_ids, desc=f"Processing {video_name}")):
        img_path = images_root / video_folder / video_id / f"{video_id}_{frame_id}.jpg"
        person_bboxes = person_bboxes_per_frame[frame_idx]

        if not img_path.exists():
            print(f"Warning: Image not found: {img_path}")
            empty_heads = {pid: [] for pid in person_bboxes}
            video_head_list.append(empty_heads)
            video_gaze_list.append({pid: [] for pid in person_bboxes})
            video_inout_list.append({pid: [] for pid in person_bboxes})
            continue

        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"Warning: Could not load image: {img_path}")
            empty_heads = {pid: [] for pid in person_bboxes}
            video_head_list.append(empty_heads)
            video_gaze_list.append({pid: [] for pid in person_bboxes})
            video_inout_list.append({pid: [] for pid in person_bboxes})
            continue

        frame_transformed = yolov5_transform(frame).to(device)
        head_bboxes, _, _, _, _ = head_detection_module.detect_one(
            frame_transformed, frame, draw=False, include_person=False
        )

        frame_head_bboxes = {pid: [] for pid in person_bboxes}

        if len(head_bboxes) > 0 and len(person_bboxes) > 0:
            for person_id, person_bbox in person_bboxes.items():
                best_head = None
                best_iou = 0.0
                for head_bbox in head_bboxes:
                    iou = calculate_iou(person_bbox, head_bbox)
                    if iou > best_iou and iou >= head_matching_iou_thres:
                        best_iou = iou
                        best_head = head_bbox
                if best_head is not None:
                    if hasattr(best_head, "tolist"):
                        head_coords = best_head.tolist()
                    else:
                        head_coords = list(best_head)
                    frame_head_bboxes[person_id] = [float(coord) for coord in head_coords]

        frame_gaze_dict = {}
        frame_inout_dict = {}

        for person_id, head_bbox in frame_head_bboxes.items():
            if not head_bbox:
                frame_gaze_dict[person_id] = np.zeros((64, 64), dtype=np.float32)
                frame_inout_dict[person_id] = 0.0
                hx_memory.pop(person_id, None)
                continue

            hidden_state = hx_memory.get(person_id)
            head_bbox_np = np.asarray(head_bbox, dtype=np.float32)

            with torch.no_grad():
                heatmap, inout, hx, _, _, _ = gaze_following_module.detect_one(
                    frame,
                    head_bbox_np,
                    hidden_state,
                    draw=False,
                )

            hx_memory[person_id] = (hx[0].detach(), hx[1].detach()) if hx is not None else None

            if heatmap is None:
                heatmap_array = np.zeros((64, 64), dtype=np.float32)
            else:
                heatmap_array = heatmap.detach().cpu().numpy() if isinstance(heatmap, torch.Tensor) else np.asarray(heatmap)

            if inout is None:
                inout_modulated = 0.0
            else:
                inout_value = inout.detach().cpu().numpy() if isinstance(inout, torch.Tensor) else np.asarray(inout)
                inout_modulated = float(np.squeeze(1 / (1 + np.exp(-inout_value))))

            frame_gaze_dict[person_id] = heatmap_array
            frame_inout_dict[person_id] = inout_modulated

        video_head_list.append(frame_head_bboxes)
        video_gaze_list.append(frame_gaze_dict)
        video_inout_list.append(frame_inout_dict)

    return video_head_list, video_gaze_list, video_inout_list


def extract_mvrhoi_gaze(mvrhoi_root="D:/MVRHOI", output_dir=None, split="train"):
    """Main function to extract gaze features for MVRHOI dataset"""
    
    split = split.lower()
    if split not in {"train", "val"}:
        raise ValueError(f"Unsupported split '{split}'. Expected 'train' or 'val'.")

    mvrhoi_root = Path(mvrhoi_root)
    if output_dir is None:
        output_dir = mvrhoi_root / "VidHOI_gaze"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=== MVRHOI Gaze Extraction ===")
    print(f"Split: {split}")
    print(f"MVRHOI root: {mvrhoi_root}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {device}")
    
    # Initialize modules
    print("Initializing head detection and gaze following modules...")
    (
        head_detection_module,
        gaze_following_module,
        img_size,
        yolov5_stride,
        head_matching_iou_thres,
    ) = initialize_modules()
    
    # Load MVRHOI annotations
    annotation_filename = f"{split}_frame_annots.json"
    print(f"Loading MVRHOI annotations from {annotation_filename}...")
    videos_data = load_mvrhoi_annotations(mvrhoi_root, annotation_filename)
    print(f"Found {len(videos_data)} videos")
    
    # Prepare output files
    prefix = f"{split}_frame"
    output_head_filename = output_dir / f"{prefix}_heads_gt_bbox"
    output_gaze_filename = output_dir / f"{prefix}_gazes_gt_bbox"
    output_inout_filename = output_dir / f"{prefix}_inout_gt_bbox"
    
    # Open shelve databases
    output_head_dict = shelve.open(str(output_head_filename))
    output_gaze_dict = shelve.open(str(output_gaze_filename))
    output_inout_dict = shelve.open(str(output_inout_filename))
    
    try:
        images_root = mvrhoi_root / "images"
        
        # Process each video
        for video_name, video_data in videos_data.items():
            try:
                video_head_list, video_gaze_list, video_inout_list = process_mvrhoi_video(
                    video_name,
                    video_data,
                    images_root,
                    head_detection_module,
                    gaze_following_module,
                    img_size,
                    yolov5_stride,
                    head_matching_iou_thres,
                )
                
                # Store in shelve databases
                output_head_dict[video_name] = video_head_list
                output_gaze_dict[video_name] = video_gaze_list
                output_inout_dict[video_name] = video_inout_list
                
                # Sync to disk
                output_head_dict.sync()
                output_gaze_dict.sync()
                output_inout_dict.sync()
                
                print(f"✓ Completed {video_name}: {len(video_head_list)} frames")
                
            except Exception as e:
                print(f"✗ Error processing {video_name}: {e}")
                continue
    
    finally:
        # Close databases
        output_head_dict.close()
        output_gaze_dict.close()
        output_inout_dict.close()
    
    print(f"\n=== Extraction Complete ===")
    print(f"Head bboxes saved to: {output_head_filename}")
    print(f"Gaze heatmaps saved to: {output_gaze_filename}")
    print(f"Gaze inout saved to: {output_inout_filename}")
    
    return output_dir


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract gaze features for MVRHOI dataset')
    parser.add_argument('--mvrhoi_root', type=str, default='D:/MVRHOI', help='MVRHOI dataset root directory')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for gaze features')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val'], help='Dataset split to process')
    
    args = parser.parse_args()
    
    # Set global device
    global device
    device = args.device if torch.cuda.is_available() else "cpu"
    
    # Run extraction
    extract_mvrhoi_gaze(args.mvrhoi_root, args.output_dir, args.split)


if __name__ == "__main__":
    main()