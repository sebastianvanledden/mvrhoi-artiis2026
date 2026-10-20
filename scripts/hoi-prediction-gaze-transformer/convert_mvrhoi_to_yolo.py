#!/usr/bin/env python3
"""Convert MVRHOI annotations to YOLOv5 format."""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2


def clip_bbox(x: float, y: float, w: float, h: float, img_width: int, img_height: int) -> Tuple[float, float, float, float]:
    """
    Clip bounding box to image boundaries.
    
    Args:
        x, y, w, h: Original bbox in CVAT format (top-left x, y, width, height)
        img_width, img_height: Image dimensions
    
    Returns:
        Clipped bbox (x, y, w, h)
    """
    # Clip to image boundaries
    x_min = max(0.0, x)
    y_min = max(0.0, y)
    x_max = min(float(img_width), x + w)
    y_max = min(float(img_height), y + h)
    
    # Ensure valid box (at least 1 pixel)
    if x_max <= x_min:
        x_max = min(float(img_width), x_min + 1.0)
    if y_max <= y_min:
        y_max = min(float(img_height), y_min + 1.0)
    
    # Convert back to x, y, w, h
    clipped_x = x_min
    clipped_y = y_min
    clipped_w = x_max - x_min
    clipped_h = y_max - y_min
    
    return clipped_x, clipped_y, clipped_w, clipped_h


def is_duplicate_bbox(class_id: int, x_center: float, y_center: float, w_norm: float, h_norm: float, 
                       seen_boxes: Set[Tuple[int, float, float, float, float]], 
                       tolerance: float = 1e-6) -> bool:
    """
    Check if this bounding box already exists in the current frame.
    
    Args:
        class_id: Object class ID
        x_center, y_center, w_norm, h_norm: Normalized YOLO bbox coordinates
        seen_boxes: Set of already processed boxes in current frame
        tolerance: Floating point comparison tolerance
    
    Returns:
        True if duplicate, False otherwise
    """
    # Round to avoid floating point precision issues
    bbox_key = (
        class_id,
        round(x_center, 6),
        round(y_center, 6),
        round(w_norm, 6),
        round(h_norm, 6)
    )
    
    if bbox_key in seen_boxes:
        return True
    
    seen_boxes.add(bbox_key)
    return False


def load_label_map(map_path: Path) -> Dict[int, int]:
    """Load a mapping from raw object ids to target ids."""

    with map_path.open("r") as f:
        data = json.load(f)

    mapping: Dict[int, int] = {}

    def _coerce_pair(key_obj, val_obj) -> Optional[Tuple[int, int]]:
        try:
            key_int = int(key_obj)
            val_int = int(val_obj)
        except (TypeError, ValueError):
            return None
        return key_int, val_int

    if isinstance(data, dict):
        if all(isinstance(v, (int, str)) or v is None for v in data.values()):
            for raw_id, mapped_id in data.items():
                pair = _coerce_pair(raw_id, mapped_id)
                if pair is not None:
                    mapping[pair[0]] = pair[1]
        if not mapping and "labels" in data and isinstance(data["labels"], list):
            for entry in data["labels"]:
                if not isinstance(entry, dict):
                    continue
                pair = _coerce_pair(entry.get("source_id"), entry.get("target_id"))
                if pair is not None:
                    mapping[pair[0]] = pair[1]
        if not mapping:
            for key in ("mapping", "object_map", "vidor_to_vidhoi", "vidor_to_config_mapping"):
                nested = data.get(key)
                if isinstance(nested, dict):
                    for raw_id, mapped in nested.items():
                        if isinstance(mapped, dict):
                            mapped = mapped.get("config_id", mapped.get("suggested_config_id"))
                        pair = _coerce_pair(raw_id, mapped)
                        if pair is not None:
                            mapping[pair[0]] = pair[1]

    if not mapping:
        raise ValueError(f"No usable entries found in object mapping file {map_path}")

    return mapping


def load_mvrhoi_annotations(anno_dir: Path) -> List[Dict]:
    """Load all MVRHOI annotation files."""
    anno_files = list(anno_dir.glob("*_annotations_n.json"))
    print(f"Found {len(anno_files)} annotation files")
    
    all_videos = []
    for anno_file in anno_files:
        with open(anno_file) as f:
            data = json.load(f)
            all_videos.append({
                'file': anno_file.name,
                'data': data
            })
    
    return all_videos


def extract_frames_and_labels(
    video_data: Dict,
    video_dir: Path,
    output_images: Path,
    output_labels: Path,
    video_filename: str,
    object_label_map: Dict[int, int],
    object_default_label: int,
) -> Tuple[int, int, int, int]:
    """
    Extract frames at keyframe indices and create YOLO labels.
    
    Returns:
        Tuple of (frames_extracted, boxes_clipped, duplicates_removed)
    """
    annotation = video_data['data']['annotation']
    video_info = annotation['video']
    object_map = annotation['objectAnnotationListMap']
    
    # Find corresponding video file
    video_name = video_filename.replace('_annotations_n.json', '')
    video_paths = list(video_dir.glob(f"{video_name}*.mp4"))
    
    if not video_paths:
        print(f"⚠️  Video not found for {video_name}")
        return 0
    
    video_path = video_paths[0]
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"⚠️  Cannot open video {video_path}")
        return 0
    
    img_width = int(video_info['width'])
    img_height = int(video_info['height'])
    fps = video_info['fps']
    
    frames_extracted = 0
    boxes_clipped = 0
    duplicates_removed = 0
    unmapped_objects = 0
    
    # Process each annotated keyframe
    for frame_id_str, objects in object_map.items():
        frame_idx = int(frame_id_str)
        
        # Skip if no objects
        if not objects:
            continue
        
        # Seek to frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret or frame is None:
            print(f"⚠️  Failed to read frame {frame_idx} from {video_name}")
            continue
        
        # Save frame
        frame_name = f"{video_name}_f{frame_idx:06d}"
        img_path = output_images / f"{frame_name}.jpg"
        success = cv2.imwrite(str(img_path), frame)
        
        if not success:
            print(f"⚠️  Failed to write image: {img_path}")
            continue
        
        # Verify image was written correctly
        verify_img = cv2.imread(str(img_path))
        if verify_img is None:
            print(f"⚠️  Corrupted image: {img_path}, removing...")
            img_path.unlink(missing_ok=True)
            continue
        
        # Create YOLO label file
        label_path = output_labels / f"{frame_name}.txt"
        seen_boxes: Set[Tuple[int, float, float, float, float]] = set()

        with open(label_path, "w") as f:
            for obj in objects:
                class_id = int(obj["labelId"])
                x = obj["x"]
                y = obj["y"]
                w = obj["width"]
                h = obj["height"]

                # Clip bounding box to image boundaries
                x_clipped, y_clipped, w_clipped, h_clipped = clip_bbox(x, y, w, h, img_width, img_height)

                # Track if clipping occurred
                if (x_clipped != x or y_clipped != y or w_clipped != w or h_clipped != h):
                    boxes_clipped += 1

                # Convert to YOLO format (center_x, center_y, width, height, all normalized)
                x_center = (x_clipped + w_clipped / 2) / img_width
                y_center = (y_clipped + h_clipped / 2) / img_height
                w_norm = w_clipped / img_width
                h_norm = h_clipped / img_height

                # Check for duplicates and apply mapping (if any)
                if object_label_map:
                    mapped_class = object_label_map.get(class_id)
                    if mapped_class is None:
                        unmapped_objects += 1
                        mapped_class = object_default_label
                else:
                    mapped_class = class_id

                if is_duplicate_bbox(mapped_class, x_center, y_center, w_norm, h_norm, seen_boxes):
                    duplicates_removed += 1
                    continue

                f.write(f"{mapped_class} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
        
        frames_extracted += 1
    
    cap.release()
    return frames_extracted, boxes_clipped, duplicates_removed, unmapped_objects


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert MVRHOI annotations to YOLOv5 format.")
    parser.add_argument("--raw-root", type=Path, default=Path("d:/mvrhoi_raw"), help="Root directory with raw MVRHOI data (videos/, annotations/).")
    parser.add_argument("--output-dir", type=Path, default=Path("d:/mvrhoi_dataset"), help="Destination directory for YOLO images and labels.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train/val split ratio.")
    parser.add_argument("--object-map", type=Path, default=None, help="Optional JSON mapping raw object ids to VidHOI ids.")
    parser.add_argument("--object-default-label", type=int, default=0, help="Fallback label id when mapping is missing an entry.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val split.")
    return parser.parse_args()


def main():
    args = parse_args()

    mvrhoi_raw = args.raw_root
    anno_dir = mvrhoi_raw / "annotations"
    video_dir = mvrhoi_raw / "videos"

    output_dir = args.output_dir
    output_images_train = output_dir / "images" / "train"
    output_images_val = output_dir / "images" / "val"
    output_labels_train = output_dir / "labels" / "train"
    output_labels_val = output_dir / "labels" / "val"
    
    # Create directories
    for d in [output_images_train, output_images_val, output_labels_train, output_labels_val]:
        d.mkdir(parents=True, exist_ok=True)

    if args.object_map is not None:
        object_label_map = load_label_map(args.object_map)
        print(f"Loaded {len(object_label_map)} object mappings from {args.object_map}")
    else:
        object_label_map = {}
    object_default_label = args.object_default_label
    
    print("=" * 80)
    print("MVRHOI → YOLOv5 Converter")
    print("=" * 80)
    
    # Load annotations
    all_videos = load_mvrhoi_annotations(anno_dir)
    
    # Split into train/val (80/20)
    random.seed(args.seed)
    random.shuffle(all_videos)
    split_idx = int(len(all_videos) * args.train_ratio)
    train_videos = all_videos[:split_idx]
    val_videos = all_videos[split_idx:]
    
    print(f"\n📊 Dataset split:")
    print(f"   Train videos: {len(train_videos)}")
    print(f"   Val videos: {len(val_videos)}")
    
    # Process training set
    print(f"\n🔄 Processing training set...")
    train_frames = 0
    train_clipped = 0
    train_duplicates = 0
    train_unmapped = 0
    for video in train_videos:
        frames, clipped, duplicates, unmapped = extract_frames_and_labels(
            video,
            video_dir,
            output_images_train,
            output_labels_train,
            video['file'],
            object_label_map,
            object_default_label,
        )
        train_frames += frames
        train_clipped += clipped
        train_duplicates += duplicates
        train_unmapped += unmapped
        
        extra = ""
        if clipped > 0:
            extra += f", {clipped} boxes clipped"
        if duplicates > 0:
            extra += f", {duplicates} duplicates removed"
        if unmapped > 0 and object_label_map:
            extra += f", {unmapped} objects fell back to default label"
        print(f"   {video['file']}: {frames} frames{extra}")
    
    # Process validation set
    print(f"\n🔄 Processing validation set...")
    val_frames = 0
    val_clipped = 0
    val_duplicates = 0
    val_unmapped = 0
    for video in val_videos:
        frames, clipped, duplicates, unmapped = extract_frames_and_labels(
            video,
            video_dir,
            output_images_val,
            output_labels_val,
            video['file'],
            object_label_map,
            object_default_label,
        )
        val_frames += frames
        val_clipped += clipped
        val_duplicates += duplicates
        val_unmapped += unmapped
        
        extra = ""
        if clipped > 0:
            extra += f", {clipped} boxes clipped"
        if duplicates > 0:
            extra += f", {duplicates} duplicates removed"
        if unmapped > 0 and object_label_map:
            extra += f", {unmapped} objects fell back to default label"
        print(f"   {video['file']}: {frames} frames{extra}")
    
    print(f"\n✅ Conversion complete!")
    print(f"   Train frames: {train_frames}")
    print(f"   Val frames: {val_frames}")
    print(f"   Total: {train_frames + val_frames}")
    
    if train_clipped + val_clipped > 0:
        print(f"\n📏 Bounding boxes clipped to image boundaries: {train_clipped + val_clipped}")
    if train_duplicates + val_duplicates > 0:
        print(f"🔍 Duplicate bounding boxes removed: {train_duplicates + val_duplicates}")
    if object_label_map and (train_unmapped + val_unmapped) > 0:
        print(f"⚠️  Objects missing from mapping (used default label): {train_unmapped + val_unmapped}")
    print(f"\n📁 Output directory: {output_dir}")
    print(f"\n🚀 Ready to train YOLOv5:")
    print(f"   python modules/object_tracking/yolov5/train.py \\")
    print(f"       --data configs/mvrhoi.yaml \\")
    print(f"       --weights weights/yolov5/yolov5s.pt \\")
    print(f"       --epochs 50 \\")
    print(f"       --batch 16 \\")
    print(f"       --img 640 \\")
    print(f"       --project runs/mvrhoi \\")
    print(f"       --name yolov5s_finetune")


if __name__ == '__main__':
    main()
