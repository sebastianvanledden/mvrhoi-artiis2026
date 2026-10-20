#!/usr/bin/env python3
"""
Verify MVRHOI YOLOv5 dataset and visualize samples.

Usage:
    python verify_mvrhoi_dataset.py --num-samples 5
"""

import argparse
import cv2
import random
from pathlib import Path
from collections import defaultdict


def load_yolo_label(label_path: Path, img_width: int, img_height: int):
    """Load YOLO format labels and convert to pixel coordinates."""
    boxes = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            
            class_id = int(parts[0])
            x_center = float(parts[1]) * img_width
            y_center = float(parts[2]) * img_height
            width = float(parts[3]) * img_width
            height = float(parts[4]) * img_height
            
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)
            
            boxes.append((class_id, x1, y1, x2, y2))
    
    return boxes


def visualize_sample(img_path: Path, label_path: Path, class_names: dict, output_dir: Path):
    """Visualize a sample with bounding boxes."""
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"⚠️  Cannot load image: {img_path}")
        return
    
    h, w = img.shape[:2]
    
    # Load and draw boxes
    boxes = load_yolo_label(label_path, w, h)
    
    for class_id, x1, y1, x2, y2 in boxes:
        class_name = class_names.get(class_id, f"unknown_{class_id}")
        color = (0, 255, 0)  # Green
        
        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{class_name} ({class_id})"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1 - text_h - 4), (x1 + text_w, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    # Save
    output_path = output_dir / img_path.name
    cv2.imwrite(str(output_path), img)
    print(f"✅ Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', type=str, default='d:/mvrhoi_yolov5_dataset')
    parser.add_argument('--num-samples', type=int, default=5)
    parser.add_argument('--output-dir', type=str, default='d:/mvrhoi_yolov5_dataset/dataset_verification')
    args = parser.parse_args()
    
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # MVRHOI class names
    '''class_names = {
        0: "default", 1: "person", 2: "door", 3: "container", 4: "button_switch",
        5: "knob_valve", 6: "lever_handle", 7: "hand_object", 8: "portable",
        9: "keycard", 10: "tool_device", 11: "rideable", 12: "vehicle",
        13: "structure", 14: "projectile", 15: "weapon", 16: "pickup",
        17: "furniture", 18: "grappling_hook", 19: "structure_item",
        20: "surface", 21: "character", 22: "non_human", 23: "creature",
        24: "menu", 25: "inventory", 26: "text_field", 27: "ui_widget",
        28: "hand_right", 29: "hand_left"
    }'''
    # VidHOI class names (78)
    class_names = {
        0: "person",  1: "car",  2: "guitar",  3: "chair",  4: "handbag",  5: "toy",
        6: "baby_seat",  7: "cat",  8: "bottle",  9: "backpack",  10: "motorcycle",
        11: "ball/sports_ball",  12: "laptop",  13: "table",  14: "surfboard",  15: "camera",
        16: "sofa",  17: "screen/monitor",  18: "bicycle",  19: "vegetables",  20: "dog",
        21: "fruits",  22: "cake",  23: "cellphone",  24: "cup",  25: "bench",  26: "snowboard",
        27: "skateboard",  28: "bread",  29: "bus/truck",  30: "ski",  31: "suitcase",
        32: "stool",  33: "bat",  34: "elephant",  35: "fish",  36: "baby_walker",  37: "dish",
        38: "watercraft",  39: "scooter",  40: "pig",  41: "refrigerator",  42: "horse",
        43: "crab",  44: "bird",  45: "piano",  46: "cattle/cow",  47: "lion",  48: "chicken",
        49: "camel",  50: "electric_fan",  51: "toilet",  52: "sheep/goat",  53: "rabbit",
        54: "train",  55: "penguin",  56: "hamster/rat",  57: "snake",  58: "frisbee",
        59: "aircraft",  60: "oven",  61: "racket",
        62: "faucet",  63: "antelope",  64: "duck",  65: "stop_sign",  66: "sink",
        67: "kangaroo",  68: "stingray",  69: "turtle",  70: "tiger",  71: "crocodile",
        72: "bear",  73: "microwave",  74: "traffic_light",  75: "panda",  76: "leopard",  77: "squirrel"
    }
    
    print("=" * 80)
    print("MVRHOI Dataset Verification")
    print("=" * 80)
    
    # Check directories
    train_img_dir = dataset_dir / "images" / "train"
    train_label_dir = dataset_dir / "labels" / "train"
    val_img_dir = dataset_dir / "images" / "val"
    val_label_dir = dataset_dir / "labels" / "val"
    
    for d in [train_img_dir, train_label_dir, val_img_dir, val_label_dir]:
        if not d.exists():
            print(f"❌ Missing directory: {d}")
            return
    
    # Count files
    train_imgs = list(train_img_dir.glob("*.jpg"))
    train_labels = list(train_label_dir.glob("*.txt"))
    val_imgs = list(val_img_dir.glob("*.jpg"))
    val_labels = list(val_label_dir.glob("*.txt"))
    
    print(f"\n📊 Dataset statistics:")
    print(f"   Train: {len(train_imgs)} images, {len(train_labels)} labels")
    print(f"   Val:   {len(val_imgs)} images, {len(val_labels)} labels")
    
    if len(train_imgs) != len(train_labels):
        print(f"⚠️  Mismatch: {len(train_imgs)} images but {len(train_labels)} labels in train set")
    if len(val_imgs) != len(val_labels):
        print(f"⚠️  Mismatch: {len(val_imgs)} images but {len(val_labels)} labels in val set")
    
    # Analyze class distribution
    class_counts = defaultdict(int)
    total_boxes = 0
    
    for label_path in train_labels + val_labels:
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    class_counts[class_id] += 1
                    total_boxes += 1
    
    print(f"\n📦 Class distribution ({total_boxes} total boxes):")
    for class_id in sorted(class_counts.keys()):
        class_name = class_names.get(class_id, f"unknown_{class_id}")
        count = class_counts[class_id]
        pct = count / total_boxes * 100
        print(f"   {class_id:2d} {class_name:20s}: {count:5d} ({pct:5.1f}%)")
    
    # Visualize random samples
    print(f"\n🎨 Visualizing {args.num_samples} random samples...")
    all_imgs = train_imgs + val_imgs
    samples = random.sample(all_imgs, min(args.num_samples, len(all_imgs)))
    
    for img_path in samples:
        # Find corresponding label
        if "train" in str(img_path):
            label_path = train_label_dir / (img_path.stem + ".txt")
        else:
            label_path = val_label_dir / (img_path.stem + ".txt")
        
        if label_path.exists():
            visualize_sample(img_path, label_path, class_names, output_dir)
    
    print(f"\n✅ Verification complete! Check visualizations in: {output_dir}")
    print(f"\n🚀 Ready to train!")
    print(f"   python modules/object_tracking/yolov5/train.py \\")
    print(f"       --data configs/mvrhoi.yaml \\")
    print(f"       --weights weights/yolov5/yolov5s.pt \\")
    print(f"       --epochs 50 --batch 16 --img 640")


if __name__ == '__main__':
    main()
