#!/usr/bin/env python3
"""Split a flat VidHOI annotation file into train/val subsets grouped by frame ranges."""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split VidHOI annotations into train/val JSON files.")
    parser.add_argument("--annotations", type=Path, required=True, help="Input annotations.json in VidHOI format.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where train_frame_annots.json and val_frame_annots.json will be written.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Fraction of videos to reserve for validation (0-1). Remaining videos form the train split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for reproducible splits.",
    )
    return parser.parse_args()


def group_by_frame_range(entries: List[Dict]) -> List[Dict]:
    video_frames: Dict[Tuple[str, str], Dict[int, List[Dict]]] = defaultdict(lambda: defaultdict(list))
    for entry in entries:
        key = (entry["video_folder"], entry["video_id"])
        try:
            frame_idx = int(entry["frame_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid frame_id in entry: {entry.get('frame_id')}" ) from exc
        video_frames[key][frame_idx].append(entry)

    groups: List[Dict] = []

    for (video_folder, video_id), frames_map in video_frames.items():
        sorted_frames = sorted(frames_map.keys())
        if not sorted_frames:
            continue

        def finalize_range(range_frames: List[int]) -> None:
            groups.append(
                {
                    "video_folder": video_folder,
                    "video_id": video_id,
                    "start_frame": range_frames[0],
                    "end_frame": range_frames[-1],
                    "entries": [ann for frame in range_frames for ann in frames_map[frame]],
                }
            )

        current_frames: List[int] = [sorted_frames[0]]
        for frame in sorted_frames[1:]:
            if frame == current_frames[-1] + 1:
                current_frames.append(frame)
            else:
                finalize_range(current_frames)
                current_frames = [frame]
        finalize_range(current_frames)

    return groups


def split_frame_ranges(groups: List[Dict], val_ratio: float, seed: int) -> Tuple[List[Dict], List[Dict]]:
    if not 0.0 <= val_ratio <= 1.0:
        raise ValueError("val-ratio must be between 0 and 1")
    if len(groups) < 2:
        raise ValueError("Need at least two frame ranges to create non-empty splits")

    rng = random.Random(seed)
    shuffled = groups[:]
    rng.shuffle(shuffled)

    total_entries = sum(len(group["entries"]) for group in shuffled)
    target_val_entries = int(round(total_entries * val_ratio))

    if target_val_entries <= 0:
        raise ValueError("Validation ratio too low: no frame ranges selected for validation")
    if target_val_entries >= total_entries:
        raise ValueError("Validation ratio too high: no frame ranges left for training")

    val_groups: List[Dict] = []
    train_groups: List[Dict] = []
    val_entries = 0
    train_entries = 0

    for idx, group in enumerate(shuffled):
        remaining_groups = len(shuffled) - idx - 1
        need_val = target_val_entries - val_entries
        if need_val > 0 and (remaining_groups >= 1 or train_groups):
            val_groups.append(group)
            val_entries += len(group["entries"])
        else:
            train_groups.append(group)
            train_entries += len(group["entries"])

    if not val_groups:
        raise ValueError("Validation ratio too low: no frame ranges selected for validation")
    if not train_groups:
        moved = val_groups.pop()
        val_entries -= len(moved["entries"])
        train_groups.append(moved)
        train_entries += len(moved["entries"])
        if not val_groups:
            raise ValueError("Validation ratio too high: no frame ranges left for training")

    return train_groups, val_groups


def main() -> None:
    args = parse_args()
    entries = json.loads(args.annotations.read_text())
    groups = group_by_frame_range(entries)
    train_groups, val_groups = split_frame_ranges(groups, args.val_ratio, args.seed)

    train_entries = [ann for group in train_groups for ann in group["entries"]]
    val_entries = [ann for group in val_groups for ann in group["entries"]]

    train_entries.sort(key=lambda ann: (ann["video_folder"], ann["video_id"], int(ann["frame_id"]), ann["person_id"], ann["object_id"], ann["action_class"]))
    val_entries.sort(key=lambda ann: (ann["video_folder"], ann["video_id"], int(ann["frame_id"]), ann["person_id"], ann["object_id"], ann["action_class"]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "train_frame_annots.json").write_text(json.dumps(train_entries))
    (args.output_dir / "val_frame_annots.json").write_text(json.dumps(val_entries))

    print(
        "Frame ranges total: {total} | train: {train} | val: {val}".format(
            total=len(groups), train=len(train_groups), val=len(val_groups)
        )
    )
    total_entries = len(train_entries) + len(val_entries)
    val_ratio_actual = len(val_entries) / total_entries if total_entries else 0.0
    print(
        "Train annotations: {train} | Val annotations: {val} | Actual val ratio: {ratio:.3f}".format(
            train=len(train_entries),
            val=len(val_entries),
            ratio=val_ratio_actual,
        )
    )


if __name__ == "__main__":
    main()
