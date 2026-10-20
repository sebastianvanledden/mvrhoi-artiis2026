#!/usr/bin/env python3
"""Visualize VidHOI frame annotations for quick spot checks."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2

COLOR_PERSON = (36, 255, 255)  # yellow-ish in BGR
COLOR_OBJECT = (255, 128, 0)   # orange-ish in BGR
FONT = cv2.FONT_HERSHEY_SIMPLEX


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw VidHOI annotations on exported frames.")
    parser.add_argument("--annotations", type=Path, required=True, help="Path to VidHOI annotation JSON (flat list).")
    parser.add_argument("--frames-dir", type=Path, required=True, help="Root directory with exported frames.")
    parser.add_argument("--video-folder", type=str, required=True, help="Video folder token (e.g., '10').")
    parser.add_argument("--video-id", type=str, required=True, help="Video id token (e.g., 'Video_01').")
    parser.add_argument("--output-dir", type=Path, default=Path("preview_frames"), help="Where to store visualizations.")
    parser.add_argument("--max-frames", type=int, default=5, help="Maximum number of frames to render (chronological order).")
    parser.add_argument("--line-thickness", type=int, default=2, help="Bounding box line thickness in pixels.")
    parser.add_argument("--text-scale", type=float, default=0.6, help="Font scale for labels.")
    return parser.parse_args()


def group_entries(entries: List[Dict]) -> Dict[str, List[Dict]]:
    per_frame: Dict[str, List[Dict]] = defaultdict(list)
    for entry in entries:
        frame_id = entry["frame_id"]
        per_frame[frame_id].append(entry)
    return per_frame


def draw_boxes(img, frame_entries: List[Dict], thickness: int, text_scale: float) -> None:
    for item in frame_entries:
        person_box = item["person_box"]
        object_box = item["object_box"]
        action_id = item["action_class"]
        person_id = item["person_id"]
        object_id = item["object_id"]

        cv2.rectangle(
            img,
            (person_box["xmin"], person_box["ymin"]),
            (person_box["xmax"], person_box["ymax"]),
            COLOR_PERSON,
            thickness,
        )
        cv2.rectangle(
            img,
            (object_box["xmin"], object_box["ymin"]),
            (object_box["xmax"], object_box["ymax"]),
            COLOR_OBJECT,
            thickness,
        )

        label = f"P{person_id} -> A{action_id} -> O{object_id}"
        text_org = (person_box["xmin"], max(15, person_box["ymin"] - 5))
        cv2.putText(img, label, text_org, FONT, text_scale, COLOR_PERSON, 1, cv2.LINE_AA)


def main() -> None:
    args = parse_args()
    annotations = json.loads(args.annotations.read_text())
    filtered = [
        entry
        for entry in annotations
        if entry["video_folder"] == args.video_folder and entry["video_id"] == args.video_id
    ]
    if not filtered:
        raise RuntimeError("No entries found for the requested video tokens.")

    frames = group_entries(filtered)
    sorted_frames = sorted(frames.keys())[: args.max_frames]

    out_dir = args.output_dir / args.video_folder / args.video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    for frame_id in sorted_frames:
        frame_path = (
            args.frames_dir
            / args.video_folder
            / args.video_id
            / f"{args.video_id}_{frame_id}.jpg"
        )
        if not frame_path.exists():
            print(f"Warning: frame not found {frame_path}")
            continue
        image = cv2.imread(str(frame_path))
        if image is None:
            print(f"Warning: failed to load {frame_path}")
            continue
        draw_boxes(image, frames[frame_id], args.line_thickness, args.text_scale)
        dest = out_dir / f"{args.video_id}_{frame_id}.jpg"
        cv2.imwrite(str(dest), image)
        print(f"Saved {dest}")


if __name__ == "__main__":
    main()
