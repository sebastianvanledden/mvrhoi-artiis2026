#!/usr/bin/env python3
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import cv2

# Developer note: converts MVR-HOI video annotations into VidHOI-style frame entries.
#
# For each *_annotations_n.json file the script:
#   1. Builds per-frame object metadata via build_frame_objects, normalizing boxes
#      and remapping labels with optional aliasing for human subjects.
#   2. Expands actionAnnotationList intervals into frame-wise subject-object-action
#      tuples in build_frame_interactions, remapping predicates when configured.
#   3. Joins the object and interaction dictionaries inside convert_video to emit
#      deduplicated VidHOI entries, optionally synthesizing egocentric subject
#      boxes, and exports the corresponding resized frames.
#
# Example:
#   annotation = {
#       "objectAnnotationListMap": {
#           "45": [
#               {"instance": 1, "labelId": 1, "x": 100, "y": 120, "width": 80, "height": 200},
#               {"instance": 5, "labelId": 22, "x": 300, "y": 200, "width": 150, "height": 120},
#           ]
#       },
#       "actionAnnotationList": [
#           {"start": 1.5, "end": 1.5, "action": 32, "description": "1;5"} # subject 1, object 5
#       ],
#   }
#   produces frame_objects[45] with person/object boxes, frame_interactions[45] = [(1, 5, 32, 32)],
#   and convert_video emits:
#       {
#           "video_folder": "1",
#           "video_id": "Video_1",
#           "frame_id": "000045",
#           "person_box": {...},
#           "object_box": {...},
#           "person_id": 3,      # instance*2 + 1
#           "object_id": 12,      # instance*2 + 2
#           "object_class": 22,
#           "action_class": 32,
#       }
#   (other fields include fps metadata, middle_frame_timestamp, *_raw ids, etc.).

# VidHOI class index that represents humans in the target label space (NOT the source id).
# Downstream conversions remap any source label flagged as a person to this id. In VidHOI, this is 0.
PERSON_LABEL_ID = 0


def load_label_map(map_path: Path) -> Dict[int, int]:
    """Load a mapping from source label ids to VidHOI ids."""
    with map_path.open("r") as f:
        data = json.load(f)

    mapping: Dict[int, int] = {}

    def _coerce_key_val(key_obj, val_obj) -> Optional[Tuple[int, int]]:
        try:
            key_int = int(key_obj)
        except (TypeError, ValueError):
            return None
        if val_obj is None:
            return None
        try:
            val_int = int(val_obj)
        except (TypeError, ValueError):
            return None
        return key_int, val_int

    if isinstance(data, dict):
        # Case 1: direct mapping {"54": 17, ...}
        if all(isinstance(v, (int, str)) or v is None for v in data.values()):
            for raw_id, mapped_id in data.items():
                pair = _coerce_key_val(raw_id, mapped_id)
                if pair is not None:
                    mapping[pair[0]] = pair[1]
        # Case 2: nested mapping similar to predicate mapping files
        if not mapping and "vidor_to_config_mapping" in data:
            nested = data["vidor_to_config_mapping"]
            if isinstance(nested, dict):
                for raw_id, info in nested.items():
                    if not isinstance(info, dict):
                        continue
                    mapped = info.get("config_id")
                    if mapped is None:
                        mapped = info.get("suggested_config_id")
                    pair = _coerce_key_val(raw_id, mapped)
                    if pair is not None:
                        mapping[pair[0]] = pair[1]
        # Case 3: generic `mapping` or `object_map` key
        if not mapping:
            for key in ("mapping", "object_map", "vidor_to_vidhoi"):
                nested = data.get(key)
                if isinstance(nested, dict):
                    for raw_id, mapped_id in nested.items():
                        pair = _coerce_key_val(raw_id, mapped_id)
                        if pair is not None:
                            mapping[pair[0]] = pair[1]
        # Case 4: explicit label list [{"source_id": x, "target_id": y}, ...]
        if not mapping and isinstance(data.get("labels"), list):
            for entry in data["labels"]:
                if not isinstance(entry, dict):
                    continue
                src = entry.get("source_id", entry.get("src_id"))
                dst = entry.get("target_id", entry.get("dst_id"))
                pair = _coerce_key_val(src, dst)
                if pair is not None:
                    mapping[pair[0]] = pair[1]

    if not mapping:
        raise ValueError(
            f"No usable entries found in object mapping file {map_path}. "
            "Expected a dict of raw-id -> vidhoi-id pairs or a nested 'vidor_to_config_mapping'."
        )

    return mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert MVR-HOI annotations and videos to the VidHOI frame format."
    )
    parser.add_argument("--annotations-dir", type=Path, required=True, help="Directory with *_annotations_n.json files.")
    parser.add_argument("--videos-dir", type=Path, required=True, help="Directory with *_fps_normalized.mp4 videos.")
    parser.add_argument("--output-annotations", type=Path, required=True, help="Path to write the aggregated VidHOI annotation json.")
    parser.add_argument("--output-frames-dir", type=Path, required=True, help="Directory to dump resized frames.")
    parser.add_argument("--target-width", type=int, default=640)
    parser.add_argument(
        "--target-height",
        type=int,
        default=0,
        help="Final frame height. If 0, keep aspect ratio based on target width.",
    )
    parser.add_argument(
        "--alias-human-labels",
        type=str,
        default="",
        help="Comma separated label IDs that should be treated as human subjects.",
    )
    parser.add_argument(
        "--allow-ego-user",
        action="store_true",
        help="Allow missing subject boxes to be replaced with an egocentric proxy.",
    )
    parser.add_argument(
        "--ego-box-margin",
        type=float,
        default=0.05,
        help="Relative margin for synthetic egocentric subject boxes (0-0.49).",
    )
    parser.add_argument(
        "--object-map",
        type=Path,
        default=None,
        help="Optional JSON file mapping raw object labels to VidHOI object ids.",
    )
    parser.add_argument(
        "--object-default-label",
        type=int,
        default=0,
        help="Fallback VidHOI object id when a raw label is missing from the mapping.",
    )
    parser.add_argument(
        "--predicate-map",
        type=Path,
        default=None,
        help="Optional JSON file mapping raw action ids to VidHOI predicate ids.",
    )
    parser.add_argument(
        "--predicate-default-label",
        type=int,
        default=0,
        help="Fallback VidHOI predicate id when a raw action id is missing from the mapping.",
    )
    return parser.parse_args()


def load_annotation(annotation_path: Path) -> Dict:
    encodings = ("utf-8", "utf-8-sig", "latin-1")
    last_unicode_error: Optional[UnicodeDecodeError] = None
    last_decode_error: Optional[json.JSONDecodeError] = None
    last_text_preview: str = ""
    for encoding in encodings:
        try:
            text = annotation_path.read_text(encoding=encoding)
        except UnicodeDecodeError as err:
            last_unicode_error = err
            continue
        try:
            return json.loads(text)
        except json.JSONDecodeError as err:
            if not text.strip():
                raise ValueError(f"Annotation file {annotation_path} is empty or whitespace-only.") from err
            last_decode_error = err
            last_text_preview = text[:200].replace("\n", "\\n")
            continue
    if last_decode_error is not None:
        raise ValueError(
            f"Failed to parse JSON in {annotation_path} after trying encodings {encodings}. "
            f"First 200 characters: {last_text_preview}"
        ) from last_decode_error
    if last_unicode_error is not None:
        raise UnicodeDecodeError(
            last_unicode_error.encoding,
            last_unicode_error.object,
            last_unicode_error.start,
            last_unicode_error.end,
            f"Failed to decode {annotation_path} with encodings {encodings}"
        ) from last_unicode_error
    raise RuntimeError(f"Failed to load annotation from {annotation_path}")


def parse_label_list(txt: str) -> Set[int]:
    values: Set[int] = set()
    if not txt:
        return values
    for part in txt.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            values.add(int(part))
        except ValueError:
            raise ValueError(f"Invalid label id '{part}' in alias-human-labels argument")
    return values


def scale_bbox(x: float, y: float, w: float, h: float, sx: float, sy: float, tw: int, th: int) -> Dict[str, int]:
    """Scale and clip bounding box to image boundaries."""
    xmin = max(0.0, min(float(tw - 1), x * sx))
    ymin = max(0.0, min(float(th - 1), y * sy))
    xmax = max(0.0, min(float(tw - 1), (x + w) * sx))
    ymax = max(0.0, min(float(th - 1), (y + h) * sy))
    if xmax <= xmin:
        xmax = min(float(tw - 1), xmin + 1.0)
    if ymax <= ymin:
        ymax = min(float(th - 1), ymin + 1.0)
    return {
        "xmin": int(round(xmin)),
        "ymin": int(round(ymin)),
        "xmax": int(round(xmax)),
        "ymax": int(round(ymax)),
    }


def bbox_to_tuple(bbox: Dict[str, int]) -> Tuple[int, int, int, int]:
    """Convert bbox dict to tuple for hashing/comparison."""
    return (bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"])


def is_duplicate_entry(entry: Dict, seen_triplets: Set[Tuple]) -> bool:
    """Check if this frame-person-object-action combination already exists."""
    triplet = (
        entry["video_folder"],
        entry["video_id"],
        entry["frame_id"],
        bbox_to_tuple(entry["person_box"]),
        bbox_to_tuple(entry["object_box"]),
        entry["action_class"],
    )
    if triplet in seen_triplets:
        return True
    seen_triplets.add(triplet)
    return False


def build_frame_objects(
    annotation: Dict,
    sx: float,
    sy: float,
    tw: int,
    th: int,
    alias_human_labels: Set[int],
    object_label_map: Dict[int, int],
    object_default_label: int,
) -> Dict[int, Dict[int, Dict]]:
    frame_objects: Dict[int, Dict[int, Dict]] = defaultdict(dict)
    object_map = annotation.get("objectAnnotationListMap", {})
    for frame_idx_str, objects in object_map.items():
        try:
            frame_idx = int(frame_idx_str)
        except ValueError:
            continue
        for obj in objects:
            inst_raw = obj.get("instance")
            label_raw = obj.get("labelId")
            if inst_raw is None or label_raw is None:
                continue
            try:
                inst_id = int(inst_raw)
                label_id = int(label_raw)
            except (TypeError, ValueError):
                continue
            bbox = scale_bbox(obj["x"], obj["y"], obj["width"], obj["height"], sx, sy, tw, th)
            if label_id == PERSON_LABEL_ID or label_id in alias_human_labels:
                converted_label = PERSON_LABEL_ID
            else:
                converted_label = object_label_map.get(label_id, object_default_label)
            frame_objects[frame_idx][inst_id] = {
                "label": converted_label,
                "raw_label": label_id,
                "bbox": bbox,
            }
    return frame_objects


def build_frame_interactions(
    annotation: Dict,
    fps: float,
    total_frames: int,
    predicate_label_map: Dict[int, int],
    predicate_default_label: Optional[int],
) -> Dict[int, List[Tuple[int, int, int, int]]]:
    frame_interactions: Dict[int, List[Tuple[int, int, int, int]]] = defaultdict(list)
    for action in annotation.get("actionAnnotationList", []):
        description = action.get("description", "")
        if ";" not in description:
            continue
        subj_str, obj_str = description.split(";", 1)
        try:
            subj_id = int(subj_str.strip())
            obj_id = int(obj_str.strip())
        except ValueError:
            continue
        action_raw = int(action.get("action", 0))
        if predicate_label_map:
            if predicate_default_label is None:
                raise ValueError("Predicate mapping supplied without --predicate-default-label")
            action_id = int(predicate_label_map.get(action_raw, predicate_default_label))
        else:
            action_id = action_raw
        start_time = max(0.0, float(action.get("start", 0.0)))
        end_time = max(start_time, float(action.get("end", start_time)))
        start_frame = max(0, int(math.floor(start_time * fps)))
        end_frame = min(total_frames - 1, int(math.ceil(end_time * fps)))
        for frame_idx in range(start_frame, end_frame + 1):
            frame_interactions[frame_idx].append((subj_id, obj_id, action_id, action_raw))
    return frame_interactions


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def export_frames(video_path: Path, frame_ids: Iterable[int], dest_dir: Path, target_size: Tuple[int, int]) -> None:
    requested = sorted(set(frame_ids))
    if not requested:
        return
    ensure_dir(dest_dir)
    requested_set = set(requested)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video {video_path}")
    next_idx_iter = iter(requested)
    try:
        next_needed = next(next_idx_iter)
    except StopIteration:
        next_needed = None
    frame_idx = 0
    written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx in requested_set:
            resized = cv2.resize(frame, target_size)
            frame_name = f"{dest_dir.name}_{frame_idx:06d}.jpg"
            cv2.imwrite(str(dest_dir / frame_name), resized)
            written += 1
            if next_needed is not None and frame_idx == next_needed:
                try:
                    next_needed = next(next_idx_iter)
                except StopIteration:
                    next_needed = None
        frame_idx += 1
        if written == len(requested_set):
            break
    cap.release()
    missing = len(requested_set) - written
    if missing:
        missing_frames = sorted(requested_set - set(range(frame_idx)))
        raise RuntimeError(f"Could not export {missing} frames for {video_path}, missing indices: {missing_frames[:5]}")


def compute_target_size(orig_width: int, orig_height: int, target_width: int, target_height: int) -> Tuple[int, int]:
    if target_height > 0:
        return target_width, target_height
    scale = float(target_width) / float(orig_width)
    scaled_height = max(1, int(round(orig_height * scale)))
    return target_width, scaled_height


def clamp_margin(margin: float) -> float:
    if margin < 0.0:
        return 0.0
    if margin > 0.49:
        return 0.49
    return margin


def synthetic_ego_bbox(tw: int, th: int, margin: float) -> Dict[str, int]:
    margin = clamp_margin(margin)
    xmin = int(round(tw * margin))
    ymin = int(round(th * margin))
    xmax = int(round(tw * (1.0 - margin)))
    ymax = int(round(th * (1.0 - margin)))
    if xmax <= xmin:
        xmax = min(tw - 1, xmin + 1)
    if ymax <= ymin:
        ymax = min(th - 1, ymin + 1)
    return {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}


def convert_video(
    annotation_path: Path,
    video_path: Path,
    output_frames_root: Path,
    target_width: int,
    target_height: int,
    alias_human_labels: Set[int],
    object_label_map: Dict[int, int],
    object_default_label: int,
    predicate_label_map: Dict[int, int],
    predicate_default_label: Optional[int],
    allow_ego_user: bool,
    ego_box_margin: float,
) -> Tuple[List[Dict], List[int]]:
    data = load_annotation(annotation_path)
    annotation = data["annotation"]
    video_meta = annotation["video"]
    fps = float(video_meta["fps"])
    orig_width = int(video_meta["width"])
    orig_height = int(video_meta["height"])
    total_frames = int(video_meta["frames"])
    tw, th = compute_target_size(orig_width, orig_height, target_width, target_height)
    sx = float(tw) / float(orig_width)
    sy = float(th) / float(orig_height)
    frame_objects = build_frame_objects(
        annotation,
        sx,
        sy,
        tw,
        th,
        alias_human_labels,
        object_label_map,
        object_default_label,
    )
    frame_interactions = build_frame_interactions(
        annotation,
        fps,
        total_frames,
        predicate_label_map,
        predicate_default_label,
    )

    stem = annotation_path.name.replace("_annotations_n.json", "")
    if "_" not in stem:
        raise RuntimeError(f"Unexpected annotation filename {annotation_path.name}")
    folder_token, video_token = stem.split("_", 1)
    video_folder = folder_token
    video_id = video_token

    entries: List[Dict] = []
    frames_used: List[int] = []
    synthetic_count = 0
    duplicate_count = 0
    seen_triplets: Set[Tuple] = set()
    
    for frame_idx in sorted(frame_interactions.keys()):
        frame_objects_map = frame_objects.get(frame_idx)
        if not frame_objects_map:
            continue
        interactions = frame_interactions[frame_idx]
        frame_used = False
        for subj_inst, obj_inst, action_id, action_raw in interactions:
            subj_data = frame_objects_map.get(subj_inst)
            if subj_data is None and allow_ego_user:
                bbox = synthetic_ego_bbox(tw, th, ego_box_margin)
                subj_data = {
                    "label": PERSON_LABEL_ID,
                    "raw_label": PERSON_LABEL_ID,
                    "bbox": bbox,
                }
                frame_objects_map[subj_inst] = subj_data
                synthetic_count += 1
            obj_data = frame_objects_map.get(obj_inst)
            if subj_data is None or obj_data is None:
                continue
            if subj_data["label"] != PERSON_LABEL_ID:
                continue
            
            entry = {
                "video_folder": video_folder,
                "video_id": video_id,
                "frame_id": f"{frame_idx:06d}",
                "video_fps": fps,
                "height": th,
                "width": tw,
                "middle_frame_timestamp": frame_idx / fps,
                "person_box": subj_data["bbox"],
                "object_box": obj_data["bbox"],
                "person_id": subj_inst * 2 + 1,
                "object_id": obj_inst * 2 + 2,
                "object_class": obj_data["label"],
                "object_class_raw": obj_data["raw_label"],
                "action_class": action_id,
                "action_class_raw": action_raw,
            }
            
            # Check for duplicates
            if is_duplicate_entry(entry, seen_triplets):
                duplicate_count += 1
                continue
            
            frame_used = True
            entries.append(entry)
        if frame_used:
            frames_used.append(frame_idx)
    if frames_used:
        dest_dir = output_frames_root / video_folder / video_id
        export_frames(video_path, frames_used, dest_dir, (tw, th))
    return entries, frames_used, {"synthetic_subjects": synthetic_count, "duplicates_removed": duplicate_count}


def main() -> None:
    args = parse_args()
    annotations_dir = args.annotations_dir
    videos_dir = args.videos_dir
    output_frames_dir = args.output_frames_dir
    ensure_dir(output_frames_dir)

    if args.object_map is not None:
        object_label_map = load_label_map(args.object_map)
        print(f"Loaded {len(object_label_map)} object label mappings from {args.object_map}")
    else:
        object_label_map = {}

    if args.predicate_map is not None:
        predicate_label_map = load_label_map(args.predicate_map)
        print(f"Loaded {len(predicate_label_map)} predicate label mappings from {args.predicate_map}")
    else:
        predicate_label_map = {}

    all_entries: List[Dict] = []
    alias_human_labels = parse_label_list(args.alias_human_labels)
    ego_box_margin = clamp_margin(args.ego_box_margin)
    annotation_files = sorted(annotations_dir.glob("*_annotations_n.json"))
    synthetic_total = 0
    duplicates_total = 0
    for annotation_path in annotation_files:
        stem = annotation_path.name.replace("_annotations_n.json", "")
        video_path = videos_dir / f"{stem}_fps_normalized.mp4"
        if not video_path.exists():
            print(f"Skipping {annotation_path.name}: missing video {video_path.name}")
            continue
        video_entries, _, stats = convert_video(
            annotation_path,
            video_path,
            output_frames_dir,
            args.target_width,
            args.target_height,
            alias_human_labels,
            object_label_map,
            args.object_default_label,
            predicate_label_map,
            args.predicate_default_label if predicate_label_map else None,
            args.allow_ego_user,
            ego_box_margin,
        )
        synthetic_total += stats.get("synthetic_subjects", 0)
        duplicates_total += stats.get("duplicates_removed", 0)
        all_entries.extend(video_entries)
        extra = ""
        if stats.get("synthetic_subjects", 0):
            extra = f", synthetic subjects: {stats['synthetic_subjects']}"
        if stats.get("duplicates_removed", 0):
            extra += f", duplicates removed: {stats['duplicates_removed']}"
        print(f"Processed {annotation_path.name}: {len(video_entries)} entries{extra}")
    if not all_entries:
        raise RuntimeError("No annotation entries generated. Check inputs.")
    ensure_dir(args.output_annotations.parent)
    with args.output_annotations.open("w") as f:
        json.dump(all_entries, f)
    print(f"Wrote {len(all_entries)} frame annotations to {args.output_annotations}")
    if synthetic_total and not args.allow_ego_user:
        print("Warning: synthetic subjects counted without enabling --allow-ego-user")
    elif synthetic_total:
        print(f"Synthetic egocentric subject boxes created: {synthetic_total}")
    if duplicates_total:
        print(f"Total duplicate entries removed: {duplicates_total}")


if __name__ == "__main__":
    main()
