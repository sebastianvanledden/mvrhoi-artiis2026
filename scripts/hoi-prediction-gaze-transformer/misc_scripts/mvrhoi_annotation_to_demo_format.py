#!/usr/bin/env python3
"""Utility converting a single MVR-HOI annotation file to the Demo HOI JSON format.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from mvrhoi_convert_mvrhoi_to_vidhoi import (
    PERSON_LABEL_ID,
    build_frame_interactions,
    build_frame_objects,
    clamp_margin,
    compute_target_size,
    load_annotation,
    load_label_map,
    synthetic_ego_bbox,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a single MVR-HOI annotation file to the Demo HOI JSON format."
    )
    parser.add_argument("--annotation", type=Path, required=True, help="Path to *_annotations_n.json file.")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the Demo-style HOI JSON (will be overwritten).",
    )
    parser.add_argument(
        "--target-width",
        type=int,
        default=0,
        help="Final width for bounding boxes. When 0, keep original width.",
    )
    parser.add_argument(
        "--target-height",
        type=int,
        default=0,
        help="Final height for bounding boxes. When 0, keep aspect ratio using target-width (or keep original if both zero).",
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
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Only export frames where index %% stride == 0. Set to 0 to match the source fps (one window per second).",
    )
    parser.add_argument(
        "--vidhoi-object-labels",
        type=Path,
        default=Path("vidhoi_related/obj_categories.json"),
        help="Path to VidHOI object categories json for target class names.",
    )
    parser.add_argument(
        "--vidhoi-predicate-labels",
        type=Path,
        default=Path("vidhoi_related/pred_categories.json"),
        help="Path to VidHOI predicate categories json for target class names.",
    )
    parser.add_argument(
        "--human-label",
        type=int,
        default=PERSON_LABEL_ID,
        help="VidHOI label id representing humans in the TARGET class set (defaults to PERSON_LABEL_ID).",
    )
    return parser.parse_args()


def parse_label_list(txt: str) -> Set[int]:
    values: Set[int] = set()
    if not txt:
        return values
    for part in txt.split(","):
        piece = part.strip()
        if not piece:
            continue
        try:
            values.add(int(piece))
        except ValueError as exc:
            raise ValueError(f"Invalid label id '{piece}' in alias-human-labels argument") from exc
    return values


def ensure_label_names(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Label file {path} not found")
    with path.open("r") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [str(x) for x in data]
    if isinstance(data, dict):
        if "obj_classes" in data and isinstance(data["obj_classes"], list):
            return [str(x) for x in data["obj_classes"]]
        if "predicates" in data and isinstance(data["predicates"], list):
            return [str(x) for x in data["predicates"]]
    raise ValueError(f"Unsupported label format in {path}")


def load_reverse_object_map(map_path: Path) -> Dict[int, Dict[str, List]]:
    with map_path.open("r") as f:
        data = json.load(f)

    entries: List[Dict] = []
    if isinstance(data, dict):
        if isinstance(data.get("labels"), list):
            entries.extend(data["labels"])
        elif all(isinstance(v, (int, str)) for v in data.values()):
            for raw_id, target in data.items():
                entries.append({"source_id": raw_id, "target_id": target})
        else:
            entries.append(data)
    elif isinstance(data, list):
        entries.extend(data)
    else:
        raise ValueError(f"Unsupported object mapping format in {map_path}")

    reverse_store: Dict[int, Dict[str, object]] = {}

    def coerce_int(value) -> Optional[int]:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def coerce_str(value) -> Optional[str]:
        if value is None:
            return None
        value_str = str(value).strip()
        return value_str or None

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        target_id = coerce_int(
            entry.get("target_id")
            or entry.get("target")
            or entry.get("vidhoi_id")
            or entry.get("target_idx")
            or entry.get("config_id")
        )
        if target_id is None:
            if "source" in entry and "target" in entry:
                target_id = coerce_int(entry.get("target"))
            if target_id is None:
                continue

        source_id = coerce_int(
            entry.get("source_id")
            or entry.get("source")
            or entry.get("mvrhoi_id")
            or entry.get("raw_id")
        )
        source_name = coerce_str(
            entry.get("source_name")
            or entry.get("source_label")
            or entry.get("source")
            or entry.get("notes")
            or entry.get("raw_name")
        )
        if source_name is None and source_id is not None:
            source_name = f"source_{source_id}"

        target_store = reverse_store.setdefault(
            target_id,
            {
                "source_ids": set(),
                "source_names": set(),
                "source_pairs": {},
                "name_only": set(),
            },
        )

        if source_id is not None:
            target_store["source_ids"].add(source_id)
        if source_name is not None:
            target_store["source_names"].add(source_name)
        if source_id is not None:
            names_set = target_store.setdefault("source_pairs", {}).setdefault(source_id, set())
            names_set.add(source_name)
        elif source_name is not None:
            target_store["name_only"].add(source_name)

    reverse_map: Dict[int, Dict[str, List]] = {}
    for target_id, info in reverse_store.items():
        source_pairs_list: List[Dict[str, object]] = []
        for src_id, names in sorted((info.get("source_pairs") or {}).items()):
            source_pairs_list.append(
                {
                    "source_id": src_id,
                    "source_names": sorted(name for name in names if name),
                }
            )
        reverse_map[target_id] = {
            "source_ids": sorted(info["source_ids"]),
            "source_names": sorted(info["source_names"]),
            "source_pairs": source_pairs_list,
            "name_only": sorted(info.get("name_only", set())),
        }
    return reverse_map


def build_object_display_name(base_name: str, mapping_entry: Optional[Dict[str, List]]) -> str:
    if not mapping_entry:
        return base_name
    candidate_names: Set[str] = set()
    for collection in ("source_names", "name_only"):
        for val in mapping_entry.get(collection, []) or []:
            if val and val.lower() != base_name.lower():
                candidate_names.add(str(val))
    for pair in mapping_entry.get("source_pairs", []) or []:
        for val in pair.get("source_names", []) or []:
            if val and val.lower() != base_name.lower():
                candidate_names.add(str(val))
    if not candidate_names:
        return base_name
    extras = ", ".join(sorted(candidate_names))
    return f"{base_name} [{extras}]"


def is_duplicate_entry(entry: Dict, seen_keys: Set[Tuple]) -> bool:
    key = (
        entry["frame"]["index"],
        tuple(entry["subject"]["bbox"]),
        tuple(entry["object"]["bbox"]),
        entry["interaction"]["name"],
        entry["interaction"].get("raw_action_id"),
    )
    if key in seen_keys:
        return True
    seen_keys.add(key)
    return False


def resolve_label_name(label_idx: int, names: List[str]) -> str:
    if 0 <= label_idx < len(names):
        return str(names[label_idx])
    return str(label_idx)


def convert_annotation(args: argparse.Namespace) -> List[Dict]:
    data = load_annotation(args.annotation)
    annotation = data["annotation"]
    video_meta = annotation["video"]
    fps = float(video_meta["fps"])
    orig_width = int(video_meta["width"])
    orig_height = int(video_meta["height"])
    total_frames = int(video_meta["frames"])

    target_width = args.target_width or orig_width
    target_height = args.target_height
    if target_height == 0 and args.target_width == 0:
        target_height = orig_height
    tw, th = compute_target_size(orig_width, orig_height, target_width, target_height)
    sx = float(tw) / float(orig_width)
    sy = float(th) / float(orig_height)

    stride = args.frame_stride
    if stride <= 0:
        stride = max(1, int(round(fps)))

    if args.object_map is not None:
        object_label_map = load_label_map(args.object_map)
    else:
        object_label_map = {}

    if args.predicate_map is not None:
        predicate_label_map = load_label_map(args.predicate_map)
    else:
        predicate_label_map = {}

    alias_human_labels = parse_label_list(args.alias_human_labels)
    ego_box_margin = clamp_margin(args.ego_box_margin)

    frame_objects = build_frame_objects(
        annotation,
        sx,
        sy,
        tw,
        th,
        alias_human_labels,
        object_label_map,
        args.object_default_label,
    )
    frame_interactions = build_frame_interactions(
        annotation,
        fps,
        total_frames,
        predicate_label_map,
        args.predicate_default_label if predicate_label_map else None,
    )

    vidhoi_object_names = ensure_label_names(args.vidhoi_object_labels)
    vidhoi_predicate_names = ensure_label_names(args.vidhoi_predicate_labels)

    reverse_map: Dict[int, Dict[str, List]] = {}
    if args.object_map is not None and args.object_map.exists():
        reverse_map = load_reverse_object_map(args.object_map)

    results: List[Dict] = []
    seen_keys: Set[Tuple] = set()

    for frame_idx in sorted(frame_interactions.keys()):
        if frame_idx % stride != 0:
            continue
        objects_in_frame = frame_objects.get(frame_idx)
        if not objects_in_frame:
            continue
        interactions = frame_interactions[frame_idx]
        for subj_inst, obj_inst, action_id, action_raw in interactions:
            subj_data = objects_in_frame.get(subj_inst)
            if subj_data is None and args.allow_ego_user:
                subj_data = {
                    "label": PERSON_LABEL_ID,
                    "raw_label": PERSON_LABEL_ID,
                    "bbox": synthetic_ego_bbox(tw, th, ego_box_margin),
                }
                objects_in_frame[subj_inst] = subj_data
            obj_data = objects_in_frame.get(obj_inst)
            if subj_data is None or obj_data is None:
                continue
            if subj_data["label"] != args.human_label:
                continue

            subject_label = subj_data["label"]
            object_label = obj_data["label"]
            predicate_label = action_id

            subject_name = resolve_label_name(subject_label, vidhoi_object_names)
            object_name = resolve_label_name(object_label, vidhoi_object_names)
            predicate_name = resolve_label_name(predicate_label, vidhoi_predicate_names)

            subject_mapping = reverse_map.get(subject_label)
            object_mapping = reverse_map.get(object_label)

            subject_bbox = [int(subj_data["bbox"][key]) for key in ("xmin", "ymin", "xmax", "ymax")]
            object_bbox = [int(obj_data["bbox"][key]) for key in ("xmin", "ymin", "xmax", "ymax")]

            entry = {
                "frame": {
                    "index": frame_idx,
                    "total": total_frames,
                },
                "subject": {
                    "name": subject_name,
                    "display_name": build_object_display_name(subject_name, subject_mapping),
                    "id": int(subj_inst),
                    "bbox": subject_bbox,
                    "weight": 1.0,
                    "mapped_source_ids": list(subject_mapping.get("source_ids", [])) if subject_mapping else [],
                    "mapped_source_names": list(subject_mapping.get("source_names", [])) if subject_mapping else [],
                    "mapped_source_pairs": list(subject_mapping.get("source_pairs", [])) if subject_mapping else [],
                    "mapped_name_only": list(subject_mapping.get("name_only", [])) if subject_mapping else [],
                    "raw_subject_label": int(subj_data["raw_label"]),
                },
                "interaction": {
                    "name": predicate_name,
                    "pair_indices": [int(subj_inst), int(obj_inst)],
                    "raw_action_id": int(action_raw),
                },
                "object": {
                    "name": object_name,
                    "display_name": build_object_display_name(object_name, object_mapping),
                    "id": int(obj_inst),
                    "bbox": object_bbox,
                    "weight": 1.0,
                    "mapped_source_ids": list(object_mapping.get("source_ids", [])) if object_mapping else [],
                    "mapped_source_names": list(object_mapping.get("source_names", [])) if object_mapping else [],
                    "mapped_source_pairs": list(object_mapping.get("source_pairs", [])) if object_mapping else [],
                    "mapped_name_only": list(object_mapping.get("name_only", [])) if object_mapping else [],
                    "raw_object_label": int(obj_data["raw_label"]),
                },
                "score": 1.0,
            }

            if is_duplicate_entry(entry, seen_keys):
                continue
            results.append(entry)

    return results


def main() -> None:
    args = parse_args()
    entries = convert_annotation(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(entries, f, indent=2)
    print(f"Wrote {len(entries)} HOI entries to {args.output}")


if __name__ == "__main__":
    main()
