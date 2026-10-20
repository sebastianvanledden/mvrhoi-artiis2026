#!/usr/bin/env python3
"""Check metadata integrity, metric exports, and excluded artifact types."""
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = {
    ".mp4", ".mkv", ".avi", ".mov", ".webm", ".png", ".jpg", ".jpeg",
    ".gif", ".webp", ".bmp", ".tif", ".tiff", ".pt", ".pth", ".ckpt",
    ".onnx", ".safetensors", ".h5", ".bin", ".npz", ".npy", ".zip",
    ".tar", ".gz", ".7z",
}


def candidate_files():
    """Inspect commit candidates, including accidentally tracked ignored artifacts."""
    try:
        top = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', '--show-toplevel'],
                             check=True, capture_output=True, text=True).stdout.strip()
        if Path(top).resolve() == ROOT:
            result = subprocess.run(['git', '-C', str(ROOT), 'ls-files', '--cached', '--others',
                                     '--exclude-standard', '-z'], check=True, capture_output=True)
            return [ROOT / name for name in sorted(set(result.stdout.decode().split('\0'))) if name]
    except (OSError, subprocess.CalledProcessError):
        pass
    return [p for p in ROOT.rglob('*') if p.is_file() and '.git' not in p.relative_to(ROOT).parts]


def main():
    errors = []
    candidates = candidate_files()
    for path in candidates:
        if path.is_file() and path.suffix.lower() in FORBIDDEN:
            errors.append(f"Excluded artifact: {path.relative_to(ROOT)}")
        if path.is_file() and path.suffix.lower() == ".json":
            try:
                json.loads(path.read_text())
            except (ValueError, UnicodeError) as exc:
                errors.append(f"Invalid JSON: {path.relative_to(ROOT)}: {exc}")
    for entry in (json.loads((ROOT / "data/metadata_provenance.json").read_text())
                  + json.loads((ROOT / "data/code_provenance.json").read_text())):
        path = ROOT / entry["destination"]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            errors.append(f"Source-copy digest mismatch: {entry['destination']}")
    with (ROOT / "results/experiments.csv").open(newline="") as handle:
        experiments = list(csv.DictReader(handle))
    identifiers = [row["experiment_id"] for row in experiments]
    if len(identifiers) != len(set(identifiers)):
        errors.append("Duplicate experiment IDs")
    for experiment in experiments:
        try:
            with (ROOT / experiment["metrics_file"]).open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            if len(rows) != 1:
                raise ValueError("expected one aggregate metric row")
            for key in ("mAP_QPIC", "mAP_rare_QPIC", "mAP_non_rare_QPIC"):
                score = float(rows[0][key])
                if not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError(f"invalid {key}: {score}")
        except (OSError, ValueError, KeyError) as exc:
            errors.append(f"Invalid metrics for {experiment['experiment_id']}: {exc}")
    summary = json.loads((ROOT / "data/splits/ego_split_summary.json").read_text())
    memberships = {}
    for split in ("train", "val"):
        with (ROOT / f"data/splits/ego_{split}_frames.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        memberships[split] = {(r["video_folder"], r["video_id"], r["frame_id"]) for r in rows}
        videos = {(a, b) for a, b, _ in memberships[split]}
        if len(rows) != len(memberships[split]) or len(rows) != summary[split]["unique_frames"]:
            errors.append(f"Frame membership count mismatch: {split}")
        if len(videos) != summary[split]["unique_videos"]:
            errors.append(f"Video membership count mismatch: {split}")
    shared_frames = len(memberships["train"] & memberships["val"])
    shared_videos = len({r[:2] for r in memberships["train"]} & {r[:2] for r in memberships["val"]})
    if shared_frames != summary["shared_frames"] or shared_videos != summary["shared_videos"]:
        errors.append("Split overlap summary mismatch")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Companion checks passed: {len(experiments)} experiments; no excluded artifacts among commit candidates.")
    print(f"Historical split: {shared_videos} shared videos, {shared_frames} shared frames.")
    print("Training/inference reproducibility and publication readiness are not verified.")


if __name__ == "__main__":
    main()
