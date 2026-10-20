#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
from collections import defaultdict

def analyze_annotations(annotations_dir):
    files = list(Path(annotations_dir).glob('*_annotations_n.json'))
    print(f"Total videos: {len(files)}")

    if not files:
        print(f"ERROR: No annotation files found in {annotations_dir}")
        return

    total_frames = 0
    total_objects = 0
    label_counts = defaultdict(int)

    for f in files:
        with open(f) as file:
            data = json.load(file)
        objs = data['annotation']['objectAnnotationListMap']
        
        for frame_id, obj_list in objs.items():
            total_frames += 1
            total_objects += len(obj_list)
            
            for obj in obj_list:
                label_counts[obj['labelId']] += 1

    print(f"\nTotal annotated keyframes: {total_frames}")
    print(f"Total object bounding boxes: {total_objects}")
    print(f"Average objects per frame: {total_objects/total_frames:.1f}")

    print(f"\nObject class distribution:")
    for label_id in sorted(label_counts.keys()):
        print(f"  Class {label_id}: {label_counts[label_id]} instances")

    print(f"\nTotal unique object classes: {len(label_counts)}")

def main():
    parser = argparse.ArgumentParser(
        description='Analyze MVRHOI annotation statistics'
    )
    parser.add_argument(
        '--annotations-dir',
        type=str,
        required=True,
        help='Path to directory containing *_annotations_n.json files'
    )
    args = parser.parse_args()
    
    analyze_annotations(args.annotations_dir)

if __name__ == '__main__': 
    main()