'''
Script for normalising the FPS of videos for annotation.
Only normalises the FPS to 30fps, sets the colour space uniformly to bt709 and
retains the original resolution.
Additionally reads metadata (environment/perspective) from the overall info CSV and
writes it to the output files during FFmpeg transcoding (container + video stream).
Audio is optionally removed.
'''

import csv
import os
import subprocess
from pathlib import Path

# Configuration
input_dir = "/Volumes/MVR256_Dataset/metaverse-dataset/output-videos"
output_dir = "/Volumes/MVR256_Dataset/metaverse-dataset/annotation-videos"
metadata_csv = Path(__file__).resolve().parent / "video_info" / "info_mvr256.csv"

# Target fps and standart audio-removal
target_fps = 30
remove_audio = True
# Use native Encoder 
video_encoder = os.environ.get("VIDEO_ENCODER", "h264_videotoolbox")

# Create target directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

# Supported formats
video_extensions = ('.mp4', '.m4v')

def load_metadata(csv_path: Path) -> dict:
    """Read the CSV and return a mapping of filename -> {environment, perspective}."""
    mapping = {}
    if not csv_path.exists():
        print(f"⚠️  Info: Metadata-CSV '{csv_path}' not found. No metadata will be added.")
        return mapping

    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=';')
        for row in reader:
            filename = row.get("filename")
            if not filename:
                continue
            mapping[filename] = {
                "environment": (row.get("environment") or "").strip(),
                "perspective": (row.get("perspective") or "").strip()
            }

    return mapping


metadata_lookup = load_metadata(metadata_csv)


# Loop through all file in input folder
for idx, filename in enumerate(sorted(os.listdir(input_dir))):
    if filename.lower().endswith(video_extensions):
        input_path = os.path.join(input_dir, filename)
        # split name and extension
        name_without_ext = os.path.splitext(filename)[0]
        
        # FPS-normalized outpout
        output_filename = f"{name_without_ext}_fps_normalized.mp4"
        output_path = os.path.join(output_dir, output_filename)

        # Check if file exists
        if os.path.exists(output_path):
            print(f"Skipping {idx+1}: {filename} → {output_filename} (already exists)")
            continue

        cmd = [
            "ffmpeg",
            "-i", input_path,
            "-r", str(target_fps),
            "-vf", "fps=30",             # FPS-filtering
            "-pix_fmt", "yuv420p",       # Consistent pixel-formating
            "-colorspace", "bt709",      # Standard HD colorspace (sRGB compatible)
            "-color_primaries", "bt709", # Standard HD color primaries
            "-color_trc", "bt709",       # Standard HD transfer characteristics
            "-color_range", "tv",        # TV range (16-235) - standard for most videos
            "-movflags", "use_metadata_tags",  # Custom QuickTime-Tags
            "-c:v", video_encoder,        # configurable Video-Encoder
            "-crf", "18"                 # High quality
        ]

        metadata = metadata_lookup.get(filename)
        if metadata:
            for key, value in metadata.items():
                if value:
                    # Container-wide Tag + first Videostream
                    cmd += [
                        "-metadata", f"{key}={value}",
                        "-metadata:s:v:0", f"{key}={value}"
                    ]
        else:
            print(f"  ℹ️  No metadata found for '{filename}'.")

        if remove_audio:
            cmd += ["-an"]

        cmd += ["-y", output_path]

        # Execute FFmpeg 
        if metadata:
            print(
                f"Normalized FPS for Video {idx+1}: {filename} → {output_filename} "
                f"(Environment: {metadata.get('environment') or 'n/a'}, Perspective: {metadata.get('perspective') or 'n/a'})"
            )
        else:
            print(f"Normalized FPS for Video {idx+1}: {filename} → {output_filename}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"  ❌ Error with {filename}: {result.stderr}")
        else:
            print(f"  ✅ Successfully processed: {output_filename}")

print(f"\nVideos have been successfully FPS-normalized:") 
print(f"  Output folder: '{output_dir}'")
