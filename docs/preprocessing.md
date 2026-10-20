# Data Preparation and Preprocessing

## Overview

The MVRHOI processing workflow converts source recordings and interval-based interaction annotations into inputs suitable for the STTran-Gaze-based HOI pipeline. It covers video normalization, annotation conversion, label mapping, frame extraction, range-level dataset splitting, detector-label generation, and the preparation of Oracle- and Detection-Mode inputs.

The corresponding MVRHOI-specific implementations are documented in the [source inventory](source_inventory.md). The released scripts form an **overlay** for the upstream pipeline; they are not a self-contained replacement for all upstream components.

## Video normalization

The paper describes preprocessing with **FFmpeg** to **30 frames per second**, **bt709** color space, and **no audio**. The exact historical FFmpeg command, version, resizing and pixel-format settings, and source/output checksums are not established in the available release documentation. Consequently, this description records the processing intent but does not constitute a byte-identical reconstruction recipe.

The original recordings and extracted frames are not part of the Git repository. See [`artifacts/README.md`](../artifacts/README.md) for information about external inputs.

## Annotation conversion

The existing MVRHOI converter performs interval-to-frame conversion, object and predicate label mapping, bounding-box processing, duplicate removal, and frame extraction. The relevant implementation is `misc_scripts/mvrhoi_convert_mvrhoi_to_vidhoi.py` within `hoi-prediction-gaze-transformer/`.

The converter supports **`--allow-ego-user`** to introduce synthetic ego-user subjects where applicable. The experimental record also includes a Detection-Mode variant without these synthetic subjects. Ego-user inclusion is therefore an experimental choice and should be recorded with the generated dataset and checkpoint.

## Dataset splitting

After conversion, the archived splitter groups contiguous annotated frame ranges and assigns them to training or validation using **seed `1234`** and a target validation ratio of **`0.2`**. The split is **not video-disjoint**. Preserve the published split membership files when examining or comparing the archived results; do not silently regenerate the split.

See [dataset documentation](dataset.md) for the prepared ego-dataset counts and the implications of overlap at the recording level.

## Detector and inference inputs

The available detector converter prepares YOLO-format labels from the MVRHOI annotations. Separate scripts extract gaze-related inputs for Oracle Mode and detection/tracking/gaze-related inputs for Detection Mode.

Detection-derived caches and outputs depend on the detector checkpoint and its configuration. Where such files are created outside the repository, retain the corresponding checkpoint identifiers and processing settings to make the provenance of the generated inputs traceable.

The upstream STTran-Gaze framework, including its external dependencies, is required for the full pipeline. See the [fine-tuning guide](../scripts/finetune.md) for the available overlay workflow and [reproduction guidance](reproduction.md) for supported and unsupported reproduction levels.

## Reproducibility boundary

The repository preserves existing MVRHOI-specific processing implementations rather than presenting a newly reconstructed preprocessing pipeline. Reusing the scripts with newly acquired inputs may therefore produce a comparable dataset preparation, but should not be described as an exact recreation of the paper's inputs without additional provenance checks.
