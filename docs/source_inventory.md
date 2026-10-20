# Pipeline Source Inventory and Provenance

## Implementation overview

The repository distributes **MVRHOI-specific additions** to the STTran-Gaze-based pipeline. The relevant code is organized as an overlay under [`scripts/hoi-prediction-gaze-transformer/`](../scripts/hoi-prediction-gaze-transformer/), rather than as a complete, independently executable fork of the upstream framework.

The table below identifies the implementations documented in the original training-workspace. **These paths describe the original workspace layout**, not necessarily directly executable paths from the repository root.

| Task                                     | Original training-workspace implementation                                                               |
| ---------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| Video normalization                      | `hoi-prediction-gaze-transformer/misc_scripts/mvrhoi_video_preprocessing.py`                             |
| Annotation conversion and ego subjects   | `hoi-prediction-gaze-transformer/misc_scripts/mvrhoi_convert_mvrhoi_to_vidhoi.py`                        |
| Frame-range splitting                    | `hoi-prediction-gaze-transformer/misc_scripts/mvrhoi_split_vidhoi_annotations.py`                        |
| Detector dataset conversion              | `hoi-prediction-gaze-transformer/convert_mvrhoi_to_yolo.py`                                              |
| Oracle gaze extraction                   | `hoi-prediction-gaze-transformer/misc_scripts/extract_mvrhoi_gaze.py`                                    |
| Detection, tracking, and gaze extraction | `hoi-prediction-gaze-transformer/misc_scripts/extract_mvrhoi_detection_gaze.py`                          |
| HOI fine-tuning                          | `hoi-prediction-gaze-transformer/finetune_mvrhoi.py`                                                     |
| Batch inference                          | `hoi-prediction-gaze-transformer/eval_inference_vidhoi.py`                                               |
| Metric evaluation                        | `hoi-prediction-gaze-transformer/eval_metrics_vidhoi.py`, `common/metrics.py`, `common/metrics_utils.py` |
| Qualitative inference                    | `hoi-prediction-gaze-transformer/run_mvrhoi.py`                                                          |
| Dataset verification                     | `hoi-prediction-gaze-transformer/verify_mvrhoi_dataset.py`                                               |

## Upstream dependencies and overlay setup

The full model pipeline depends on the upstream **STTran-Gaze** implementation and its dependencies. The release includes MVRHOI-specific scripts, detector conversion, and dataset verification, but it does **not** vendor all upstream components.

In particular, `common/`, `modules/` (including YOLOv5), `train_vidhoi.py`, and the batch evaluators remain upstream dependencies. The setup helper described in the [fine-tuning guide](../scripts/finetune.md) copies the MVRHOI overlay into an external upstream checkout.

The original training workspace contains local modifications. Compatibility of the released overlay with an **unmodified upstream revision** has not yet been independently verified. The exact historical upstream revision and all modified dependencies are not fully established; users should not assume that the upstream repository's latest branch recreates the paper's execution environment.

Preserving the original package layout is important for imports and relative paths. Machine-specific locations from the original workspace should be replaced with explicit local configuration when adapting the code.

## Source records and licensing

The repository's [`data/code_provenance.json`](../data/code_provenance.json) records copied source files and their digests. This provides an inventory of the included material, but does not independently prove the precise code revision used for every historical experiment.

Third-party software retains its applicable license obligations.

## Other training components

The Java **HOIGazeAnalyzer** application, MMFG exports/renderers, and PMD results are supplementary materials from the broader training workspace. They are **not required** to regenerate the archived quantitative tables distributed with the current paper companion.

The training pipeline overview contained an apparent credential and is not reproduced here. Historical training logs and configuration exports require appropriate review before any additional publication; this inventory intentionally does not disclose potential credentials.

## Related documentation

See [preprocessing](preprocessing.md) for data preparation, [experiment protocol](experiment_protocol.md) for the archived training and evaluation settings, and [reproduction](reproduction.md) for the currently supported workflows.
