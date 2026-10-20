# Configuration files

All pipeline YAML configurations live in this directory. Historical settings are preserved for reference; their presence does not establish the exact effective configuration of every reported experiment.

| File | Role |
| --- | --- |
| `mvrhoi_vidhoi_yolov5.yaml` | Editable detector dataset template with the original 78-class VidHOI vocabulary. Change `path` to your converted dataset location. |
| `mvrhoi_finetune_rtx3060.yaml` | Historical candidate configuration for STTran-Gaze fine-tuning with a frozen backbone. |
| `mvrhoi_finetune_rtx3060_unfreeze_tail.yaml` | Historical candidate configuration for the backbone fine-tuning stage. |
| `eval_mvrhoi_f0.yaml` | Historical candidate configuration for model evaluation. |
| `metrics.yaml` | Archived metric settings, including IoU threshold and rare-class limit. |
| `detector_opt.yaml` | Saved detector-run options, including historical machine-specific paths. |
| `detector_hyp.yaml` | Saved detector-run hyperparameters. |

The [setup helper](../scripts/prepare_upstream.py) copies the detector dataset template and the HOI training, evaluation, and metric configurations to the upstream checkout's `configs/`. It leaves the saved `detector_opt.yaml` and `detector_hyp.yaml` as reference files here. Follow the [fine-tuning guide](../scripts/finetune.md) for commands and the [experiment protocol](../docs/experiment_protocol.md) for unresolved historical details.

`vidat_config/` contains the annotation-tool configuration and is separate from the pipeline YAML files. Source paths and checksums are recorded in `data/metadata_provenance.json` and `data/code_provenance.json` at the repository root.
