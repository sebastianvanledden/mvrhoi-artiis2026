# Fine-tuning STTran-Gaze on MVRHOI

This companion repository supplies MVRHOI scripts, metadata, and archived configurations as an overlay for [STTran-Gaze](https://github.com/nizhf/hoi-prediction-gaze-transformer). The core `common/`, `modules/`, training, and evaluation implementations remain in that repository. The overlay is not a standalone training package.

These commands describe the workflow. The implementation contains training changes (including backbone learning-rate scaling and backbone checkpoint export); upstream compatibility and the historical source revision still need verification. See [experiment protocol](../docs/experiment_protocol.md). Regenerating the archived result tables is independently supported.

## 1. Set up the pipeline checkout

From the companion repository root:

```sh
git clone --recurse-submodules https://github.com/nizhf/hoi-prediction-gaze-transformer.git upstream
python scripts/prepare_upstream.py --checkout upstream
python scripts/prepare_upstream.py --checkout upstream --apply
cd upstream
```

The preparation script copies the MVRHOI overlay, archived HOI configs, detector config, mapping files, and original VidHOI label dictionaries into the paths used below. It prints a plan by default. Once reviewed, repeat it with `--apply` to copy the files, then enter `upstream`. Existing files with different contents require `--overwrite`; use a dedicated checkout. It does not install dependencies, copy checkpoints, change the core upstream implementation, or fetch data.

Use the upstream installation instructions: the original environment uses Python 3.9, PyTorch 1.10, torchvision 0.11, and CUDA 11.3. The overlay's `requirements.txt` is a historical Windows environment snapshot, including Windows-only packages; it is not a portable lockfile. Start from the upstream environment and resolve platform-specific packages there. A 12 GB RTX 3060 was used for the reported runs.

All subsequent commands run from the **pipeline checkout root**, except the explicitly shown YOLOv5 commands. Commands are single lines and work in Windows Command Prompt or a POSIX shell. Change the example paths to your local locations. Use `python3` if your environment names Python that way.

Obtain the initialization and auxiliary weights described in [external artifacts](../artifacts/README.md). Missing assets may be downloaded by some upstream components, but automatic downloads are not guaranteed. The fine-tuned MVRHOI model is a different artifact from the VidHOI initialization checkpoint.

## 2. Convert raw annotations and frames

The converter reads `*_annotations_n.json` annotations and normalized videos. Inspect [preprocessing](../docs/preprocessing.md) before choosing source videos. The example assumes raw annotations and videos in separate directories:

```sh
python misc_scripts/mvrhoi_convert_mvrhoi_to_vidhoi.py --annotations-dir D:/MVRHOI_RAW/annotations --videos-dir D:/MVRHOI_RAW/videos --output-annotations D:/MVRHOI/VidHOI_annotation/mvrhoi_all_annots.json --output-frames-dir D:/MVRHOI/images --object-map mvrhoi_related/mvrhoi_object_mapping.json --object-default-label 1 --predicate-map mvrhoi_related/mvrhoi_predicate_mapping.json --predicate-default-label 0 --allow-ego-user
```

`--allow-ego-user` synthesizes a proxy person box when an interaction has no subject box. Omit it for a separately prepared no-ego variant. Existing exported frames are reused, so use a fresh output directory when changing frame preprocessing.

The original VidHOI vocabulary has **78 object classes and 50 predicates**. `data/labels_mvrhoi/` preserves the available prepared dataset's **30 objects and 48 predicates**. These dictionaries have different orderings and must not be interchanged. This example maps raw annotations into the original VidHOI vocabulary, using `data/labels_vidhoi/`; it does not reconstruct the prepared MVRHOI class ordering used by every archived result.

Mappings are copied from `data/mappings/` to `mvrhoi_related/` during setup. Their `target_id` values refer to the original VidHOI vocabulary. ID 0 is `person`; fallback object ID 1 is `car`. Explicit mappings take precedence over fallbacks, including the mapping of raw `default` to person. Inspect mapping approximations and fallback counts before training.

For **new exploratory runs**, the included splitter groups contiguous annotated frame ranges:

```sh
python misc_scripts/mvrhoi_split_vidhoi_annotations.py --annotations D:/MVRHOI/VidHOI_annotation/mvrhoi_all_annots.json --output-dir D:/MVRHOI/VidHOI_annotation --val-ratio 0.2 --seed 1234
```

For **paper reproduction**, retain the existing memberships from `data/splits/ego_train_frames.csv` and `ego_val_frames.csv`; filter converted entries by `(video_folder, video_id, frame_id)` instead of creating a new split. Reconcile class ordering and preprocessing with the archived dataset before claiming a matching experiment. Different ranges from the same video can occur in both splits; the historical split has no shared exact frames.

Copy `obj_categories.json`, `pred_categories.json`, and `pred_split_categories.json` from the pipeline's `vidhoi_related/` to `D:/MVRHOI/VidHOI_annotation/` for this original-vocabulary workflow. The expected layout is:

```text
D:/MVRHOI/
  images/<video_folder>/<video_id>/<video_id>_<frame_id>.jpg
  VidHOI_annotation/
    train_frame_annots.json
    val_frame_annots.json
    obj_categories.json
    pred_categories.json
    pred_split_categories.json
  VidHOI_detection/    # generated for detection mode
  VidHOI_gaze/         # generated gaze shelves
```

## 3. Train the object detector

The overlay includes `convert_mvrhoi_to_yolo.py`, `verify_mvrhoi_dataset.py`, and `configs/mvrhoi_vidhoi_yolov5.yaml`. YOLOv5 itself is provided by the recursively cloned upstream repository at `modules/object_tracking/yolov5/`.

The detector converter expects `<raw-root>/annotations/` and `<raw-root>/videos/`. It creates its own **video-level** random split (default seed 42, train ratio 0.8), which differs from the HOI frame-range split. The original detector split membership has not been independently established. These commands create a new detector dataset:

```sh
python convert_mvrhoi_to_yolo.py --raw-root D:/MVRHOI_RAW --output-dir D:/mvrhoi_yolo5v_to_vidhoi_dataset --object-map mvrhoi_related/mvrhoi_object_mapping.json --object-default-label 1 --seed 42 --train-ratio 0.8
python verify_mvrhoi_dataset.py --dataset-dir D:/mvrhoi_yolo5v_to_vidhoi_dataset --output-dir D:/mvrhoi_yolo5v_to_vidhoi_dataset/dataset_verification --num-samples 10
```

Edit `configs/mvrhoi_vidhoi_yolov5.yaml` so `path` matches that output directory. Keep `nc: 78` and its class ordering aligned with `vidhoi_related/obj_categories.json`. Then:

```sh
cd modules/object_tracking/yolov5
python train.py --data ../../../configs/mvrhoi_vidhoi_yolov5.yaml --weights ../../../weights/yolov5/vidor_yolov5l.pt --epochs 50 --batch 16 --img 640 --project ../../../runs/mvrhoi_finetune --name yolov5l_vidhoi_classes_finetune --workers 2
python val.py --data ../../../configs/mvrhoi_vidhoi_yolov5.yaml --weights ../../../runs/mvrhoi_finetune/yolov5l_vidhoi_classes_finetune/weights/best.pt --img 640
cd ../../..
```

Batch size 16 and 50 epochs match the saved detector options. Lower the batch size if necessary and record the change. Historical directory names contain `yolov5s`, while saved initialization options identify `yolov5l`; the new example uses `yolov5l` consistently. The archived `configs/detector_opt.yaml` preserves historical paths and is not an executable portable config.

## 4. Extract gaze and detection inputs

Ground-truth boxes (Oracle Mode):

```sh
python misc_scripts/extract_mvrhoi_gaze.py --mvrhoi_root D:/MVRHOI --split train
python misc_scripts/extract_mvrhoi_gaze.py --mvrhoi_root D:/MVRHOI --split val
```

The script writes zero heatmaps when head detection fails. Review its head matching threshold if changing the data.

Detection Mode, using the detector from step 3:

```sh
python misc_scripts/extract_mvrhoi_detection_gaze.py --mvrhoi_root D:/MVRHOI --split train --yolo_weights runs/mvrhoi_finetune/yolov5l_vidhoi_classes_finetune/weights/best.pt --detection_tag yolov5l_deepsort_mvrhoi_ft
python misc_scripts/extract_mvrhoi_detection_gaze.py --mvrhoi_root D:/MVRHOI --split val --yolo_weights runs/mvrhoi_finetune/yolov5l_vidhoi_classes_finetune/weights/best.pt --detection_tag yolov5l_deepsort_mvrhoi_ft
```

The detector trained above already emits VidHOI IDs; do not apply the raw MVRHOI-to-VidHOI map to those IDs again. The extractor's optional `--object-map` is for detectors whose outputs are still in a source label space. Confirm the checkpoint's class names first. Auxiliary weights are needed for head detection, gaze following, and DeepSORT.

## 5. Fine-tune STTran-Gaze

Setup copies the HOI configs from the companion's `configs/` into the pipeline's `configs/`. Review `batch_size`, `train_ratio`, `dim_transformer_ffn`, learning rates, and clip lengths. A checkpoint must match the model architecture and vocabulary. Runtime helper settings such as gradient accumulation and mixed precision are not necessarily consumed by the trainer.

Example Oracle training with original VidHOI initialization:

```sh
python finetune_mvrhoi.py --checkpoint weights/sttrangaze/f0_final.pt --data D:/MVRHOI --cfg configs/mvrhoi_finetune_rtx3060.yaml --project runs/mvrhoi_finetune --name sttran_mvrhoi --epochs 50 --gaze cross --global-token --disable-wandb
```

The example's 50 epochs are a workflow illustration. The paper describes 45 frozen-backbone epochs followed by 9 with selected backbone layers unfrozen; the exact transition checkpoint and effective settings remain to be verified.

For a second stage, pass the selected first-stage HOI checkpoint via `--checkpoint`, use `configs/mvrhoi_finetune_rtx3060_unfreeze_tail.yaml`, and add `--finetune-backbone`. If continuing from an already adapted backbone, also pass its matching `--backbone-checkpoint`. This wrapper initializes model weights; it does not restore optimizer/scheduler state as an exact training resume would.

The trainer uses `backbone_lr_mult` to scale backbone learning rates. Verify that the installed trainer consumes it and exports `backbone_last.pt`; the setup script does not patch the upstream trainer. Use `--strict-load` to require exact state-dictionary compatibility. Without it, mismatched tensors are skipped and the run is not a full checkpoint restoration.

The trainer saves epoch checkpoints and `last.pt` under `runs/mvrhoi_finetune/<name>/weights/`. A backbone checkpoint is exported when backbone fine-tuning is enabled. `best.pt` relies on an internal validation partition (`train_ratio < 1.0`); otherwise evaluate candidate epochs on the held-out split. Record that selection procedure and preserve the matching backbone. Logs include `train.log`, `metrics.csv`, `class_ap.csv`, and `train_losses.json`, depending on the trainer revision.

## 6. Evaluate

The batch evaluator and metric script come from the upstream repository. Confirm their supported options with `--help` for the installed revision. Example Oracle evaluation after a stage that exported a fine-tuned backbone:

```sh
python eval_inference_vidhoi.py runs/mvrhoi_finetune/sttran_mvrhoi/weights/last.pt --data D:/MVRHOI --cfg configs/eval_mvrhoi_f0.yaml --project runs/mvrhoi_finetune --name sttran_mvrhoi_eval --gaze cross --global-token --backbone runs/mvrhoi_finetune/sttran_mvrhoi/weights/backbone_last.pt
python eval_metrics_vidhoi.py runs/mvrhoi_finetune/sttran_mvrhoi_eval/eval/all_results.json --data D:/MVRHOI
```

For a frozen ImageNet backbone, use `--backbone weights/backbone` instead. For Detection Mode add `--detection yolov5l_deepsort_mvrhoi_ft` to inference and use its emitted result path for metrics. This is a value of `--detection`, not a standalone flag. Preserve the metric configuration and class dictionaries alongside the result; output CSV names and paths depend on the evaluator revision. The extra QPIC/per-class exports in the workspace may require its modified evaluator.

## 7. Qualitative inference

Place the adapted detector at `weights/yolov5/mvrhoi_yolov5l.pt` (the runner's expected filename). Use the final HOI checkpoint together with its matching backbone:

```sh
python run_mvrhoi.py --source path/to/video.mp4 --model weights/mvrhoi/last.pt --backbone weights/mvrhoi/backbone_last.pt --eval-cfg eval_mvrhoi_f0.yaml --weights-root weights --cfg-root configs --video --print
```

`--eval-cfg` is resolved relative to `--cfg-root`; do not repeat `configs/` in that argument. An absolute config path also works. This example requires the original VidHOI dictionaries expected by the runner; validate checkpoint/vocabulary compatibility before using an archived MVRHOI checkpoint. Outputs go to `output/<video_name>/` by default. Keep generated media, caches, runs, and checkpoints outside the companion commit.
