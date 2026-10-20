# Experimental Protocol and Evaluation

## Experiment overview

The repository's [`results/experiments.csv`](../results/experiments.csv) indexes the archived experiment entries and links them to saved metric exports. The available records cover pretrained **Oracle** and **Detection** baselines, the **adapted full model** in both modes, a **Detection-Mode variant without ego-user subjects**, and an adapted **Detection-Mode run without detector fine-tuning**.

The archived full-model results summarized in the companion README are:

| Evaluation setting                  | Archived QPIC mAP |
| ----------------------------------- | ----------------: |
| Adapted full model — Oracle Mode    |        **0.6760** |
| Adapted full model — Detection Mode |        **0.1450** |

These figures are read from the archived evaluation records. Matching a saved score does not, by itself, establish the identity of the source dataset, trained checkpoint, or evaluation code revision.

## HOI model adaptation

The paper describes STTran-Gaze adaptation on an **NVIDIA RTX 3060 with 12 GB VRAM** in two training stages: **45 epochs with the backbone frozen**, followed by **9 epochs with selected backbone layers unfrozen**.

The archived YAML files in [`configs/`](../configs/) document candidate historical settings. They should not all be interpreted as verified, executable commands for the reported runs. In particular, the precise transition checkpoint, unfrozen layer selection, effective runtime flags, random seed, optimizer, scheduler, and checkpoint-selection rule have not all been confirmed against original execution logs. Some helper settings may not be consumed by the training implementation.

## Object detector fine-tuning

The YOLOv5 detector was fine-tuned for **50 epochs** with **batch size 16**. These values are supported by the saved detector options, archived as [`configs/detector_opt.yaml`](../configs/detector_opt.yaml), and by the author's confirmation. An earlier paper draft specified batch size 8; **16 is the confirmed historical setting**.

Historical experiment directories include `yolov5s` in their names, while the saved options identify `vidor_yolov5l.pt` as the initial weights. Directory names alone are insufficient to establish the actual model architecture. The architecture and exact checkpoint provenance should be resolved from the checkpoint itself before making stronger claims.

## Evaluation configuration

The archived metric configuration specifies an **intersection-over-union (IoU) threshold of 0.5** and a **rare-class threshold of 25**. The available evaluator derives rare and non-rare categories from the counts in the **evaluated ground truth**, rather than from a separately fixed list based on training-set frequencies. This convention matters when comparing the reported QPIC-derived scores with other HOI evaluations.

## Interpretation and limitations

The archived training/validation split assigns **contiguous annotated ranges**, not entire recordings, to subsets. Recordings can occur in both subsets, although the documented prepared files contain no identical frame identifiers across subsets. Consequently, the archived scores should not be interpreted as demonstrating generalization to wholly unseen recordings.

The source annotations were not independently validated. In addition, approximate label mappings and the treatment of synthetic ego-user subjects may affect comparisons between experimental variants.

For a description of the data split, see [dataset documentation](dataset.md). For the available reproduction routes, see [reproduction guidance](reproduction.md).
