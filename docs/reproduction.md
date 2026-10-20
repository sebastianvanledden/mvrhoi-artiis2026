# Reproducibility Guide

## Scope

This repository is an **research companion release**. It supports regenerating the paper's result tables from archived metric exports. Recomputing metrics from predictions, rerunning inference, and repeating training require additional external inputs or historical implementation details that have not been verified as a complete reproduction package.

The following routes are ordered from the currently supported operation to the more demanding research workflows.

## Regenerate tables from archived metrics

**Status: available.**

Run the following commands from the repository root using **Python 3.9 or newer**:

```bash
python3 scripts/validate_repository.py
python3 scripts/generate_tables.py
```

The generator reads archived `all_metrics.csv` files referenced by [`results/experiments.csv`](../results/experiments.csv) and writes CSV and Markdown summaries to **`results/tables/`**, with QPIC values formatted to four decimal places.

The repository validator can use Git metadata to distinguish tracked and eligible untracked files from ignored local assets. In source archives without Git metadata, it scans all files; keep excluded recordings, weights, and other large local artifacts outside such archives.

**These commands regenerate tables from existing scores. They do not rerun inference, recompute HOI metrics from raw predictions, or independently validate model performance.**

## Recompute metrics from predictions

**Status: not packaged as a verified workflow.**

Metric recomputation requires the upstream evaluator, any experiment-specific modifications, its dependencies, full inference prediction JSON, matching label dictionaries, and a verified metric configuration. The original prediction dumps are excluded from this repository.

The historical entry point is `eval_metrics_vidhoi.py`, with related metric implementations under `common/metrics.py` and `common/metrics_utils.py`. An exact command for each experiment cannot be provided reliably without confirming the evaluator revision and input provenance.

See [experiment protocol](experiment_protocol.md) for the documented IoU and rare-class settings.

## Rerun inference

Inference requires legally accessible recordings or appropriately prepared frame inputs, matching annotation and split metadata, the HOI model checkpoint and corresponding fine-tuned backbone, detector and gaze/tracking dependencies, and a compatible upstream STTran-Gaze environment with the MVRHOI overlay.

The required recordings, extracted frames, model weights, and raw prediction files are not distributed in this repository. No verified public artifact URLs for these assets have been established in the available documentation. Consult [`artifacts/README.md`](../artifacts/README.md) for their known release status.

## Repeat model training

Training requires the annotated data and split memberships, detector and STTran initialization weights, preprocessing and training dependencies, historical runtime settings, and suitable GPU resources.

The [fine-tuning guide](../scripts/finetune.md) documents the available overlay setup and illustrative commands. The paper describes **45 + 9 epochs** of staged HOI adaptation and **50 epochs** of detector fine-tuning with **batch size 16**. However, archived configurations are research records rather than fully verified one-command reproduction recipes. Some checkpoint, implementation, and execution details remain unresolved.

## Comparability of new runs

To compare new results with the archived paper rows, preserve the released label order, mapping conventions, ego-user configuration, and split memberships. Replacing missing checkpoints with unrelated weights, modifying the label mapping, or generating a new random split produces a **new experiment**, not an exact reproduction of the corresponding archived row.

For a task-to-source mapping, see the [source inventory](source_inventory.md). The [dataset](dataset.md), [annotation](annotation.md), and [preprocessing](preprocessing.md) documents describe further assumptions affecting comparability.
