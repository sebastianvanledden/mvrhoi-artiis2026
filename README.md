# Detection of Avatar Interactions in Metaverse Recordings

Companion repository for the ARTIIS 2026 paper by Sebastian van Ledden, Patrick Steinert, Stefan Wagenpfeil, and Matthias Hemmje.

The paper studies STTran-Gaze adaptation to MVRHOI based on the [256-MetaverseRecords Dataset](https://patricksteinert.de/256-metaverse-records-dataset/): 71 recordings and 595 annotated interaction instances, as stated in the paper. The archived full-model QPIC mAP is 0.6760 in Oracle Mode and 0.1450 in Detection Mode.

## Release scope

This repository includes annotations, taxonomy, label dictionaries and mappings, historical ego split memberships, MVRHOI preprocessing/training/inference extensions, detector conversion and verification scripts, archived configurations, and selected saved metrics. It is a research companion, not a verified end-to-end reproduction release.

The core pipeline and YOLOv5 implementation are dependencies of the [upstream STTran-Gaze repository](https://github.com/nizhf/hoi-prediction-gaze-transformer). Use the [fine-tuning guide](scripts/finetune.md) and its overlay setup instructions.

Recordings, extracted frames, model weights, and raw inference dumps are excluded from Git. Local copies may exist in ignored directories. See [external artifacts](artifacts/README.md) for the expected assets and release status.

## Start here

- [Dataset and split protocol](docs/dataset.md)
- [Annotation and taxonomy](docs/annotation.md)
- [Fine-tuning and detector workflow](scripts/finetune.md)
- [Reproduction routes](docs/reproduction.md)
- [Experiment details](docs/experiment_protocol.md)
- [Source inventory](docs/source_inventory.md)

## Regenerate the reported tables

Python 3.9 or newer is sufficient for the repository utilities. The validator also uses Git, when available, to inspect tracked and eligible untracked files while allowing ignored local artifacts.

```sh
python3 scripts/validate_repository.py
python3 scripts/generate_tables.py
```

The generator writes `results/tables/` from saved metrics. These commands do not recompute HOI metrics or validate model performance. In a source archive without Git metadata, the validator scans all files, so keep excluded local artifacts outside the archive.

## Repository contents

`data/` contains annotations and metadata, not recordings. `configs/` contains historical candidate configurations and the detector dataset template; see its [configuration guide](configs/README.md). `scripts/hoi-prediction-gaze-transformer/` contains the MVRHOI overlay for the upstream checkout. `results/experiments.csv` connects paper rows to archived metric exports. `docs/` records the protocol.

## Citation and licensing

Final citation details will be added when available.

This repository combines original research materials, adaptations of existing research, and code derived from third-party research. **No single license applies to every file.** The scope of each permission must be considered separately.

### Original source code

Original code authored specifically for this project will be released under the MIT License. This applies only to files for which the project author(s) hold the necessary rights. See [`licenses/MVRHOI_CODE_MIT.txt`](licenses/MVRHOI_CODE_MIT.txt).

### MVRHOI annotations and associated data

The newly authored MVRHOI annotation materials will be available under CC BY-NC 4.0. See [`licenses/MVRHOI_CC.txt`](licenses/MVRHOI_CC.txt).

The MVRHOI annotations were created using recordings from [256-MetaverseRecords Dataset](https://patricksteinert.de/256-metaverse-records-dataset/), which is distributed under CC BY-NC 4.0. The source recordings' attribution and noncommercial-use restrictions must be respected wherever they apply. A license for the underlying recordings does not automatically establish the ownership or licensing status of newly created annotations, mappings, or other files.

Original recordings and extracted frames are **not distributed** in this repository. See [dataset documentation](docs/dataset.md) and [external artifacts](artifacts/README.md).

### STTran-Gaze

The MVRHOI overlay in `scripts/hoi-prediction-gaze-transformer/` extends the [STTran-Gaze implementation](https://github.com/nizhf/hoi-prediction-gaze-transformer) by Zhifan Ni and collaborators.

The upstream source workspace identifies the license as **MIT**, with the notice **Copyright (c) 2023 nizhf**. The complete upstream license text is preserved in [`licenses/STTRAN_GAZE_MIT.txt`](licenses/STTRAN_GAZE_MIT.txt). This notice applies to the corresponding upstream-derived code, including relevant adaptations.

**Research reference:** Ni, Z., Valls Mascaró, E., Ahn, H. and Lee, D. (2023) 'Human–Object Interaction Prediction in Videos through Gaze Following', _Computer Vision and Image Understanding_, 233, 103741. DOI: 10.1016/j.cviu.2023.103741.

### External software dependencies

The upstream pipeline, YOLOv5, DeepSORT, and gaze-following implementations are external dependencies and are **not included** in this repository. Their licenses, notices, and any model-weight distribution conditions must be consulted at the respective upstream sources when installing or redistributing them.

The historical `requirements.txt` is an environment inventory, not a license grant.

### MVR256 source recordings

MVRHOI annotations are based on recordings from the [256-MetaverseRecords Dataset](https://patricksteinert.de/256-metaverse-records-dataset/). The source dataset is available under **Creative Commons Attribution–NonCommercial 4.0 International (CC BY-NC 4.0)**, requiring attribution and limiting covered reuse to noncommercial purposes.

**Source attribution:** Patrick Steinert, Stefan Wagenpfeil, and Matthias L. Hemmje (as identified in the MVR256 dataset license information). Consult the dataset's official citation and licensing information for the complete bibliographic reference and any additional conditions.

The recordings and extracted frames are **not included** in this repository.

### Taxonomy and annotation materials

The taxonomy adapts Manninen's interaction taxonomy as described in the accompanying paper.

**Research reference** Manninen, T. (2001) 'Rich Interaction in the Context of Networked Virtual Environments — Experiences Gained from the Multi-player Games Domain', _Joint Proceedings of HCI 2001 and IHM 2001 Conference_, 383.398. DOI: 10.1007/978-1-4471-0353-0_23.
