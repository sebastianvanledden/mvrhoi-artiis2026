# MVRHOI Dataset and Access

## Dataset overview

MVRHOI is a research dataset of annotated avatar interactions derived from third-person metaverse recordings. The accompanying ARTIIS paper describes **71 recordings** and **595 annotated human–object interaction (HOI) instances** across four environments.

| Environment       | Recordings |
| ----------------- | ---------: |
| Roblox            |         20 |
| Spatial           |         20 |
| World of Warcraft |         11 |
| Fortnite          |         20 |
| **Total**         |     **71** |

The recording count, interaction-instance count, number of distinct frames, and number of frame-level annotation entries represent different quantities and should not be used interchangeably. The source recordings originate from the [256-MetaverseRecords Dataset](https://patricksteinert.de/256-metaverse-records-dataset/); the released MVRHOI materials provide derived annotations and supporting metadata rather than copies of the recordings.

## Included data and external recordings

The repository's [`data/`](../data/) directory contains the available annotations, label dictionaries, taxonomy, mappings, and archived split information. Original video recordings and extracted frames are **not distributed** in this repository. Their availability and any known access conditions are described in [`artifacts/README.md`](../artifacts/README.md).

An independently verified manifest connecting every released annotation to a stable source-recording identifier, checksum, duration, resolution, and preprocessing version is not part of the documented release. Accordingly, access to precisely the same raw inputs used in every historical experiment is not currently guaranteed.

MVRHOI builds on source material distributed under **CC BY-NC 4.0**. The source dataset's attribution and noncommercial conditions remain relevant to the underlying recordings. The licensing status of the newly authored annotations and taxonomy adaptations is addressed separately in the repository's [licensing section](../README.md#licensing-and-attribution).

## Archived training/validation split

The available splitter groups **contiguous annotated frame ranges** and assigns those ranges to training or validation. Its documented settings are a random **seed of `1234`** and a target validation fraction of **`0.2`**.

This is a **range-level split**, not a video-disjoint split. In the available prepared dataset with ego-user annotations, the saved memberships correspond to the following counts:

| Property                               | Archived count |
| -------------------------------------- | -------------: |
| Training annotation entries            |         19,860 |
| Validation annotation entries          |          4,987 |
| Recordings represented in training     |             55 |
| Recordings represented in validation   |             34 |
| Recordings represented in both subsets |             30 |

No exact frame identifiers overlap between the two subsets in these prepared files. Nevertheless, frames from the **same recording** can occur in both training and validation. The results therefore assess held-out annotated ranges rather than generalization to entirely unseen recordings. The relationship between these prepared split files and **every** historical experiment has not been established conclusively.

For comparisons with the archived results, use the **saved split memberships** rather than regenerating a random split. A video-disjoint split would constitute a different evaluation protocol and should be reported separately.

## Related documentation

See [annotation and taxonomy](annotation.md) for label interpretation, [preprocessing](preprocessing.md) for annotation conversion and splitting, and [experiment protocol](experiment_protocol.md) for the evaluation settings and their limitations.
