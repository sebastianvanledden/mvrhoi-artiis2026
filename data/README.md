# Included annotations and metadata

- `annotations/`: 60 original `*_annotations_n.json` files currently supplied with this companion; this file count does not establish coverage of all 71 recordings described in the paper.
- `annotation_guide/Annotation Guideline.pdf`: supplied annotation guideline, deliberately included as documentation.
- `taxonomy/`: hierarchical interaction taxonomy and its HTML visualization.
- `../configs/vidat_config/mvrhoi_config.json`: annotation-tool configuration.
- `mappings/`: raw MVRHOI-to-original-VidHOI object and predicate mappings.
- `labels_vidhoi/`: original VidHOI ordering (78 objects, 50 predicates).
- `labels_mvrhoi/`: available prepared MVRHOI ordering (30 objects, 48 predicates). Do not substitute these for the original vocabulary without reconciling IDs and checkpoints.
- `mvr_info/`: available video metadata.
- `splits/ego_train_frames.csv`, `ego_val_frames.csv`: historical unique frame memberships, including source video folder and ID.
- `splits/ego_split_summary.json`: annotation-entry counts and video/frame overlap summary.
- `metadata_provenance.json`: inventoried metadata source paths and SHA-256 digests; modified copies retain prior digests and notes.
- `code_provenance.json`: source paths and SHA-256 digests of the supplied pipeline overlay and detector config at initial packaging. These workspace sources do not identify a verified upstream commit.
