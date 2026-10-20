# Annotation, Taxonomy, and Label Mappings

## Annotation resources

MVRHOI represents interactions between avatars and objects or other entities. The released data includes interaction annotations, label dictionaries, mappings, and a hierarchical taxonomy. The available annotation guideline is provided as a PDF under [`data/annotation_guide/`](../data/annotation_guide/).

The source annotations describe interaction intervals; conversion scripts produce the frame-oriented inputs required by the HOI framework.

The annotation process was **not independently validated**, as stated in the paper. Consequently, the release should not be interpreted as a separately quality-audited benchmark.

The guideline PDF is the available reference for annotation practice.

The used configuration file for the Vidat annotation software is provided under ['configs/vidat_config/mvrhoi_config.json'](../configs/vidat_config/mvrhoi_config.json).

## Taxonomy representation

The hierarchical taxonomy JSON provided in [`data/taxonomy/`](../data/taxonomy/) is the experiment's taxonomy. As described in the paper it is an extended taxonomy based on Manninen's taxonomy of interaction types. It preserves the labels used in the annotations while distinguishing diagram-derived hierarchy nodes from extensions required by the annotation data.

With a python server you can visualize the taxonomy using the provided taxonomy_visualization.html.

## Mapping to the model label space

The repository preserves the available mappings from the annotation labels to the VidHOI-oriented model label space. These transformations are not always one-to-one. For example, avatar labels are mapped to the model's **`person`** category, while some metaverse-specific object types use approximate correspondences, such as mapping **menu** to a **screen/monitor** category.

Such approximations, as well as unmapped/default labels and annotations excluded during conversion, can affect both the training examples and interpretation of predictions. The model label dictionaries retain the ordering found in the available prepared dataset.

Historical mapping copies existed in `Annotation/Mapping/`, `mvrhoi_related/`, and the raw-data mapping directory. Reusers should preserve the released mappings alongside any derived dataset and record modifications explicitly.

## Usage considerations

For the archived experiments, treat the released taxonomy, mapping files, and label order as part of the experiment definition. Changes to category identities or mapping rules create a different label space and can invalidate direct metric comparisons.

See [dataset documentation](dataset.md) for recording composition and split details, [preprocessing](preprocessing.md) for the conversion workflow, and [experiment protocol](experiment_protocol.md) for evaluation constraints.
