# External data and model artifacts

Recordings, extracted frames, weights, and generated inference dumps are excluded from Git.

## Assets expected in the pipeline checkout

| Artifact                          | Expected location                                    | Purpose                                                |
| --------------------------------- | ---------------------------------------------------- | ------------------------------------------------------ |
| VidHOI STTran-Gaze initialization | `weights/sttrangaze/f0_final.pt`                     | Start adaptation; distinct from the final MVRHOI model |
| Original VidOR YOLOv5l            | `weights/yolov5/vidor_yolov5l.pt`                    | Initialize detector training                           |
| ResNet backbone                   | `weights/backbone/`                                  | ImageNet feature initialization; follow upstream setup |
| Semantic word vectors             | `weights/semantic/`                                  | STTran semantic embeddings                             |
| CrowdHuman head detector          | `weights/yolov5/crowdhuman_yolov5m.pt`               | Head detection for gaze extraction                     |
| Gaze-following model              | `weights/detecting_attended/model_videoatttarget.pt` | Gaze heatmaps                                          |
| DeepSORT assets                   | `weights/deep_sort/`                                 | Detection-mode tracking                                |
| Adapted HOI model                 | `weights/mvrhoi/last.pt`                             | Final MVRHOI inference                                 |
| Matching adapted backbone         | `weights/mvrhoi/backbone_last.pt`                    | Must match the HOI model/stage                         |
| Adapted detector                  | `weights/yolov5/mvrhoi_yolov5l.pt`                   | Final MVRHOI detection                                 |

Consult the [upstream installation and weight instructions](https://github.com/nizhf/hoi-prediction-gaze-transformer#inference-on-an-arbitrary-video) for upstream download sources. Some components download assets automatically; verify successful retrieval and exact versions. Recordings and frames must be obtained separately; see [dataset access](../docs/dataset.md).
