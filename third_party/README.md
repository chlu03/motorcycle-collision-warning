# Third-Party Code

This folder contains code from other open-source projects. It is kept close to the
original so the project's own code in `src/` stays clearly separated.

| Folder | Source | License | Notes |
|---|---|---|---|
| `yolov5/` | [ultralytics/yolov5](https://github.com/ultralytics/yolov5) (v5.0, 2021) | GPL-3.0 | Object detection. Only the inference code is kept. |
| `deep_sort/` | [ZQPei/deep_sort_pytorch](https://github.com/ZQPei/deep_sort_pytorch) | MIT (see `deep_sort/LICENSE`) | Multi-object tracking. |

`yolov5/utils/distance.py` and the speed drawing in `yolov5/utils/plots.py` come from an
earlier open-source YOLOv5 + DeepSORT distance and speed estimation project. They are no
longer used by the pipeline; the current implementation is in `src/distance.py` and `src/speed.py`.

## Model weights (not included)

| File | Place it at |
|---|---|
| YOLOv5s COCO weights | `weights/yolov5s.pt` |
| DeepSORT ReID checkpoint `ckpt.t7` | `third_party/deep_sort/deep_sort/deep/checkpoint/ckpt.t7` |
