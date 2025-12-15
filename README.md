# Combinatorial Open-World Detection

## Requirements
- Linux or macOS with Python ≥ 3.8.
- Install [PyTorch ≥ 1.9.0, torchvision](https://pytorch.org/#install),
  [Detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html),
  timm, and einops.
- Prepare datasets:
  - Download [COCO](https://cocodataset.org/#download) and [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/).
  - Convert annotation format using `coco_to_voc.py`.
  - Move all images to `datasets/JPEGImages` and annotations to `datasets/Annotations`.
  - Additional instructions are available [here](https://github.com/JosephKJ/OWOD/issues/59#issuecomment-897747744).

## Getting Started
* Training for open world object detection:
  ```
  bash run_owod.sh
  ```
  Evaluation for open world object detection:
  ```
  bash test_owod.sh
  ```
* Visualize the results:
  ```
  python demo.py -i LIST_OF_IMAGES
  ```
* Note that we are using an ImageNet pre-trained backbone. 
* We aggregate experiments from several randomized runs, weights from one such run is available [here](https://drive.google.com/file/d/1SXc8hj9_W1N8JKq4voZ6rrOhMIef6kP4/view?usp=sharing).

## Acknowledgement

Our implementation is based on [OrthogonalDet](https://github.com/feifeiobama/OrthogonalDet), [RandBox](https://github.com/scuwyh2000/RandBox) among others which uses [Detectron2](https://github.com/facebookresearch/detectron2) and [Sparse R-CNN](https://github.com/PeizeSun/SparseR-CNN).
