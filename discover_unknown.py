# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import os
import os.path as osp
import time
import json
import tqdm
import torch

from detectron2.structures import Boxes, Instances, BoxMode
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from core import add_config
from core.predictor import VisualizationDemo
from core.pascal_voc import register_pascal_voc

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/M-OWODB/t4_ft.yaml",
        metavar="FILE",
        help="path to config file",
    )
    # Either provide image ids on the command line or a text file with image ids (without extension)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '-i', "--input", nargs="+", help="A list of space separated input image ids (without extension)"
    )
    group.add_argument(
        '--input-txt', help="Path to a txt file where each line contains an image id (without extension)"
    )
    parser.add_argument(
        "--output",
        help="Path to output JSON file to save results. If not provided, only prints to console.",
    )
    parser.add_argument(
        "--task",
        help="Task being trained/evaluated for. <SuperSplit/tasksplit> format eg: M-OWOD/t2",
    )
    parser.add_argument(
        '-c',
        "--confidence-threshold",
        type=float,
        default=0.15,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=['MODEL.WEIGHTS', 'output/backup/model_0109999.pth'],
        nargs=argparse.REMAINDER,
    )
    parser.add_argument('-u', '--unknown', action='store_true', help='emphasize unknown color')
    return parser

if __name__ == "__main__":
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    
    super_task = args.task.split('/')[0]
    register_pascal_voc('my_selection_split', './datasets/', super_task, args.task, cfg)
    
    # Build a mapping from image id (without extension) to its ground truth annotations.
    dataset_dicts = DatasetCatalog.get("my_selection_split")
    gt_mapping = {}
    for d in dataset_dicts:
        # Extract the image id from the file name, e.g., "2008_000371" from "datasets/JPEGImages/2008_000371.jpg"
        image_id = osp.splitext(osp.basename(d["file_name"]))[0]
        gt_mapping[image_id] = d.get("annotations", [])

    miner = VisualizationDemo(cfg)

    # Get the list of image ids from command-line or text file
    if args.input_txt:
        with open(args.input_txt, "r") as f:
            image_ids = [line.strip() for line in f if line.strip()]
    else:
        image_ids = args.input

    results = {}
    for image_id in tqdm.tqdm(image_ids):
        # Build the image path assuming images are stored under 'datasets/JPEGImages'
        path = osp.join('datasets/JPEGImages', image_id + '.jpg')
        img = read_image(path, format="BGR")
        height, width = img.shape[:2]

        # Convert the saved ground truth annotations (if any) into an Instances object.
        gt_annos = gt_mapping.get(image_id, [])
        gt_instances = None
        if gt_annos:
            boxes, classes = [], []
            for anno in gt_annos:
                bbox = anno.get("bbox", [])
                # If bbox is in XYWH format (and if bbox_mode indicates so), convert it:
                if "bbox_mode" in anno and anno["bbox_mode"] == BoxMode.XYWH_ABS:
                    bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
                boxes.append(bbox)
                classes.append(anno.get("category_id"))
            if boxes:
                gt_boxes = Boxes(boxes)
                gt_instances = Instances((height, width))
                gt_instances.gt_boxes = gt_boxes
                gt_instances.gt_classes = torch.tensor(classes)

        # Pass both image and ground truth to the predictor.        
        predictions = miner.mine_unknown_per_image(img, gt_instances=gt_instances)
        
        if len(predictions["labels"]) > 0 :
            results[image_id] = predictions 

    # Convert results to JSON and print to console
    json_output = json.dumps(results, indent=4)
    # If an output path is provided, save the JSON there
    if args.output:
        with open(args.output, "w") as f:
            f.write(json_output)
