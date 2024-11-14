# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision

from .coco import build as build_coco
from .coco import  build_RGBT_FLIR, build_RGBT_LLVIP

def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco
    return dataset.coco


def build_dataset(image_set, args):
    print(args.dataset_file) 
    if args.dataset_file == 'flir_fusion':
        return build_RGBT_FLIR(image_set, args)
    if args.dataset_file == 'llvip_fusion':
        return build_RGBT_LLVIP(image_set, args)  
          
    if args.dataset_file == 'coco_panoptic':
        # to avoid making panopticapi required for coco
        from .coco_panoptic import build as build_coco_panoptic
        return build_coco_panoptic(image_set, args)
    if args.dataset_file == 'o365':
        from .o365 import build_o365_combine
        return build_o365_combine(image_set, args)
    if args.dataset_file == 'vanke':
        from .vanke import build_vanke
        return build_vanke(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
