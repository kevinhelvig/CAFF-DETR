# khelvig
# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from torchvision
# ------------------------------------------------------------------------

"""
Copy-Paste from torchvision, but add utility of caching images on memory
"""
import torch 
from torchvision.datasets.vision import VisionDataset
from PIL import Image
import os
import os.path
import tqdm
from io import BytesIO


class CocoDetection(VisionDataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(self, root, annFile, transform=None, target_transform=None, transforms=None,
                 cache_mode=False, local_rank=0, local_size=1):
        super(CocoDetection, self).__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.cache_mode = cache_mode
        self.local_rank = local_rank
        self.local_size = local_size
        if cache_mode:
            self.cache = {}
            self.cache_images()

    def cache_images(self):
        self.cache = {}
        for index, img_id in zip(tqdm.trange(len(self.ids)), self.ids):
            if index % self.local_size != self.local_rank:
                continue
            path = self.coco.loadImgs(img_id)[0]['filename']
            with open(os.path.join(self.root, path), 'rb') as f:
                self.cache[path] = f.read()

    def get_image(self, path):
        if self.cache_mode:
            if path not in self.cache.keys():
                with open(os.path.join(self.root, path), 'rb') as f:
                    self.cache[path] = f.read()
            return Image.open(BytesIO(self.cache[path])).convert('RGB')
        return Image.open(os.path.join(self.root, path)).convert('RGB')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name_IR']
        img = self.get_image(path)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)

# %% Modification to capture both spectra 
import os
from typing import Any, Callable, List, Optional, Tuple

import torch
import torch.utils.data as data

from torchvision.utils import _log_api_usage_once

class CocoDetection_RGBT_FLIR(VisionDataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(self, root_rgb, root_thermal, ann_file, transform=None, target_transform=None, transforms=None,
                 cache_mode=False, local_rank=0, local_size=1):
        super(CocoDetection_RGBT, self).__init__(root_rgb, transforms, transform, target_transform)
        self.root_rgb = root_rgb
        self.root_thermal = root_thermal
        self.ann_file = ann_file 
        from pycocotools.coco import COCO
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.cache_mode = cache_mode
        self.local_rank = local_rank
        self.local_size = local_size
        
        if cache_mode:
            self.cache = {}
            self.cache_images()
        
    
    def cache_images(self):
        self.cache = {}
        for index, img_id in zip(tqdm.trange(len(self.ids)), self.ids):
            if index % self.local_size != self.local_rank:
                continue
            path_thermal = self.coco.loadImgs(img_id)[0]['file_name_IR']
            with open(os.path.join(self.root_thermal, path_thermal), 'rb') as f:
                self.cache[path_thermal] = f.read()
            path_rgb = self.coco.loadImgs(img_id)[0]['file_name_RGB']
            with open(os.path.join(self.root_rgb, path_rgb), 'rb') as f:
                self.cache[path_rgb] = f.read()
                
    def get_image(self, path, spectrum):
        if spectrum == 'ir': 
            if self.cache_mode:
                if path not in self.cache.keys():
                    with open(os.path.join(self.root_thermal, path), 'rb') as f:
                        self.cache[path] = f.read()
                return Image.open(BytesIO(self.cache[path])).convert('RGB')
            return Image.open(os.path.join(self.root_thermal, path)).convert('RGB')
        if spectrum == 'v': 
            if self.cache_mode:
                if path not in self.cache.keys():
                    with open(os.path.join(self.root_rgb, path), 'rb') as f:
                        self.cache[path] = f.read()
                return Image.open(BytesIO(self.cache[path])).convert('RGB')
            return Image.open(os.path.join(self.root_rgb, path)).convert('RGB')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path_thermal = coco.loadImgs(img_id)[0]['file_name_IR']
        path_RGB = coco.loadImgs(img_id)[0]['file_name_RGB']

        img_thermal = self.get_image(path_thermal, 'ir')
        img_RGB = self.get_image(path_RGB, 'v')

        if self.transforms is not None:
            img_RGB, img_thermal, target = self.transforms(img_RGB, img_thermal, target)

        return img_RGB, img_thermal, target

    def __len__(self):
        return len(self.ids)
        
class CocoDetection_RGBT_LLVIP(VisionDataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(self, root_rgb, root_thermal, ann_file, transform=None, target_transform=None, transforms=None,
                 cache_mode=False, local_rank=0, local_size=1):
        super(CocoDetection_RGBT_LLVIP, self).__init__(root_rgb, transforms, transform, target_transform)
        self.root_rgb = root_rgb
        self.root_thermal = root_thermal
        self.ann_file = ann_file 
        from pycocotools.coco import COCO
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.cache_mode = cache_mode
        self.local_rank = local_rank
        self.local_size = local_size
        
        if cache_mode:
            self.cache = {}
            self.cache_images()
        
    
    def cache_images(self):
        self.cache = {}
        for index, img_id in zip(tqdm.trange(len(self.ids)), self.ids):
            if index % self.local_size != self.local_rank:
                continue
            path_thermal = self.coco.loadImgs(img_id)[0]['file_name']
            with open(os.path.join(self.root_thermal, path_thermal), 'rb') as f:
                self.cache[path_thermal] = f.read()
            path_rgb = self.coco.loadImgs(img_id)[0]['file_name']
            with open(os.path.join(self.root_rgb, path_rgb), 'rb') as f:
                self.cache[path_rgb] = f.read()
                
    def get_image(self, path, spectrum):
        if spectrum == 'ir': 
            if self.cache_mode:
                if path not in self.cache.keys():
                    with open(os.path.join(self.root_thermal, path), 'rb') as f:
                        self.cache[path] = f.read()
                return Image.open(BytesIO(self.cache[path])).convert('RGB')
            return Image.open(os.path.join(self.root_thermal, path)).convert('RGB')
        if spectrum == 'v': 
            if self.cache_mode:
                if path not in self.cache.keys():
                    with open(os.path.join(self.root_rgb, path), 'rb') as f:
                        self.cache[path] = f.read()
                return Image.open(BytesIO(self.cache[path])).convert('RGB')
            return Image.open(os.path.join(self.root_rgb, path)).convert('RGB')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        #print(index)
        img_id = self.ids[index]
        #print(img_id)
        ann_ids = coco.getAnnIds(imgIds=img_id)
        #print(ann_ids) 
        target = coco.loadAnns(ann_ids)
        
        path_RGB = coco.loadImgs(img_id)[0]['file_name']
        path_thermal = coco.loadImgs(img_id)[0]['file_name']
        """
        path_RGB = coco.loadImgs(img_id)[0]['file_name']
        path_thermal = coco.loadImgs(img_id)[0]['file_name'] 
        """ 
        img_thermal = self.get_image(path_thermal, 'ir')
        img_RGB = self.get_image(path_RGB, 'v')

        if self.transforms is not None:
            img_RGB, img_thermal, target = self.transforms(img_RGB, img_thermal, target)

        return img_RGB, img_thermal, target

    def __len__(self):
        return len(self.ids)

