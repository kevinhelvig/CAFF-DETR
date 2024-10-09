# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 14:45:33 2022
Adaptation of DETR implemented augmentation (COCO) to apply these on image pairs
@author: khelv
"""

import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision.transforms.functional import to_tensor, to_pil_image
from util.box_ops import box_xyxy_to_cxcywh
from util.misc import interpolate
import random 

def misalign_img(img, n):
    # Convertir l'image PIL en tenseur PyTorch
    img_tensor = to_tensor(img)
    
    # Décaler l'image vers le haut et vers la droite
    img_decalee = torch.roll(img_tensor, shifts=(-n, -n), dims=(1, 2))
    
    # Remplir les zones vides avec des zéros
    img_decalee[:, :n, :] = 0  # Pour le décalage vers le haut
    img_decalee[:, :, :n] = 0  # Pour le décalage vers la droite

    # Reconvertir le tenseur décalé en image PIL
    img_decalee_pil = to_pil_image(img_decalee)

    return img_decalee_pil


def crop(image_vis, image_ir, target, region):
    cropped_image_vis, cropped_image_ir = F.crop(image_vis, *region), F.crop(image_ir, *region)
    target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd"]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    if "masks" in target:
        # FIXME should we update the area here if there are no boxes?
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append("masks")

    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target or "masks" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "boxes" in target:
            cropped_boxes = target['boxes'].reshape(-1, 2, 2)
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        else:
            keep = target['masks'].flatten(1).any(1)

        for field in fields:
            target[field] = target[field][keep]

    return cropped_image_vis, cropped_image_ir, target


def hflip(image_vis, image_ir, target):
    flipped_image_vis, flipped_image_ir = F.hflip(image_vis), F.hflip(image_ir)

    w, h = image_vis.size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    if "masks" in target:
        target['masks'] = target['masks'].flip(-1)

    return flipped_image_vis, flipped_image_ir, target


def resize(image_vis, image_ir, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image_vis.size, size, max_size)
    rescaled_image_vis = F.resize(image_vis, size)
    rescaled_image_ir = F.resize(image_ir, size)

    if target is None:
        return rescaled_image_vis, rescaled_image_ir, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image_vis.size, image_vis.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target['masks'] = interpolate(
            target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    return rescaled_image_vis, rescaled_image_ir, target


def pad(image_vis, image_ir, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_image_vis = F.pad(image_vis, (0, 0, padding[0], padding[1]))
    padded_image_ir = F.pad(image_ir, (0, 0, padding[0], padding[1]))

    if target is None:
        return padded_image_vis, padded_image_ir, None
    target = target.copy()
    # should we do something wrt the original size?
    target["size"] = torch.tensor(padded_image_vis[::-1])
    if "masks" in target:
        target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[0], 0, padding[1]))
    return padded_image_vis, padded_image_ir, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img_vis, img_ir, target):
        region = T.RandomCrop.get_params(img_vis, self.size)
        return crop(img_vis, img_ir, target, region)


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img_vis: PIL.Image.Image, img_ir: PIL.Image.Image, target: dict):
        w = random.randint(self.min_size, min(img_vis.width, self.max_size)) # Convention: visible as reference
        h = random.randint(self.min_size, min(img_vis.height, self.max_size))
        region = T.RandomCrop.get_params(img_vis, [h, w])
        return crop(img_vis, img_ir, target, region)


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img_vis, img_ir, target):
        image_width, image_height = img_vis.size  # Convention: visible as reference
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(img_vis, img_ir, target, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img_vis, img_ir, target):
        if random.random() < self.p:
            return hflip(img_vis, img_ir, target)
        return img_vis, img_ir, target


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img_vis, img_ir, target=None):
        size = random.choice(self.sizes)
        return resize(img_vis, img_ir, target, size, self.max_size)


class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img_vis, img_ir, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img_vis, img_ir, target, (pad_x, pad_y))


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img_vis, img_ir, target):
        if random.random() < self.p:
            return self.transforms1(img_vis, img_ir, target)
        return self.transforms2(img_vis, img_ir, target)


class ToTensor(object):
    def __call__(self, img_vis, img_ir, target):
        return F.to_tensor(img_vis), F.to_tensor(img_ir), target


class RandomErasing(object):

    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img_vis, img_ir, target):
        return self.eraser(img_vis), self.eraser(img_ir), target


class Misalign_img(object):
    def __init__(self, n=50):
        self.n = n

    def __call__(self, img_vis, img_ir, target):
        return misalign_img(img_vis, self.n), img_ir, target    
        
        
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image_vis, image_ir, target=None):
        image_vis = F.normalize(image_vis, mean=self.mean, std=self.std)
        image_ir = F.normalize(image_ir, mean=self.mean, std=self.std)

        if target is None:
            return image_vis, image_ir, None
        target = target.copy()
        h, w = image_vis.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        return image_vis, image_ir, target

class ResizeDebug(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        return resize(img_vis, img_ir, target, self.size)

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image_vis, image_ir, target):
        for t in self.transforms:
            image_vis, image_ir, target = t(image_vis, image_ir, target)
        return image_vis, image_ir, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string




"""
class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: PIL.Image.Image, target: dict):
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        region = T.RandomCrop.get_params(img, [h, w])
        return crop(img, target, region)
"""
"""
class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))
"""
"""
class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target
"""
"""
class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)
"""
"""
class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))
"""
"""
class RandomSelect(object):
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)
"""
"""
class ToTensor(object):
    def __call__(self, img, target):
        return F.to_tensor(img), target
"""
"""
class RandomErasing(object):

    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, target):
        return self.eraser(img), target
"""
"""
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        return image, target
"""
"""
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
"""
