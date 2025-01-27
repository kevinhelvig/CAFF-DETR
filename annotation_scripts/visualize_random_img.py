#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 2025

@author: khelvig
"""

import os
import json
import random
import argparse
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plot_coco_annotations(image_path, annotations, class_ids, category_map, image_path_secondary=None):
    """Plot COCO annotations on one or two images."""
    # Charger la première image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Charger la seconde image si nécessaire
    if image_path_secondary:
        img_secondary = cv2.imread(image_path_secondary)
        if img_secondary is None:
            raise FileNotFoundError(f"Image not found: {image_path_secondary}")
        img_secondary = cv2.cvtColor(img_secondary, cv2.COLOR_BGR2RGB)

        # Configuration des sous-graphes
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        axes[0].imshow(img)
        axes[0].axis('off')
        axes[1].imshow(img_secondary)
        axes[1].axis('off')

        # Ajouter les annotations aux deux images
        for ax, image in zip(axes, [img, img_secondary]):
            for bbox, class_id in zip(annotations, class_ids):
                x, y, w, h = bbox
                rect = Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
                label = category_map.get(class_id, f"Class {class_id}")
                ax.text(x, y - 10, label, color='red', fontsize=10, weight='bold')

    else:
        # Affichage pour une seule image
        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.axis('off')
        ax = plt.gca()

        # Ajouter les annotations à l'image
        for bbox, class_id in zip(annotations, class_ids):
            x, y, w, h = bbox
            rect = Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            label = category_map.get(class_id, f"Class {class_id}")
            ax.text(x, y - 10, label, color='red', fontsize=10, weight='bold')

    plt.show()


def load_coco_annotations(annotation_file, images_dir):
    """Load COCO annotations and return a random image with its annotations."""
    with open(annotation_file, 'r') as f:
        coco_data = json.load(f)

    images = coco_data['images']
    annotations = coco_data['annotations']
    categories = coco_data['categories']

    # Create a category map
    category_map = {cat['id']: cat['name'] for cat in categories}

    # Select a random image
    random_image = random.choice(images)
    image_id = random_image['id']
    image_path = os.path.join(images_dir, random_image['file_name'])

    # Get annotations for this image
    image_annotations = [ann for ann in annotations if ann['image_id'] == image_id]
    bboxes = [ann['bbox'] for ann in image_annotations]
    class_ids = [ann['category_id'] for ann in image_annotations]

    return image_path, bboxes, class_ids, category_map, random_image['file_name']


def main():
    parser = argparse.ArgumentParser(description="Visualize COCO annotations on an image or a pair of images.")
    parser.add_argument("--images_dir", type=str, required=True, help="Directory containing the images.")
    parser.add_argument("--annotation_file", type=str, required=True, help="COCO annotations JSON file.")
    parser.add_argument("--mode", type=str, choices=["mono", "multi"], default="mono",
                        help="Display mode: 'mono' for single image, 'multi' for paired images.")
    parser.add_argument("--secondary_images_dir", type=str,
                        help="Directory for secondary images (required for 'multi' mode).")
    args = parser.parse_args()

    # Load a random image and its annotations
    image_path, bboxes, class_ids, category_map, file_name = load_coco_annotations(args.annotation_file, args.images_dir)

    if args.mode == "multi":
        if not args.secondary_images_dir:
            raise ValueError("Secondary images directory must be specified in 'multi' mode.")
        # Find the corresponding image in the secondary directory
        image_path_secondary = os.path.join(args.secondary_images_dir, file_name)
        if not os.path.exists(image_path_secondary):
            raise FileNotFoundError(f"Secondary image not found: {image_path_secondary}")

        # Plot the annotations on both images
        plot_coco_annotations(image_path, bboxes, class_ids, category_map, image_path_secondary)
    else:
        # Plot the annotations on the single image
        plot_coco_annotations(image_path, bboxes, class_ids, category_map)


if __name__ == "__main__":
    main()



