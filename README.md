# CAFF-DETR
Repository for the paper __"CAFF-DINO: Multi-spectral object detection transformers with cross-attention features fusion" [Helvig et al.]__, accepted in the __20 <sup> th </sup> IEEE Workshop Perception Beyond the visible Spectrum__, taking part of the CVPR 2024 conference. The work explores the adaptation of DETRs architectures for backbone features fusion on IR-visible data, through cross-attention fusion.

# Road-Map :construction: 
- __Best architecture released soon__ : CAFF-DINO :t-rex: 
- Several __other adaptations of DEtection TRansformers (DETR, H-DETR) released later__ :robot: 
- Annotation files for LLVIP and FLIR dataset, converted in COCO format :notebook_with_decorative_cover:

# Demo animations :movie_camera: 
<figure>
<p align="center">
  <img src="illustrations\pairs_dino_testset_flir2-ezgif.com-optimize.gif" alt="Alt Text 1">
  <img src="illustrations\pairs_dino_testset_llvip_v2.gif" alt="Alt Text 2">
</p>
<figcaption style="text-align: center; font-style: italic;"> <p> <i> Examples of multi-spectral object detections using our CAFF-DINO architecture, on FLIR then LLVIP. Labels are in red, while model's detections are in blue. 
Object Detection's confidence threshold is 50 %.</i> </p> </figcaption>
</figure>

# Core principles :bulb: 

The core principles of the proposed fusion are described in the illustrations bellow. See the paper for more theoretical stuffs. 

<figure>
<p align="center">
  <img src="illustrations\fusion_DETR_globalpathway.png" alt="Alt Text 1" width="500" height="500" >
</p>
<figcaption style="text-align: center; font-style: italic;"> <p> <i> Illustration of the global detection transformer model : the proposed fusion approach inserts feature fusion operation (CAFF module) at each level of monospectral backbones, merging both modalities.</i> </p> </figcaption>
</figure>

<figure>
<p align="center">
  <img src="illustrations\Hcaff_module_schematics.png" alt="Alt Text 1" width="500" height="500">
</p>
<figcaption style="text-align: center; font-style: italic;"> <p> <i> Illustration of the features fusion module. The main idea is to use a cross-attention operation to extract meaningful correlations between both spectra's features, as a complementary information, before a convolutional features fusion layer.</i> </p> </figcaption>
</figure>

# Annotations :bookmark_tabs: 
-- TO DO -- 

The annotations files are adapted from the standard MS-COCO format. A key "image_IR" is added, to load the correspondant thermal image. 

# Use :rocket: 
-- TO DO -- 

* Requirements install : identical to DETR-DINO original implementation
  ````
  cd ./CAFF-DINO
  pip install -r requirements.txt
  ````
* Training from scratch
  You need to donwload the pretrained mono-spectra DETR-DINO in the proper folder (pretrained)
  Example of training command (LINUX system : might need code adaptation to work with Slurm)
  ````
  CUDA_VISIBLE_DEVICES=0  python ./CAFF-DINO/main.py -c /d/khelvig/DINO-LLVIP/DINO_twostreams/config/DINO/DINO_5scale_swin.py --dataset_file 'fusion' --coco_path = ./dataset_files/LLVIP --pretrain_model_path ./pretrained/checkpoint0027_5scale_swin-001.pth --output_dir ./output_files
  ````
* Fine-tune a pretrained one's
  You may fine-tune a pretrained model, for specific, smaller fusion datasets. 
  Our pretrained models can be donwloaded here: LLVIP and FLIR. Let replace args --pretrained_model_path by args --resume to make it properly.

# Reference to prior work :bookmark: 
- IR-Visible fusion model using attention operation for YOLO-v5 head: [CFT-YOLO v5](https://github.com/DocF/multispectral-object-detection).
* DETR-DINO original architecture [here](https://github.com/IDEA-Research/DINO).

# Cite :closed_book: 
If the proposed fusion architecture is used for academic purpose, please consider citing our work: 

```
to complete
```

