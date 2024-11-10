# Implementation of CAFF Fusion Module for DETR-DINO (DINO) Architecture :t-rex:

This repository contains an implementation of the CAFF (Cross-Attention Feature Fusion) module designed for the DETR-DINO architecture. The CAFF module aims to enhance feature fusion across different levels in the DETR-DINO model, improving performance on object detection tasks.

:warning:For training "from scratch" on the target datasets (LLVIP/FLIR/custom) you __must__ use pretrained weights (COCO mono-spectrum initialization) in order to have proper performance. You can find them in the original [DETR-DINO github page](https://github.com/IDEA-Research/DINO). 

__see the Legacy_README.md for the original mono-spectrum architecture details, hyper-parameters settings ...__ <br> 
<br>

* __[11/10/2024]__ : __first release__, the code is as easy to manipulate as the other DETRs. Weights and logs incoming (new/longer trainings🧑‍🍳).  🖋️
  
## Overview :mag_right:

Links to the original mono-spectrum implementation <br>
* Github: [https://github.com/IDEA-Research/DINO](https://github.com/IDEA-Research/DINO) <br>
* Paper: [DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection](https://arxiv.org/abs/2203.03605)

## Installation :minidisc:

Clone the associated repository and install the required dependencies (identical to the original implementation)

```bash
cd ./CAFF-DINO
pip install -r requirements.txt
```

Run (train/inference) is comparable to the mono-spectrum DINO implementation.

 * __Example of training command ("from scratch" i.e. from COCO to LLVIP or FLIR)__: 
```bash
CUDA_VISIBLE_DEVICES=0  python3 ./CAFF-DINO/main.py -c ./CAFF-DINO/config/DINO/DINO_5scale_swin.py --dataset_file=flir_fusion --coco_path=./FLIR_aligned_coco --pretrain_model_coco=./coco_pretrained_weights/checkpoint0027_5scale_swin-001.pth
```

"pretrain_model_coco" is for coco-monospectrum weights init., which is __critical to obtain proper performance__. You need to download them from the original DETR-DINO implementation.

 * __Example of training command ("from LLVIP/FLIR" i.e. from LLVIP/FLIR to custom data)__: 
```bash
CUDA_VISIBLE_DEVICES=0  python3 ./CAFF-DINO/main.py -c ./CAFF-DINO/config/DINO/DINO_5scale_swin.py --dataset_file=flir_fusion --coco_path=./FLIR-style_dataset --pretrain_model_path=./flir_training_output/checkpoint_best_regular.pth
```

"pretrain_model_path" is for multi-spectral weights init. from LLVIP or FLIR to train on a custom limited dataset with a starting point (data needs to follow either "FLIR-coco" organization" or "LLVIP-coco", as described in the main page). A proper visualization code should be added.

## Weights :weight_lifting:
Due to issues with the preservation of previous trainings we are working on new trains of the proposed model. We will released the trained models for LLVIP and FLIR-aligned dataset (best mean average precision obtained), with the associated log files as soon as possible : __small variation of performance__ compared with the paper may be present. <br> 
* LLVIP (not yet available) [[Checkpoint]()][[log]()] <br>
* FLIR-aligned  (not yet available) [[Checkpoint]()][[log]()]
