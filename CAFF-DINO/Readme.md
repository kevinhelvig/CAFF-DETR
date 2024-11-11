# Implementation of CAFF Fusion Module for DETR-DINO (DINO) Architecture :t-rex:

This repository contains an implementation of the CAFF (Cross-Attention Feature Fusion) module designed for the DETR-DINO architecture. The CAFF module aims to enhance feature fusion across different levels in the DETR-DINO model, improving performance on object detection tasks.

:warning:For training "from scratch" on the target datasets (LLVIP/FLIR/custom) you __must__ use pretrained weights (COCO mono-spectrum initialization) in order to have proper performance. You can find them in the original [DETR-DINO github page](https://github.com/IDEA-Research/DINO). 

__see the Legacy_README.md for the original mono-spectrum architecture details, hyper-parameters settings ...__ <br> 
<br>
* __[11/11/2024]__ : The original weights and logs after training are released: new/longer trainings should be added in the next future🧑‍🍳.
* __[11/10/2024]__ : __first release__, the code is as easy to manipulate as the other DETRs.   🖋️
  
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
We release the original training weights and log files (11 epochs for training) <br> 
* LLVIP - 11 epochs [[Checkpoint](https://zenodo.org/records/14065648/files/checkpoint_best_regular_CAFF-DINO_LLVIP-training_11epochs.pth?download=1)][[log](https://zenodo.org/records/14065648/files/log_CAFF-DINO_LLVIP-training_11epochs.txt?download=1)] <br>
* FLIR-aligned - 11 epochs [[Checkpoint](https://zenodo.org/records/14065648/files/checkpoint_best_regular_CAFF-DINO_FLIR-training_11epochs.pth?download=1)][[log](https://zenodo.org/records/14065648/files/log_CAFF-DINO_FLIR-training_11epochs.txt?download=1)]

We will add longer training results during the next weeks 🧑‍🍳. __Small variation of performance__ compared with the paper may be present.
* LLVIP - 31 epochs (not yet available) [[Checkpoint]()][[log]()] <br>
* FLIR-aligned - 31 epochs  (not yet available) [[Checkpoint]()][[log]()]
