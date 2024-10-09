# Implementation of CAFF Fusion Module for Lite-DINO Architecture :lizard:

This repository contains an implementation of the CAFF (Cross-Attention Feature Fusion) module designed for the Lite-DINO architecture. The CAFF module aims to enhance feature fusion across different levels in the Lite-DINO model, improving performance on object detection tasks.

:warning: Please note that the proposed model is adapted for Resnet-50 features extraction, then following the fusion module settings described in the paper for this backbone.

The lite-DINO architecture is lighter and faster in inference than the DETR-DINO approach, while providing reasonable detection performance. 
__see Legacy_README.md for the original mono-spectrum architecture details, hyper-parameters settings ...__ <br> 
<br>
[10/09/2024] : __first release__ of the model, weights are coming. __The script needs several cleaning__ (less hacks, proper functions and classes to load each dataset properly => __incoming in the next weeks__  :lizard:

## Overview :mag_right:

Link to the original mono-spectrum implementation <br>
Github: [https://github.com/IDEA-Research/Lite-DETR](https://github.com/IDEA-Research/Lite-DETR) <br>
Paper: [Lite DETR : An Interleaved Multi-Scale Encoder for Efficient DETR](https://arxiv.org/pdf/2303.07335.pdf)

## Installation :minidisc:

Clone the associated repository and install the required dependencies (identical to the original implementation)

```bash
cd ./CAFF-DETR/CAFF-Lite-DINO
pip install -r requirements.txt
```

Run (train/inference) is comparable to the CAFF-DINO implementation. Need several manual modification to run depending of the input data (__code cleaning is planned__). Read scripts (main.py) for details on how to run properly. 

## Weights :weight_lifting:
We released the trained models for LLVIP and FLIR-aligned dataset (best mean average precision obtained), with the associated log files. <br> 
* LLVIP [[Checkpoint](https://zenodo.org/records/13908222/files/checkpoint_best_CAFF-LITE-DINO_LLVIP.pth?download=1)][[log](https://zenodo.org/records/13908222/files/log_LLVIP_CAFF-LITE-DINO.txt?download=1)] <br>
* FLIR-aligned [[Checkpoint](https://zenodo.org/records/13908222/files/checkpoint_best_CAFF-LITE-DINO_FLIR.pth?download=1)][[log](https://zenodo.org/records/13908222/files/log_FLIR-Aligned_CAFF-LITE-DINO.txt?download=1)]
