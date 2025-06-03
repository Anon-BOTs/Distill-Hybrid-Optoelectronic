<div align='center'>

# <p><strong><em>PHOENIX:</em></strong></p> Photonic Distillation Transfers Electronic Knowledge to Hybrid Optical Neural Networks
Authors
</div>

## Introduction

💥PHOENIX presented the first optoelectrically fused neural network deployment framework for object detection tasks, demonstrating its performance in industrial-level large datasets (e.g., COCO) and benchmark models. 
Compared to state-of-the-art electronic models, our solution achieved approximately 85.0% accuracy. The accuracy was further improved to 93.0% through our novel knowledge distillation strategy. 
Furthermore, we achieved 86.5% energy reduction and 11.3× speed acceleration compared to equivalent edge GPUs by successfully transferring spatial attention knowledge from the electronic domain to the photonic domain, 
making it an ideal choice for real-time, energy-critical industrial applications. This technique not only bridges the performance gap but also offers an alternative physically interpretable platform for AI. Our universal framework paves the way for extending ONN deployment to a wider range of deep learning models and applications, whether based on CNN or Transformer architectures, providing a compelling choice for real-time, energy-critical scenarios such as autonomous driving, smart surveillance, and industrial automation.💥
<p align="center">
  <img src="asserts/intro.png" alt="pipeline" width="1000"/>
</p>

## Overview
💥 Overview of the proposed hybrid photoelectronic object detection framework, ***PHOENIX***.
The system is built upon a state-of-the-art baseline detector, where early-stage extracts low/mid-level
features by the CNN-based or Transformer-based method. The key component of the knowledge
distillation module transfers all-electronic ’teacher’ backbone to the ONN ’student’ stages, enhancing
their functional capabilities. Finally, features output by the ONN-processed segment of the backbone
are fed to a task-specific detection head for object classification and localization.

<p align="center">
  <img src="asserts/framework.png" alt="pipeline" width="1000"/>
</p> 

## Details architecture of ONN and Distillation module.

<div style="display: flex; justify-content: space-between;">
    <img src="asserts/onn_arch.png" alt="left" style="margin-right: auto;" width="45%" height="350" />
    <img src="asserts/distill.png" alt="right" style="margin-left: auto;" width="45%" height="450"/>
</div>

## Qualitative Rendered Visualization
### ONN experimental architecture and the Distribution of the Diffracted Field simulated in FDTD.
<p align="center">
  <img src="asserts/onn_vis.png" alt="pipeline" width="1000"/>
</p> 

### Detection performance on MS-COCO dataset
<p align="center">
  <img src="asserts/vis_coco.png" alt="pipeline" width="1000"/>
</p> 

### Detection performance on nuScenes dataset
<p align="center">
  <img src="asserts/vis_nus.png" alt="pipeline" width="1000"/>
</p> 

## Main Results
### Object Detection Performance on MS-COCO dataset
![img.png](asserts/main_table_coco.png)
### Object Detection Performance on MS-COCO on nuScenes dataset
![img.png](asserts/main_table_nus.png)
### Ablation studies.
<div style="display: flex; justify-content: space-between;">
    <img src="asserts/img.png" alt="left" style="margin-left: auto;" width="45%"  />
    <img src="asserts/img_1.png" alt="right" style="margin-right: auto;" width="45%"/>

</div>
<div style="display: flex; justify-content: space-between;">
    <img src="asserts/img_2.png" alt="left" style="margin-left: auto;" width="45%" />
    <img src="asserts/img_3.png" alt="left" style="margin-right: auto;" width="50%" />
</div>

More comprehensive ablation studies can be found at APPENDIX SECTION. 

## Getting Started
## Installation
This project is based on MMDetection3D, which can be constructed as follows.

- Install PyTorch [v1.9.1](https://pytorch.org/get-started/previous-versions/) and mmDetection3D [v0.17.3](https://github.com/open-mmlab/mmdetection3d/tree/v0.17.3) following the [instructions](https://github.com/open-mmlab/mmdetection3d/blob/v0.17.3/docs/getting_started.md).
- Install the required environment

```
conda create -n phoenix python=3.8
conda activate phoenix
conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 cudatoolkit=11.1 -c pytorch -c conda-forge

pip install mmcv-full==1.3.11 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9/index.html
pip install mmdet==2.14.0 mmsegmentation==0.14.1 tifffile-2021.11.2 numpy==1.19.5 protobuf==3.19.4 scikit-image==0.19.2 pycocotools==2.0.0 nuscenes-devkit==1.0.5 spconv-cu111 gpustat numba scipy pandas matplotlib Cython shapely loguru tqdm future fire yacs jupyterlab scikit-image pybind11 tensorboardX tensorboard easydict pyyaml open3d addict pyquaternion awscli timm typing-extensions==4.7.1
```
### Data Preparation
Please download the MS-COCO and nuScenes dataset from the official website.

### Training & Testing
You can train and eval the model following the instructions. For example:
```
# Training
bash tools/dist_train.sh projects/configs/exp/distill_800mf_onn_v2.py 8

# test
bash tools/dist_test.sh projects/configs/exp/distill_800mf_onn_v2.py 8

# run speed
python tools/test_4stage_cnn.py $config $ckpt --eval bbox
```
### Weights
#### 1. Object Detection
| Method             |  Config        | mAP  | Relative SOTA |  Model |
|--------------------|-------------------------------------------------------------------------|------|---------------|--------
| **1xONN-nuScenes** | [**1xONN**](projects/configs/gaussianpretrain/gp_0.075_convnext.py) | 41.8 | 93%           | [Google](https://drive.google.com/file/d/1EiyUJ6mAfCo6ASaeK5zhH0dhJtzQJcAu/view?usp=drive_link)
| **1xONN-COCO**     | [**1xONN**](projects/exp/distill_800mf_onn_v2.py)                   | 26.3 | 97.1%         | [Google](https://drive.google.com/file/d/1eT1MrrZpBY1UBRa2NKz5mRkmbf0USvmW/view?usp=drive_link)


## News

- [2025-05-29] 🚀 The codebase is initialed and the inference code and associated weights have been released. We are diligently preparing for a clean, optimized version. Stay tuned for the complete code release, which is coming soon..

## TODO
* Release training code.