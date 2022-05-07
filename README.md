# CPT

This is the code for paper "[CPT: Colorful Prompt Tuning for Pre-trained Vision-Language Models](https://arxiv.org/abs/2109.11797)".

## Recent Updates
- [x] 2022.05.06 Initialize CPT for grounding, VRD, GQA, and VCR codes.


## Quick links

* [Overview](#overview)
* [Install](#install)
* [Dataset](#dataset)
* [Object Detector](#object-detector)
* [IETrans](#ietrans)
    * [Preparation](#preparation)
    * [Training](#training)
* [Bugs or questions?](#bugs-or-questions)
* [Acknowledgement](#acknowledgement)

## Overview
![alt text](demo/teaser.png "Illustration of CPT")

The code is based on two sub-repos. The prompt-feat is used to extract visual features with the help of pre-trained object detector. The Oscar is the pre-trained vision and language model to conduct inference.

## Install

We wrap all the commands in `install.sh`. You can directly run `bash install.sh`. Or:

```bash
# you can direcly run by 
# bash install.sh

# create a new environment
conda create --name cpt python=3.7
conda activate cpt

# install pytorch1.6
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch

export INSTALL_DIR=$PWD

# install apex
cd $INSTALL_DIR
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext

# install requirements
pip install -r requirements.txt

# install prompt_feat
cd prompt_feat
python setup.py build develop

# install oscar
cd Oscar
# install transformers
git clone git@github.com:huggingface/transformers.git
cd transformers
git reset --hard 067923d3267325f525f4e46f357360c191ba562e
cd ..
# install coco_caption
git clone git@github.com:LuoweiZhou/coco-caption.git
cd coco_caption
git reset --hard de6f385503ac9a4305a1dcdc39c02312f9fa13fc
# ./get_stanford_models.sh
cd ..

python setup.py build develop

unset INSTALL_DIR
```



## Tasks
### Visual Grounding

Visual Grounding task is to find the visual region corresponding to a query sentence e.g.: the black horse.

#### Data

Please download the data first.

#### Configuration

Before

#### Feature Extraction

#### CPT Inference

#### Evaluation





## Bugs or questions?
If you have any questions related to the code or the paper, feel free to email Ao Zhang (`zhanga6@outlook.com`). If you encounter any problems when using the code, or want to report a bug, you can open an issue. Please try to specify the problem with details so we can help you better and quicker!


## Acknowledgement
The code is built on [scene_graph_benchmark](https://github.com/microsoft/scene_graph_benchmark) and [Oscar](https://github.com/microsoft/Oscar)
Thanks for their excellent codes.