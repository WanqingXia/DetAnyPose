# Anypose
 
This project aims to combine some state-of-the-art algorithms and pre-trained models together to achieve 6D pose estimation of everything, as long as the 3D CAD model is available.

The overall architecture is chatGPT + mmdetection + Segment Anything + DINOv2 + Megapose. Start from a blur instruction, chatGPT will  determine the object that's needs. The object name will then be input to mmdetection for bounding box. The bounding box will be used by SAM to segment out the object. Finally, the coarse object pose will be found by matching rendered templates with the segmented image by DINOv2 and refined by ICP.

## Base Development Environment Setup

###Clone project and create environment with conda
```shell
git clone https://github.com/WanqingXia/SiameseViT.git -b newArchi

conda env create -f environment.yaml
conda activate anypose
```

#### Dependencies Installation

```shell
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
pip install -U openmim
mim install "mmcv>=2.0.0"

conda install pinocchio -c conda-forge
conda install -c conda-forge firefox geckodriver

pip install -r requirements.txt
# build from source
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection; pip install -e .; cd ..

git clone https://github.com/facebookresearch/dinov2.git

pip install git+https://github.com/facebookresearch/segment-anything.git
pip install git+https://github.com/openai/CLIP.git

```