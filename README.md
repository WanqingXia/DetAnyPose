# Zero-shot 6D Estimation
 
This project aims to combine some state-of-the-art algorithms and pre-trained models together to achieve 6D pose estimation of everything, as long as the 3D CAD model is available.

The overall architecture is chatGPT + mmdetection + Segment Anything + DINOv2 + ICP refinement. Start from a blur instruction, chatGPT will  determine the object that's needs. The object name will then be input to mmdetection for bounding box. The bounding box will be used by SAM to segment out the object. Finally, the coarse object pose will be found by matching rendered templates with the segmented image by DINOv2 and refined by ICP.

## Base Development Environment Setup

```shell
conda create -n mmdet-sam python=3.8 -y
conda activate mmdet-sam
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmengine

git clone https://github.com/open-mmlab/playground.git
cd playground
```

### 1 Open-Vocabulary + SAM

Use Open-Vocabulary object detectors with SAM models. Currently we support Detic.

#### Dependencies Installation

```shell
pip install -U openmim
mim install "mmcv>=2.0.0"

# build from source
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection; pip install -e .; cd ..

pip install git+https://github.com/facebookresearch/segment-anything.git
pip install git+https://github.com/openai/CLIP.git
```