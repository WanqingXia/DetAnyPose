# ViTpose
 
This project aims to combine some state-of-the-art algorithms and pre-trained models together to achieve 6D pose estimation of everything, as long as the 3D CAD model is available.

The overall architecture is mmdetection + Segment Anything + DINOv2 + Megapose. The object name will first be input to mmdetection for bounding box. The bounding box will be used by SAM to segment out the object. Then the segmented images are used by DINOv2 to score the objects. Finally, the coarse object pose will be found by matching rendered templates with the segmented image by Megapose and refined.

## Base Development Environment Setup

### Clone project and create environment with conda
```shell
git clone https://github.com/WanqingXia/SiameseViT.git -b newArchi

conda env create -f environment.yaml
conda activate ViTpose
```

### Dependencies Installation

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

### Download Pre-trained Models

```shell
mkdir models; cd models
# DINOv2 L14 without register weight
wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth
# SAM ViT_h weight
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
# Detic weight
wget https://download.openmmlab.com/mmdetection/v3.0/detic/detic_centernet2_swin-b_fpn_4x_lvis-coco-in21k/detic_centernet2_swin-b_fpn_4x_lvis-coco-in21k_20230120-0d301978.pth

sudo -v ; curl https://rclone.org/install.sh | sudo bash
rclone copyto inria_data:megapose-models/ megapose-models/ --exclude="**epoch**" --config ./megapose/rclone.conf -P

cd ..
```

### Download BOP Dataset (ycbv only)

```shell
mkdir -p bop_datasets; cd bop_datasets
export SRC=https://huggingface.co/datasets/bop-benchmark/datasets/resolve/main
wget $SRC/ycbv/ycbv_base.zip         # Base archive with dataset info, camera parameters, etc.
wget $SRC/ycbv/ycbv_models.zip       # 3D object models.
wget $SRC/ycbv/ycbv_test_bop19.zip       # test images.

unzip ycbv_base.zip             # Contains folder "ycbv".
unzip ycbv_models.zip -d ycbv     # Unpacks to "ycbv".
unzip ycbv_test_bop19.zip -d ycbv   # Unpacks to "ycbv".
```

### Running the code

```shell
# run the demo with visualised results for each step
python demo.py

# run the full pipeline on the BOP YCBV dataset
python generate/generate_ycb.py # generate rendered images
python run_BOP_YCBV.py
```