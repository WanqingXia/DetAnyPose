# Copyright (c) OpenMMLab. All rights reserved.
# Refer from https://github.com/IDEA-Research/Grounded-Segment-Anything
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# Grounding DINO
try:
    import groundingdino
    import groundingdino.datasets.transforms as T
    from groundingdino.models import build_model
    from groundingdino.util import get_tokenlizer
    from groundingdino.util.utils import (clean_state_dict,
                                          get_phrases_from_posmap)
    grounding_dino_transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
except ImportError:
    groundingdino = None

# mmdet
try:
    import mmdet
    from mmdet.apis import inference_detector, init_detector
except ImportError:
    mmdet = None

import sys

from mmengine.config import Config
from PIL import Image
# segment anything
from segment_anything import SamPredictor, sam_model_registry

sys.path.append('../')
from mmdet_sam.utils import apply_exif_orientation, get_file_list  # noqa

# GLIP
try:
    import maskrcnn_benchmark

    from mmdet_sam.predictor_glip import GLIPDemo
except ImportError:
    maskrcnn_benchmark = None


class MMDet_SAM:
    def __init__(self, device):
        self.image = None
        self.image_path = ""
        self.pred_dict = {}
        self.det_config = "./mmdet_sam/configs/Detic_LI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.py"  # path to det config file, no default specified
        self.det_weight = "./models/detic_centernet2_swin-b_fpn_4x_lvis-coco-in21k_20230120-0d301978.pth"  # path to det weight file, no default specified
        self.only_det = False  # Default is the equivalent of not using --only-det
        self.not_show_label = False  # Default is the equivalent of not using --not-show-label
        self.sam_type = 'vit_h'  # Default sam type
        self.sam_weight = './models/sam_vit_h_4b8939.pth'  # Default path to checkpoint file
        self.out_dir = 'outputs'  # Default output directory
        self.box_thr = 0.3  # Default box threshold
        self.det_device = device  # Default device used for det inference
        self.sam_device = device  # Default device used for sam inference
        self.cpu_off_load = False  # Default is the equivalent of not using --cpu-off-load
        self.use_detic_mask = False  # Default is the equivalent of not using --use-detic-mask
        self.text_prompt = ""  # text prompt, no default specified
        self.text_thr = 0.25  # Default text threshold
        self.apply_original_groudingdino = False  # Default is the equivalent of not using --apply-original-groudingdino
        self.apply_other_text = False  # Default is the equivalent of not using --apply-other-text
        os.makedirs(self.out_dir, exist_ok=True)

        if groundingdino is None and maskrcnn_benchmark is None and mmdet is None:
            raise RuntimeError('detection model is not installed,\
                     please install it follow README')
        self.det_model = self.build_detecter()
        self.sam_model = None

        if not self.cpu_off_load:
            if 'glip' in self.det_config:
                self.det_model.model = self.det_model.model.to(self.det_device)
                self.det_model.device = self.det_device
            else:
                self.det_model = self.det_model.to(self.det_device)

        if not self.only_det:
            build_sam = sam_model_registry[self.sam_type]
            self.sam_model = SamPredictor(build_sam(checkpoint=self.sam_weight))
            if not self.cpu_off_load:
                self.sam_model.mode = self.sam_model.model.to(self.sam_device)

    def build_detecter(self):
        config = Config.fromfile(self.det_config)
        if 'init_cfg' in config.model.backbone:
            config.model.backbone.init_cfg = None
        if 'detic' in self.det_config and not self.use_detic_mask:
            config.model.roi_head.mask_head = None
        detecter = init_detector(
            config, self.det_weight, device='cpu', cfg_options={})
        return detecter

    def run_detector(self, image, prompt):
        self.image = image  # image need to be read by cv2 and convert to RGB format
        self.text_prompt = prompt

        if 'Detic' in self.det_config:
            from projects.Detic.detic.utils import get_text_embeddings
            text_prompt = self.text_prompt
            text_prompt = text_prompt.lower()
            text_prompt = text_prompt.strip()
            if text_prompt.endswith('.'):
                text_prompt = text_prompt[:-1]
            custom_vocabulary = text_prompt.split('.')
            self.det_model.dataset_meta['classes'] = [
                c.strip() for c in custom_vocabulary
            ]
            embedding = get_text_embeddings(custom_vocabulary=custom_vocabulary)
            self._reset_cls_layer_weight(embedding)

        result = inference_detector(self.det_model, self.image)
        pred_instances = result.pred_instances[
            result.pred_instances.scores > self.box_thr]

        self.pred_dict['boxes'] = pred_instances.bboxes
        self.pred_dict['scores'] = pred_instances.scores.cpu().numpy().tolist()
        self.pred_dict['labels'] = [
            self.det_model.dataset_meta['classes'][label]
            for label in pred_instances.labels
        ]
        if self.use_detic_mask:
            self.pred_dict['masks'] = pred_instances.masks

        if self.pred_dict['boxes'].shape[0] == 0:
            # print('No objects detected !')
            return self.pred_dict

        if not self.only_det:
            self.sam_model.set_image(self.image)

            transformed_boxes = self.sam_model.transform.apply_boxes_torch(
                self.pred_dict['boxes'], image.shape[:2])
            transformed_boxes = transformed_boxes.to(self.sam_model.model.device)

            masks, _, _ = self.sam_model.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False)
            self.pred_dict['masks'] = masks

        return self.pred_dict

    def draw_outcome(self, image, pred, show_result=False, save_copy=False, random_color=True, show_label=True):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        self.pred_dict = pred
        with_mask = 'masks' in self.pred_dict
        labels = self.pred_dict['labels']
        scores = self.pred_dict['scores']

        bboxes = self.pred_dict['boxes'].cpu().numpy()
        for box, label, score in zip(bboxes, labels, scores):
            x0, y0 = box[0], box[1]
            w, h = box[2] - box[0], box[3] - box[1]
            plt.gca().add_patch(
                plt.Rectangle((x0, y0),
                              w,
                              h,
                              edgecolor='green',
                              facecolor=(0, 0, 0, 0),
                              lw=2))

            if show_label:
                if isinstance(score, str):
                    plt.gca().text(x0, y0, f'{label}|{score}', color='green')
                else:
                    plt.gca().text(
                        x0, y0, f'{label}|{round(score, 2)}', color='green')

        if with_mask:
            masks = self.pred_dict['masks'].cpu().numpy()
            for mask in masks:
                if random_color:
                    color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
                else:
                    color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
                h, w = mask.shape[-2:]
                mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
                plt.gca().imshow(mask_image)

        plt.axis('off')
        if save_copy:
            save_path = os.path.join(self.out_dir, self.text_prompt + '.png')
            plt.savefig(save_path)
        if show_result:
            plt.show()


    def _reset_cls_layer_weight(self, weight):
        if type(weight) == str:
            print(f'Resetting cls_layer_weight from file: {weight}')
            zs_weight = torch.tensor(
                np.load(weight),
                dtype=torch.float32).permute(1, 0).contiguous()  # D x C
        else:
            zs_weight = weight
        zs_weight = torch.cat(
            [zs_weight, zs_weight.new_zeros(
                (zs_weight.shape[0], 1))], dim=1)  # D x (C + 1)
        zs_weight = F.normalize(zs_weight, p=2, dim=0)
        zs_weight = zs_weight.to(next(self.det_model.parameters()).device)
        num_classes = zs_weight.shape[-1]

        for bbox_head in self.det_model.roi_head.bbox_head:
            bbox_head.num_classes = num_classes
            del bbox_head.fc_cls.zs_weight
            bbox_head.fc_cls.zs_weight = zs_weight







