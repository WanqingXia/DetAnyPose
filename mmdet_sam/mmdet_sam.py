# Copyright (c) OpenMMLab. All rights reserved.
# Refer from https://github.com/IDEA-Research/Grounded-Segment-Anything
import argparse
import os

import cv2
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
from mmengine.utils import ProgressBar
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

class mmdet_sam:
    def __init__(self):
        self.image = ""  # path to image file, no default specified
        self.det_config = "configs/Detic_LI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.py"  # path to det config file, no default specified
        self.det_weight = "../models/detic_centernet2_swin-b_fpn_4x_lvis-coco-in21k_20230120-0d301978.pth"  # path to det weight file, no default specified
        self.only_det = False  # Default is the equivalent of not using --only-det
        self.not_show_label = False  # Default is the equivalent of not using --not-show-label
        self.sam_type = 'vit_h'  # Default sam type
        self.sam_weight = '../models/sam_vit_h_4b8939.pth'  # Default path to checkpoint file
        self.out_dir = 'outputs'  # Default output directory
        self.box_thr = 0.3  # Default box threshold
        self.det_device = 'cuda:0'  # Default device used for det inference
        self.sam_device = 'cuda:0'  # Default device used for sam inference
        self.cpu_off_load = False  # Default is the equivalent of not using --cpu-off-load
        self.use_detic_mask = False  # Default is the equivalent of not using --use-detic-mask
        self.text_prompt = None  # text prompt, no default specified
        self.text_thr = 0.25  # Default text threshold
        self.apply_original_groudingdino = False  # Default is the equivalent of not using --apply-original-groudingdino
        self.apply_other_text = False  # Default is the equivalent of not using --apply-other-text

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

        os.makedirs(self.out_dir, exist_ok=True)

        files, source_type = get_file_list(self.image)
        progress_bar = ProgressBar(len(files))
        for image_path in files:
            save_path = os.path.join(self.out_dir, self.text_prompt + '.' + os.path.basename(image_path).split(".")[1])
            det_model, pred_dict = run_detector(det_model, image_path, args)

            if pred_dict['boxes'].shape[0] == 0:
                print('No objects detected !')
                continue

            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if not only_det:

                if cpu_off_load:
                    sam_model.model = sam_model.model.to(args.sam_device)

                sam_model.set_image(image)

                transformed_boxes = sam_model.transform.apply_boxes_torch(
                    pred_dict['boxes'], image.shape[:2])
                transformed_boxes = transformed_boxes.to(sam_model.model.device)

                masks, _, _ = sam_model.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes,
                    multimask_output=False)
                pred_dict['masks'] = masks

                if cpu_off_load:
                    sam_model.model = sam_model.model.to('cpu')
            for i in range(pred_dict['masks'].size(0)):
                # Normalize the tensor to be in the range [0, 255]
                mask = pred_dict['masks'][i].mul(255).byte()
                # Convert to PIL Image
                img = Image.fromarray(mask.squeeze().cpu().numpy(), 'L')
                # Save image with the channel number in the name
                img.save(f'./mask_channel_{i}.png')

            draw_and_save(
                image, pred_dict, save_path, show_label=not args.not_show_label)
            progress_bar.update()

    def build_detecter(self):
        config = Config.fromfile(self.det_config)
        if 'init_cfg' in config.model.backbone:
            config.model.backbone.init_cfg = None
        if 'detic' in self.det_config and not self.use_detic_mask:
            config.model.roi_head.mask_head = None
        detecter = init_detector(
            config, self.det_weight, device='cpu', cfg_options={})
        return detecter

    def run_detector(model, image_path, args):
        pred_dict = {}

        if args.cpu_off_load:
            if 'glip' in args.det_config:
                model.model = model.model.to(args.det_device)
                model.device = args.det_device
            else:
                model = model.to(args.det_device)

        if 'GroundingDINO' in args.det_config:
            image_pil = Image.open(image_path).convert('RGB')  # load image
            image_pil = apply_exif_orientation(image_pil)
            image, _ = grounding_dino_transform(image_pil, None)  # 3, h, w

            text_prompt = args.text_prompt
            text_prompt = text_prompt.lower()
            text_prompt = text_prompt.strip()
            if not text_prompt.endswith('.'):
                text_prompt = text_prompt + '.'

            # Original GroundingDINO use text-thr to get class name,
            # the result will always result in categories that we don't want,
            # so we provide a category-restricted approach to address this

            if not args.apply_original_groudingdino:
                # custom label name
                custom_vocabulary = text_prompt[:-1].split('.')
                label_name = [c.strip() for c in custom_vocabulary]
                tokens_positive = []
                start_i = 0
                separation_tokens = ' . '
                for _index, label in enumerate(label_name):
                    end_i = start_i + len(label)
                    tokens_positive.append([(start_i, end_i)])
                    if _index != len(label_name) - 1:
                        start_i = end_i + len(separation_tokens)
                tokenizer = get_tokenlizer.get_tokenlizer('bert-base-uncased')
                tokenized = tokenizer(
                    args.text_prompt, padding='longest', return_tensors='pt')
                positive_map_label_to_token = create_positive_dict(
                    tokenized, tokens_positive, list(range(len(label_name))))

            image = image.to(next(model.parameters()).device)

            with torch.no_grad():
                outputs = model(image[None], captions=[text_prompt])

            logits = outputs['pred_logits'].cpu().sigmoid()[0]  # (nq, 256)
            boxes = outputs['pred_boxes'].cpu()[0]  # (nq, 4)

            if not args.apply_original_groudingdino:
                logits = convert_grounding_to_od_logits(
                    logits, len(label_name),
                    positive_map_label_to_token)  # [N, num_classes]

            # filter output
            logits_filt = logits.clone()
            boxes_filt = boxes.clone()
            filt_mask = logits_filt.max(dim=1)[0] > args.box_thr
            logits_filt = logits_filt[filt_mask]  # num_filt, 256
            boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

            if args.apply_original_groudingdino:
                # get phrase
                tokenlizer = model.tokenizer
                tokenized = tokenlizer(text_prompt)
                # build pred
                pred_labels = []
                pred_scores = []
                for logit, box in zip(logits_filt, boxes_filt):
                    pred_phrase = get_phrases_from_posmap(logit > args.text_thr,
                                                          tokenized, tokenlizer)
                    pred_labels.append(pred_phrase)
                    pred_scores.append(str(logit.max().item())[:4])
            else:
                scores, pred_phrase_idxs = logits_filt.max(1)
                # build pred
                pred_labels = []
                pred_scores = []
                for score, pred_phrase_idx in zip(scores, pred_phrase_idxs):
                    pred_labels.append(label_name[pred_phrase_idx])
                    pred_scores.append(str(score.item())[:4])

            pred_dict['labels'] = pred_labels
            pred_dict['scores'] = pred_scores

            size = image_pil.size
            H, W = size[1], size[0]
            for i in range(boxes_filt.size(0)):
                boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
                boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
                boxes_filt[i][2:] += boxes_filt[i][:2]
            pred_dict['boxes'] = boxes_filt
        elif 'glip' in args.det_config:
            image = cv2.imread(image_path)
            # caption
            text_prompt = args.text_prompt
            text_prompt = text_prompt.lower()
            text_prompt = text_prompt.strip()
            if not text_prompt.endswith('.') and not args.apply_other_text:
                text_prompt = text_prompt + '.'

            custom_vocabulary = text_prompt[:-1].split('.')
            label_name = [c.strip() for c in custom_vocabulary]

            # top_predictions = model.inference(image, label_name)
            if args.apply_other_text:
                top_predictions = model.inference(
                    image, args.text_prompt, use_other_text=True)
            else:
                top_predictions = model.inference(
                    image, args.text_prompt, use_other_text=False)
            scores = top_predictions.get_field('scores').tolist()
            labels = top_predictions.get_field('labels').tolist()

            if args.apply_other_text:
                new_labels = []
                if model.entities and model.plus:
                    for i in labels:
                        if i <= len(model.entities):
                            new_labels.append(model.entities[i - model.plus])
                        else:
                            new_labels.append('object')
                else:
                    new_labels = ['object' for i in labels]
            else:
                new_labels = [label_name[i] for i in labels]

            pred_dict['labels'] = new_labels
            pred_dict['scores'] = scores
            pred_dict['boxes'] = top_predictions.bbox
        else:
            result = inference_detector(model, image_path)
            pred_instances = result.pred_instances[
                result.pred_instances.scores > args.box_thr]

            pred_dict['boxes'] = pred_instances.bboxes
            pred_dict['scores'] = pred_instances.scores.cpu().numpy().tolist()
            pred_dict['labels'] = [
                model.dataset_meta['classes'][label]
                for label in pred_instances.labels
            ]
            if args.use_detic_mask:
                pred_dict['masks'] = pred_instances.masks

        if args.cpu_off_load:
            if 'glip' in args.det_config:
                model.model = model.model.to('cpu')
                model.device = 'cpu'
            else:
                model = model.to('cpu')
        return model, pred_dict


    def create_positive_dict(self, tokenized, tokens_positive, labels):
        """construct a dictionary such that positive_map[i] = j,
        if token i is mapped to j label"""

        positive_map_label_to_token = {}

        for j, tok_list in enumerate(tokens_positive):
            for (beg, end) in tok_list:
                beg_pos = tokenized.char_to_token(beg)
                end_pos = tokenized.char_to_token(end - 1)

                assert beg_pos is not None and end_pos is not None
                positive_map_label_to_token[labels[j]] = []
                for i in range(beg_pos, end_pos + 1):
                    positive_map_label_to_token[labels[j]].append(i)

        return positive_map_label_to_token

    def convert_grounding_to_od_logits(self, logits,
                                       num_classes,
                                       positive_map,
                                       score_agg='MEAN'):
        """
        logits: (num_query, max_seq_len)
        num_classes: 80 for COCO
        """
        assert logits.ndim == 2
        assert positive_map is not None
        scores = torch.zeros(logits.shape[0], num_classes).to(logits.device)
        # 256 -> 80, average for each class
        # score aggregation method
        if score_agg == 'MEAN':  # True
            for label_j in positive_map:
                scores[:, label_j] = logits[:,
                                     torch.LongTensor(positive_map[label_j]
                                                      )].mean(-1)
        else:
            raise NotImplementedError
        return scores

    def draw_and_save(self, image,
                      pred_dict,
                      save_path,
                      random_color=True,
                      show_label=True):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)

        with_mask = 'masks' in pred_dict
        labels = pred_dict['labels']
        scores = pred_dict['scores']

        bboxes = pred_dict['boxes'].cpu().numpy()
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
            masks = pred_dict['masks'].cpu().numpy()
            for mask in masks:
                if random_color:
                    color = np.concatenate(
                        [np.random.random(3), np.array([0.6])], axis=0)
                else:
                    color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
                h, w = mask.shape[-2:]
                mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
                plt.gca().imshow(mask_image)

        plt.axis('off')
        plt.savefig(save_path)

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







