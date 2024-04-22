import os
import cv2
from scipy.io import loadmat
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from mmdet_sam import mmdet_sam
from fbdinov2 import fbdinov2
from utils.choose import choose_from_viewpoints
from utils.convert import Convert_String
from utils.calculate_iou import calculate_iou
import time


root_path = '/media/iai-lab/wanqing/YCB_Video_Dataset'
folder_paths = sorted([p for p in (Path(root_path) / 'data').glob('*') if p.is_dir()])[:60]  # up to 0059
# remove 0003, 0011, 0015, 0019, 0025, 0032, 0033, 0036, 0047, 0048, 0057 due to 052_extra_large_clamp
remove_folders = ['0003', '0011', '0015', '0019', '0025', '0032', '0033', '0036', '0047', '0048', '0057']
filtered_paths = [path for path in folder_paths if not any(substring in str(path) for substring in remove_folders)]
MMDet_SAM = mmdet_sam.MMDet_SAM()
DINOv2 = fbdinov2.DINOv2("./viewpoints_42")
Convert_String = Convert_String()
object_list = sorted(os.listdir(root_path + '/models'))
obj_list_modified = object_list.copy()
obj_list_modified.pop(19)  # remove '052_extra_large_clamp'

for folder in filtered_paths:
    tic = time.time()
    object_count = dict.fromkeys(obj_list_modified, 0)
    mmdet_fail = dict.fromkeys(obj_list_modified, 0)
    isolation_fail = dict.fromkeys(obj_list_modified, 0)
    identification_fail = dict.fromkeys(obj_list_modified, 0)
    success = dict.fromkeys(obj_list_modified, 0)
    print('Evaluating images from %s' % folder)
    txt_files = sorted(folder.rglob('*.txt'))
    for num, txt_file in tqdm(enumerate(txt_files), total=len(txt_files), desc=f'Processing images in {folder}'):
        image_path = str(txt_file).replace('box.txt', 'color.png')
        label_path = str(txt_file).replace('box.txt', 'label.png')
        metad_path = str(txt_file).replace('box.txt', 'meta.mat')
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label_img = np.array(Image.open(label_path))
        mat = loadmat(metad_path)
        with open(txt_file, 'r') as f:
            content = f.readlines()
        for line_num, line in enumerate(content):
            obj_name = line.split(' ')[0]
            gpt_name = Convert_String.convert(obj_name)
            object_count[obj_name] += 1
            pred = MMDet_SAM.run_detector(image.copy(), image_path, gpt_name)
            if len(pred['labels']) == 0:
                mmdet_fail[obj_name] += 1
            else:
                obj_num = object_list.index(obj_name) + 1
                ground_mask = (label_img == obj_num)
                best_iou = 0
                for pred_mask in pred['masks']:
                    pred_mask = pred_mask.cpu().numpy().astype(np.uint8)
                    pred_mask = np.transpose(pred_mask, (1, 2, 0))
                    pred_mask = np.squeeze(pred_mask, axis=-1)
                    iou = calculate_iou(pred_mask, ground_mask)
                    if iou > best_iou:
                        best_iou = iou
                if best_iou < 0.7:
                    mmdet_fail[obj_name] += 1
                else:
                    vp_img_path, vp_pose, best_pred, embed_img, iso_img = choose_from_viewpoints(image, pred, DINOv2)
                    best_mask = pred['masks'][best_pred]
                    best_mask = best_mask.cpu().numpy().astype(np.uint8)
                    best_mask = np.transpose(best_mask, (1, 2, 0))
                    best_mask = np.squeeze(best_mask, axis=-1)
                    if calculate_iou(best_mask, ground_mask) < 0.7:
                        isolation_fail[obj_name] += 1
                    else:
                        ref_pose = np.array(mat['poses'][:, :, line_num])
                        best_vp = DINOv2.search_img(ref_pose, obj_name)
                        best_vp = str(best_vp).replace('matrix.txt', 'color.png')
                        if vp_img_path != best_vp:
                            # if the isolation failed, do not count the fail for identification
                            identification_fail[obj_name] += 1
                        else:
                            success[obj_name] += 1
    toc = time.time()
    # Identify keys where the value in object_count is 0
    keys_to_remove = [key for key, value in object_count.items() if value == 0]

    # Remove these keys from all dictionaries
    for key in keys_to_remove:
        del object_count[key]
        del mmdet_fail[key]
        del isolation_fail[key]
        del identification_fail[key]
        del success[key]

    # Save the dictionary to a text file
    with open('outputs/' + folder.name + '.txt', 'w') as file:
        file.write(f'Time taken: {toc-tic} \n')
        file.write('object_count \n')
        for key, value in object_count.items():
            file.write(f'{key}: {value}\n')
        file.write('mmdet_sam_fail \n')
        for key, value in mmdet_fail.items():
            file.write(f'{key}: {value}\n')
        file.write('isolation_fail \n')
        for key, value in isolation_fail.items():
            file.write(f'{key}: {value}\n')
        file.write('identification_fail \n')
        for key, value in identification_fail.items():
            file.write(f'{key}: {value}\n')
        file.write('success \n')
        for key, value in success.items():
            file.write(f'{key}: {value}\n')




