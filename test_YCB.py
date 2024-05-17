import os
import cv2
from scipy.io import loadmat
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from mmdet_sam import mmdet_sam
from fbdinov2 import fbdinov2
from megapose import nvmegapose
from utils.choose import validate_preds
from utils.convert import Convert_String
from utils.calculate_iou import calculate_iou
import time
import csv

device = 'cuda:0'
root_path = '/media/iai-lab/wanqing/YCB_Video_Dataset'
folder_paths = sorted([p for p in (Path(root_path) / 'data').glob('*') if p.is_dir()])[:60]  # up to 0059
# remove 0003, 0011, 0015, 0019, 0025, 0032, 0033, 0036, 0047, 0048, 0057 due to 052_extra_large_clamp
remove_folders = ['0003', '0011', '0015', '0019', '0025', '0032', '0033', '0036', '0047', '0048', '0057']
filtered_paths = [path for path in folder_paths if not any(substring in str(path) for substring in remove_folders)]
MMDet_SAM = mmdet_sam.MMDet_SAM(device)
DINOv2 = fbdinov2.DINOv2("./viewpoints_42", device)
Megapose = nvmegapose.Megapose(device)
Convert_String = Convert_String()
object_list = sorted(os.listdir(root_path + '/models'))
obj_list_modified = object_list.copy()
obj_list_modified.pop(19)  # remove '052_extra_large_clamp'
# Initialize the list to store rows of data
data = []


def test_all():
    for folder in filtered_paths:
        print('Evaluating images from %s' % folder)
        txt_files = sorted(folder.rglob('*.txt'))

        # Process every 10th file
        for num in tqdm(range(0, len(txt_files), 10), desc=f'Processing images in {folder}'):
            txt_file = txt_files[num]
            image_path = str(txt_file).replace('box.txt', 'color.png')
            depth_path = str(txt_file).replace('box.txt', 'depth.png')
            label_path = str(txt_file).replace('box.txt', 'label.png')
            metad_path = str(txt_file).replace('box.txt', 'meta.mat')
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            label_img = np.array(Image.open(label_path))
            depth = np.array(Image.open(depth_path), dtype=np.float32) / 10000
            mat = loadmat(metad_path)
            with open(txt_file, 'r') as f:
                content = f.readlines()
            for line_num, line in enumerate(content):
                obj_name = line.split(' ')[0]
                gpt_name = Convert_String.convert(obj_name)
                pred = MMDet_SAM.run_detector(image.copy(), gpt_name)

                tic = time.time()
                success_flag = False
                obj_num = 0
                pose_estimation = 0
                if len(pred['labels']) > 0:
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
                    if best_iou > 0.7:
                        best_pred = validate_preds(image.copy(), pred, DINOv2)
                        best_mask = pred['masks'][best_pred]
                        best_mask = best_mask.cpu().numpy().astype(np.uint8)
                        best_mask = np.transpose(best_mask, (1, 2, 0))
                        best_mask = np.squeeze(best_mask, axis=-1)

                        if calculate_iou(best_mask, ground_mask) > 0.7:
                            bbox = np.round(pred['boxes'][best_pred].cpu().numpy()).astype(int)
                            ycb_name = Convert_String.convert(pred['labels'][best_pred])
                            pose_estimation = Megapose.inference(image.copy(), depth, ycb_name, bbox)
                            success_flag = True

                if success_flag:
                    poses = pose_estimation.poses.cpu().numpy()
                    R = poses[0, :3, :3]
                    T = poses[0, :3, 3] * 1000
                    t = time.time() - tic
                    # Flatten R and t for CSV
                    R_str = ' '.join(map(str, R.flatten()))
                    T_str = ' '.join(map(str, T))
                    # Create the row
                    row = [int(folder.name), num+1, obj_num, 1, R_str, T_str, t]
                    data.append(row)
                else:
                    R = np.zeros((3, 3))
                    T = np.zeros(3)
                    t = time.time() - tic
                    # Flatten R and t for CSV
                    R_str = ' '.join(map(str, R.flatten()))
                    T_str = ' '.join(map(str, T))
                    # Create the row
                    row = [int(folder.name), num+1, obj_num, 0, R_str, T_str, t]
                    data.append(row)
            if num > 100:
                return data


# Write to CSV
csv_file = 'outputs/result_ycbv-test.csv'
data_out = test_all()
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header
    writer.writerow(['scene_id', 'im_id', 'obj_id', 'score', 'R', 't', 'time'])
    # Write the data
    writer.writerows(data_out)



