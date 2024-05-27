import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from mmdet_sam import mmdet_sam
from fbdinov2 import fbdinov2
from megapose import nvmegapose
from utils.choose import validate_preds
from utils.convert import Convert_YCB
import time
import csv
import json
import torch

device = torch.device('cuda:0')
root_path = './bop_datasets/ycbv'
folder_paths = sorted([p for p in (Path(root_path) / 'test').glob('*') if p.is_dir()])
Convert_YCB = Convert_YCB()

MMDet_SAM = mmdet_sam.MMDet_SAM(device)
DINOv2 = fbdinov2.DINOv2(device, "./data/ycbv_generated")
Megapose = nvmegapose.Megapose(device, Convert_YCB)

object_list = Convert_YCB.get_object_list()

# Initialize the list to store result
data = []

def run_all():
    for count, folder in enumerate(folder_paths):
        print('Evaluating images from %s' % folder)
        rgb_files = sorted((folder / "rgb").rglob('*.png'))
        dep_files = sorted((folder / "depth").rglob('*.png'))
        mask_files = sorted((folder / "mask").rglob('*.png'))
        objs_in_scene = int(len(mask_files) / len(rgb_files))
        t = 0

        for num in tqdm(range(0, len(rgb_files)), desc=f'Processing images in {folder}'):
            rgb_path = rgb_files[num]
            dep_path = dep_files[num]
            rgb = cv2.imread(str(rgb_path))
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            depth = Image.open(dep_path)
            depth = np.array(depth, dtype=np.float32) / 10000

            with open((folder / "scene_gt.json"), 'r') as f:
                scene_data = json.load(f)
            first_row_key = list(scene_data.keys())[0]  # Get the key for the first row
            first_row_objects = scene_data[first_row_key]  # Get the list of objects in the first row
            obj_ids = [obj['obj_id'] for obj in first_row_objects]  # Extract all obj_id values

            for index, obj_id in enumerate(obj_ids):
                tic = time.time()
                success_flag = False
                pose_estimation = 0

                ycb_name = Convert_YCB.convert_number(obj_id)
                desc_name = Convert_YCB.convert_name(ycb_name)
                pred = MMDet_SAM.run_detector(rgb.copy(), desc_name)

                if len(pred['labels']) > 0:
                    # """Testing code for Detic + SAM + DINOv2 scores"""
                    # best_pred = validate_preds(rgb.copy(), pred, DINOv2)
                    # bbox = np.round(pred['boxes'][best_pred].cpu().numpy()).astype(int)
                    # ycb_name = Convert_YCB.convert_name(pred['labels'][best_pred])
                    #
                    # mask = pred['masks'][best_pred].cpu().numpy().astype(np.uint8)
                    # mask = np.transpose(mask, (1, 2, 0))
                    # rgb = np.array(rgb, dtype=np.uint8)
                    # rgb_masked = rgb * mask
                    # mask = mask.squeeze(axis=-1)
                    # depth_masked = depth * mask
                    #
                    # pose_estimation = Megapose.inference(rgb_masked, depth_masked, ycb_name, bbox)
                    # success_flag = True

                    """Testing code for Detic + DINOv2 (need to modify the validate_preds function)"""
                    # best_pred = validate_preds(rgb.copy(), pred, DINOv2)
                    # bbox = np.round(pred['boxes'][best_pred].cpu().numpy()).astype(int)
                    # ycb_name = Convert_YCB.convert_name(pred['labels'][best_pred])
                    # pose_estimation = Megapose.inference(rgb.copy(), depth, ycb_name, bbox)
                    # success_flag = True

                    """Testing code for Detic + SAM"""
                    best_pred = np.argmax(pred['scores'])
                    bbox = np.round(pred['boxes'][best_pred].cpu().numpy()).astype(int)
                    ycb_name = Convert_YCB.convert_name(pred['labels'][best_pred])

                    mask = pred['masks'][best_pred].cpu().numpy().astype(np.uint8)
                    mask = np.transpose(mask, (1, 2, 0))
                    rgb = np.array(rgb, dtype=np.uint8)
                    rgb_masked = rgb * mask
                    mask = mask.squeeze(axis=-1)
                    depth_masked = depth * mask

                    pose_estimation = Megapose.inference(rgb_masked, depth_masked, ycb_name, bbox)
                    success_flag = True

                    """Testing code for Detic"""
                    # best_pred = np.argmax(pred['scores'])
                    # bbox = np.round(pred['boxes'][best_pred].cpu().numpy()).astype(int)
                    # ycb_name = Convert_YCB.convert_name(pred['labels'][best_pred])
                    # pose_estimation = Megapose.inference(rgb.copy(), depth, ycb_name, bbox)
                    # success_flag = True

                t += time.time() - tic
                if success_flag:
                    poses = pose_estimation.poses.cpu().numpy()
                    R = poses[0, :3, :3]
                    T = poses[0, :3, 3] * 1000
                    # Flatten R and t for CSV
                    R_str = ' '.join(map(str, R.flatten()))
                    T_str = ' '.join(map(str, T))
                    # Create the row
                    row = [int(folder.name), int(rgb_path.stem), obj_id, 1, R_str, T_str, t]
                    data.append(row)
                else:
                    R = np.zeros((3, 3))
                    T = np.zeros(3)
                    # Flatten R and t for CSV
                    R_str = ' '.join(map(str, R.flatten()))
                    T_str = ' '.join(map(str, T))
                    # Create the row
                    row = [int(folder.name), int(rgb_path.stem), obj_id, 0, R_str, T_str, t]
                    data.append(row)
                if index == objs_in_scene - 1:
                    for i in range(len(data) - objs_in_scene, len(data)):
                        data[i][-1] = t
                    t = 0

    return data


# Write to CSV
csv_file = 'outputs/DeticSam_ycbv-test.csv'
data_out = run_all()
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header
    writer.writerow(['scene_id', 'im_id', 'obj_id', 'score', 'R', 't', 'time'])
    # Write the data
    writer.writerows(data_out)



