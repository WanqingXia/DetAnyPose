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
from utils.calculate_iou import calculate_iou
import time
import csv
import json

device = 'cuda:0'
root_path = './bop_datasets/ycbv'
folder_paths = sorted([p for p in (Path(root_path) / 'test').glob('*') if p.is_dir()])
Convert_YCB = Convert_YCB()

MMDet_SAM = mmdet_sam.MMDet_SAM(device)
DINOv2 = fbdinov2.DINOv2(device, "./viewpoints_42")
Megapose = nvmegapose.Megapose(device, Convert_YCB)

object_list = Convert_YCB.get_object_list()

# Initialize the list to store result
data = []

def test_all():
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
                t1 = time.time()
                pred = MMDet_SAM.run_detector(rgb.copy(), desc_name)
                t2 = time.time()
                # print("mmdet+sam time {}".format(t2-t1))

                if len(pred['labels']) > 0:
                    """Testing code for DINOv2 scores, only masked areas"""
                    best_pred = validate_preds(rgb.copy(), pred, DINOv2)
                    t3 = time.time()
                    # print("Dinov2 time {}".format(t3 - t2))
                    bbox = np.round(pred['boxes'][best_pred].cpu().numpy()).astype(int)
                    ycb_name = Convert_YCB.convert_name(pred['labels'][best_pred])

                    mask = pred['masks'][best_pred].cpu().numpy().astype(np.uint8)
                    mask = np.transpose(mask, (1, 2, 0))
                    rgb = np.array(rgb, dtype=np.uint8)
                    rgb_masked = rgb * mask
                    mask = mask.squeeze(axis=-1)
                    depth_masked = depth * mask

                    pose_estimation = Megapose.inference(rgb_masked, depth_masked, ycb_name, bbox)
                    t4 = time.time()
                    # print("Megapose time {}".format(t4 - t3))
                    success_flag = True

                    """Testing code for DINOv2 scores"""
                    # best_pred = validate_preds(rgb.copy(), pred, DINOv2)
                    # bbox = np.round(pred['boxes'][best_pred].cpu().numpy()).astype(int)
                    # ycb_name = Convert_YCB.convert_name(pred['labels'][best_pred])
                    # pose_estimation = Megapose.inference(rgb.copy(), depth, ycb_name, bbox)
                    # success_flag = True

                    """Testing code for DINOv2 scores with ground truth masks"""
                    # mask_path = mask_files[(objs_in_scene * num + index)]
                    # mask = Image.open(mask_path)
                    # mask = np.array(mask, dtype=bool)
                    # best_iou = 0
                    # for pred_mask in pred['masks']:
                    #     pred_mask = pred_mask.cpu().numpy().astype(np.uint8)
                    #     pred_mask = np.transpose(pred_mask, (1, 2, 0))
                    #     pred_mask = np.squeeze(pred_mask, axis=-1)
                    #     iou = calculate_iou(pred_mask, mask)
                    #     if iou > best_iou:
                    #         best_iou = iou
                    #
                    # if best_iou > 0.6:
                    #     best_pred = validate_preds(rgb.copy(), pred, DINOv2)
                    #     best_mask = pred['masks'][best_pred]
                    #     best_mask = best_mask.cpu().numpy().astype(np.uint8)
                    #     best_mask = np.transpose(best_mask, (1, 2, 0))
                    #     best_mask = np.squeeze(best_mask, axis=-1)
                    #
                    #     if calculate_iou(best_mask, mask) > 0.6:
                    #         bbox = np.round(pred['boxes'][best_pred].cpu().numpy()).astype(int)
                    #         ycb_name = Convert_YCB.convert_name(pred['labels'][best_pred])
                    #         pose_estimation = Megapose.inference(rgb.copy(), depth, ycb_name, bbox)
                    #         success_flag = True

                    """Testing code for best scores"""
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
                # print("Next object!")

    return data


# Write to CSV
csv_file = 'outputs/resultv8_ycbv-test.csv'
data_out = test_all()
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header
    writer.writerow(['scene_id', 'im_id', 'obj_id', 'score', 'R', 't', 'time'])
    # Write the data
    writer.writerows(data_out)



