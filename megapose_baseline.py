import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from megapose import nvmegapose
from utils.convert import Convert_YCB
import time
import csv
import json

device = 'cuda:0'
root_path = './bop_datasets/ycbv'
folder_paths = sorted([p for p in (Path(root_path) / 'test').glob('*') if p.is_dir()])
Convert_YCB = Convert_YCB()
Megapose = nvmegapose.Megapose(device, Convert_YCB)
object_list = Convert_YCB.get_object_list()
# Initialize the list to store result
data = []


def baseline():
    for count, folder in enumerate(folder_paths):
        print('Evaluating images from %s' % folder)
        rgb_files = sorted((folder / "rgb").rglob('*.png'))
        dep_files = sorted((folder / "depth").rglob('*.png'))
        mask_files = sorted((folder / "mask").rglob('*.png'))
        objs_in_scene = int(len(mask_files) / len(rgb_files))
        t = 0

        with open((folder / "scene_gt.json"), 'r') as f:
            scene_data = json.load(f)
        first_row_key = list(scene_data.keys())[0]  # Get the key for the first row
        first_row_objects = scene_data[first_row_key]  # Get the list of objects in the first row
        obj_ids = [obj['obj_id'] for obj in first_row_objects]  # Extract all obj_id values
        with open((folder / "scene_gt_info.json"), 'r') as f:
            scene_gt_data = json.load(f)
        gt_info = list(scene_gt_data.items())

        for num in tqdm(range(0, len(rgb_files)), desc=f'Processing images in {folder}'):
            rgb_path = rgb_files[num]
            dep_path = dep_files[num]
            rgb = cv2.imread(str(rgb_path))
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            depth = Image.open(dep_path)
            depth = np.array(depth, dtype=np.float32) / 10000
            gt_info_img = gt_info[num][1]

            for index, obj_id in enumerate(obj_ids):
                tic = time.time()
                ycb_name = Convert_YCB.convert_number(obj_id)
                bbox = np.array(gt_info_img[index]["bbox_visib"], dtype=int)
                bbox = np.array([bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1]], dtype=int)
                pose_estimation = Megapose.inference(rgb.copy(), depth, ycb_name, bbox)
                t += time.time() - tic

                poses = pose_estimation.poses.cpu().numpy()
                R = poses[0, :3, :3]
                T = poses[0, :3, 3] * 1000
                # Flatten R and t for CSV
                R_str = ' '.join(map(str, R.flatten()))
                T_str = ' '.join(map(str, T))
                # Create the row
                row = [int(folder.name), int(rgb_path.stem), obj_id, 1, R_str, T_str, t]
                data.append(row)

                if index == objs_in_scene - 1:
                    for i in range(len(data) - objs_in_scene, len(data)):
                        data[i][-1] = t
                    t = 0

    return data


# Write to CSV
csv_file = 'outputs/resultv4_ycbv-test.csv'
data_out = baseline()
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header
    writer.writerow(['scene_id', 'im_id', 'obj_id', 'score', 'R', 't', 'time'])
    # Write the data
    writer.writerows(data_out)



