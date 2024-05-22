import cv2
from pathlib import Path
from tqdm import tqdm
from mmdet_sam import mmdet_sam
from utils.convert import Convert_YCB
import time
import csv
import json

device = 'cuda:0'
root_path = './bop_datasets/ycbv'
folder_paths = sorted([p for p in (Path(root_path) / 'test').glob('*') if p.is_dir()])
Convert_YCB = Convert_YCB()

MMDet_SAM = mmdet_sam.MMDet_SAM(device)
MMDet_SAM.only_det = True
object_list = Convert_YCB.get_object_list()

# Initialize the list to store result
data = []

def eval_all():
    for count, folder in enumerate(folder_paths):
        print('Evaluating images from %s' % folder)
        rgb_files = sorted((folder / "rgb").rglob('*.png'))
        mask_files = sorted((folder / "mask").rglob('*.png'))
        objs_in_scene = int(len(mask_files) / len(rgb_files))
        t = 0

        for num in tqdm(range(0, len(rgb_files)), desc=f'Processing images in {folder}'):
            rgb_path = rgb_files[num]
            rgb = cv2.imread(str(rgb_path))
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

            with open((folder / "scene_gt.json"), 'r') as f:
                scene_data = json.load(f)
            first_row_key = list(scene_data.keys())[0]  # Get the key for the first row
            first_row_objects = scene_data[first_row_key]  # Get the list of objects in the first row
            obj_ids = [obj['obj_id'] for obj in first_row_objects]  # Extract all obj_id values

            for index, obj_id in enumerate(obj_ids):
                tic = time.time()
                success_flag = False
                if obj_id == 16:
                    ycb_name = Convert_YCB.convert_number(obj_id)
                    desc_name = Convert_YCB.convert_name(ycb_name)
                    pred = MMDet_SAM.run_detector(rgb.copy(), desc_name)
                    # print("mmdet+sam time {}".format(t2-t1))

                    if len(pred['boxes']) > 0:
                        """Testing code for DINOv2 scores, only masked areas"""
                        success_flag = True

                t += time.time() - tic
                if success_flag:
                    # Create the row
                    row = [int(folder.name), int(rgb_path.stem), obj_id, 1, t]
                    data.append(row)
                else:
                    # Create the row
                    row = [int(folder.name), int(rgb_path.stem), obj_id, 0, t]
                    data.append(row)
                if index == objs_in_scene - 1:
                    for i in range(len(data) - objs_in_scene, len(data)):
                        data[i][-1] = t
                    t = 0
                # print("Next object!")

    return data


# Write to CSV
csv_file = 'outputs/resultsam3_ycbv-test.csv'
data_out = eval_all()
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header
    writer.writerow(['scene_id', 'im_id', 'obj_id', 'score', 'time'])
    # Write the data
    writer.writerows(data_out)



