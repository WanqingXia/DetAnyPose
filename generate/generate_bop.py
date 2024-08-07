import os
from pathlib import Path

# Third Party
import numpy as np
import json
# MegaPose
from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset
from megapose.datasets.scene_dataset import CameraData
from megapose.utils.load_model import load_named_model

def make_object_dataset(cad_model_dir: Path, convert) -> RigidObjectDataset:
    rigid_objects = []
    mesh_units = "mm"
    object_plys = sorted(cad_model_dir.rglob('*.ply'))
    print("Loading all CAD models from {}, default unit {}, this may take a long time".
          format(cad_model_dir, mesh_units))
    for num, object_ply in enumerate(object_plys):
        label = convert.convert_number(num)
        two_digit_str = str(num + 1).zfill(2)
        rigid_objects.append(RigidObject(label=(two_digit_str+'_'+label), mesh_path=object_ply, mesh_units=mesh_units))
    rigid_object_dataset = RigidObjectDataset(rigid_objects)
    return rigid_object_dataset


def generate_bop(convert, d_name):
    device = "cuda:0"
    model_name = "megapose-1.0-RGB-multi-hypothesis-icp"
    json_file_path = Path("./bop_datasets/" + d_name + "/camera_data.json")
    camera_data = CameraData.from_json(json_file_path.read_text())
    models_path = Path("./models/megapose-models")
    cad_path = Path("./bop_datasets/" + d_name + "/models")
    save_dir_root = Path("./data/" + d_name + "_generated")
    object_dataset = make_object_dataset(cad_path, convert)
    pose_estimator = load_named_model(model_name, models_path, object_dataset).to(device)
    # for image size 480*640, the detection is default 100*100, in the middle of the image
    detection = np.array([270, 190, 369, 289], dtype=int)
    for num, label in enumerate(convert.get_object_list()):
        two_digit_str = str(num + 1).zfill(2)
        folder_name = two_digit_str + '_' + label
        save_dir = save_dir_root / folder_name
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        pose_estimator.image_generation(
            save_dir=save_dir, detection=detection, K=camera_data.K, label=folder_name, device=device
        )
