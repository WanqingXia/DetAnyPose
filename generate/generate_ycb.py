import os
from pathlib import Path

# Third Party
import numpy as np

# MegaPose
from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset
from megapose.datasets.scene_dataset import CameraData
from megapose.utils.load_model import load_named_model
from utils.convert import Convert_YCB


def make_ycb_object_dataset(cad_model_dir: Path, Convert_YCB) -> RigidObjectDataset:
    rigid_objects = []
    mesh_units = "mm"
    object_plys = sorted(cad_model_dir.rglob('*.ply'))
    print("Loading all CAD models from {}, default unit {}, this may take a long time".
          format(cad_model_dir, mesh_units))
    for num, object_ply in enumerate(object_plys):
        label = Convert_YCB.convert_number(num + 1)
        rigid_objects.append(RigidObject(label=label, mesh_path=object_ply, mesh_units=mesh_units))
    rigid_object_dataset = RigidObjectDataset(rigid_objects)
    return rigid_object_dataset


if __name__ == "__main__":
    Convert_YCB = Convert_YCB()
    device = 'cuda:0'
    model_name = "megapose-1.0-RGB-multi-hypothesis-icp"
    camera_data = CameraData.from_json((Path("../data/ycbv_camera_data.json")).read_text())
    models_path = Path("../models/megapose-models")
    cad_path = Path("../bop_datasets/ycbv/models")
    save_dir_root = Path("../data/ycbv_generated")
    object_dataset = make_ycb_object_dataset(cad_path, Convert_YCB)
    pose_estimator = load_named_model(model_name, models_path, object_dataset).to(device)
    # for image size 480*640, the detection is default 100*100, in the middle of the image
    detection = np.array([270, 190, 369, 289], dtype=int)
    for label in Convert_YCB.get_object_list():
        save_dir = save_dir_root / label
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        pose_estimator.image_generation(
            save_dir=save_dir, detection=detection, K=camera_data.K, label=label, device=device
        )
