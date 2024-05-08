# Standard Library
import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Union

# Third Party
import numpy as np
from bokeh.io import export_png
from bokeh.plotting import gridplot
from PIL import Image

# MegaPose
from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset
from megapose.datasets.scene_dataset import CameraData, ObjectData
from megapose.inference.types import (
    DetectionsType,
    ObservationTensor,
    PoseEstimatesType,
)
from megapose.inference.utils import make_detections_from_object_data
from megapose.lib3d.transform import Transform
from megapose.panda3d_renderer import Panda3dLightData
from megapose.panda3d_renderer.panda3d_scene_renderer import Panda3dSceneRenderer
from megapose.utils.conversion import convert_scene_observation_to_panda3d
from megapose.utils.load_model import NAMED_MODELS, load_named_model
from megapose.utils.logging import get_logger
from megapose.visualization.bokeh_plotter import BokehPlotter
from megapose.visualization.utils import make_contour_overlay

logger = get_logger(__name__)


class Megapose:
    def __init__(self, device):
        self.device = device
        self.model_name = "megapose-1.0-RGB-multi-hypothesis-icp"
        self.model_info = NAMED_MODELS[self.model_name]
        self.camera_data = CameraData.from_json((Path("./camera_data.json")).read_text())
        self.models_path = Path("./models/megapose-models")
        self.cad_path = Path("./data/drill")

    def inference(self, rgb, depth, label, bbox):
        """
        :param rgb: np array of the RGB image, np.uint8 type
        :param depth: np array of the depth image, np.float32 type or None
        :param label: object name in string format
        :param bbox: bounding box of the object [xmin, ymin, xmax, ymax] format
        :return: prediction result in RT
        """
        # make sure the size of camera input and images are same
        assert rgb.shape[:2] == self.camera_data.resolution
        assert depth.shape[:2] == self.camera_data.resolution
        observation = ObservationTensor.from_numpy(rgb, depth, self.camera_data.K).cuda()

        object_data = [ObjectData(label=label, bbox_modal=bbox)]
        detections = make_detections_from_object_data(object_data).cuda()
        object_dataset = self.make_object_dataset(self.cad_path)

        logger.info(f"Loading model {self.model_name}.")
        pose_estimator = load_named_model(self.model_name, self.models_path, object_dataset).cuda()

        logger.info(f"Running inference.")
        output, _ = pose_estimator.run_inference_pipeline(
            observation, detections=detections, run_detector=False, **self.model_info["inference_parameters"]
        )

        self.save_predictions(Path("./data/drill/outputs"), output)
        return

    def output_visualization(self, example_dir: Path) -> None:
        rgb, _, camera_data = self.load_observation(example_dir, load_depth=False)
        camera_data.TWC = Transform(np.eye(4))
        object_datas = self.load_object_data(example_dir / "outputs" / "object_data.json")
        object_dataset = self.make_object_dataset(example_dir)

        renderer = Panda3dSceneRenderer(object_dataset)

        camera_data, object_datas = convert_scene_observation_to_panda3d(camera_data, object_datas)
        light_datas = [
            Panda3dLightData(
                light_type="ambient",
                color=((1.0, 1.0, 1.0, 1)),
            ),
        ]
        renderings = renderer.render_scene(
            object_datas,
            [camera_data],
            light_datas,
            render_depth=False,
            render_binary_mask=False,
            render_normals=False,
            copy_arrays=True,
        )[0]

        plotter = BokehPlotter()

        fig_rgb = plotter.plot_image(rgb)
        fig_mesh_overlay = plotter.plot_overlay(rgb, renderings.rgb)
        contour_overlay = make_contour_overlay(
            rgb, renderings.rgb, dilate_iterations=1, color=(0, 255, 0)
        )["img"]
        fig_contour_overlay = plotter.plot_image(contour_overlay)
        fig_all = gridplot([[fig_rgb, fig_contour_overlay, fig_mesh_overlay]], toolbar_location=None)
        vis_dir = example_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)
        export_png(fig_mesh_overlay, filename=vis_dir / "mesh_overlay.png")
        export_png(fig_contour_overlay, filename=vis_dir / "contour_overlay.png")
        export_png(fig_all, filename=vis_dir / "all_results.png")
        logger.info(f"Wrote visualizations to {vis_dir}.")
        return

    def load_observation(self, example_dir: Path, load_depth: bool = False) -> Tuple[
        np.ndarray, Union[None, np.ndarray], CameraData]:
        camera_data = CameraData.from_json((example_dir / "camera_data.json").read_text())

        rgb = np.array(Image.open(example_dir / "image_rgb.png"), dtype=np.uint8)
        assert rgb.shape[:2] == camera_data.resolution

        depth = None
        if load_depth:
            depth = np.array(Image.open(example_dir / "image_depth.png"), dtype=np.float32) / 1000
            assert depth.shape[:2] == camera_data.resolution

        return rgb, depth, camera_data

    def load_observation_tensor(self,
            example_dir: Path,
            load_depth: bool = False,
    ) -> ObservationTensor:
        rgb, depth, camera_data = self.load_observation(example_dir, load_depth)
        observation = ObservationTensor.from_numpy(rgb, depth, camera_data.K)
        return observation

    def load_object_data(self, data_path: Path) -> List[ObjectData]:
        object_data = json.loads(data_path.read_text())
        object_data = [ObjectData.from_json(d) for d in object_data]
        return object_data

    def load_detections(self, example_dir: Path) -> DetectionsType:
        input_object_data = self.load_object_data(example_dir / "inputs/object_data.json")
        detections = make_detections_from_object_data(input_object_data).cuda()
        return detections

    def make_object_dataset(self, example_dir: Path) -> RigidObjectDataset:
        rigid_objects = []
        mesh_units = "mm"
        object_dirs = (example_dir / "meshes").iterdir()
        for object_dir in object_dirs:
            label = object_dir.name
            mesh_path = None
            for fn in object_dir.glob("*"):
                if fn.suffix in {".obj", ".ply"}:
                    assert not mesh_path, f"there multiple meshes in the {label} directory"
                    mesh_path = fn
            assert mesh_path, f"couldnt find a obj or ply mesh for {label}"
            rigid_objects.append(RigidObject(label=label, mesh_path=mesh_path, mesh_units=mesh_units))
            # TODO: fix mesh units
        rigid_object_dataset = RigidObjectDataset(rigid_objects)
        return rigid_object_dataset

    def save_predictions(self,
            example_dir: Path,
            pose_estimates: PoseEstimatesType,
    ) -> None:
        labels = pose_estimates.infos["label"]
        poses = pose_estimates.poses.cpu().numpy()
        object_data = [
            ObjectData(label=label, TWO=Transform(pose)) for label, pose in zip(labels, poses)
        ]
        object_data_json = json.dumps([x.to_json() for x in object_data])
        output_fn = example_dir / "outputs" / "object_data.json"
        output_fn.parent.mkdir(exist_ok=True)
        output_fn.write_text(object_data_json)
        logger.info(f"Wrote predictions: {output_fn}")
        return
