#!/usr/bin/python3.8
# -*- encoding: utf-8 -*-

import blenderproc as bproc
import numpy as np
import os
import cv2
import h5py
import shutil
import png
from tqdm import tqdm

"""
This is the main script to generate all the images for individual object 
with BlenderProc. The camera position and light position will be reset for
each image. Run the script by calling "blenderproc run generate.py" in terminal
"""


# functions for debug, only uncomment this when debugging is needed
# import debugpy
# debugpy.listen(5678)
# debugpy.wait_for_client()

# exception handler
def handler(func, path, exc_info):
    print("Inside handler")
    print(exc_info)


def calc_box(dataPath):
    class Point(object):
        def __init__(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z

    class File(object):
        count = 0
        filepaths = []

        def get_file_paths(self, base_path):
            folders = [d for d in sorted(os.listdir(base_path)) if os.path.isdir(os.path.join(base_path, d))]
            for folder in folders:
                folder_path = os.path.join(base_path, folder)
                files = os.listdir(folder_path)
                for file in files:
                    file_path = os.path.join(folder_path, file)
                    if os.path.isfile(file_path) and file[-3:] == "obj" and file[-10:-4] != "simple":
                        # print("file path:", file_path)
                        self.filepaths.append(file_path)
                        self.count += 1

    class Normalize(object):
        minP = Point(1000, 10000, 10000)
        maxP = Point(0, 0, 0)

        def reset_points(self):
            self.minP = Point(1000, 10000, 10000)
            self.maxP = Point(0, 0, 0)

        def get_bounding_box(self, p):
            # Get min and max for x, y, z of an object
            self.minP.x = p.x if p.x < self.minP.x else self.minP.x
            self.minP.y = p.y if p.y < self.minP.y else self.minP.y
            self.minP.z = p.z if p.z < self.minP.z else self.minP.z
            self.maxP.x = p.x if p.x > self.maxP.x else self.maxP.x
            self.maxP.y = p.y if p.y > self.maxP.y else self.maxP.y
            self.maxP.z = p.z if p.z > self.maxP.z else self.maxP.z

        def get_bounding_box_length(self):
            # Get the length for bounding box
            l = self.maxP.x - self.minP.x
            w = self.maxP.y - self.minP.y
            h = self.maxP.z - self.minP.z
            return l, w, h

        def get_bounding_box_diag_length(self, l, w, h):
            # Get the diagonal length
            diag_rect = np.sqrt(l ** 2 + w ** 2)
            diag_box = np.sqrt(diag_rect ** 2 + h ** 2)
            return round(diag_box, 3)

        def read_points(self, filename):
            # read all points in an obj file
            with open(filename) as f:
                points = []
                while 1:
                    line = f.readline()
                    if not line:
                        break
                    strs = line.split(" ")
                    if strs[0] == "v":
                        points.append(Point(float(strs[1]), float(strs[2]), float(strs[3])))
                    if strs[0] == "vt":
                        break
            return points
    """
    Calculate the diagonal length of the bounding boxes of each object in the YCB_Video_Dataset.

    Args:
        dataPath (str): The path to the YCB_Video_Dataset directory.

    Returns:
        list: A list of tuples containing the file path and diagonal length of each object.

    """
    print("Calculating diagonal length for every model")
    dataFile = File()
    dataFile.get_file_paths(dataPath)
    dataNormalize = Normalize()
    paths = []
    diags = []

    for file in dataFile.filepaths:
        # read points from obj file
        points = dataNormalize.read_points(file)
        for point in points:
            dataNormalize.get_bounding_box(point)
            # get the length and diagnoal length of bounding box
        length, width, height = dataNormalize.get_bounding_box_length()
        diag = dataNormalize.get_bounding_box_diag_length(length, width, height)
        dataNormalize.reset_points()
        paths.append(file)
        diags.append(diag)
    print("Finished calculating diagonal length")
    return paths, diags


# Create the output path for each object
def create_output_path(filepath: str, output_folder: str) -> str:
    """
    This function creates the output path for the given object file.

    Args:
        filepath (str): The path to the object file.
        output_folder (str): The path to the output folder.

    Returns:
        str: The path to the output folder for the given object file.

    """
    obj_name = filepath.split("/")[-2]
    out_parent = os.path.join(output_folder, obj_name)

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    if os.path.exists(out_parent):
        shutil.rmtree(out_parent, onerror=handler)

    os.mkdir(out_parent)

    return out_parent


# Sample camera points and save to txt
def sample_points(radius: float, sample: int) -> np.ndarray:
    """
    This function generates a set of random points on a sphere with a given radius and sample size.

    Args:
        radius (float): The radius of the sphere.
        sample (int): The number of points to generate.

    Returns:
        np.ndarray: A numpy array of shape (sample, 3) containing the generated points.

    """
    points = [[0, 0, 0] for _ in range(sample)]
    for n in range(sample):  # formula
        phi = np.arccos(-1.0 + (2.0 * (n + 1) - 1.0) / sample)
        theta = np.sqrt(sample * np.pi) * phi
        points[n][0] = radius * np.cos(theta) * np.sin(phi)
        points[n][1] = radius * np.sin(theta) * np.sin(phi)
        points[n][2] = radius * np.cos(phi)
    return np.array(points)


def render(obj_path: str, diag_length: float, save_path: str, flag: bool):
    """
    This function is used to render images of the given object with BlenderProc.

    Args:
        obj_path (str): The path to the object file.
        diag_length (float): The length of the diagonal of the bounding box of the object.
        save_path (str): The path to the folder where the images should be saved.
        flag (bool): A boolean indicating whether to render depth images or not.

    Returns:
        None

    """
    cam_points = sample_points(radius=diag_length * 3, sample=400)
    # The core function for rendering images, render a colour and depth image
    # for each camera location
    bproc.init()

    # load the objects into the scene
    objs = bproc.loader.load_obj(obj_path)

    # define a light and set its location and energy level
    light = bproc.types.Light()

    # define the camera resolution
    bproc.camera.set_resolution(256, 256)

    # locate object center
    # poi = bproc.object.compute_poi(objs)
    poi = np.array([0, 0, 0])
    # activate depth rendering
    if flag:
        bproc.renderer.enable_depth_output(activate_antialiasing=False)
        K = bproc.camera.get_intrinsics_as_K_matrix()
        with open(os.path.join("./camera_intrinsics.txt"), 'wb') as cam:
            np.savetxt(cam, K)

    for num, cam_location in tqdm(enumerate(cam_points), desc="Rendering images for" + save_path.split("/")[-1]):

        # set light
        light.set_location(cam_location)
        light.set_type("POINT")
        light.set_energy(20)

        # set camera
        rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - cam_location)
        cam2world_matrix = bproc.math.build_transformation_mat(cam_location, rotation_matrix)
        bproc.camera.add_camera_pose(cam2world_matrix)

        # render the whole pipeline
        data = bproc.renderer.render()

        # write the data to a .hdf5 container
        bproc.writer.write_hdf5(os.path.join(save_path, str(num)), data)

        # reset the scene (clear the camera and light)
        bproc.utility.reset_keyframes()

        with h5py.File(os.path.join(save_path, str(num), '0.hdf5'), 'r') as h5f:
            colours = np.array(h5f["colors"])[..., ::-1].copy()
            cv2.imwrite(os.path.join(save_path, f'color-{num}.png'), colours)
            with open(os.path.join(save_path, f'depth-{num}.png'), 'wb') as im:
                float_arr = np.array(h5f["depth"])
                mask = np.array(h5f["depth"]) < 100
                int_arr = float_arr * mask * 10000
                writer = png.Writer(width=256, height=256, bitdepth=16, greyscale=True)
                writer.write(im, int_arr.astype(np.int16))
            # with open(os.path.join(mask_path, f'mask-{num}'), 'wb') as dat:
            # savearr = np.array(h5f["depth"]) < 100
            # np.save(dat, savearr)
            with open(os.path.join(save_path, f'matrix-{num}.txt'), 'wb') as dat:
                np.savetxt(dat, cam2world_matrix)
        shutil.rmtree(os.path.join(save_path, str(num)))


if __name__ == "__main__":
    # Get the paths to object model and the model size
    paths, diags = calc_box('/media/iai-lab/wanqing/YCB_Video_Dataset/models')
    first_execution_flag = True
    for obj_path, diag_l in zip(paths, diags):
        # using a wrong file to protect generated files
        save_path = create_output_path(obj_path, output_folder=obj_path.split('models')[0] + 'YCB_objects')
        render(obj_path, diag_l, save_path, first_execution_flag)
        first_execution_flag = False
