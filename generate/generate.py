#!/usr/bin/python3
# -*- encoding: utf-8 -*-

import blenderproc as bproc
import numpy as np
import sys
import os
import cv2
import h5py
import shutil
import png

"""
Author： Wanqing Xia
Email: wxia612@aucklanduni.ac.nz

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


def invert_rt_matrix(rt_matrix):
    # Extract the rotation matrix (upper-left 3x3)
    R = rt_matrix[:3, :3]

    # Extract the translation vector (top three elements of the fourth column)
    T = rt_matrix[:3, 3]

    # Invert the rotation (transpose)
    R_inv = R.T

    # Invert the translation
    T_inv = -np.dot(R_inv, T)

    # Construct the inverse RT matrix
    rt_matrix_inv = np.eye(4)
    rt_matrix_inv[:3, :3] = R_inv
    rt_matrix_inv[:3, 3] = T_inv

    return rt_matrix_inv


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


def fibonacci_sample_sphere(radius: float, samples: int) -> np.ndarray:
    """
    Generates points on the surface of a sphere using the Fibonacci method.
    :param samples: Number of points to generate
    :param radius: Radius of the sphere
    :return: List of points on the sphere surface
    """
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius_at_y = np.sqrt(1 - y * y) * radius  # radius at y, scaled by the desired radius

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius_at_y
        z = np.sin(theta) * radius_at_y
        y *= radius  # scale y coordinate by the desired radius

        points.append((x, y, z))

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
    origin = [0, 0, 0]
    # load the objects into the scene
    objs = bproc.loader.load_obj(obj_path)

    # Sample camera locations by fibonacci lattice
    cam_points = fibonacci_sample_sphere(radius=diag_length * 6, samples=4000)

    # define a light and set its location and energy level
    light = bproc.types.Light()

    # Set the camera intrinsics to match YCB video dataset
    image_size = 256
    camera_intrinsics = np.array([
        [1078, 0, (image_size-1)/2],
        [0, 1078, (image_size-1)/2],
        [0, 0, 1]
    ])
    bproc.camera.set_intrinsics_from_K_matrix(camera_intrinsics, image_size, image_size)

    # Change coordinate frame of transformation matrix from OpenCV to Blender coordinates
    cam2world = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    cam2world = bproc.math.change_source_coordinate_frame_of_transformation_matrix(cam2world, ["X", "-Y", "-Z"])

    # activate depth rendering
    if flag:
        bproc.renderer.enable_depth_output(activate_antialiasing=False)

    for num, cam_location in enumerate(cam_points):
        # set light
        light.set_location(origin)
        light.set_type("POINT")
        light.set_energy(40)

        # get the camera rotation and translation matrix
        rotation_matrix = bproc.camera.rotation_from_forward_vec(np.array(origin) - cam_location)
        cam2world_matrix = bproc.math.build_transformation_mat(cam_location, rotation_matrix)
        # convert camera RT matrix to object RT matrix (exchange the place of camera and object)
        obj_matrix = invert_rt_matrix(cam2world_matrix)
        # Making the object stay in the middle
        obj_matrix[:3, 3] = [0, 0, diag_length * 6]
        # set object location and rotation
        objs[0].set_local2world_mat(obj_matrix)

        # set camera
        bproc.camera.add_camera_pose(cam2world)

        # render the whole pipeline
        data = bproc.renderer.render()

        # write the data to a .hdf5 container
        bproc.writer.write_hdf5(os.path.join(save_path, str(num)), data)

        # reset the scene (clear the camera and light)
        bproc.utility.reset_keyframes()

        with h5py.File(os.path.join(save_path, str(num), '0.hdf5'), 'r') as h5f:
            padded_num = "{:04d}".format(num)
            colours = np.array(h5f["colors"])[..., ::-1].copy()
            with open(os.path.join(save_path, f'{padded_num}-depth.png'), 'wb') as im:
                depth_original = np.array(h5f["depth"])
                mask = np.array(h5f["depth"]) < 100
                depth_enlarged = depth_original * mask * 10000
                writer = png.Writer(width=256, height=256, bitdepth=16, greyscale=True)
                writer.write(im, depth_enlarged.astype(np.int16))
                masked_img = colours * np.expand_dims(mask, axis=-1)
                cv2.imwrite(os.path.join(save_path, f'{padded_num}-color.png'), masked_img)
            with open(os.path.join(save_path, f'{padded_num}-matrix.txt'), 'wb') as dat:
                np.savetxt(dat, cam2world_matrix)
        shutil.rmtree(os.path.join(save_path, str(num)))


if __name__ == "__main__":
    paths, diags = [], []
    # Get the paths to object model and the model size
    with open(os.path.dirname(os.path.abspath(__file__)) + "/diameters.txt", "r") as f:
        for line in f:
            # Split the line at the first space
            parts = line.split(maxsplit=1)
            paths.append(parts[0])
            diags.append(parts[1].strip())

    first_execution_flag = True
    bproc.init()
    for obj_path, diag_l in zip(paths, np.float64(diags)):
        print(obj_path, diag_l)
        # create an output folder for each object model
        save_path = create_output_path(obj_path, output_folder=obj_path.split('models')[0] + 'YCB_objects')
        # render image for each view point
        render(obj_path, diag_l, save_path, first_execution_flag)
        first_execution_flag = False
        bproc.clean_up()
