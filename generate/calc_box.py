import os
import numpy as np
from vedo import *

"""
Author： Wanqing Xia
Email: wxia612@aucklanduni.ac.nz

This script is used to calculate the diagonal length of the bounding boxes
of each object, the diagonal length will be used to control the camera distance
in generate.py, the calculated length and object path will be saved to "diameters.txt"
This script cannot be used inside generate.py unless the 'vedo' package is installed in Blender
"""


def calc_box(dataPath):
    folders = [d for d in sorted(os.listdir(dataPath)) if os.path.isdir(os.path.join(dataPath, d))]
    filepaths = []
    paths = []
    diags = []

    for folder in folders:
        folder_path = os.path.join(dataPath, folder)
        files = os.listdir(folder_path)
        for file in files:
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path) and file[-3:] == "obj" and file[-10:-4] != "simple":
                filepaths.append(file_path)

    for file in filepaths:
        mesh = Mesh(file)
        paths.append(file)
        diags.append(round(mesh.diagonal_size(), 3))
    print("Finished calculating diagonal length")
    return paths, diags


if __name__ == "__main__":
    paths, diags = calc_box('/home/wanqing/YCB_Video_Dataset/models')
    names = ['052_extra_large_clamp']
    with open(os.path.dirname(os.path.abspath(__file__)) + "/diameters.txt", "w") as f:
        for path, diag in zip(paths, diags):
            if not any(substring in path for substring in names):
                f.write("%s %.3f\n" % (path, diag))
