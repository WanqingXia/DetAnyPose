import numpy as np
import scipy
import os

"""
Author： Wanqing Xia
Email: wxia612@aucklanduni.ac.nz

This script is used to search the closest generated image with an pose from YCB video dataset
"""


def angle_between_vectors(v1, v2):
    """Calculate the angle in degrees between vectors 'v1' and 'v2'."""
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    angle_radians = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    return np.degrees(angle_radians)


# Load the pose from YCB video dataset
mat = scipy.io.loadmat('000001-meta.mat')
data = mat['poses']
rotation_matrix_ori = np.array(data[:3, :3, 1])

# Select the folder by filename
dir = '/home/wanqing/YCB_Video_Dataset/YCB_objectstest/003_cracker_box'
# List all files in the directory
files = sorted(os.listdir(dir))
# Filter out files that end with '.txt'
txt_files = [file for file in files if file.endswith('.txt')]

# Find the closest pose
tmp = [1000, 1000, 1000]
angle_diff = [0, 0, 0]
file = ''
for txt in txt_files:
    rotation_matrix_gen = np.loadtxt(os.path.join(dir, txt))[:3, :3]
    angle_diff = [angle_between_vectors(rotation_matrix_ori[0, :], rotation_matrix_gen[0, :]),
                  angle_between_vectors(rotation_matrix_ori[1, :], rotation_matrix_gen[1, :]),
                  angle_between_vectors(rotation_matrix_ori[2, :], rotation_matrix_gen[2, :])]

    if np.abs(angle_diff[2]) < np.abs(tmp[2]):
        tmp = angle_diff
        file = txt

print(f"Smallest Angular Differences in X, Y, Z: {tmp} in file {file} \n")
