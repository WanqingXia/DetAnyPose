import numpy as np
import scipy
import os
import time

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
txt = open('/media/iai-lab/wanqing/YCB_Video_Dataset/data/0001/000006-box.txt', 'r')
cont = txt.readlines()
mat = scipy.io.loadmat('/media/iai-lab/wanqing/YCB_Video_Dataset/data/0001/000006-meta.mat')
data = mat['poses']
rotation_matrix_ori = np.array(data[:3, :3, 2])

# Select the folder by filename
dir = '/media/iai-lab/wanqing/YCB_Video_Dataset/YCB_objects/004_sugar_box'
# List all files in the directory
files = sorted(os.listdir(dir))
# Filter out files that end with '.txt'
txt_files = [file for file in files if file.endswith('.txt')]

# Find the closest pose
tmp = [1000, 1000, 1000]
angle_diff = [0, 0, 0]
file = ''
time1, time2, time3 = 0, 0, 0
for txt in txt_files:
    tic = time.time()
    rotation_matrix_gen = np.loadtxt(os.path.join(dir, txt))
    toc = time.time()
    angle_diff = [angle_between_vectors(rotation_matrix_ori[0, :], rotation_matrix_gen[0, :3]),
                  angle_between_vectors(rotation_matrix_ori[1, :], rotation_matrix_gen[1, :3]),
                  angle_between_vectors(rotation_matrix_ori[2, :], rotation_matrix_gen[2, :3])]
    # print(angle_diff)
    tac = time.time()
    if np.abs(angle_diff[2]) < np.abs(tmp[2]):
        tmp = angle_diff
        file = txt
    tbc = time.time()
    time1 += toc - tic
    time2 += tac - toc
    time3 += tbc - tac

print("time taken1 " + str(time1))
print("time taken2 " + str(time2))
print("time taken3 " + str(time3))
print(f"Smallest Angular Differences in X, Y, Z: {tmp} in file {file} \n")
