import numpy as np

"""
This script is to calculate the angles between a point and its neighbouring points
need 100+ points to keep the angles around 20 degrees
need 400+ points to keep the angles around 10 degrees

The logic in this script can also be used to select neighbouring pictures later
"""

camera_path = './positions.txt'
output_path = './angles.txt'

num_lines = sum(1 for line in open(camera_path))
lines = np.zeros((num_lines, 3))
out = np.zeros((num_lines, 7))

with open(camera_path, "r") as f:
    for num, line in enumerate(f.readlines()):
        line = [float(x) for x in line.split()]
        lines[num, :] = line

for num, line in enumerate(lines):
    angles = np.ones(4) * 100
    for count, others in enumerate(lines):
        angle = np.arccos(np.clip(np.dot(line / np.linalg.norm(line), others / np.linalg.norm(others)), -1.0, 1.0))
        if angle < 0.02:
            pass
        elif angle < np.max(angles):
            angles[np.where(angles == np.max(angles))[0][0]] = angle

    angles = (angles * 180) / np.pi
    out[num, :3] = line
    out[num, 3:] = np.sort(angles)

np.savetxt(output_path, out, fmt='%1.4f')
