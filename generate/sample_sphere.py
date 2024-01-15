import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""
This is the script to sample camera points on a sphere
Use the function by input the file path to save all points, sphere radius and number of points
"""

def sample_points(path, radius, sample):
    points = [[0, 0, 0] for _ in range(sample)]

    for n in range(sample):
        phi = np.arccos(-1.0 + (2.0 * (n + 1) - 1.0) / sample)
        theta = np.sqrt(sample * np.pi) * phi
        points[n][0] = radius * np.cos(theta) * np.sin(phi)
        points[n][1] = radius * np.sin(theta) * np.sin(phi)
        points[n][2] = radius * np.cos(phi)

    points = np.array(points)
    savepath = os.path.join(path,"positions.txt")
    np.savetxt(savepath, points, fmt='%1.4f')
    ##################################################################
    # This block of the code is used to visualise the sampled sphere
    # fig = plt.figure("sphere points uniform")
    # ax = fig.add_subplot(111, projection='3d')
    # ax.set_aspect('equal')
    #
    # u = np.linspace(0, 2 * np.pi, 100)
    # v = np.linspace(0, np.pi, 100)
    # x = (radius-0.01) * np.outer(np.cos(u), np.sin(v))
    # y = (radius-0.01) * np.outer(np.sin(u), np.sin(v))
    # z = (radius-0.01) * np.outer(np.ones(np.size(u)), np.cos(v))
    #
    # ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='k', linewidth=1, antialiased=False)
    # ax.scatter(points[:,0], points[:,1], points[:,2], color='r')
    # ax.scatter(points[:,0], points[:,1], -points[:,2], color='r')
    #
    # plt.show()
    ####################################################################
    return savepath

if __name__ == "__main__":
    savepath = sample_points('./', radius=0.5, sample=400)
