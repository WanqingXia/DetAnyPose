import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import distance_matrix

"""
Author： Wanqing Xia
Email: wxia612@aucklanduni.ac.nz

This is the script to calculate the min, mean, max angular distance between camera points sampled 
on a sphere surrounding the object, helps us to determine the sampling density
"""


def fibonacci_sphere(samples=1, radius=1):
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

    return points


def angular_distance(point1, point2):
    """
    Calculate the angular distance in degrees between two points on a sphere.
    """
    inner_product = np.dot(point1, point2) / (np.linalg.norm(point1) * np.linalg.norm(point2))
    angle_rad = np.arccos(np.clip(inner_product, -1.0, 1.0))
    return np.degrees(angle_rad)


if __name__ == "__main__":
    # Generate 4000 points
    radius = 5
    points = fibonacci_sphere(42, radius)  # radius is 1 for simplicity

    # Calculate distance matrix
    dist_matrix = distance_matrix(points, points)

    # Sort each row in the distance matrix and take the distances to the 5 nearest neighbors
    nearest_dists = np.sort(dist_matrix, axis=1)[:, 1:6]

    # Calculate the angular distances for each point to its 5 nearest neighbors
    angular_dists = []
    for i in range(len(points)):
        for j in range(5):
            neighbor_idx = np.where(dist_matrix[i] == nearest_dists[i, j])[0][0]
            angular_dists.append(angular_distance(points[i], points[neighbor_idx]))

    # Calculate min, max, and mean of the angular distances
    min_angular_dist = np.min(angular_dists)
    max_angular_dist = np.max(angular_dists)
    mean_angular_dist = np.mean(angular_dists)
    print("min angular distance: ", min_angular_dist)
    print("max angular distance:", max_angular_dist)
    print("mean angular distance: ", mean_angular_dist)

    # Unpack points for plotting
    xs, ys, zs = zip(*points)

    # Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Draw sphere
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    x = radius * np.cos(u) * np.sin(v)
    y = radius * np.sin(u) * np.sin(v)
    z = radius * np.cos(v)
    ax.plot_wireframe(x, y, z, color="r", alpha=0.1)

    # Draw points
    ax.scatter(xs, ys, zs, color="b", s=1)

    # Labels and show
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Fibonacci Sphere with Points')
    plt.show()
