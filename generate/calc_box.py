import os
import numpy as np

"""
This script is used to calculate the diagonal length of the bounding boxes
of each object, the diagonal length will be used to control the camera distance
in generate.py, the calculated length and object path will be saved to "diameters.txt"
"""


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


def calc_box(dataPath):
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


if __name__ == "__main__":
    paths, diags = calc_box('/media/iai-lab/wanqing/YCB_Video_Dataset/models')
    with open(os.path.dirname(os.path.abspath(__file__)) + "/diameters.txt", "w") as f:
        for path, diag in zip(paths, diags):
            f.write("%s %f\n" % (path, diag))

