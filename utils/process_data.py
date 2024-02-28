import glob
import hashlib
import logging
import os
import random
import shutil

from pathlib import Path
import scipy

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
import warnings

from concurrent.futures import ProcessPoolExecutor, as_completed
import scipy.io
import cProfile


def get_hash(paths):
    # Returns a single hash value of a list of paths (files or dirs)
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.md5(str(size).encode())  # hash sizes
    h.update(''.join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash

def angle_between_vectors(v1, v2):
    """Calculate the angle in degrees between vectors 'v1' and 'v2'."""
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    angle_radians = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    return np.degrees(angle_radians)


def create_folder(path):
    # Create folder
    if os.path.exists(path):
        pass  # delete output folder
    else:
        os.makedirs(path)  # make new output folder
    return path

class Process:
    def __init__(self, path, img_size=256):
        self.img_size = img_size
        self.path = Path(path)
        self.test_categories = ['006_mustard_bottle', '019_pitcher_base', '021_bleach_cleanser']
        self.train_categories = ['002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can',
                                 '008_pudding_box', '011_banana', '024_bowl', '025_mug', '036_wood_block',
                                 '037_scissors', '061_foam_brick']
        self.folders = ['0001', '0004', '0007', '0013', '0020', '0021', '0031', '0041',
                        '0051', '0055', '0071', '0074', '0076', '0078', '0082', '0084',
                        '0091']  # current camera setting only match images before 60
        self.data_paths = [(self.path / 'data') / subpath for subpath in self.folders]
        self.gen_paths = sorted([p for p in Path(self.path / 'YCB_objects').glob('*') if p.is_dir()])
        self.save_path = self.path / 'YCB_pairs'
        self.train_data_path = create_folder(path=(self.save_path / 'train_data'))
        self.test_data_path = create_folder(path=(self.save_path / 'test_data'))
        self.gen_folders_names = [str(p.name) for p in self.gen_paths]
        self.obj_names = sorted([p.name for p in Path(self.path / 'models').glob('*') if p.is_dir()])
        self.train_data, self.test_data = {}, {}

        try:
            f = []  # original text files
            g = []  # generated text files
            for p in self.data_paths:
                f += sorted(list(p.rglob('*.txt')))
            self.txt_files = sorted(f)
            for p in self.gen_paths:
                g += sorted(list(p.rglob('*.txt')))
            self.gen_txt_files = sorted(g)
            assert self.txt_files, f'No text label file found, check your data path'
            assert self.gen_txt_files, f'No generated text file found, check your YCB_objects path'
        except Exception as e:
            raise Exception(f'Error loading data from {path}: {e}\n')

        train_temp = Path(__file__).parent / 'train_temp_data.cache.npy'
        test_temp = Path(__file__).parent / 'test_temp_data.cache.npy'
        try:
            self.train_data = np.load(train_temp, allow_pickle=True).item()
            self.test_data = np.load(test_temp, allow_pickle=True).item()
        except:
            self.process_images()
            np.save(train_temp, self.train_data)  # save cache for next time
            np.save(test_temp, self.test_data)  # save cache for next time

        train_cache_path = Path(__file__).parent.parent / 'cache' / 'train.cache'
        test_cache_path = Path(__file__).parent.parent / 'cache' / 'test.cache'
        self.train_save, self.test_save = [], []

        workers = 18  # Number of workers, maxed at the number of threads or the dataset size
        # Process training data
        with ProcessPoolExecutor(max_workers=workers) as executor:
            # Prepare data for processing
            train_tasks = [(ori, gen) for ori, gen in list(self.train_data.items())]
            # Process data in parallel
            futures = [executor.submit(self.allocate_data, ori, gen, self.train_data_path) for ori, gen in train_tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing training data"):
                result = future.result()
                self.train_save.append(result)

        # Save processed training data
        np.save(train_cache_path, self.train_save)
        train_cache_path.with_suffix('.cache.npy').rename(train_cache_path)  # remove .npy suffix

        # Repeat the process for test data
        with ProcessPoolExecutor(max_workers=workers) as executor:
            test_tasks = [(ori, gen) for ori, gen in list(self.train_data.items())]
            futures = [executor.submit(self.allocate_data, ori, gen, self.test_data_path) for ori, gen in test_tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing testing data"):
                result = future.result()
                self.train_save.append(result)

        # Save processed testing data
        np.save(test_cache_path, self.test_save)
        test_cache_path.with_suffix('.cache.npy').rename(test_cache_path)  # remove .npy suffix

    def process_images(self):
        print('Scanning YCB_Video_Dataset/YCB_objects images and labels...')
        gen_txt_content = self.read_folders_contents()
        print('Scanning YCB_Video_Dataset/YCB_objects finished')
        for data_path in tqdm(self.txt_files, desc='Scanning YCB_Video_Dataset/data images and labels...',
                              total=len(self.txt_files)):
            txt_file, mat_file = self.verify_data_paths(data_path)
            if txt_file:
                mat = scipy.io.loadmat(mat_file)
                with open(txt_file, 'r') as file:
                    labels = file.readlines()
                    for num, label in enumerate(labels):
                        obj_name = label.split(' ')[0]
                        gen_txt_path = self.search_imgs(gen_txt_content[obj_name],
                                                        np.array(mat['poses'][:3, :3, num]))
                        obj_save_path = str(txt_file / obj_name)
                        gen_save_path = str(gen_txt_path)
                        if obj_name in self.train_categories:
                            self.train_data[obj_save_path] = gen_save_path
                        elif obj_name in self.test_categories:
                            self.test_data[obj_save_path] = gen_save_path
                        else:
                            raise Exception(f'Unrecognised object name: {obj_name} \n')

    def read_folders_contents(self):
        """
        Reads the contents of text files in the given list of folders and returns a dictionary.
        The keys of the dictionary are the folder names, and the values are lists of tuples.
        Each tuple consists of a file path and its content.

        Parameters:
        folders (list): A list of folder paths to read text files from.

        Returns:
        dict: A dictionary where each key is a folder name, and each value is a sub-dictionary.
              Each sub-dictionary contains the path to a text file as key and its content as value.
        """
        all_folders_contents = {}
        for folder in self.gen_paths:
            folder_name = folder.name  # Extracts the last part of the path as the folder name
            folder_contents = {}
            if os.path.exists(folder):
                for filename in sorted(list(folder.rglob('*.txt'))):
                    file_path = folder / filename
                    if os.path.isfile(file_path):
                        folder_contents[file_path] = np.loadtxt(file_path)
                print(f'load data from {folder} finished, {len(folder_contents)}/4000 data loaded')
                all_folders_contents[folder_name] = folder_contents
            else:
                print(f"The folder {folder} does not exist.")

        return all_folders_contents

    def verify_data_paths(self, file_path):
        """
        Verifies that the given file path corresponds to a valid image-label pair.

        Args:
            file_path (Path): The path to the image-label pair.

        Returns:
            Tuple[Path, Path, int, int, List[str]]: A tuple containing the following values:

                - The path to the image-label pair if it is valid.
                - The path to the generated text file if it exists.
                - The number of valid image-label pairs found.
                - The number of missing image-label pairs.
                - A list of any warnings or error messages.

        Raises:
            ValueError: If the given file path is not a file.
        """
        base_name = file_path.name.split('-')[0]
        # Create the new file name
        metad_file_path = file_path.parent / (base_name + '-meta.mat')

        return file_path, metad_file_path
        color_img_path = file_path.parent / (base_name + '-color.png')
        depth_img_path = file_path.parent / (base_name + '-depth.png')
        label_img_path = file_path.parent / (base_name + '-label.png')

        # Verify one image-label pair
        if os.path.isfile(file_path):
            if os.path.isfile(color_img_path):
                if os.path.isfile(depth_img_path):
                    if os.path.isfile(label_img_path):
                        if os.path.isfile(metad_file_path):
                            pass
                        else:
                            warnings.warn('matlab mat file {} missing'.format(metad_file_path), UserWarning)
                    else:
                        warnings.warn('label image file {} missing'.format(label_img_path), UserWarning)
                else:
                    warnings.warn('depth image file {} missing'.format(depth_img_path), UserWarning)
            else:
                warnings.warn('color image file {} missing'.format(color_img_path), UserWarning)
        else:
            warnings.warn('txt file {} missing'.format(file_path), UserWarning)

        return file_path, metad_file_path

    def allocate_data(self, ori_path, gen_path, save_path):
        folder_path = self.load_image(Path(ori_path), Path(gen_path), save_path)
        return folder_path

    def load_image(self, txt_path_obj, gen_txt_path, save_path):
        """
        loads image from dataset index

        Returns:
            tuple: tuple containing the following values:

                - color_isolated (numpy.ndarray): isolated color image of the object of interest
                - depth_isolated (numpy.ndarray): isolated depth image of the object of interest
                - pose_ori (numpy.ndarray): original pose of the object of interest
                - txt_path (str): path to the text file containing the object's pose and other metadata
                - color_gen (numpy.ndarray): generated color image of a different object
                - depth_gen (numpy.ndarray): generated depth image of a different object
                - pose_gen (numpy.ndarray): generated pose of the object of interest
                - gen_txt_path (str): path to the generated text file containing the generated pose and other metadata
                - neg_color (numpy.ndarray): negative color image of a different object
                - neg_depth (numpy.ndarray): negative depth image of a different object
                - neg_txt_path (str): path to the negative text file containing the negative pose and other metadata
                - obj_name (str): name of the object

        """
        # Extract the object name from the path of the current text file.
        obj_name = txt_path_obj.name
        # Get the parent directory of the current text file.
        txt_path = txt_path_obj.parent
        # Extract the base name of the dataset or image set from the directory name.
        base_name = txt_path.name.split('-')[0]
        new_path = create_folder(save_path / (txt_path.parent.name + '-' + base_name + '-' + obj_name))
        new_path = Path(new_path)

        # Construct paths for the associated color, depth, label images, and metadata file, using the base name.
        color_img_path = txt_path.parent / (base_name + '-color.png')
        depth_img_path = txt_path.parent / (base_name + '-depth.png')
        label_img_path = txt_path.parent / (base_name + '-label.png')
        metad_file_path = txt_path.parent / (base_name + '-meta.mat')

        # Load the color, depth, and label images as NumPy arrays.
        color_image = np.array(Image.open(color_img_path))
        depth_image = np.array(Image.open(depth_img_path))
        label_image = np.array(Image.open(label_img_path))
        # Load the metadata file which contains additional information like object poses.
        mat = scipy.io.loadmat(metad_file_path)

        # Find the index of the current object in a predefined list of object names, adjusting index to start from 1.
        obj_num = self.obj_names.index(obj_name) + 1
        # Find the index of the current object in the metadata's class indexes.
        obj_index = list(mat['cls_indexes']).index(obj_num)
        # Extract the pose information for the current object from the metadata.
        pose_ori = np.array(mat['poses'][:3, :, obj_index])

        # Create a mask where the label image matches the current object number.
        isolated_mask = (label_image == obj_num)
        # Isolate the color and depth images using the generated mask.
        color_isolated = self.isolate_image(color_image, isolated_mask, color_img_path)
        depth_isolated = self.isolate_image(depth_image, isolated_mask, depth_img_path)
        Image.fromarray(color_isolated.astype('uint8'), 'RGB').save(new_path / 'color_original.png')
        Image.fromarray(depth_isolated.astype('I')).save(new_path / 'depth_original.png')
        np.savetxt(new_path / 'pose_original.txt', pose_ori)

        # For generated data, determine the paths for color and depth images, and load the pose information.
        color_img_path = gen_txt_path.parent / (gen_txt_path.name.split('-')[0] + '-color.png')
        depth_img_path = gen_txt_path.parent / (gen_txt_path.name.split('-')[0] + '-depth.png')
        pose_gen = np.loadtxt(gen_txt_path)[:3, :]
        shutil.copy(color_img_path, new_path / 'color_generated.png')
        shutil.copy(depth_img_path, new_path / 'depth_generated.png')
        np.savetxt(new_path / 'pose_generated.txt', pose_gen)

        # Retrieve negative sample images (i.e., images not containing the object of interest) for the current object.
        neg_color_path, neg_depth_path, neg_txt_path = self.get_negative_imgs(obj_name)
        shutil.copy(neg_color_path, new_path / 'color_negative.png')
        shutil.copy(neg_depth_path, new_path / 'depth_negative.png')

        with open(new_path / 'name_and_path.txt', 'w') as f:
            f.write(obj_name)
            f.write(str(txt_path))
            f.write(str(gen_txt_path))
            f.write(str(neg_txt_path))

        return Path(new_path)

    def search_imgs(self, gen_poses, pose_ori):
        """
        Find the closest pose in the generated poses list to the original pose.

        Args:
            gen_poses (dict): A dictionary containing the generated poses and their corresponding file names.
            pose_ori (numpy.ndarray): The original pose of the object of interest.

        Returns:
            str: The file name of the generated pose that is closest to the original pose.

        """
        # Find the closest pose
        tmp = [1000, 1000, 1000]
        file = ''
        for file_name, file_content in gen_poses.items():
            angle_diff = [angle_between_vectors(pose_ori[0, :], file_content[0, :3]),
                          angle_between_vectors(pose_ori[1, :], file_content[1, :3]),
                          angle_between_vectors(pose_ori[2, :], file_content[2, :3])]
            if np.abs(angle_diff[2]) < np.abs(tmp[2]):
                tmp = angle_diff
                file = file_name
        return file

    def get_negative_imgs(self, obj_name):
        """
        Returns a negative image and its metadata for a given object name.

        Parameters:
        obj_name (str): The name of the object for which to retrieve negative images.

        Returns:
        tuple: A tuple containing the following values:

            - neg_color_gen (numpy.ndarray): A negative color image of a different object.
            - neg_depth_gen (numpy.ndarray): A negative depth image of a different object.
            - neg_txt_path (str): The path to the negative text file containing the negative pose and other metadata.

        """
        files = self.gen_folders_names.copy()
        files.remove(obj_name)
        neg_obj = random.choice(files)
        txt_files = sorted((self.path / 'YCB_objects' / neg_obj).rglob('*.txt'))
        # Filter out files that end with '.txt'
        neg_file = random.choice(txt_files)
        neg_txt_path = self.path / 'YCB_objects' / neg_obj / neg_file
        neg_color_img_path = neg_txt_path.parent / (neg_txt_path.name.split('-')[0] + '-color.png')
        neg_depth_img_path = neg_txt_path.parent / (neg_txt_path.name.split('-')[0] + '-depth.png')

        return neg_color_img_path, neg_depth_img_path, neg_txt_path

    def isolate_image(self, input_img, mask, img_file):
        """
        Isolates an object of interest from a color or depth image using a binary mask.

        Args:
            input_img (np.ndarray): The color or depth image from which to isolate the object.
            mask (np.ndarray): A binary mask indicating the region of interest.
            img_file (str): The path to the image file.

        Returns:
            np.ndarray: The isolated object.

        Raises:
            Exception: If an error occurs while isolating the object.

        """
        if input_img.ndim == 2:  # depth image
            # Create a new 256x256 numpy array filled with 0
            background = np.zeros((self.img_size, self.img_size))
            # Use the mask to find the ROI in the color image
            # This creates a masked version of the image where unmasked areas are set to 0
            masked_image = input_img * mask

        else:  # color image
            # Create a new 256x256x3 numpy array filled with 0
            background = np.zeros((self.img_size, self.img_size, 3))
            # Use the mask to find the ROI in the color image
            # This creates a masked version of the image where unmasked areas are set to 0
            masked_image = input_img * np.expand_dims(mask, axis=-1)

        # Find the bounding box of the masked area to extract only the relevant part
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if rows.size == 0:
            print("The 'rows' array is empty.")
            stop = 1
        if cols.size == 0:
            print("The 'cols' array is empty.")
            stop = 1
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # Extract the ROI
        roi = masked_image[rmin:rmax + 1, cmin:cmax + 1]
        max_dimension = 200
        roi_height, roi_width = roi.shape[:2]
        if roi_width > max_dimension or roi_height > max_dimension:
            if roi_width > roi_height:
                new_width = max_dimension
                new_height = int((max_dimension / roi_width) * roi_height)
            else:
                new_height = max_dimension
                new_width = int((max_dimension / roi_height) * roi_width)
            roi_width = new_width
            roi_height = new_height
            if input_img.ndim == 2:
                roi_img = Image.fromarray(roi.astype('I'))
            else:
                roi_img = Image.fromarray(roi.astype('uint8'), 'RGB')

            roi_img = roi_img.resize((roi_width, roi_height))
            roi = np.array(roi_img)

        # Calculate where to place the ROI in the center of the background
        start_x = (self.img_size - roi_width) // 2
        start_y = (self.img_size - roi_height) // 2
        end_x = start_x + roi_width
        end_y = start_y + roi_height

        try:
            # Copy the ROI into the center of the background
            if input_img.ndim == 2:
                background[start_y:end_y, start_x:end_x] = roi
            else:
                background[start_y:end_y, start_x:end_x, :] = roi
        except Exception as e:
            raise Exception(f'Error isolating the image {img_file}: {e}\n')
        return background


if __name__ == '__main__':
    dataroot = '/media/iai-lab/wanqing/YCB_Video_Dataset'
    Process(dataroot, img_size=256)
