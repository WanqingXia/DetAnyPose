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

def get_hash(paths):
    # Returns a single hash value of a list of paths (files or dirs)
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.md5(str(size).encode())  # hash sizes
    h.update(''.join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash


def create_dataloader(path, type, imgsz, batch_size, workers=8):
    dataset = LoadImagesAndLabels(path, type, imgsz, batch_size)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=workers,
                                             sampler=torch.utils.data.RandomSampler(dataset),
                                             pin_memory=True,
                                             )
    return dataloader, dataset


class LoadImagesAndLabels(Dataset):  # for training/testing
    def __init__(self, path, type, img_size=256, batch_size=16):
        self.img_size = img_size
        self.batch_size = batch_size
        self.path = Path(path)
        assert type in ('train', 'test'), f'{type} is not train or test'
        self.type = type
        self.test_categories = ['006_mustard_bottle', '019_pitcher_base', '021_bleach_cleanser']
        self.train_categories = ['002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can',
                                 '008_pudding_box', '011_banana', '024_bowl', '025_mug', '036_wood_block',
                                 '037_scissors', '061_foam_brick']
        self.folders = ['0001', '0004', '0007', '0013', '0020', '0021', '0031', '0041',
                        '0051', '0055', '0071', '0074', '0076', '0078', '0082', '0084', '0091']  # current camera setting only match images before 60
        self.data_paths = [(self.path / 'data') / subpath for subpath in self.folders]
        self.gen_paths = sorted([p for p in Path(self.path / 'YCB_objects').glob('*') if p.is_dir()])
        self.gen_folders_names = [str(p.name) for p in self.gen_paths]
        self.obj_names = sorted([p.name for p in Path(self.path / 'models').glob('*') if p.is_dir()])

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

        # Check cache
        train_cache_path = Path(__file__).parent.parent / 'cache' / 'train_data.cache'
        test_cache_path = Path(__file__).parent.parent / 'cache' / 'test_data.cache'
        try:
            if self.type == 'train':
                train_cache = np.load(train_cache_path, allow_pickle=True).item()
                assert (train_cache['version'] == 1.0 and
                        train_cache['hash'] == get_hash(str(self.txt_files) + str(self.gen_txt_files)))
            elif self.type == 'test':
                test_cache = np.load(test_cache_path, allow_pickle=True).item()
                assert (test_cache['version'] == 1.1 and
                        test_cache['hash'] == get_hash(str(self.txt_files) + str(self.gen_txt_files)))
            exists = True  # load dict
        except:
            train_cache, test_cache = self.cache_labels(train_cache_path, test_cache_path)
            exists = False  # cache

        # Display cache
        current_cache = train_cache if self.type == 'train' else test_cache
        nf, nm, n = current_cache.pop('results')
        if exists:
            d = f"Scanning images and labels for {self.type} dataset... {n} found, {nm} missing"
            tqdm(None, desc=d, total=n, initial=n)  # display cache results
            if current_cache['msgs']:
                logging.info('\n'.join(current_cache['msgs']))  # display warnings

        if self.type == 'train':
            assert n > 0, f'No labels in {train_cache_path}. Cannot train without labels.'
            [train_cache.pop(k) for k in ('hash', 'version', 'msgs')]  # remove items
            train_txt_cache = list(train_cache.keys())
            self.train_txt_files = [self.path / sub_path for sub_path in train_txt_cache]
            train_gen_cache = list(train_cache.values())
            self.train_gen_files = [self.path / sub_path for sub_path in train_gen_cache]
            self.train_indices = range(len(self.train_txt_files))

        elif self.type == 'test':
            assert n > 0, f'No labels in {test_cache_path}. Cannot train without labels.'
            [test_cache.pop(k) for k in ('hash', 'version', 'msgs')]  # remove items
            test_txt_cache = list(test_cache.keys())
            self.test_txt_files = [self.path / sub_path for sub_path in test_txt_cache]
            test_gen_cache = list(test_cache.values())
            self.test_gen_files = [self.path / sub_path for sub_path in test_gen_cache]
            self.test_indices = range(len(self.test_txt_files))

    def __len__(self):
        return len(self.train_txt_files) if self.type == 'train' else len(self.test_gen_files)

    def __getitem__(self, index):
        index = self.train_indices[index] if self.type == 'train' else self.test_indices[index]

        # Load image
        c_ori, d_ori, p_ori, t_p, c_gen, d_gen, p_gen, gt_p, c_neg, d_neg, nt_p, obj_name = self.load_image(index)

        # Convert
        c_ori = c_ori.transpose((2, 0, 1))  # HWC to CHW
        c_ori = np.ascontiguousarray(c_ori).astype(np.float32)
        c_ori = torch.from_numpy(c_ori)

        c_gen = c_gen.transpose((2, 0, 1))  # HWC to CHW
        c_gen = np.ascontiguousarray(c_gen).astype(np.float32)
        c_gen = torch.from_numpy(c_gen)

        c_neg = c_neg.transpose((2, 0, 1))  # HWC to CHW
        c_neg = np.ascontiguousarray(c_neg).astype(np.float32)
        c_neg = torch.from_numpy(c_neg)

        d_ori = d_ori[np.newaxis, :, :]
        d_ori = np.ascontiguousarray(d_ori).astype(np.float32)
        d_ori = torch.from_numpy(d_ori)

        d_gen = d_gen[np.newaxis, :, :]
        d_gen = np.ascontiguousarray(d_gen).astype(np.float32)
        d_gen = torch.from_numpy(d_gen)

        d_neg = d_neg[np.newaxis, :, :]
        d_neg = np.ascontiguousarray(d_neg).astype(np.float32)
        d_neg = torch.from_numpy(d_neg)

        p_ori = torch.from_numpy(p_ori)
        p_gen = torch.from_numpy(p_gen)

        return c_ori, d_ori, p_ori, t_p, c_gen, d_gen, p_gen, gt_p, c_neg, d_neg, nt_p, obj_name

    def cache_labels(self, path_train=Path('./train_data.cache'), path_test=Path('./test_data.cache')):
        """
        Cache dataset labels, check images
        Args:
            path_train (Path, optional): path to train cache file. Defaults to Path('./train_data.cache')
            path_test (Path, optional): path to test cache file. Defaults to Path('./test_data.cache')
        Returns:
            tuple: tuple containing two dictionaries, one for train and one for test, each with keys as image paths and values as generated text paths
        """
        x = {}  # dict for train txt file
        y = {}  # dict for test txt file
        nf, nm, msgs = 0, 0, []  # number found, corrupt, messages in original data

        print('Scanning YCB_Video_Dataset/YCB_objects images and labels...')
        gen_txt_content = self.read_folders_contents()
        print('Scanning YCB_Video_Dataset/YCB_objects finished')

        num_train, num_test = 0, 0
        for data_path in tqdm(self.txt_files, desc='Scanning YCB_Video_Dataset/data images and labels...',
                              total=len(self.txt_files)):
            txt_file, mat_file, f, m, msg = self.verify_data_paths(data_path)
            split_str = str(self.path) + '/'
            nf += f
            nm += m
            if txt_file:
                mat = scipy.io.loadmat(mat_file)
                with open(txt_file, 'r') as file:
                    labels = file.readlines()
                    for num, label in enumerate(labels):
                        obj_name = label.split(' ')[0]
                        gen_txt_path = self.search_imgs(gen_txt_content[obj_name], np.array(mat['poses'][:3, :3, num]))
                        obj_save_path = str(txt_file / obj_name).split(split_str)[1]
                        gen_save_path = str(gen_txt_path).split(split_str)[1]
                        if obj_name in self.train_categories:
                            x[obj_save_path] = gen_save_path
                            num_train += 1
                        elif obj_name in self.test_categories:
                            y[obj_save_path] = gen_save_path
                            num_test += 1
                        else:
                            raise Exception(f'Unrecognised object name: {obj_name} \n')
                if msg:
                    msgs.append(msg)
        if msgs:
            logging.info('\n'.join(msgs))

        print('Scanning YCB_Video_Dataset/data finished')

        x['hash'] = get_hash(str(self.txt_files) + str(self.gen_txt_files))
        x['results'] = nf, nm, num_train
        x['msgs'] = msgs  # warnings
        x['version'] = 1.0  # cache version
        try:
            np.save(path_train, x)  # save cache for next time
            path_train.with_suffix('.cache.npy').rename(path_train)  # remove .npy suffix
            print(f'New cache created: {path_train}')
        except Exception as e:
            print(f'WARNING: Cache directory {path_train.parent} is not writeable: {e}')  # path not writeable

        y['hash'] = get_hash(str(self.txt_files) + str(self.gen_txt_files))
        y['results'] = nf, nm, num_test
        y['msgs'] = msgs  # warnings
        y['version'] = 1.1  # cache version
        try:
            np.save(path_test, y)  # save cache for next time
            path_test.with_suffix('.cache.npy').rename(path_test)  # remove .npy suffix
            print(f'New cache created: {path_test}')
        except Exception as e:
            print(f'WARNING: Cache directory {path_test.parent} is not writeable: {e}')  # path not writeable

        return x, y

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
        color_img_path = file_path.parent / (base_name + '-color.png')
        depth_img_path = file_path.parent / (base_name + '-depth.png')
        label_img_path = file_path.parent / (base_name + '-label.png')
        metad_file_path = file_path.parent / (base_name + '-meta.mat')
        # Verify one image-label pair
        nf, nm, msg = 0, 0, []  # number found, missing
        if os.path.isfile(file_path) and os.path.isfile(color_img_path) and os.path.isfile(depth_img_path) \
                and os.path.isfile(label_img_path) and os.path.isfile(metad_file_path):
            nf = 1
        else:
            nm = 1
            if not os.path.isfile(file_path):
                warnings.warn('txt file missing', UserWarning)
            if not os.path.isfile(color_img_path):
                warnings.warn('color image file missing', UserWarning)
            if not os.path.isfile(depth_img_path):
                warnings.warn('depth image file missing', UserWarning)
            if not os.path.isfile(label_img_path):
                warnings.warn('label image file missing', UserWarning)
            if not os.path.isfile(metad_file_path):
                warnings.warn('matlab mat file missing', UserWarning)
            msg = f'WARNING: Ignoring corrupted image and/or label {file_path}'

        if nf:
            return file_path, metad_file_path, nf, nm, msg
        else:
            return None, None, nf, nm, msg

    def load_image(self, i):
        """
        loads 1 image from dataset index 'i'

        Args:
            i (int): index of the image to be loaded

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
        # Choose the appropriate list of text files based on whether this is for training or testing data.
        txt_path_obj = self.train_txt_files[i] if self.type == 'train' else self.test_txt_files[i]
        # Extract the object name from the path of the current text file.
        obj_name = txt_path_obj.name
        # Get the parent directory of the current text file.
        txt_path = txt_path_obj.parent
        # Extract the base name of the dataset or image set from the directory name.
        base_name = txt_path.name.split('-')[0]

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

        # For generated data, determine the paths for color and depth images, and load the pose information.
        gen_txt_path = self.train_gen_files[i] if self.type == 'train' else self.test_gen_files[i]
        color_img_path = gen_txt_path.parent / (gen_txt_path.name.split('-')[0] + '-color.png')
        depth_img_path = gen_txt_path.parent / (gen_txt_path.name.split('-')[0] + '-depth.png')
        pose_gen = np.loadtxt(gen_txt_path)[:3, :]
        color_gen = np.array(Image.open(color_img_path))
        depth_gen = np.array(Image.open(depth_img_path))

        # Retrieve negative sample images (i.e., images not containing the object of interest) for the current object.
        neg_color, neg_depth, neg_txt_path = self.get_negative_imgs(obj_name)

        # Return a tuple containing all processed and relevant information for the current sample.
        return (color_isolated, depth_isolated, pose_ori, str(txt_path), color_gen, depth_gen, pose_gen,
                str(gen_txt_path), neg_color, neg_depth, str(neg_txt_path), obj_name)

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
        time1, time2, time3 = 0, 0, 0
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
        neg_color_gen = np.array(Image.open(neg_color_img_path))
        neg_depth_gen = np.array(Image.open(neg_depth_img_path))

        return neg_color_gen, neg_depth_gen, neg_txt_path

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

def angle_between_vectors(v1, v2):
    """Calculate the angle in degrees between vectors 'v1' and 'v2'."""
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    angle_radians = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    return np.degrees(angle_radians)

def create_folder(path='./new'):
    # Create folder
    if os.path.exists(path):
        shutil.rmtree(path)  # delete output folder
    os.makedirs(path)  # make new output folder
    return path
