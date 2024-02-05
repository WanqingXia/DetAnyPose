# YOLOv5 dataset utils and dataloaders

import glob
import hashlib
import logging
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import ThreadPool, Pool
from pathlib import Path
from threading import Thread
import scipy

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
import warnings


# Parameters
NUM_THREADS = min(8, os.cpu_count())  # number of multiprocessing threads


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
        nf_o, nm_o, nf_g, nm_g, n = current_cache.pop('results')
        if exists:
            d = f"Scanning original images and labels for {self.type}... {nf_o} found, {nm_o} missing"
            tqdm(None, desc=d, total=n, initial=n)  # display cache results
            if current_cache['msgs_o']:
                logging.info('\n'.join(current_cache['msgs_o']))  # display warnings
            d = f"Scanning generated images and labels for {self.type}... {nf_o} found, {nm_o} missing"
            tqdm(None, desc=d, total=n, initial=n)  # display cache results
            if current_cache['msgs_g']:
                logging.info('\n'.join(current_cache['msgs_g']))  # display warnings

        if self.type == 'train':
            assert n > 0, f'No labels in {train_cache_path}. Cannot train without labels.'
            [train_cache.pop(k) for k in ('hash', 'version', 'msgs_o', 'msgs_g')]  # remove items
            train_txt_cache = list(train_cache.keys())
            self.train_txt_files = [self.path / sub_path for sub_path in train_txt_cache]
            train_gen_cache = list(train_cache.values())
            self.train_gen_files = [self.path / sub_path for sub_path in train_gen_cache]
            self.train_indices = range(len(self.train_txt_files))

        elif self.type == 'test':
            assert n > 0, f'No labels in {test_cache_path}. Cannot train without labels.'
            [test_cache.pop(k) for k in ('hash', 'version', 'msgs_o', 'msgs_g')]  # remove items
            test_txt_cache = list(test_cache.keys())
            self.test_txt_files = [self.path / sub_path for sub_path in test_txt_cache]
            test_gen_cache = list(test_cache.values())
            self.test_gen_files = [self.path / sub_path for sub_path in test_gen_cache]
            self.test_indices = range(len(self.test_txt_files))

    def cache_labels(self, path_train=Path('./train_data.cache'), path_test=Path('./test_data.cache')):
        # Cache dataset labels, check images
        x = {}  # dict for train txt file
        y = {}  # dict for test txt file
        nf_o, nm_o, msgs_o = 0, 0, []  # number found, corrupt, messages in original data
        nf_g, nm_g, msgs_g = 0, 0, []  # number found, corrupt, messages in generated data
        desc = f"Scanning YCB_Video_Dataset/YCB_objects images and labels..."
        with Pool(NUM_THREADS) as pool:
            pbar = tqdm(pool.imap_unordered(self.verify_gen_paths, self.gen_txt_files),
                        desc=desc, total=len(self.gen_txt_files))
            for nf, nm, msg in pbar:
                nf_g += nf
                nm_g += nm
                if msg:
                    msgs_g.append(msg)
                pbar.desc = f"{desc}{nf_g} found, {nm_g} corrupted"
                if nf_g == 10:
                    break
        pbar.close()
        if msgs_g:
            logging.info('\n'.join(msgs_g))
        print('Scanning YCB_Video_Dataset/YCB_objects finished')

        desc = f"Scanning YCB_Video_Dataset/data images and labels..."
        num_train, num_test = 0, 0
        with Pool(NUM_THREADS) as pool:
            pbar = tqdm(pool.imap_unordered(self.verify_data_paths, self.txt_files),
                        desc=desc, total=len(self.txt_files))
            for txt_file, mat_file, nf, nm, msg in pbar:
                split_str = str(self.path) + '/'
                nf_o += nf
                nm_o += nm
                if txt_file:
                    mat = scipy.io.loadmat(mat_file)
                    with open(txt_file, 'r') as f:
                        labels = f.readlines()
                        for num, label in enumerate(labels):
                            obj_name = label.split(' ')[0]
                            gen_txt_path = self.search_imgs(obj_name, np.array(mat['poses'][:3, :3, num]))
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
                    msgs_o.append(msg)
                pbar.desc = f"{desc}{nf_o} found, {nm_o} corrupted"
                if nf_o == 10:
                    break
        pbar.close()
        if msgs_o:
            logging.info('\n'.join(msgs_o))
        print('Scanning YCB_Video_Dataset/data finished')

        x['hash'] = get_hash(str(self.txt_files) + str(self.gen_txt_files))
        x['results'] = nf_o, nm_o, nf_g, nm_g, num_train
        x['msgs_o'] = msgs_o  # warnings
        x['msgs_g'] = msgs_g  # warnings
        x['version'] = 1.0  # cache version
        try:
            np.save(path_train, x)  # save cache for next time
            path_train.with_suffix('.cache.npy').rename(path_train)  # remove .npy suffix
            print(f'New cache created: {path_train}')
        except Exception as e:
            print(f'WARNING: Cache directory {path_train.parent} is not writeable: {e}')  # path not writeable

        y['hash'] = get_hash(str(self.txt_files) + str(self.gen_txt_files))
        y['results'] = nf_o, nm_o, nf_g, nm_g, num_test
        y['msgs_o'] = msgs_o  # warnings
        y['msgs_g'] = msgs_g  # warnings
        y['version'] = 1.1  # cache version
        try:
            np.save(path_test, y)  # save cache for next time
            path_test.with_suffix('.cache.npy').rename(path_test)  # remove .npy suffix
            print(f'New cache created: {path_test}')
        except Exception as e:
            print(f'WARNING: Cache directory {path_test.parent} is not writeable: {e}')  # path not writeable

        return x, y

    def verify_gen_paths(self, file_path):
        return 1, 0, []  # bypassing filecheck, comment this line if file check is needed
        file_path = Path(file_path)
        color_img_path = file_path.parent / (file_path.name.split('-')[0] + '-color.png')
        depth_img_path = file_path.parent / (file_path.name.split('-')[0] + '-depth.png')
        nf, nm, msg = 0, 0, []  # number found, missing
        if os.path.isfile(file_path) and os.path.isfile(color_img_path) and os.path.isfile(depth_img_path):
            nf = 1
        else:
            nm = 1
            if not os.path.isfile(file_path):
                warnings.warn('txt file missing', UserWarning)
            if not os.path.isfile(color_img_path):
                warnings.warn('color image file missing', UserWarning)
            if not os.path.isfile(depth_img_path):
                warnings.warn('depth image file missing', UserWarning)
            msg = f'WARNING: Ignoring corrupted image and/or label {file_path}'
        return nf, nm, msg

    def verify_data_paths(self, file_path):
        base_name = file_path.name.split('-')[0]
        # Create the new file name
        color_img_path = file_path.parent / (base_name + '-color.png')
        depth_img_path = file_path.parent / (base_name + '-depth.png')
        label_img_path = file_path.parent / (base_name + '-label.png')
        metad_file_path = file_path.parent / (base_name + '-meta.mat')
        return file_path, metad_file_path, 1, 0, []  # bypassing filecheck, comment this line if file check is needed
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

    def __len__(self):
        return len(self.train_txt_files) if self.type == 'train' else len(self.test_gen_files)

    def __getitem__(self, index):
        index = self.train_indices[index] if self.type == 'train' else self.test_indices[index]

        # Load image
        c_ori, d_ori, p_ori, t_p, c_gen, d_gen, p_gen, gt_p, c_neg, d_neg, nt_p, obj_name = self.load_image(index)

        # Convert
        c_ori = c_ori.transpose((2, 0, 1))  # HWC to CHW
        c_ori = np.ascontiguousarray(c_ori)
        c_ori = torch.from_numpy(c_ori)

        c_gen = c_gen.transpose((2, 0, 1))  # HWC to CHW
        c_gen = np.ascontiguousarray(c_gen)
        c_gen = torch.from_numpy(c_gen)

        c_neg = c_neg.transpose((2, 0, 1))  # HWC to CHW
        c_neg = np.ascontiguousarray(c_neg)
        c_neg = torch.from_numpy(c_neg)

        d_ori = d_ori[np.newaxis, :, :]
        d_ori = np.ascontiguousarray(d_ori)
        d_ori = torch.from_numpy(d_ori)

        d_gen = d_gen[np.newaxis, :, :]
        d_gen = np.ascontiguousarray(d_gen)
        d_gen = torch.from_numpy(d_gen)

        d_neg = d_neg[np.newaxis, :, :]
        d_neg = np.ascontiguousarray(d_neg)
        d_neg = torch.from_numpy(d_neg)

        p_ori = torch.from_numpy(p_ori)
        p_gen = torch.from_numpy(p_gen)

        return c_ori, d_ori, p_ori, t_p, c_gen, d_gen, p_gen, gt_p, c_neg, d_neg, nt_p, obj_name

    def load_image(self, i):
        # loads 1 image from dataset index 'i', returns im, original hw, resized hw
        txt_path_obj = self.train_txt_files[i] if self.type == 'train' else self.test_txt_files[i]
        obj_name = txt_path_obj.name
        txt_path = txt_path_obj.parent
        base_name = txt_path.name.split('-')[0]
        # Create the new file name
        color_img_path = txt_path.parent / (base_name + '-color.png')
        depth_img_path = txt_path.parent / (base_name + '-depth.png')
        label_img_path = txt_path.parent / (base_name + '-label.png')
        metad_file_path = txt_path.parent / (base_name + '-meta.mat')
        color_image = np.array(Image.open(color_img_path))
        depth_image = np.array(Image.open(depth_img_path))
        label_image = np.array(Image.open(label_img_path))
        mat = scipy.io.loadmat(metad_file_path)
        obj_num = self.obj_names.index(obj_name) + 1  # starts from 1
        obj_index = list(mat['cls_indexes']).index(obj_num)
        pose_ori = np.array(mat['poses'][:3, :, obj_index])

        isolated_mask = (label_image == obj_num)
        color_isolated = self.isolate_image(color_image, isolated_mask, color_img_path)
        depth_isolated = self.isolate_image(depth_image, isolated_mask, depth_img_path)

        gen_txt_path = self.train_gen_files[i] if self.type == 'train' else self.test_gen_files[i]
        color_img_path = gen_txt_path.parent / (gen_txt_path.name.split('-')[0] + '-color.png')
        depth_img_path = gen_txt_path.parent / (gen_txt_path.name.split('-')[0] + '-depth.png')
        pose_gen = np.loadtxt(gen_txt_path)[:3, :]
        color_gen = np.array(Image.open(color_img_path))
        depth_gen = np.array(Image.open(depth_img_path))

        neg_color, neg_depth, neg_txt_path = self.get_negative_imgs(obj_name)

        return (color_isolated, depth_isolated, pose_ori, str(txt_path), color_gen, depth_gen, pose_gen,
                str(gen_txt_path), neg_color, neg_depth, str(neg_txt_path), obj_name)

    def search_imgs(self, obj_name, pose_ori):
        txt_files = [p for p in self.gen_txt_files if obj_name in str(p)]
        # Find the closest pose
        tmp = [1000, 1000, 1000]
        file = ''
        time1, time2, time3 = 0, 0, 0
        for txt in txt_files:
            tic = time.time()
            pose_gen = np.loadtxt(txt)
            toc = time.time()
            angle_diff = [angle_between_vectors(pose_ori[0, :], pose_gen[0, :3]),
                          angle_between_vectors(pose_ori[1, :], pose_gen[1, :3]),
                          angle_between_vectors(pose_ori[2, :], pose_gen[2, :3])]
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

        return file

    def get_negative_imgs(self, obj_name):
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
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # Extract the ROI
        roi = masked_image[rmin:rmax + 1, cmin:cmax + 1]

        # Calculate where to place the ROI in the center of the background
        roi_height, roi_width = roi.shape[:2]
        start_x = (self.img_size - roi_width) // 2
        start_y = (self.img_size - roi_height) // 2

        # Ensure that the ROI does not exceed background dimensions
        end_x = start_x + roi_width
        end_y = start_y + roi_height

        # if start_x < 0:
        #     start_x = 0
        # if start_y < 0:
        #     start_y = 0
        # if end_x > 256:
        #     end_x = 256
        # if end_y > 256:
        #     end_y = 256
        #

        try:
            # Adjust ROI size if it's larger than the background
            # roi = roi[:end_y - start_y, :end_x - start_x]
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
