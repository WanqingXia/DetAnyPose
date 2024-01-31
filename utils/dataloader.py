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
                                             collate_fn=LoadImagesAndLabels.collate_fn
                                             )
    return dataloader, dataset


def angle_between_vectors(v1, v2):
    """Calculate the angle in degrees between vectors 'v1' and 'v2'."""
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    angle_radians = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    return np.degrees(angle_radians)


class LoadImagesAndLabels(Dataset):  # for training/testing
    def __init__(self, path, type, img_size=256, batch_size=16):
        self.img_size = img_size
        self.path = Path(path)
        assert type in ('train', 'val', 'test'), f'{type} is not train, val or test'
        self.type = type
        self.test_categories = ['006_mustard_bottle', '019_pitcher_base', '021_bleach_cleanser']
        self.train_categories = ['002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can',
                                 '008_pudding_box', '011_banana', '024_bowl', '025_mug', '036_wood_block',
                                 '037_scissors', '061_foam_brick']
        self.folders = ['0001', '0004', '0007', '0013', '0020', '0021', '0031', '0041',
                        '0051', '0055', '0071', '0074', '0076', '0078', '0082', '0084', '0091']
        self.data_paths = [(self.path / 'data') / subpath for subpath in self.folders]
        self.obj_names = sorted(glob.glob(str(self.path / 'models')))

        try:
            f = []  # image files
            for p in self.data_paths:
                p = Path(p)  # os-agnostic
                f += glob.glob(str(p / '**' / '*.*'), recursive=True)
            self.txt_files = sorted([x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() == 'txt'])
            assert self.txt_files, f'No text file found'
        except Exception as e:
            raise Exception(f'Error loading data from {path}: {e}\n')

        # Check cache
        train_cache_path = Path(__file__) / 'train_data.cache'
        test_cache_path = Path(__file__) / 'test_data.cache'
        try:
            train_cache = np.load(train_cache_path, allow_pickle=True).item()
            test_cache = np.load(test_cache_path, allow_pickle=True).item()
            exists = True  # load dict
            assert train_cache['version'] == 1.0 and train_cache['hash'] == get_hash(self.txt_files)
            assert test_cache['version'] == 1.1 and test_cache['hash'] == get_hash(self.txt_files)
        except:
            train_cache, test_cache = self.cache_labels(train_cache_path, test_cache_path)
            exists = False  # cache

        # Display cache
        nf_t, nm_t, ne_t, nc_t, n_t = train_cache.pop('results')  # found, missing, empty, corrupted, total
        nf_e, nm_e, ne_e, nc_e, n_e = test_cache.pop('results')  # found, missing, empty, corrupted, total
        if exists:
            d = f"Scanning '{train_cache_path}' images and labels... {nf_t} found, {nm_t} missing, {ne_t} empty, {nc_t} corrupted"
            tqdm(None, desc=d, total=n_t, initial=n_t)  # display cache results
            if train_cache['msgs']:
                logging.info('\n'.join(train_cache['msgs']))  # display warnings
            d = f"Scanning '{test_cache_path}' images and labels... {nf_e} found, {nm_e} missing, {ne_e} empty, {nc_e} corrupted"
            tqdm(None, desc=d, total=n_e, initial=n_e)  # display cache results
            if test_cache['msgs']:
                logging.info('\n'.join(test_cache['msgs']))  # display warnings
        assert nf_t > 0, f'No labels in {train_cache_path}. Cannot train without labels.'
        assert nf_e > 0, f'No labels in {test_cache_path}. Cannot test without labels.'

        # Read cache
        [train_cache.pop(k) for k in ('hash', 'version', 'msgs')]  # remove items
        [test_cache.pop(k) for k in ('hash', 'version', 'msgs')]  # remove items
        self.train_files = zip(*train_cache.values())
        self.test_files = zip(*test_cache.values())

        self.train_indices = range(len(self.train_files))
        self.test_indices = range(len(self.test_files))

    def cache_labels(self, path_train=Path('./train_data.cache'), path_test=Path('./test_data.cache')):
        # Cache dataset labels, check images
        x = {}  # dict for train txt file
        y = {}  # dict for test txt file
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"Scanning '{path.parent / path.stem}' images and labels..."
        with Pool(NUM_THREADS) as pool:
            pbar = tqdm(pool.imap_unordered(self.verify_paths, self.txt_files),
                        desc=desc, total=len(self.txt_files))
            for im_file, l, shape, segments, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x[im_file] = [l, shape, segments]
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc}{nf} found, {nm} missing, {ne} empty, {nc} corrupted"

        pbar.close()
        if msgs:
            logging.info('\n'.join(msgs))
        if nf == 0:
            logging.info(f'WARNING: No labels found in {path}. See {HELP_URL}')
        x['hash'] = get_hash(self.txt_files)
        x['results'] = nf, nm, ne, nc, len(self.txt_files)
        x['msgs'] = msgs  # warnings
        x['version'] = 1.0  # cache version
        try:
            np.save(path, x)  # save cache for next time
            path.with_suffix('.cache.npy').rename(path)  # remove .npy suffix
            logging.info(f'New cache created: {path}')
        except Exception as e:
            logging.info(f'WARNING: Cache directory {path.parent} is not writeable: {e}')  # path not writeable
        return x

    def separate_train_test(self):
        train_files, test_files = [], []
        for file in self.txt_files:
            with open(file, 'r') as f:
                labels = f.readlines()
                for label in labels:
                    obj_name = label.split(' ')[0]
                    if obj_name in self.test_categories:
                        test_files.append(file / obj_name)
                    elif obj_name in self.train_categories:
                        train_files.append(file / obj_name)
                    else:
                        raise Exception(f'Unrecognised object name: {obj_name} \n')
        return train_files, test_files

    def verify_paths(self):
        # Verify one image-label pair
        im_file, lb_file, prefix = args
        nm, nf, ne, nc = 0, 0, 0, 0  # number missing, found, empty, corrupt
        try:
            # verify images
            im = Image.open(im_file)
            im.verify()  # PIL verify
            shape = exif_size(im)  # image size
            assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
            assert im.format.lower() in IMG_FORMATS, f'invalid image format {im.format}'
            if im.format.lower() in ('jpg', 'jpeg'):
                with open(im_file, 'rb') as f:
                    f.seek(-2, 2)
                    assert f.read() == b'\xff\xd9', 'corrupted JPEG'

            # verify labels
            segments = []  # instance segments
            if os.path.isfile(lb_file):
                nf = 1  # label found
                with open(lb_file, 'r') as f:
                    l = [x.split() for x in f.read().strip().splitlines() if len(x)]
                    if any([len(x) > 8 for x in l]):  # is segment
                        classes = np.array([x[0] for x in l], dtype=np.float32)
                        segments = np.array([x[1:5] for x in l], dtype=np.float32)  # (cls, xy1...)
                        # segments = [np.array(x[1:5], dtype=np.float32).reshape(-1, 2) for x in l]  # (cls, xy1...)
                        attributes = np.array([x[5:] for x in l], dtype=np.float32)
                        l = np.concatenate((classes.reshape(-1, 1), segments, attributes), 1)  # (cls, xywh)
                        # l = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments), attributes), 1)  # (cls, xywh)
                    l = np.array(l, dtype=np.float32)
                if len(l):
                    assert l.shape[1] == 21, 'labels require 21 columns each'
                    assert (l >= 0).all(), 'negative labels'
                    assert (l[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels'
                    assert np.unique(l, axis=0).shape[0] == l.shape[0], 'duplicate labels'
                else:
                    ne = 1  # label empty
                    l = np.zeros((0, 5), dtype=np.float32)
            else:
                nm = 1  # label missing
                l = np.zeros((0, 5), dtype=np.float32)
            return im_file, l, shape, segments, nm, nf, ne, nc, ''
        except Exception as e:
            nc = 1
            msg = f'{prefix}WARNING: Ignoring corrupted image and/or label {im_file}: {e}'
            return [None, None, None, None, nm, nf, ne, nc, msg]

    def __len__(self):
        return len(self.train_files) if self.type == 'train' else len(self.test_files)

    def __getitem__(self, index):
        index = self.indices[index]  # linear, shuffled, or image_weights

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

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes

    def load_image(self, i):
        # loads 1 image from dataset index 'i', returns im, original hw, resized hw
        text = self.train_files[i] if self.type == 'train' else self.test_files[i]
        obj_name = Path(text).name
        txt_path = Path(text).parent
        base_name = txt_path.stem.split('-')[0]
        # Create the new file name
        color_img_path = base_name + '-color.png'
        depth_img_path = base_name + '-depth.png'
        label_img_path = base_name + '-label.png'
        mat_file_path = base_name + '-meta.mat'
        color_image = np.array(Image.open(color_img_path))
        depth_image = np.array(Image.open(depth_img_path))
        label_image = np.array(Image.open(label_img_path))
        mat = scipy.io.loadmat(mat_file_path)
        obj_index = mat['cls_indexs'].index(self.obj_names.index(obj_name))
        pose_ori = np.array(mat['poses'][:3, :, obj_index])

        isolated_mask = (label_image == self.obj_names.index(obj_name))
        color_isolated = self.isolate_image(color_image, isolated_mask, text)
        depth_isolated = self.isolate_image(depth_image, isolated_mask, text)

        pose_gen, color_gen, depth_gen, gen_txt_path = self.search_imgs(obj_name, pose_ori)

        neg_color, neg_depth, neg_txt_path = self.get_negative_imgs(obj_name)

        return (color_isolated, depth_isolated, pose_ori, txt_path, color_gen, depth_gen, pose_gen,
                gen_txt_path, neg_color, neg_depth, neg_txt_path, obj_name)

    def search_imgs(self, obj_name, pose_ori):
        files = sorted(os.listdir(self.path / 'YCB_objects' / obj_name))
        # Filter out files that end with '.txt'
        txt_files = [file for file in files if file.endswith('.txt')]

        # Find the closest pose
        tmp = [1000, 1000, 1000]
        angle_diff = [0, 0, 0]
        file = ''
        for txt in txt_files:
            pose_gen = np.loadtxt(Path(self.path / 'YCB_objects' / obj_name / txt))[:3, :3]
            angle_diff = [angle_between_vectors(pose_ori[0, :], pose_gen[0, :]),
                          angle_between_vectors(pose_ori[1, :], pose_gen[1, :]),
                          angle_between_vectors(pose_ori[2, :], pose_gen[2, :])]

            if np.abs(angle_diff[2]) < np.abs(tmp[2]):
                tmp = angle_diff
                file = txt
        gen_txt_path = Path(self.path / 'YCB_objects' / obj_name / file)
        color_img_path = gen_txt_path.stem.split('-')[0] + '-color.png'
        depth_img_path = gen_txt_path.stem.split('-')[0] + '-depth.png'
        pose_gen = np.loadtxt(gen_txt_path)[:3, :]
        color_gen = np.array(Image.open(color_img_path))
        depth_gen = np.array(Image.open(depth_img_path))
        return pose_gen, color_gen, depth_gen, gen_txt_path

    def get_negative_imgs(self, obj_name):
        files = sorted(os.listdir(self.path / 'YCB_objects'))
        files.remove(obj_name)
        neg_obj = random.choice(files)
        neg_files = sorted(os.listdir(self.path / 'YCB_objects' / neg_obj))
        # Filter out files that end with '.txt'
        txt_files = [file for file in neg_files if file.endswith('.txt')]
        neg_file = random.choice(txt_files)
        neg_txt_path = Path(self.path / 'YCB_objects' / neg_obj / neg_file)
        neg_color_img_path = neg_txt_path.stem.split('-')[0] + '-color.png'
        neg_depth_img_path = neg_txt_path.stem.split('-')[0] + '-depth.png'
        neg_color_gen = np.array(Image.open(neg_color_img_path))
        neg_depth_gen = np.array(Image.open(neg_depth_img_path))

        return neg_color_gen, neg_depth_gen, neg_txt_path

    def isolate_image(self, input_img, mask, text):
        if input_img.ndim == 2:  # depth image
            # Create a new 256x256 numpy array filled with 0
            background = np.full((256, 256), 0, dtype=input_img.dtype)
            # Use the mask to find the ROI in the color image
            # This creates a masked version of the image where unmasked areas are set to 0
            masked_image = input_img * mask

        else:  # color image
            # Create a new 256x256x3 numpy array filled with 70
            background = np.full((256, 256, 3), 70, dtype=input_img.dtype)
            # Use the mask to find the ROI in the color image
            # This creates a masked version of the image where unmasked areas are set to 0
            masked_image = np.zeros_like(input_img)
            for i in range(3):  # color_image has 3 channels
                masked_image[:, :, i] = input_img[:, :, i] * mask

        # Find the bounding box of the masked area to extract only the relevant part
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # Extract the ROI
        roi = masked_image[rmin:rmax + 1, cmin:cmax + 1]

        # Calculate where to place the ROI in the center of the background
        roi_height, roi_width = roi.shape[:2]
        start_x = (256 - roi_width) // 2
        start_y = (256 - roi_height) // 2

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
            roi = roi[:end_y - start_y, :end_x - start_x]
            # Copy the ROI into the center of the background
            background[start_y:end_y, start_x:end_x, :] = roi
        except Exception as e:
            raise Exception(f'Error isolating the image {text}: {e}\n')
        return background


def create_folder(path='./new'):
    # Create folder
    if os.path.exists(path):
        shutil.rmtree(path)  # delete output folder
    os.makedirs(path)  # make new output folder
