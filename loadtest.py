from utils.dataloader import create_dataloader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


dataroot = '/media/iai-lab/wanqing/YCB_Video_Dataset'
ori = np.array(Image.open(dataroot + '/YCB_pairs/train_data/0001-000001-002_master_chef_can/color_original.png'))
gen = np.array(Image.open(dataroot + '/YCB_pairs/train_data/0001-000001-002_master_chef_can/color_generated.png'))
ori_pose = np.loadtxt(dataroot + '/YCB_pairs/train_data/0001-000001-002_master_chef_can/pose_original.txt')
gen_pose = np.loadtxt(dataroot + '/YCB_pairs/train_data/0001-000001-002_master_chef_can/pose_generated.txt')
ori_dis = ori_pose[2, 3]
gen_dis = gen_pose[2, 3]

ori_count = np.count_nonzero(np.any(ori != [0, 0, 0], axis=2))
gen_count = np.count_nonzero(np.any(gen != [0, 0, 0], axis=2))
print(ori_dis, gen_dis, ori_count, gen_count)
