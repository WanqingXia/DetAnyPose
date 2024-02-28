from utils.dataloader import create_dataloader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


dataroot = '/media/iai-lab/wanqing/YCB_Video_Dataset'

what = np.load('./cache/test.cache', allow_pickle=True).tolist()
stop = 1