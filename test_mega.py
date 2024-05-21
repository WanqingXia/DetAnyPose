import cv2
import numpy as np
from PIL import Image
from megapose import nvmegapose
from utils.convert import Convert_YCB


device = 'cuda:0'
Convert_YCB = Convert_YCB()
Megapose = nvmegapose.Megapose(device, Convert_YCB)

rgb = cv2.imread('./data/drill/image_rgb.png')
rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
depth = np.array(Image.open('./data/drill/image_depth.png'), dtype=np.float32) / 10000

rgb = np.array(rgb, dtype=np.uint8)
bbox = np.array([295, 215, 344, 264], dtype=int)
Megapose.inference(rgb, depth, Convert_YCB.convert_name('drill'), bbox)
