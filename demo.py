import cv2
import numpy as np
from PIL import Image

from mmdet_sam import mmdet_sam
from fbdinov2 import fbdinov2
from megapose import nvmegapose
from utils.choose import choose_from_viewpoints, validate_preds
from scipy.io import loadmat


device = 'cuda:0'
MMDet_SAM = mmdet_sam.MMDet_SAM(device)
DINOv2 = fbdinov2.DINOv2("./viewpoints_42", device)
Megapose = nvmegapose.Megapose(device)

mat = loadmat('./data/drill/image_meta.mat')
rgb = cv2.imread('./data/drill/image_rgb.png')
rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
depth = np.array(Image.open('./data/drill/image_depth.png'), dtype=np.float32) / 10000

pred = MMDet_SAM.run_detector(rgb.copy(), 'drill')
# MMDet_SAM.draw_outcome(image.copy(), pred, show_result=False, save_copy=False)

best_pred = 0
if len(pred['labels']) == 0:
    # Nothing detected
    pass
else:
    best_pred = validate_preds(rgb, pred, DINOv2)

# mask = pred['masks'][best_pred].cpu().numpy().astype(np.uint8)
# mask = np.transpose(mask, (1, 2, 0))
#
rgb = np.array(rgb, dtype=np.uint8)
# rgb_masked = rgb * mask
#
# mask = mask.squeeze(axis=-1)
# depth_masked = depth * mask

# Convert the NumPy array to a PIL Image
# image = Image.fromarray(rgb_masked)
# # Save the image
# image.save('output_image.png')
#
bbox = np.round(pred['boxes'][best_pred].cpu().numpy()).astype(int)
Megapose.inference(rgb, depth, pred['labels'][best_pred], bbox)

