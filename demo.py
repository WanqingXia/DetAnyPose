import cv2
from mmdet_sam import mmdet_sam
from fbdinov2 import fbdinov2

# image_path = './test.png'
# MMDet_SAM = mmdet_sam.MMDet_SAM()
# image = cv2.imread(image_path)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
# pred = MMDet_SAM.run_detector(image, image_path, 'drill')
# MMDet_SAM.draw_outcome(image, pred, show_result=False, save_copy=False)

DINOv2 = fbdinov2.DINOv2("./viewpoints_42")

stop = 1
