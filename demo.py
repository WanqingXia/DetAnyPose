import cv2
from mmdet_sam import mmdet_sam

image_path = './test.png'
mmdet_sam_model = mmdet_sam.mmdet_sam_model()
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

pred = mmdet_sam_model.run_detector(image, image_path, 'drill')
mmdet_sam_model.draw_outcome(image, pred, show_result=True, save_copy=True)

