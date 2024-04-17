import cv2
from mmdet_sam import mmdet_sam
from fbdinov2 import fbdinov2
from utils.choose import choose_from_viewpoints

image_path = './test.png'
MMDet_SAM = mmdet_sam.MMDet_SAM()
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

pred = MMDet_SAM.run_detector(image.copy(), image_path, 'white_bleach_bottle')
# MMDet_SAM.draw_outcome(image.copy(), pred, show_result=False, save_copy=False)

DINOv2 = fbdinov2.DINOv2("./viewpoints_42")

if len(pred['labels']) == 0:
    # Nothing detected
    pass
else:
    vp_img_path, vp_pose, best_pred, embed_img, iso_img = choose_from_viewpoints(image, pred, DINOv2, save=True)


