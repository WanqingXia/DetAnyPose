from pathlib import Path
from mmdet_sam import mmdet_sam
from fbdinov2 import fbdinov2
from megapose import nvmegapose

device = 'cuda:0'
Megapose = nvmegapose.Megapose(device)
MMDet_SAM = mmdet_sam.MMDet_SAM(device)
DINOv2 = fbdinov2.DINOv2("./viewpoints_42", device)


Megapose.run_inference(Path('./data/drill'))
