import torch
from torchsummary import summary

dinov2_vitg14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg').cuda()
summary(dinov2_vitg14_reg, input_size=(3, 224, 224))  # Adjust the input size (3, 224, 224) as per your model

