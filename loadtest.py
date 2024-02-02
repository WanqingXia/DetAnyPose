from utils.dataloader import create_dataloader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


dataroot = '/media/iai-lab/wanqing/YCB_Video_Dataset'
train_loader, train_set = create_dataloader(dataroot, type='train', imgsz=256, batch_size=8, workers=8)
test_loader, test_set = create_dataloader(dataroot, type='test', imgsz=256, batch_size=8, workers=8)

for data1, data2 in tqdm(zip(train_loader, test_loader)):

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    count = 0
    for i, tensor in enumerate(data1):
        # Assuming the input tensor's shape is (8, C, H, W),
        # where C can be 3 for RGB or 1 for grayscale images,
        # and we're only interested in the first slice of the batch.
        if i == 0 or i == 1 or i == 4 or i == 5 or i == 8 or i == 9:
            slice = tensor[0]  # Take the first slice of the batch

            if slice.shape[0] == 3:
                # If the tensor is RGB, transpose it to HxWxC for plotting
                image = slice.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            else:
                # If the tensor is grayscale, squeeze it to remove the channel dimension
                image = slice.squeeze(0).cpu().numpy()
                min_val = image.min()
                max_val = image.max()
                # Avoid division by zero in case of a constant image
                normalized_image = (image - min_val) / (max_val - min_val) * 255
                image = normalized_image.astype(np.uint8)

            plt.imsave(f'image_{i + 1}.png', image)

        else:
            count += 1

    stop = 1

