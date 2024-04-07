import os
import random
import re
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torch.nn.functional as F


from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from torch.nn.modules.distance import PairwiseDistance
import json
from PIL import Image
import matplotlib.pyplot as plt

class Similarity(nn.Module):
    def __init__(self, metric="cosine", chunk_size=64):
        super(Similarity, self).__init__()
        self.metric = metric
        self.chunk_size = chunk_size

    def forward(self, query, reference):
        query = F.normalize(query, dim=-1)
        reference = F.normalize(reference, dim=-1)
        similarity = F.cosine_similarity(query, reference, dim=-1)
        return similarity.clamp(min=0.0, max=1.0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg').to(device)

# Assuming the two lines of code you've provided are executed before this

# 1. Load the main image
image_path = '0006-000100-color.png'
image = Image.open(image_path).convert('RGB')

# 2. Load the binary masks and convert them to boolean masks
mask_paths = ['mask_channel_0.png']
masks = [Image.open(mask_path) for mask_path in mask_paths]
bool_masks = [(np.array(mask) > 0) for mask in masks]

rgb_normalize = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])
results = []
for mask in bool_masks:
    # Apply the mask to isolate the masked area
    masked_image = Image.fromarray(np.where(mask[..., None], np.array(image), 0).astype('uint8'))

    # Find the bounding box of the mask
    bbox = np.where(mask)
    if bbox[0].size and bbox[1].size:
        min_row, max_row, min_col, max_col = np.min(bbox[0]), np.max(bbox[0]), np.min(bbox[1]), np.max(bbox[1])
        # Crop the masked image to the bounding box of the mask
        masked_image_cropped = masked_image.crop((min_col, min_row, max_col + 1, max_row + 1))
    else:
        # Fallback if the mask does not cover any part of the image
        masked_image_cropped = masked_image

    # Resize the cropped image
    masked_image_resized = masked_image_cropped.resize((224, 224))

    # Normalize the image
    normalized_image = rgb_normalize(masked_image_resized)

    # Forward pass the image with the model
    normalized_image_batch = normalized_image.unsqueeze(0).to(device)
    with torch.no_grad():  # Ensuring we're not tracking gradients for inference
        output = model(normalized_image_batch)
    results.append(output.cpu())  # Assuming you're processing outputs later

# 1. Load all images ending with "color.png" from the specified directory
directory_path = '/home/iai-lab/Documents/YCB_Video_Dataset/viewpoints_42/009_gelatin_box'
image_files = sorted([os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith('color.png')])

# 2 & 3. Crop, resize, and normalize all images
processed_images = []
for image_path in image_files:
    image = Image.open(image_path).convert('RGB')
    # Resize the image directly, assuming we want to maintain the aspect ratio
    image_resized = np.array(image)[16:240, 16:240, :]
    cropped_image = Image.fromarray(image_resized)
    # Normalize the image
    normalized_image = rgb_normalize(cropped_image)
    processed_images.append(normalized_image)

# 4. Concatenate and pass them as a single batch to the model
database = []
for image_tensor in processed_images:
    # Add an extra batch dimension and transfer to GPU
    image_tensor_batch = image_tensor.unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():  # Ensure no extra memory is used for gradients
        output = model(image_tensor_batch)

    # Store or process the output here as needed
    database.append(output.cpu())
# Instantiate the similarity calculation module
similarity_calc = Similarity()

# Assuming 'results' and 'database' are lists of tensors
cosine_similarities = []

# Convert lists of tensors to batches for processing
results_batch = torch.stack(results).squeeze(1)  # Ensure shape is [m, 1536] from [m, 1, 1536]
database_batch = torch.stack(database).squeeze(1)  # Ensure shape is [n, 1536] from [n, 1, 1536]

# Calculate cosine similarity for each tensor in 'results' against all tensors in 'database'
for query_tensor in results_batch:
    # Expand dimensions of query_tensor to [1, 1536] to match the forward method's expectation
    query_tensor = query_tensor.unsqueeze(0)

    # Compute similarity scores with the entire database batch
    # The 'similarity_calc' expects the first tensor to have a shape of [1, 1536]
    # and the second tensor to have a shape of [n, 1536]
    similarity_scores = similarity_calc(query_tensor, database_batch)

    # Collect the scores
    cosine_similarities.append(similarity_scores)

stop = 1

