import os
import random
import re
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from torch.nn.modules.distance import PairwiseDistance
import json
from PIL import Image
import matplotlib.pyplot as plt

from utils.dataloader import create_dataloader
from utils.process_data import create_folder
from losses.triplet_loss import TripletLoss
from models.simple_vit import SimpleViT

def angle_between_rotation_matrices(m1, m2):
    def angle_between_vectors(v1, v2):
        v1_u = v1 / np.linalg.norm(v1)
        v2_u = v2 / np.linalg.norm(v2)
        angle_radians = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
        return np.degrees(angle_radians)
    """Calculate the angle in degrees between vectors 'v1' and 'v2'."""
    angle_diff = [angle_between_vectors(m1[0, :3], m2[0, :3]),
                  angle_between_vectors(m1[1, :3], m2[1, :3]),
                  angle_between_vectors(m1[2, :3], m2[2, :3])]
    return angle_diff

def read_folders_contents(dataroot):
    """
    Reads the contents of text files in the given list of folders and returns a dictionary.
    The keys of the dictionary are the folder names, and the values are lists of tuples.
    Each tuple consists of a file path and its content.

    Parameters:
    folders (list): A list of folder paths to read text files from.

    Returns:
    dict: A dictionary where each key is a folder name, and each value is a sub-dictionary.
          Each sub-dictionary contains the path to a text file as key and its content as value.
    """
    dataroot = Path(dataroot)
    gen_paths = sorted([p for p in Path(dataroot / 'YCB_objects').glob('*') if p.is_dir()])
    all_folders_contents = {}
    for folder in gen_paths:
        folder_name = folder.name  # Extracts the last part of the path as the folder name
        folder_contents = {}
        if os.path.exists(folder):
            for filename in sorted(list(folder.rglob('*.txt'))):
                file_path = folder / filename
                if os.path.isfile(file_path):
                    folder_contents[file_path] = np.loadtxt(file_path)
            print(f'load data from {folder} finished, {len(folder_contents)}/4000 data loaded')
            all_folders_contents[folder_name] = folder_contents
        else:
            print(f"The folder {folder} does not exist.")

    return all_folders_contents


def forward_pass(imgs, model, batch_size):
    imgs = imgs.cuda()
    embeddings = model(imgs)

    # Split the embeddings into Anchor, Positive, and Negative embeddings
    anc_embeddings = embeddings[:batch_size]
    pos_embeddings = embeddings[batch_size: batch_size * 2]
    neg_embeddings = embeddings[batch_size * 2:]

    return anc_embeddings, pos_embeddings, neg_embeddings, model


def load_all_caches(cache_dir):
    all_caches = {}  # This dictionary will hold all cache contents

    # Iterate through all files in the cache directory
    for filename in os.listdir(cache_dir):
        # Construct the full file path
        file_path = os.path.join(cache_dir, filename)

        # Check if the file is a cache file (adjust the condition based on your cache file naming pattern)
        if filename.endswith("_cache.pt"):
            # Extract folder_name from the filename (adjust the slicing according to your naming pattern)
            folder_name = filename[:-len("_cache.pt")]

            # Load the cache file
            cache_content = torch.load(file_path)

            # Add the loaded content to the all_caches dict under the extracted folder_name
            all_caches[folder_name] = cache_content

    return all_caches


def test_dino():
    # load trained model
    model_path = './results/dino'
    cache_dir = "./cache"  # Define the cache directory

    # create dataloader for testing
    dataroot = '/home/iai-lab/Documents/YCB_Video_Dataset'
    batch_size = 1  # modify base on your device
    image_size = 256
    num_workers = 18
    inference_size = 1000  # 1000 pairs from train and test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg').to(device)

    gen_content = read_folders_contents(dataroot)

    # Loop through each string and sub-dict
    with torch.no_grad():
        for folder_name in tqdm(
                list(gen_content.keys())):  # Use list() to avoid RuntimeError for changing dict size during iteration
            # Construct the expected cache file name or path based on folder_name
            # This is an example; adjust the pattern to match how your cache files are named
            cache_file_path = os.path.join(cache_dir, f"{folder_name}_cache.pt")

            # Check if the cache file exists
            if os.path.exists(cache_file_path):
                print(f"Cache found for {folder_name}. Removing from processing queue.")
                # Remove the entry from gen_content
                del gen_content[folder_name]

    # Loop through each string and sub-dict
    if gen_content is not None:
        with torch.no_grad():
            for folder_name, sub_dict in tqdm(gen_content.items()):
                # For each sub-dict, get the path and ndarray
                processed_sub_dict = {}
                for path, pose in sub_dict.items():
                    # load the img
                    base_name = path.name.split('-')[0]
                    # Create the new file name
                    img_file_path = path.parent / (base_name + '-color.png')
                    img = np.array(Image.open(img_file_path))
                    img = img.transpose((2, 0, 1))  # HWC to CHW
                    img = np.ascontiguousarray(img).astype(np.float32)
                    img = torch.from_numpy(img)
                    img = img.to(device)
                    img = img.unsqueeze(0)
                    embedding = model(img)
                    img = img[:, :, 16:240, 16:240]
                    # dep_file_path = path.parent / (base_name + '-depth.png')
                    # dep = np.array(Image.open(dep_file_path)) / 10000
                    # dep = np.expand_dims(dep, axis=0)  # Add channel dimension
                    # dep = np.ascontiguousarray(dep).astype(np.float32)
                    # dep = torch.from_numpy(dep)
                    # dep = dep.to(device)
                    # dep = dep.unsqueeze(0)  # Add batch dimension
                    # # Concatenate RGB and depth images along the channel dimension to form an RGBD image
                    # rgbd = torch.cat((img, dep), dim=1)
                    embedding = model(img)

                    # Combine the ndarray and the 1D array into a list
                    combined_list = [pose, embedding.cpu().numpy()]  # Move data to CPU before converting to numpy
                    processed_sub_dict[path] = combined_list

                    # Suggest to the GPU to free up unused memory
                    torch.cuda.empty_cache()

                # Save the processed_sub_dict to a file after each folder is processed
                cache_file_path = os.path.join(cache_dir, f"{folder_name}_cache.pt")
                torch.save(processed_sub_dict, cache_file_path)
                print(f"Saved cache for folder {folder_name} at {cache_file_path}")

                # Optionally clear processed_sub_dict to free up memory
                del processed_sub_dict

        embedded_gen_contents = load_all_caches(cache_dir)

        train_loader, train_set = create_dataloader(dataroot,
                                                    type='train',
                                                    imgsz=image_size,
                                                    batch_size=batch_size,
                                                    workers=num_workers)
        trained_results = inference(train_loader, model, device, embedded_gen_contents, batch_size, inference_size, 'trained')
        del train_loader, train_set
        test_loader, test_set = create_dataloader(dataroot,
                                                  type='test',
                                                  imgsz=image_size,
                                                  batch_size=batch_size,
                                                  workers=num_workers)
        untrained_results = inference(test_loader, model, device, embedded_gen_contents, batch_size, inference_size, 'untrained')
        del test_loader, test_set
        # Writing JSON data
        trained_json = model_path + '/trained_results.json'
        untrained_json = model_path + '/untrained_results.json'
        trained_graph = model_path + '/trained_hist.png'
        untrained_graph = model_path + '/untrained_hist.png'

        draw_histogram(trained_results, trained_graph)
        draw_histogram(untrained_results, untrained_graph)

        with open(trained_json, 'w') as f:
            json.dump(trained_results, f, indent=4)
        print('Trained result saved to {}'.format(trained_json))
        with open(untrained_json, 'w') as f:
            json.dump(untrained_results, f, indent=4)
        print('Untrained result saved to {}'.format(untrained_json))


def inference(dataloader, model, device, gen_content, batch_size, inference_size, source):
    results = []
    l2_distance = PairwiseDistance(p=2)
    # we first process the pairs in the dataloader to get ground truth
    for batch, data in tqdm(enumerate(dataloader), total=min(inference_size / batch_size, len(dataloader)),
                                        desc=f'Processing data for {source} data'):
        anc_img = data[0][:, :, 16:240, 16:240]
        pos_img = data[1][:, :, 16:240, 16:240]
        neg_img = data[2][:, :, 16:240, 16:240]
        concatenated_data = torch.cat((anc_img, pos_img, neg_img), dim=0)
        anc_embeddings, pos_embeddings, neg_embeddings, model = forward_pass(
            imgs=concatenated_data,
            model=model,
            batch_size=batch_size
        )

        for anc, pos, ori_pose, gen_pose, obj_name, ori_path, gen_path in (
                zip(anc_embeddings, pos_embeddings, data[6], data[7], data[8][0], data[8][1], data[8][2])):
            smallest_dist, s_path, s_pose = 1000, [], []
            for path, c_list in gen_content[obj_name].items():
                temp_dist = l2_distance.forward(anc, torch.from_numpy(c_list[1]).to(device))
                if temp_dist < smallest_dist:
                    s_path = path
                    smallest_dist = temp_dist
                    s_pose = c_list[0]

            new_object = {
                "original": str(ori_path),
                "pre_selected_gen": str(gen_path),
                "angular_dist_sgen": angle_between_rotation_matrices(ori_pose, gen_pose),
                "distance_sgen": round(l2_distance.forward(anc, pos).item(), 3),
                "mod_selected_gen": str(s_path),
                "angular_dist_mgen": angle_between_rotation_matrices(ori_pose, s_pose),
                "distance_mgen": round(smallest_dist.item(), 3),
            }
            results.append(new_object)
        if batch == inference_size / batch_size:
            return results


def draw_histogram(json_result, save_name):
    differences = []
    # Iterate through each JSON object in the file
    for item in json_result:
        # Extract the third numbers from the 'angular_dist_sgen' and 'angular_dist_mgen'
        third_num_sgen = item['angular_dist_sgen'][2]
        third_num_mgen = item['angular_dist_mgen'][2]
        # Calculate the absolute difference
        diff = abs(third_num_sgen - third_num_mgen)
        # Append the difference to the list
        differences.append(diff)
    # Create bins for the range 0-1 to 179-180
    bins = np.arange(0, 181, 1)
    # Calculate the histogram
    hist, bin_edges = np.histogram(differences, bins=bins)
    # Draw the bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(bin_edges[:-1], hist, width=1, edgecolor='black')
    plt.xlabel('Difference in Degrees')
    plt.ylabel('Frequency')
    plt.title('Frequency of Differences Between "angular_dist_sgen" and "angular_dist_mgen"')
    plt.xticks(np.arange(0, 181, 10))
    # Save the plot to a file
    plt.savefig(save_name, dpi=300)  # Saves the plot as a PNG file with 300 DPI


if __name__ == '__main__':
    test_dino()
