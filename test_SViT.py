import os
import random

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

def test_SViT():
    # load trained model
    model_path = './results/03-19_14:35/best.pt'

    embedding_dimension = 512
    image_size = 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleViT(
        image_size=image_size,
        patch_size=32,
        num_classes=embedding_dimension,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048
    )
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # create dataloader for testing
    dataroot = '/home/iai-lab/Documents/YCB_Video_Dataset'
    batch_size = 50  # modify base on your device
    num_workers = 18  # modify base on your device
    inference_size = 1000  # 1000 pairs from train and test
    train_loader, train_set = create_dataloader(dataroot,
                                                type='train',
                                                imgsz=image_size,
                                                batch_size=batch_size,
                                                workers=num_workers)
    test_loader, test_set = create_dataloader(dataroot,
                                              type='test',
                                              imgsz=image_size,
                                              batch_size=batch_size,
                                              workers=num_workers)

    gen_content = read_folders_contents(dataroot)
    # Loop through each string and sub-dict
    with torch.no_grad():
        for folder_name, sub_dict in tqdm(gen_content.items()):
            # For each sub-dict, get the path and ndarray
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

                # Combine the ndarray and the 1D array into a list
                combined_list = [pose, embedding]
                # Save it back to the original structure
                # Each sub-dict now has a path and a list of ndarray and 1D array
                sub_dict[path] = combined_list

        trained_results = inference(train_loader, model, gen_content, batch_size, inference_size, 'trained')
        untrained_results = inference(test_loader, model, gen_content, batch_size, inference_size, 'untrained')

        # Writing JSON data
        with open('./results/trained_results.json', 'w') as f:
            json.dump(trained_results, f, indent=4)
        print('Trained result saved to {}/trained_results.json\n'.format(model_path.split('/best')[0]))
        with open('./results/untrained_results.json', 'w') as f:
            json.dump(untrained_results, f, indent=4)
        print('Untrained result saved to {}/trained_results.json\n'.format(model_path.split('/best')[0]))


def inference(dataloader, model, gen_content, batch_size, inference_size, source):
    results = []
    l2_distance = PairwiseDistance(p=2)
    # we first process the pairs in the dataloader to get ground truth
    for batch, data in tqdm(enumerate(dataloader), total=min(inference_size / batch_size, len(dataloader)),
                                        desc=f'Processing data for {source} data'):

        concatenated_data = torch.cat((data[0], data[1], data[2]), dim=0)
        anc_embeddings, pos_embeddings, neg_embeddings, model = forward_pass(
            imgs=concatenated_data,
            model=model,
            batch_size=batch_size
        )

        for anc, pos, ori_pose, gen_pose, obj_name, ori_path, gen_path in (
                zip(anc_embeddings, pos_embeddings, data[6], data[7], data[8][0], data[8][1], data[8][2])):
            smallest_dist, s_path, s_pose = 1000, [], []
            for path, c_list in gen_content[obj_name].items():
                temp_dist = l2_distance.forward(anc, c_list[1])
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


if __name__ == '__main__':
    test_SViT()
