import os
import torch
import numpy as np
import hashlib
import json
from pathlib import Path
import torchvision.transforms as T
from PIL import Image

class DINOv2:
    def __init__(self, viewpoints_path):
        self.viewpoints_path = viewpoints_path
        self.viewpoints_poses = {}
        self.viewpoints_images = {}
        self.viewpoints_embeddings = {}
        os.makedirs("./cache", exist_ok=True)
        self.cache = "./cache/embeddings.pt"
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
        self.out_dir = './outputs'  # Default output directory
        self.device = 'cuda:0'  # Default device used for det inference
        self.model.to(self.device)
        os.makedirs(self.out_dir, exist_ok=True)

        # Read the contents of the viewpoints file
        self.read_folders_contents()
        # Ensure the dictionary is sorted by keys to maintain order
        serialized_dict = json.dumps(self.viewpoints_images, sort_keys=True)
        # Use hashlib to generate a hash from the serialized string
        hash_value = hashlib.sha256(serialized_dict.encode()).hexdigest()
        self.load_cache(hash_value)

    def forward(self, img):
        # check if the image is already the size for DINOv2
        dinov2_size = (224, 224)
        if img.size != dinov2_size:
            # Resize the image
            resize_transform = T.Resize(dinov2_size)
            img = resize_transform(img)

        img = img.transpose((2, 0, 1))  # HWC to CHW
        img = np.ascontiguousarray(img).astype(np.float32)  # Ensure the image is contiguous and convert it to float32
        img = torch.from_numpy(img)  # Convert the numpy array to a PyTorch tensor.
        rgb_normalise = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        img = rgb_normalise(img / 255.)  # Normalise and scale
        img = img.to(self.device)
        img = img.unsqueeze(0)  # Add a batch dimension
        return self.model(img)

    def read_folders_contents(self):
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
        gen_paths = sorted([p for p in Path(self.viewpoints_path).glob('*') if p.is_dir()])
        for folder in gen_paths:
            folder_name = str(folder.name) # Extracts the last part of the path as the folder name
            sub_folder_contents = {}
            if os.path.exists(folder):
                for filename in sorted(list(folder.rglob('*.txt'))):
                    txt_file_path = str(folder / filename)
                    img_file_path = str(folder / str(filename).replace('matrix.txt', 'color.png'))
                    if os.path.isfile(txt_file_path):
                        sub_folder_contents[txt_file_path] = np.loadtxt(txt_file_path)
                        self.viewpoints_images[folder_name].append(img_file_path)
                print(f'load data from {folder} finished, {len(sub_folder_contents)} data loaded')
                self.viewpoints_poses[folder_name] = sub_folder_contents
            else:
                print(f"The folder {folder} does not exist.")

    def load_cache(self, hash_value):
        if os.path.isfile(self.cache):
            # cache exist, check hash value
            loaded_data = torch.load(self.cache)
            if hash_value == loaded_data['hash']:
                self.viewpoints_embeddings = loaded_data['tensors'].to(self.device)
                return
            else:
                print(f"The cache file {self.cache} is not the same as the hash value {hash_value}.")
                os.remove(self.cache)

        for folder, image_list in self.viewpoints_images.items():
            for image in image_list:
                img = Image.open(image)
                img = self.forward(img)
                self.viewpoints_embeddings[folder] = img

        data_to_save = {
            'tensors': self.viewpoints_embeddings,
            'hash': hash_value
        }
        torch.save(data_to_save, self.cache)

    @staticmethod
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
