import os
import torch
import numpy as np
import hashlib
import json
from pathlib import Path
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm


class DINOv2:
    def __init__(self, device, viewpoints_path):
        self.device = device  # Default device used for det inference
        self.viewpoints_path = viewpoints_path
        self.viewpoints_poses = {}
        self.viewpoints_images = {}
        self.viewpoints_embeddings = {}
        self.cache = "./data/embeddings48.pt"
        self.model = torch.hub.load('dinov2', 'dinov2_vitl14', source='local', pretrained=True)
        self.model.to(self.device)
        self.model.load_state_dict(torch.load('./models/dinov2_vitl14_pretrain.pth'))
        self.out_dir = './outputs'  # Default output directory
        os.makedirs(self.out_dir, exist_ok=True)
        # Resize the image
        self.dinov2_size = (224, 224)
        self.resize_transform = T.Resize(self.dinov2_size)

        # Read the contents of the viewpoints file
        self.read_folders_contents()
        # Ensure the dictionary is sorted by keys to maintain order
        serialized_dict = json.dumps(self.viewpoints_images, sort_keys=True)
        # Use hashlib to generate a hash from the serialized string
        hash_value = hashlib.sha256(serialized_dict.encode()).hexdigest()
        if os.path.isfile(self.cache):
            # cache exist, check hash value
            loaded_data = torch.load(self.cache)
            if hash_value == loaded_data['hash']:
                self.viewpoints_embeddings = loaded_data['tensors']
            else:
                print(f"The cache file {self.cache} is not the same as the hash value {hash_value}.")
                os.remove(self.cache)
                self.create_cache(hash_value)
        else:
            self.create_cache(hash_value)

    def forward(self, img):
        if not isinstance(img, np.ndarray):
            raise ValueError("The image must be converted to np array before processing.")
        with torch.no_grad():
            height, width = img.shape[:2]

            if height != width:
                # Determine the size of the new square image.
                new_size = max(height, width)

                # Create a new square image with zeros and three channels.
                padded_image = np.zeros((new_size, new_size, 3), dtype=img.dtype)

                # Determine the starting positions to center the original image.
                start_y = (new_size - height) // 2
                start_x = (new_size - width) // 2

                # Copy the original image into the center of the padded image.
                padded_image[start_y:start_y + height, start_x:start_x + width, :] = img

            img = np.transpose(img, (2, 0, 1))  # HWC to CHW
            img = np.ascontiguousarray(img).astype(np.float32)  # Ensure the image is contiguous, convert it to float32
            img = torch.from_numpy(img)  # Convert the numpy array to a PyTorch tensor.
            rgb_normalise = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            img = rgb_normalise(img / 255.)  # Normalise and scale
            # check if the image is already the size for DINOv2
            if img.size != self.dinov2_size:
                img = self.resize_transform(img)
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
            folder_name = str(folder.name)
            if folder_name not in self.viewpoints_images:
                self.viewpoints_images[folder_name] = []
            # Extracts the last part of the path as the folder name
            if os.path.exists(folder):
                for png_file in sorted(list(folder.rglob('*.png'))):
                    png_stem = png_file.stem
                    png_stem = png_stem.split('_')
                    if png_stem[0] == 'rgb' and int(png_stem[1]) % 12 == 0:
                        self.viewpoints_images[folder_name].append(str(png_file))  # convert to string for json
                print(f'load data from {folder} finished, {len(self.viewpoints_images[folder_name])} data loaded')
            else:
                print(f"The folder {folder} does not exist.")

    def create_cache(self, hash_value):
        for folder, image_list in tqdm(self.viewpoints_images.items()):
            tensors_list = []
            for image in image_list:
                img = Image.open(Path(image))
                img = np.array(img)
                img = self.forward(img)
                tensors_list.append(img.cpu())
                img.detach()
                del img
                torch.cuda.empty_cache()  # Clear unused memory
            tensor_cat = torch.cat(tensors_list, dim=0)
            self.viewpoints_embeddings[folder] = tensor_cat

        data_to_save = {
            'tensors': self.viewpoints_embeddings,
            'hash': hash_value
        }
        torch.save(data_to_save, self.cache)

    def search_img(self, ref_poses, obj_name):
        """
        Find the closest pose in the viewpoint list to the original pose.

        Args:
            ref_poses (numpy.ndarray): The original pose of the object of interest.
            obj_name (str): The name of the object

        Returns:
            str: The file name of the generated pose that is closest to the original pose.

        """
        # Find the closest pose
        tmp = [1000, 1000, 1000]
        path = ''
        for image_path, pose in self.viewpoints_poses[obj_name].items():
            angle_diff = self.angle_between_rotation_matrices(ref_poses, pose)
            if np.abs(angle_diff[2]) < np.abs(tmp[2]):
                tmp = angle_diff
                path = image_path
        return path

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
