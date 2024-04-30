import cv2
import numpy as np
import os

def process_images(root_dir):
    # Walk through each directory and subdirectory
    for subdir, dirs, files in os.walk(root_dir):
        paired_files = {}
        
        # Prepare pairs of color and depth images
        for file in files:
            if file.endswith('-color.png') or file.endswith('-depth.png'):
                base_name = file.split('-')[0]
                if base_name not in paired_files:
                    paired_files[base_name] = {}
                if '-color.png' in file:
                    paired_files[base_name]['color'] = os.path.join(subdir, file)
                elif '-depth.png' in file:
                    paired_files[base_name]['depth'] = os.path.join(subdir, file)
        
        # Process each pair
        for key, paths in paired_files.items():
            if 'color' in paths and 'depth' in paths:
                color_image_path = paths['color']
                depth_image_path = paths['depth']

                # Read color and depth images
                color_img = cv2.imread(color_image_path)
                depth_img = cv2.imread(depth_image_path, -1)  # Load depth image as is (16-bit)

                # Get non-black pixel coordinates
                indices = np.where(np.all(color_img != [0, 0, 0], axis=-1))
                if indices[0].size == 0 or indices[1].size == 0:
                    continue  # Skip if no non-black pixels found

                # Calculate bounding box and make it square
                x_min, x_max = min(indices[1]), max(indices[1])
                y_min, y_max = min(indices[0]), max(indices[0])
                box_size = max(x_max - x_min, y_max - y_min)

                # Center the square box
                x_center = (x_min + x_max) // 2
                y_center = (y_min + y_max) // 2
                x_min = max(0, x_center - box_size // 2)
                y_min = max(0, y_center - box_size // 2)
                x_max = min(color_img.shape[1], x_center + box_size // 2)
                y_max = min(color_img.shape[0], y_center + box_size // 2)

                # Crop and resize both images
                cropped_color_img = color_img[y_min:y_max, x_min:x_max]
                cropped_depth_img = depth_img[y_min:y_max, x_min:x_max]

                resized_color_img = cv2.resize(cropped_color_img, (224, 224), interpolation=cv2.INTER_LINEAR)
                resized_depth_img = cv2.resize(cropped_depth_img, (224, 224), interpolation=cv2.INTER_NEAREST)

                # Save images back, replacing the original
                cv2.imwrite(color_image_path, resized_color_img)
                cv2.imwrite(depth_image_path, resized_depth_img)

if __name__ == '__main__':
    root_directory = './viewpoints_42'  # Change this to your directory
    process_images(root_directory)

