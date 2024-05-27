import cv2
import numpy as np
import shutil
from utils.similarity import CosineSimilarity
from utils.convert import Convert_YCB


def get_embedding(image, mask, box, dinov2):
    # Apply the mask directly to the original image
    masked_img = cv2.bitwise_and(image, image, mask=mask)

    # Crop the masked image using the bounding box
    x0, y0, x1, y1 = map(int, box)
    cropped_masked_img = masked_img[y0:y1, x0:x1]

    # Only used for testing without SAM
    # cropped_masked_img = image[y0:y1, x0:x1]

    # Calculate the size needed for a square image
    height, width, _ = cropped_masked_img.shape
    side_length = max(width, height)
    square_img = np.zeros((side_length, side_length, 3), dtype=np.uint8)

    # Calculate the position to place the cropped image in the square image
    x_offset = (side_length - width) // 2
    y_offset = (side_length - height) // 2
    square_img[y_offset:y_offset + height, x_offset:x_offset + width] = cropped_masked_img

    # Resize the square image to dinov2's required input size
    final_img_resized = cv2.resize(square_img, dinov2.dinov2_size, interpolation=cv2.INTER_LANCZOS4)

    # Forward through DinoV2 and get embedding
    img_array = np.array(final_img_resized)
    embed_img = dinov2.forward(img_array)
    embed_img = embed_img.detach().cpu()  # Detach from GPU
    return embed_img, final_img_resized


def choose_from_viewpoints(img, pred, dinov2, save=False):
    """
    :param img: RGB cv2 image
    :param pred: the prediction of masks which includes 'boxes', 'scores', 'labels', 'masks'
    :param dinov2: DinoV2 model, used to embed the image to a tensor of (1,1536)
    :return: filename: filename of the chosen viewpoint
             pose: object pose of the chosen viewpoint
    """
    num_predictions = len(pred['labels'])
    CosineSim = CosineSimilarity()
    L2Dist = L2Distance()
    convert_string = Convert_YCB()
    best_pred = 0
    if num_predictions > 1:
        embed_imgs = []
        isolated_imgs = []
        original_label = pred['labels'][0]
        label = convert_string.convert_name(original_label)
        cos_similarities = np.zeros((num_predictions, len(dinov2.viewpoints_embeddings[label])))
        pair_similarities = np.zeros((1, len(dinov2.viewpoints_embeddings[label])))

        # Process each prediction mask
        for i in range(num_predictions):
            img_copy = img.copy()  # preserve the original image
            mask = pred['masks'][i].cpu().numpy().astype(np.uint8)
            mask = np.transpose(mask, (1, 2, 0))  # Change order to (H, W, C) for CV2
            box = pred['boxes'][i].cpu().numpy().astype(int)  # Format: [x0, y0, x1, y1]
            embed_img, isolated_img = get_embedding(img_copy, mask, box, dinov2)

            # Calculate similarity
            reference_embedding = dinov2.viewpoints_embeddings[label]
            for col, ref in enumerate(reference_embedding):
                cos_similarities[i, col] = CosineSim(embed_img, ref).item()
            embed_imgs.append(embed_img)
            isolated_imgs.append(isolated_img)

        # Choose the best viewpoint
        best_candidate_index = np.argmax(np.mean(cos_similarities, axis=1))
        embed_img = embed_imgs[best_candidate_index]
        best_pred = best_candidate_index
        reference_embedding = dinov2.viewpoints_embeddings[label]
        for col, ref in enumerate(reference_embedding):
            # pair_similarities[0, col] = L2Dist(embed_img, ref).item()
            pair_similarities[0, col] = CosineSim(embed_img, ref).item()
        isolated_img = isolated_imgs[best_candidate_index]
        # best_vp_index = np.argmin(pair_similarities)
        best_vp_index = np.argmax(pair_similarities)
        vp_img_path = dinov2.viewpoints_images[label][best_vp_index]
        vp_pose = list(dinov2.viewpoints_poses[label].values())[best_vp_index]

    else:
        # Single prediction case
        img_copy = img.copy()  # preserve the original image
        mask = pred['masks'][0].cpu().numpy().astype(np.uint8)
        mask = np.transpose(mask, (1, 2, 0))  # Change order to (H, W, C) for CV2
        box = pred['boxes'][0].cpu().numpy().astype(int)  # Format: [x0, y0, x1, y1]
        original_label = pred['labels'][0]
        label = convert_string.convert_name(original_label)
        pair_similarities = np.zeros((1, len(dinov2.viewpoints_embeddings[label])))

        embed_img, isolated_img = get_embedding(img_copy, mask, box, dinov2)
        reference_embedding = dinov2.viewpoints_embeddings[label]
        for col, ref in enumerate(reference_embedding):
            # pair_similarities[0, col] = L2Dist(embed_img, ref).item()
            pair_similarities[0, col] = CosineSim(embed_img, ref).item()

        # best_vp_index = np.argmin(pair_similarities)
        best_vp_index = np.argmax(pair_similarities)
        vp_img_path = dinov2.viewpoints_images[label][best_vp_index]
        vp_pose = list(dinov2.viewpoints_poses[label].values())[best_vp_index]

    if save:
        cv2.imwrite('./outputs/isolated.jpg', cv2.cvtColor(isolated_img, cv2.COLOR_RGB2BGR))
        shutil.copy(vp_img_path, './outputs/best_vp.png')

    return vp_img_path, vp_pose, best_pred, embed_img, isolated_img


def validate_preds(img, pred, dinov2):
    """
    :param img: RGB cv2 image
    :param pred: the prediction of masks which includes 'boxes', 'scores', 'labels', 'masks'
    :param dinov2: DinoV2 model, used to embed the image to a tensor of (1,1536)
    :return: filename: filename of the chosen viewpoint
             pose: object pose of the chosen viewpoint
    """
    num_predictions = len(pred['labels'])
    CosineSim = CosineSimilarity()
    convert_string = Convert_YCB()
    best_pred = 0
    if num_predictions > 1:
        original_label = pred['labels'][0]
        label = convert_string.convert_name(original_label)
        cos_similarities = np.zeros((num_predictions, len(dinov2.viewpoints_embeddings[label])))

        # Process each prediction mask
        for i in range(num_predictions):
            img_copy = img.copy()  # preserve the original image
            mask = pred['masks'][i].cpu().numpy().astype(np.uint8)
            mask = np.transpose(mask, (1, 2, 0))  # Change order to (H, W, C) for CV2
            box = pred['boxes'][i].cpu().numpy().astype(int)  # Format: [x0, y0, x1, y1]
            embed_img, _ = get_embedding(img_copy, mask, box, dinov2)

            # Calculate similarity
            reference_embedding = dinov2.viewpoints_embeddings[label]
            cos_similarities[i, :] = CosineSim(embed_img, reference_embedding)

        # Choose the best viewpoint
        best_pred = np.argmax(np.mean(cos_similarities, axis=1))

    return best_pred
