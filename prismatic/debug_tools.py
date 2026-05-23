import torch
import cv2
import numpy as np

# Define the denormalization function
def denormalize_from_DINO(tensor, mean = (0.484375, 0.455078125, 0.40625), std = (0.228515625, 0.2236328125, 0.224609375)):
    if len(tensor.shape) == 3:
        dtype = tensor.dtype
        mean = torch.tensor(mean, dtype=dtype).reshape(3, 1, 1)
        std = torch.tensor(std, dtype=dtype).reshape(3, 1, 1)
        return tensor * std + mean
    elif len(tensor.shape) == 4:
        dtype = tensor.dtype
        mean = torch.tensor(mean, dtype=dtype).reshape(1, 3, 1, 1)
        std = torch.tensor(std, dtype=dtype).reshape(1, 3, 1, 1)
        return tensor * std + mean
    elif len(tensor.shape) == 5: # bz, tz, cz, hz, wz
        dtype = tensor.dtype
        mean = torch.tensor(mean, dtype=dtype).reshape(1, 1, 3, 1, 1)
        std = torch.tensor(std, dtype=dtype).reshape(1, 1, 3, 1, 1)
        return tensor * std + mean

    else:
        print('Error shape', tensor.shape)
        1/0

def torch2numpy(tensor):
    if not torch.is_tensor(tensor) and len(tensor.shape) == 3:
        raise TypeError('Input should be a 2D image tensor [C, H, W].')
    tensor = torch.permute(tensor.detach(), [1, 2, 0]).cpu().numpy()
    tensor = (tensor * 255).astype(np.uint8)
    return tensor

def draw_bbox(image, bbox, label=None):    
    # Get the image dimensions
    img_height, img_width, _ = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Denormalize the bbox coordinates
    x_center, y_center, width, height = bbox.tolist()
    x_center, y_center, width, height = (
        x_center * img_width,
        y_center * img_height,
        width * img_width,
        height * img_height,
    )

    # Calculate the top-left and bottom-right coordinates
    x_min = int(x_center - width / 2)
    y_min = int(y_center - height / 2)
    x_max = int(x_center + width / 2)
    y_max = int(y_center + height / 2)

    # Ensure coordinates are within bounds
    x_min = max(0, min(img_width - 1, x_min))
    y_min = max(0, min(img_height - 1, y_min))
    x_max = max(0, min(img_width - 1, x_max))
    y_max = max(0, min(img_height - 1, y_max))

    # Draw the bounding box
    color = (0,255,0)
    thickness = 2
    image = cv2.rectangle(image, pt1=(x_min,y_min), pt2=(x_max,y_max), 
                          color=color, thickness=thickness)

    if label is not None:
        # Place the label above the bounding box
        label_position = (x_min, max(y_min - 10, 0))  # Offset above the top-left corner
        image = cv2.putText(image, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 255, 0), 1, cv2.LINE_AA)
        
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def write_image(img, name, bboxes=None, labels=None):
    if bboxes is None:
        cv2.imwrite('./DEBUG/'+name+'.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    elif labels is None:
        for ind in range(bboxes.shape[0]):
            bbox = bboxes[ind]
            img = draw_bbox(img, bbox)
        cv2.imwrite('./DEBUG/'+name+'.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    else:
        for label, bbox in zip(labels, bboxes):
            img = draw_bbox(img, bbox, label)
        cv2.imwrite('./DEBUG/'+name+'.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


import torch
import numpy as np
import os
from PIL import Image, ImageDraw

def generate_colors(num_colors):
    """Generate a set of distinguishable colors."""
    np.random.seed(42)  # Ensure consistent colors
    colors = [
        tuple(np.random.randint(0, 255, size=3).tolist()) for _ in range(num_colors)
    ]
    return colors

def draw_bboxes_on_images(
    images, bboxes, save_dir="output_images", objectness_threshold=None
):
    """
    Draw bounding boxes on images, concatenate images across the window dimension, and save the results.
    Uses predefined colors for consistency across the window and includes the objectness threshold in filenames.

    Args:
        images (torch.Tensor): Tensor of shape [bz, window, cc, h, w] with values in range [0, 1].
        bboxes (torch.Tensor): Tensor of shape [bz, window, box_num, 5] (cx, cy, w, h, objectness).
        save_dir (str): Directory to save the output images.
        objectness_threshold (float, optional): If set, only boxes with objectness >= threshold are drawn.

    Returns:
        None (images are saved to `save_dir`).
    """
    bz, window, cc, h, w = images.shape
    box_num = bboxes.shape[2]  # Get number of objects per window
    os.makedirs(save_dir, exist_ok=True)

    # Generate predefined colors (1 color per object index)
    predefined_colors = generate_colors(box_num)

    # Convert threshold to string for filename
    threshold_str = f"th{objectness_threshold:.2f}" if objectness_threshold is not None else "thNone"

    for b in range(bz):
        images_list = []

        for w_idx in range(window):
            # Convert image tensor to numpy and scale to [0, 255]
            img_np = (images[b, w_idx].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)

            # Draw bounding boxes
            draw = ImageDraw.Draw(img_pil)
            for obj_idx, box in enumerate(bboxes[b, w_idx]):
                cx, cy, bw, bh, objectness = box.cpu().numpy()

                # Apply objectness filtering if threshold is set
                if objectness_threshold is None or objectness >= objectness_threshold:
                    x1 = int((cx - bw / 2) * w)
                    y1 = int((cy - bh / 2) * h)
                    x2 = int((cx + bw / 2) * w)
                    y2 = int((cy + bh / 2) * h)

                    # Assign a predefined color based on object index
                    color = predefined_colors[obj_idx]

                    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

            images_list.append(img_pil)

        # Concatenate images horizontally across the window dimension
        concatenated_image = np.hstack([np.array(img) for img in images_list])
        concatenated_pil = Image.fromarray(concatenated_image)

        # Save the concatenated image with threshold in filename
        filename = f"batch_{b}_{threshold_str}.png"
        concatenated_pil.save(os.path.join(save_dir, filename))

    print(f"Images saved in {save_dir}")


def apply_colormap(depth_map):
    """Applies a colormap to a grayscale depth map (normalized between 0-255)."""
    depth_map = (depth_map * 255).astype(np.uint8)  # Normalize to [0, 255]
    depth_map_color = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)  # Apply color mapping
    return depth_map_color

import pickle
def draw_bboxes_with_depth(
    images, bboxes, depths=None, window_step=1, save_dir="output_images", objectness_threshold=None, interactable_only=False, **kwargs
):
    """
    Draw bounding boxes on images, optionally overlay depth maps inside bounding box regions, 
    concatenate images across the window, and save.

    Args:
        images (torch.Tensor): Tensor of shape [bz, window, cc, h, w] with values in range [0, 1].
        bboxes (torch.Tensor): Tensor of shape [bz, window, box_num, 5] (cx, cy, w, h, objectness).
        depths (torch.Tensor or None): Tensor of shape [bz, window, box_num, 1, 40, 40] with values in range [0, 1], or None.
        save_dir (str): Directory to save the output images.
        objectness_threshold (float, optional): If set, only boxes with objectness >= threshold are drawn.

    Returns:
        None (images are saved to `save_dir`).
    """
    bz, window, cc, h, w = images.shape
    box_num = bboxes.shape[2]  # Get number of objects per window
    os.makedirs(save_dir, exist_ok=True)

    # Generate predefined colors (1 color per object index)
    predefined_colors = generate_colors(box_num)

    # Convert threshold to string for filename
    threshold_str = f"th{objectness_threshold:.2f}" if objectness_threshold is not None else "thNone"
    depth_str = "depth" if depths is not None else "nodepth"  # Include depth status in filename

    for b in range(bz):
        raw_images_list = []
        images_list = []

        for w_idx in np.arange(0, window, window_step):
            # Convert image tensor to numpy and scale to [0, 255]
            img_np = (images[b, w_idx].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            img_pil = img_np.copy()

            # Draw bounding boxes
            for obj_idx, box in enumerate(bboxes[b, w_idx]):
                if not interactable_only and obj_idx not in [0, 2, 3, 4, 6, 7, 8, 9, 10, 11, 14, 15]:
                    continue      
                cx, cy, bw, bh, objectness = box.cpu().numpy()[:5]
                if len(box.cpu().numpy()) == 6:
                    interactable_score = box.cpu().numpy()[5]
                else:
                    interactable_score = 1

                # Apply objectness filtering if threshold is set
                if objectness_threshold is None or objectness >= objectness_threshold:
                    if (interactable_only and interactable_score < 0.5):
                        continue
                    x1 = int((cx - bw / 2) * w)
                    y1 = int((cy - bh / 2) * h)
                    x2 = int((cx + bw / 2) * w)
                    y2 = int((cy + bh / 2) * h)

                    # Assign a predefined color based on object index
                    color = predefined_colors[obj_idx]
                    cv2.rectangle(img_pil, (x1, y1), (x2, y2), color, thickness=3)
                        
                    # Overlay depth map inside the bounding box (only if depths is provided)
                    if depths is not None:
                        depth_map = depths[b, w_idx, obj_idx, 0].cpu().numpy()  # Shape (40, 40)
                        depth_resized = cv2.resize((depth_map*255).astype(np.uint8), (x2 - x1, y2 - y1))
                        depth_colored = cv2.applyColorMap(depth_resized, cv2.COLORMAP_JET)  # Apply colormap
                        alpha = 0.5
                        img_pil[y1:y2, x1:x2] = cv2.addWeighted(img_pil[y1:y2, x1:x2], 1 - alpha, depth_colored, alpha, 0)

                    if interactable_only and depths is not None:
                        tmp = img_np.copy()
                        cv2.rectangle(tmp, (x1, y1), (x2, y2), color, thickness=3)
                        tmp[y1:y2, x1:x2] = cv2.addWeighted(tmp[y1:y2, x1:x2], 1 - alpha, depth_colored, alpha, 0)
                        filename = f"batch_{b}_obj{obj_idx}.png"
                        tmp = Image.fromarray(tmp)
                        tmp.save(os.path.join(save_dir, filename))
                        print("interaction saved at", os.path.join(save_dir, filename))


            images_list.append(img_pil)
            raw_images_list.append(img_np)

        # Concatenate images horizontally across the window dimension
        concatenated_image = np.hstack([np.array(img) for img in images_list])
        concatenated_pil = Image.fromarray(concatenated_image)
        raw_concatenated_image = np.hstack([np.array(img) for img in raw_images_list])
        raw_concatenated_pil = Image.fromarray(raw_concatenated_image)

        # Save the concatenated image with threshold & depth status in filename
        filename = f"batch_{b}_{threshold_str}_{depth_str}.png"
        if 'wrist' in kwargs:
            filename = filename.replace('batch_', 'batch_wrist_')
        concatenated_pil.save(os.path.join(save_dir, filename))
        filename = f"batch_{b}_raw.png"
        if 'wrist' in kwargs:
            filename = filename.replace('batch_', 'batch_wrist_')
        raw_concatenated_pil.save(os.path.join(save_dir, filename))
        if 'features' in kwargs:
            print("saving features...")
            pred_feat = kwargs['features'].detach().cpu()
            # Save tensor using pickle
            with open(os.path.join(save_dir, f"batch_{b}_{threshold_str}_{depth_str}.pkl"), "wb") as f:
                pickle.dump(pred_feat[b], f)

    print(f"Images saved in {save_dir}")