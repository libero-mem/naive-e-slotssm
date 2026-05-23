import argparse
import json
import os
import time

import cv2
import h5py
import numpy as np
import tqdm
from libero.libero import benchmark
import robosuite.utils.transform_utils as T

IMAGE_RESOLUTION = 256

def get_task_nouns(task_description):
    if task_description in TASK_NOUN_DICT:
        return TASK_NOUN_DICT[task_description]
    print(task_description)
    # 1/0
    return []

def min_max_normalize(image, old_min=0, old_max=1, new_min=0, new_max=1):
    """
    Normalize an image using Min-Max normalization.

    Parameters:
        image (numpy array): Input image.
        new_min (float): Minimum value of the normalized range.
        new_max (float): Maximum value of the normalized range.

    Returns:
        normalized_image (numpy array): Normalized image.
    """
    # Convert the image to float for processing
    image = image.astype(np.float32)

    # Get min and max of the original image
    old_min = np.min(image)
    old_max = np.max(image)
    # if DEBUG_MODE:
    #     print(old_min, old_max)
    # Apply Min-Max normalization formula
    normalized_image = (image - old_min) / (old_max - old_min) * (new_max - new_min) + new_min

    return (normalized_image*255).astype(np.uint8)

def get_visual(obs, key, normalize_depth=True):
    """
    Processes a visual observation (image, segmentation, or depth) from obs.

    Args:
        obs (dict): The observation dictionary from the environment.
        key (str): The key to retrieve the image/segmentation/depth.
        normalize_depth (bool): Whether to normalize depth images automatically.

    Returns:
        np.ndarray: The processed visual output.
    """
    # Fetch and flip horizontally
    visual = obs[key][:, ::-1]

    # Depth maps need normalization
    if 'depth' in key:
        if 'agentview' in key:
            visual = min_max_normalize(visual, old_min=0.97, old_max=1.0)
        elif 'eye_in_hand' in key:
            visual = min_max_normalize(visual, old_min=0.71, old_max=1.0)
        else:
            if normalize_depth:
                visual = min_max_normalize(visual)  # fallback

    # Segmentations need to be uint8
    if 'segmentation' in key:
        visual = visual.astype(np.uint8)

    # Rotate 180 degrees
    visual = cv2.rotate(visual, cv2.ROTATE_180)

    return visual

color4label = {}
def get_color4label(label):
    if label in color4label:
        return color4label[label]
    else:
        new_color = np.random.randint(0, 255, size=(3, ))
        color4label[label] = tuple(new_color.astype(np.int32))
        return color4label[label]

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from torchvision.ops.boxes import box_area


# Convert from center format (cx, cy, w, h) to corner format (x0, y0, x1, y1)
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


# Convert from corner format (x0, y0, x1, y1) to center format (cx, cy, w, h)
def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# Compute IoU between two sets of boxes
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)  # [N]
    area2 = box_area(boxes2)  # [M]

    # Intersection top-left & bottom-right
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # width-height intersection
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    union = area1[:, None] + area2 - inter
    eps = 1e-6  # avoid division by zero
    iou = inter / (union + eps)

    return iou, union


# Compute Generalized IoU (GIoU) between two sets of boxes
def generalized_box_iou(boxes1, boxes2):
    # Sanity check for box validity
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()

    iou, union = box_iou(boxes1, boxes2)

    # Compute enclosing box
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)
    area = wh[:, :, 0] * wh[:, :, 1]  # enclosing box area

    eps = 1e-6
    giou = iou - (area - union) / (area + eps)

    return giou

class HungarianMatcherboxTRACK(nn.Module):
    def __init__(self, cost_bbox: float = 1, cost_giou: float = 1):
        """
        Initializes the Hungarian matcher.

        Args:
            cost_bbox (float): Weight for bbox L1 distance cost.
            cost_giou (float): Weight for GIoU cost.
        """
        super().__init__()
        assert cost_bbox != 0 or cost_giou != 0, 'All costs cannot be 0'
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Performs the matching.

        Args:
            outputs: Tensor of shape [B, ..., num_queries, horizon, 5]
            targets: List of ground-truth tensors for each batch item

        Returns:
            Tuple of (pred_indices, target_indices) with matched indices per batch
        """
        bs = outputs.shape[0]
        num_queries = outputs.shape[-3]
        horizon = outputs.shape[-2]

        indices = []

        for b in range(bs):
            out_bbox = outputs[b].float()  # [num_queries, horizon, 5]

            if targets[b] is None:
                continue

            tgt_boxes = targets[b].float().to(out_bbox.device)

            # Permute to [num_queries, horizon, 5] if needed
            out_bbox = out_bbox.permute(1, 0, 2)
            tgt_boxes = tgt_boxes.permute(1, 0, 2)

            # Compute L1 distance over all horizon steps, sum across time
            cost_bbox = torch.cdist(out_bbox, tgt_boxes, p=1).sum(0)

            # Compute GIoU-based cost
            cost_giou = 0
            for h in range(horizon):
                cost_giou += -generalized_box_iou(
                    box_cxcywh_to_xyxy(out_bbox[h, :, :4]),
                    box_cxcywh_to_xyxy(tgt_boxes[h, :, :4])
                )

            # Final cost matrix
            C = self.cost_bbox * cost_bbox + self.cost_giou * cost_giou  # [num_queries, num_queries]
            C = C.cpu()

            try:
                row_ind, col_ind = linear_sum_assignment(C)
                indices.append((torch.as_tensor(row_ind, dtype=torch.int64),
                                torch.as_tensor(col_ind, dtype=torch.int64)))
            except Exception as e:
                print("Hungarian matching failed for batch", b)
                print("Cost matrix:\n", C)
                raise e

        return indices


class TrajectoryRecorder:
    def __init__(self, image_resolution=128):
        self.image_resolution = image_resolution

    def reset_trackstate(self):
        # Storage
        self.states = []
        self.actions = []
        self.ee_states = []
        self.gripper_states = []
        self.joint_states = []
        self.robot_states = []
        self.agentview_images = []
        self.eye_in_hand_images = []
        self.agentview_depths = []
        self.eye_in_hand_depths = []
        self.agentview_segs = []
        self.eye_in_hand_segs = []
        self.agentview_boxes = []
        self.eye_in_hand_boxes = []
        self.cur_obs = None
        self.collision_states = {}
        for noun in self.nouns:
            label = self.remapper[noun]
            self.collision_states[label] = 0
        self.matcher = HungarianMatcherboxTRACK(2.5, 2)

    def reset_object_mappers(self, env):
        ##########################################
        # """ Functions to handle utilities """"
        def simplify(text):
            for key, value in SIMPLIFYING_DICT.items():
                if key in text:
                    return text.replace(key, value)
            return text

        def revise_remapper(remapper, double_filter=True):
            for key, value in remapper.items():
                value = value.split(' ')
                value = ' '.join(value)

                if double_filter:
                    value = simplify(value)

                remapper[key] = value

            return remapper

        def get_manual_geomid(geom_name2id, geom_names=['']):
            ids = []
            for name in geom_names:
                ids.append(geom_name2id[name])
            return ids

        def not_match_any(geom_id, selected_ids):
            if geom_id is None:
                return False
            for id in selected_ids:
                if id in geom_id:
                    return False
            return True
        ##########################################

        nouns = ['gripper0']
        remapper = {
            'gripper0': 'robot'
        }
        
        # Retrieve the mapping from geometry IDs to names
        geom_name2id = {env.sim.model.geom_id2name(geom_id): geom_id  
                        for geom_id in range(env.sim.model.ngeom)}
        geom_ids = list(geom_name2id.keys())

        # Retrieve the mapping from each noun to geom id
        id_list = {}
        for noun in nouns:
            re_noun = noun.replace(' ', '_')
            id_list[re_noun] = []
            for geom in geom_ids:
                # print(geom)
                if geom is not None and re_noun in geom:
                    if re_noun == 'book':
                        if geom not in ['office_book_shelf', 'black_book_1', 'black_book_2']:
                            id_list[re_noun].append(geom_name2id[geom])
                    elif re_noun == 'shelf':
                        if geom not in ['office_book_shelf']:
                            id_list[re_noun].append(geom_name2id[geom])
                    else:
                        id_list[re_noun].append(geom_name2id[geom])
        id_list['gripper0'] = []
        for geom in geom_ids:
            # print(geom)
            if geom is not None and 'gripper0' in geom:
                id_list['gripper0'].append(geom_name2id[geom])

        if 'wooden_two_layer_shelf_1_g0' in geom_name2id:
            keywords = ['cabinet', 'cabinet shelf']
            for keyword in keywords:
                if keyword in id_list and len(id_list[keyword]) == 0:
                    id_list[keyword] = get_manual_geomid(geom_name2id, geom_names=['wooden_two_layer_shelf_1_g0', 'wooden_two_layer_shelf_1_g1', 'wooden_two_layer_shelf_1_g2', 'wooden_two_layer_shelf_1_g3', 'wooden_two_layer_shelf_1_g4', 'wooden_two_layer_shelf_1_g5', 'wooden_two_layer_shelf_1_g6'])

        if 'wooden_cabinet_1_g0' in geom_name2id:
            if 'top_drawer' in id_list:
                id_list['top_drawer'] = get_manual_geomid(geom_name2id, geom_names=['wooden_cabinet_1_g6', 'wooden_cabinet_1_g7', 'wooden_cabinet_1_g8', 'wooden_cabinet_1_g9', 'wooden_cabinet_1_g10', 'wooden_cabinet_1_g11', 'wooden_cabinet_1_g12', 'wooden_cabinet_1_g13', 'wooden_cabinet_1_g14', 'wooden_cabinet_1_g15', 'wooden_cabinet_1_g16', 'wooden_cabinet_1_g17', 'wooden_cabinet_1_g18'])
            if 'middle_drawer' in id_list:
                id_list['middle_drawer'] = get_manual_geomid(geom_name2id, geom_names=['wooden_cabinet_1_g19', 'wooden_cabinet_1_g20', 'wooden_cabinet_1_g21', 'wooden_cabinet_1_g22', 'wooden_cabinet_1_g23', 'wooden_cabinet_1_g24', 'wooden_cabinet_1_g25', 'wooden_cabinet_1_g26', 'wooden_cabinet_1_g27', 'wooden_cabinet_1_g28', 'wooden_cabinet_1_g29', 'wooden_cabinet_1_g30', 'wooden_cabinet_1_g31'])
            if 'bottom_drawer' in id_list:
                id_list['bottom_drawer'] = get_manual_geomid(geom_name2id, geom_names=['wooden_cabinet_1_g32', 'wooden_cabinet_1_g33', 'wooden_cabinet_1_g34', 'wooden_cabinet_1_g35', 'wooden_cabinet_1_g36', 'wooden_cabinet_1_g37', 'wooden_cabinet_1_g38', 'wooden_cabinet_1_g39', 'wooden_cabinet_1_g40', 'wooden_cabinet_1_g41', 'wooden_cabinet_1_g42'])
        elif 'white_cabinet_1_g0' in geom_name2id:
            if 'top_drawer' in id_list:
                id_list['top_drawer'] = get_manual_geomid(geom_name2id, geom_names=['white_cabinet_1_g6', 'white_cabinet_1_g7', 'white_cabinet_1_g8', 'white_cabinet_1_g9', 'white_cabinet_1_g10', 'white_cabinet_1_g11', 'white_cabinet_1_g12', 'white_cabinet_1_g13', 'white_cabinet_1_g14', 'white_cabinet_1_g15', 'white_cabinet_1_g16', 'white_cabinet_1_g17', 'white_cabinet_1_g18'])
            if 'middle_drawer' in id_list:
                id_list['middle_drawer'] = get_manual_geomid(geom_name2id, geom_names=['white_cabinet_1_g19', 'white_cabinet_1_g20', 'white_cabinet_1_g21', 'white_cabinet_1_g22', 'white_cabinet_1_g23', 'white_cabinet_1_g24', 'white_cabinet_1_g25', 'white_cabinet_1_g26', 'white_cabinet_1_g27', 'white_cabinet_1_g28', 'white_cabinet_1_g29', 'white_cabinet_1_g30', 'white_cabinet_1_g31'])
            if 'bottom_drawer' in id_list:
                id_list['bottom_drawer'] = get_manual_geomid(geom_name2id, geom_names=['white_cabinet_1_g32', 'white_cabinet_1_g33', 'white_cabinet_1_g34', 'white_cabinet_1_g35', 'white_cabinet_1_g36', 'white_cabinet_1_g37', 'white_cabinet_1_g38', 'white_cabinet_1_g39', 'white_cabinet_1_g40', 'white_cabinet_1_g41', 'white_cabinet_1_g42'])
        
        # if 'desk_caddy_1_g0' in geom_name2id: # doesnt workp
        #     if 'front_compartment' in id_list:
        #         id_list['front_compartment'] = get_manual_geomid(geom_name2id, geom_names=['desk_caddy_1_g1'])
        #     if 'left_compartment' in id_list:
        #         id_list['left_compartment'] = get_manual_geomid(geom_name2id, geom_names=['desk_caddy_1_g0'])
        #     if 'right_compartment' in id_list:
        #         id_list['right_compartment'] = get_manual_geomid(geom_name2id, geom_names=['desk_caddy_1_g4', 'desk_caddy_1_g5'])

        remain_geom_ids = [geom_id for geom_id in geom_ids if not_match_any(geom_id, id_list.keys())]
        remain_obj_keys = [geom_id[:-3] for geom_id in remain_geom_ids if '_g0' in geom_id and 'robot0' not in geom_id]

        for key in remain_obj_keys:
            if 'chefmate_8_frypan' in key:
                remapper[key] = 'frying pan'
            else:
                remapper[key] = key.replace('_',' ')
            re_noun = key
            id_list[re_noun] = []
            for geom in geom_ids:
                # print(geom)
                if geom is not None and re_noun in geom:
                    if re_noun == 'black_book_1' and geom == 'black_book_1':
                        continue
                    id_list[re_noun].append(geom_name2id[geom])
        nouns+= remain_obj_keys

        remapping_values = list(remapper.values()) # assert that all verbs belong to this remapper
        self.remapper = revise_remapper(remapper)
        self.remapping_values = remapping_values
        self.id_list = id_list
        self.nouns = nouns

    def record(self, obs, action):
        """
        Record one step worth of data
        """

        # Save action
        self.actions.append(action)

        # Save gripper, joint, and end-effector state
        if "robot0_gripper_qpos" in obs:
            self.gripper_states.append(obs["robot0_gripper_qpos"])
        self.joint_states.append(obs["robot0_joint_pos"])
        self.ee_states.append(
            np.hstack((
                obs["robot0_eef_pos"],
                T.quat2axisangle(obs["robot0_eef_quat"]),
            ))
        )
        self.robot_states.append(
            np.concatenate([obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"]])
        )

        self.cur_obs = obs

        # Save rotated images
        self.agentview_images.append(get_visual(obs, "agentview_image"))
        self.eye_in_hand_images.append(get_visual(obs, "robot0_eye_in_hand_image"))

        # Process segmentations
        ego_objs, exo_objs, ego_seg, exo_seg = self.process_segmentations(
            obs,
        )

        self.eye_in_hand_boxes.append(ego_objs)
        ego_image = get_visual(obs, "robot0_eye_in_hand_image")
        for label, data in ego_objs.items():
            ind, bbox = data
            color = get_color4label(label)
            color = ( int (color [ 0 ]), int (color [ 1 ]), int (color [ 2 ])) 

            x_min, y_min, w, h = bbox
            x_max, y_max = int((x_min + w)*IMAGE_RESOLUTION), int((y_min + h)*IMAGE_RESOLUTION)
            x_min, y_min = int(x_min*IMAGE_RESOLUTION), int(y_min*IMAGE_RESOLUTION)
            cv2.rectangle(ego_image, (x_min, y_min), (x_max, y_max), color=color, thickness=2)
            text_position = (x_min, y_min - 10 if y_min - 10 > 10 else y_min + 10)
            cv2.putText(ego_image, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=color, thickness=2)
        ego_image_boxed = ego_image

        self.agentview_boxes.append(exo_objs)
        exo_image = get_visual(obs, "agentview_image")
        for label, data in exo_objs.items():
            ind, bbox = data
            color = get_color4label(label)
            color = ( int (color [ 0 ]), int (color [ 1 ]), int (color [ 2 ])) 

            x_min, y_min, w, h = bbox
            x_max, y_max = int((x_min + w)*IMAGE_RESOLUTION), int((y_min + h)*IMAGE_RESOLUTION)
            x_min, y_min = int(x_min*IMAGE_RESOLUTION), int(y_min*IMAGE_RESOLUTION)
            cv2.rectangle(exo_image, (x_min, y_min), (x_max, y_max), color=color, thickness=2)
            text_position = (x_min, y_min - 10 if y_min - 10 > 10 else y_min + 10)
            cv2.putText(exo_image, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=color, thickness=2)
        exo_image_boxed = exo_image

        self.eye_in_hand_segs.append(ego_seg)
        self.agentview_segs.append(exo_seg)

        # Save depths
        exo_depth = get_visual(obs, "agentview_depth")
        ego_depth = get_visual(obs, "robot0_eye_in_hand_depth")
        self.agentview_depths.append(exo_depth)
        self.eye_in_hand_depths.append(ego_depth)
        visuals = {
            'exo_rgb' : self.agentview_images[-1],  # 'exo rgb'
            'exo_rgb_boxed' : exo_image_boxed,  # 'exo rgb boxed'
            'exo_depth' : exo_depth,  # 'exo depth'
            'exo_seg' : exo_seg,  # 'exo seg'

            'ego_rgb' : self.eye_in_hand_images[-1],  # 'ego rgb'
            'ego_rgb_boxed' : ego_image_boxed,  # 'ego rgb boxed'
            'ego_depth' : ego_depth,  # 'ego depth'
            'ego_seg' : ego_seg,  # 'ego seg'
        }
        return visuals

    def process_segmentations(self, obs):
        ego_objs = {}
        exo_objs = {}
        ego_seg = np.zeros_like(obs['robot0_eye_in_hand_segmentation_robot0_eye_in_hand'])[:,:,0]
        exo_seg = np.zeros_like(obs['agentview_segmentation_agentview'])[:,:,0]

        for ind, noun in enumerate(self.nouns):
            label = self.remapper[noun]

            seg = get_visual(obs, 'robot0_eye_in_hand_segmentation_robot0_eye_in_hand')
            seg_mask = seg_to_mask(seg, self.id_list, noun.replace(' ', '_'))
            ego_seg += seg_mask * ((ind + 1) * 10)
            ego_box = get_bbox(seg_mask)

            seg = get_visual(obs, 'agentview_segmentation_agentview')
            seg_mask = seg_to_mask(seg, self.id_list, noun.replace(' ', '_'))
            exo_seg += seg_mask * ((ind + 1) * 10)
            exo_box = get_bbox(seg_mask)

            if ego_box is not None:
                ego_objs[label] = [(ind + 1) * 10, list(ego_box)]
            if exo_box is not None:
                exo_objs[label] = [(ind + 1) * 10, list(exo_box)]

        return ego_objs, exo_objs, ego_seg.astype(np.uint8), exo_seg.astype(np.uint8)

    def get_collision_object(self, contact_geom):
        for key, values in self.id_list.items():
            if contact_geom in values:
                label = self.remapper[key]
                # if 'bowl' in label:
                #     self.collision_states[label] = self.collision_states['robot']
                return label
        return None

    def update_collision_state(self, contact_key):
        if contact_key is not None and 'bowl' in contact_key:
            self.collision_states['robot'] += 1
            self.collision_states['robot'] = min(self.collision_states['robot'], 16)
            self.collision_states[contact_key] = self.collision_states['robot']
        print("Collision states...")
        print(self.collision_states)
        return

    def update_collision_state_with_objs(self, visuals=None):
        for _, noun in enumerate(self.nouns):
            label = self.remapper[noun]
            if label in self.agentview_boxes[-1]:
                self.agentview_boxes[-1][label].append(self.collision_states[label])
            if label in self.eye_in_hand_boxes[-1]:
                self.eye_in_hand_boxes[-1][label].append(self.collision_states[label])
        
        if visuals is not None:
            exo_image_contact = visuals['exo_rgb'].copy()
            ego_image_contact = visuals['ego_rgb'].copy()
            for label, data in self.agentview_boxes[-1].items():
                ind, bbox, contact = data
                color = get_color4label(label)
                color = ( int (color [ 0 ]), int (color [ 1 ]), int (color [ 2 ])) 

                x_min, y_min, w, h = bbox
                x_max, y_max = int((x_min + w)*IMAGE_RESOLUTION), int((y_min + h)*IMAGE_RESOLUTION)
                x_min, y_min = int(x_min*IMAGE_RESOLUTION), int(y_min*IMAGE_RESOLUTION)
                cv2.rectangle(exo_image_contact, (x_min, y_min), (x_max, y_max), color=color, thickness=2)
                text_position = (x_min, y_min - 10 if y_min - 10 > 10 else y_min + 10)
                cv2.putText(exo_image_contact, str(contact), text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=color, thickness=2)

            for label, data in self.eye_in_hand_boxes[-1].items():
                ind, bbox, contact = data
                color = get_color4label(label)
                color = ( int (color [ 0 ]), int (color [ 1 ]), int (color [ 2 ])) 

                x_min, y_min, w, h = bbox
                x_max, y_max = int((x_min + w)*IMAGE_RESOLUTION), int((y_min + h)*IMAGE_RESOLUTION)
                x_min, y_min = int(x_min*IMAGE_RESOLUTION), int(y_min*IMAGE_RESOLUTION)
                cv2.rectangle(ego_image_contact, (x_min, y_min), (x_max, y_max), color=color, thickness=2)
                text_position = (x_min, y_min - 10 if y_min - 10 > 10 else y_min + 10)
                cv2.putText(ego_image_contact, str(contact), text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=color, thickness=2)

            visuals['exo_rgb_boxed_contact'] = exo_image_contact  # 'exo rgb boxed'
            visuals['ego_rgb_boxed_contact'] = ego_image_contact  # 'ego rgb boxed'
        return

    def get_data(self):
        """Return all the recorded data (optional to implement)."""
        data = {
            'states': self.states,
            'actions': self.actions,
            'ee_states': self.ee_states,
            'gripper_states': self.gripper_states,
            'joint_states': self.joint_states,
            'robot_states': self.robot_states,
            'agentview_images': self.agentview_images,
            'eye_in_hand_images': self.eye_in_hand_images,
            'agentview_depths': self.agentview_depths,
            'eye_in_hand_depths': self.eye_in_hand_depths,
            'agentview_segs': self.agentview_segs,
            'eye_in_hand_segs': self.eye_in_hand_segs,
            'agentview_boxes': self.agentview_boxes,
            'eye_in_hand_boxes': self.eye_in_hand_boxes,
        }
        return data

    def get_matches(self, preds, targets):
        return self.matcher(preds, targets)

import matplotlib.cm as cm
# deterministic shuffling of values to map each geom ID to a random int in [0, 255]
rstate = np.random.RandomState(seed=8)
inds = np.arange(256)
rstate.shuffle(inds)

def segmentation_to_rgb(seg_im):
    """
    Helper function to visualize segmentations as RGB frames.
    NOTE: assumes that geom IDs go up to 255 at most - if not,
    multiple geoms might be assigned to the same color.
    """
    # ensure all values lie within [0, 255]
    # seg_im = seg_im[:,:,0]
    seg_im = np.mod(seg_im, 256)
    # use @inds to map each geom ID to a color
    mapped = (255.0 * cm.rainbow(inds[seg_im], 3)).astype(np.uint8)[..., :3]
    return mapped

def imshow(name, image):
    # image = cv2.rotate(image.astype(np.uint8), cv2.ROTATE_180)
    cv2.imshow(name, image)

# import spacy
# # Load the spaCy English model
# nlp = spacy.load("en_core_web_sm")

def process_nonessential(noun):
    to_removes = ['both', 'the', 'a', 'an']
    # Split the string into words
    words = noun.split()
    # Filter out the words that need to be removed
    filtered_words = [word for word in words if word not in to_removes]
    # Join the filtered words back into a string
    return ' '.join(filtered_words)

mapper = {
    'cream cheese box': 'cream_cheese',
    'moka pot': 'moka_pot',
    'moka pots': 'moka_pot',
    'white mug': 'porcelain_mug',
    'yellow and white mug': 'white_yellow_mug',
    'right plate': 'plate_1',
    'left plate': 'plate_2 ',
    'cookie box': 'cookies',
    'middle black bowl': 'akita_black_bowl_2',
    'back black bowl': 'akita_black_bowl_3',
    'frying pan': 'chefmate_8_frypan',
    'right moka pot': 'moka_pot_1',
    'cabinet shelf': 'wooden_two_layer_shelf',
    'left bowl': 'akita_black_bowl_1',
    'right bowl': 'akita_black_bowl_2',
    'red mug': 'red_coffee_mug'
}

SIMPLIFYING_DICT = {
    'akita black bowl': 'bowl',
    'wooden cabinet': 'cabinet',
    'wine bottle': 'bottle',
    'wine rack': 'rack',
    'white cabinet': 'cabinet'
}

def check_contact(env, id_list):
    # Access the simulation data
    sim = env.sim
    touched = False
    # Iterate through all contacts
    for i in range(sim.data.ncon):
        contact = sim.data.contact[i]
        
        # Get the geom names involved in the contact
        geom1 = sim.model.geom_id2name(contact.geom1)
        geom2 = sim.model.geom_id2name(contact.geom2)
        
        # Check if the contact is between the gripper and the object
        if ("gripper0_finger" in geom1) and ("gripper" not in geom2):
            # print(geom1, geom2)
            # print(f"Object '{geom2}' is in contact with the gripper!")
            # Extract the contact position (in world coordinates)
            # contact_position = contact.pos  # This gives the x, y, z position of contact in world space
            # print(f"Contact position: {contact_position}")
            # Create a temporary site (red sphere) at the contact point
            # site_id = sim.model.site_name2id("gripper0_grip_site")
            # sim.model.site_rgba[site_id] = np.array([1, 0, 0, 1])  # Red color
            # sim.data.site_xpos[site_id] = contact_position
            # sim.forward()

            for key, values in id_list.items():
                if contact.geom2 in values:
                    return key
            
            return None # table or background

        elif ("gripper0_finger" in geom2) and ("gripper" not in geom1):
            # print(geom1, geom2)
            # print(f"Object '{geom1}' is in contact with the gripper!")
            # Extract the contact position (in world coordinates)
            # contact_position = contact.pos  # This gives the x, y, z position of contact in world space
            # print(f"Contact position: {contact_position}")
            # Create a temporary site (red sphere) at the contact point
            # site_id = sim.model.site_name2id("gripper0_grip_site")
            # sim.model.site_rgba[site_id] = np.array([1, 0, 0, 1])  # Red color
            # sim.data.site_xpos[site_id] = contact_position

            # sim.forward()

            for key, values in id_list.items():
                if contact.geom1 in values:
                    return key

            return None # table or background

    # print("Gripper is not in contact with anything.")
    return None

def extract_clean_nouns(sentence):
    doc = nlp(sentence)
    # Extract noun phrases and standalone nouns
    nouns = [chunk.text for chunk in doc.noun_chunks]
    # Remove determiners like 'a', 'an', 'the'
    cleaned_nouns = [' '.join(token.text for token in nlp(noun) if token.pos_ != 'DET') for noun in nouns]
    cleaned_nouns = [process_nonessential(noun) for noun in nouns]
    return cleaned_nouns

# Define the JSON file path
json_file = "nouns_storage.json"
# Function to save nouns to JSON
def save_nouns_to_json(noun_dict, sentence, nouns):
    # Add the new sentence and its nouns
    noun_dict[sentence] = nouns

    # Write back to the JSON file
    with open(json_file, "w") as file:
        json.dump(noun_dict, file, indent=4)

# Function to retrieve nouns from JSON
def get_nouns_from_json(json_file):
    if os.path.exists(json_file):
        with open(json_file, "r") as file:
            data = json.load(file)
        return data
    else:
        return {}


def seg_to_mask(seg_image, id_list, goal):
    ids = id_list[goal]
    # Create a mask
    mask = np.isin(seg_image, ids)
    return mask


def get_bbox(mask, height=256, width=256):
    """
    Extract bounding boxes and the overarching bounding box from a binary mask.

    Args:
        mask (np.ndarray): Binary mask as a NumPy array (2D).
                          Non-zero pixels are considered part of the mask.

    Returns:
        Tuple[List[Tuple[int, int, int, int]], Tuple[int, int, int, int]]:
            - A list of bounding boxes in (x_min, y_min, x_max, y_max) format.
            - The overarching bounding box in (x_min, y_min, x_max, y_max) format.
    """
    H, W = height, width

    # Ensure the mask is binary
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8) * 255

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Extract individual bounding boxes
    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area > 10:
            bounding_boxes.append((x, y, x + w, y + h))

    # Compute the overarching bounding box
    if bounding_boxes:
        x_min = min(bbox[0] for bbox in bounding_boxes)
        y_min = min(bbox[1] for bbox in bounding_boxes)
        x_max = max(bbox[2] for bbox in bounding_boxes)
        y_max = max(bbox[3] for bbox in bounding_boxes)
        x_min, y_min, x_max, y_max = x_min/W, y_min/H, x_max/W, y_max/H
        box = (x_min, y_min, x_max - x_min, y_max - y_min)
        overarching_box = np.array(box)
    else:
        # If no bounding boxes were found
        overarching_box = None

    return overarching_box

DEBUG_MODE = False
from mem_task_object_nouns import TASK_NOUN_DICT

import imageio
import re

def set_init_state(env, init_state):
    env.sim.set_state_from_flattened(init_state)
    env.sim.forward()
    env._check_success()
    env._post_process()
    env._update_observables(force=True)
    return env._get_observations()

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() 
            for text in re.split(r'(\d+)', s)]

def collision_check(env):
    # Access the simulation data
    sim = env.sim
    # Iterate through all contacts
    for i in range(sim.data.ncon):
        contact = sim.data.contact[i]
        
        # Get the geom names involved in the contact
        geom1 = sim.model.geom_id2name(contact.geom1)
        geom2 = sim.model.geom_id2name(contact.geom2)
        
        # Check if the contact is between the gripper and the object
        if ("gripper0_finger" in geom1) and ("gripper" not in geom2):
            return contact.geom2, geom2
        elif ("gripper0_finger" in geom2) and ("gripper" not in geom1):
            return contact.geom1, geom1

    return None, None



import cv2
import torch

class InteractionMonitor:
    def __init__(self, env, recorder, retry_limit=1):
        self.env = env
        self.recorder = recorder
        self.retry_limit = retry_limit
        self.prev_contact_key = None
        self.prev_grip = False
        self.object_keys = None
        self.object_cnt = None

    def get_current_objs(self, non=False):
        """ This function returns objects and the corresponding bboxes, interactoin count
        """
        data_dict = self.recorder.agentview_boxes[-1] # {obj: [seg_id, tlwh_bbox, interaction_cnt]}
        if self.object_keys is None:
            all_obj_keys = list(data_dict.keys())
            self.object_keys = all_obj_keys
        else:
            all_obj_keys = self.object_keys
        
        all_obj_bboxes = []
        for key in all_obj_keys:
            if key in data_dict:
                all_obj_bboxes.append(data_dict[key][1])
            else:
                all_obj_bboxes.append(np.zeros(4, dtype=np.float64))
        all_obj_bboxes = torch.tensor(np.array([list(all_obj_bboxes)])) # 1 x O x 4
        all_obj_bboxes[:,:,:2] += all_obj_bboxes[:,:,2:] / 2                               # tlwh to (cx cy w h)
        all_obj_bboxes = all_obj_bboxes.unsqueeze(-2) # add temporal dimension -> [1 x O x 1 x 4]

        if self.object_cnt is None:
            if non:
                self.object_cnt = {key: [0] for key in all_obj_keys} 
            else:
                self.object_cnt = {key: [data_dict[key][2] - (data_dict[key][2] % 2)] for key in all_obj_keys} 
        else:
            if non:
                for key, value in data_dict.items():
                    self.object_cnt[key] = [0]
            else:
                for key, value in data_dict.items():
                    self.object_cnt[key] = [value[2] - (value[2] % 2)]
        all_obj_cnts = torch.tensor([list([self.object_cnt[key] for key in all_obj_keys])]) # 1 x O x 1
        all_obj_cnts = all_obj_cnts.unsqueeze(-2)     # add temporal dimension -> [1 x O x 1 x 1]
        return all_obj_bboxes, all_obj_cnts

    def step(self, obs, action, time_step):
        # Record visual state
        visuals = self.recorder.record(obs, action)

        # Collision check
        contact_geom, geom = collision_check(self.env)
        curr_contact_key = self.recorder.get_collision_object(contact_geom)
        curr_grip = action[-1] > 0

        # print('time', time_step)
        # print('-- curr_contact_key', curr_contact_key)
        # print('-- prev_collision', self.prev_contact_key)
        # print('-- curr_grip', curr_grip)
        # print('-- prev_grip', self.prev_grip)
        # print('')

        counted = False
        if curr_contact_key != self.prev_contact_key and curr_grip != self.prev_grip:
            counted = True

            # retry = self.retry_limit
            # while retry >= 1:
            #     key = cv2.waitKey(1)
            #     if key == ord('c'):
            #         print("Count collision")
            #         counted = True
            #         break
            #     if key == ord('v'):
            #         print("Ignore collision")
            #         counted = False
            #         break

            if counted:
                if curr_grip:
                    self.recorder.update_collision_state(curr_contact_key)
                else:
                    self.recorder.update_collision_state(self.prev_contact_key)
                self.prev_grip = curr_grip
                self.prev_contact_key = curr_contact_key

        self.recorder.update_collision_state_with_objs(visuals)

        return visuals

    def get_matches(self, preds, targets):
        return  self.recorder.get_matches(preds, targets)
