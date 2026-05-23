"""
data_utils.py

General utilities and classes for facilitating data loading and collation.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Sequence, Tuple
import numpy as np

import torch
from torch.nn.utils.rnn import pad_sequence

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


def tree_map(fn: Callable, tree: dict) -> dict:
    """Maps a function over a nested dictionary."""
    return {k: tree_map(fn, v) if isinstance(v, dict) else fn(v) for k, v in tree.items()}


def tree_map_with_key(fn: Callable, tree: dict, keys: Sequence = ()) -> dict:
    """Maps a function over a nested dictionary."""
    return {
        k: tree_map_with_key(fn, v, (*keys, k)) if isinstance(v, dict) else fn((*keys, k), v) for k, v in tree.items()
    }


@dataclass
class PaddedCollatorForLanguageModeling:
    model_max_length: int
    pad_token_id: int
    default_image_resolution: Tuple[int, int, int]
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        self.dummy_pixel_values = torch.zeros(self.default_image_resolution, dtype=self.pixel_values_dtype)

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        pixel_values = [instance["pixel_values"] for instance in instances]

        # For now, we only support Tokenizers with `padding_side = "right"` during Training (but plan to extend!)
        #   => Handle padding via RNN Utils => `pad_sequence`
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        # Truncate (if necessary)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]

        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(self.pad_token_id)

        # === Handle "unimodal" (language-only) vs. "multimodal" ===

        # Some examples are "language-only" --> build a Tensor of `multimodal_indices` that we can slice into easily
        multimodal_indices = torch.tensor(
            [idx for idx in range(len(pixel_values)) if pixel_values[idx] is not None], dtype=torch.long
        )

        # Stack all `pixel_values` --> depending on type (torch.Tensor, or Dict[str, torch.Tensor]) & presence of None
        if len(multimodal_indices) == 0:
            pixel_values = torch.stack([self.dummy_pixel_values for _ in range(len(input_ids))])
        elif isinstance(pv_example := pixel_values[multimodal_indices[0]], torch.Tensor):
            pixel_values = torch.stack(
                [
                    pixel_values[idx] if idx in multimodal_indices else self.dummy_pixel_values
                    for idx in range(len(input_ids))
                ]
            )
        elif isinstance(pv_example, dict):
            pixel_values = {
                k: torch.stack(
                    [
                        pixel_values[idx][k] if idx in multimodal_indices else self.dummy_pixel_values
                        for idx in range(len(input_ids))
                    ]
                )
                for k in pv_example
            }
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        return dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            multimodal_indices=multimodal_indices,
        )


@dataclass
class PaddedCollatorForActionPrediction:
    model_max_length: int
    pad_token_id: int
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        pixel_values = [instance["pixel_values"] for instance in instances]
        if "dataset_name" in instances[0]:
            dataset_names = [instance["dataset_name"] for instance in instances]
        else:
            dataset_names = None

        # For now, we only support Tokenizers with `padding_side = "right"` during training
        #   => Handle padding via RNN Utils => `pad_sequence`
        assert self.padding_side == "right", f"Invalid Tokenizer `{self.padding_side = }`"
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        # Truncate (if necessary)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]

        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(self.pad_token_id)

        # [Contract] For VLA Training =>> No "Unimodal" Data!
        assert all([pv is not None for pv in pixel_values]), "Invalid VLA Example with `pixel_values = None`!"

        # Stack all `pixel_values` --> depending on type is torch.Tensor or Dict[str, torch.Tensor]
        if isinstance(pixel_values[0], torch.Tensor):
            pixel_values = torch.stack(pixel_values)
        elif isinstance(pixel_values[0], dict):
            pixel_values = {
                k: torch.stack([pixel_values[idx][k] for idx in range(len(input_ids))]) for k in pixel_values[0]
            }
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        output = dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        if dataset_names is not None:
            output["dataset_names"] = dataset_names
        return output


@dataclass
class PaddedCollatorForActionPredictionV2:
    model_max_length: int
    pad_token_id: int
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        if "dataset_name" in instances[0]:
            dataset_names = [instance["dataset_name"] for instance in instances]
        else:
            dataset_names = None

        # For now, we only support Tokenizers with `padding_side = "right"` during training
        #   => Handle padding via RNN Utils => `pad_sequence`
        assert self.padding_side == "right", f"Invalid Tokenizer `{self.padding_side = }`"
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        # Truncate (if necessary)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]

        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(self.pad_token_id)

        # # Stack all `pixel_values` --> depending on type is torch.Tensor or Dict[str, torch.Tensor]
        # pixel_values = [instance["pixel_values"] for instance in instances]
        # # [Contract] For VLA Training =>> No "Unimodal" Data!
        # assert all([pv is not None for pv in pixel_values]), "Invalid VLA Example with `pixel_values = None`!"
        # if isinstance(pixel_values[0], torch.Tensor):
        #     pixel_values = torch.stack(pixel_values)
        # elif isinstance(pixel_values[0], dict):
        #     pixel_values = {
        #         k: torch.stack([pixel_values[idx][k] for idx in range(len(input_ids))]) for k in pixel_values[0]
        #     }
        # else:
        #     raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        # Stack all additional `pixel_values` --> depending on type is torch.Tensor or Dict[str, torch.Tensor]
        all_pixel_values = [instance["all_pixel_values"] for instance in instances]
        all_pixel_values = torch.stack(all_pixel_values)
        if instances[0]["all_wrist_values"] is not None:
            all_wrist_values = [instance["all_wrist_values"] for instance in instances]
            all_wrist_values = torch.stack(all_wrist_values)
        else:
            all_wrist_values = None

        output = dict(
            # pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            all_pixel_values=all_pixel_values,
            all_wrist_values=all_wrist_values,
        )
        if dataset_names is not None:
            output["dataset_names"] = dataset_names
        return output


@dataclass
class PaddedCollatorForActionPredictionV3:
    model_max_length: int
    pad_token_id: int
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, windowed_labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        windowed_labels = torch.stack(windowed_labels)
        if "dataset_name" in instances[0]:
            dataset_names = [instance["dataset_name"] for instance in instances]
        else:
            dataset_names = None

        # For now, we only support Tokenizers with `padding_side = "right"` during training
        #   => Handle padding via RNN Utils => `pad_sequence`
        assert self.padding_side == "right", f"Invalid Tokenizer `{self.padding_side = }`"
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)

        # Truncate (if necessary)
        input_ids = input_ids[:, : self.model_max_length]

        # Windowed labels
        _windowed_labels = []
        for idx in range(windowed_labels.shape[1]):
            labels = windowed_labels[:,idx]
            labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
            labels = labels[:, : self.model_max_length]
            _windowed_labels.append(labels)
        windowed_labels = torch.stack(_windowed_labels, dim=1)

        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(self.pad_token_id)

        # # Stack all `pixel_values` --> depending on type is torch.Tensor or Dict[str, torch.Tensor]
        # pixel_values = [instance["pixel_values"] for instance in instances]
        # # [Contract] For VLA Training =>> No "Unimodal" Data!
        # assert all([pv is not None for pv in pixel_values]), "Invalid VLA Example with `pixel_values = None`!"
        # if isinstance(pixel_values[0], torch.Tensor):
        #     pixel_values = torch.stack(pixel_values)
        # elif isinstance(pixel_values[0], dict):
        #     pixel_values = {
        #         k: torch.stack([pixel_values[idx][k] for idx in range(len(input_ids))]) for k in pixel_values[0]
        #     }
        # else:
        #     raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")
        
        # Stack all additional `pixel_values` --> depending on type is torch.Tensor or Dict[str, torch.Tensor]
        all_pixel_values = [instance["all_pixel_values"] for instance in instances]
        all_pixel_values = torch.stack(all_pixel_values)
        if instances[0]["all_wrist_values"] is not None:
            all_wrist_values = [instance["all_wrist_values"] for instance in instances]
            all_wrist_values = torch.stack(all_wrist_values)
        else:
            all_wrist_values = None
        if instances[0]["all_pixel_depth_values"] is not None:
            all_pixel_depth_values = [instance["all_pixel_depth_values"] for instance in instances]
            all_pixel_depth_values = torch.stack(all_pixel_depth_values)
        else:
            all_pixel_depth_values = None
        if instances[0]["all_wrist_depth_values"] is not None:
            all_wrist_depth_values = [instance["all_wrist_depth_values"] for instance in instances]
            all_wrist_depth_values = torch.stack(all_wrist_depth_values)
        else:
            all_wrist_depth_values = None
        if instances[0]["all_pixel_seg_values"] is not None:
            all_pixel_seg_values = [instance["all_pixel_seg_values"] for instance in instances]
            all_pixel_seg_values = torch.stack(all_pixel_seg_values)
        else:
            all_pixel_seg_values = None
        if instances[0]["all_wrist_seg_values"] is not None:
            all_wrist_seg_values = [instance["all_wrist_seg_values"] for instance in instances]
            all_wrist_seg_values = torch.stack(all_wrist_seg_values)
        else:
            all_wrist_seg_values = None

        if "reasoning_on_image" in instances[0]:
            reasoning_on_image = [instance["reasoning_on_image"] for instance in instances]
        else:
            reasoning_on_image = None
        if "reasoning_on_wrist_image" in instances[0]:
            reasoning_on_wrist_image = [instance["reasoning_on_wrist_image"] for instance in instances]
        else:
            reasoning_on_wrist_image = None
        if "lang_nouns_ids" in instances[0]:
            lang_nouns_ids = [instance["lang_nouns_ids"] for instance in instances]
        else:
            lang_nouns_ids = None

        if "task" in instances[0]:
            tasks = [instance["task"] for instance in instances]
        else:
            tasks = None

        output = dict(
            # pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=windowed_labels,
            all_pixel_values=all_pixel_values,
            all_wrist_values=all_wrist_values,
            all_pixel_depth_values=all_pixel_depth_values,
            all_wrist_depth_values=all_wrist_depth_values,
            all_pixel_seg_values=all_pixel_seg_values,
            all_wrist_seg_values=all_wrist_seg_values,
            reasoning_on_image=reasoning_on_image,
            reasoning_on_wrist_image=reasoning_on_wrist_image,
            lang_nouns_ids=lang_nouns_ids,
            tasks=tasks
        )
        if dataset_names is not None:
            output["dataset_names"] = dataset_names
        return output


@dataclass
class PaddedCollatorForActionPredictionV3T:
    model_max_length: int
    pad_token_id: int
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        full_input_ids, full_labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        if "dataset_name" in instances[0]:
            dataset_names = [instance["dataset_name"] for instance in instances]
        else:
            dataset_names = None

        # For now, we only support Tokenizers with `padding_side = "right"` during training
        #   => Handle padding via RNN Utils => `pad_sequence`
        assert self.padding_side == "right", f"Invalid Tokenizer `{self.padding_side = }`"
        input_ids = []
        for datum in full_input_ids:
            input_ids += datum
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)

        labels = []
        for datum in full_labels:
            labels += datum
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        # Truncate (if necessary)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]

        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(self.pad_token_id)

        # # Stack all `pixel_values` --> depending on type is torch.Tensor or Dict[str, torch.Tensor]
        # pixel_values = [instance["pixel_values"] for instance in instances]
        # # [Contract] For VLA Training =>> No "Unimodal" Data!
        # assert all([pv is not None for pv in pixel_values]), "Invalid VLA Example with `pixel_values = None`!"
        # if isinstance(pixel_values[0], torch.Tensor):
        #     pixel_values = torch.stack(pixel_values)
        # elif isinstance(pixel_values[0], dict):
        #     pixel_values = {
        #         k: torch.stack([pixel_values[idx][k] for idx in range(len(input_ids))]) for k in pixel_values[0]
        #     }
        # else:
        #     raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        # Stack all additional `pixel_values` --> depending on type is torch.Tensor or Dict[str, torch.Tensor]
        all_pixel_values = [instance["all_pixel_values"] for instance in instances]
        all_pixel_values = torch.stack(all_pixel_values)
        if instances[0]["all_wrist_values"] is not None:
            all_wrist_values = [instance["all_wrist_values"] for instance in instances]
            all_wrist_values = torch.stack(all_wrist_values)
        else:
            all_wrist_values = None
        if instances[0]["all_pixel_depth_values"] is not None:
            all_pixel_depth_values = [instance["all_pixel_depth_values"] for instance in instances]
            all_pixel_depth_values = torch.stack(all_pixel_depth_values)
        else:
            all_pixel_depth_values = None
        if instances[0]["all_wrist_depth_values"] is not None:
            all_wrist_depth_values = [instance["all_wrist_depth_values"] for instance in instances]
            all_wrist_depth_values = torch.stack(all_wrist_depth_values)
        else:
            all_wrist_depth_values = None

        if "reasoning_on_image" in instances[0]:
            reasoning_on_image = [instance["reasoning_on_image"] for instance in instances]
        else:
            reasoning_on_image = None
        if "reasoning_on_wrist_image" in instances[0]:
            reasoning_on_wrist_image = [instance["reasoning_on_wrist_image"] for instance in instances]
        else:
            reasoning_on_wrist_image = None
        if "lang_nouns_ids" in instances[0]:
            lang_nouns_ids = [instance["lang_nouns_ids"] for instance in instances]
        else:
            lang_nouns_ids = None

        if "task" in instances[0]:
            tasks = [instance["task"] for instance in instances]
        else:
            tasks = None

        output = dict(
            # pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            all_pixel_values=all_pixel_values,
            all_wrist_values=all_wrist_values,
            all_pixel_depth_values=all_pixel_depth_values,
            all_wrist_depth_values=all_wrist_depth_values,
            reasoning_on_image=reasoning_on_image,
            reasoning_on_wrist_image=reasoning_on_wrist_image,
            lang_nouns_ids=lang_nouns_ids,
            tasks=tasks
        )
        if dataset_names is not None:
            output["dataset_names"] = dataset_names
        return output



def preprocess_reasoning_bboxes(reasoning_bboxes, device=None):
    # reasoning_bboxes : List[List[tuple(name, box)]]
    batched_rois = reasoning_bboxes
    revised_reasoning_bboxes = []
    for i in range(len(batched_rois)):
        rois = batched_rois[i]
        rois = [torch.tensor(roi[1]) for roi in rois]
        revised_reasoning_bboxes.append(torch.stack(rois, dim=0).to(device))

    return revised_reasoning_bboxes

def preprocess_reasoning_bboxes_v2(reasoning_bboxes, device=None):
    # reasoning_bboxes : List[List[tuple(name, box)]]
    bz = len(reasoning_bboxes)
    horizon = len(reasoning_bboxes[0])
    
    revised_reasoning_bboxes = []
    for i in range(bz):
        labeled_reasoning_bboxes = {}
        for j in range(horizon):
            for roi in reasoning_bboxes[i][j]:
                if roi[0] not in labeled_reasoning_bboxes:
                    labeled_reasoning_bboxes[roi[0]] = []
                    
        batch_reasoning_keys = labeled_reasoning_bboxes.keys()

        for j in range(horizon):
            reasoning_dict_ij = dict(reasoning_bboxes[i][j])
            for key in batch_reasoning_keys:
                if key in reasoning_dict_ij:
                    labeled_reasoning_bboxes[key].append(
                        torch.tensor(np.concatenate([reasoning_dict_ij[key], [1.0]])).float()
                    )
                else:
                    labeled_reasoning_bboxes[key].append(
                        torch.tensor(np.zeros(5)).float()
                    )
        
        for key in batch_reasoning_keys:
            labeled_reasoning_bboxes[key] = torch.stack(labeled_reasoning_bboxes[key], dim=0).to(device)
        revised_reasoning_bboxes.append(labeled_reasoning_bboxes)

    return revised_reasoning_bboxes


def preprocess_reasoning_bboxes_v3(reasoning_bboxes, lang_nouns, device=None):
    # reasoning_bboxes : List[List[tuple(name, box)]]
    bz = len(reasoning_bboxes)
    horizon = len(reasoning_bboxes[0])
    
    revised_reasoning_bboxes = []
    for i in range(bz):
        labeled_reasoning_bboxes = {}
        for j in range(horizon):
            for roi in reasoning_bboxes[i][j]:
                # if roi[0] not in labeled_reasoning_bboxes:
                #     labeled_reasoning_bboxes[roi[0]] = []
                if roi not in labeled_reasoning_bboxes:
                    labeled_reasoning_bboxes[roi] = []
                    
        batch_reasoning_keys = labeled_reasoning_bboxes.keys()
        active_objs = list(lang_nouns[i])

        for j in range(horizon):
            reasoning_dict_ij = dict(reasoning_bboxes[i][j])
            for key in batch_reasoning_keys:
                if key in reasoning_dict_ij:
                    if key in active_objs:
                        labeled_reasoning_bboxes[key].append(
                            torch.tensor(np.concatenate([reasoning_dict_ij[key], [1.0, 1.0]])).float()
                        )
                    else:
                        labeled_reasoning_bboxes[key].append(
                            torch.tensor(np.concatenate([reasoning_dict_ij[key], [1.0, 0.0]])).float()
                        )
                else:
                    labeled_reasoning_bboxes[key].append(
                        torch.tensor(np.zeros(6)).float()
                    )
        
        for key in batch_reasoning_keys:
            labeled_reasoning_bboxes[key] = torch.stack(labeled_reasoning_bboxes[key], dim=0).to(device)
        revised_reasoning_bboxes.append(labeled_reasoning_bboxes)

    return revised_reasoning_bboxes

# def preprocess_reasoning_bboxes(reasoning_bboxes, lang_nouns=None, device=None):
#     # reasoning_bboxes : List[{object_name: xywh in [0,1],...}]
#     # lang_nouns : List[{object_name: tokenized_name,...}]
#     # english_lang_nouns, revised_reasoning_bboxes, revised_lang_nouns
    
#     english_lang_nouns = {}
#     revised_reasoning_bboxes = {}
#     revised_lang_nouns = {}

#     for data_name in ['robot', 'objects', 'all']:
#         _english_lang_nouns = []
#         _revised_reasoning_bboxes = []
#         _revised_lang_nouns = []

#         for i, bboxes in enumerate(reasoning_bboxes):
#             keys = lang_nouns[i].keys()
#             bbox_list = []
#             tokens_list = []
#             selected_keys = []
#             for key in keys:
#                 if key not in bboxes:
#                     continue
#                 if data_name == 'robot' and key != 'robot':
#                     continue
#                 if data_name == 'objects' and key == 'robot':
#                     continue
                
#                 bbox_list.append(torch.tensor(bboxes[key]))
#                 lang_nouns[i][key][0] = 32000
#                 tokens_list.append(lang_nouns[i][key])
#                 selected_keys.append(key)
#             if len(bbox_list) != 0:
#                 bbox_list = torch.stack(bbox_list, dim=0).to(device)
#             else:
#                 bbox_list = None

#             if len(tokens_list) != 0:
#                 tokens_list = torch.stack(tokens_list, dim=0).to(device)
#             else:
#                 tokens_list = None
            
#             _revised_reasoning_bboxes.append(bbox_list)
#             _revised_lang_nouns.append(tokens_list)
#             _english_lang_nouns.append(selected_keys)
        
#         revised_reasoning_bboxes[data_name] = _revised_reasoning_bboxes
#         revised_lang_nouns[data_name] = _revised_lang_nouns
#         english_lang_nouns[data_name] = _english_lang_nouns
    
#     return english_lang_nouns, revised_reasoning_bboxes, revised_lang_nouns


def split_gripper_object_data(data):
    return data[:,0:1], data[:,1:]

import os
import cv2
from torch.nn import functional as F
def crop_bboxes_with_score(
    image_tensor, # torch.Tensor,
    bboxes, # List[Dict[str: torch.Tensor]] ,
    patch_resize: tuple = (40, 40),
    save_debug: bool = False,
    debug_dir: str = None,
    is_depth: bool = False
) -> torch.Tensor:
    """
    Crops and resizes bounding boxes from a batch of images.

    Args:
        image_tensor (torch.Tensor):
            A batch of horizon of images of shape (B, T, C, H, W).
        bboxes (List[Dict[str: torch.Tensor]]):
            A list of dictionaries of len B, each dict has `key`: label, `value`: (T, Ni, 4), 
            where each bbox in (cx, cy, w, h, score) format [0, 1]. 
            Coordinates are normalized in [0,1]. There are ~max(Ni)*B*T such bboxes.
        patch_resize (tuple):
            The (height, width) to resize each crop.
        save_debug (bool):
            If True, saves each cropped patch for debugging via cv2.
        debug_dir (str):
            Directory (or prefix) to save debug images.

    Returns:
        List[Dict[str: torch.Tensor]]: Cropped patches that correspond with `bboxes`.
    """
    # Unpack shapes
    B, T, C, H, W = image_tensor.shape

    # Prepare an output tensor to hold all patches
    out_patches = []

    # Ensure debug directory exists if we're saving patches
    if save_debug and debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)

    for b_idx in range(B):
        batched_patches = {}
        for key, value in bboxes[b_idx].items():
            # key   is str
            # value is tensor of boxes
            cropped_patches = []

            for t_idx in range(T):
                cx, cy, w, h, obj = value[t_idx][:5]

                # (Optional) If you want to skip or filter by score, do it here:
                # if score < 0.5:  # or your threshold
                #     continue

                # Convert from normalized [0,1] to pixel coordinates
                cx_pix = cx * W
                cy_pix = cy * H
                w_pix  = w  * W
                h_pix  = h  * H

                # Compute integer pixel bounds for the crop
                x1 = int(cx_pix - 0.5 * w_pix)
                y1 = int(cy_pix - 0.5 * h_pix)
                x2 = int(cx_pix + 0.5 * w_pix)
                y2 = int(cy_pix + 0.5 * h_pix)

                # Clamp to image boundaries
                x1 = max(0, min(x1, W))
                x2 = max(0, min(x2, W))
                y1 = max(0, min(y1, H))
                y2 = max(0, min(y2, H))
                if obj == 0 or (y2-y1 == 0 and x2-x1 == 0):
                    cropped_patches.append(-torch.ones((1, patch_resize[0], patch_resize[1])).to(image_tensor.device).float())
                    continue
                    
                # Crop the patch [shape: (C, crop_h, crop_w)]
                patch = image_tensor[b_idx, t_idx, :, y1:y2, x1:x2].unsqueeze(0)

                # Resize to (patch_resize[0], patch_resize[1]) using bilinear interpolation
                patch_resized = F.interpolate(
                    patch,
                    size=patch_resize,
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)  # => shape (C, patch_resize[0], patch_resize[1])
                if is_depth:
                    patch_resized = patch_resized[:1,:,:].float()

                cropped_patches.append(patch_resized)

                if save_debug and debug_dir is not None:
                    # Convert to NumPy for cv2
                    patch_np = patch_resized.permute(1, 2, 0).cpu().numpy()

                    # If your tensor is in [0,1], you may want to scale up to [0,255].
                    patch_np = (patch_np * 255).astype("uint8")
                    
                    # Note: By default, this is in RGB. If you need BGR for OpenCV, do:
                    # patch_np = patch_np[..., ::-1]

                    # Write out the patch
                    out_path = os.path.join(
                        debug_dir, f"patch_b{b_idx}_{key}_t{t_idx}.png"
                    )
                    cv2.imwrite(out_path, patch_np)
            batched_patches[key] = torch.stack(cropped_patches, dim=0)
        out_patches.append(batched_patches)

    # for b_idx in range(B):
    #     print(bboxes[b_idx].keys())
    #     print(out_patches[b_idx].keys())
    # 1/0
    return out_patches

def filter_by_active_labels(reasoning_data, lang_nouns):
    revised_reasoning_data = []
    # print(reasoning_data)
    # print(lang_nouns)
    for bi, data in enumerate(reasoning_data):
        revised_reasoning_data.append({})
        active_objs = list(lang_nouns[bi])
        for key, value in data.items():
            if key in active_objs:
                revised_reasoning_data[bi][key] = value
    # print(revised_reasoning_data)
    return revised_reasoning_data

def _merge_bboxes_util(bbox1, bbox2):
    # Extract bounding box coordinates: cx, cy, w, h
    cx1, cy1, w1, h1, score = bbox1[:,0], bbox1[:,1], bbox1[:,2], bbox1[:,3], bbox1[:,4]
    cx2, cy2, w2, h2, score = bbox2[:,0], bbox2[:,1], bbox2[:,2], bbox2[:,3], bbox2[:,4]

    # Convert bbox1 to corner coordinates (x_min, y_min, x_max, y_max)
    x_min1 = cx1 - w1 / 2
    y_min1 = cy1 - h1 / 2
    x_max1 = cx1 + w1 / 2
    y_max1 = cy1 + h1 / 2

    # Convert bbox2 to corner coordinates (x_min, y_min, x_max, y_max)
    x_min2 = cx2 - w2 / 2
    y_min2 = cy2 - h2 / 2
    x_max2 = cx2 + w2 / 2
    y_max2 = cy2 + h2 / 2

    # Calculate the merged bounding box (min and max)
    x_min_merged = torch.min(x_min1, x_min2)
    y_min_merged = torch.min(y_min1, y_min2)
    x_max_merged = torch.max(x_max1, x_max2)
    y_max_merged = torch.max(y_max1, y_max2)

    # Convert the merged bounding box back to cx, cy, w, h format
    cx_merged = (x_min_merged + x_max_merged) / 2
    cy_merged = (y_min_merged + y_max_merged) / 2
    w_merged = x_max_merged - x_min_merged
    h_merged = y_max_merged - y_min_merged

    if bbox1.shape[1] > 5:
        interactable = bbox1[:,5]
        res = torch.stack([cx_merged, cy_merged, w_merged, h_merged, score, interactable], dim=-1)
    else:
        res = torch.stack([cx_merged, cy_merged, w_merged, h_merged, score], dim=-1)
    return res

def merge_bboxes_to_interaction(reasoning_data):
    revised_reasoning_data = []
    # print(reasoning_data)
    # print(lang_nouns)
    for bi, data in enumerate(reasoning_data):
        revised_reasoning_data.append({})
        for key, value in data.items():
            revised_reasoning_data[bi][key] = _merge_bboxes_util(value, reasoning_data[bi]['robot'])
    # print(revised_reasoning_data)
    return revised_reasoning_data


def merge_bboxes(reasoning_data):
    revised_reasoning_data = []
    # print(reasoning_data)
    # print(lang_nouns)
    for bi, data in enumerate(reasoning_data):
        revised_reasoning_data.append({})
        for key, value in data.items():
            if 'interaction' not in revised_reasoning_data[bi]:
                revised_reasoning_data[bi]['interaction'] = value
            else:
                revised_reasoning_data[bi]['interaction'] = _merge_bboxes_util(revised_reasoning_data[bi]['interaction'], value)
    # print(revised_reasoning_data)
    return revised_reasoning_data
