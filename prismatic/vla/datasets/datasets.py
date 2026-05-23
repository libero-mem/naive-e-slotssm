"""
datasets.py

Lightweight PyTorch Dataset Definition for wrapping RLDS TFDS Pipeline; just defines transform from RLDS default
format to OpenVLA, IterableDataset shim.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Type

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.data_utils import tree_map
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets.rlds import make_interleaved_dataset, make_single_dataset
from prismatic.vla.datasets.rlds import make_interleaved_dataset_v2, make_single_dataset_v2
from prismatic.vla.datasets.rlds import make_interleaved_dataset_v3, make_single_dataset_v3
from prismatic.vla.datasets.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights
from prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

def process_reasoning_cues(reasoning, new_size=224, ori_size=256):

    object_dict = {}
    objects = str(reasoning)[2:-1].split('#')
    for object in objects:
        noun, details = object.split(':')
        # print(noun, len(noun))
        if noun == 'gripper':
            noun = 'robot'
        segid, bbox = details.split(',')
        if '\\n' in bbox:
            bbox = bbox.replace('[', '').replace(']', '').replace('\\n', '').strip()
            bbox = bbox.split(' ')
        else:
            bbox = bbox[1:-1].split(' ')
        bbox = [int(p) for p in bbox if p != '']
        bbox = np.array(bbox)/ori_size
        bbox[:2] += bbox[2:]/2
        object_dict[str(noun)] = bbox
    # print('')
    # 1/0
    return object_dict

def process_reasoning_cues_v2(reasoning, new_size=224, ori_size=256):

    object_dict = {}
    objects = str(reasoning)[2:-1].split('#')
    for object in objects:
        if len(object.split(',')) == 3:
            noun_segid, bbox, itrn_cnt = object.split(',')
            noun, segid = noun_segid.split(':')
        else:
            noun_segid, bbox = object.split(',')
            noun, segid = noun_segid.split(':')
            itrn_cnt = str(0)
        # print(noun, len(noun))
        if noun == 'gripper':
            noun = 'robot'
        if '\\n' in bbox:
            bbox = bbox.replace('[', '').replace(']', '').replace('\\n', '').strip()
            bbox = bbox.split(' ')
        else:
            bbox = bbox[1:-1].split(' ')
        bbox = [int(p) for p in bbox if p != '']
        bbox = np.array(bbox)/ori_size
        bbox[:2] += bbox[2:]/2
        object_dict[str(noun)] = [itrn_cnt, int(segid), bbox]
    # print('')
    # 1/0
    return object_dict

def align_cues_by_name(_reasoning_on_image, _reasoning_on_wrist_image):
    image_keys = set(_reasoning_on_image.keys())
    wrist_keys = set(_reasoning_on_wrist_image.keys())
    elements_in_ab = list(image_keys & wrist_keys)
    elements_in_a_not_b = list(image_keys - wrist_keys)
    elements_in_b_not_a = list(wrist_keys - image_keys)

    reasoning_on_image, reasoning_on_wrist_image = [], []
    for key in elements_in_ab:
        reasoning_on_image.append((key, _reasoning_on_image[key]))
        reasoning_on_wrist_image.append((key, _reasoning_on_wrist_image[key]))
    for key in elements_in_a_not_b:
        reasoning_on_image.append((key, _reasoning_on_image[key]))
    for key in elements_in_b_not_a:
        reasoning_on_wrist_image.append((key, _reasoning_on_wrist_image[key]))

    # print(image_keys); print(wrist_keys)
    # print(elements_in_ab)
    # print(elements_in_a_not_b)
    # print(elements_in_b_not_a)
    # print(reasoning_on_image)
    # print(reasoning_on_wrist_image); 1/0

    return reasoning_on_image, reasoning_on_wrist_image

@dataclass
class RLDSBatchTransform:
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"][0]
        lang = rlds_batch["task"]["language_instruction"].decode().lower()
        if "language_instruction_nouns" in rlds_batch["task"]:
            lang_nouns = rlds_batch["task"]["language_instruction_nouns"].decode().lower().replace('gripper', 'robot')
            lang_nouns = lang_nouns.split('. ')

        # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        # print(prompt_builder.get_prompt())
        # print(input_ids)
        labels = list(input_ids)
        lang_nouns_ids = {}
        if "reasoning_on_image" in rlds_batch["observation"]:
            reasoning_on_image = process_reasoning_cues(rlds_batch["observation"]["reasoning_on_image"][0])
            reasoning_on_wrist_image = process_reasoning_cues(rlds_batch["observation"]["reasoning_on_wrist_image"][0])
            reasoning_on_image, reasoning_on_wrist_image = align_cues_by_name(reasoning_on_image, reasoning_on_wrist_image)
            for noun in lang_nouns:
                noun_ids = self.base_tokenizer(noun).input_ids
                noun_ids = self.base_tokenizer.pad(
                    {"input_ids": noun_ids}, 
                    padding="max_length",  # or "max_length"
                    max_length=30,      # specify if using "max_length"
                    return_tensors="pt"
                ).input_ids
                lang_nouns_ids[noun] = noun_ids
        else:
            reasoning_on_image = {}
            reasoning_on_wrist_image = {}

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)

        print("prompt", prompt_builder.get_prompt())
        print("input ids", input_ids)

        all_pixel_values = []
        for img_data in rlds_batch["observation"]["image_primary"]:
            img_data = img_data[:,:,:]
            # import cv2
            # cv2.imwrite('./tmp_img_training.png', img_data) #; 1/0

            img = Image.fromarray(img_data)
            # img.save("img.png")
            pixel_values = self.image_transform(img)
            all_pixel_values.append(pixel_values)
        all_pixel_values = torch.stack(all_pixel_values)

        all_wrist_values = None
        if "image_wrist" in rlds_batch["observation"]:
            all_wrist_values = []
            for img_data in rlds_batch["observation"]["image_wrist"]:
                img_data = img_data[:,:,:]
                # import cv2
                # cv2.imwrite('./tmp_wrist_img_training.png', img_data); 1/0

                wrist_img = Image.fromarray(img_data)
                # wrist_img.save("img_wrist_img.png")

                wrist_values = self.image_transform(wrist_img)
                all_wrist_values.append(wrist_values)
            all_wrist_values = torch.stack(all_wrist_values)

        all_pixel_depth_values = None
        if "depth_primary" in rlds_batch["observation"]:
            all_pixel_depth_values = []
            for img_data in rlds_batch["observation"]["depth_primary"]:
                img_data = img_data[:,:,:]
                img_data = np.concatenate([img_data, img_data, img_data], axis=-1)
                img_data = Image.fromarray(img_data)
                # img_data.save("img_data_depth.png")

                pixel_values = self.image_transform(img_data)
                all_pixel_depth_values.append(pixel_values)
            all_pixel_depth_values = torch.stack(all_pixel_depth_values)

        all_wrist_depth_values = None
        if "depth_wrist" in rlds_batch["observation"]:
            all_wrist_depth_values = []
            for img_data in rlds_batch["observation"]["depth_wrist"]:
                img_data = img_data[:,:,:]
                img_data = np.concatenate([img_data, img_data, img_data], axis=-1)
                img_data = Image.fromarray(img_data)
                # img_data.save("img_wrist_data_depth.png"); 1/0

                wrist_values = self.image_transform(img_data)
                all_wrist_depth_values.append(wrist_values)
            all_wrist_depth_values = torch.stack(all_wrist_depth_values)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(len(action) + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        # print(lang_nouns)
        # print(rlds_batch["observation"]["reasoning_on_image"])
        # print(rlds_batch["observation"]["reasoning_on_wrist_image"])
        # 1/0

        return dict(pixel_values=all_pixel_values[0], input_ids=input_ids, labels=labels, dataset_name=dataset_name,
                    all_pixel_values=all_pixel_values, all_wrist_values=all_wrist_values,
                    all_pixel_depth_values=all_pixel_depth_values, all_wrist_depth_values=all_wrist_depth_values,
                    reasoning_on_image=reasoning_on_image, reasoning_on_wrist_image=reasoning_on_wrist_image,
                    lang_nouns_ids=lang_nouns_ids)



class RLDSDataset(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: RLDSBatchTransform,
        resize_resolution: Tuple[int, int],
        shuffle_buffer_size: int = 256_000,
        train: bool = True,
        image_aug: bool = False,
        window_size: int = 1,
        load_camera_views: Tuple[str] = ("primary",),
        load_depth: bool = False,
        cropping = True
    ) -> None:
        """Lightweight wrapper around RLDS TFDS Pipeline for use with PyTorch/OpenVLA Data Loaders."""
        self.data_root_dir, self.data_mix, self.batch_transform = data_root_dir, data_mix, batch_transform

        # Configure RLDS Dataset(s)
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
            mixture_spec = [(self.data_mix, 1.0)]

        # fmt: off
        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=load_camera_views,
            load_depth=load_depth,
            load_proprio=False,
            load_language=True,
            action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,
        )
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=window_size,                                      # If we wanted to feed / predict more than one step
                future_action_window_size=0,                        # For action chunking
                skip_unlabeled=True,                                # Skip trajectories without language labels
                goal_relabeling_strategy="uniform",                 # Goals are currently unused
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,
                depth_resize_size=resize_resolution,
                num_parallel_calls=16,                          # For CPU-intensive ops (decoding, resizing, etc.)
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
        )

        # If applicable, enable image augmentations
        if image_aug:
            if cropping:
                rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : dict(
                    random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                    random_brightness=[0.2],
                    random_contrast=[0.8, 1.2],
                    random_saturation=[0.8, 1.2],
                    random_hue=[0.05],
                    augment_order=[
                        "random_resized_crop",
                        "random_brightness",
                        "random_contrast",
                        "random_saturation",
                        "random_hue",
                    ],
                )}),
            else:
                rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : dict(
                    random_brightness=[0.2],
                    random_contrast=[0.8, 1.2],
                    random_saturation=[0.8, 1.2],
                    random_hue=[0.05],
                    augment_order=[
                        "random_brightness",
                        "random_contrast",
                        "random_saturation",
                        "random_hue",
                    ],
                )}),

        # fmt: on

        # Initialize RLDS Dataset
        if load_depth:
            self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config, multimodal=True)
        else:
            self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

    def make_dataset(self, rlds_config, multimodal=False):
        if multimodal:
            return make_interleaved_dataset_v2(**rlds_config)
        return make_interleaved_dataset(**rlds_config)

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            yield self.batch_transform(rlds_batch)

    def __len__(self) -> int:
        return self.dataset_length

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")


import dlimp as dl

class MyStreamingLoader:
    def __init__(self, dataset: dl.DLataset, batch_size: int, batch_transform=None, collate_fn=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.batch_transform = batch_transform or (lambda x: x)
        self.collate_fn = collate_fn
        self.iterator = dataset.iterator()
        self.slots = [None] * batch_size
        self.indices = [0] * batch_size

        # Preload initial trajectories
        for i in range(batch_size):
            self._load_new_traj(i)

    def _load_new_traj(self, i: int):
        try:
            traj = next(self.iterator)
            self.slots[i] = traj
            self.indices[i] = 0
        except StopIteration:
            self.slots[i] = None
            self.indices[i] = -1

    def __iter__(self):
        return self

    def __next__(self):
        batch_obs, batch_act, batch_task = [], [], []

        for i in range(self.batch_size):
            traj = self.slots[i]
            idx = self.indices[i]

            if traj is None or idx >= traj["action"].shape[0]:
                self._load_new_traj(i)
                traj = self.slots[i]
                idx = self.indices[i]

            if traj is None:
                raise StopIteration

            obs = tf.nest.map_structure(lambda x: x[idx], traj["observation"])
            act = traj["action"][idx]
            task = traj["task"]

            batch_obs.append(obs)
            batch_act.append(act)
            batch_task.append(task)

            self.indices[i] += 1

        batch = {
            "observation": tf.nest.map_structure(lambda *x: np.stack(x), *batch_obs),
            "action": np.stack(batch_act),
            "task": tf.nest.map_structure(lambda *x: np.stack(x), *batch_task),
        }

        return self.batch_transform(batch)


class RLDSDatasetV3(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: RLDSBatchTransform,
        resize_resolution: Tuple[int, int],
        shuffle_buffer_size: int = 256_000,
        train: bool = True,
        image_aug: bool = False,
        window_size: int = 1,
        load_camera_views: Tuple[str] = ("primary",),
        load_depth: bool = False,
        cropping = True
    ) -> None:
        """Lightweight wrapper around RLDS TFDS Pipeline for use with PyTorch/OpenVLA Data Loaders."""
        self.data_root_dir, self.data_mix, self.batch_transform = data_root_dir, data_mix, batch_transform

        # Configure RLDS Dataset(s)
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
            mixture_spec = [(self.data_mix, 1.0)]

        # fmt: off
        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=load_camera_views,
            load_depth=load_depth,
            load_proprio=False,
            load_language=True,
            action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,
        )
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=window_size,                                      # If we wanted to feed / predict more than one step
                future_action_window_size=0,                        # For action chunking
                skip_unlabeled=True,                                # Skip trajectories without language labels
                goal_relabeling_strategy="uniform",                 # Goals are currently unused
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,
                depth_resize_size=resize_resolution,
                num_parallel_calls=16,                          # For CPU-intensive ops (decoding, resizing, etc.)
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
        )

        # If applicable, enable image augmentations
        if image_aug:
            if cropping:
                rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : dict(
                    random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                    random_brightness=[0.2],
                    random_contrast=[0.8, 1.2],
                    random_saturation=[0.8, 1.2],
                    random_hue=[0.05],
                    augment_order=[
                        "random_resized_crop",
                        "random_brightness",
                        "random_contrast",
                        "random_saturation",
                        "random_hue",
                    ],
                )}),
            else:
                rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : dict(
                    random_brightness=[0.2],
                    random_contrast=[0.8, 1.2],
                    random_saturation=[0.8, 1.2],
                    random_hue=[0.05],
                    augment_order=[
                        "random_brightness",
                        "random_contrast",
                        "random_saturation",
                        "random_hue",
                    ],
                )}),

        # fmt: on

        # Initialize RLDS Dataset
        if load_depth:
            self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config, multimodal=True)
        else:
            self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

    def make_dataset(self, rlds_config, multimodal=False):
        # if multimodal:
        #     return make_interleaved_dataset_v2(**rlds_config)
        return make_interleaved_dataset_v3(**rlds_config)

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            yield self.batch_transform(rlds_batch)

    def __len__(self) -> int:
        return self.dataset_length

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")


class RLDSDatasetV3_1(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: RLDSBatchTransform,
        resize_resolution: Tuple[int, int],
        shuffle_buffer_size: int = 256_000,
        train: bool = True,
        image_aug: bool = False,
        window_size: int = 1,
        future_action_window_size: int = 5,
        load_camera_views: Tuple[str] = ("primary",),
        load_depth: bool = False,
        cropping = True
    ) -> None:
        """Lightweight wrapper around RLDS TFDS Pipeline for use with PyTorch/OpenVLA Data Loaders."""
        self.data_root_dir, self.data_mix, self.batch_transform = data_root_dir, data_mix, batch_transform

        # Configure RLDS Dataset(s)
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
            mixture_spec = [(self.data_mix, 1.0)]

        # fmt: off
        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=load_camera_views,
            load_depth=load_depth,
            load_proprio=False,
            load_language=True,
            action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,
        )
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=window_size,                                      # If we wanted to feed / predict more than one step
                future_action_window_size=future_action_window_size,                        # For action chunking
                skip_unlabeled=True,                                # Skip trajectories without language labels
                goal_relabeling_strategy="uniform",                 # Goals are currently unused
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,
                depth_resize_size=resize_resolution,
                num_parallel_calls=16,                          # For CPU-intensive ops (decoding, resizing, etc.)
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
        )

        # If applicable, enable image augmentations
        if image_aug:
            if cropping:
                rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : dict(
                    random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                    random_brightness=[0.2],
                    random_contrast=[0.8, 1.2],
                    random_saturation=[0.8, 1.2],
                    random_hue=[0.05],
                    augment_order=[
                        "random_resized_crop",
                        "random_brightness",
                        "random_contrast",
                        "random_saturation",
                        "random_hue",
                    ],
                )}),
            else:
                rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : dict(
                    random_brightness=[0.2],
                    random_contrast=[0.8, 1.2],
                    random_saturation=[0.8, 1.2],
                    random_hue=[0.05],
                    augment_order=[
                        "random_brightness",
                        "random_contrast",
                        "random_saturation",
                        "random_hue",
                    ],
                )}),

        # fmt: on

        # Initialize RLDS Dataset
        if load_depth:
            self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config, multimodal=True)
        else:
            self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

    def make_dataset(self, rlds_config, multimodal=False):
        # if multimodal:
        #     return make_interleaved_dataset_v2(**rlds_config)
        return make_interleaved_dataset_v3(**rlds_config)

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            yield self.batch_transform(rlds_batch)

    def __len__(self) -> int:
        return self.dataset_length

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")


class EpisodicRLDSDataset(RLDSDataset):
    """Returns full episodes as list of steps instead of individual transitions (useful for visualizations)."""

    def make_dataset(self, rlds_config):
        per_dataset_kwargs = rlds_config["dataset_kwargs_list"]
        assert len(per_dataset_kwargs) == 1, "Only support single-dataset `mixes` for episodic datasets."

        return make_single_dataset(
            per_dataset_kwargs[0],
            train=rlds_config["train"],
            traj_transform_kwargs=rlds_config["traj_transform_kwargs"],
            frame_transform_kwargs=rlds_config["frame_transform_kwargs"],
        )

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            out = [
                self.batch_transform(tree_map(lambda x: x[i], rlds_batch))  # noqa: B023
                for i in range(rlds_batch["action"].shape[0])
            ]
            yield out


class DummyDataset(Dataset):
    def __init__(
        self,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
    ) -> None:
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn

        # Note =>> We expect the dataset to store statistics for action de-normalization. Specifically, we store the
        # per-dimension 1st and 99th action quantile. The values below correspond to "no normalization" for simplicity.
        self.dataset_statistics = {
            "dummy_dataset": {
                "action": {"q01": np.zeros((7,), dtype=np.float32), "q99": np.ones((7,), dtype=np.float32)}
            }
        }

    def __len__(self):
        # TODO =>> Replace with number of elements in your dataset!
        return 10000

    def __getitem__(self, idx):
        # TODO =>> Load image, action and instruction from disk -- we use dummy values
        image = Image.fromarray(np.asarray(np.random.rand(224, 224, 3) * 255.0, dtype=np.uint8))
        action = np.asarray(np.random.rand(7), dtype=np.float32)
        instruction = "do something spectacular"

        # Add instruction to VLA prompt
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {instruction}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF .forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(image)
        print("prompt", prompt_builder.get_prompt())
        print("input ids", input_ids)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(len(action) + 1)] = IGNORE_INDEX

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)


@dataclass
class RLDSBatchTransformV2:
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"][0]
        lang = rlds_batch["task"]["language_instruction"].decode().lower()
        language_instruction_nouns = ['none']
        if "language_instruction_nouns" in rlds_batch["task"]:
            language_instruction_nouns = rlds_batch["task"]["language_instruction_nouns"].decode().lower().replace('gripper', 'robot')
            language_instruction_nouns = list(language_instruction_nouns.split('. '))

        # Label tokens.        
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])
        # Tokenize (w/ `base_tokenizer`)
        labels = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(labels)
        # print(prompt_builder.get_prompt())
        # print(labels)

        # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            {"from": "gpt", "value": "_ _ _ _ _ _ _ _"},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])
        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        input_ids[-9] = 29871
        # print(prompt_builder.get_prompt())
        # print(input_ids)

        lang_nouns_ids = {}
        if "reasoning_on_image" in rlds_batch["observation"]:
            reasoning_on_image = process_reasoning_cues(rlds_batch["observation"]["reasoning_on_image"][0])
            reasoning_on_wrist_image = process_reasoning_cues(rlds_batch["observation"]["reasoning_on_wrist_image"][0])
            lang_nouns = list(set(list(reasoning_on_image.keys()) + list(reasoning_on_wrist_image.keys()) + language_instruction_nouns))
            for noun in lang_nouns:
                noun_ids = self.base_tokenizer(noun).input_ids
                noun_ids = self.base_tokenizer.pad(
                    {"input_ids": noun_ids}, 
                    padding="max_length",  # or "max_length"
                    max_length=30,      # specify if using "max_length"
                    return_tensors="pt"
                ).input_ids
                lang_nouns_ids[noun] = noun_ids
            reasoning_on_image, reasoning_on_wrist_image = align_cues_by_name(reasoning_on_image, reasoning_on_wrist_image)
        else:
            reasoning_on_image = []
            reasoning_on_wrist_image = []

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        if input_ids.shape != labels.shape:
            print(input_ids)
            print(labels)

        all_pixel_values = []
        for img_data in rlds_batch["observation"]["image_primary"]:
            img_data = img_data[:,:,:]
            # import cv2
            # cv2.imwrite('./tmp_img_training.png', img_data) #; 1/0

            img = Image.fromarray(img_data)
            # img.save("img.png")
            pixel_values = self.image_transform(img)
            all_pixel_values.append(pixel_values)
        all_pixel_values = torch.stack(all_pixel_values)

        all_wrist_values = None
        if "image_wrist" in rlds_batch["observation"]:
            all_wrist_values = []
            for img_data in rlds_batch["observation"]["image_wrist"]:
                img_data = img_data[:,:,:]
                # import cv2
                # cv2.imwrite('./tmp_wrist_img_training.png', img_data); 1/0

                wrist_img = Image.fromarray(img_data)
                # wrist_img.save("img_wrist_img.png")

                wrist_values = self.image_transform(wrist_img)
                all_wrist_values.append(wrist_values)
            all_wrist_values = torch.stack(all_wrist_values)

        all_pixel_depth_values = None
        if "depth_primary" in rlds_batch["observation"]:
            all_pixel_depth_values = []
            for img_data in rlds_batch["observation"]["depth_primary"]:
                img_data = img_data[:,:,:]
                img_data = np.concatenate([img_data, img_data, img_data], axis=-1)
                # import cv2
                # cv2.imwrite('./tmp_depth_img_training.png', img_data); 1/0

                img_data = Image.fromarray(img_data)
                # img_data.save("img_data_depth.png")

                pixel_values = self.image_transform(img_data)
                all_pixel_depth_values.append(pixel_values)
            all_pixel_depth_values = torch.stack(all_pixel_depth_values)

        all_wrist_depth_values = None
        if "depth_wrist" in rlds_batch["observation"]:
            all_wrist_depth_values = []
            for img_data in rlds_batch["observation"]["depth_wrist"]:
                img_data = img_data[:,:,:]
                img_data = np.concatenate([img_data, img_data, img_data], axis=-1)
                img_data = Image.fromarray(img_data)
                # img_data.save("img_wrist_data_depth.png"); 1/0

                wrist_values = self.image_transform(img_data)
                all_wrist_depth_values.append(wrist_values)
            all_wrist_depth_values = torch.stack(all_wrist_depth_values)
        
        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(len(action) + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        # print(lang_nouns)
        # print(rlds_batch["observation"]["reasoning_on_image"])
        # print(rlds_batch["observation"]["reasoning_on_wrist_image"])
        # 1/0

        task = {
            'instruction': prompt_builder.get_prompt(),
            'instruction_nouns': language_instruction_nouns
        }

        return dict(pixel_values=all_pixel_values[0], input_ids=input_ids, labels=labels, dataset_name=dataset_name,
                    all_pixel_values=all_pixel_values, all_wrist_values=all_wrist_values,
                    all_pixel_depth_values=all_pixel_depth_values, all_wrist_depth_values=all_wrist_depth_values,
                    reasoning_on_image=reasoning_on_image, reasoning_on_wrist_image=reasoning_on_wrist_image,
                    lang_nouns_ids=lang_nouns_ids, task=task)


@dataclass
class RLDSBatchTransformV2T:
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        dataset_name = rlds_batch["dataset_name"]
        lang = rlds_batch["task"]["language_instruction"].decode().lower()
        if "language_instruction_nouns" in rlds_batch["task"]:
            language_instruction_nouns = rlds_batch["task"]["language_instruction_nouns"].decode().lower().replace('gripper', 'robot')
            language_instruction_nouns = list(language_instruction_nouns.split('. '))

        full_labels = []
        full_input_ids = []
        # print(len(rlds_batch["action"]))
        # print(rlds_batch["action"])
        for t in range(len(rlds_batch["action"])):
            action = rlds_batch["action"][t]
            # Label tokens.        
            prompt_builder = self.prompt_builder_fn("openvla")
            conversation = [
                {"from": "human", "value": f"What action should the robot take to {lang}?"},
                {"from": "gpt", "value": self.action_tokenizer(action)},
            ]
            for turn in conversation:
                prompt_builder.add_turn(turn["from"], turn["value"])
            # Tokenize (w/ `base_tokenizer`)
            labels = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
            labels = list(labels)
            # print(prompt_builder.get_prompt())
            # print(labels)

            # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
            prompt_builder = self.prompt_builder_fn("openvla")
            conversation = [
                {"from": "human", "value": f"What action should the robot take to {lang}?"},
                {"from": "gpt", "value": "_ _ _ _ _ _ _ _"},
            ]
            for turn in conversation:
                prompt_builder.add_turn(turn["from"], turn["value"])
            # Tokenize (w/ `base_tokenizer`)
            input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
            input_ids[-9] = 29871
            # print(prompt_builder.get_prompt())
            # print(input_ids)

            # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
            #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
            input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
            if input_ids.shape != labels.shape:
                print(input_ids)
                print(labels)
                1/0
            
            # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
            labels[: -(len(action) + 1)] = IGNORE_INDEX
            if not self.predict_stop_token:
                labels[-1] = IGNORE_INDEX

            full_labels.append(labels)
            full_input_ids.append(input_ids)

        full_labels = full_labels
        full_input_ids = full_input_ids

        lang_nouns_ids = {}
        if "reasoning_on_image" in rlds_batch["observation"]:
            reasoning_on_image = process_reasoning_cues(rlds_batch["observation"]["reasoning_on_image"][0])
            reasoning_on_wrist_image = process_reasoning_cues(rlds_batch["observation"]["reasoning_on_wrist_image"][0])
            lang_nouns = list(set(list(reasoning_on_image.keys()) + list(reasoning_on_wrist_image.keys()) + language_instruction_nouns))
            for noun in lang_nouns:
                noun_ids = self.base_tokenizer(noun).input_ids
                noun_ids = self.base_tokenizer.pad(
                    {"input_ids": noun_ids}, 
                    padding="max_length",  # or "max_length"
                    max_length=30,      # specify if using "max_length"
                    return_tensors="pt"
                ).input_ids
                lang_nouns_ids[noun] = noun_ids
            reasoning_on_image, reasoning_on_wrist_image = align_cues_by_name(reasoning_on_image, reasoning_on_wrist_image)
        else:
            reasoning_on_image = []
            reasoning_on_wrist_image = []

        all_pixel_values = []
        for img_data in rlds_batch["observation"]["image_primary"]:
            img_data = img_data[:,:,:]
            # import cv2
            # cv2.imwrite('./tmp_img_training.png', img_data) #; 1/0

            img = Image.fromarray(img_data)
            # img.save("img.png")
            pixel_values = self.image_transform(img)
            all_pixel_values.append(pixel_values)
        all_pixel_values = torch.stack(all_pixel_values)

        all_wrist_values = None
        if "image_wrist" in rlds_batch["observation"]:
            all_wrist_values = []
            for img_data in rlds_batch["observation"]["image_wrist"]:
                img_data = img_data[:,:,:]
                # import cv2
                # cv2.imwrite('./tmp_wrist_img_training.png', img_data); 1/0

                wrist_img = Image.fromarray(img_data)
                # wrist_img.save("img_wrist_img.png")

                wrist_values = self.image_transform(wrist_img)
                all_wrist_values.append(wrist_values)
            all_wrist_values = torch.stack(all_wrist_values)

        all_pixel_depth_values = None
        if "depth_primary" in rlds_batch["observation"]:
            all_pixel_depth_values = []
            for img_data in rlds_batch["observation"]["depth_primary"]:
                img_data = img_data[:,:,:]
                img_data = np.concatenate([img_data, img_data, img_data], axis=-1)
                # import cv2
                # cv2.imwrite('./tmp_depth_img_training.png', img_data); 1/0

                img_data = Image.fromarray(img_data)
                # img_data.save("img_data_depth.png")

                pixel_values = self.image_transform(img_data)
                all_pixel_depth_values.append(pixel_values)
            all_pixel_depth_values = torch.stack(all_pixel_depth_values)

        all_wrist_depth_values = None
        if "depth_wrist" in rlds_batch["observation"]:
            all_wrist_depth_values = []
            for img_data in rlds_batch["observation"]["depth_wrist"]:
                img_data = img_data[:,:,:]
                img_data = np.concatenate([img_data, img_data, img_data], axis=-1)
                img_data = Image.fromarray(img_data)
                # img_data.save("img_wrist_data_depth.png"); 1/0

                wrist_values = self.image_transform(img_data)
                all_wrist_depth_values.append(wrist_values)
            all_wrist_depth_values = torch.stack(all_wrist_depth_values)
        
        # print(lang_nouns)
        # print(rlds_batch["observation"]["reasoning_on_image"])
        # print(rlds_batch["observation"]["reasoning_on_wrist_image"])
        # 1/0

        task = {
            'instruction': prompt_builder.get_prompt(),
            'instruction_nouns': language_instruction_nouns
        }

        return dict(pixel_values=all_pixel_values[0], input_ids=full_input_ids, labels=full_labels, dataset_name=dataset_name,
                    all_pixel_values=all_pixel_values, all_wrist_values=all_wrist_values,
                    all_pixel_depth_values=all_pixel_depth_values, all_wrist_depth_values=all_wrist_depth_values,
                    reasoning_on_image=reasoning_on_image, reasoning_on_wrist_image=reasoning_on_wrist_image,
                    lang_nouns_ids=lang_nouns_ids, task=task)


LANGUAGE_INSTRUCTION_NOUNS = {
    "put the carrot on the white plate": ['robot', 'carrot', 'plate'],
    "open the middle drawer of the cabinet": ['robot', 'middle drawer', 'cabinet'],
    "put the bowl on the stove": ['robot', 'bowl', 'stove'],
    "put the wine bottle on top of the cabinet": ['robot', 'bottle', 'cabinet'],
    "open the top drawer and put the bowl inside": ['robot', 'top drawer', 'bowl'],
    "put the bowl on top of the cabinet": ['robot', 'bowl', 'cabinet'],
    "push the plate to the front of the stove": ['robot', 'plate', 'stove'],
    "put the cream cheese in the bowl": ['robot', 'cream cheese', 'bowl'],
    "turn on the stove": ['robot', 'stove'],
    "put the bowl on the plate": ['robot', 'bowl', 'plate'],
    "put the wine bottle on the rack": ['robot', 'bottle', 'rack'],
    "swap the 2 bowls using the intermediary plate": ['robot', 'bowl', 'plate'],
    "swap the 3 bowls from left to right using the intermediary plate": ['robot', 'bowl', 'plate'],
    "pick the bowl from the plate and place it back 1 time": ['robot', 'bowl', 'plate'],
    "pick the bowl from the plate and place it back 3 times": ['robot', 'bowl', 'plate'],
    "pick the bowl from the plate and place it back 5 times": ['robot', 'bowl', 'plate'],
    "pick the bowl from the plate and place it back 7 times": ['robot', 'bowl', 'plate'],

    "pick up the black bowl between the plate and the ramekin and place it on the plate": ['robot', 'bowl', 'plate', 'ramekin'],
    "pick up the black bowl from table center and place it on the plate": ['robot', 'bowl', 'plate'],
    "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate": ['robot', 'bowl', 'cabinet', 'plate'],
    "pick up the black bowl next to the cookie box and place it on the plate": ['robot', 'bowl', 'cookies', 'plate'],
    "pick up the black bowl next to the plate and place it on the plate": ['robot', 'bowl', 'plate'],
    "pick up the black bowl next to the ramekin and place it on the plate": ['robot', 'bowl', 'ramekin', 'plate'],
    "pick up the black bowl on the cookie box and place it on the plate": ['robot', 'bowl', 'cookies', 'plate'],
    "pick up the black bowl on the ramekin and place it on the plate": ['robot', 'bowl', 'ramekin', 'plate'],
    "pick up the black bowl on the stove and place it on the plate": ['robot', 'bowl', 'stove', 'plate'],
    "pick up the black bowl on the wooden cabinet and place it on the plate": ['robot', 'bowl', 'cabinet', 'plate'],

    "turn on the stove and put the moka pot on it": ['robot', 'stove', 'moka pot'],
    "put the black bowl in the bottom drawer of the cabinet and close it": ['robot', 'bowl', 'cabinet'],
    "put the yellow and white mug in the microwave and close it": ['robot', 'yellow and white mug', 'microwave'],
    "put both moka pots on the stove": ['robot', 'moka pot', 'stove'],
    "put both the alphabet soup and the cream cheese box in the basket": ['robot', 'alphabet soup', 'cream cheese'],
    "put both the alphabet soup and the tomato sauce in the basket": ['robot', 'alphabet soup', 'tomato'],
    "put both the cream cheese box and the butter in the basket": ['robot', 'cream cheese', 'butter', 'basket'],
    "put the white mug on the left plate and put the yellow and white mug on the right plate": ['robot', 'white mug', 'plate', 'yellow and white mug'],
    "put the white mug on the plate and put the chocolate pudding to the right of the plate": ['robot', 'white mug', 'plate', 'chocolate pudding'],
    "pick up the book and place it in the back compartment of the caddy": ['robot', 'book', 'caddy'],

    "pick up the bowl and place it back on the plate": ['robot', 'pick up', 'bowl', 'place back on', 'plate'],
    "lift the bottle and put it down on the plate": ['robot', 'lift', 'bottle', 'put down on', 'plate'],
    "lift the bowl and place it back on the plate 3 times": ['robot', 'lift', 'bowl', 'place back on', 'plate', 'three times'],
    "pick up the bottle and put it down the plate 3 times": ['robot', 'pick up', 'bottle', 'put down', 'plate', 'three times'],
    "lift the bowl and place it back on the plate 5 times": ['robot', 'lift', 'bowl', 'place back on', 'plate', 'five times'],
    "pick up the bowl and place it on the plate 7 times": ['robot', 'pick up', 'bowl', 'place on', 'plate', 'seven times'],
    "swap the 2 bowls on their plates using the empty plate": ['robot', 'swap', '2 bowls', 'on their plates', 'using', 'empty plate'],
    "rotate the 3 bowls on their plates from left to right using the empty plate": ['robot', 'rotate', '3 bowls', 'on their plates', 'from left to right', 'using', 'empty plate'],
    "put the cream cheese in the nearest basket and place that basket in the center": ['robot', 'put', 'cream cheese', 'in', 'nearest basket', 'place', 'that basket', 'in the center'],
    "put the cream cheese in the nearest basket and place the empty basket in the center": ['robot', 'put', 'cream cheese', 'in', 'nearest basket', 'place', 'empty basket', 'in the center'],

    "pick up the butter and place it in the basket": ['robot', 'pick up', 'butter', 'place', 'in', 'basket'],
    "pick up the bbq sauce and place it in the basket": ['robot', 'pick up', 'bbq sauce', 'place', 'in', 'basket'],
    "pick up the cream cheese and place it in the basket": ['robot', 'pick up', 'cream cheese', 'place', 'in', 'basket'],
    "pick up the salad dressing and place it in the basket": ['robot', 'pick up', 'salad dressing', 'place', 'in', 'basket'],
    "pick up the orange juice and place it in the basket": ['robot', 'pick up', 'orange juice', 'place', 'in', 'basket'],
    "pick up the alphabet soup and place it in the basket": ['robot', 'pick up', 'alphabet soup', 'place', 'in', 'basket'],
    "pick up the tomato sauce and place it in the basket": ['robot', 'pick up', 'tomato sauce', 'place', 'in', 'basket'],
    "pick up the ketchup and place it in the basket": ['robot', 'pick up', 'ketchup', 'place', 'in', 'basket'],
    "pick up the chocolate pudding and place it in the basket": ['robot', 'pick up', 'chocolate pudding', 'place', 'in', 'basket'],
    "pick up the milk and place it in the basket": ['robot', 'pick up', 'milk', 'place', 'in', 'basket'],
}

import cv2

@dataclass
class RLDSBatchTransformV3:
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        horizon = len(rlds_batch["action"])
        dataset_name = rlds_batch["dataset_name"]
        lang = rlds_batch["task"]["language_instruction"].decode().lower()
        if "language_instruction_nouns" in rlds_batch["task"] and lang in LANGUAGE_INSTRUCTION_NOUNS:
            language_instruction_nouns = LANGUAGE_INSTRUCTION_NOUNS[lang]
        else:
            language_instruction_nouns = ['robot']

        windowed_labels = []
        for hidx in range(horizon):
            action = rlds_batch["action"][hidx]
            # Label tokens.        
            prompt_builder = self.prompt_builder_fn("openvla")
            conversation = [
                {"from": "human", "value": f"What action should the robot take?"}, #  to {lang}  , we are relying on CLIP's embeddings for true actions
                {"from": "gpt", "value": self.action_tokenizer(action)},
            ]
            for turn in conversation:
                prompt_builder.add_turn(turn["from"], turn["value"])
            # Tokenize (w/ `base_tokenizer`)
            labels = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
            labels = list(labels)
            windowed_labels.append(labels)
            # print(prompt_builder.get_prompt())
            # print(labels)
            
        # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take?"}, #  to {lang}  , we are relying on CLIP's embeddings for true actions
            {"from": "gpt", "value": "_ _ _ _ _ _ _ _"},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])
        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        input_ids[-9] = 29871
        # print(prompt_builder.get_prompt())
        # print(input_ids)

        # here we get nouns, bboxes of interest across `horizon`
        lang_nouns_ids = {}
        reasoning_on_image = []
        reasoning_on_wrist_image = []
        if "reasoning_on_image" in rlds_batch["observation"]:
            _lang_nouns = list(language_instruction_nouns)
            for hidx in range(len(rlds_batch["observation"]["reasoning_on_image"])):
                _reasoning_on_image = process_reasoning_cues_v2(rlds_batch["observation"]["reasoning_on_image"][hidx])
                if 'reasoning_on_wrist_image' in rlds_batch["observation"]:
                    _reasoning_on_wrist_image = process_reasoning_cues_v2(rlds_batch["observation"]["reasoning_on_wrist_image"][hidx])
                    # _lang_nouns = list(set(list(_reasoning_on_image.keys()) + list(_reasoning_on_wrist_image.keys()) + _lang_nouns))
                    # _reasoning_on_image, _reasoning_on_wrist_image = align_cues_by_name(_reasoning_on_image, _reasoning_on_wrist_image)
                    reasoning_on_wrist_image.append(_reasoning_on_wrist_image)
                reasoning_on_image.append(_reasoning_on_image)

            for noun in _lang_nouns:
                noun_ids = self.base_tokenizer(noun).input_ids
                noun_ids = self.base_tokenizer.pad(
                    {"input_ids": noun_ids}, 
                    padding="max_length",  # or "max_length"
                    max_length=30,      # specify if using "max_length"
                    return_tensors="pt"
                ).input_ids
                lang_nouns_ids[noun] = noun_ids

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, windowed_labels = torch.tensor(input_ids), torch.tensor(windowed_labels)
        if input_ids.shape != windowed_labels[0].shape:
            print('input_ids', input_ids.shape)
            print('windowed_labels', windowed_labels.shape)
            1/0

        all_pixel_values = []
        for img_data in rlds_batch["observation"]["image_primary"]:
            img_data = img_data[:,:,:]
            # import cv2
            # cv2.imwrite('./tmp_img_training.png', img_data); 1/0

            img = Image.fromarray(img_data)
            # img.save("img.png")
            pixel_values = self.image_transform(img)
            all_pixel_values.append(pixel_values)
        all_pixel_values = torch.stack(all_pixel_values)
        all_pixel_seg_values = []
        for img_data in rlds_batch["observation"]["image_primary_seg"]:
            img_data = img_data[:,:,0]
            # import cv2
            # cv2.imwrite('./tmp_img_training.png', img_data) #; 1/0
            # img.save("img.png")
            pixel_values = torch.from_numpy(img_data)
            all_pixel_seg_values.append(pixel_values)
        all_pixel_seg_values = torch.stack(all_pixel_seg_values)

        all_wrist_values = None
        all_wrist_seg_values = None
        if "image_wrist" in rlds_batch["observation"]:
            all_wrist_values = []
            for img_data in rlds_batch["observation"]["image_wrist"]:
                img_data = img_data[:,:,:]
                # import cv2
                # cv2.imwrite('./tmp_wrist_img_training.png', img_data); 1/0

                wrist_img = Image.fromarray(img_data)
                # wrist_img.save("img_wrist_img.png")

                wrist_values = self.image_transform(wrist_img)
                all_wrist_values.append(wrist_values)
            all_wrist_values = torch.stack(all_wrist_values)

            all_wrist_seg_values = []
            for img_data in rlds_batch["observation"]["image_wrist_seg"]:
                img_data = img_data[:,:,0]
                # import cv2
                # cv2.imwrite('./tmp_img_training.png', img_data) #; 1/0
                # img.save("img.png")
                wrist_values = torch.from_numpy(img_data)
                all_wrist_seg_values.append(wrist_values)
            all_wrist_seg_values = torch.stack(all_wrist_seg_values)

        all_pixel_depth_values = None
        if "depth_primary" in rlds_batch["observation"]:
            all_pixel_depth_values = []
            for img_data in rlds_batch["observation"]["depth_primary"]:
                img_data = img_data[:,:,:]
                img_data = np.concatenate([img_data, img_data, img_data], axis=-1)
                # import cv2
                # cv2.imwrite('./tmp_depth_img_training.png', img_data); 1/0

                img_data = Image.fromarray(img_data)
                # img_data.save("img_data_depth.png")

                pixel_values = self.image_transform(img_data)
                all_pixel_depth_values.append(pixel_values)
            all_pixel_depth_values = torch.stack(all_pixel_depth_values)

        all_wrist_depth_values = None
        if "depth_wrist" in rlds_batch["observation"]:
            all_wrist_depth_values = []
            for img_data in rlds_batch["observation"]["depth_wrist"]:
                img_data = img_data[:,:,:]
                img_data = np.concatenate([img_data, img_data, img_data], axis=-1)
                img_data = Image.fromarray(img_data)
                # img_data.save("img_wrist_data_depth.png"); 1/0

                wrist_values = self.image_transform(img_data)
                all_wrist_depth_values.append(wrist_values)
            all_wrist_depth_values = torch.stack(all_wrist_depth_values)
        
        # print(lang_nouns)
        # print(rlds_batch["observation"]["reasoning_on_image"])
        # print(rlds_batch["observation"]["reasoning_on_wrist_image"])
        # 1/0

        task = {
            'full_instruction': prompt_builder.get_prompt(),
            'instruction': lang,
            'instruction_nouns': language_instruction_nouns
        }

        return dict(pixel_values=all_pixel_values[0], input_ids=input_ids, labels=windowed_labels, dataset_name=dataset_name,
                    all_pixel_values=all_pixel_values, all_wrist_values=all_wrist_values,
                    all_pixel_depth_values=all_pixel_depth_values, all_wrist_depth_values=all_wrist_depth_values,
                    all_pixel_seg_values=all_pixel_seg_values, all_wrist_seg_values=all_wrist_seg_values,
                    reasoning_on_image=reasoning_on_image, reasoning_on_wrist_image=reasoning_on_wrist_image,
                    lang_nouns_ids=lang_nouns_ids, task=task)


@dataclass
class RLDSBatchTransformV3_1:
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        horizon = len(rlds_batch["observation"]["reasoning_on_image"])
        future_horizon = len(rlds_batch["action"]) - horizon
        action_dim = len(rlds_batch["action"][0])
        dataset_name = rlds_batch["dataset_name"]
        lang = rlds_batch["task"]["language_instruction"].decode().lower()
        if "language_instruction_nouns" in rlds_batch["task"] and lang in LANGUAGE_INSTRUCTION_NOUNS:
            language_instruction_nouns = LANGUAGE_INSTRUCTION_NOUNS[lang]
        else:
            language_instruction_nouns = ['robot']

        windowed_labels = []
        for hidx in range(horizon):
            action_sequence = ''
            for fidx in range(future_horizon):
                action = self.action_tokenizer(rlds_batch["action"][hidx+fidx])
                action_sequence = action_sequence + action

            # Label tokens.        
            prompt_builder = self.prompt_builder_fn("openvla")
            conversation = [
                {"from": "human", "value": f"What action should the robot take?"}, #  to {lang}  , we are relying on CLIP's embeddings for true actions
                {"from": "gpt", "value": action_sequence},
            ]
            for turn in conversation:
                prompt_builder.add_turn(turn["from"], turn["value"])
            # Tokenize (w/ `base_tokenizer`)
            labels = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
            labels = list(labels)
            windowed_labels.append(labels)
            # print(prompt_builder.get_prompt())
            # print(labels)
            
        # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
        prompt_builder = self.prompt_builder_fn("openvla")
        placeholder_seg = "_ _ _ _ _ _ _ _"
        for fidx in range(future_horizon-1):
            placeholder_seg += " _ _ _ _ _ _ _"
        conversation = [
            {"from": "human", "value": f"What action should the robot take?"}, #  to {lang}  , we are relying on CLIP's embeddings for true actions
            {"from": "gpt", "value": placeholder_seg},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])
        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        input_ids[-2-(action_dim*future_horizon)] = 29871
        # print(prompt_builder.get_prompt())
        # print(len(input_ids)); 1/0

        # here we get nouns, bboxes of interest across `horizon`
        lang_nouns_ids = {}
        reasoning_on_image = []
        reasoning_on_wrist_image = []
        if "reasoning_on_image" in rlds_batch["observation"]:
            _lang_nouns = list(language_instruction_nouns)
            for hidx in range(len(rlds_batch["observation"]["reasoning_on_image"])):
                _reasoning_on_image = process_reasoning_cues_v2(rlds_batch["observation"]["reasoning_on_image"][hidx])
                if 'reasoning_on_wrist_image' in rlds_batch["observation"]:
                    _reasoning_on_wrist_image = process_reasoning_cues_v2(rlds_batch["observation"]["reasoning_on_wrist_image"][hidx])
                    # _lang_nouns = list(set(list(_reasoning_on_image.keys()) + list(_reasoning_on_wrist_image.keys()) + _lang_nouns))
                    # _reasoning_on_image, _reasoning_on_wrist_image = align_cues_by_name(_reasoning_on_image, _reasoning_on_wrist_image)
                    reasoning_on_wrist_image.append(_reasoning_on_wrist_image)
                reasoning_on_image.append(_reasoning_on_image)

            for noun in _lang_nouns:
                noun_ids = self.base_tokenizer(noun).input_ids
                noun_ids = self.base_tokenizer.pad(
                    {"input_ids": noun_ids}, 
                    padding="max_length",  # or "max_length"
                    max_length=30,      # specify if using "max_length"
                    return_tensors="pt"
                ).input_ids
                lang_nouns_ids[noun] = noun_ids

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, windowed_labels = torch.tensor(input_ids), torch.tensor(windowed_labels)
        if input_ids.shape != windowed_labels[0].shape:
            print('input_ids\n', input_ids)
            print('windowed_labels\n', windowed_labels[0])
            1/0

        all_pixel_values = []
        for img_data in rlds_batch["observation"]["image_primary"]:
            img_data = img_data[:,:,:]
            import cv2
            # cv2.imwrite('./tmp_img_training.png', img_data); 1/0

            img = Image.fromarray(img_data)
            # img.save("img.png")
            pixel_values = self.image_transform(img)
            all_pixel_values.append(pixel_values)
        all_pixel_values = torch.stack(all_pixel_values)
        all_pixel_seg_values = []
        for img_data in rlds_batch["observation"]["image_primary_seg"]:
            img_data = img_data[:,:,0]
            # import cv2
            # cv2.imwrite('./tmp_img_training.png', img_data) #; 1/0
            # img.save("img.png")
            pixel_values = torch.from_numpy(img_data)
            all_pixel_seg_values.append(pixel_values)
        all_pixel_seg_values = torch.stack(all_pixel_seg_values)

        all_wrist_values = None
        all_wrist_seg_values = None
        if "image_wrist" in rlds_batch["observation"]:
            all_wrist_values = []
            for img_data in rlds_batch["observation"]["image_wrist"]:
                img_data = img_data[:,:,:]
                # import cv2
                # cv2.imwrite('./tmp_wrist_img_training.png', img_data); 1/0

                wrist_img = Image.fromarray(img_data)
                # wrist_img.save("img_wrist_img.png")

                wrist_values = self.image_transform(wrist_img)
                all_wrist_values.append(wrist_values)
            all_wrist_values = torch.stack(all_wrist_values)

            all_wrist_seg_values = []
            for img_data in rlds_batch["observation"]["image_wrist_seg"]:
                img_data = img_data[:,:,0]
                # import cv2
                # cv2.imwrite('./tmp_img_training.png', img_data) #; 1/0
                # img.save("img.png")
                wrist_values = torch.from_numpy(img_data)
                all_wrist_seg_values.append(wrist_values)
            all_wrist_seg_values = torch.stack(all_wrist_seg_values)

        all_pixel_depth_values = None
        if "depth_primary" in rlds_batch["observation"]:
            all_pixel_depth_values = []
            for img_data in rlds_batch["observation"]["depth_primary"]:
                img_data = img_data[:,:,:]
                img_data = np.concatenate([img_data, img_data, img_data], axis=-1)
                # import cv2
                # cv2.imwrite('./tmp_depth_img_training.png', img_data); 1/0

                img_data = Image.fromarray(img_data)
                # img_data.save("img_data_depth.png")

                pixel_values = self.image_transform(img_data)
                all_pixel_depth_values.append(pixel_values)
            all_pixel_depth_values = torch.stack(all_pixel_depth_values)

        all_wrist_depth_values = None
        if "depth_wrist" in rlds_batch["observation"]:
            all_wrist_depth_values = []
            for img_data in rlds_batch["observation"]["depth_wrist"]:
                img_data = img_data[:,:,:]
                img_data = np.concatenate([img_data, img_data, img_data], axis=-1)
                img_data = Image.fromarray(img_data)
                # img_data.save("img_wrist_data_depth.png"); 1/0

                wrist_values = self.image_transform(img_data)
                all_wrist_depth_values.append(wrist_values)
            all_wrist_depth_values = torch.stack(all_wrist_depth_values)
        
        # print(lang_nouns)
        # print(rlds_batch["observation"]["reasoning_on_image"])
        # print(rlds_batch["observation"]["reasoning_on_wrist_image"])
        # 1/0

        task = {
            'full_instruction': prompt_builder.get_prompt(),
            'instruction': lang,
            'instruction_nouns': language_instruction_nouns
        }

        return dict(pixel_values=all_pixel_values[0], input_ids=input_ids, labels=windowed_labels, dataset_name=dataset_name,
                    all_pixel_values=all_pixel_values, all_wrist_values=all_wrist_values,
                    all_pixel_depth_values=all_pixel_depth_values, all_wrist_depth_values=all_wrist_depth_values,
                    all_pixel_seg_values=all_pixel_seg_values, all_wrist_seg_values=all_wrist_seg_values,
                    reasoning_on_image=reasoning_on_image, reasoning_on_wrist_image=reasoning_on_wrist_image,
                    lang_nouns_ids=lang_nouns_ids, task=task)
