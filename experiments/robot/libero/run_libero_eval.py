"""
run_libero_eval.py

Runs a model in a LIBERO simulation environment.

Usage:
    # OpenVLA:
    # IMPORTANT: Set `center_crop=True` if model is fine-tuned with augmentations
    python experiments/robot/libero/run_libero_eval.py \
        --model_family openvla \
        --pretrained_checkpoint <CHECKPOINT_PATH> \
        --task_suite_name [ libero_spatial | libero_object | libero_goal | libero_10 | libero_90 ] \
        --center_crop [ True | False ] \
        --run_id_note <OPTIONAL TAG TO INSERT INTO RUN ID FOR LOGGING> \
        --use_wandb [ True | False ] \
        --wandb_project <PROJECT> \
        --wandb_entity <ENTITY>

    LIBERO_Goal
    CUDA_VISIBLE_DEVICES=6 python run_libero_eval.py \
        --model_family openvla \
        --slot_type eobject \
        --number_of_slots 16 \
        --pretrained_checkpoint "/cm/shared/weights/openvla/adapters/output_hf_model_openx+libero_goal+b4+lr-1e-05+lora-r64+dropout-0.0--image_aug--multiview1000--eobject-action" \
        --pretrained_checkpoint2 "/cm/shared/weights/openvla/adapters/output_hf_model_openx+libero_goal+b4+lr-1e-05+lora-r64+dropout-0.0--image_aug--multiview1000--eobject-action-continue" \
        --custom_param_checkpoint "/cm/shared/weights/openvla/adapters/output_hf_model_openx+libero_goal+b4+lr-1e-05+lora-r64+dropout-0.0--image_aug--multiview1000--eobject-action-continue/relate_object_bboxes_w_mask_actionable_s16h16.safetensors" \
        --task_suite_name libero_goal \
        --center_crop False \
        --saved_dir node4_eobject_slot-16-libero_goal

    CUDA_VISIBLE_DEVICES=6 python run_libero_eval.py \
        --model_family openvla \
        --slot_type erelation \
        --number_of_slots 16 \
        --pretrained_checkpoint "/cm/shared/weights/openvla/adapters/output_hf_model_openx+libero_goal+b4+lr-1e-05+lora-r64+dropout-0.0--image_aug--multiview1000--erelate-action" \
        --pretrained_checkpoint2 "/cm/shared/weights/openvla/adapters/output_hf_model_openx+libero_goal+b4+lr-1e-05+lora-r64+dropout-0.0--image_aug--multiview1000--erelate-action-continue" \
        --custom_param_checkpoint "/cm/shared/weights/openvla/adapters/output_hf_model_openx+libero_goal+b4+lr-1e-05+lora-r64+dropout-0.0--image_aug--multiview1000--erelate-action-continue/relate_object_bboxes_w_mask_actionable_s16h16.safetensors" \
        --task_suite_name libero_goal \
        --center_crop False \
        --saved_dir node4_erelation_slot-16-libero_goal
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
import torch
import draccus
import numpy as np
import tqdm
from libero.libero import benchmark
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

# import wandb

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../../../")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig, SlotVLAV2Config, CustomOpenVLAConfig, ObjectCentricVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction, EmbodiedObjectSlot, EmbodiedRelationSlot, EmbodiedObject_LangSlot, EmbodiedRelation_LangSlot
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from peft import PeftModel
from safetensors.torch import load_file

import json
def get_vla(cfg):
    """Loads and returns a VLA model from checkpoint."""
    # Load VLA checkpoint.
    print("[*] Instantiating Pretrained VLA model")
    # print("[*] Loading in F16 with Flash-Attention Enabled")
    print("[*] Loading in BF16 with Flash-Attention Enabled")

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    adapter_dir = cfg.pretrained_checkpoint.replace('/ckpts/', '/adapters/')

    try:
        1/0
        base_vla = AutoModelForVision2Seq.from_pretrained(
            adapter_dir, 
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            # torch_dtype=torch.float16,
            load_in_8bit=cfg.load_in_8bit,
            load_in_4bit=cfg.load_in_4bit,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
            local_files_only=True
        ).to('cuda')
        print("Loaded full base.")

    except:
        # 1/0
        base_vla = AutoModelForVision2Seq.from_pretrained(
            "/cm/shared/workspace/output_hf_model_openx", 
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            # torch_dtype=torch.float16,
            load_in_8bit=cfg.load_in_8bit,
            load_in_4bit=cfg.load_in_4bit,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
        ).to('cuda')
        print("Loaded base.")

        base_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
        base_vla = base_vla.merge_and_unload()
        base_vla = PeftModel.from_pretrained(base_vla, cfg.pretrained_checkpoint2)
        base_vla = base_vla.merge_and_unload()
        print("Merged CustomOpenVLA LLM LoRA.")
    
    if cfg.slot_type == 'oc_wo_filter':
        vla = EmbodiedObjectSlot(base_model=base_vla, number_of_slots = cfg.number_of_slots)
    elif cfg.slot_type == 'orc_wo_filter':
        vla = EmbodiedRelationSlot(base_model=base_vla, number_of_slots = cfg.number_of_slots)
    elif cfg.slot_type == 'oc':
        vla = EmbodiedObject_LangSlot(base_model=base_vla, number_of_slots = cfg.number_of_slots)
    elif cfg.slot_type == 'orc':
        vla = EmbodiedRelation_LangSlot(base_model=base_vla, number_of_slots = cfg.number_of_slots)
    
    weights = load_file(cfg.custom_param_checkpoint)
    vla.load_state_dict(weights, strict=False)
    vla.object_centric_tokenizer.requires_grad_(False)
    vla.object_centric_bbox_head.requires_grad_(False)
    vla.object_centric_mask_head.requires_grad_(False)
    vla.requires_grad_(False)
    vla = vla.to('cuda')

    # Move model to device.
    # Note: `.to()` is not supported for 8-bit or 4-bit bitsandbytes models, but the model will
    #       already be set to the right devices and casted to the correct dtype upon loading.
    if not cfg.load_in_8bit and not cfg.load_in_4bit:
        vla = vla.to('cuda')

    # Load dataset stats used during finetuning (for action un-normalization).
    dataset_statistics_path = os.path.join(adapter_dir, "dataset_statistics.json")
    if os.path.isfile(dataset_statistics_path):
        with open(dataset_statistics_path, "r") as f:
            norm_stats = json.load(f)
        vla.norm_stats = norm_stats
    else:
        print(
            "WARNING: No local dataset_statistics.json file found for current checkpoint.\n"
            "You can ignore this if you are loading the base VLA (i.e. not fine-tuned) checkpoint."
            "Otherwise, you may run into errors when trying to call `predict_action()` due to an absent `unnorm_key`."
        )

    return vla

def get_model(cfg, wrap_diffusion_policy_for_droid=False):
    """Load model for evaluation."""
    if cfg.model_family == "openvla" or cfg.model_family == "customvla" or cfg.model_family == 'objectvla':
        model = get_vla(cfg)
    else:
        raise ValueError("Unexpected `model_family` found in config.")
    print(f"Loaded model: {type(model)}")
    return model

def visualize_all(visuals):
    # Get the corresponding visualizations
    row1 = [
        visuals['exo_rgb'],  # 'exo rgb'
        visuals['exo_rgb_boxed'],  # 'exo rgb'
        visuals['exo_rgb_boxed_contact'],  # 'exo rgb'
        visuals['exo_depth'],  # 'exo depth'
        visuals['exo_seg'],  # 'exo seg'
    ]

    row2 = [
        visuals['ego_rgb'],  # 'ego rgb'
        visuals['ego_rgb_boxed'],  # 'ego rgb'
        visuals['ego_rgb_boxed_contact'],  # 'ego rgb'
        visuals['ego_depth'],  # 'ego depth'
        visuals['ego_seg'],  # 'ego seg'
    ]

    # Function to preprocess images (convert grayscale to RGB and resize)
    def preprocess_image(image, target_size=(300, 300)):
        if len(image.shape) == 2:  # Convert grayscale to RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, target_size)
        return image

    # Process all images to uniform size and color format and concatenate horizontally
    row1_concat = np.hstack([preprocess_image(img) for img in row1])
    row2_concat = np.hstack([preprocess_image(img) for img in row2])

    # Concatenate vertically to form a 2x4 grid
    final_image = np.vstack([row1_concat, row2_concat])
    cv2.imshow(task_description, final_image)

from dataclasses import dataclass, field
@dataclass
class MambaCache:
    """Inference parameters that are passed to the main model in order
    to efficienly calculate and store the context during inference."""
    seqlen_offset: int = 0
    key_value_memory_dict: dict = field(default_factory=dict)

@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    slot_type: str = "relation"                      # relation, object, erelation, eobject
    number_of_slots: int = 16                        # number of slots in the model
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    pretrained_checkpoint2: Union[str, Path] = ""     # Pretrained checkpoint path
    custom_param_checkpoint: Union[str, Path] = ""   # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization
    
    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_goal"            # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 20                        # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 20                   # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs
    saved_dir: str = "libero_goal_01_slots"             # Libero goal slots

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under

    seed: int = 7                                    # Random Seed (for reproducibility)

    writing_extra_waits: bool = False
    # fmt: on

import re

def natural_key(s):
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', s)]

MAX_STEPS = {
    'KITCHEN_SCENE1_pick_the_bowl_from_the_plate_and_place_it_back_1_time': 275,
    'KITCHEN_SCENE1_pick_the_bowl_from_the_plate_and_place_it_back_3_times': 365,
    'KITCHEN_SCENE1_pick_the_bowl_from_the_plate_and_place_it_back_5_times': 640,
    'KITCHEN_SCENE1_pick_the_bowl_from_the_plate_and_place_it_back_7_times': 800,
    'KITCHEN_SCENE1_swap_the_3_bowls_from_left_to_right_using_the_intermediary_plate': 635,
    'KITCHEN_SCENE1_swap_the_2_bowls_using_the_intermediary_plate': 505
}

import cv2
import pickle
from mem_object_centric_tracker import TrajectoryRecorder, InteractionMonitor

IMAGE_RESOLUTION = 256
from PIL import Image

def process_inputs(processor, pixel_values, device):
    image = Image.fromarray(pixel_values)
    image = image.convert("RGB")
    
    # Build VLA prompt
    action_dim = 7; future_horizon = 5
    prompt = f"In: What action should the robot take?\nOut: "
    placeholder_seg = "_ _ _ _ _ _ _ _"
    for fidx in range(future_horizon-1):
        placeholder_seg += " _ _ _ _ _ _ _"
    prompt = prompt + placeholder_seg + "</s>"

    # Process inputs.
    inputs = processor(prompt, image).to(device, dtype=torch.float16)
    inputs['input_ids'][0][-2-(action_dim*future_horizon)] = 29871

    return inputs

def get_subgoal_states(model, indices, all_obj_cnts, cnt_embedding_table, horizon):
    slot_specific_cnt = []
    for s in range(model.object_token_num):
        slot_specific_cnt.append(None)

    # get matched indices
    src, dst = indices[0]
    for i, src_idx in enumerate(src):
        dst_idx = dst[i]
        slot_specific_cnt[src_idx.item()] = cnt_embedding_table[all_obj_cnts[0][dst_idx.item()]].squeeze(1)
    # zero init the None's
    for s in range(model.object_token_num):
        if slot_specific_cnt[s] is None:
            slot_specific_cnt[s] = torch.stack([cnt_embedding_table[0]]*horizon)

    # slot cnts
    subgoal_states = torch.stack([torch.stack(slot_specific_cnt)])
    subgoal_states = subgoal_states.permute([0, 2, 1, 3]) # 'b o t d -> b t o d'
    return subgoal_states

def get_current_slots(vla, batch, task_texts, in_past_tokens, device_id='cuda'):
    pixel_values = batch["pixel_values"].to(torch.bfloat16).to(device_id)
    outputs = vla.get_obj_slots(pixel_values, task_texts, in_past_tokens)
    return outputs


def get_normalized_actions(vla, batch, slot_outputs, device_id):
    # get object-centric dynamics
    patch_features = slot_outputs['patch_features']
    slotted_features = slot_outputs['visual_tokens']
    if 'interactable_features' in slot_outputs:
        slotted_features, _ = vla.select_top_k_slots(
                            slotted_features, slot_outputs['interactable_features'], k=4)
    clip_embeddings = slot_outputs['texts']
    clip_attention_mask = torch.logical_not(slot_outputs['texts_attn'])
    llama_input_ids = batch["input_ids"].to(device_id)
    llama_attention_mask = batch["attention_mask"].to(device_id)

    continuous_actions_pred = vla.decode_continuous_actions(
        patch_features=patch_features,
        slotted_features=slotted_features,
        clip_embeddings=clip_embeddings,
        clip_attention_mask=clip_attention_mask,
        llama_input_ids=llama_input_ids,
        llama_attention_mask=llama_attention_mask,
        llama_labels=None
    ) # will return output of [cross-modality-slots, cross-modality-bboxes]
    action_chunk, action_dim = continuous_actions_pred.shape[1:]

    return continuous_actions_pred[:,0,:]

def denormalize_actions(model, unnorm_key, normalized_actions):
    # Unnormalize actions
    action_norm_stats = model.get_action_stats(unnorm_key)
    mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
    action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
    actions = np.where(
        mask,
        0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
        normalized_actions,
    )
    return actions


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    # if "image_aug" in cfg.pretrained_checkpoint:
    #     assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # [OpenVLA] Set action un-normalization key
    cfg.unnorm_key = cfg.task_suite_name

    # # Load model
    model = get_model(cfg)

    # [OpenVLA] Check that the model contains the action un-normalization key
    if cfg.model_family == "openvla":
        # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
        # with the suffix "_no_noops" in the dataset name)
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

    # [OpenVLA] Get Hugging Face processor
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)

    # Initialize local logging
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging as well
    # if cfg.use_wandb:
    #     wandb.init(
    #         entity=cfg.wandb_entity,
    #         project=cfg.wandb_project,
    #         name=run_id,
    #     )

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    extra_waits_dict = {}
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)

        # Set trajectory recorder for object centric labels
        recorder = TrajectoryRecorder(
                image_resolution=IMAGE_RESOLUTION
        )
        recorder.reset_object_mappers(env)

        # Start episodes
        task_episodes, task_successes = 0, 0
        extra_waits = []
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            # Reset environment
            env.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])
            
            # Setup
            t = 0
            replay_images = []
            if cfg.task_suite_name == "libero_spatial":
                max_steps = 220  # longest training demo has 193 steps
            elif cfg.task_suite_name == "libero_object":
                max_steps = 280  # longest training demo has 254 steps
            elif cfg.task_suite_name == "libero_goal":
                max_steps = 300  # longest training demo has 270 steps
            elif cfg.task_suite_name == "libero_10":
                max_steps = 520  # longest training demo has 505 steps
            elif cfg.task_suite_name == "libero_90":
                max_steps = 400  # longest training demo has 373 steps

            t = 0
            while t < cfg.num_steps_wait:
                # try:
                # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                # and we need to wait for them to fall
                obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
            
            # Process observations and task
            task_texts = [task_description]
            texts = ['robot ' + task_description]
            img = get_libero_image(obs, resize_size)
            img = img[:,::-1,:]
            batch_data = process_inputs(processor=processor, pixel_values=img, device='cuda')
            # pixel_values    = batch_data['pixel_values']    #  torch.Size([1, 6, 224, 224])
            # input_ids       = batch_data['input_ids']       #  torch.Size([1, 50])
            # attention_mask  = batch_data['attention_mask']  # torch.Size([1, 50])
            horizon = 1
            batch_data['pixel_values'] = batch_data['pixel_values'].unsqueeze(1).repeat(1,horizon,1,1,1)
            running_batch_data = batch_data['pixel_values']
            in_past_tokens_data = [None, None]
            # Replicate frames across temporal dimension for initialization purposes
            # Get the object-centric slot embeddings
            with torch.autocast("cuda", dtype=torch.bfloat16):
                for idx, in_past_tokens in enumerate(in_past_tokens_data):
                    _slot_outputs = get_current_slots(model, batch_data, texts, in_past_tokens, device_id='cuda')
                    in_past_tokens_data[idx] = _slot_outputs['visual_tokens']
                    
            # Query model to get several prelim actions
            with torch.autocast("cuda", dtype=torch.bfloat16):
                actions = get_normalized_actions(model, batch_data, _slot_outputs, device_id='cuda')
                actions = actions.detach().float().cpu().numpy()
                unnormalized_actions = denormalize_actions(model, cfg.unnorm_key, actions)

            ## Start running the rest of the model here
            while t < max_steps + cfg.num_steps_wait:
                # Get preprocessed image
                img = get_libero_image(obs, resize_size)
                img = img[:,::-1,:]
                batch_data = process_inputs(processor=processor, pixel_values=img, device='cuda')
                # pixel_values    = batch_data['pixel_values']    #  torch.Size([1, 6, 224, 224])
                # input_ids       = batch_data['input_ids']       #  torch.Size([1, 50])
                # attention_mask  = batch_data['attention_mask']  # torch.Size([1, 50])
                # Save preprocessed image for replay video
                replay_images.append(img)

                # Getting frames across temporal dimension for initialization purposes
                batch_data['pixel_values'] = batch_data['pixel_values'].unsqueeze(1)
                running_batch_data = torch.cat([running_batch_data, batch_data['pixel_values']], dim=1)[:,1:]
                # Get the object-centric slot embeddings
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    for idx, in_past_tokens in enumerate(in_past_tokens_data):
                        _slot_outputs = get_current_slots(model, batch_data, texts, in_past_tokens, device_id='cuda')
                        in_past_tokens_data[idx] = _slot_outputs['visual_tokens']
                                        
                # Query model to get several prelim actions
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    actions = get_normalized_actions(model, batch_data, _slot_outputs, device_id='cuda')
                    actions = actions.detach().float().cpu().numpy()
                    unnormalized_actions = denormalize_actions(model, cfg.unnorm_key, actions)
                    action = unnormalized_actions[0]

                # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
                action = normalize_gripper_action(action, binarize=True)

                # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
                # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
                # if cfg.model_family == "openvla":
                #     action = invert_gripper_action(action)

                # Execute action in environment
                obs, reward, done, info = env.step(action.tolist())
                t += 1
                if done:
                    task_successes += 1
                    total_successes += 1
                    break

                if ((t - cfg.num_steps_wait) >= 8 and (t - cfg.num_steps_wait) % 8 == 0):
                    in_past_tokens_data.insert(0, None)
                    in_past_tokens_data = in_past_tokens_data[:2]

                # # Get wait times
                # cv2.imshow('obs', img)
                # _key_ = cv2.waitKey(1)
                # if _key_ == ord('k'):
                #     extra_wait_time = t-cfg.num_steps_wait
                #     print('extra wait =', extra_wait_time)
                # elif _key_ == ord('q'):
                #     break

                # except Exception as e:
                #     print(f"Caught exception: {e}")
                #     log_file.write(f"Caught exception: {e}\n")
                #     break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            cv2.destroyAllWindows()
            save_rollout_video(
                replay_images, total_episodes, success=done, task_description=task_description, log_file=log_file, saved_dir=cfg.saved_dir
            )
            # # Log current results
            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.flush()
            
        # Log final results
        print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
        log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
        log_file.flush()
        # if cfg.use_wandb:
        #     wandb.log(
        #         {
        #             f"success_rate/{task_description}": float(task_successes) / float(task_episodes),
        #             f"num_episodes/{task_description}": task_episodes,
        #         }
        #     )

    # Save local log file
    log_file.close()

    # Push total metrics and local log file to wandb
    # if cfg.use_wandb:
    #     wandb.log(
    #         {
    #             "success_rate/total": float(total_successes) / float(total_episodes),
    #             "num_episodes/total": total_episodes,
    #         }
    #     )
    #     wandb.save(local_log_filepath)


if __name__ == "__main__":
    eval_libero()
