"""
Training script for an OpenVLA-based robot vision-language model with
object-centric slot supervision and continuous action prediction.

This script:
- Loads a pretrained OpenVLA model and processor from HuggingFace.
- Freezes the main vision backbone and optionally applies LoRA / 4-bit quantization.
- Wraps the base model with a custom SlotSSM-based object-centric action module.
- Loads pretrained object-centric weights from a .safetensors checkpoint.
- Trains on an RLDS-formatted robotics dataset with multi-view RGB and depth inputs.
- Uses language instructions, object masks, bounding boxes, segmentation labels,
  and per-object interaction annotations to supervise object-centric predictions.
- Builds temporal context from horizon windows and future-action windows.
- Predicts continuous robot actions from fused slot, text, and visual features.
- Optimizes an L1 loss on continuous action trajectories.
- Periodically saves object-centric weights during training.

Expected data:
- An RLDS dataset under `data_root_dir`
- Samples containing:
  - image tensors
  - segmentation maps
  - object metadata
  - language instructions
  - per-object reasoning and interaction annotations
  - action token sequences

How to run:

Single-GPU:
    CUDA_VISIBLE_DEVICES=0 torchrun \
        --standalone \
        --nnodes 1 \
        --nproc-per-node 1 \
        vla-scripts/finetune-3_slotssm_to_action.py \
        --vla_path "openvla/openvla-7b" \
        --data_root_dir "/path/to/datasets" \
        --dataset_name "droid_wipe" \
        --run_root_dir "/path/to/checkpoints" \
        --adapter_tmp_dir "/path/to/adapters" \
        --load_slot_path "/path/to/object_centric_bwd25_fwd6.safetensors" \
        --interaction_cache_path "/path/to/cnt_text_embeddings.pt" \
        --batch_size 16 \
        --horizon_size 32 \
        --bwd_steps 25 \
        --fwd_steps 6 \
        --number_of_slots 16 \
        --learning_rate 2e-5 \
        --grad_accumulation_steps 1 \
        --save_steps 5000 \
        --image_aug True \
        --modality_key 1000

Multi-GPU:
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
        --standalone \
        --nnodes 1 \
        --nproc-per-node 4 \
        vla-scripts/finetune-3_slotssm_to_action.py \
        --vla_path "openvla/openvla-7b" \
        --data_root_dir "/path/to/datasets" \
        --dataset_name "droid_wipe" \
        --load_slot_path "/path/to/object_centric_bwd25_fwd6.safetensors" \
        --interaction_cache_path "/path/to/cnt_text_embeddings.pt" \
        --batch_size 16 \
        --horizon_size 32 \
        --bwd_steps 25 \
        --fwd_steps 6 \
        --number_of_slots 16 \
        --learning_rate 2e-5 \
        --grad_accumulation_steps 1 \
        --save_steps 5000 \
        --image_aug True \
        --modality_key 1000

Modality key format:
    [RGB main, RGB wrist, Depth main, Depth wrist]

Examples:
    1000 -> RGB main only
    1100 -> RGB main + RGB wrist
    1111 -> All modalities
    0010 -> Depth main only

Notes:
- Designed for GPU training and distributed execution via `torchrun`.
- `horizon_size` should match `bwd_steps + fwd_steps + 1`.
- This is a research/debugging-oriented training pipeline.
- Temporal context comes from horizon windows, not explicit temporal labels.
"""

from __future__ import annotations

import gc
import os
import sys
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import draccus
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import tqdm
import wandb
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from safetensors.torch import load_file, save_file
from scipy.ndimage import center_of_mass
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
)

from prismatic.debug_tools import denormalize_from_DINO
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import (
    EmbodiedDecodedSlotSSM,
    OpenVLAForActionPrediction,
)
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.losses.training_losses import RobotSSMObjectLossWithTrack
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPredictionV3
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransformV3_1, RLDSDatasetV3_1
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODALITIES = ("rgb_main", "rgb_wrist", "depth_main", "depth_wrist")

def process_interactions(reasoning_on_image):
    batched_interactions = []
    for batch_entry in reasoning_on_image: # [B]
        horizoned_interactions = {}
        processed_interactions = {}
        for horizon_entry in batch_entry:  # [H]
            # find all keys of current horizon step
            for key, datum in horizon_entry.items(): # [O]
                # if is first value in the horizon
                if key not in horizoned_interactions:
                    horizoned_interactions[key] = datum[0]

        # for each current observable entry
        for horizon_entry in batch_entry:  # [H]
            # find all global keys
            for key in list(horizoned_interactions.keys()): # [O]
                # update key if it is visible
                if key in horizon_entry:
                    horizoned_interactions[key] = horizon_entry[key][0]
                # update interaction key as latest recorded
                if key not in processed_interactions:
                    processed_interactions[key] = []
                processed_interactions[key].append(horizoned_interactions[key])
        
        batched_interactions.append(processed_interactions)
    return batched_interactions

def process_seg_ids(reasoning_on_image):
    batched_seg_ids = []
    for batch_entry in reasoning_on_image: # [B]
        horizoned_seg_ids = {}
        processed_seg_ids = {}
        for horizon_entry in batch_entry:  # [H]
            # find all keys of current horizon step
            for key, datum in horizon_entry.items(): # [O]
                # if is first value in the horizon
                if key not in horizoned_seg_ids:
                    horizoned_seg_ids[key] = datum[1]
        processed_seg_ids = horizoned_seg_ids
        batched_seg_ids.append(processed_seg_ids)
    return batched_seg_ids

def process_bboxes(reasoning_on_image):
    batched_bboxes = []
    for batch_entry in reasoning_on_image: # [B]
        processed_bboxes = {}
        for horizon_entry in batch_entry:  # [H]
            # find all keys of current horizon step
            for key, datum in horizon_entry.items(): # [O]
                # if is first value in the horizon
                if key not in processed_bboxes:
                    processed_bboxes[key] = []

        # for each current observable entry
        for horizon_entry in batch_entry:  # [H]
            # find all global keys
            for key in list(processed_bboxes.keys()): # [O]
                # update key if it is visible
                if key in horizon_entry:
                    bbox_obs = horizon_entry[key][2]
                    bbox_obs = np.concatenate([bbox_obs] + [np.array([1], dtype=np.float64)])
                else:
                    bbox_obs = np.zeros(5, dtype=np.float64)
                processed_bboxes[key].append(bbox_obs)

        batched_bboxes.append(processed_bboxes)
    return batched_bboxes

def visualize_and_save_labeled_masks(seg_tensor, meta_list, output_dir="_segmentation_outputs", images=None):
    os.makedirs(output_dir, exist_ok=True)
    B, H, size, _ = seg_tensor.shape
    cmap = plt.cm.get_cmap("tab20", 20)  # up to 20 object colors

    for b in range(B):
        for h in range(H):
            # Convert to numpy and squeeze to [H, W]
            mask = seg_tensor[b][h].cpu().numpy() if seg_tensor[b].ndim == 3 else seg_tensor[b].cpu().numpy()

            for obj_name, obj_value in meta_list[b].items():
                color = np.array(cmap(obj_value % 20)[:3])  # map label to color
                colored = np.zeros((size, size, 3), dtype=np.float32)
                binary_mask = mask == obj_value
                for c in range(3):
                    colored[:, :, c][binary_mask] = color[c]

                # Label with object name at center of mass
                y, x = center_of_mass(binary_mask)
                plt.text(x, y, obj_name, fontsize=8, color="white",
                            bbox=dict(facecolor='black', alpha=0.5, pad=1))

                # Save image
                plt.figure(figsize=(4, 4))
                plt.imshow(colored)
                plt.axis("off")
                save_path = os.path.join(output_dir, f"mask_{b}_{h}_{obj_name}.png")
                plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
                plt.close()
            if images is not None:
                # Convert to numpy and squeeze to [H, W]
                image = (images[b,h].cpu().numpy()*255).astype(np.uint8)
                # Transpose to (H, W, C)
                image = np.transpose(image, (1, 2, 0))

                # Save with matplotlib
                save_path = os.path.join(output_dir, f"image_{b}_{h}.png")
                plt.imsave(save_path, image)

    print(f"Saved {B} visualizations to: {output_dir}")

@dataclass
class FinetuneConfig:
    # fmt: off
    vla_path: str = "openvla/openvla-7b"                            # Path to OpenVLA model (on HuggingFace Hub)

    # Directory Paths
    data_root_dir: Path = Path("datasets/open-x-embodiment")        # Path to Open-X dataset directory
    dataset_name: str = "droid_wipe"                                # Name of fine-tuning dataset (e.g., `droid_wipe`)
    run_root_dir: Path = Path("runs")                               # Path to directory to store logs & checkpoints
    adapter_tmp_dir: Path = Path("vla-scripts/tmp")                     # Temporary directory for LoRA weights before fusing
    load_slot_path: Optional[Path] = None                            # Stage-2 SlotSSM checkpoint
    interaction_cache_path: Path = Path("cnt_text_embeddings.pt")    # Cached interaction-text embeddings

    # Fine-tuning Parameters
    batch_size: int = 16                                            # Fine-tuning batch size
    horizon_size: int = 32                                          # Temporal window size
    bwd_steps: int = 25                                             # Back token prediction size
    fwd_steps: int = 6                                              # Forward token prediction size
    number_of_slots: int = 16                                       # The number of slots of the model
    max_steps: int = 31000                                          # Max number of fine-tuning steps
    save_steps: int = 5000                                          # Interval for checkpoint saving
    learning_rate: float = 2e-5                                     # Fine-tuning learning rate
    grad_accumulation_steps: int = 1                                # Gradient accumulation steps
    image_aug: bool = True                                          # Whether to train with image augmentations
    shuffle_buffer_size: int = 100_000                              # Dataloader shuffle buffer size (can reduce if OOM)
    save_latest_checkpoint_only: bool = True                        # Whether to save only one checkpoint per run and
                                                                    #   continually overwrite the latest checkpoint
                                                                    #   (If False, saves all checkpoints)

    # LoRA Arguments
    use_lora: bool = True                                           # Whether to use LoRA fine-tuning
    lora_rank: int = 32                                             # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                                       # Dropout applied to LoRA weights
    use_quantization: bool = False                                  # Whether to 4-bit quantize VLA for LoRA fine-tuning
                                                                    #   => CAUTION: Reduces memory but hurts performance

    # Tracking Parameters
    run_id_note: Optional[str] = None                               # Optional suffix for the experiment ID
    use_wandb: bool = True                                          # Log the main process to Weights & Biases
    wandb_entity: str = "nhat"                                      # W&B entity
    wandb_project: str = "libero-mem"                               # W&B project
    log_steps: int = 10                                             # Console and W&B logging interval

    # fmt: on
    modality_key: str = "1000"     # Binary characters for using RGB main, RGB wrist, depth main, depth wrist
    resume: bool = False                                           # Whether to resume old run
    debug: bool = False                                           # Whether to use debug mode
    train_rotation: bool = False                                  # Include xyz rotation dimensions in the action loss
    action_loss_start_step: int = 16                              # Ignore early horizon steps with little temporal context

MODALITIES = ['rgb_main', 'rgb_wrist', 'depth_main', 'depth_wrist']
def process_modality_key(modality_key):
    if len(modality_key) != len(MODALITIES) or any(bit not in "01" for bit in modality_key):
        raise ValueError(
            f"modality_key must contain four binary digits, got {modality_key!r}"
        )
    modality_flags = {}
    modality_num = 0
    for i, modal in enumerate(MODALITIES):
        if modality_key[i] == '1':
            modality_flags[modal] = True
            modality_num += 1
            print('Using', modal, 'data.')
        else:
            modality_flags[modal] = False
    return modality_flags, modality_num

def get_linear_module_names(module, parent_name='', exclude_pattern=''):
    linear_module_names = []
    for name, sub_module in module.named_children():
        # Construct the full module name
        full_name = f"{parent_name}.{name}" if parent_name else name

        if isinstance(sub_module, torch.nn.Linear):
            if exclude_pattern == '' or all(pattern not in full_name for pattern in exclude_pattern):
                linear_module_names.append(full_name)
        else:
            # Recursively search in child modules
            linear_module_names.extend(
                get_linear_module_names(sub_module, full_name, exclude_pattern)
            )
    return linear_module_names

def _get_matched_pairs(src_data, target_data, indices):
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])

    idx = (batch_idx, src_idx)
    src_data = src_data[idx]
    target_data = torch.cat([t[i] for t, (_, i) in zip(target_data, indices)], dim=0)
    return src_data, target_data

def get_slot_specific_subgoals(
        outputs,
        object_token_num, horizon,
        object_gts,
        all_obj_itrn_cnts,
        cnt_embedding_table,
        object_loss,
        object_centric_text_encoder,
        cache_path: Path,
    ):
    bz = outputs['visual_tokens'].shape[0]
    device = outputs["visual_tokens"].device
    new_key = False
    if "0" not in cnt_embedding_table:
        cnt_embedding_table["0"] = (
            object_centric_text_encoder.encode_simple(["0"], device)
            .detach()
            .reshape(-1)
        )
        new_key = True
    # Get the object interaction count embeddings
    all_obj_itrn_cnts_mapped = []
    for b in range(bz):
        batched_subgoal_encodings = []
        for obj_itrn in all_obj_itrn_cnts[b]:
            obj_subgoals = []
            for key in obj_itrn:
                if key not in cnt_embedding_table:
                    new_key = True
                    obj_subgoal_encodings = (
                        object_centric_text_encoder.encode_simple([key], device)
                        .detach()
                        .reshape(-1)
                    )
                    cnt_embedding_table[key] = obj_subgoal_encodings
                else:
                    obj_subgoal_encodings = cnt_embedding_table[key].reshape(-1)
                obj_subgoals.append(obj_subgoal_encodings)

            batched_subgoal_encodings.append(torch.stack(obj_subgoals))

        obj_itrn_cnts_mapped = torch.stack(batched_subgoal_encodings)
        # Output shape will be [B, H, W, encoded_dim]
        all_obj_itrn_cnts_mapped.append(obj_itrn_cnts_mapped)

    if new_key and (not dist.is_initialized() or dist.get_rank() == 0):
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {key: value.detach().cpu() for key, value in cnt_embedding_table.items()},
            cache_path,
        )

    # align the gt data with slot predictions
    object_preds = {'bboxes': outputs['bboxes'].permute(0, 2, 1, 3)}
    indices = object_loss.get_match_indices(object_preds, object_gts)
    
    subgoal_states = []
    for b in range(bz):
        slot_specific_cnt = []
        for s in range(object_token_num):
            slot_specific_cnt.append(None)
        
        # get matched indices
        src, dst = indices[b]
        for i, src_idx in enumerate(src):
            dst_idx = dst[i]
            slot_specific_cnt[src_idx.item()] = all_obj_itrn_cnts_mapped[b][dst_idx.item()]
        # zero init the None's
        for s in range(object_token_num):
            if slot_specific_cnt[s] is None:
                zero_embedding = cnt_embedding_table["0"].reshape(-1)
                slot_specific_cnt[s] = zero_embedding.unsqueeze(0).expand(
                    horizon, -1
                )

        # slot cnts
        slot_specific_cnt = torch.stack(slot_specific_cnt)
        # slot_specific_cnt += _general_clip_embeddings
        subgoal_states.append(slot_specific_cnt)

    subgoal_states = torch.stack(subgoal_states)
    subgoal_states = subgoal_states.permute([0, 2, 1, 3]) # 'b o t d -> b t o d'
    return subgoal_states, indices


import gc

@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    print(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`")

    # [Validate] Ensure GPU Available & Set Device / Distributed Context
    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    if cfg.grad_accumulation_steps < 1:
        raise ValueError("grad_accumulation_steps must be at least 1")
    if cfg.max_steps < 1 or cfg.save_steps < 1:
        raise ValueError("max_steps and save_steps must be at least 1")
    if cfg.log_steps < 1:
        raise ValueError("log_steps must be at least 1")
    if cfg.horizon_size != cfg.bwd_steps + cfg.fwd_steps + 1:
        raise ValueError(
            "horizon_size must equal bwd_steps + fwd_steps + 1; "
            f"got {cfg.horizon_size} != {cfg.bwd_steps} + {cfg.fwd_steps} + 1"
        )
    if not 0 <= cfg.action_loss_start_step < cfg.horizon_size:
        raise ValueError(
            "action_loss_start_step must be within the horizon; "
            f"got {cfg.action_loss_start_step} for horizon {cfg.horizon_size}"
        )
    if cfg.use_quantization:
        raise NotImplementedError(
            "Stage-3 4-bit training is not supported yet because the newly "
            "constructed action modules require explicit mixed device placement"
        )
    modality_flags, _ = process_modality_key(cfg.modality_key)
    if not modality_flags["rgb_main"] or any(
        modality_flags[name] for name in ("rgb_wrist", "depth_main", "depth_wrist")
    ):
        raise ValueError(
            "The current SlotSSM tokenizer supports primary RGB only; "
            "use --modality_key 1000"
        )
    if cfg.load_slot_path is None:
        raise ValueError(
            "Stage 3 requires --load_slot_path pointing to the Stage-2 "
            "object_centric_bwd*_fwd*.safetensors checkpoint"
        )
    if not cfg.load_slot_path.is_file():
        raise FileNotFoundError(f"Stage-2 checkpoint not found: {cfg.load_slot_path}")
    if not cfg.interaction_cache_path.is_file():
        raise FileNotFoundError(
            f"Interaction embedding cache not found: {cfg.interaction_cache_path}"
        )

    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()

    # Configure Unique Experiment ID & Log Directory
    exp_id = (
        f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps * distributed_state.num_processes}"
        f"+lr-{cfg.learning_rate}"
    )
    if cfg.use_lora:
        exp_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    if cfg.use_quantization:
        exp_id += "+q-4bit"
    if cfg.run_id_note is not None:
        exp_id += f"--{cfg.run_id_note}"
    if cfg.image_aug:
        exp_id += "--image_aug"
    exp_id += "--multiview_bwdfwd"
    exp_id += cfg.modality_key + "--fined"

    # Start =>> Build Directories
    run_dir, adapter_dir = cfg.run_root_dir / exp_id, cfg.adapter_tmp_dir / exp_id
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(adapter_dir, exist_ok=True)
    training_state_path = adapter_dir / "training_state.pt"
    training_signature = {
        "bwd_steps": cfg.bwd_steps,
        "fwd_steps": cfg.fwd_steps,
        "number_of_slots": cfg.number_of_slots,
        "action_loss_start_step": cfg.action_loss_start_step,
        "wandb_entity": cfg.wandb_entity,
        "wandb_project": cfg.wandb_project,
    }
    resume_state = None
    if cfg.resume:
        if not training_state_path.is_file():
            raise FileNotFoundError(
                f"Cannot resume: training state not found at {training_state_path}"
            )
        resume_state = torch.load(training_state_path, map_location="cpu")
        if bool(resume_state.get("use_lora")) != cfg.use_lora:
            raise ValueError(
                "Resume configuration disagrees with the saved run: "
                f"saved use_lora={resume_state.get('use_lora')}, "
                f"requested use_lora={cfg.use_lora}"
            )
        if resume_state.get("training_signature") != training_signature:
            raise ValueError(
                "Resume configuration disagrees with the saved model signature: "
                f"saved={resume_state.get('training_signature')}, "
                f"requested={training_signature}"
            )
        if int(resume_state["completed_steps"]) >= cfg.max_steps:
            raise ValueError(
                "The saved run has already reached the requested max_steps: "
                f"{resume_state['completed_steps']} >= {cfg.max_steps}"
            )

    # Quantization Config =>> only if LoRA fine-tuning
    quantization_config = None
    if cfg.use_quantization:
        assert cfg.use_lora, "Quantized training only supported for LoRA fine-tuning!"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )

    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Load OpenVLA Processor and Model using HF AutoClasses
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=False)

    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )

    # Device Placement =>> note that BitsAndBytes automatically handles for quantized training
    if cfg.use_quantization:
        vla = prepare_model_for_kbit_training(vla)
    else:
        vla = vla.to(device_id)

    vla.vision_backbone.requires_grad_(False)

    # [LoRA] Wrap Model w/ PEFT `LoraConfig` =>> by default we set `target_modules=all-linear`
    if cfg.use_lora:
        if cfg.resume:
            adapter_config_path = adapter_dir / "adapter_config.json"
            if not adapter_config_path.is_file():
                raise FileNotFoundError(
                    f"Cannot resume LoRA: adapter config not found at {adapter_config_path}"
                )
            vla = PeftModel.from_pretrained(
                vla,
                adapter_dir,
                is_trainable=True,
            )
        else:
            # Adapt language-model linear layers, not the frozen vision/projector path.
            linear_module_names = get_linear_module_names(
                vla,
                exclude_pattern=["vision_backbone", "projector"],
            )
            lora_config = LoraConfig(
                r=cfg.lora_rank,
                lora_alpha=min(cfg.lora_rank, 16),
                lora_dropout=cfg.lora_dropout,
                target_modules=linear_module_names,
                init_lora_weights="gaussian",
            )
            vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()

    vla = EmbodiedDecodedSlotSSM(
        base_model=vla,
        number_of_slots=cfg.number_of_slots,
        backward_step=cfg.bwd_steps,
        forward_step=cfg.fwd_steps,
    )
    slot_state_dict = load_file(str(cfg.load_slot_path), device="cpu")
    incompatible_keys = vla.load_state_dict(slot_state_dict, strict=False)
    required_prefixes = (
        "object_centric_tokenizer.",
        "object_centric_bbox_head.",
        "object_centric_mask_head.",
        "object_centric_pre_ssm_fusion.",
        "object_centric_ssm.",
        "object_centric_latent_decoder.",
    )
    absent_prefixes = [
        prefix
        for prefix in required_prefixes
        if not any(key.startswith(prefix) for key in slot_state_dict)
    ]
    if absent_prefixes:
        raise RuntimeError(
            "The Stage-2 checkpoint is incomplete; missing tensor groups: "
            + ", ".join(absent_prefixes)
        )
    if incompatible_keys.unexpected_keys:
        raise RuntimeError(
            "The Stage-2 checkpoint contains keys unknown to the Stage-3 model: "
            + ", ".join(incompatible_keys.unexpected_keys[:20])
        )
    print(
        f"Loaded {len(slot_state_dict)} Stage-2 tensors from {cfg.load_slot_path}"
    )
    del slot_state_dict

    if cfg.resume:
        stage3_checkpoint_path = adapter_dir / resume_state["checkpoint_name"]
        if not stage3_checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Cannot resume: Stage-3 checkpoint not found at {stage3_checkpoint_path}"
            )
        stage3_state_dict = load_file(str(stage3_checkpoint_path), device="cpu")
        stage3_incompatible = vla.load_state_dict(stage3_state_dict, strict=False)
        action_prefixes = (
            "object_centric_action_slot_fusion.",
            "object_centric_action_slot_projector.",
            "object_centric_action_text_projector.",
            "object_centric_action_head.",
        )
        missing_action_prefixes = [
            prefix
            for prefix in action_prefixes
            if not any(key.startswith(prefix) for key in stage3_state_dict)
        ]
        if missing_action_prefixes or stage3_incompatible.unexpected_keys:
            raise RuntimeError(
                "Saved Stage-3 checkpoint is incompatible. "
                f"Missing action groups: {missing_action_prefixes}; "
                f"unexpected keys: {stage3_incompatible.unexpected_keys[:20]}"
            )
        print(
            f"Restored {len(stage3_state_dict)} Stage-3 tensors from "
            f"{stage3_checkpoint_path}"
        )
        del stage3_state_dict

    # Stage 3 treats the object tokenizer and temporal model as a frozen
    # representation. Only newly introduced action modules and optional LoRA
    # adapters are optimized.
    for module in (
        vla.object_centric_tokenizer,
        vla.object_centric_bbox_head,
        vla.object_centric_mask_head,
        vla.object_centric_pre_ssm_fusion,
        vla.object_centric_ssm,
        vla.object_centric_latent_decoder,
    ):
        module.requires_grad_(False)
    if not cfg.use_lora:
        vla.base_model.requires_grad_(False)

    vla = vla.to(device_id)

    # Wrap VLA in PyTorch DDP Wrapper for Multi-GPU Training
    vla = DDP(
        vla,
        device_ids=[device_id],
        find_unused_parameters=True,
        gradient_as_bucket_view=True,
    )

    # Create Optimizer =>> note that we default to a simple constant learning rate!
    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters remain after Stage-3 freezing")
    if distributed_state.is_main_process:
        trainable_count = sum(parameter.numel() for parameter in trainable_params)
        total_count = sum(parameter.numel() for parameter in vla.parameters())
        print(
            f"Stage-3 trainable parameters: {trainable_count:,} / "
            f"{total_count:,} ({100 * trainable_count / total_count:.3f}%)"
        )
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)
    completed_steps = 0
    if cfg.resume:
        optimizer.load_state_dict(resume_state["optimizer"])
        completed_steps = int(resume_state["completed_steps"])
        print(
            f"Resuming at optimizer step {completed_steps}; "
            "the RLDS sample stream restarts from a new iterator"
        )

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # Load Fine-tuning Dataset =>> note that we use an RLDS-formatted dataset following Open X-Embodiment by default.
    #   =>> If you want to use a non-RLDS dataset (e.g., a standard PyTorch Dataset) see the following commented block.
    #   =>> Note that our training code does not loop over epochs because the RLDS loader does this implicitly; if using
    #       your own Dataset, make sure to add the appropriate logic to the training loop!
    #
    # ---
    # from prismatic.vla.datasets import DummyDataset
    #
    # vla_dataset = DummyDataset(
    #     action_tokenizer,
    #     processor.tokenizer,
    #     image_transform=processor.image_processor.apply_transform,
    #     prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    # )
    # ---
    # Create Collator and DataLoader
    collator = PaddedCollatorForActionPredictionV3(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    if cfg.debug:
        print("Debug mode uses small batch size=",2)
        cfg.batch_size = 2

    window_size = cfg.horizon_size # 60
    batch_transform = RLDSBatchTransformV3_1(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
        include_segmentation=False,
    )
    vla_dataset = RLDSDatasetV3_1(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        # resize_resolution=tuple(vla.module.config.image_sizes),
        resize_resolution=(224, 224),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
        window_size=window_size,
        future_action_window_size=5,
        load_camera_views=("primary",),
        load_depth=False,
        cropping=False
    )
    
    # [Important] Save Dataset Statistics =>> used to de-normalize actions for inference!
    if distributed_state.is_main_process:
        save_dataset_statistics(vla_dataset.dataset_statistics, adapter_dir)
        processor.save_pretrained(adapter_dir)
    # 1/0
    dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
        pin_memory=True,
    )

    wandb_run = None
    wandb_run_id = (
        resume_state.get("wandb_run_id")
        if resume_state is not None
        else None
    )
    if distributed_state.is_main_process and cfg.use_wandb:
        wandb_config = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(cfg).items()
        }
        wandb_run = wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=f"slotssm-action+{exp_id}",
            dir=str(run_dir),
            config=wandb_config,
            id=wandb_run_id,
            resume="allow" if wandb_run_id is not None else None,
        )
        wandb_run_id = wandb_run.id
        wandb_run.summary["trainable_parameters"] = sum(
            parameter.numel() for parameter in trainable_params
        )
        wandb_run.summary["total_parameters"] = sum(
            parameter.numel() for parameter in vla.parameters()
        )
        wandb_run.summary["stage2_checkpoint"] = str(cfg.load_slot_path)
        print(f"W&B run: {wandb_run.url}")

    # Deque to store recent train metrics (used for computing smoothened metrics for gradient accumulation)
    recent_l1_losses = deque(maxlen=cfg.grad_accumulation_steps)

    # Train!
    object_loss = RobotSSMObjectLossWithTrack()
    cnt_embedding_table = torch.load(
        cfg.interaction_cache_path,
        map_location="cpu",
    )
    for key, value in cnt_embedding_table.items():
        cnt_embedding_table[key] = value.reshape(-1).to(device_id)
    gc.collect()
    torch.cuda.empty_cache()
    action_dimension_mask = torch.tensor(
        [1, 1, 1, 1, 1, 1, 1]
        if cfg.train_rotation
        else [1, 1, 1, 0, 0, 0, 1],
        dtype=torch.float32,
        device=device_id,
    ).view(1, 1, 1, 7)

    def save_checkpoint(step: int) -> None:
        if not distributed_state.is_main_process:
            return
        processor.save_pretrained(adapter_dir)
        if cfg.use_lora:
            vla.module.base_model.save_pretrained(adapter_dir)
        object_centric_state_dict = {
            key: value.detach().contiguous()
            for key, value in vla.module.state_dict().items()
            if key.startswith("object_centric_")
        }
        suffix = "" if cfg.save_latest_checkpoint_only else f"_step{step}"
        checkpoint_path = (
            adapter_dir
            / (
                f"object_centric_bwd{cfg.bwd_steps}_fwd{cfg.fwd_steps}"
                f"_actionable{suffix}.safetensors"
            )
        )
        save_file(object_centric_state_dict, str(checkpoint_path))
        state_tmp_path = training_state_path.with_suffix(".pt.tmp")
        torch.save(
            {
                "completed_steps": step,
                "optimizer": optimizer.state_dict(),
                "checkpoint_name": checkpoint_path.name,
                "use_lora": cfg.use_lora,
                "training_signature": training_signature,
                "wandb_run_id": wandb_run_id,
            },
            state_tmp_path,
        )
        os.replace(state_tmp_path, training_state_path)
        if wandb_run is not None:
            wandb_run.summary["latest_checkpoint"] = str(checkpoint_path)
            wandb_run.summary["latest_checkpoint_step"] = step
        print(f"Saved Stage-3 checkpoint for step {step} at {checkpoint_path}")

    with tqdm.tqdm(
        total=cfg.max_steps,
        initial=completed_steps,
        desc="SlotSSM-Action",
        dynamic_ncols=True,
        disable=not distributed_state.is_main_process,
    ) as progress:
        vla.train()
        for frozen_module in (
            vla.module.base_model.vision_backbone,
            vla.module.object_centric_tokenizer,
            vla.module.object_centric_bbox_head,
            vla.module.object_centric_mask_head,
            vla.module.object_centric_pre_ssm_fusion,
            vla.module.object_centric_ssm,
            vla.module.object_centric_latent_decoder,
        ):
            frozen_module.eval()
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(dataloader):
            pixel_values = batch["all_pixel_values"].to(
                device=device_id,
                dtype=torch.bfloat16,
                non_blocking=True,
            )
            reasoning_on_image = batch["reasoning_on_image"]
            all_interaction_cnts = process_interactions(reasoning_on_image)
            all_bboxes = process_bboxes(reasoning_on_image)
            all_obj_keys = [
                list(sample_bboxes.keys()) for sample_bboxes in all_bboxes
            ]
            all_obj_bboxes = [
                [sample_bboxes[key] for key in all_obj_keys[index]]
                for index, sample_bboxes in enumerate(all_bboxes)
            ]
            all_obj_itrn_cnts = [
                [sample_interactions[key] for key in all_obj_keys[index]]
                for index, sample_interactions in enumerate(all_interaction_cnts)
            ]

            batch_size, horizon = pixel_values.shape[:2]
            object_gts = object_loss.preprocess(
                all_obj_bboxes,
                all_obj_segids=None,
                all_pixel_seg_values=None,
                all_interaction_cnts=all_obj_itrn_cnts,
                include_segs=False,
            )

            # The Stage-1 object tokenizer is frozen. Run it outside the DDP
            # forward and retain only its values for matching and Stage-3.
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                texts = [
                    "robot " + task["instruction"] for task in batch["tasks"]
                ]
                object_outputs = vla.module.get_obj_slots(
                    pixel_values,
                    texts,
                    batch_vision_backbone=True,
                    compute_masks=False,
                )
                subgoal_states, _ = get_slot_specific_subgoals(
                    object_outputs,
                    vla.module.object_token_num,
                    horizon,
                    object_gts,
                    all_obj_itrn_cnts,
                    cnt_embedding_table,
                    object_loss,
                    vla.module.object_centric_text_encoder,
                    cfg.interaction_cache_path,
                )
                object_outputs = {
                    key: object_outputs[key]
                    for key in (
                        "visual_tokens",
                        "patch_features",
                        "texts",
                        "texts_attn",
                    )
                }

            input_ids = batch["input_ids"].to(device_id, non_blocking=True)
            attention_mask = batch["attention_mask"].to(
                device_id,
                non_blocking=True,
            )
            is_optimizer_step = (
                (batch_idx + 1) % cfg.grad_accumulation_steps == 0
            )
            sync_context = nullcontext() if is_optimizer_step else vla.no_sync()
            with sync_context:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    continuous_actions_pred = vla(
                        object_outputs=object_outputs,
                        subgoal_states=subgoal_states,
                        llama_input_ids=input_ids,
                        llama_attention_mask=attention_mask,
                        action_start_step=cfg.action_loss_start_step,
                    )

                    loss_horizon = horizon - cfg.action_loss_start_step
                    expected_shape = (batch_size, loss_horizon, 5, 7)
                    if continuous_actions_pred.shape != expected_shape:
                        raise RuntimeError(
                            "Stage-3 action decoder violated its output contract: "
                            f"got {tuple(continuous_actions_pred.shape)}, "
                            f"expected {expected_shape}"
                        )

                    action_token_count = 5 * 7
                    action_token_ids = batch["labels"][
                        :, :, -1 - action_token_count : -1
                    ]
                    if (action_token_ids < 0).any():
                        raise RuntimeError(
                            "Action target slice contains padding/ignore tokens; "
                            "check RLDSBatchTransformV3_1 prompt construction"
                        )
                    continuous_actions_gt = torch.as_tensor(
                        action_tokenizer.decode_token_ids_to_actions(
                            action_token_ids.cpu().numpy()
                        ),
                        device=device_id,
                        dtype=continuous_actions_pred.dtype,
                    ).reshape(batch_size, horizon, 5, 7)[
                        :, cfg.action_loss_start_step :
                    ]

                    absolute_error = (
                        continuous_actions_pred - continuous_actions_gt
                    ).abs() * action_dimension_mask
                    denominator = (
                        continuous_actions_pred.shape[0]
                        * continuous_actions_pred.shape[1]
                        * continuous_actions_pred.shape[2]
                        * action_dimension_mask.sum()
                    )
                    action_l1_loss = absolute_error.sum() / denominator
                    normalized_loss = (
                        action_l1_loss / cfg.grad_accumulation_steps
                    )
                normalized_loss.backward()

            recent_l1_losses.append(action_l1_loss.item())
            smoothened_l1_loss = sum(recent_l1_losses) / len(recent_l1_losses)
            next_step = completed_steps + 1
            should_log = (
                distributed_state.is_main_process
                and is_optimizer_step
                and next_step % cfg.log_steps == 0
            )
            grad_norm = None
            if should_log:
                parameter_grad_norms = [
                    parameter.grad.detach().float().norm(2)
                    for parameter in trainable_params
                    if parameter.grad is not None
                ]
                grad_norm = (
                    torch.linalg.vector_norm(
                        torch.stack(parameter_grad_norms),
                        ord=2,
                    )
                    if parameter_grad_norms
                    else torch.zeros((), device=device_id)
                )

            if is_optimizer_step:
                optimizer.step()
                completed_steps += 1
                progress.update()

            if should_log:
                prediction_values = continuous_actions_pred.detach().float()
                target_values = continuous_actions_gt.detach().float()
                per_dimension_mae = (
                    prediction_values - target_values
                ).abs().mean(dim=(0, 1, 2))
                metrics = {
                    "train/action_l1_loss": smoothened_l1_loss,
                    "train/learning_rate": optimizer.param_groups[0]["lr"],
                    "train/grad_norm": grad_norm.item(),
                    "actions/pred_mean": prediction_values.mean().item(),
                    "actions/pred_std": prediction_values.std().item(),
                    "actions/target_mean": target_values.mean().item(),
                    "actions/target_std": target_values.std().item(),
                    "actions/mae_x": per_dimension_mae[0].item(),
                    "actions/mae_y": per_dimension_mae[1].item(),
                    "actions/mae_z": per_dimension_mae[2].item(),
                    "actions/mae_roll": per_dimension_mae[3].item(),
                    "actions/mae_pitch": per_dimension_mae[4].item(),
                    "actions/mae_yaw": per_dimension_mae[5].item(),
                    "actions/mae_gripper": per_dimension_mae[6].item(),
                    "system/gpu_allocated_gib": (
                        torch.cuda.memory_allocated(device_id) / 2**30
                    ),
                    "system/gpu_reserved_gib": (
                        torch.cuda.memory_reserved(device_id) / 2**30
                    ),
                    "system/gpu_peak_gib": (
                        torch.cuda.max_memory_allocated(device_id) / 2**30
                    ),
                }
                console_metrics = {
                    "step": completed_steps,
                    "l1": round(metrics["train/action_l1_loss"], 6),
                    "grad_norm": round(metrics["train/grad_norm"], 4),
                    "pred_std": round(metrics["actions/pred_std"], 4),
                    "target_std": round(metrics["actions/target_std"], 4),
                    "gpu_gib": round(metrics["system/gpu_allocated_gib"], 2),
                }
                progress.write(str(console_metrics))
                progress.set_postfix(
                    loss=f"{metrics['train/action_l1_loss']:.4f}",
                    grad=f"{metrics['train/grad_norm']:.3f}",
                    gpu=f"{metrics['system/gpu_allocated_gib']:.1f}G",
                )
                if wandb_run is not None:
                    wandb_run.log(metrics, step=completed_steps)
                torch.cuda.reset_peak_memory_stats(device_id)

            if is_optimizer_step:
                optimizer.zero_grad()
                should_save = (
                    completed_steps % cfg.save_steps == 0
                    or completed_steps >= cfg.max_steps
                )
                if should_save:
                    save_checkpoint(completed_steps)
                    distributed_state.wait_for_everyone()

                if completed_steps >= cfg.max_steps:
                    break

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    finetune()
