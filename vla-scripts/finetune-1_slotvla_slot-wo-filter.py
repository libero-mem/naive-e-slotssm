"""
Fine-tuning script for an OpenVLA-based robot vision-language model with
object-centric supervision.

This script:
- Loads a pretrained OpenVLA model and processor from HuggingFace.
- Freezes the main vision backbone, projector, and language model.
- Wraps the base model with a custom object-centric slot-attention head.
- Optionally initializes the object-centric modules from a Safetensors checkpoint.
- Optionally enables LoRA or 4-bit quantization for parameter-efficient training.
- Trains on an RLDS-formatted robotics dataset with RGB/depth sequences.
- Uses object masks and bounding boxes to supervise object-centric predictions.
- Optimizes a combined loss over bounding boxes, masks, and objectness.
- Periodically saves object-centric weights during training.

Expected data:
- An RLDS dataset under `data_root_dir`
- Samples containing:
  - image tensors
  - segmentation maps
  - object metadata
  - per-object reasoning annotations

How to run:

Single-GPU:
    CUDA_VISIBLE_DEVICES=0 torchrun \
        --standalone \
        --nnodes 1 \
        --nproc-per-node 1 \
        vla-scripts/finetune-1_slotvla_slot-wo-filter.py \
        --vla_path "openvla/openvla-7b" \
        --data_root_dir "/path/to/datasets" \
        --dataset_name "real_data" \
        --run_root_dir "/path/to/checkpoints" \
        --adapter_tmp_dir "/path/to/adapters" \
        --batch_size 16 \
        --horizon_size 36 \
        --number_of_slots 16 \
        --learning_rate 1e-5 \
        --grad_accumulation_steps 1 \
        --save_steps 250 \
        --image_aug True \
        --modality_key 1000

Multi-GPU:
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
        --standalone \
        --nnodes 1 \
        --nproc-per-node 4 \
        vla-scripts/finetune-1_slotvla_slot-wo-filter.py \
        --vla_path "openvla/openvla-7b" \
        --data_root_dir "/path/to/datasets" \
        --dataset_name "real_data"

Modality key format:
    [RGB main, RGB wrist, Depth main, Depth wrist]

Examples:
    1000 -> RGB main only
    1100 -> RGB main + RGB wrist
    1111 -> All modalities
    0010 -> Depth main only

Notes:
- Designed for GPU training and distributed execution via `torchrun`.
- This is a research/debugging-oriented training pipeline.
- Some helper functions and debug statements may intentionally interrupt execution
  during development.

Example output:
- Object-centric checkpoint saved as:
  `pretrained_slots-wo-filters_s16.safetensors`

- Logged losses may include:
  - `loss_bbox`
  - `loss_giou`
  - `loss_objectness`
  - `loss_obj_seg`
"""

# from __future__ import annotations

import os
import resource
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import draccus
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import tqdm
from accelerate import PartialState
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from safetensors.torch import load_file, save_file
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

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import (
    OpenVLAForActionPrediction,
    OpenVLAForActionPrediction_SlotAtt,
)
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.losses.training_losses import RobotSSMObjectLossWithTrack
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPredictionV3
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransformV3, RLDSDatasetV3
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics
from scipy.ndimage import center_of_mass


MODALITIES = ("rgb_main", "rgb_wrist", "depth_main", "depth_wrist")

def process_modality_key(modality_key: str) -> tuple[dict[str, bool], int]:
    """Convert a 4-char binary modality string into flags."""
    if len(modality_key) != len(MODALITIES):
        raise ValueError(f"modality_key must have length {len(MODALITIES)}, got {modality_key!r}")
    if any(value not in {"0", "1"} for value in modality_key):
        raise ValueError(f"modality_key must contain only '0' and '1', got {modality_key!r}")

    modality_flags: dict[str, bool] = {}
    modality_num = 0

    for enabled, modality in zip(modality_key, MODALITIES):
        is_on = enabled == "1"
        modality_flags[modality] = is_on
        if is_on:
            modality_num += 1
            print(f"Using {modality} data.")

    return modality_flags, modality_num


def get_linear_module_names(
    module: torch.nn.Module,
    parent_name: str = "",
    exclude_patterns: Sequence[str] = (),
) -> list[str]:
    """Recursively collect names of Linear layers, excluding selected subtrees."""
    names: list[str] = []

    for child_name, child_module in module.named_children():
        full_name = f"{parent_name}.{child_name}" if parent_name else child_name

        if isinstance(child_module, torch.nn.Linear):
            if not any(pattern in full_name for pattern in exclude_patterns):
                names.append(full_name)
        else:
            names.extend(get_linear_module_names(child_module, full_name, exclude_patterns))

    return names


def filter_by_active_objs(reasoning_on_image, batched_active_objs):
    """Keep only object entries that match active language nouns."""
    filtered = []

    for batch_reasoning, active_objs in zip(reasoning_on_image, batched_active_objs):
        batch_filtered = []
        for step_reasoning in batch_reasoning:
            step_filtered = {
                obj_name: obj_value
                for obj_name, obj_value in step_reasoning.items()
                if any(active_obj in obj_name for active_obj in active_objs)
            }
            batch_filtered.append(step_filtered)
        filtered.append(batch_filtered)

    return filtered


def process_interactions(reasoning_on_image):
    """Extract interaction counts per object over the horizon."""
    batched_interactions = []

    for batch_entry in reasoning_on_image:
        horizon_interactions: dict[str, int] = {}
        processed: dict[str, list[int]] = {}

        for horizon_entry in batch_entry:
            for key, datum in horizon_entry.items():
                if key not in horizon_interactions:
                    horizon_interactions[key] = datum[0]

        for horizon_entry in batch_entry:
            for key in horizon_interactions.keys():
                if key in horizon_entry:
                    horizon_interactions[key] = horizon_entry[key][0]
                processed.setdefault(key, []).append(horizon_interactions[key])

        batched_interactions.append(processed)

    return batched_interactions


def process_seg_ids(reasoning_on_image):
    """Build per-object segmentation-ID trajectories over the horizon."""
    batched_seg_ids = []

    for batch_entry in reasoning_on_image:
        object_keys: list[str] = []

        for horizon_entry in batch_entry:
            for key in horizon_entry:
                if key not in object_keys:
                    object_keys.append(key)

        seg_ids = {
            key: [horizon_entry[key][1] if key in horizon_entry else -1 for horizon_entry in batch_entry]
            for key in object_keys
        }
        batched_seg_ids.append(seg_ids)

    return batched_seg_ids


def process_bboxes(reasoning_on_image):
    """Build per-object bbox trajectories over the horizon."""
    batched_bboxes = []

    for batch_entry in reasoning_on_image:
        object_bboxes: dict[str, list[np.ndarray]] = {}

        for horizon_entry in batch_entry:
            for key in horizon_entry.keys():
                object_bboxes.setdefault(key, [])

        for horizon_entry in batch_entry:
            for key in object_bboxes.keys():
                if key in horizon_entry:
                    bbox = np.asarray(horizon_entry[key][2], dtype=np.float64)
                    bbox = np.concatenate([bbox, np.array([1.0], dtype=np.float64)])
                else:
                    bbox = np.zeros(5, dtype=np.float64)
                object_bboxes[key].append(bbox)

        batched_bboxes.append(object_bboxes)

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
    output_dir: Path = Path(".")                                    # Directory for the object-centric safetensors checkpoint
    load_slot_path: Optional[Path] = None                            # Optional object-centric initialization checkpoint

    # Fine-tuning Parameters
    batch_size: int = 16                                            # Fine-tuning batch size
    horizon_size: int = 16                                          # Temporal window size
    number_of_slots: int = 16                                       # The number of slots of the model
    max_steps: int = 31000                                          # Max number of fine-tuning steps
    save_steps: int = 5000                                          # Interval for checkpoint saving
    learning_rate: float = 2e-5                                     # Fine-tuning learning rate
    grad_accumulation_steps: int = 1                                # Gradient accumulation steps
    image_aug: bool = True                                          # Whether to train with image augmentations
    shuffle_buffer_size: int = 10_000                               # RLDS shuffle buffer; large windows consume substantial RAM
    # LoRA Arguments
    use_lora: bool = False                                           # Whether to use LoRA fine-tuning
    lora_rank: int = 32                                             # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                                       # Dropout applied to LoRA weights
    use_quantization: bool = False                                  # Whether to 4-bit quantize VLA for LoRA fine-tuning
                                                                    #   => CAUTION: Reduces memory but hurts performance

    run_id_note: Optional[str] = None                               # Extra note for logging, Weights & Biases

    # fmt: on
    modality_key: str = "1000"     # Binary characters for using RGB main, RGB wrist, depth main, depth wrist
    resume: bool = False                                           # Whether to resume old run
    debug: bool = False                                           # Whether to use debug mode


@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    print(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`")

    # [Validate] Ensure GPU Available & Set Device / Distributed Context
    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()

    # Configure Unique Experiment ID & Log Directory
    exp_id = (
        f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
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
    exp_id += "--multiview"
    exp_id += cfg.modality_key + "--object"

    # Start =>> Build Directories
    run_dir, adapter_dir = cfg.run_root_dir / exp_id, cfg.adapter_tmp_dir / exp_id
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(adapter_dir, exist_ok=True)
    os.makedirs(cfg.output_dir, exist_ok=True)

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
    vla.vision_backbone.requires_grad_(False)
    vla.projector.requires_grad_(False)
    vla.language_model.requires_grad_(False)
    # Stage 1 calls only the vision backbone through `get_obj_slots`; keeping the
    # 7B language model and projector would waste most of the resident GPU memory.
    del vla.language_model
    del vla.projector
    vla = OpenVLAForActionPrediction_SlotAtt(model=vla, number_of_slots=cfg.number_of_slots)

    if cfg.load_slot_path is None:
        if distributed_state.is_main_process:
            print("No slot checkpoint provided; training object-centric modules from their default initialization.")
    elif cfg.load_slot_path.is_file():
        if distributed_state.is_main_process:
            print(f"Loading initial slot weights from {cfg.load_slot_path}")
        incompatible_keys = vla.load_state_dict(load_file(str(cfg.load_slot_path), device="cpu"), strict=False)
        if distributed_state.is_main_process:
            print(
                "Loaded slot checkpoint with "
                f"{len(incompatible_keys.missing_keys)} missing and "
                f"{len(incompatible_keys.unexpected_keys)} unexpected keys."
            )
    elif distributed_state.is_main_process:
        print(
            f"Slot checkpoint not found at {cfg.load_slot_path}; "
            "training object-centric modules from their default initialization."
        )

    vla.object_centric_tokenizer.requires_grad_(True)
    vla.object_centric_bbox_head.requires_grad_(True)
    vla.object_centric_mask_head.requires_grad_(True)

    # Device Placement =>> note that BitsAndBytes automatically handles for quantized training
    if cfg.use_quantization:
        vla = prepare_model_for_kbit_training(vla)
    else:
        vla = vla.to(device_id)

    # [LoRA] Wrap Model w/ PEFT `LoraConfig` =>> by default we set `target_modules=all-linear`
    if cfg.use_lora:
        # Get linear module names not in backbone
        linear_module_names = get_linear_module_names(vla, exclude_patterns=("vision_backbone",))

        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules=linear_module_names,
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()

    # Wrap VLA in PyTorch DDP Wrapper for Multi-GPU Training
    vla = DDP(vla, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)

    # Create Optimizer =>> note that we default to a simple constant learning rate!
    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

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

    modality_flags, _ = process_modality_key(cfg.modality_key)
    unsupported_modalities = [
        name for name, enabled in modality_flags.items()
        if enabled and name != "rgb_main"
    ]
    if not modality_flags["rgb_main"] or unsupported_modalities:
        raise ValueError(
            "Stage 1 SlotAtt currently consumes main RGB only; use --modality_key 1000. "
            f"Unsupported enabled modalities: {unsupported_modalities}"
        )

    window_size = cfg.horizon_size
    batch_transform = RLDSBatchTransformV3(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    )
    vla_dataset = RLDSDatasetV3(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        # resize_resolution=tuple(vla.module.config.image_sizes),
        resize_resolution=(224, 224),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
        window_size=window_size,
        load_camera_views=("primary",),
        load_depth=False,
        cropping=False
    )
    
    # [Important] Save Dataset Statistics =>> used to de-normalize actions for inference!
    if distributed_state.is_main_process:
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

    dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
    )

    # Train!
    object_loss = RobotSSMObjectLossWithTrack(interact_required=False)
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()
        torch.cuda.reset_peak_memory_stats(device_id)
        completed_steps = 0
        for batch_idx, batch in enumerate(dataloader):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pixel_values = batch["all_pixel_values"].to(torch.bfloat16).to(device_id)

                reasoning_on_image=batch["reasoning_on_image"]

                all_seg_ids = process_seg_ids(reasoning_on_image)               # [B x H] lists of dicts of seg ids
                all_bboxes = process_bboxes(reasoning_on_image)                 # [B x H] lists of dicts of bboxes [cx cy w h, obj] in [0,1]

                # all_images = denormalize_from_DINO(batch["all_pixel_values"][:,:,:3,:,:]).to(device_id)    
                all_obj_keys = [list(all_bboxes[b].keys()) for b in range(len(all_bboxes))]
                all_obj_bboxes = [[horizon_bboxes[key] for key in all_obj_keys[i]] for i,horizon_bboxes in enumerate(all_bboxes)]    # [B x O x H x 5]
                all_obj_segids = [[horizon_seg_ids[key] for key in all_obj_keys[i]] for i,horizon_seg_ids in enumerate(all_seg_ids)] # [B x O x H]
                all_pixel_seg_values = batch["all_pixel_seg_values"]
                bz, horizon, H, W = all_pixel_seg_values.shape
                # downsampling the seg values
                all_pixel_seg_values = all_pixel_seg_values.reshape(bz*horizon, 1, H, W)
                all_pixel_seg_values = F.interpolate(all_pixel_seg_values, size=(64, 64))
                all_pixel_seg_values = all_pixel_seg_values.reshape(bz, horizon, 64, 64)
                # print(all_bboxes.shape))
                # print(all_pixel_seg_values.shape)
                # print(torch.unique(all_pixel_seg_values))
                # visualize_and_save_labeled_masks(all_pixel_seg_values, all_seg_ids, images=all_images)

                with torch.autocast("cuda", dtype=torch.bfloat16):
                    # # sample values
                    # num_objs = 8
                    # pred_bboxes = torch.rand(cfg.batch_size, window_size, num_objs, 5)
                    # pred_segs = torch.rand(cfg.batch_size, window_size, num_objs, 224, 224)
                    # print('sample_pred_bboxes', pred_bboxes.shape)
                    # print('sample_pred_segs', pred_segs.shape)
                    # pred_bboxes = pred_bboxes.permute(0, 2, 1, 3)
                    # pred_segs = pred_segs.permute(0, 2, 1, 3, 4)
                    outputs = vla.module.get_obj_slots(pixel_values)
                    object_preds = {'bboxes': outputs['bboxes'].permute(0, 2, 1, 3),
                                    'segs': outputs['masks'].permute(0, 2, 1, 3, 4)}
                    object_gts = object_loss.preprocess(
                        all_obj_bboxes,
                        all_obj_segids, all_pixel_seg_values,
                        None,
                    )
                    losses, _ = object_loss(object_preds, object_gts)
                    updated_losses = object_loss.update_weights(
                        losses,
                        weights={
                            'loss_bbox': 1,
                            'loss_giou': 1,
                            'loss_objectness': 1,
                            'loss_obj_seg': 1,
                        }
                    )

                loss = sum(updated_losses.values())

            # Normalize loss to account for gradient accumulation
            normalized_loss = loss / cfg.grad_accumulation_steps

            # Backward pass
            normalized_loss.backward()

            # Optimizer Step
            is_optimizer_step = (batch_idx + 1) % cfg.grad_accumulation_steps == 0
            if is_optimizer_step:
                optimizer.step()
                optimizer.zero_grad()
                completed_steps += 1
                progress.update()

            if distributed_state.is_main_process and is_optimizer_step and completed_steps % 10 == 0:
                loss_metrics = {
                    name: value.detach().float().item()
                    for name, value in updated_losses.items()
                }
                mask_iou = 1.0 - object_loss.get_jaccard_evaluations().detach().float().item()
                mask_logits = object_preds["segs"].detach().float()
                print(
                    {
                        "step": completed_steps,
                        **loss_metrics,
                        "loss_total": sum(loss_metrics.values()),
                        "mask_iou": mask_iou,
                        "mask_logit_min": mask_logits.min().item(),
                        "mask_logit_max": mask_logits.max().item(),
                        "gpu_allocated_gib": torch.cuda.memory_allocated(device_id) / 2**30,
                        "gpu_reserved_gib": torch.cuda.memory_reserved(device_id) / 2**30,
                        "gpu_peak_gib": torch.cuda.max_memory_allocated(device_id) / 2**30,
                        "host_peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
                    }
                )
                torch.cuda.reset_peak_memory_stats(device_id)

            # Save Model Checkpoint =>> by default, only keeps the latest checkpoint, continually overwriting it!
            should_save = (
                completed_steps > 0
                and is_optimizer_step
                and (completed_steps % cfg.save_steps == 0 or completed_steps >= cfg.max_steps)
            )
            if should_save:

                if distributed_state.is_main_process:
                    checkpoint_path = cfg.output_dir / f"pretrained_slots-wo-filters_s{cfg.number_of_slots}.safetensors"
                    print(f"Saving model checkpoint for step {completed_steps} to {checkpoint_path}")

                    merged_state_dict = vla.module.state_dict()
                    object_centric_state_dict = {k: v for k, v in merged_state_dict.items() if 'object_centric' in k}
                    save_file(object_centric_state_dict, str(checkpoint_path))

                # Wait for processor and adapter weights to be saved by main process
                dist.barrier()
            
            if completed_steps >= cfg.max_steps:
                break

if __name__ == "__main__":
    finetune()
