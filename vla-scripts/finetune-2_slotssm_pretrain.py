"""
Training script for an OpenVLA-based robot vision-language model with
object-centric and temporal slot supervision.

This script:
- Loads a pretrained OpenVLA model and processor from HuggingFace.
- Freezes the main vision backbone, projector, and language model.
- Wraps the base model with a custom SlotSSM-based object-centric module.
- Loads pretrained object-centric weights from a .safetensors checkpoint.
- Optionally enables LoRA or 4-bit quantization for parameter-efficient training.
- Trains on primary-camera RGB from an RLDS-formatted robotics dataset.
- Uses language instructions, object masks, bounding boxes, segmentation labels,
  and per-object interaction annotations to supervise object-centric predictions.
- Builds temporal supervision over backward and forward horizon windows to
  reconstruct past slot states and predict future slot dynamics.
- Conditions the causal SlotSSM on the global task and matched per-slot subgoals.
- Optimizes temporal slot reconstruction while reporting frozen object-head
  bounding-box, mask, and objectness metrics.
- Periodically saves object-centric weights during training.

Expected data:
- An RLDS dataset under `data_root_dir`
- Samples containing:
  - image tensors
  - segmentation maps
  - object metadata
  - language instructions
  - per-object reasoning and interaction annotations

How to run:

Single-GPU:
    CUDA_VISIBLE_DEVICES=0 torchrun \
        --standalone \
        --nnodes 1 \
        --nproc-per-node 1 \
        vla-scripts/finetune-2_slotssm_pretrain.py \
        --vla_path "openvla/openvla-7b" \
        --data_root_dir "/path/to/datasets" \
        --dataset_name "droid_wipe" \
        --run_root_dir "/path/to/checkpoints" \
        --adapter_tmp_dir "/path/to/adapters" \
        --batch_size 16 \
        --horizon_size 32 \
        --bwd_steps 31 \
        --number_of_slots 16 \
        --learning_rate 2e-5 \
        --grad_accumulation_steps 1 \
        --save_steps 5000 \
        --image_aug True \
        --modality_key 1000 \
        --load_slot_path "/path/to/pretrained_slots.safetensors"

Multi-GPU:
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
        --standalone \
        --nnodes 1 \
        --nproc-per-node 4 \
        vla-scripts/finetune-2_slotssm_pretrain.py \
        --vla_path "openvla/openvla-7b" \
        --data_root_dir "/path/to/datasets" \
        --dataset_name "droid_wipe"

Notes:
- Designed for GPU training and distributed execution via `torchrun`.
- The current SlotSSM implementation accepts primary RGB only (`modality_key=1000`).
- This is a research/debugging-oriented training pipeline.
- Temporal supervision uses neighboring slot embeddings across the horizon
  to learn both backward reconstruction and forward slot prediction.
- Some helper functions and debug statements may intentionally interrupt
  execution during development.

Example output:
- Object-centric checkpoint saved as:
  `object_centric_bwd{bwd_steps}.safetensors`

- Logged losses may include:
  - `loss_bbox`
  - `loss_giou`
  - `loss_objectness`
  - `loss_obj_seg`
  - `loss_reconstruction`
"""

import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import draccus
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import tqdm
import wandb
from accelerate import PartialState
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from safetensors.torch import load_file, save_file
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from scipy.ndimage import center_of_mass

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction, OpenVLAForActionPrediction_SlotSSM
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.losses.training_losses import RobotSSMObjectLossWithTrack
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPredictionV3
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransformV3, RLDSDatasetV3
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


MODALITIES = ("rgb_main", "rgb_wrist", "depth_main", "depth_wrist")


def process_modality_key(modality_key: str) -> tuple[dict[str, bool], int]:
    if len(modality_key) != 4 or any(c not in "01" for c in modality_key):
        raise ValueError(f"modality_key must be a 4-char binary string, got {modality_key!r}")

    modality_flags = {name: modality_key[i] == "1" for i, name in enumerate(MODALITIES)}
    modality_num = sum(modality_flags.values())
    return modality_flags, modality_num

def get_linear_module_names(
    module: torch.nn.Module,
    parent_name: str = "",
    exclude_patterns: tuple[str, ...] = (),
) -> list[str]:
    names: list[str] = []
    for child_name, child in module.named_children():
        full_name = f"{parent_name}.{child_name}" if parent_name else child_name
        if isinstance(child, torch.nn.Linear):
            if not any(pat in full_name for pat in exclude_patterns):
                names.append(full_name)
        else:
            names.extend(get_linear_module_names(child, full_name, exclude_patterns))
    return names

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

                # Save image
                plt.figure(figsize=(4, 4))
                plt.imshow(colored)
                if binary_mask.any():
                    y, x = center_of_mass(binary_mask)
                    plt.text(
                        x,
                        y,
                        obj_name,
                        fontsize=8,
                        color="white",
                        bbox={"facecolor": "black", "alpha": 0.5, "pad": 1},
                    )
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
    load_slot_path: Optional[Path] = None                            # Stage-1 object-centric checkpoint
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
    # LoRA Arguments
    use_lora: bool = False                                           # Whether to use LoRA fine-tuning
    lora_rank: int = 32                                             # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                                       # Dropout applied to LoRA weights
    use_quantization: bool = False                                  # Whether to 4-bit quantize VLA for LoRA fine-tuning
                                                                    #   => CAUTION: Reduces memory but hurts performance

    run_id_note: Optional[str] = None                               # Optional suffix for the experiment ID
    use_wandb: bool = True                                          # Log the main process to Weights & Biases
    wandb_entity: str = "nhat"                                      # W&B entity
    wandb_project: str = "libero-mem"                               # W&B project
    log_steps: int = 10                                             # Console and W&B logging interval

    # fmt: on
    modality_key: str = "1000"     # Binary characters for using RGB main, RGB wrist, depth main, depth wrist
    debug: bool = False            # Whether to use a small batch for smoke testing


def get_slot_specific_subgoals(
    outputs,
    object_token_num,
    horizon,
    object_gts,
    all_obj_itrn_cnts,
    cnt_embedding_table,
    object_loss,
    object_centric_text_encoder,
    cache_path: Path,
):
    bz = outputs["visual_tokens"].shape[0]
    device = outputs["visual_tokens"].device
    # Get the object interaction count embeddings
    all_obj_itrn_cnts_mapped = []
    new_key = False
    if "0" not in cnt_embedding_table:
        cnt_embedding_table["0"] = (
            object_centric_text_encoder.encode_simple(["0"], device)
            .detach()
            .reshape(-1)
        )
        new_key = True

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
        cpu_cache = {key: value.detach().cpu() for key, value in cnt_embedding_table.items()}
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cpu_cache, cache_path)

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
                slot_specific_cnt[s] = zero_embedding.unsqueeze(0).expand(horizon, -1)

        # slot cnts
        slot_specific_cnt = torch.stack(slot_specific_cnt)
        # slot_specific_cnt += _general_clip_embeddings
        subgoal_states.append(slot_specific_cnt)

    subgoal_states = torch.stack(subgoal_states)
    subgoal_states = subgoal_states.permute([0, 2, 1, 3]) # 'b o t d -> b t o d'
    return subgoal_states, indices

def horizon_projection(
    visual_tokens: torch.Tensor,   # [B, H, O, D]
    skip_steps: int = 0,
    backward_steps: int = 1,
    forward_steps: int = 0,
):
    """
    Project tokens from the horizon axis into a fixed window around each step.

    Returns:
        proj_tokens: [B, H, O, W, D]
        proj_mask:   [B, H, O, W]  (bool)

    Notes:
      - W = backward_steps + forward_steps
      - We include offsets [-backward_steps, ..., -1, 1, ..., forward_steps]
        (the current step 0 is excluded).
      - Mask is 1 where the indexed step exists and < H - skip_steps, else 0.
    """
    B, H, O, D = visual_tokens.shape
    device = visual_tokens.device

    # Build the offset window (exclude current step 0)
    offs_past   = torch.arange(-backward_steps, 0, device=device)
    offs_future = torch.arange(1, forward_steps + 1, device=device)
    offsets = torch.cat([offs_past, offs_future], dim=0)  # [W]
    W = offsets.numel()

    if W == 0:
        empty_tokens = visual_tokens.new_zeros(B, H, O, 0, D)
        empty_mask   = visual_tokens.new_zeros(B, H, O, 0, dtype=torch.bool)
        return empty_tokens, empty_mask

    # Indices into the horizon dimension for each step
    base = torch.arange(H, device=device).unsqueeze(1)     # [H, 1]
    idx  = base + offsets.unsqueeze(0)                     # [H, W]

    # Valid if within [0, H-skip_steps)
    valid = (idx >= 0) & (idx < H - skip_steps)            # [H, W]
    idx_clamped = idx.clamp(0, max(H - 1, 0))              # safe for gather

    # Gather: replacing the horizon dim with [H, W]
    gathered = visual_tokens[:, idx_clamped, :, :]         # [B, H, W, O, D]
    proj_tokens = gathered.permute(0, 1, 3, 2, 4).contiguous()  # [B, H, O, W, D]

    # Broadcast mask to [B, H, O, W]
    proj_mask = valid.unsqueeze(0).unsqueeze(2).expand(B, H, O, W)

    return proj_tokens, proj_mask


@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    print(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`")

    # [Validate] Ensure GPU Available & Set Device / Distributed Context
    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    if cfg.grad_accumulation_steps < 1:
        raise ValueError("grad_accumulation_steps must be at least 1")
    if cfg.max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    if cfg.save_steps < 1:
        raise ValueError("save_steps must be at least 1")
    if cfg.log_steps < 1:
        raise ValueError("log_steps must be at least 1")
    if cfg.horizon_size != cfg.bwd_steps + cfg.fwd_steps + 1:
        raise ValueError(
            "horizon_size must equal bwd_steps + fwd_steps + 1; "
            f"got {cfg.horizon_size} != {cfg.bwd_steps} + {cfg.fwd_steps} + 1"
        )

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
    exp_id += f"{cfg.modality_key}"

    # Start =>> Build Directories
    run_dir, adapter_dir = cfg.run_root_dir / exp_id, cfg.adapter_tmp_dir / exp_id
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(adapter_dir, exist_ok=True)

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
    del vla.language_model
    vla = OpenVLAForActionPrediction_SlotSSM(
        base_model=vla,
        number_of_slots=cfg.number_of_slots,
        backward_step=cfg.bwd_steps,
        forward_step=cfg.fwd_steps,
    )
    
    if cfg.load_slot_path is None:
        raise ValueError(
            "Stage 2 requires a Stage-1 object-centric checkpoint; "
            "provide --load_slot_path /path/to/checkpoint.safetensors"
        )
    if not cfg.load_slot_path.is_file():
        raise FileNotFoundError(f"Stage-1 checkpoint not found: {cfg.load_slot_path}")
    slot_state_dict = load_file(str(cfg.load_slot_path), device="cpu")
    incompatible_keys = vla.load_state_dict(slot_state_dict, strict=False)
    stage1_prefixes = (
        "object_centric_tokenizer.",
        "object_centric_bbox_head.",
        "object_centric_mask_head.",
    )
    absent_stage1_prefixes = [
        prefix
        for prefix in stage1_prefixes
        if not any(key.startswith(prefix) for key in slot_state_dict)
    ]
    missing_stage1_keys = [
        key
        for key in incompatible_keys.missing_keys
        if key.startswith(stage1_prefixes)
    ]
    if absent_stage1_prefixes or missing_stage1_keys:
        raise RuntimeError(
            "The Stage-1 checkpoint is incomplete. "
            f"Absent module prefixes: {absent_stage1_prefixes}; "
            f"missing model keys: {missing_stage1_keys}"
        )

    if distributed_state.is_main_process:
        missing_key_groups = Counter(
            next(
                (
                    prefix
                    for prefix in (
                        "base_model.",
                        "clip_model.",
                        "object_centric_pre_ssm_fusion.",
                        "object_centric_ssm.",
                        "object_centric_latent_decoder.",
                    )
                    if key.startswith(prefix)
                ),
                key.split(".", 1)[0] + ".",
            )
            for key in incompatible_keys.missing_keys
        )
        print(
            f"Loaded all {len(slot_state_dict)} Stage-1 tensors from {cfg.load_slot_path}. "
            "Missing keys belong to modules intentionally absent from Stage 1: "
            f"{dict(sorted(missing_key_groups.items()))}. "
            f"Unexpected keys: {len(incompatible_keys.unexpected_keys)}."
        )
        if cfg.debug:
            print("Full missing-key list:", incompatible_keys.missing_keys)
    vla.object_centric_tokenizer.requires_grad_(False)
    vla.object_centric_bbox_head.requires_grad_(False)
    vla.object_centric_mask_head.requires_grad_(False)

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
        print("Debug mode uses batch_size=2")
        cfg.batch_size = 2

    modality_flags, _ = process_modality_key(cfg.modality_key)
    unsupported_modalities = [
        name
        for name, enabled in modality_flags.items()
        if enabled and name != "rgb_main"
    ]
    if not modality_flags["rgb_main"] or unsupported_modalities:
        raise ValueError(
            "Stage 2 SlotSSM currently consumes primary RGB only; "
            "use --modality_key 1000. "
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
        cropping=False,
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

    wandb_run = None
    if distributed_state.is_main_process and cfg.use_wandb:
        wandb_config = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(cfg).items()
        }
        wandb_run = wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=f"slotssm+{exp_id}",
            dir=str(run_dir),
            config=wandb_config,
        )
        wandb_run.summary["trainable_parameters"] = sum(
            parameter.numel() for parameter in trainable_params
        )
        print(f"W&B run: {wandb_run.url}")

    # Train!
    object_loss = RobotSSMObjectLossWithTrack()
    cnt_embedding_table = {}
    if cfg.interaction_cache_path.is_file():
        cnt_embedding_table = torch.load(
            cfg.interaction_cache_path,
            map_location="cpu",
            weights_only=True,
        )
        for key, value in cnt_embedding_table.items():
            cnt_embedding_table[key] = value.reshape(-1).to(device_id)

    with tqdm.tqdm(
        total=cfg.max_steps,
        desc="SlotSSM",
        dynamic_ncols=True,
        disable=not distributed_state.is_main_process,
    ) as progress:
        vla.train()
        optimizer.zero_grad()
        completed_steps = 0
        for batch_idx, batch in enumerate(dataloader):

            with torch.autocast("cuda", dtype=torch.bfloat16):                
                pixel_values = batch["all_pixel_values"].to(torch.bfloat16).to(device_id)
                reasoning_on_image=batch["reasoning_on_image"]

                all_interaction_cnts = process_interactions(reasoning_on_image) # [B x H] lists of dicts of interactions
                all_seg_ids = process_seg_ids(reasoning_on_image)               # [B x H] lists of dicts of seg ids
                all_bboxes = process_bboxes(reasoning_on_image)                 # [B x H] lists of dicts of bboxes [cx cy w h, obj] in [0,1]

                all_obj_keys = [list(all_bboxes[b].keys()) for b in range(len(all_bboxes))]
                all_obj_bboxes = [[horizon_bboxes[key] for key in all_obj_keys[i]] for i,horizon_bboxes in enumerate(all_bboxes)]    # [B x O x H x 5]
                all_obj_segids = [[horizon_seg_ids[key] for key in all_obj_keys[i]] for i,horizon_seg_ids in enumerate(all_seg_ids)] # [B x O x H]
                all_obj_itrn_cnts = [[horizon_itrn_cnts[key] for key in all_obj_keys[i]] for i,horizon_itrn_cnts in enumerate(all_interaction_cnts)] # [B x O x H]

                all_pixel_seg_values = batch["all_pixel_seg_values"]
                bz, horizon, H, W = all_pixel_seg_values.shape
                # downsampling the seg values
                all_pixel_seg_values = all_pixel_seg_values.reshape(bz*horizon, 1, H, W)
                all_pixel_seg_values = F.interpolate(
                    all_pixel_seg_values.float(),
                    size=(64, 64),
                    mode="nearest",
                )
                all_pixel_seg_values = all_pixel_seg_values.reshape(bz, horizon, 64, 64)
                object_gts = object_loss.preprocess(
                    all_obj_bboxes,
                    all_obj_segids, all_pixel_seg_values,
                    all_obj_itrn_cnts
                )

                with torch.autocast("cuda", dtype=torch.bfloat16):
                    texts = ["robot " + task["instruction"] for task in batch["tasks"]]
                    outputs = vla.module.get_obj_slots(pixel_values, texts)

                    ############################################
                    subgoal_states, indices = get_slot_specific_subgoals(
                        outputs,
                        vla.module.object_token_num, horizon,
                        object_gts,
                        all_obj_itrn_cnts,
                        cnt_embedding_table,
                        object_loss,
                        vla.module.object_centric_text_encoder,
                        cfg.interaction_cache_path,
                    )
                    original_slots = outputs['visual_tokens']

                    # can filter temporally of the next slots
                    outputs = vla.module.get_slot_dynamics(
                        original_slots,
                        subgoal_states,
                        outputs['patch_features'], 
                        outputs['texts'], 
                        outputs['texts_attn'],
                        outputs,
                        redundant_steps=0 # no need to ssm-calculate these steps into the future
                    )

                    # Evaluate the decoder's final (+fwd_steps) token against the
                    # corresponding future object target. These heads are detached
                    # in the model and therefore provide metrics only.
                    losses = {}
                    if cfg.fwd_steps > 0:
                        future_object_preds = {
                            "bboxes": outputs["nxt_bboxes"]
                            .permute(0, 2, 1, 3)[:, :, :-cfg.fwd_steps],
                            "segs": outputs["nxt_masks"]
                            .permute(0, 2, 1, 3, 4)[:, :, :-cfg.fwd_steps],
                        }
                        future_object_gts = {
                            key: [sample[:, cfg.fwd_steps:] for sample in value]
                            for key, value in object_gts.items()
                        }
                        losses, _ = object_loss(
                            future_object_preds,
                            future_object_gts,
                            indices=indices,
                        )
                    
                    nxt_tokens = outputs['nxt_tokens']       # bz, horizon, o, projected_past, d
                    visual_tokens = outputs['visual_tokens']  # bz, horizon, o, d
                    # visual_tokens: [B, H, O, D]  (from outputs)
                    # nxt_tokens:    [B, H, O, W, D]  (model's predictions)
                    gt_tokens, vis_mask = horizon_projection(
                        visual_tokens,
                        skip_steps=0,
                        backward_steps=vla.module.backward_steps,
                        forward_steps=vla.module.forward_steps
                    )
                    if nxt_tokens.shape != gt_tokens.shape:
                        raise RuntimeError(
                            "Temporal prediction and target shapes do not match: "
                            f"{tuple(nxt_tokens.shape)} vs {tuple(gt_tokens.shape)}"
                        )
                    if not vis_mask.any():
                        raise RuntimeError("Temporal reconstruction has no valid target positions")

                    # The visual mask denotes where the horizon index can predict past states,
                    #   for horizon_index = 0 the model predict the furthest future step
                    #   for horizon_index = 1 the model predict the furthest future step - 1
                    #   ...
                    #   for horizon_index = horizon-1 the model predict the past (furthest) step
                    # Same shape as nxt_tokens now → index with the same mask
                    loss_reconstruction = F.mse_loss(
                        nxt_tokens[vis_mask],          # [N_selected, D]
                        gt_tokens.detach()[vis_mask],  # [N_selected, D]
                        reduction='mean'
                    )
                    losses['loss_reconstruction'] = loss_reconstruction
                    
                    updated_losses = object_loss.update_weights(
                        losses,
                        weights={
                            'loss_reconstruction': 1,
                            'loss_bbox': 1,
                            'loss_giou': 1,
                            'loss_objectness': 1,
                            'loss_obj_seg': 10
                        }
                    )

                loss = loss_reconstruction

            # Normalize loss to account for gradient accumulation
            normalized_loss = loss / cfg.grad_accumulation_steps

            # Backward pass
            normalized_loss.backward()

            is_optimizer_step = (batch_idx + 1) % cfg.grad_accumulation_steps == 0
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
                    torch.linalg.vector_norm(torch.stack(parameter_grad_norms), ord=2)
                    if parameter_grad_norms
                    else torch.zeros((), device=device_id)
                )

            if is_optimizer_step:
                optimizer.step()
                completed_steps += 1
                progress.update()

            if should_log:
                valid_predictions = nxt_tokens.detach()[vis_mask]
                valid_targets = gt_tokens.detach()[vis_mask]
                metrics = {
                    "train/reconstruction_loss": loss_reconstruction.detach().float().item(),
                    "train/learning_rate": optimizer.param_groups[0]["lr"],
                    "train/grad_norm": grad_norm.item(),
                    "tokens/pred_mean": valid_predictions.float().mean().item(),
                    "tokens/pred_std": valid_predictions.float().std().item(),
                    "tokens/target_mean": valid_targets.float().mean().item(),
                    "tokens/target_std": valid_targets.float().std().item(),
                    "system/gpu_allocated_gib": torch.cuda.memory_allocated(device_id) / 2**30,
                    "system/gpu_reserved_gib": torch.cuda.memory_reserved(device_id) / 2**30,
                    "system/gpu_peak_gib": torch.cuda.max_memory_allocated(device_id) / 2**30,
                }
                metrics.update({
                    f"future_metrics/{name}": value.detach().float().item()
                    for name, value in updated_losses.items()
                    if name != "loss_reconstruction"
                })
                if cfg.fwd_steps > 0:
                    metrics["future_metrics/jaccard_loss"] = (
                        object_loss.get_jaccard_evaluations().detach().float().item()
                    )

                console_metrics = {
                    "step": completed_steps,
                    "recon": round(metrics["train/reconstruction_loss"], 6),
                    "grad_norm": round(metrics["train/grad_norm"], 4),
                    "pred_std": round(metrics["tokens/pred_std"], 4),
                    "target_std": round(metrics["tokens/target_std"], 4),
                    "gpu_gib": round(metrics["system/gpu_allocated_gib"], 2),
                }
                progress.write(str(console_metrics))
                progress.set_postfix(
                    loss=f"{metrics['train/reconstruction_loss']:.4f}",
                    grad=f"{metrics['train/grad_norm']:.3f}",
                    gpu=f"{metrics['system/gpu_allocated_gib']:.1f}G",
                )
                if wandb_run is not None:
                    wandb_run.log(metrics, step=completed_steps)
                torch.cuda.reset_peak_memory_stats(device_id)

            if is_optimizer_step:
                optimizer.zero_grad()

            # Save Model Checkpoint =>> by default, only keeps the latest checkpoint, continually overwriting it!
            should_save = (
                is_optimizer_step
                and completed_steps > 0
                and (completed_steps % cfg.save_steps == 0 or completed_steps >= cfg.max_steps)
            )
            if should_save:

                if distributed_state.is_main_process:
                    print(f"Saving Model Checkpoint for Step {completed_steps} at", adapter_dir)

                    merged_state_dict = vla.module.state_dict()
                    object_centric_state_dict = {k: v for k, v in merged_state_dict.items() if 'object_centric' in k}
                    checkpoint_path = (
                        adapter_dir
                        / f"object_centric_bwd{cfg.bwd_steps}_fwd{cfg.fwd_steps}.safetensors"
                    )
                    save_file(object_centric_state_dict, str(checkpoint_path))
                    if wandb_run is not None:
                        wandb_run.summary["latest_checkpoint"] = str(checkpoint_path)
                        wandb_run.summary["latest_checkpoint_step"] = completed_steps

                # Wait for processor and adapter weights to be saved by main process
                distributed_state.wait_for_everyone()

            if completed_steps >= cfg.max_steps:
                break

    if wandb_run is not None:
        wandb_run.finish()

if __name__ == "__main__":
    finetune()
