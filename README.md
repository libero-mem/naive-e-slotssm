# Naive Embodied-SlotSSM Reimplementation

Unofficial reimplementation of **Embodied-SlotSSM**, an object-centric temporal memory modeling framework for robotic manipulation, accepted at **AAAI 2026**.

This repository is based on the paper:

> **Rethinking Progression of Memory State in Robotic Manipulation: An Object-Centric Perspective**

Project page:  
https://libero-mem.github.io/

Paper:  
https://arxiv.org/abs/2511.11478

LIBERO-Mem dataset:  
https://github.com/libero-mem/libero-mem  
https://huggingface.co/datasets/libero-mem/LIBERO-Mem

This repository contains PyTorch training scripts for:
- object-centric slot learning
- object-centric temporal memory learning with SlotSSM
- object-centric embodied action prediction

using OpenVLA-based robot vision-language-action backbones.

---

# Overview

Most vision-language-action (VLA) models rely on dense visual tokenization, producing hundreds of visual tokens per frame.

Embodied-SlotSSM instead models manipulation through compact object-centric memory states. The core idea is to represent the scene using object slots and then model how these object states progress over time.

This repository focuses on:

- object-centric slot representations
- object-level temporal memory states
- SlotSSM-based memory progression
- language-conditioned object interaction modeling
- compact visuomotor representations for action prediction

The training pipeline is organized into three stages:

1. **Object-Centric Slot Learning**
2. **SlotSSM Temporal Memory Pretraining**
3. **Embodied Action Prediction**

---

# Pipeline Overview

```text
RGB-D Observations
      |
      v
Stage 1: Object-Centric Slot Learning
      |
      v
pretrained_slots-wo-filters_s16.safetensors
      |
      v
Stage 2: SlotSSM Temporal Memory Pretraining
      |
      v
object_centric_bwd{bwd_steps}_fwd{fwd_steps}.safetensors
      |
      v
Stage 3: Embodied Action Prediction
      |
      v
object_centric_bwd{bwd_steps}+fwd{fwd_steps}_actionable.safetensors
      |
      v
Continuous Robot Actions
```

---

# Repository Structure

```text
vla-scripts/
├── finetune-1_slotvla_slot-wo-filter.py
├── finetune-2_slotssm_pretrain.py
└── finetune-3_slotssm_to_action.py
```

---

# Training Pipeline

## Stage 1 — Object-Centric Slot Learning

Script:

```text
finetune-1_slotvla_slot-wo-filter.py
```

This stage trains object-centric visual representations from RGB-D trajectories and object annotations.

It learns:

- object-centric slot representations
- segmentation-aware object slots
- object bounding box prediction
- object mask prediction
- objectness prediction
- interactable object prediction

This stage does **not** train temporal memory dynamics or embodied action prediction. Its purpose is to produce reusable object-centric slot weights.

### Input

Stage 1 uses:

- RGB / RGB-D observations
- segmentation maps
- object bounding boxes
- object metadata
- language instructions
- per-object interaction labels

### Output

Stage 1 produces a pretrained object-centric slot checkpoint:

```text
pretrained_slots-wo-filters_s16.safetensors
```

---

## Stage 2 — SlotSSM Temporal Memory Pretraining

Script:

```text
finetune-2_slotssm_pretrain.py
```

This stage trains temporal memory progression over object-centric slots.

It loads the pretrained object-centric slot weights from Stage 1 and trains a SlotSSM module to model how object states evolve across a horizon window.

It learns:

- object-centric temporal memory representations
- backward slot reconstruction
- forward slot forecasting
- temporal object dynamics
- memory-state progression over object slots

Unlike Stage 3, this stage does **not** decode robot actions.

### Input

Stage 2 uses:

- pretrained object-centric slot weights from Stage 1
- RGB / RGB-D trajectory windows
- object masks
- object bounding boxes
- object interaction annotations
- language instructions

### Output

Stage 2 produces a SlotSSM temporal memory checkpoint:

```text
object_centric_bwd{bwd_steps}_fwd{fwd_steps}.safetensors
```

For example:

```text
object_centric_bwd25_fwd6.safetensors
```

---

## Stage 3 — Embodied Action Prediction

Script:

```text
finetune-3_slotssm_to_action.py
```

This stage trains the embodied policy.

It loads the temporal SlotSSM checkpoint from Stage 2 and trains an action decoder that predicts continuous robot actions from object-centric memory states.

It learns:

- object-centric embodied policy representations
- continuous robot action prediction
- language-conditioned action decoding
- action prediction from fused visual, textual, and slot-memory features

Most object-centric and temporal SlotSSM components are frozen during this stage. The trainable components are mainly used to adapt the model for embodied action prediction.

### Input

Stage 3 uses:

- pretrained SlotSSM memory checkpoint from Stage 2
- RGB / RGB-D trajectory windows
- language instructions
- object interaction annotations
- tokenized action labels
- continuous action supervision

### Output

Stage 3 produces an actionable embodied policy checkpoint:

```text
object_centric_bwd{bwd_steps}+fwd{fwd_steps}_actionable.safetensors
```

For example:

```text
object_centric_bwd25+fwd6_actionable.safetensors
```

---

# Checkpoint Flow

```text
Stage 1:
    finetune-1_slotvla_slot-wo-filter.py
        |
        | saves
        v
    pretrained_slots-wo-filters_s16.safetensors

Stage 2:
    finetune-2_slotssm_pretrain.py
        |
        | loads
        v
    pretrained_slots-wo-filters_s16.safetensors
        |
        | saves
        v
    object_centric_bwd25_fwd6.safetensors

Stage 3:
    finetune-3_slotssm_to_action.py
        |
        | loads
        v
    object_centric_bwd25_fwd6.safetensors
        |
        | saves
        v
    object_centric_bwd25+fwd6_actionable.safetensors
```

---

# Installation

## Recommended Environment

- Python 3.10+
- CUDA 11.8+
- PyTorch 2.1+

## Install Dependencies

```bash
pip install torch torchvision torchaudio
pip install transformers accelerate peft safetensors draccus tqdm matplotlib scipy
```

Optional logging:

```bash
pip install wandb tensorboard
```

---

# Dataset

This repository uses the **LIBERO-Mem** dataset.

LIBERO-Mem extends LIBERO with memory-oriented object-centric supervision, including:

- RGB observations
- depth observations
- object masks
- object bounding boxes
- temporal object tracking
- object interaction annotations
- language instructions
- long-horizon manipulation trajectories

Dataset repository:

```text
https://github.com/libero-mem/libero-mem
```

HuggingFace dataset:

```text
https://huggingface.co/datasets/libero-mem/LIBERO-Mem
```

Expected structure:

```text
<data_root_dir>/
└── <dataset_name>/
    └── 1.0.0/
```

Example:

```text
/path/to/LIBERO-Mem/
└── libero_mem/
    └── 1.0.0/
```

---

# Usage

## Stage 1 — Train Object-Centric Slots

```bash
CUDA_VISIBLE_DEVICES=0 torchrun \
    --standalone \
    --nnodes 1 \
    --nproc-per-node 1 \
    vla-scripts/finetune-1_slotvla_slot-wo-filter.py \
    --vla_path "output_hf_model_openx" \
    --data_root_dir "/path/to/LIBERO-Mem" \
    --dataset_name libero_mem \
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
```

Expected output:

```text
pretrained_slots-wo-filters_s16.safetensors
```

---

## Stage 2 — Train SlotSSM Temporal Memory

Before running Stage 2, make sure the Stage 1 checkpoint is available:

```text
pretrained_slots-wo-filters_s16.safetensors
```

Depending on your local script configuration, update the Stage 2 checkpoint path so that it loads the Stage 1 output.

```bash
CUDA_VISIBLE_DEVICES=0 torchrun \
    --standalone \
    --nnodes 1 \
    --nproc-per-node 1 \
    vla-scripts/finetune-2_slotssm_pretrain.py \
    --vla_path "output_hf_model_openx" \
    --data_root_dir "/path/to/LIBERO-Mem" \
    --dataset_name libero_mem \
    --run_root_dir "/path/to/checkpoints" \
    --adapter_tmp_dir "/path/to/adapters" \
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
```

Expected output:

```text
object_centric_bwd25_fwd6.safetensors
```

---

## Stage 3 — Train Embodied Action Policy

Before running Stage 3, make sure the Stage 2 checkpoint is available:

```text
object_centric_bwd25_fwd6.safetensors
```

Depending on your local script configuration, update the Stage 3 checkpoint path so that it loads the Stage 2 output.

```bash
CUDA_VISIBLE_DEVICES=0 torchrun \
    --standalone \
    --nnodes 1 \
    --nproc-per-node 1 \
    vla-scripts/finetune-3_slotssm_to_action.py \
    --vla_path "output_hf_model_openx" \
    --data_root_dir "/path/to/LIBERO-Mem" \
    --dataset_name libero_mem \
    --run_root_dir "/path/to/checkpoints" \
    --adapter_tmp_dir "/path/to/adapters" \
    --lora_rank 64 \
    --grad_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --number_of_slots 16 \
    --horizon_size 32 \
    --bwd_steps 25 \
    --fwd_steps 6 \
    --batch_size 6 \
    --image_aug True \
    --save_steps 1000 \
    --modality_key 1000
```

Expected output:

```text
object_centric_bwd25+fwd6_actionable.safetensors
```

---

# Multi-GPU Training

All scripts support distributed execution with `torchrun`.

Example:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --standalone \
    --nnodes 1 \
    --nproc-per-node 4 \
    vla-scripts/finetune-2_slotssm_pretrain.py \
    --vla_path "output_hf_model_openx" \
    --data_root_dir "/path/to/LIBERO-Mem" \
    --dataset_name libero_mem \
    --run_root_dir "/path/to/checkpoints" \
    --adapter_tmp_dir "/path/to/adapters" \
    --batch_size 16 \
    --horizon_size 32 \
    --bwd_steps 25 \
    --fwd_steps 6 \
    --number_of_slots 16 \
    --learning_rate 2e-5 \
    --save_steps 5000 \
    --image_aug True \
    --modality_key 1000
```

---

# Modality Key

The `modality_key` controls which observation modalities are used.

Format:

```text
[RGB main, RGB wrist, Depth main, Depth wrist]
```

Examples:

```text
1000 -> RGB main only
1100 -> RGB main + RGB wrist
1111 -> all modalities
0010 -> depth main only
```

---

# Important Notes

- This repository is a research reimplementation and may differ from the original codebase of FPT Software.
- The training scripts are experimental and may contain debugging utilities.
- Stage 2 learns object-centric temporal memory states; Stage 3 uses those states for action prediction.
- Stage 3 does not train explicit temporal reconstruction labels; it uses horizon-based context and continuous action supervision.
- Checkpoint paths may be hard-coded in some scripts and should be updated before training.

---

# Outputs

Typical outputs include:

```text
pretrained_slots-wo-filters_s16.safetensors
object_centric_bwd25_fwd6.safetensors
object_centric_bwd25+fwd6_actionable.safetensors
```

Depending on the script and configuration, outputs are saved under:

```text
<adapter_tmp_dir>/<experiment_name>/
```

or:

```text
<run_root_dir>/<experiment_name>/
```

---

# Citation

If you find this repository useful, please cite:

```bibtex
@inproceedings{chung2026rethinking,
  title={Rethinking Progression of Memory State in Robotic Manipulation: An Object-Centric Perspective},
  author={Chung, Nhat and Hanyu, Taisei and Nguyen, Toan and Le, Huy and Bumgarner, Frederick and Nguyen, Duy Minh Ho and Vo, Khoa and Yamazaki, Kashu and Rainwater, Chase and Kieu, Tung and Nguyen, Anh and Le, Ngan},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)},
  year={2026}
}
```

```bibtex
@inproceedings{hanyu2026slotvla,
  title={SlotVLA: Towards Modeling of Object-Relation Representations in Robotic Manipulation},
  author={Hanyu, Taisei and Chung, Nhat and Le, Huy and Nguyen, Toan and Ikebe, Yuki and Gunderman, Anthony and Nguyen, Duy Minh Ho and Vo, Khoa and Kieu, Tung and Yamazaki, Kashu and Rainwater, Chase and Nguyen, Anh and Le, Ngan},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2026}
}
