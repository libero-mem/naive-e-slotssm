import argparse
import importlib.util
import json
import os
import re
import sys
from collections import deque
from pathlib import Path

import h5py
import numpy as np
import torch
from peft import PeftModel
from safetensors.torch import load_file
from scipy.optimize import linear_sum_assignment
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import EmbodiedDecodedSlotSSM, OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor


def import_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_upstream_rollout_tools(libero_mem_root):
    modeling = sys.modules["prismatic.extern.hf.modeling_prismatic"]
    modeling.EmbodiedDecodedSlotNaive = EmbodiedDecodedSlotSSM
    modeling.EmbodiedDecodedSlotSSMv2 = EmbodiedDecodedSlotSSM
    sys.path.insert(0, str(libero_mem_root / "scripts"))
    sys.path.insert(0, str(libero_mem_root / "libero"))
    import mem_6_run_evaluation_env_pred as rollout_tools

    binder_module = import_module_from_path(
        "slotssm_state_binder",
        PROJECT_ROOT / "experiments/robot/libero/slotssm_state_binder.py",
    )
    return rollout_tools, binder_module.OnlineSubgoalStateBinder


def load_policy(args, device):
    adapter_dir = args.adapter_dir
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    base_model = AutoModelForVision2Seq.from_pretrained(
        args.base_model,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    ).to(device)
    base_model = PeftModel.from_pretrained(base_model, adapter_dir).merge_and_unload()
    model = EmbodiedDecodedSlotSSM(
        base_model=base_model,
        number_of_slots=args.number_of_slots,
        backward_step=args.bwd_steps,
        forward_step=args.fwd_steps,
    )

    stage2_state = load_file(str(args.stage2_checkpoint), device="cpu")
    stage2_result = model.load_state_dict(stage2_state, strict=False)
    if stage2_result.unexpected_keys:
        raise RuntimeError(f"Unexpected Stage-2 tensors: {stage2_result.unexpected_keys[:10]}")

    stage3_state = load_file(str(args.stage3_checkpoint), device="cpu")
    stage3_result = model.load_state_dict(stage3_state, strict=False)
    required_action_groups = (
        "object_centric_action_slot_fusion.",
        "object_centric_action_slot_projector.",
        "object_centric_action_text_projector.",
        "object_centric_action_head.",
    )
    missing_groups = [
        group
        for group in required_action_groups
        if not any(key.startswith(group) for key in stage3_state)
    ]
    if missing_groups or stage3_result.unexpected_keys:
        raise RuntimeError(
            f"Incompatible Stage-3 checkpoint; missing={missing_groups}, "
            f"unexpected={stage3_result.unexpected_keys[:10]}"
        )

    with (adapter_dir / "dataset_statistics.json").open() as stats_file:
        model.norm_stats = json.load(stats_file)
    model = model.to(device).eval()
    model.requires_grad_(False)
    processor = AutoProcessor.from_pretrained(adapter_dir, trust_remote_code=False)
    count_embeddings = torch.load(args.interaction_cache, map_location="cpu")
    for key, value in count_embeddings.items():
        count_embeddings[key] = value.reshape(-1).to(device)

    return model, processor, count_embeddings


def raw_initial_state_episodes(task_file, mode, episodes_per_task, natural_sort_key):
    with h5py.File(task_file, "r") as handle:
        data = handle["data"]
        demo_keys = sorted(data.keys(), key=natural_sort_key)
        if mode == "unseen":
            selected_keys = demo_keys[max(0, len(demo_keys) - episodes_per_task) :]
        else:
            selected_keys = demo_keys[:episodes_per_task]
        episodes = []
        for demo_key in selected_keys:
            demo = data[demo_key]
            initial_state = demo["init_state"][()] if "init_state" in demo else demo["states"][()][0]
            episodes.append((demo_key, initial_state, len(demo["actions"])))
    return episodes


def image_batch(rollout_tools, processor, obs, device):
    image = rollout_tools.get_libero_image(obs, 256)
    image = image[:, ::-1, :]
    batch = rollout_tools.process_inputs(processor, image, device)
    batch["pixel_values"] = batch["pixel_values"].to(device=device, dtype=torch.bfloat16)
    return image, batch


def update_tracker(recorder, env, obs, action, task_description, rollout_tools, binder):
    visuals = recorder.record(
        obs,
        action,
        seg_to_mask_fn=rollout_tools.seg_to_mask,
        get_bbox_fn=rollout_tools.get_bbox,
    )
    recorder.monitor_gripper_collision(env)
    env._check_success(inc=True)
    recorder.update_subgoal_state(env.get_satisfied_subgoals(task_description))
    recorder.update_collision_state_with_objs(visuals)
    binder.update(recorder)


def predict_action(model, batch, pixel_history, binder, task_text):
    device = batch["input_ids"].device
    pixel_values = torch.stack([frame[0] for frame in pixel_history]).unsqueeze(0)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        object_outputs = model.get_obj_slots(
            pixel_values,
            [f"robot {task_text}"],
            batch_vision_backbone=True,
            compute_masks=False,
        )
        subgoal_states = binder.bind(object_outputs)
        action_chunk = model(
            object_outputs=object_outputs,
            subgoal_states=subgoal_states,
            llama_input_ids=batch["input_ids"].to(device),
            llama_attention_mask=batch["attention_mask"].to(device),
            action_start_step=pixel_values.shape[1] - 1,
        )
    return action_chunk[0, 0, 0].float().cpu().numpy(), object_outputs


def denormalize_actions(model, unnorm_key, normalized_actions):
    stats = model.norm_stats[unnorm_key]["action"]
    mask = np.asarray(stats.get("mask", np.ones_like(stats["q01"], dtype=bool)))
    low = np.asarray(stats["q01"])
    high = np.asarray(stats["q99"])
    return np.where(mask, 0.5 * (normalized_actions + 1) * (high - low) + low, normalized_actions)


def grounding_step_metrics(predicted_boxes, gt_objects, previous_assignments, metrics):
    predicted_boxes = predicted_boxes.detach().float().cpu().numpy()
    gt_labels = list(gt_objects)
    gt_boxes = np.asarray([gt_objects[label][1] for label in gt_labels], dtype=np.float32)
    gt_boxes = np.column_stack(
        [gt_boxes[:, 0] + gt_boxes[:, 2] / 2, gt_boxes[:, 1] + gt_boxes[:, 3] / 2, gt_boxes[:, 2:]]
    )
    candidate_indices = np.flatnonzero(predicted_boxes[:, 4] >= 0.5)
    if not len(candidate_indices):
        metrics["ground_truth_objects"] += len(gt_labels)
        metrics["false_negatives"] += len(gt_labels)
        return previous_assignments
    candidate_boxes = predicted_boxes[candidate_indices, :4]
    pred_xyxy = np.column_stack(
        [
            candidate_boxes[:, 0] - candidate_boxes[:, 2] / 2,
            candidate_boxes[:, 1] - candidate_boxes[:, 3] / 2,
            candidate_boxes[:, 0] + candidate_boxes[:, 2] / 2,
            candidate_boxes[:, 1] + candidate_boxes[:, 3] / 2,
        ]
    )
    gt_xyxy = np.column_stack(
        [
            gt_boxes[:, 0] - gt_boxes[:, 2] / 2,
            gt_boxes[:, 1] - gt_boxes[:, 3] / 2,
            gt_boxes[:, 0] + gt_boxes[:, 2] / 2,
            gt_boxes[:, 1] + gt_boxes[:, 3] / 2,
        ]
    )
    top_left = np.maximum(pred_xyxy[:, None, :2], gt_xyxy[None, :, :2])
    bottom_right = np.minimum(pred_xyxy[:, None, 2:], gt_xyxy[None, :, 2:])
    intersection_wh = np.maximum(bottom_right - top_left, 0)
    intersection = intersection_wh[..., 0] * intersection_wh[..., 1]
    pred_area = np.prod(np.maximum(pred_xyxy[:, 2:] - pred_xyxy[:, :2], 0), axis=1)
    gt_area = np.prod(np.maximum(gt_xyxy[:, 2:] - gt_xyxy[:, :2], 0), axis=1)
    ious = intersection / np.maximum(pred_area[:, None] + gt_area[None, :] - intersection, 1e-8)
    row_indices, col_indices = linear_sum_assignment(1.0 - ious)
    matched = [(r, c, ious[r, c]) for r, c in zip(row_indices, col_indices) if ious[r, c] >= 0.5]
    matched_slots = {candidate_indices[r] for r, _, _ in matched}
    metrics["ground_truth_objects"] += len(gt_labels)
    metrics["true_positives"] += len(matched)
    metrics["false_positives"] += len(candidate_indices) - len(matched_slots)
    metrics["false_negatives"] += len(gt_labels) - len(matched)
    metrics["matched_iou_sum"] += sum(iou for _, _, iou in matched)
    current_assignments = {}
    for row, col, _ in matched:
        label = gt_labels[col]
        slot_index = int(candidate_indices[row])
        current_assignments[label] = slot_index
        if label in previous_assignments:
            metrics["tracking_transitions"] += 1
            metrics["tracking_id_switches"] += int(previous_assignments[label] != slot_index)
    return current_assignments


def parse_args():
    default_adapter = Path(
        "/mnt/data/data_nhat/LIBERO-Mem/weights/adapters/"
        "openvla-7b+libero_mem+b64+lr-2e-05+lora-r32+dropout-0.0"
        "--image_aug--multiview_bwdfwd1000--fined"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero_mem_repo", type=Path, default=Path.home() / "libero-mem")
    parser.add_argument("--raw_data_dir", type=Path, default=Path("/mnt/data/data_nhat/LIBERO-Mem-Raw"))
    parser.add_argument("--adapter_dir", type=Path, default=default_adapter)
    parser.add_argument(
        "--stage2_checkpoint",
        type=Path,
        default=Path(
            "/mnt/data/data_nhat/LIBERO-Mem/weights/adapters/"
            "openvla-7b+libero_mem+b16+lr-0.0001--image_aug--multiview1000/"
            "object_centric_bwd25_fwd6.safetensors"
        ),
    )
    parser.add_argument("--stage3_checkpoint", type=Path, default=None)
    parser.add_argument(
        "--interaction_cache",
        type=Path,
        default=Path("/mnt/data/data_nhat/LIBERO-Mem/weights/cnt_text_embeddings.pt"),
    )
    parser.add_argument("--base_model", default="openvla/openvla-7b")
    parser.add_argument("--eval_mode", choices=("seen", "unseen"), default="unseen")
    parser.add_argument("--episodes_per_task", type=int, default=20)
    parser.add_argument("--task_id", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/mnt/data/data_nhat/LIBERO-Mem/evaluation/slotssm-raw-unseen"),
    )
    parser.add_argument("--save_videos", action="store_true")
    parser.add_argument("--grounding_metrics", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--number_of_slots", type=int, default=16)
    parser.add_argument("--bwd_steps", type=int, default=25)
    parser.add_argument("--fwd_steps", type=int, default=6)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.stage3_checkpoint is None:
        args.stage3_checkpoint = args.adapter_dir / "object_centric_bwd25_fwd6_actionable.safetensors"
    args.libero_mem_repo = args.libero_mem_repo.resolve()
    args.raw_data_dir = args.raw_data_dir.resolve()
    args.adapter_dir = args.adapter_dir.resolve()
    args.stage2_checkpoint = args.stage2_checkpoint.resolve()
    args.stage3_checkpoint = args.stage3_checkpoint.resolve()
    args.interaction_cache = args.interaction_cache.resolve()
    args.output_dir = args.output_dir.resolve()
    if not args.libero_mem_repo.is_dir() or not args.raw_data_dir.is_dir():
        raise FileNotFoundError("LIBERO-Mem source checkout or raw HDF5 directory is missing")
    if not torch.cuda.is_available():
        raise RuntimeError("LIBERO-Mem SlotSSM evaluation requires a CUDA GPU")

    os.chdir(args.libero_mem_repo / "scripts")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0")
    rollout_tools, binder_class = load_upstream_rollout_tools(args.libero_mem_repo)
    model, processor, count_embeddings = load_policy(args, device)

    from libero.libero import benchmark
    from prismatic.losses.training_losses import RobotSSMObjectLossWithTrack

    task_suite = benchmark.get_benchmark_dict()["libero_mem"]()
    task_ids = range(task_suite.n_tasks) if args.task_id is None else [args.task_id]
    object_loss = RobotSSMObjectLossWithTrack(seg_required=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / f"results_{args.eval_mode}.json"
    if args.resume and results_path.exists():
        with results_path.open() as results_file:
            results = json.load(results_file).get("task_results", {})
    else:
        results = {}
    horizon = args.bwd_steps + args.fwd_steps + 1

    for task_id in task_ids:
        task = task_suite.get_task(task_id)
        task_description = task.language.strip()
        model_task_text = re.sub(r"^\d+\s+", "", task_description).strip()
        task_file = args.raw_data_dir / f"{task.name}_demo.hdf5"
        episodes = raw_initial_state_episodes(
            task_file,
            args.eval_mode,
            args.episodes_per_task,
            rollout_tools.natural_sort_key,
        )
        env, _ = rollout_tools.get_libero_env(task, "llava", resolution=256)
        goal_length = env.get_goal_sequence_len()
        task_results = results.get(task.name, {})
        max_steps = args.max_steps or max(
            rollout_tools.TASK_LENGTHS.get(model_task_text.lower(), 0),
            max(length for _, _, length in episodes),
        )

        for demo_key, initial_state, _ in episodes:
            if demo_key in task_results and (
                not args.grounding_metrics or "grounding" in task_results[demo_key]
            ):
                continue
            env.reset()
            rollout_tools.set_init_state(env, initial_state)
            env.reset_subgoal_progress()
            dummy_action = rollout_tools.get_libero_dummy_action("llava")
            for _ in range(20):
                obs, _, _, _ = env.step(dummy_action)

            recorder = rollout_tools.TrajectoryRecorder(image_resolution=256)
            recorder.reset_object_mappers(env)
            recorder.reset_trackstate()
            binder = binder_class(
                horizon,
                args.number_of_slots,
                object_loss,
                model.object_centric_text_encoder,
                count_embeddings,
            )
            grounding = {
                "ground_truth_objects": 0,
                "true_positives": 0,
                "false_positives": 0,
                "false_negatives": 0,
                "matched_iou_sum": 0.0,
                "tracking_transitions": 0,
                "tracking_id_switches": 0,
            }
            previous_assignments = {}
            update_tracker(
                recorder,
                env,
                obs,
                dummy_action,
                model_task_text,
                rollout_tools,
                binder,
            )
            image, batch = image_batch(rollout_tools, processor, obs, device)
            pixel_history = deque([batch["pixel_values"]] * horizon, maxlen=horizon)
            replay_images = []
            done = False
            last_action = dummy_action

            for _ in range(max_steps):
                replay_images.append(image)
                normalized_action, object_outputs = predict_action(
                    model, batch, pixel_history, binder, model_task_text
                )
                if args.grounding_metrics:
                    previous_assignments = grounding_step_metrics(
                        object_outputs["bboxes"][0, -1],
                        recorder.agentview_boxes[-1],
                        previous_assignments,
                        grounding,
                    )
                action = denormalize_actions(model, "libero_mem", normalized_action)
                action[3:6] = 0.0
                obs, _, done, _ = env.step(action.tolist())
                last_action = action

                update_tracker(
                    recorder,
                    env,
                    obs,
                    last_action,
                    model_task_text,
                    rollout_tools,
                    binder,
                )
                image, batch = image_batch(rollout_tools, processor, obs, device)
                pixel_history.append(batch["pixel_values"])
                satisfied = env.get_satisfied_subgoals(model_task_text)
                done = bool(done or len(satisfied) >= goal_length)
                if done:
                    break

            satisfied = env.get_satisfied_subgoals(model_task_text)
            task_results[demo_key] = {
                "success": bool(done),
                "tiered_success": len(satisfied) / goal_length,
                "unseen_state_labels": sorted(binder.unseen_state_labels),
            }
            if args.grounding_metrics:
                detections = grounding["true_positives"] + grounding["false_positives"]
                task_results[demo_key]["grounding"] = {
                    "precision": grounding["true_positives"] / max(detections, 1),
                    "recall": grounding["true_positives"] / max(grounding["ground_truth_objects"], 1),
                    "mean_iou": grounding["matched_iou_sum"] / max(grounding["true_positives"], 1),
                    "tracking_id_switches": grounding["tracking_id_switches"],
                    "tracking_transitions": grounding["tracking_transitions"],
                    "tracking_id_switch_rate": grounding["tracking_id_switches"] / max(grounding["tracking_transitions"], 1),
                }
            if args.save_videos:
                rollout_tools.save_rollout_video(
                    replay_images,
                    sum(len(items) for items in results.values()) + len(task_results),
                    success=bool(done),
                    task_description=task_description,
                    saved_dir=args.output_dir.name,
                )
            results[task.name] = task_results
            with results_path.open("w") as results_file:
                json.dump(
                    {
                        "eval_mode": args.eval_mode,
                        "episodes_per_task": args.episodes_per_task,
                        "task_results": results,
                    },
                    results_file,
                    indent=2,
                )
            print(
                f"{task.name} {demo_key}: success={bool(done)}, "
                f"tiered={task_results[demo_key]['tiered_success']:.3f}"
            )

        success_rate = sum(item["success"] for item in task_results.values()) / len(task_results)
        print(f"{task.name}: success_rate={success_rate:.3f}")
        env.close()

    print(f"Saved evaluation results to {results_path}")


if __name__ == "__main__":
    main()