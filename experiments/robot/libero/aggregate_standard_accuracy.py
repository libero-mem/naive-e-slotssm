import os
import re
import fnmatch
import statistics
from collections import defaultdict

# ======= CONFIG =======
ROOT = "./rollouts"  # root directory that contains all run folders
MODELS = {
    "Object Slot": "node4_object_slot",
    "Relation Slot": "node4_relation_slot",
    "L Object Slot": "node4_eobject_slot",
    "L Relation Slot": "node4_erelation_slot",
}
DATASETS = ["libero_10", "libero_goal", "libero_object", "libero_spatial"]
VARIANTS = [""]  # three runs per (model, dataset) , "-copy", "-rev"
INCLUDE_AVERAGE_COL = True        # set False if you don't want the Average column
# ======================

FNAME_RE = re.compile(r"episode=(\d+)--success=(True|False)--task=(.*)\.mp4$")

def parse_folder(folder_path):
    """Scan one run folder; return dict: task -> success_rate (0..1)."""
    if not os.path.isdir(folder_path):
        return {}

    task_hits = defaultdict(int)
    task_total = defaultdict(int)

    for root, _, files in os.walk(folder_path):
        for f in files:
            if not f.endswith(".mp4"):
                continue
            m = FNAME_RE.search(f)
            if not m:
                continue
            _, success_str, task = m.groups()
            task_total[task] += 1
            task_hits[task] += 1 if success_str == "True" else 0

    return {t: (task_hits[t] / task_total[t]) for t in task_total}

def find_run_folder(model, dataset, variant):
    """
    Find a folder matching: {model}*-{dataset}{variant}, e.g.
      node4_relation_slot-24-libero_goal-copy
    Accept any slot number with wildcard.
    """
    pattern = f"{model}*-{dataset}{variant}"
    for name in os.listdir(ROOT):
        full = os.path.join(ROOT, name)
        if fnmatch.fnmatch(name, pattern) and os.path.isdir(full):
            return full
    return None

def aggregate_model_dataset(model_dirname, dataset):
    """
    Return dict: task -> (mean, std) across available runs for one (model,dataset).
    """
    per_run = []
    for v in VARIANTS:
        folder = find_run_folder(model_dirname, dataset, v)
        if folder:
            per_run.append(parse_folder(folder))

    results = {}
    # union of tasks across runs
    all_tasks = set()
    for r in per_run:
        all_tasks.update(r.keys())

    for task in all_tasks:
        vals = [r[task] for r in per_run if task in r]
        if not vals:
            continue
        mean = statistics.fmean(vals)
        std = statistics.stdev(vals) if len(vals) >= 2 else 0.0
        results[task] = (mean, std)
    return results

def format_val(mean_std):
    """Format 'mean±std' to 3 decimals, or '—' for missing."""
    if mean_std is None:
        return f"{0:.2f}"
    m, s = mean_std
    return f"{m:.2f}"

def print_dataset_table(dataset, per_model_stats):
    """
    per_model_stats: dict with keys 'Object Slot', 'Relation Slot'
      each maps: task_name -> (mean, std)
    Prints a Markdown table:
      LIBERO-XYZ | Task 1 | Task 2 | ... | Average
      Object Slot | m±s | ...
      Relation Slot | m±s | ...
    """
    # derive task list: union across models, sorted by task name (stable)
    all_tasks = set()
    for stats in per_model_stats.values():
        all_tasks.update(stats.keys())
    tasks_sorted = sorted(all_tasks)

    # prepare mapping Task i -> underlying name
    # (we only show "Task i" in headers)
    headers = [dataset.replace("_", " ").upper()]
    headers.extend([f"Task {i+1}" for i in range(len(tasks_sorted))])
    if INCLUDE_AVERAGE_COL:
        headers.append("Average")

    # header row + separator
    print("\n" + " | ".join(headers))
    print(" | ".join(["---"] * len(headers)))

    # each model row
    for model_label in ["Object Slot", "Relation Slot", "L Object Slot", "L Relation Slot"]:
        stats = per_model_stats.get(model_label, {})
        row = [model_label]
        vals_for_avg = []
        for tname in tasks_sorted:
            ms = stats.get(tname, None)
            row.append(format_val(ms))
            if ms is not None:
                vals_for_avg.append(ms[0])  # mean only for dataset-level average
        if INCLUDE_AVERAGE_COL:
            avg = statistics.fmean(vals_for_avg) if vals_for_avg else 0.0
            row.append(f"{avg:.2f}")
        print(" | ".join(row))

# ===== MAIN: compute and print one table per dataset =====
for dataset in DATASETS:
    per_model_stats = {}
    for model_label, model_dirname in MODELS.items():
        per_model_stats[model_label] = aggregate_model_dataset(model_dirname, dataset)
    # Pretty title like "Table 1", "Table 2", ...
    idx = DATASETS.index(dataset) + 1
    print(f"\nTable {idx}")
    print_dataset_table(dataset=dataset.replace("libero_", "LIBERO-"),
                        per_model_stats=per_model_stats)
