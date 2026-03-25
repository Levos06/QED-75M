#!/usr/bin/env python3
"""
Merge multiple lighteval result JSON files (from sequential task runs) into one combined file.

Usage:
  python merge_results.py [results_dir]
  python merge_results.py ./results_full
  python merge_results.py ./results_full/results/local-sllm

Output: merged_results.json in the same directory as the input files.
"""

import argparse
import json
import math
from pathlib import Path


def find_result_files(results_dir: Path) -> list[Path]:
    """Find all results_*.json files, recursively or in model subdirs."""
    files = list(results_dir.rglob("results_*.json"))
    if not files and (results_dir / "results").exists():
        # Try results/model_name/
        for sub in (results_dir / "results").iterdir():
            if sub.is_dir():
                files.extend(sub.glob("results_*.json"))
    return sorted(files)


def merge_results(files: list[Path]) -> dict:
    merged = {
        "config_general": None,
        "results": {},
        "config_tasks": {},
        "summary_tasks": {},
        "summary_general": None,
        "versions": {},
    }
    task_accs = []
    task_stderrs = []

    for f in files:
        with open(f, encoding="utf-8") as fp:
            data = json.load(fp)

        if merged["config_general"] is None:
            merged["config_general"] = data.get("config_general", {})
            merged["summary_general"] = data.get("summary_general", {})
        merged["versions"].update(data.get("versions", {}))

        for key, val in data.get("results", {}).items():
            if key == "all":
                continue
            if key in merged["results"]:
                continue  # Keep first occurrence
            merged["results"][key] = val
            if "acc" in val:
                task_accs.append(val["acc"])
                task_stderrs.append(val.get("acc_stderr", 0))

        for key, val in data.get("config_tasks", {}).items():
            if key not in merged["config_tasks"]:
                merged["config_tasks"][key] = val

        for key, val in data.get("summary_tasks", {}).items():
            if key not in merged["summary_tasks"]:
                merged["summary_tasks"][key] = val

    # Overall "all" metric: mean of task accuracies
    if task_accs:
        mean_acc = sum(task_accs) / len(task_accs)
        # Pooled stderr: sqrt(mean(stderr^2)) as approximation
        if task_stderrs:
            mean_stderr = math.sqrt(sum(s**2 for s in task_stderrs) / len(task_stderrs))
        else:
            mean_stderr = 0.0
        merged["results"]["all"] = {"acc": mean_acc, "acc_stderr": mean_stderr}

    # MMLU average: mean over all mmlu:* subjects only
    mmlu_accs = []
    mmlu_stderrs = []
    for key, val in merged["results"].items():
        if key.startswith("mmlu:") and "acc" in val:
            mmlu_accs.append(val["acc"])
            mmlu_stderrs.append(val.get("acc_stderr", 0))
    if mmlu_accs:
        mmlu_mean = sum(mmlu_accs) / len(mmlu_accs)
        mmlu_stderr = math.sqrt(sum(s**2 for s in mmlu_stderrs) / len(mmlu_stderrs)) if mmlu_stderrs else 0.0
        merged["results"]["mmlu|0"] = {"acc": mmlu_mean, "acc_stderr": mmlu_stderr}

    return merged


def main():
    parser = argparse.ArgumentParser(description="Merge lighteval sequential task results into one JSON")
    parser.add_argument("results_dir", type=str, default="./results_full", nargs="?")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output file path")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Directory not found: {results_dir}")
        return 1

    files = find_result_files(results_dir)
    if not files:
        print(f"No results_*.json files found in {results_dir}")
        return 1

    print(f"Found {len(files)} result files")
    merged = merge_results(files)

    if args.output:
        out_path = Path(args.output)
    else:
        # Place next to the first file
        out_path = files[0].parent / "merged_results.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(merged, fp, indent=2, ensure_ascii=False)

    # Exclude synthetic keys from task count
    n_tasks = len([k for k in merged["results"] if k not in ("all", "mmlu|0")])
    print(f"Merged {n_tasks} tasks -> {out_path}")
    if "all" in merged["results"]:
        acc = merged["results"]["all"]["acc"]
        stderr = merged["results"]["all"]["acc_stderr"]
        print(f"Overall acc: {acc:.4f} ± {stderr:.4f}")
    if "mmlu|0" in merged["results"]:
        acc = merged["results"]["mmlu|0"]["acc"]
        stderr = merged["results"]["mmlu|0"]["acc_stderr"]
        print(f"MMLU (avg):  {acc:.4f} ± {stderr:.4f}")
    return 0


if __name__ == "__main__":
    exit(main())
