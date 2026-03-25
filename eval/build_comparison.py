#!/usr/bin/env python3
"""
Build comparison table, radar chart and bar charts from summary_*.json.
MMLU: average only (mmlu|0), no per-topic breakdown.
"""

import json
import os
from pathlib import Path

# Metrics to extract (results key -> display label)
METRIC_KEYS = [
    ("hellaswag|0", "HellaSwag"),
    ("arc:challenge|0", "ARC-Challenge"),
    ("arc:easy|0", "ARC-Easy"),
    ("winogrande|0", "WinoGrande"),
    ("truthfulqa:mc|0", "TruthfulQA-MC"),
    ("mmlu|0", "MMLU (avg)"),
]

# Short names for models
MODEL_LABELS = {
    "openai-community-gpt2": "GPT-2",
    "EleutherAI-gpt-neo-125m": "GPT-Neo 125M",
    "local-sllm": "Local",
    "EleutherAI-pythia-160m-deduped": "Pythia 160M",
    "HuggingFaceTB-SmolLM2-135M": "SmolLM2 135M",
}

SUMMARY_FILES = [
    "summary_gpt2.json",
    "summary_gpt-neo-125m.json",
    "summary_local.json",
    "summary_pythia-160m-deduped.json",
    "summary_SmolLM2-135M.json",
]


def load_metrics(base_dir: Path):
    """Load metrics from all summary files. Returns list of (model_short_name, metrics_dict)."""
    rows = []
    for fname in SUMMARY_FILES:
        path = base_dir / fname
        if not path.exists():
            print(f"Skip (not found): {path}")
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        model_name = data["config_general"]["model_name"]
        short = MODEL_LABELS.get(model_name, model_name)
        results = data["results"]
        metrics = {}
        for key, label in METRIC_KEYS:
            if key in results and "acc" in results[key]:
                metrics[label] = round(results[key]["acc"] * 100, 2)
            else:
                metrics[label] = None
        rows.append((short, metrics))
    return rows


def write_table(rows, base_dir: Path):
    """Write comparison table as CSV and Markdown."""
    labels = [r[1].keys() for r in rows]
    cols = list(METRIC_KEYS[i][1] for i in range(len(METRIC_KEYS)))
    csv_path = base_dir / "comparison_table.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("Model," + ",".join(cols) + "\n")
        for short, metrics in rows:
            vals = [str(metrics.get(c, "")) for c in cols]
            f.write(f"{short}," + ",".join(vals) + "\n")
    print(f"Written: {csv_path}")

    md_path = base_dir / "comparison_table.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| Model | " + " | ".join(cols) + " |\n")
        f.write("|" + "---|" * (len(cols) + 1) + "\n")
        for short, metrics in rows:
            vals = [str(metrics.get(c, "")) for c in cols]
            f.write("| " + short + " | " + " | ".join(vals) + " |\n")
    print(f"Written: {md_path}")


def plot_radar_and_bars(rows, base_dir: Path):
    """Plot radar chart and bar charts. Requires matplotlib."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not found. Install with: pip install matplotlib")
        return

    cols = [m[1] for m in METRIC_KEYS]
    n_metrics = len(cols)
    model_names = [r[0] for r in rows]
    data = np.array([[r[1].get(c) or 0 for c in cols] for r in rows])

    # Radar: one polygon per model
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))
    colors = plt.cm.Set1(np.linspace(0, 1, len(model_names)))
    for i, (name, vals) in enumerate(zip(model_names, data)):
        vals_closed = vals.tolist() + [vals[0]]
        ax.plot(angles, vals_closed, "o-", linewidth=2, label=name, color=colors[i])
        ax.fill(angles, vals_closed, alpha=0.15, color=colors[i])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cols, size=9)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"])
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.0), fontsize=9)
    ax.set_title("Benchmark comparison (accuracy %)", pad=20)
    plt.tight_layout()
    radar_path = base_dir / "comparison_radar.png"
    plt.savefig(radar_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Written: {radar_path}")

    # Bar charts: one subplot per metric
    n_models = len(model_names)
    x = np.arange(n_models)
    width = 0.75

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    for idx, (ax, col) in enumerate(zip(axes, cols)):
        vals = [r[1].get(col) or 0 for r in rows]
        bars = ax.bar(x, vals, width, color=colors)
        ax.set_ylabel("Acc %")
        ax.set_title(col)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha="right")
        ax.set_ylim(0, 100)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{v:.1f}", ha="center", va="bottom", fontsize=8, rotation=0)
    plt.suptitle("Benchmark comparison by model", y=1.02)
    plt.tight_layout()
    bar_path = base_dir / "comparison_bars.png"
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Written: {bar_path}")


def main():
    base_dir = Path(__file__).resolve().parent
    rows = load_metrics(base_dir)
    if not rows:
        print("No data loaded.")
        return
    write_table(rows, base_dir)
    plot_radar_and_bars(rows, base_dir)


if __name__ == "__main__":
    main()
