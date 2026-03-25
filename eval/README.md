# LLM Evaluation Toolkit

A comprehensive toolkit for evaluating small language models using the [LightEval](https://github.com/huggingface/lighteval) framework.

---

## 📁 Files Overview

| File | Description |
|------|-------------|
| `evaluate_local.py` | Main evaluation script – supports local checkpoints and HuggingFace models |
| `run_eval_local.sh` | Quick evaluation of local model (few tasks, limited samples) |
| `run_full_eval.sh` | Full benchmark suite for local models (57 tasks: MMLU, GSM8K, ARC, HellaSwag, etc.) |
| `run_full_eval_hf.sh` | Full benchmark suite for HuggingFace models (Pythia, GPT-2, SmolLM, etc.) |
| `build_comparison.py` | Generate comparison tables and charts from multiple evaluation results |
| `inspect_model.py` | Inspect model checkpoint structure |
| `merge_results.py` | Merge multiple evaluation result files |

---

## Quick Start

### Prerequisites

```bash
pip install lighteval torch transformers accelerate
```

### Evaluate a Local Model

```bash
# Quick test (100 samples)
./run_eval_local.sh model.pt tokenizer.json "hellaswag,arc:challenge" 100 mps ./results_quick

# Full evaluation (all benchmarks)
./run_full_eval.sh model.pt tokenizer.json mps 8 ./results_full 0
```

### Evaluate a HuggingFace Model

```bash
# Full evaluation on Pythia 160M
./run_full_eval_hf.sh EleutherAI/pythia-160m-deduped mps 8 ./results_pythia 1

# Evaluate GPT-2
./run_full_eval_hf.sh openai-community/gpt2 mps 8 ./results_gpt2 1
```

---

## Command Line Arguments

### `evaluate_local.py`

| Argument | Description | Default |
|----------|-------------|---------|
| `--model_path` | Path to local `.pt` checkpoint | `None` |
| `--model_name` | HuggingFace model ID (overrides `model_path`) | `None` |
| `--tokenizer_name` | Path to tokenizer or HF model ID | `./tokenizer.json` |
| `--tasks` | Comma-separated list of tasks | `gsm8k` |
| `--max_samples` | Max samples per task (for quick testing) | `None` |
| `--batch_size` | Batch size for evaluation | `1` |
| `--output_dir` | Directory to save results | `./results` |
| `--device` | Device to run on (`cpu`, `mps`, `cuda`) | Auto-detect |
| `--sequential_tasks` | Run tasks one-by-one to save memory | `False` |
| `--classification_only` | Skip generative tasks (GSM8K, TruthfulQA:gen) | `False` |

---

## Available Benchmarks

### Core Benchmarks (included in full eval)

| Benchmark | Description | Type |
|-----------|-------------|------|
| **MMLU** (57 subjects) | Multi-task language understanding | Log-likelihood |
| **GSM8K** | Grade school math problems | Generative |
| **HellaSwag** | Common sense reasoning | Log-likelihood |
| **ARC-Challenge** | Science QA | Log-likelihood |
| **ARC-Easy** | Easier science QA | Log-likelihood |
| **WinoGrande** | Coreference resolution | Log-likelihood |
| **TruthfulQA:MC** | Truthfulness multiple choice | Log-likelihood |
| **TruthfulQA:Gen** | Truthfulness generative | Generative |

### Additional Available Tasks

See `registry_keys.txt` in the parent directory for the full list of 300+ tasks including:
- Legal benchmarks (LexGLUE, LegalBench)
- Medical benchmarks (MedQA, PubMedQA)
- Math benchmarks (MATH, AIME)
- Code benchmarks (LiveCodeBench, HumanEval)
- Multilingual tasks (XCOPA, XStoryCloze)

---

## Results Format

After evaluation, results are saved to `output_dir/`:

```
results_full/
├── details_*.json      # Per-sample results (if --save-details)
├── results_*.json      # Aggregated metrics
└── summary_*.json      # Summary for comparison
```

### Example Output (`summary.json`)

```json
{
  "config_general": {
    "model_name": "local-sllm",
    "total_evaluation_time_secondes": "1792.74"
  },
  "results": {
    "hellaswag|0": { "acc": 0.2566, "acc_stderr": 0.0044 },
    "arc:challenge|0": { "acc": 0.2169, "acc_stderr": 0.0120 },
    "mmlu:abstract_algebra|0": { "acc": 0.3131, "acc_stderr": 0.0468 }
  }
}
```

---

## Comparing Models

After running evaluations on multiple models, generate comparison visualizations:

```bash
# Copy summary files to the same directory
cp results_model1/summary.json summary_model1.json
cp results_model2/summary.json summary_model2.json

# Generate comparison table and charts
python build_comparison.py
```

### Output Files

| File | Description |
|------|-------------|
| `comparison_table.md` | Markdown table with accuracy scores |
| `comparison_table.csv` | CSV format for further analysis |
| `comparison_radar.png` | Radar chart comparing all models |
| `comparison_bars.png` | Bar charts for each benchmark |

### Example Comparison Table

| Model | HellaSwag | ARC-Challenge | WinoGrande | TruthfulQA-MC | MMLU (avg) |
|-------|-----------|---------------|------------|---------------|------------|
| GPT-2 | 25.02 | 19.04 | 49.05 | 40.07 | 23.22 |
| Pythia 160M | 24.97 | 19.98 | 49.21 | 44.24 | 23.24 |
| Local | 25.71 | 22.63 | 50.55 | 46.08 | 25.43 |

---

## Advanced Usage

### Custom Task List

Edit the `TASKS` variable in shell scripts or pass directly:

```bash
python evaluate_local.py \
    --model_path model.pt \
    --tasks "mmlu:math,mmlu:physics,gsm8k,hellaswag" \
    --max_samples 500
```

### Memory Optimization

For models that cause OOM errors:

```bash
# Run tasks sequentially (frees memory between tasks)
python evaluate_local.py --sequential_tasks ...

# Reduce batch size
python evaluate_local.py --batch_size 4 ...

# Classification-only mode (skip slow generative tasks)
./run_full_eval.sh model.pt tokenizer.json mps 8 ./results 1
```

### MPS (Apple Silicon) Tips

```bash
# MPS works best with batch_size=8 for log-likelihood tasks
# For generation tasks, reduce to batch_size=4 if OOM

# Clear cache between runs
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
```

---

## Example Workflows

### Workflow 1: Quick Sanity Check

```bash
# Test on 100 samples of 2 tasks
./run_eval_local.sh model.pt tokenizer.json "hellaswag,arc:challenge" 100 mps
```

### Workflow 2: Full Benchmark

```bash
# Run full suite overnight
./run_full_eval.sh model.pt tokenizer.json mps 8 ./results_full 0
```

### Workflow 3: Compare Multiple Models

```bash
# Evaluate all models
./run_full_eval_hf.sh EleutherAI/pythia-160m-deduped mps 8 ./pythia 1
./run_full_eval_hf.sh openai-community/gpt2 mps 8 ./gpt2 1
./run_full_eval.sh model.pt tokenizer.json mps 8 ./local 1

# Generate comparison
python build_comparison.py
```

---

## Notes

1. **Log-likelihood vs Generative**: Tasks are patched to use log-likelihood accuracy (classification) instead of generation for ~20x speedup on MMLU, HellaSwag, ARC, etc.

2. **Sequential Tasks**: Use `--sequential_tasks` to avoid 16GB+ RAM growth when evaluating many tasks.

3. **Classification-Only Mode**: Use `--classification_only` to skip GSM8K and TruthfulQA:gen (generative tasks take ~3 hours).

4. **Tokenizer**: For local models, ensure tokenizer has proper special tokens mapped (`<bos>`, `<eos>`, `<pad>`).

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| OOM on MPS | Reduce `batch_size`, use `--sequential_tasks` |
| Slow evaluation | Use `--classification_only` for quick results |
| Missing tokenizer | Provide explicit `--tokenizer_name` |
| Task not found | Check `registry_keys.txt` for valid task names |

---

## License

This toolkit uses LightEval under the Apache 2.0 license.

---

## Acknowledgments

- [LightEval](https://github.com/huggingface/lighteval) – HuggingFace
- [Transformers](https://github.com/huggingface/transformers) – HuggingFace
- [EleutherAI](https://github.com/EleutherAI) – Pythia models
