#!/bin/bash

# Configuration
# batch_size 8: good speed for loglikelihood; generation tasks use same batch (slower but OK)
# CLASSIFICATION_ONLY: set to 1 to skip gsm8k and truthfulqa:gen (faster, classification tasks only)
MODEL_PATH=${1:-"model.pt"}
TOKENIZER_PATH=${2:-"/Users/levosadchi/Desktop/transformer4/data/tokenizer/tokenizer.json"}
DEVICE=${3:-"mps"}
BATCH_SIZE=${4:-"8"}
OUTPUT_DIR=${5:-"./results_full"}
CLASSIFICATION_ONLY=${6:-"0"}

# Comprehensive benchmark list (Standard suites)
# ORDER: Fast loglikelihood tasks FIRST, slow generation tasks (gsm8k, truthfulqa:gen) LAST
# This shows progress quickly and avoids 3h wait on first task
TASKS="hellaswag,arc:challenge,arc:easy,winogrande,truthfulqa:mc,mmlu:abstract_algebra,mmlu:anatomy,mmlu:astronomy,mmlu:business_ethics,mmlu:clinical_knowledge,mmlu:college_biology,mmlu:college_chemistry,mmlu:college_computer_science,mmlu:college_mathematics,mmlu:college_medicine,mmlu:college_physics,mmlu:computer_security,mmlu:conceptual_physics,mmlu:econometrics,mmlu:electrical_engineering,mmlu:elementary_mathematics,mmlu:formal_logic,mmlu:global_facts,mmlu:high_school_biology,mmlu:high_school_chemistry,mmlu:high_school_computer_science,mmlu:high_school_european_history,mmlu:high_school_geography,mmlu:high_school_government_and_politics,mmlu:high_school_macroeconomics,mmlu:high_school_mathematics,mmlu:high_school_microeconomics,mmlu:high_school_physics,mmlu:high_school_psychology,mmlu:high_school_statistics,mmlu:high_school_us_history,mmlu:high_school_world_history,mmlu:human_aging,mmlu:human_sexuality,mmlu:international_law,mmlu:jurisprudence,mmlu:logical_fallacies,mmlu:machine_learning,mmlu:management,mmlu:marketing,mmlu:medical_genetics,mmlu:miscellaneous,mmlu:moral_disputes,mmlu:moral_scenarios,mmlu:nutrition,mmlu:philosophy,mmlu:prehistory,mmlu:professional_accounting,mmlu:professional_law,mmlu:professional_medicine,mmlu:professional_psychology,mmlu:public_relations,mmlu:security_studies,mmlu:sociology,mmlu:us_foreign_policy,mmlu:virology,mmlu:world_religions,gsm8k,truthfulqa:gen"

echo "=========================================================="
echo "Starting FULL evaluation of local model: $MODEL_PATH"
echo "Device: $DEVICE | Batch Size: $BATCH_SIZE"
echo "Output Directory: $OUTPUT_DIR"
[ "$CLASSIFICATION_ONLY" = "1" ] && echo "Mode: classification only (no gsm8k, truthfulqa:gen)"
echo "Tasks: GSM8K, HellaSwag, ARC, Winogrande, TruthfulQA, ALL MMLU"
echo "=========================================================="

# Build extra args
EXTRA_ARGS="--sequential_tasks"
[ "$CLASSIFICATION_ONLY" = "1" ] && EXTRA_ARGS="$EXTRA_ARGS --classification_only"

# Run evaluation without sample limit (max_samples=None)
python evaluate_local.py \
    --model_path "$MODEL_PATH" \
    --tokenizer_name "$TOKENIZER_PATH" \
    --tasks "$TASKS" \
    --max_samples -1 \
    --batch_size "$BATCH_SIZE" \
    --device "$DEVICE" \
    --output_dir "$OUTPUT_DIR" \
    $EXTRA_ARGS

echo "=========================================================="
echo "FULL Evaluation finished. Results saved to $OUTPUT_DIR"
echo "=========================================================="
