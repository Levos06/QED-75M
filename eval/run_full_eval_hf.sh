#!/bin/bash
# Full evaluation for HuggingFace models (Pythia, GPT-2, etc.)
# Same setup as run_full_eval.sh: classification_only, sequential_tasks, same task list

MODEL_NAME=${1:-"EleutherAI/pythia-160m-deduped"}
DEVICE=${2:-"mps"}
BATCH_SIZE=${3:-"8"}
OUTPUT_DIR=${4:-"./results_full_hf"}
CLASSIFICATION_ONLY=${5:-"1"}

TASKS="hellaswag,arc:challenge,arc:easy,winogrande,truthfulqa:mc,mmlu:abstract_algebra,mmlu:anatomy,mmlu:astronomy,mmlu:business_ethics,mmlu:clinical_knowledge,mmlu:college_biology,mmlu:college_chemistry,mmlu:college_computer_science,mmlu:college_mathematics,mmlu:college_medicine,mmlu:college_physics,mmlu:computer_security,mmlu:conceptual_physics,mmlu:econometrics,mmlu:electrical_engineering,mmlu:elementary_mathematics,mmlu:formal_logic,mmlu:global_facts,mmlu:high_school_biology,mmlu:high_school_chemistry,mmlu:high_school_computer_science,mmlu:high_school_european_history,mmlu:high_school_geography,mmlu:high_school_government_and_politics,mmlu:high_school_macroeconomics,mmlu:high_school_mathematics,mmlu:high_school_microeconomics,mmlu:high_school_physics,mmlu:high_school_psychology,mmlu:high_school_statistics,mmlu:high_school_us_history,mmlu:high_school_world_history,mmlu:human_aging,mmlu:human_sexuality,mmlu:international_law,mmlu:jurisprudence,mmlu:logical_fallacies,mmlu:machine_learning,mmlu:management,mmlu:marketing,mmlu:medical_genetics,mmlu:miscellaneous,mmlu:moral_disputes,mmlu:moral_scenarios,mmlu:nutrition,mmlu:philosophy,mmlu:prehistory,mmlu:professional_accounting,mmlu:professional_law,mmlu:professional_medicine,mmlu:professional_psychology,mmlu:public_relations,mmlu:security_studies,mmlu:sociology,mmlu:us_foreign_policy,mmlu:virology,mmlu:world_religions,gsm8k,truthfulqa:gen"

echo "=========================================================="
echo "Starting FULL evaluation of HuggingFace model: $MODEL_NAME"
echo "Device: $DEVICE | Batch Size: $BATCH_SIZE"
echo "Output Directory: $OUTPUT_DIR"
[ "$CLASSIFICATION_ONLY" = "1" ] && echo "Mode: classification only (no gsm8k, truthfulqa:gen)"
echo "=========================================================="

EXTRA_ARGS="--sequential_tasks"
[ "$CLASSIFICATION_ONLY" = "1" ] && EXTRA_ARGS="$EXTRA_ARGS --classification_only"

python evaluate_local.py \
    --model_name "$MODEL_NAME" \
    --tasks "$TASKS" \
    --max_samples -1 \
    --batch_size "$BATCH_SIZE" \
    --device "$DEVICE" \
    --output_dir "$OUTPUT_DIR" \
    $EXTRA_ARGS

echo "=========================================================="
echo "FULL Evaluation finished. Results saved to $OUTPUT_DIR"
echo "=========================================================="
