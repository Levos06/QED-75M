#!/bin/bash

# Configuration
MODEL_PATH=${1:-"model.pt"}
TOKENIZER_PATH=${2:-"/Users/levosadchi/Desktop/transformer4/data/tokenizer/tokenizer.json"}
TASKS=${3:-"hellaswag,tiny:mmlu"}
MAX_SAMPLES=${4:-10}
DEVICE=${5:-"mps"}
OUTPUT_DIR=${6:-"./results_local"}

echo "Starting evaluation of local model: $MODEL_PATH"
echo "Tasks: $TASKS"
echo "Max samples: $MAX_SAMPLES"
echo "Device: $DEVICE"

# Run evaluation
python evaluate_local.py \
    --model_path "$MODEL_PATH" \
    --tokenizer_name "$TOKENIZER_PATH" \
    --tasks "$TASKS" \
    --max_samples "$MAX_SAMPLES" \
    --device "$DEVICE" \
    --output_dir "$OUTPUT_DIR"

echo "Evaluation finished. Results saved to $OUTPUT_DIR"
