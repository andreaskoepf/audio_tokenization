#!/bin/bash
#SBATCH --account=transfernetx
#SBATCH --partition=develbooster
#SBATCH --job-name=audio_tokenize
#SBATCH --output=logs/tokenize_%A_%a.out
#SBATCH --error=logs/tokenize_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --array=0-3

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# This script processes 4 tar files in parallel on a single node with 4 GPUs
# Designed for 2-hour time limit with automatic resume capability
#
# Usage: sbatch run_tokenize_4gpu.sh <tar_list_file> <model_path> <output_dir>
#
# tar_list_file: Text file with one tar file path per line
# model_path: Path to the local audio tokenizer model
# output_dir: Directory where JSONL outputs will be written
#
# Example:
#   sbatch run_tokenize_4gpu.sh tar_files.txt /path/to/model /path/to/output
#
# For resumable processing:
#   1. Run: python check_progress.py --tar-list tar_files.txt --output-dir /path/to/output
#   2. This generates: remaining_tar_files.txt
#   3. Resubmit: sbatch run_tokenize_4gpu.sh remaining_tar_files.txt /path/to/model /path/to/output

# Check arguments
if [ "$#" -ne 3 ]; then
    echo "Usage: sbatch $0 <tar_list_file> <model_path> <output_dir>"
    exit 1
fi

TAR_LIST_FILE=$1
MODEL_PATH=$2
OUTPUT_DIR=$3

# Create logs directory if it doesn't exist
mkdir -p logs

# Calculate which tar files this array task should process
# Each array task processes 4 tar files (one per GPU)
# Array IDs start at 0: ID 0 -> lines 1-4, ID 1 -> lines 5-8, etc.
START_IDX=$(( SLURM_ARRAY_TASK_ID * 4 + 1 ))
END_IDX=$(( (SLURM_ARRAY_TASK_ID + 1) * 4 ))

echo "=========================================="
echo "SLURM Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Processing tar files: $START_IDX to $END_IDX"
echo "Node: $SLURM_NODELIST"
echo "Model path: $MODEL_PATH"
echo "Output dir: $OUTPUT_DIR"
echo "=========================================="

# Extract the tar files for this task
TAR_FILES=($(sed -n "${START_IDX},${END_IDX}p" "$TAR_LIST_FILE"))

# Check if we have tar files to process
if [ ${#TAR_FILES[@]} -eq 0 ]; then
    echo "No tar files to process for task $SLURM_ARRAY_TASK_ID"
    exit 0
fi

echo "Found ${#TAR_FILES[@]} tar files to process"

# Launch one process per GPU in parallel
for i in "${!TAR_FILES[@]}"; do
    TAR_FILE="${TAR_FILES[$i]}"
    GPU_ID=$i

    if [ -z "$TAR_FILE" ]; then
        echo "Skipping empty tar file entry at index $i"
        continue
    fi

    echo "Starting GPU $GPU_ID: $TAR_FILE"

    # Run tokenization in background for this GPU
    # Use --skip-existing to automatically skip already processed files
    # Use the Python interpreter from the venv
    HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python tokenize_audio.py \
        --model-path "$MODEL_PATH" \
        --input-tar "$TAR_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --gpu-id $GPU_ID \
        --skip-existing \
        --log-file "logs/tokenize_${SLURM_ARRAY_TASK_ID}_gpu${GPU_ID}.log" \
        &

    # Store the PID
    PIDS[$i]=$!
done

# Wait for all background processes to complete
echo "Waiting for all GPU processes to complete..."
for i in "${!PIDS[@]}"; do
    PID=${PIDS[$i]}
    wait $PID
    EXIT_CODE=$?
    echo "GPU $i (PID $PID) finished with exit code $EXIT_CODE"
done

echo "=========================================="
echo "Array task $SLURM_ARRAY_TASK_ID completed"
echo "=========================================="
