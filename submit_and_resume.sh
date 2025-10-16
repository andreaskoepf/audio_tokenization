#!/bin/bash
# Automated script to submit jobs and handle resuming after 2-hour time limits
#
# This script submits a SLURM job and can be run repeatedly to automatically
# check progress and resubmit remaining work.
#
# Usage: ./submit_and_resume.sh <tar_list_file> <model_path> <output_dir> [max_iterations]
#
# Example:
#   ./submit_and_resume.sh all_tar_files.txt /path/to/model /path/to/output 10

set -e  # Exit on error

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <tar_list_file> <model_path> <output_dir> [max_iterations]"
    echo ""
    echo "  tar_list_file:   Initial list of tar files to process"
    echo "  model_path:      Path to audio tokenizer model"
    echo "  output_dir:      Output directory for JSONL files"
    echo "  max_iterations:  Maximum resume iterations (default: 100)"
    exit 1
fi

TAR_LIST_FILE=$1
MODEL_PATH=$2
OUTPUT_DIR=$3
MAX_ITERATIONS=${4:-100}

CURRENT_LIST="$TAR_LIST_FILE"
ITERATION=1

echo "========================================"
echo "Audio Tokenization Submission Script"
echo "========================================"
echo "Initial tar list: $TAR_LIST_FILE"
echo "Model path:       $MODEL_PATH"
echo "Output dir:       $OUTPUT_DIR"
echo "Max iterations:   $MAX_ITERATIONS"
echo ""

# Ensure logs directory exists
mkdir -p logs

while [ $ITERATION -le $MAX_ITERATIONS ]; do
    echo "========================================"
    echo "Iteration $ITERATION"
    echo "========================================"

    # Check if tar list file exists and is not empty
    if [ ! -f "$CURRENT_LIST" ]; then
        echo "Error: Tar list file not found: $CURRENT_LIST"
        exit 1
    fi

    NUM_REMAINING=$(wc -l < "$CURRENT_LIST")

    if [ "$NUM_REMAINING" -eq 0 ]; then
        echo "No tar files to process. All done!"
        echo ""
        echo "Final statistics:"
        python check_progress.py \
            --tar-list "$TAR_LIST_FILE" \
            --output-dir "$OUTPUT_DIR" \
            --check-validity
        exit 0
    fi

    echo "Tar files to process: $NUM_REMAINING"

    # Calculate number of array tasks needed (4 files per task)
    NUM_TASKS=$(( (NUM_REMAINING + 3) / 4 ))
    echo "SLURM array tasks:    $NUM_TASKS"

    # Update the array parameter in the SLURM script
    # Create a temporary modified script
    TEMP_SCRIPT="run_tokenize_4gpu_iter${ITERATION}.sh"
    sed "s/#SBATCH --array=.*/#SBATCH --array=1-${NUM_TASKS}/" run_tokenize_4gpu.sh > "$TEMP_SCRIPT"

    echo ""
    echo "Submitting job..."

    # Submit the job and capture the job ID
    SUBMIT_OUTPUT=$(sbatch "$TEMP_SCRIPT" "$CURRENT_LIST" "$MODEL_PATH" "$OUTPUT_DIR")
    JOB_ID=$(echo "$SUBMIT_OUTPUT" | grep -oP 'Submitted batch job \K[0-9]+')

    if [ -z "$JOB_ID" ]; then
        echo "Error: Failed to extract job ID from: $SUBMIT_OUTPUT"
        rm -f "$TEMP_SCRIPT"
        exit 1
    fi

    echo "Job submitted: $JOB_ID"
    echo "Waiting for job to complete..."
    echo ""

    # Wait for the job to complete
    while true; do
        # Check job status
        JOB_STATUS=$(squeue -j "$JOB_ID" -h -o "%T" 2>/dev/null | head -n 1)

        if [ -z "$JOB_STATUS" ]; then
            # Job no longer in queue, it's finished
            echo "Job $JOB_ID completed"
            break
        fi

        # Count running/pending tasks
        RUNNING=$(squeue -j "$JOB_ID" -h -t RUNNING | wc -l)
        PENDING=$(squeue -j "$JOB_ID" -h -t PENDING | wc -l)

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Job $JOB_ID - Running: $RUNNING, Pending: $PENDING"

        sleep 60  # Check every minute
    done

    # Clean up temporary script
    rm -f "$TEMP_SCRIPT"

    echo ""
    echo "Checking progress..."

    # Check progress and generate new remaining list
    REMAINING_LIST="remaining_tar_files_iter${ITERATION}.txt"

    python check_progress.py \
        --tar-list "$TAR_LIST_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --remaining-output "$REMAINING_LIST" \
        --check-validity

    CHECK_EXIT_CODE=$?

    if [ $CHECK_EXIT_CODE -eq 0 ]; then
        echo ""
        echo "All tar files processed successfully!"
        exit 0
    fi

    # Prepare for next iteration
    CURRENT_LIST="$REMAINING_LIST"
    ITERATION=$((ITERATION + 1))

    echo ""
    echo "Preparing next iteration..."
    sleep 5
done

echo ""
echo "========================================"
echo "Reached maximum iterations ($MAX_ITERATIONS)"
echo "Some files may still be remaining."
echo "Check: $CURRENT_LIST"
echo "========================================"
exit 1
