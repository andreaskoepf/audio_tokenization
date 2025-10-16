# Audio Tokenization Tool

This tool tokenizes audio files (MP3) stored in tar archives using the `bosonai/higgs-audio-v2-tokenizer` model. It's designed for large-scale batch processing on SLURM clusters with multi-GPU support.

## Features

- Processes MP3 files from tar archives
- Chunked processing for long audio files with overlapping to minimize border effects
- Multi-GPU parallel processing support
- JSONL output format for each tar file
- **Resumable processing at individual audio file level within tar archives**
- Progress tracking and comprehensive logging
- SLURM batch script for cluster processing optimized for 2-hour time limits

## Model Information

The tool uses the `bosonai/higgs-audio-v2-tokenizer`, which:
- Uses Residual Vector Quantization (RVQ) with 8 codebooks
- Converts 24 kHz mono audio to discrete tokens
- Generates 25 tokens per second (25 frames/sec)
- Each frame = 960 audio samples at 24 kHz
- Output shape: [8, T] where T is the number of frames

## Installation

Ensure you have the required dependencies:

```bash
pip install torch torchaudio boson-multimodal
```

## Usage

### 1. Download the Model

First, download the model to a local path (required for cluster nodes without internet access):

```bash
# Example using huggingface-cli
huggingface-cli download bosonai/higgs-audio-v2-tokenizer --local-dir /path/to/local/model
```

### 2. Generate List of Tar Files

Use the helper script to create a list of tar files to process:

```bash
# Generate list of all tar files in a directory
python generate_tar_list.py \
    --input-dir /path/to/tar/files \
    --output-file tar_files.txt

# Generate list with range filter (e.g., 000001.tar to 000100.tar)
python generate_tar_list.py \
    --input-dir /path/to/tar/files \
    --output-file tar_files.txt \
    --start 1 \
    --end 100
```

### 3. Process Single Tar File (Local Testing)

```bash
python tokenize_audio.py \
    --model-path /path/to/local/model \
    --input-tar /path/to/000001.tar \
    --output-dir /path/to/output \
    --gpu-id 0
```

### 4. Batch Processing on SLURM Cluster

#### Option A: Automated Resume (Recommended for 2-hour time limits)

For clusters with short time limits, use the automated submission script:

```bash
# Make scripts executable
chmod +x run_tokenize_4gpu.sh submit_and_resume.sh

# This will automatically submit, monitor, and resubmit jobs until completion
./submit_and_resume.sh tar_files.txt /path/to/model /path/to/output
```

The script will:
1. Submit a SLURM job with appropriate array size
2. Wait for the job to complete
3. Check progress and identify remaining tar files
4. Automatically resubmit a new job for remaining files
5. Repeat until all files are processed

#### Option B: Manual Submission and Resume

The SLURM script processes 4 tar files in parallel per node (one per GPU):

```bash
# Make script executable
chmod +x run_tokenize_4gpu.sh

# Submit initial batch job
sbatch run_tokenize_4gpu.sh \
    tar_files.txt \
    /path/to/local/model \
    /path/to/output

# After job completes (or hits time limit), check progress
python check_progress.py \
    --tar-list tar_files.txt \
    --output-dir /path/to/output \
    --check-validity

# This creates remaining_tar_files.txt with unprocessed files
# Resubmit with remaining files
sbatch run_tokenize_4gpu.sh \
    remaining_tar_files.txt \
    /path/to/local/model \
    /path/to/output
```

#### How the SLURM Script Works

- Each SLURM array task processes 4 tar files (one per GPU)
- Default time limit: 2 hours (configurable in script)
- Automatically skips already-completed files using `--skip-existing`
- If you have 100 tar files in `tar_files.txt`, you need 25 array tasks
- Adjust `#SBATCH --array=1-25` in the script accordingly
- For N tar files, you need `ceil(N/4)` array tasks

**SLURM Array Indexing Example:**
- Array Task 1: processes tar files 1-4 (lines 1-4 from input list)
- Array Task 2: processes tar files 5-8 (lines 5-8 from input list)
- Array Task 3: processes tar files 9-12 (lines 9-12 from input list)

Example for 1000 tar files:
```bash
# Edit run_tokenize_4gpu.sh and change:
#SBATCH --array=1-250

# Then submit
sbatch run_tokenize_4gpu.sh tar_files.txt /path/to/model /path/to/output
```

## Output Format

Each tar file produces one JSONL file with the same name:

**Input:** `000001.tar`
**Output:** `000001.jsonl`

Each line in the JSONL file contains:

```json
{
  "tar_file": "/path/to/000001.tar",
  "audio_file": "audio_001.mp3",
  "num_samples": 2880000,
  "sample_rate": 24000,
  "tokens": [[...], [...], ..., [...]],
  "token_shape": [8, 3000]
}
```

For failed files:

```json
{
  "tar_file": "/path/to/000001.tar",
  "audio_file": "corrupted_audio.mp3",
  "error": "Error message here"
}
```

## Command Line Options

### tokenize_audio.py

```
--model-path          Path to local audio tokenizer model (required)
--input-tar           Path to input tar file (required)
--output-dir          Directory for output JSONL files (required)
--gpu-id              GPU device ID (default: 0)
--sample-rate         Target sample rate (default: 24000)
--max-tokens-per-fragment  Max tokens per chunk (default: 750 = 30 sec)
--num-tokens-overlap  Token overlap between chunks (default: 25 = 1 sec)
--extensions          Audio file extensions (default: .mp3)
--log-file            Path to log file (optional)
--verbose             Enable verbose logging
--skip-existing       Skip processing if output file already exists
--force               Force overwrite without prompting
```

### generate_tar_list.py

```
--input-dir           Directory containing tar files (required)
--output-file         Output file path (required)
--pattern             Glob pattern (default: *.tar)
--prefix              Filter by filename prefix
--start               Start index for numbered files
--end                 End index for numbered files
```

### check_progress.py

```
--tar-list            Original list of tar files (required)
--output-dir          Output directory with JSONL files (required)
--remaining-output    Output file for remaining files (default: remaining_tar_files.txt)
--completed-output    Optional output file for completed files
--check-validity      Verify JSONL files are valid, not just present
--verbose             Show detailed progress per file
```

## Monitoring Progress

### Check Overall Progress

Use the `check_progress.py` script to see how many files have been completed:

```bash
python check_progress.py \
    --tar-list tar_files.txt \
    --output-dir /path/to/output \
    --check-validity \
    --verbose
```

Output example:
```
Checking progress for 1000 tar files...
Output directory: /path/to/output

================================================================================
PROGRESS SUMMARY
================================================================================
Total tar files:       1000
Completed:             847 (84.7%)
Remaining:             153 (15.3%)
```

### Monitor Running Jobs

Logs are stored in the `logs/` directory:

```bash
# View logs for specific array task and GPU
tail -f logs/tokenize_12345_6_gpu0.log

# View SLURM output
tail -f logs/tokenize_12345_6.out

# Check for errors
grep -i error logs/*.log

# Count currently running jobs
squeue -u $USER | grep audio_tokenize | wc -l
```

## Troubleshooting

### Out of Memory Errors

Reduce the chunk size:

```bash
python tokenize_audio.py \
    --max-tokens-per-fragment 500 \
    ...
```

### Handling 2-Hour Time Limits and Partial Processing

The tool supports **automatic resume at the individual audio file level** within tar archives:

- Each audio file is written to a `.jsonl.partial` file immediately after processing
- If processing is interrupted, the tool automatically resumes from where it left off
- Partial files are tracked per tar archive
- When a tar file is fully processed, the final `.jsonl` is written and `.partial` is removed

**Example workflow after interruption:**

1. **Check progress (shows partial files):**
```bash
python check_progress.py --tar-list tar_files.txt --output-dir /path/to/output --verbose
```

Output:
```
[1/100] ✓ 000001.tar
[2/100] ◐ 000002.tar (partial: 87 files processed)
[3/100] ○ 000003.tar (not started)
```

2. **Simply resubmit - partial tar files automatically resume:**
```bash
# The tool will resume 000002.tar from file 88
sbatch run_tokenize_4gpu.sh remaining_tar_files.txt /path/to/model /path/to/output
```

3. **Or use automated script:**
```bash
./submit_and_resume.sh tar_files.txt /path/to/model /path/to/output
```

**Important:** The `remaining_tar_files.txt` includes partially-processed tar files, which will resume automatically on rerun.

### Finding Invalid/Corrupted Outputs

Use `--check-validity` to find JSONL files that are empty or malformed:

```bash
python check_progress.py \
    --tar-list tar_files.txt \
    --output-dir /path/to/output \
    --check-validity

# This creates invalid_tar_files.txt if any are found
```

### Re-running Specific Files

To reprocess specific tar files (e.g., after fixing issues):

```bash
# Create a list with just the files to reprocess
echo "/path/to/000123.tar" > reprocess.txt
echo "/path/to/000456.tar" >> reprocess.txt

# Process with --force to overwrite
python tokenize_audio.py \
    --model-path /path/to/model \
    --input-tar /path/to/000123.tar \
    --output-dir /path/to/output \
    --force
```

## Technical Details

### Chunked Processing

Long audio files are processed in overlapping chunks to:
1. Avoid GPU memory exhaustion
2. Minimize border artifacts from convolution filters

Default settings:
- Chunk size: 1000 seconds (25,000 tokens at 25 tokens/sec)
- Overlap: 2 seconds (50 tokens)

### Resumable Processing

The tool implements fine-grained resume capability:

**Two-Level Progress Tracking:**
1. **Tar-level:** `check_progress.py` identifies which tar files need processing
2. **File-level:** Within each tar, `.jsonl.partial` tracks individual audio files

**How it works:**
- During processing, each completed audio file is immediately appended to `{tarname}.jsonl.partial`
- If interrupted (timeout, crash, etc.), the partial file is preserved
- On resume, the tool:
  - Reads the partial file to build a set of completed audio files
  - Skips those files when iterating through the tar archive
  - Continues appending new results to the partial file
- When all files in the tar are processed, the final `{tarname}.jsonl` is written and `.partial` is deleted

**Example:**
```
000001.jsonl.partial  <- 150 audio files processed, then job killed
                         Rerun automatically resumes at file 151
000001.jsonl          <- Created when all files complete
```

This means even if a single tar file takes longer than 2 hours, it will eventually complete over multiple job submissions.

### Token Format

Tokens are stored as a list of 8 lists (one per codebook):

```python
tokens = [
    [t0_cb0, t1_cb0, t2_cb0, ...],  # Codebook 0
    [t0_cb1, t1_cb1, t2_cb1, ...],  # Codebook 1
    ...
    [t0_cb7, t1_cb7, t2_cb7, ...]   # Codebook 7
]
```

Shape: [8, num_frames]

## Examples

### Process First 10 Tar Files

```bash
# Generate list
python generate_tar_list.py \
    --input-dir /data/audio_tars \
    --output-file test_files.txt \
    --start 1 \
    --end 10

# Process locally on single GPU
for tar in $(cat test_files.txt); do
    python tokenize_audio.py \
        --model-path /models/higgs-audio-v2 \
        --input-tar "$tar" \
        --output-dir /data/tokenized \
        --gpu-id 0
done
```

### Process All Files on Cluster

```bash
# Generate full list
python generate_tar_list.py \
    --input-dir /data/audio_tars \
    --output-file all_tar_files.txt

# Count files
NUM_FILES=$(wc -l < all_tar_files.txt)
NUM_ARRAY_TASKS=$(( (NUM_FILES + 3) / 4 ))

echo "Processing $NUM_FILES tar files with $NUM_ARRAY_TASKS array tasks"

# Edit SLURM script array parameter
sed -i "s/#SBATCH --array=.*/#SBATCH --array=1-${NUM_ARRAY_TASKS}/" run_tokenize_4gpu.sh

# Submit job
sbatch run_tokenize_4gpu.sh \
    all_tar_files.txt \
    /models/higgs-audio-v2 \
    /data/tokenized
```

## License

This tool is for processing audio data. Ensure you have proper rights to the audio files being processed.
