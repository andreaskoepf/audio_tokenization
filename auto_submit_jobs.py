#!/usr/bin/env python3
"""
Automated SLURM job submission script for audio tokenization.

This script periodically checks the number of queued/running jobs and submits
new jobs if below the threshold, tracking which tar files have been processed.

Usage:
    python auto_submit_jobs.py \
        --tarlist tarlist.txt \
        --model-path <path-to-model> \
        --output-dir <out-dir> \
        --max-jobs 62 \
        --check-interval 60
"""

import argparse
import subprocess
import time
import sys
import logging
from pathlib import Path
from typing import List, Set
import json
import tempfile
from datetime import datetime


def setup_logging(log_file=None, verbose=False):
    """Configure logging to file and console."""
    log_level = logging.DEBUG if verbose else logging.INFO
    handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    handlers.append(console_handler)

    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)

    logging.basicConfig(
        level=log_level,
        handlers=handlers,
        force=True
    )


def get_num_jobs(username: str) -> int:
    """Get the number of pending and running jobs for the user.

    This properly counts array jobs. For example, if squeue shows:
      12674967_[730-758]  <- This is 29 jobs (758-730+1)
      12674967_729        <- This is 1 job

    Returns the total count of individual jobs (array tasks).
    """
    try:
        result = subprocess.run(
            ["squeue", "--user", username, "--noheader"],
            capture_output=True,
            text=True,
            check=True
        )

        total_jobs = 0
        for line in result.stdout.strip().split('\n'):
            line = line.strip()
            if not line:
                continue

            # Parse the JOBID field (first column)
            parts = line.split()
            if not parts:
                continue

            jobid = parts[0]

            # Check if this is an array job with pending tasks: JOBID_[start-end]
            if '_[' in jobid and ']' in jobid:
                # Extract the range: "12674967_[730-758]" -> "730-758"
                range_part = jobid.split('_[')[1].rstrip(']')

                if '-' in range_part:
                    # Range format: "730-758"
                    try:
                        start, end = map(int, range_part.split('-'))
                        count = end - start + 1
                        total_jobs += count
                    except ValueError:
                        # If parsing fails, count as 1
                        logging.warning(f"Failed to parse array range: {jobid}")
                        total_jobs += 1
                else:
                    # Single task: "12674967_[730]"
                    total_jobs += 1
            else:
                # Regular job or single running array task: "12674967_729"
                total_jobs += 1

        return total_jobs
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to get job count: {e}")
        return -1


def get_processed_tar_files(output_dir: Path) -> Set[str]:
    """Get set of tar files that have been fully processed (have .jsonl output)."""
    processed = set()

    if not output_dir.exists():
        return processed

    # Look for .jsonl files (not .jsonl.partial)
    for jsonl_file in output_dir.glob("*.jsonl"):
        # Skip partial files
        if jsonl_file.suffix == '.partial':
            continue

        # The stem is the tar filename without extension
        tar_name = jsonl_file.stem
        processed.add(tar_name)

    return processed


def read_tarlist(tarlist_path: Path) -> List[str]:
    """Read tar file paths from tarlist file."""
    with open(tarlist_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def get_tar_basename(tar_path: str) -> str:
    """Get the basename of a tar file without extension."""
    return Path(tar_path).stem


def calculate_array_range(num_tars: int, tars_per_job: int = 4) -> str:
    """Calculate the SLURM array range for the given tar files.

    Array IDs start at 0:
    - Array ID 0: processes tars 0-3 (first 4 tars)
    - Array ID 1: processes tars 4-7 (next 4 tars)
    - etc.

    Args:
        num_tars: Number of tar files to process
        tars_per_job: Number of tars processed per job (default: 4)

    Returns:
        Array range string (e.g., "0-9" or "0" for single task)
    """
    # Calculate how many array tasks we need
    num_tasks = (num_tars + tars_per_job - 1) // tars_per_job

    if num_tasks == 1:
        return "0"
    else:
        return f"0-{num_tasks - 1}"


def submit_job(
    script_path: Path,
    tarlist_path: Path,
    model_path: str,
    output_dir: Path,
    unprocessed_tars: List[str],
    dry_run: bool = False
) -> bool:
    """Submit a job using sbatch.

    Args:
        script_path: Path to the SLURM script
        tarlist_path: Path to the tar list file
        model_path: Path to the model
        output_dir: Output directory
        unprocessed_tars: List of unprocessed tar file paths
        dry_run: If True, only print what would be submitted

    Returns:
        True if submission was successful, False otherwise
    """
    # Create a UNIQUE temporary tarlist with only unprocessed tars
    # Use timestamp to ensure uniqueness across multiple submissions
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    temp_tarlist = tarlist_path.parent / f".{tarlist_path.stem}.{timestamp}.tmp"

    with open(temp_tarlist, 'w') as f:
        f.write('\n'.join(unprocessed_tars) + '\n')

    # Calculate array range
    num_unprocessed = len(unprocessed_tars)
    array_range = calculate_array_range(num_unprocessed, tars_per_job=4)

    cmd = [
        "sbatch",
        f"--array={array_range}",
        str(script_path),
        str(temp_tarlist),
        model_path,
        str(output_dir)
    ]

    if dry_run:
        logging.info(f"[DRY RUN] Would submit: {' '.join(cmd)}")
        logging.info(f"[DRY RUN] Temporary tarlist: {temp_tarlist}")
        logging.info(f"[DRY RUN] Number of tars: {num_unprocessed}")
        logging.info(f"[DRY RUN] Array range: {array_range}")
        return True

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        logging.info(f"Job submitted: {result.stdout.strip()}")
        logging.info(f"Submitted {num_unprocessed} tar files for processing (array: {array_range})")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to submit job: {e}")
        logging.error(f"stdout: {e.stdout}")
        logging.error(f"stderr: {e.stderr}")
        return False


def save_state(state_file: Path, submitted_tars: Set[str]):
    """Save the state of submitted tar files."""
    with open(state_file, 'w') as f:
        json.dump({"submitted": list(submitted_tars)}, f, indent=2)


def load_state(state_file: Path) -> Set[str]:
    """Load the state of submitted tar files."""
    if not state_file.exists():
        return set()

    try:
        with open(state_file, 'r') as f:
            data = json.load(f)
            return set(data.get("submitted", []))
    except Exception as e:
        logging.warning(f"Failed to load state file: {e}")
        return set()


def cleanup_old_temp_files(directory: Path, pattern: str = ".*.*.tmp", max_age_hours: int = 24):
    """Clean up old temporary tarlist files.

    Args:
        directory: Directory to search for temp files
        pattern: Glob pattern for temp files
        max_age_hours: Remove files older than this many hours
    """
    import time

    if not directory.exists():
        return

    current_time = time.time()
    max_age_seconds = max_age_hours * 3600

    for temp_file in directory.glob(pattern):
        try:
            file_age = current_time - temp_file.stat().st_mtime
            if file_age > max_age_seconds:
                temp_file.unlink()
                logging.debug(f"Cleaned up old temp file: {temp_file}")
        except Exception as e:
            logging.debug(f"Failed to clean up {temp_file}: {e}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Automated SLURM job submission for audio tokenization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--tarlist",
        type=str,
        required=True,
        help="Path to tar list file"
    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the audio tokenizer model"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for JSONL files"
    )

    parser.add_argument(
        "--sbatch-script",
        type=str,
        default="./run_tokenize_4gpu.sh",
        help="Path to the sbatch script"
    )

    parser.add_argument(
        "--max-jobs",
        type=int,
        default=62,
        help="Maximum number of concurrent jobs (pending + running)"
    )

    parser.add_argument(
        "--check-interval",
        type=int,
        default=60,
        help="Interval in seconds between checks"
    )

    parser.add_argument(
        "--username",
        type=str,
        default=None,
        help="SLURM username (default: current user from $USER)"
    )

    parser.add_argument(
        "--state-file",
        type=str,
        default=".auto_submit_state.json",
        help="File to track submission state"
    )

    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to log file (optional)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode - don't actually submit jobs"
    )

    parser.add_argument(
        "--once",
        action="store_true",
        help="Run once and exit (don't loop)"
    )

    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start processing from this index in the tarlist (0-based, default: 0)"
    )

    return parser.parse_args()


def main():
    """Main loop for automated job submission."""
    args = parse_args()

    # Setup logging
    setup_logging(args.log_file, args.verbose)

    # Get username
    import os
    username = args.username or os.environ.get('USER', 'unknown')

    # Validate paths
    tarlist_path = Path(args.tarlist)
    if not tarlist_path.exists():
        logging.error(f"Tar list file not found: {tarlist_path}")
        sys.exit(1)

    sbatch_script = Path(args.sbatch_script)
    if not sbatch_script.exists():
        logging.error(f"SBATCH script not found: {sbatch_script}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    state_file = Path(args.state_file)

    # Load all tar files
    all_tars = read_tarlist(tarlist_path)

    # Apply start index if specified
    if args.start_index > 0:
        if args.start_index >= len(all_tars):
            logging.error(f"Start index {args.start_index} is >= total tar files {len(all_tars)}")
            sys.exit(1)
        logging.info(f"Skipping first {args.start_index} entries due to --start-index")
        all_tars = all_tars[args.start_index:]

    total_tars = len(all_tars)

    logging.info("=" * 80)
    logging.info("Automated Job Submission for Audio Tokenization")
    logging.info("=" * 80)
    logging.info(f"Username: {username}")
    logging.info(f"Tar list: {tarlist_path}")
    if args.start_index > 0:
        logging.info(f"Start index: {args.start_index} (skipping first {args.start_index} entries)")
    logging.info(f"Total tar files to process: {total_tars}")
    logging.info(f"Model path: {args.model_path}")
    logging.info(f"Output dir: {output_dir}")
    logging.info(f"SBATCH script: {sbatch_script}")
    logging.info(f"Max concurrent jobs: {args.max_jobs}")
    logging.info(f"Check interval: {args.check_interval}s")
    logging.info(f"State file: {state_file}")
    if args.dry_run:
        logging.info("DRY RUN MODE - No jobs will be submitted")
    logging.info("=" * 80)

    # Load state
    submitted_tars = load_state(state_file)

    iteration = 0

    while True:
        iteration += 1
        logging.info(f"\n{'=' * 80}")
        logging.info(f"Iteration {iteration}")
        logging.info(f"{'=' * 80}")

        # Clean up old temporary tarlist files (older than 24 hours)
        if iteration % 10 == 1:  # Check every 10 iterations to avoid overhead
            cleanup_old_temp_files(tarlist_path.parent, pattern=f".{tarlist_path.stem}.*.tmp")

        # Get current job count
        num_jobs = get_num_jobs(username)
        if num_jobs < 0:
            logging.error("Failed to get job count, waiting before retry...")
            time.sleep(args.check_interval)
            continue

        logging.info(f"Current jobs (pending + running): {num_jobs}")

        # Get processed tar files
        processed_tars = get_processed_tar_files(output_dir)
        logging.info(f"Processed tar files: {len(processed_tars)}")

        # Find unprocessed and unsubmitted tars
        unprocessed_tars = []
        for tar_path in all_tars:
            tar_basename = get_tar_basename(tar_path)
            if tar_basename not in processed_tars and tar_path not in submitted_tars:
                unprocessed_tars.append(tar_path)

        num_unprocessed = len(unprocessed_tars)
        logging.info(f"Unprocessed tar files: {num_unprocessed}")

        # Calculate how many jobs we can submit
        jobs_available = args.max_jobs - num_jobs

        if num_unprocessed == 0:
            logging.info("All tar files have been processed or submitted!")
            if args.once:
                logging.info("Exiting (--once mode)")
                break
            logging.info("Monitoring for any new files or failed jobs...")
        elif jobs_available <= 0:
            logging.info(f"Job limit reached ({num_jobs}/{args.max_jobs}), waiting...")
        else:
            logging.info(f"Can submit up to {jobs_available} more jobs")

            # Each job array task processes 4 tars, so we can submit up to jobs_available tasks
            # which means jobs_available * 4 tars
            max_tars_to_submit = jobs_available * 4
            tars_to_submit = unprocessed_tars[:max_tars_to_submit]

            if tars_to_submit:
                logging.info(f"Submitting {len(tars_to_submit)} tar files...")

                success = submit_job(
                    sbatch_script,
                    tarlist_path,
                    args.model_path,
                    output_dir,
                    tars_to_submit,
                    dry_run=args.dry_run
                )

                if success:
                    # Mark these tars as submitted
                    submitted_tars.update(tars_to_submit)
                    save_state(state_file, submitted_tars)
                    logging.info(f"Updated state file: {state_file}")

        # Exit if running once
        if args.once:
            logging.info("Exiting (--once mode)")
            break

        # Wait before next check
        logging.info(f"Waiting {args.check_interval} seconds before next check...")
        time.sleep(args.check_interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("\nReceived interrupt signal, exiting...")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)
