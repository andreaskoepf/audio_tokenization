#!/usr/bin/env python3
"""
Check progress of audio tokenization and generate list of remaining tar files.

This script compares the list of input tar files against the output directory
to determine which files have been successfully processed and which remain.
"""

import argparse
from pathlib import Path
import json
import sys


def check_jsonl_valid(jsonl_path):
    """Check if JSONL file is valid and non-empty."""
    try:
        if jsonl_path.stat().st_size == 0:
            return False

        # Try to read at least one line
        with open(jsonl_path, 'r') as f:
            first_line = f.readline()
            if not first_line:
                return False
            # Verify it's valid JSON
            json.loads(first_line)
            return True
    except Exception:
        return False


def get_tar_status(tar_path, output_dir):
    """Get the processing status of a tar file.

    Returns:
        tuple: (status, details) where status is one of:
            'complete' - Final .jsonl exists
            'partial' - .jsonl.partial exists
            'not_started' - Neither file exists
        details contains file counts if partial
    """
    tar_name = Path(tar_path).stem
    jsonl_path = output_dir / f"{tar_name}.jsonl"
    partial_path = output_dir / f"{tar_name}.jsonl.partial"

    if jsonl_path.exists():
        return ('complete', None)
    elif partial_path.exists():
        try:
            count = sum(1 for _ in open(partial_path) if _.strip())
            return ('partial', count)
        except Exception:
            return ('partial', 0)
    else:
        return ('not_started', None)


def main():
    parser = argparse.ArgumentParser(
        description="Check progress and generate list of remaining tar files"
    )
    parser.add_argument(
        "--tar-list",
        type=str,
        required=True,
        help="Original list of tar files to process"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory containing output JSONL files"
    )
    parser.add_argument(
        "--remaining-output",
        type=str,
        default="remaining_tar_files.txt",
        help="Output file for remaining tar files"
    )
    parser.add_argument(
        "--completed-output",
        type=str,
        default=None,
        help="Optional output file for completed tar files"
    )
    parser.add_argument(
        "--check-validity",
        action="store_true",
        help="Check if output JSONL files are valid (not just present)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed progress information"
    )

    args = parser.parse_args()

    # Read tar file list
    tar_list_path = Path(args.tar_list)
    if not tar_list_path.exists():
        print(f"Error: Tar list file not found: {tar_list_path}", file=sys.stderr)
        sys.exit(1)

    with open(tar_list_path, 'r') as f:
        tar_files = [line.strip() for line in f if line.strip()]

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"Warning: Output directory does not exist: {output_dir}", file=sys.stderr)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Check which files are completed
    completed = []
    remaining = []
    invalid = []
    partial = []

    print(f"Checking progress for {len(tar_files)} tar files...")
    print(f"Output directory: {output_dir}")
    print()

    for idx, tar_file in enumerate(tar_files, 1):
        tar_path = Path(tar_file)
        status, details = get_tar_status(tar_file, output_dir)

        if status == 'complete':
            expected_output = output_dir / f"{tar_path.stem}.jsonl"
            if args.check_validity:
                if check_jsonl_valid(expected_output):
                    completed.append(tar_file)
                    if args.verbose:
                        print(f"[{idx}/{len(tar_files)}] ✓ {tar_path.name}")
                else:
                    invalid.append(tar_file)
                    remaining.append(tar_file)
                    if args.verbose:
                        print(f"[{idx}/{len(tar_files)}] ✗ {tar_path.name} (invalid/empty)")
            else:
                completed.append(tar_file)
                if args.verbose:
                    print(f"[{idx}/{len(tar_files)}] ✓ {tar_path.name}")

        elif status == 'partial':
            partial.append(tar_file)
            remaining.append(tar_file)
            if args.verbose:
                print(f"[{idx}/{len(tar_files)}] ◐ {tar_path.name} (partial: {details} files processed)")

        else:  # not_started
            remaining.append(tar_file)
            if args.verbose:
                print(f"[{idx}/{len(tar_files)}] ○ {tar_path.name} (not started)")

    # Summary
    print()
    print("=" * 80)
    print("PROGRESS SUMMARY")
    print("=" * 80)
    print(f"Total tar files:       {len(tar_files)}")
    print(f"Completed:             {len(completed)} ({len(completed)/len(tar_files)*100:.1f}%)")
    print(f"Partially processed:   {len(partial)} ({len(partial)/len(tar_files)*100:.1f}%)")
    print(f"Remaining:             {len(remaining)} ({len(remaining)/len(tar_files)*100:.1f}%)")
    if args.check_validity and invalid:
        print(f"Invalid/Empty outputs: {len(invalid)}")
    print()

    # Write remaining files
    remaining_path = Path(args.remaining_output)
    remaining_path.parent.mkdir(parents=True, exist_ok=True)

    with open(remaining_path, 'w') as f:
        for tar_file in remaining:
            f.write(f"{tar_file}\n")

    print(f"Remaining tar files written to: {remaining_path}")

    # Write completed files if requested
    if args.completed_output:
        completed_path = Path(args.completed_output)
        completed_path.parent.mkdir(parents=True, exist_ok=True)

        with open(completed_path, 'w') as f:
            for tar_file in completed:
                f.write(f"{tar_file}\n")

        print(f"Completed tar files written to: {completed_path}")

    # Write invalid files if any
    if args.check_validity and invalid:
        invalid_path = Path("invalid_tar_files.txt")
        with open(invalid_path, 'w') as f:
            for tar_file in invalid:
                f.write(f"{tar_file}\n")
        print(f"Invalid/empty outputs listed in: {invalid_path}")

    print()

    if remaining:
        print(f"To resume processing, use: {remaining_path}")
        sys.exit(1)  # Exit with error code to indicate incomplete processing
    else:
        print("All tar files have been processed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
