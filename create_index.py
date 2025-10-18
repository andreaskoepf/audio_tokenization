#!/usr/bin/env python3
"""
Create an index JSONL file over tokenized audio files.

This script recursively searches for tokenized audio JSONL files and creates an index
that contains all metadata but strips the large token payload. The index includes
file offsets to allow seeking directly to specific entries in the source files.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterator


def setup_logging(verbose: bool = False):
    """Configure logging to console."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def find_jsonl_files(input_dir: Path, pattern: str = "*.jsonl") -> list[Path]:
    """
    Recursively find JSONL files matching the pattern.

    Args:
        input_dir: Directory to search in
        pattern: Glob pattern for matching files (default: "*.jsonl")

    Returns:
        List of Path objects for matching files
    """
    files = sorted(input_dir.rglob(pattern))
    return files


def get_relative_path(file_path: Path, cwd: Path) -> str:
    """
    Get relative path from current working directory.

    Args:
        file_path: Absolute path to the file
        cwd: Current working directory

    Returns:
        Relative path as string
    """
    try:
        return str(file_path.relative_to(cwd))
    except ValueError:
        # If file is not relative to cwd, return absolute path
        return str(file_path.absolute())


def process_jsonl_file(
    jsonl_path: Path,
    cwd: Path,
    exclude_keys: set[str] = {"tokens"}
) -> Iterator[dict]:
    """
    Process a single JSONL file and yield index entries.

    Args:
        jsonl_path: Path to the JSONL file to process
        cwd: Current working directory for relative path calculation
        exclude_keys: Set of keys to exclude from the index (default: {"tokens"})

    Yields:
        Index entry dictionaries with metadata and file offsets
    """
    relative_path = get_relative_path(jsonl_path, cwd)

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        line_number = 0
        while True:
            # Record the byte offset at the start of the line
            start_offset = f.tell()
            line = f.readline()

            if not line:
                break

            line_number += 1

            # Skip empty lines
            if not line.strip():
                continue

            # Record the byte offset at the end of the line
            end_offset = f.tell()

            try:
                entry = json.loads(line)

                # Create index entry by copying all fields except excluded keys
                index_entry = {
                    "source_jsonl": relative_path,
                    "line_number": line_number,
                    "start_offset": start_offset,
                    "end_offset": end_offset,
                }

                # Add all fields from original entry except excluded keys
                for key, value in entry.items():
                    if key not in exclude_keys:
                        index_entry[key] = value

                yield index_entry

            except json.JSONDecodeError as e:
                logging.warning(f"Failed to parse line {line_number} in {relative_path}: {e}")
                continue


def create_index(
    input_dir: Path,
    output_file: Path,
    pattern: str = "*.jsonl",
    exclude_keys: set[str] = {"tokens"},
    exclude_patterns: list[str] = None,
    verbose: bool = False
) -> int:
    """
    Create index file from all matching JSONL files.

    Args:
        input_dir: Directory to search for JSONL files
        output_file: Path to output index file
        pattern: Glob pattern for matching input files
        exclude_keys: Set of keys to exclude from index entries
        exclude_patterns: List of filename patterns to exclude (e.g., ["*_index.jsonl"])
        verbose: Enable verbose logging

    Returns:
        Number of index entries created
    """
    setup_logging(verbose)

    # Get current working directory for relative paths
    cwd = Path.cwd()

    # Find all matching JSONL files
    logging.info(f"Searching for JSONL files in {input_dir} with pattern '{pattern}'")
    jsonl_files = find_jsonl_files(input_dir, pattern)

    # Filter out excluded patterns
    if exclude_patterns:
        original_count = len(jsonl_files)
        for exclude_pattern in exclude_patterns:
            jsonl_files = [f for f in jsonl_files if not f.match(exclude_pattern)]
        excluded_count = original_count - len(jsonl_files)
        if excluded_count > 0:
            logging.info(f"Excluded {excluded_count} files matching exclusion patterns")

    # Exclude the output file itself if it exists in the list
    output_file_abs = output_file.absolute()
    jsonl_files = [f for f in jsonl_files if f.absolute() != output_file_abs]

    if not jsonl_files:
        logging.warning("No JSONL files found matching the criteria")
        return 0

    logging.info(f"Found {len(jsonl_files)} JSONL files to index")

    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Process all files and write index
    total_entries = 0
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for jsonl_file in jsonl_files:
            logging.info(f"Processing: {jsonl_file}")
            try:
                file_entries = 0
                for index_entry in process_jsonl_file(jsonl_file, cwd, exclude_keys):
                    out_f.write(json.dumps(index_entry) + '\n')
                    file_entries += 1
                    total_entries += 1

                logging.info(f"  Indexed {file_entries} entries from {jsonl_file.name}")

            except Exception as e:
                logging.error(f"Failed to process {jsonl_file}: {e}", exc_info=True)
                continue

    logging.info(f"Successfully created index with {total_entries} entries: {output_file}")
    return total_entries


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create an index JSONL file over tokenized audio files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Input directory to search for JSONL files (searched recursively)"
    )

    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Output index JSONL file path"
    )

    parser.add_argument(
        "--pattern",
        type=str,
        default="*.jsonl",
        help="Glob pattern for matching input JSONL files"
    )

    parser.add_argument(
        "--exclude-keys",
        type=str,
        nargs="+",
        default=["tokens"],
        help="Keys to exclude from index entries (e.g., 'tokens')"
    )

    parser.add_argument(
        "--exclude-patterns",
        type=str,
        nargs="+",
        default=["*_index.jsonl", "*index.jsonl"],
        help="Filename patterns to exclude from indexing"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_file = Path(args.output_file)

    if not input_dir.exists():
        logging.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)

    if not input_dir.is_dir():
        logging.error(f"Input path is not a directory: {input_dir}")
        sys.exit(1)

    try:
        exclude_keys = set(args.exclude_keys)
        total_entries = create_index(
            input_dir,
            output_file,
            pattern=args.pattern,
            exclude_keys=exclude_keys,
            exclude_patterns=args.exclude_patterns,
            verbose=args.verbose
        )

        if total_entries == 0:
            logging.warning("No entries were indexed")
            sys.exit(1)

    except Exception as e:
        logging.error(f"Index creation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
