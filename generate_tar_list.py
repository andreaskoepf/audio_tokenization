#!/usr/bin/env python3
"""
Generate a list of tar files for batch processing.

This script helps create the input file needed for the SLURM batch script.
"""

import argparse
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Generate a list of tar files for batch processing"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing tar files"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Output file to write tar file paths"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.tar",
        help="Glob pattern for tar files (default: *.tar)"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Filter files by prefix (e.g., '00000' for files starting with 00000)"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="Start index for numbered tar files (e.g., 1 for 000001.tar)"
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="End index for numbered tar files (e.g., 100 for 000100.tar)"
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    # Find all tar files
    tar_files = sorted(input_dir.glob(args.pattern))

    # Apply prefix filter
    if args.prefix:
        tar_files = [f for f in tar_files if f.name.startswith(args.prefix)]

    # Apply index range filter for numbered tar files
    if args.start is not None or args.end is not None:
        filtered = []
        for tar_file in tar_files:
            # Try to extract number from filename (e.g., 000001.tar -> 1)
            try:
                # Remove extension and try to parse as integer
                num_str = tar_file.stem
                num = int(num_str)

                # Check if within range
                if args.start is not None and num < args.start:
                    continue
                if args.end is not None and num > args.end:
                    continue

                filtered.append(tar_file)
            except ValueError:
                # Skip files that don't have numeric names
                continue

        tar_files = filtered

    if not tar_files:
        print(f"Warning: No tar files found matching criteria", file=sys.stderr)
        sys.exit(0)

    # Write to output file
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for tar_file in tar_files:
            f.write(f"{tar_file.absolute()}\n")

    print(f"Generated list of {len(tar_files)} tar files")
    print(f"Output: {output_path}")

    # Print first few and last few
    if len(tar_files) > 0:
        print("\nFirst files:")
        for tar_file in tar_files[:5]:
            print(f"  {tar_file.name}")

        if len(tar_files) > 10:
            print("  ...")
            print("\nLast files:")
            for tar_file in tar_files[-5:]:
                print(f"  {tar_file.name}")


if __name__ == "__main__":
    main()
