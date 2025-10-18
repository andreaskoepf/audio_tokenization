#!/usr/bin/env python3
"""
Tokenize a single audio file using Higgs Audio Tokenizer.

This script processes a single audio file and outputs a JSONL file compatible
with the format produced by tokenize_audio.py.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
import torch

from tokenize_audio import (
    load_audio_mono,
    audio_metadata_to_dict,
    determine_token_audio_length,
    TokenEncoder,
    setup_logging,
)
from boson_multimodal.audio_processing.higgs_audio_tokenizer import (
    load_higgs_audio_tokenizer,
)


def tokenize_single_file(
    input_file: Path,
    output_file: Path,
    encoder: TokenEncoder
) -> dict:
    """
    Tokenize a single audio file and return the result.

    Args:
        input_file: Path to the input audio file
        output_file: Path to the output JSONL file
        encoder: TokenEncoder instance

    Returns:
        Dictionary with tokenization results
    """
    logging.info(f"Processing: {input_file}")

    try:
        # Load audio and get metadata
        with open(input_file, 'rb') as f:
            wav, metadata = load_audio_mono(f, sample_rate=encoder.sample_rate)

        num_samples = wav.shape[0]

        logging.info(f"Input: {num_samples} samples (original SR: {metadata.sample_rate} Hz, channels: {metadata.num_channels})")

        # Encode to tokens
        tokens = encoder.encode_piecewise(wav)

        # Convert to list for JSON serialization
        tokens_list = tokens.squeeze(0).cpu().tolist()

        result = {
            "audio_file": str(input_file),
            "num_samples": num_samples,
            "sample_rate": encoder.sample_rate,
            "original_metadata": audio_metadata_to_dict(metadata),
            "tokens": tokens_list,
            "token_shape": list(tokens.shape)
        }

        logging.info(f"Successfully tokenized: {num_samples} samples -> {tokens.shape}")

        # Write to output file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps(result) + '\n')

        logging.info(f"Wrote output to: {output_file}")

        return result

    except Exception as ex:
        logging.error(f"Failed to process {input_file}: {ex}", exc_info=True)
        result = {
            "audio_file": str(input_file),
            "error": str(ex)
        }

        # Write error to output file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps(result) + '\n')

        return result


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Tokenize a single audio file using Higgs Audio Tokenizer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default="bosonai/higgs-audio-v2-tokenizer",
        help="Path to the audio tokenizer model"
    )

    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Path to input audio file"
    )

    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to output JSONL file"
    )

    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="GPU device ID to use"
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=24_000,
        help="Target sample rate for audio"
    )

    parser.add_argument(
        "--max-tokens-per-fragment",
        type=int,
        default=25_000,
        help="Maximum tokens per fragment for chunked processing"
    )

    parser.add_argument(
        "--num-tokens-overlap",
        type=int,
        default=50,
        help="Number of tokens to overlap between fragments"
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
        "--force",
        action="store_true",
        help="Force overwrite existing output file without prompting"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Setup logging
    setup_logging(args.log_file, args.verbose)

    logging.info("=" * 80)
    logging.info("Single Audio File Tokenization Tool")
    logging.info("=" * 80)
    logging.info(f"Model path: {args.model_path}")
    logging.info(f"Input file: {args.input_file}")
    logging.info(f"Output file: {args.output_file}")
    logging.info(f"GPU ID: {args.gpu_id}")
    logging.info(f"Sample rate: {args.sample_rate}")

    # Check if input file exists
    input_file = Path(args.input_file)
    if not input_file.exists():
        logging.error(f"Input file not found: {input_file}")
        sys.exit(1)

    if not input_file.is_file():
        logging.error(f"Input path is not a file: {input_file}")
        sys.exit(1)

    # Check if output already exists
    output_file = Path(args.output_file)
    if output_file.exists():
        if args.force:
            logging.warning(f"Output file already exists, overwriting: {output_file}")
        else:
            logging.warning(f"Output file already exists: {output_file}")
            response = input("Overwrite? [y/N]: ").strip().lower()
            if response != 'y':
                logging.info("Aborted by user")
                sys.exit(0)

    # Setup device
    if not torch.cuda.is_available():
        logging.error("CUDA is not available. GPU is required for tokenization.")
        sys.exit(1)

    device = torch.device("cuda", args.gpu_id)
    logging.info(f"Using device: {device} ({torch.cuda.get_device_name(args.gpu_id)})")

    # Load tokenizer
    start_time = datetime.now()
    logging.info("Loading audio tokenizer...")
    try:
        audio_tokenizer = load_higgs_audio_tokenizer(args.model_path, device=device)
        samples_per_token = determine_token_audio_length(audio_tokenizer)
        logging.info(f"Tokenizer loaded. Samples per token: {samples_per_token}")
    except Exception as e:
        logging.error(f"Failed to load tokenizer: {e}", exc_info=True)
        sys.exit(1)

    # Create encoder
    encoder = TokenEncoder(
        audio_tokenizer,
        samples_per_token,
        max_tokens_per_fragment=args.max_tokens_per_fragment,
        num_tokens_overlap=args.num_tokens_overlap,
        device=device,
        sample_rate=args.sample_rate,
    )

    # Process file
    try:
        result = tokenize_single_file(
            input_file,
            output_file,
            encoder
        )

        elapsed = datetime.now() - start_time

        if "error" in result:
            logging.error("=" * 80)
            logging.error("Processing failed!")
            logging.error(f"Error: {result['error']}")
            logging.error(f"Elapsed time: {elapsed}")
            logging.error("=" * 80)
            sys.exit(1)
        else:
            logging.info("=" * 80)
            logging.info("Processing complete!")
            logging.info(f"Tokens: {result['token_shape']}")
            logging.info(f"Elapsed time: {elapsed}")
            logging.info(f"Output: {output_file}")
            logging.info("=" * 80)

    except Exception as e:
        logging.error(f"Processing failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
