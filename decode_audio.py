from pathlib import Path
import argparse
import json
import logging
import sys
import torch
import torchaudio
from datetime import datetime
from boson_multimodal.audio_processing.higgs_audio_tokenizer import (
    load_higgs_audio_tokenizer,
)
from tokenize_audio import PregressiveDecoder, determine_token_audio_length


def load_jsonl(jsonl_path: Path) -> list[dict]:
    """Load all entries from a JSONL file."""
    entries = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def parse_indices(indices_str: str, max_index: int) -> list[int]:
    """Parse index specification string into list of indices.

    Supports:
    - Single index: "5"
    - Comma-separated: "1,3,5"
    - Ranges: "1-5" or "1:5"
    - Mixed: "1,3-5,7"
    """
    indices = set()

    for part in indices_str.split(','):
        part = part.strip()
        if '-' in part or ':' in part:
            # Handle range
            sep = '-' if '-' in part else ':'
            start, end = part.split(sep)
            start = int(start.strip())
            end = int(end.strip())
            indices.update(range(start, end + 1))
        else:
            # Single index
            indices.add(int(part))

    # Filter valid indices
    valid_indices = [i for i in sorted(indices) if 0 <= i < max_index]
    return valid_indices


def list_command(args):
    """List all entries in a JSONL file."""
    jsonl_path = Path(args.input_jsonl)

    if not jsonl_path.exists():
        logging.error(f"Input file not found: {jsonl_path}")
        sys.exit(1)

    logging.info(f"Loading entries from: {jsonl_path}")
    entries = load_jsonl(jsonl_path)

    print(f"\n{'Index':<8} {'Audio File':<60} {'Samples':<12} {'Duration (s)':<12} {'Tokens':<10}")
    print("=" * 110)

    for idx, entry in enumerate(entries):
        if "error" in entry:
            audio_file = entry.get("audio_file", "unknown")
            print(f"{idx:<8} {audio_file:<60} {'ERROR':<12} {'-':<12} {'-':<10}")
        else:
            audio_file = entry.get("audio_file", "unknown")
            num_samples = entry.get("num_samples", 0)
            sample_rate = entry.get("sample_rate", 24000)
            duration = num_samples / sample_rate
            token_shape = entry.get("token_shape", [0, 0])
            num_tokens = token_shape[-1] if token_shape else 0

            print(f"{idx:<8} {audio_file:<60} {num_samples:<12} {duration:<12.2f} {num_tokens:<10}")

    print("=" * 110)
    print(f"\nTotal entries: {len(entries)}")
    successful = sum(1 for e in entries if "error" not in e)
    print(f"Successful: {successful}, Failed: {len(entries) - successful}")


def decode_command(args):
    """Decode audio from JSONL file."""
    jsonl_path = Path(args.input_jsonl)
    output_dir = Path(args.output_dir)

    if not jsonl_path.exists():
        logging.error(f"Input file not found: {jsonl_path}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load entries
    logging.info(f"Loading entries from: {jsonl_path}")
    entries = load_jsonl(jsonl_path)
    logging.info(f"Found {len(entries)} entries")

    # Parse indices if specified
    if args.indices:
        indices = parse_indices(args.indices, len(entries))
        if not indices:
            logging.error(f"No valid indices found in range 0-{len(entries)-1}")
            sys.exit(1)
        logging.info(f"Selected {len(indices)} entries to decode: {indices}")
    else:
        indices = list(range(len(entries)))
        logging.info(f"Decoding all {len(indices)} entries")

    # Setup device
    if not torch.cuda.is_available():
        logging.error("CUDA is not available. GPU is required for decoding.")
        sys.exit(1)

    device = torch.device("cuda", args.gpu_id)
    logging.info(f"Using device: {device} ({torch.cuda.get_device_name(args.gpu_id)})")

    # Load tokenizer
    logging.info("Loading audio tokenizer...")
    try:
        audio_tokenizer = load_higgs_audio_tokenizer(args.model_path, device=device)
        samples_per_token = determine_token_audio_length(audio_tokenizer)
        logging.info(f"Tokenizer loaded. Samples per token: {samples_per_token}")
    except Exception as e:
        logging.error(f"Failed to load tokenizer: {e}", exc_info=True)
        sys.exit(1)

    # Create decoder
    decoder = PregressiveDecoder(
        audio_tokenizer,
        samples_per_token=samples_per_token,
        max_tokens_per_fragment=args.max_tokens_per_fragment,
        context_len=args.context_len,
        tail_delay=args.tail_delay,
    )

    # Decode selected entries
    start_time = datetime.now()
    successful = 0
    failed = 0

    for idx in indices:
        entry = entries[idx]

        if "error" in entry:
            logging.warning(f"[{idx}] Skipping entry with error: {entry.get('audio_file', 'unknown')}")
            failed += 1
            continue

        audio_file = entry.get("audio_file", f"audio_{idx}")
        tokens = entry.get("tokens")
        sample_rate = entry.get("sample_rate", 24000)

        if tokens is None:
            logging.warning(f"[{idx}] No tokens found for {audio_file}")
            failed += 1
            continue

        logging.info(f"[{idx}/{len(indices)}] Decoding: {audio_file}")

        try:
            # Convert tokens to tensor
            tokens_tensor = torch.tensor(tokens, dtype=torch.int32)
            if tokens_tensor.ndim == 1:
                tokens_tensor = tokens_tensor.unsqueeze(0)

            # Decode
            reconstructed = decoder.decode_piecewise(tokens_tensor)

            # Prepare output filename
            audio_name = Path(audio_file).stem
            output_path = output_dir / f"{audio_name}_reconstructed.wav"

            # Ensure output doesn't exist or handle overwrite
            if output_path.exists() and not args.force:
                logging.warning(f"Output exists, skipping: {output_path}")
                continue

            # Save audio
            torchaudio.save(
                str(output_path),
                reconstructed.unsqueeze(0),
                sample_rate=sample_rate,
            )

            logging.info(f"Saved: {output_path} ({reconstructed.shape[0]} samples)")
            successful += 1

        except Exception as e:
            logging.error(f"Failed to decode {audio_file}: {e}", exc_info=True)
            failed += 1

    elapsed = datetime.now() - start_time
    logging.info("=" * 80)
    logging.info("Decoding complete!")
    logging.info(f"Successful: {successful}")
    logging.info(f"Failed: {failed}")
    logging.info(f"Elapsed time: {elapsed}")
    logging.info(f"Output directory: {output_dir}")
    logging.info("=" * 80)


def setup_logging(verbose=False):
    """Configure logging."""
    log_level = logging.DEBUG if verbose else logging.INFO

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)

    logging.basicConfig(
        level=log_level,
        handlers=[console_handler]
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Decode audio from JSONL files created by tokenize_audio.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.required = True

    # List command
    list_parser = subparsers.add_parser(
        "list",
        help="List all entries in a JSONL file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    list_parser.add_argument(
        "--input-jsonl",
        type=str,
        required=True,
        help="Path to input JSONL file"
    )
    list_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    # Decode command
    decode_parser = subparsers.add_parser(
        "decode",
        help="Decode audio from JSONL file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    decode_parser.add_argument(
        "--input-jsonl",
        type=str,
        required=True,
        help="Path to input JSONL file"
    )
    decode_parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for output WAV files"
    )
    decode_parser.add_argument(
        "--model-path",
        type=str,
        default="bosonai/higgs-audio-v2-tokenizer",
        help="Path to the audio tokenizer model"
    )
    decode_parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="GPU device ID to use"
    )
    decode_parser.add_argument(
        "--max-tokens-per-fragment",
        type=int,
        default=10_000,
        help="Maximum tokens per fragment for progressive decoding"
    )
    decode_parser.add_argument(
        "--context-len",
        type=int,
        default=50,
        help="Context length for progressive decoder"
    )
    decode_parser.add_argument(
        "--tail-delay",
        type=int,
        default=50,
        help="Tail delay for progressive decoder"
    )
    decode_parser.add_argument(
        "--indices",
        type=str,
        default=None,
        help="Indices to decode (e.g., '0', '1,3,5', '0-10', '1,5-8,10'). If not specified, decodes all entries."
    )
    decode_parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite existing output files"
    )
    decode_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    setup_logging(args.verbose)

    if args.command == "list":
        list_command(args)
    elif args.command == "decode":
        decode_command(args)


if __name__ == "__main__":
    main()
