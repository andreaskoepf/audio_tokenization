from os import PathLike
from pathlib import Path
import tarfile
from typing import BinaryIO
import torch
import argparse
import json
import logging
import sys
from datetime import datetime
from torchcodec.decoders import AudioDecoder, AudioStreamMetadata
from boson_multimodal.audio_processing.higgs_audio_tokenizer import (
    load_higgs_audio_tokenizer,
)


def audio_metadata_to_dict(metadata: AudioStreamMetadata) -> dict:
    """
    Convert AudioStreamMetadata to a dictionary.

    Args:
        metadata: AudioStreamMetadata object

    Returns:
        Dictionary with all metadata properties
    """
    return {
        "sample_rate": metadata.sample_rate,
        "num_channels": metadata.num_channels,
        "duration_seconds_from_header": metadata.duration_seconds_from_header,
        "begin_stream_seconds_from_header": metadata.begin_stream_seconds_from_header,
        "bit_rate": metadata.bit_rate,
        "codec": metadata.codec,
        "stream_index": metadata.stream_index,
        "sample_format": metadata.sample_format,
    }


def load_audio_mono(
    uri: str | PathLike | BinaryIO, sample_rate: int | None = None, trim_zeros: bool = True,
) -> tuple[torch.Tensor, AudioStreamMetadata]:
    """
    Load audio file and convert to mono at the specified sample rate.

    Args:
        uri: Path to audio file or file-like object
        sample_rate: Target sample rate for output. If None, uses the original sample rate.

    Returns:
        Tuple of (audio_tensor, metadata)
        - audio_tensor: 1D tensor with shape (num_samples,) containing normalized float audio in [-1, 1]
        - metadata: AudioStreamMetadata object with original sample_rate, num_channels, etc.
    """
    # Create AudioDecoder with desired output sample rate
    decoder = AudioDecoder(uri, sample_rate=sample_rate)

    # Get original metadata before decoding
    metadata = decoder.metadata

    # Decode all samples
    audio_samples = decoder.get_all_samples()
    wav = audio_samples.data  # shape: (num_channels, num_samples)

    # Validate tensor dimensions
    if wav.ndim not in (1, 2):
        raise ValueError(f"Expected audio tensor to have 1 or 2 dimensions, but got {wav.ndim}")
    
    # convert to mono output
    if wav.ndim == 2:
        wav = wav.mean(dim=0)
    
    # ensure dynamic range doesn't exceed [-1,1] range
    max_value = wav.abs().max()
    if max_value.item() > 1.0:
        wav = wav / max_value

    if trim_zeros:
        # remove zero elements at the beginning and end
        nonzero_indices = torch.nonzero(wav).squeeze()
        if nonzero_indices.numel() > 0:
            first_nonzero_idx = nonzero_indices[0].item()
            last_nonzero_idx = nonzero_indices[-1].item()
            wav = wav[first_nonzero_idx:last_nonzero_idx + 1]
        elif wav.numel() > 0:
            # all zeros
            wav = wav[0:0]

    return wav, metadata


def determine_token_audio_length(audio_tokenizer, num_tokens: int = 1) -> int:
    x = torch.zeros(size=(1, 8, num_tokens), dtype=torch.int32)
    y = audio_tokenizer.decode(x)
    return y.shape[-1]


class PregressiveDecoder:
    def __init__(
        self,
        audio_tokenizer,
        samples_per_token: int,
        max_tokens_per_fragment: int,
        context_len: int = 12,
        tail_delay: int = 6,
    ):
        assert max_tokens_per_fragment > 0 and context_len > 0 and tail_delay >= 0
        self.audio_tokenizer = audio_tokenizer
        self.samples_per_token = samples_per_token
        self.max_tokens_per_fragment = max_tokens_per_fragment
        self.context_len = context_len
        self.tail_delay = tail_delay
        self.buffer = torch.empty(8, 0, dtype=torch.int32)
        self.first_block = True

    @torch.inference_mode()
    def add_tokens(
        self, tokens: torch.Tensor, last_block: bool = False
    ) -> torch.Tensor:
        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(-1)
        assert tokens.ndim == 2 and tokens.shape[0] == 8

        full_buffer = torch.cat((self.buffer, tokens.cpu()), dim=-1)

        num_tokens = full_buffer.shape[-1]

        if num_tokens < self.context_len + self.tail_delay:
            self.buffer = full_buffer
            return torch.empty(0)

        decoded_segment = (
            self.audio_tokenizer.decode(full_buffer.unsqueeze(0)).squeeze(0).squeeze(0)
        )
        decoded_segment_length = decoded_segment.shape[-1]

        if last_block:
            segment_end = decoded_segment_length
        else:
            segment_end = (
                decoded_segment_length - self.tail_delay * self.samples_per_token
            )

        if self.first_block:
            segment_begin = 0
            self.first_block = False
        else:
            segment_begin = self.context_len * self.samples_per_token

        output_segment = decoded_segment[segment_begin:segment_end]
        output_segment = torch.from_numpy(output_segment)

        self.buffer = full_buffer[:, -(self.context_len + self.tail_delay) :]

        return output_segment
    
    def decode_piecewise(self, audio_tokens: torch.Tensor) -> torch.Tensor:
        segments = []
        num_tokens = audio_tokens.shape[-1]
    
        for i in range(0, num_tokens, self.max_tokens_per_fragment):
            is_last_block = i + self.max_tokens_per_fragment >= num_tokens
            output_segment = self.add_tokens(audio_tokens[:, i:i+self.max_tokens_per_fragment], last_block=is_last_block)
            print(i, output_segment.shape, is_last_block)
            segments.append(output_segment)

        sample_reconstructed = torch.cat(segments)
        return sample_reconstructed


class TokenEncoder:
    def __init__(
        self,
        audio_tokenizer,
        samples_per_token: int,
        max_tokens_per_fragment: int,
        num_tokens_overlap: int,
        device: torch.DeviceObjType,
        sample_rate: int = 24_000,
    ):
        assert max_tokens_per_fragment > 2 * num_tokens_overlap
        self.audio_tokenizer = audio_tokenizer
        self.samples_per_token = samples_per_token
        self.max_tokens_per_fragment = max_tokens_per_fragment
        self.num_tokens_overlap = num_tokens_overlap
        self.device = device
        self.sample_rate = sample_rate

    @torch.inference_mode()
    def encode_piecewise(self, wav: torch.Tensor) -> torch.Tensor:
        assert wav.ndim == 1

        num_input_samples = wav.shape[0]
        num_fragment_samples = self.samples_per_token * self.max_tokens_per_fragment
        num_overlap_samples = self.samples_per_token * self.num_tokens_overlap

        token_list = []

        pos = 0
        frag_end = -1
        while frag_end < num_input_samples:
            frag_begin = max(pos - num_overlap_samples, 0)
            frag_end = frag_begin + num_fragment_samples
            fragment_input = wav[frag_begin:frag_end]
            fragment_tokens = self.audio_tokenizer.encode(
                fragment_input.unsqueeze(0).to(self.device), self.sample_rate
            )

            # remove head (if not first fargment)
            if pos > 0:
                fragment_tokens = fragment_tokens[:, self.num_tokens_overlap :]

            # remove tail (if not last fragment)
            if frag_end < num_input_samples:
                fragment_tokens = fragment_tokens[:, : -self.num_tokens_overlap]
            token_list.append(fragment_tokens.cpu())

            pos += fragment_tokens.shape[-1] * self.samples_per_token

        tokens_output = torch.cat(token_list, dim=-1)
        return tokens_output

    def encode_piecewise_file(
        self, uri: str | PathLike | BinaryIO
    ) -> tuple[torch.Tensor, AudioStreamMetadata]:
        x, metadata = load_audio_mono(uri, sample_rate=self.sample_rate)
        return self.encode_piecewise(x), metadata

    def process_tar_file(self, fn: str | PathLike, output_jsonl: str | PathLike, extensions=(".mp3",)):
        """Process a tar file and write results to JSONL output.

        Supports resumable processing:
        - Uses a .partial file to track progress during processing
        - Each audio file is written immediately after processing
        - On resume, reads .partial file to skip already-processed files
        - Final output is written to the main JSONL file when complete
        """
        tar_path = Path(fn)
        output_path = Path(output_jsonl)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Use a .partial file for incremental progress tracking
        partial_path = output_path.with_suffix('.jsonl.partial')

        # Load already-processed files from partial file if it exists
        processed_files = set()
        results = []

        if partial_path.exists():
            logging.info(f"Found partial progress file: {partial_path}")
            try:
                with open(partial_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            result = json.loads(line)
                            results.append(result)
                            processed_files.add(result['audio_file'])
                logging.info(f"Resumed with {len(processed_files)} already-processed files")
            except Exception as ex:
                logging.warning(f"Failed to load partial progress file: {ex}. Starting from scratch.")
                processed_files = set()
                results = []

        logging.info(f"Processing tar file: {tar_path}")

        # Open partial file in append mode for incremental writing
        with tarfile.open(fn, mode="r") as tar, open(partial_path, 'a') as partial_f:
            members = [m for m in tar.getmembers() if m.isfile() and m.name.lower().endswith(extensions)]
            total_members = len(members)
            already_processed = len(processed_files)
            remaining = total_members - already_processed

            logging.info(f"Found {total_members} audio files in {tar_path.name}")
            if already_processed > 0:
                logging.info(f"Already processed: {already_processed}, Remaining: {remaining}")

            for idx, member in enumerate(members, 1):
                # Skip if already processed
                if member.name in processed_files:
                    logging.debug(f"Skipping already-processed [{idx}/{total_members}]: {member.name}")
                    continue

                logging.info(f"Processing [{idx}/{total_members}]: {member.name}")

                file_obj = tar.extractfile(member)
                if file_obj:
                    try:
                        # Load audio and get metadata
                        wav, metadata = load_audio_mono(file_obj, sample_rate=self.sample_rate)
                        num_samples = wav.shape[0]

                        logging.info(f"Input {member.name}: {num_samples} samples (original SR: {metadata.sample_rate} Hz, channels: {metadata.num_channels})")

                        # Encode to tokens
                        tokens = self.encode_piecewise(wav)

                        # Convert to list for JSON serialization
                        tokens_list = tokens.squeeze(0).cpu().tolist()

                        result = {
                            "tar_file": str(tar_path),
                            "audio_file": member.name,
                            "num_samples": num_samples,
                            "sample_rate": self.sample_rate,
                            "original_metadata": audio_metadata_to_dict(metadata),
                            "tokens": tokens_list,
                            "token_shape": list(tokens.shape)
                        }
                        results.append(result)

                        # Write immediately to partial file
                        partial_f.write(json.dumps(result) + '\n')
                        partial_f.flush()  # Ensure it's written to disk

                        logging.info(f"Successfully tokenized {member.name}: {num_samples} samples -> {tokens.shape}")

                    except Exception as ex:
                        logging.error(f"Failed to process {member.name}: {ex}", exc_info=True)
                        result = {
                            "tar_file": str(tar_path),
                            "audio_file": member.name,
                            "error": str(ex)
                        }
                        results.append(result)

                        # Write error to partial file as well
                        partial_f.write(json.dumps(result) + '\n')
                        partial_f.flush()

        # All processing complete - write final output
        logging.info(f"Writing final output to {output_path}")
        with open(output_path, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')

        # Remove partial file after successful completion
        if partial_path.exists():
            partial_path.unlink()
            logging.info(f"Removed partial progress file: {partial_path}")

        logging.info(f"Wrote {len(results)} results to {output_path}")
        return results


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
        handlers=handlers
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Tokenize audio files from tar archives using Higgs Audio Tokenizer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default="bosonai/higgs-audio-v2-tokenizer",
        help="Path to the audio tokenizer model (e.g., local path to bosonai/higgs-audio-v2-tokenizer)"
    )

    parser.add_argument(
        "--input-tar",
        type=str,
        required=True,
        help="Path to input tar file containing mp3 files"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for output JSONL files"
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
        default=25_000,  # 1000 seconds at 25 tokens/sec
        help="Maximum tokens per fragment for chunked processing"
    )

    parser.add_argument(
        "--num-tokens-overlap",
        type=int,
        default=250,  # 10 second overlap
        help="Number of tokens to overlap between fragments"
    )

    parser.add_argument(
        "--extensions",
        type=str,
        nargs="+",
        default=[".mp3"],
        help="Audio file extensions to process"
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
        "--skip-existing",
        action="store_true",
        help="Skip processing if output file already exists"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite existing output files without prompting"
    )

    return parser.parse_args()


def main():
    """Main entry point for the tokenization tool."""
    args = parse_args()

    # Setup logging
    setup_logging(args.log_file, args.verbose)

    logging.info("=" * 80)
    logging.info("Audio Tokenization Tool")
    logging.info("=" * 80)
    logging.info(f"Model path: {args.model_path}")
    logging.info(f"Input tar: {args.input_tar}")
    logging.info(f"Output dir: {args.output_dir}")
    logging.info(f"GPU ID: {args.gpu_id}")
    logging.info(f"Sample rate: {args.sample_rate}")
    logging.info(f"Max tokens per fragment: {args.max_tokens_per_fragment}")
    logging.info(f"Token overlap: {args.num_tokens_overlap}")

    # Check if input tar exists
    input_tar_path = Path(args.input_tar)
    if not input_tar_path.exists():
        logging.error(f"Input tar file not found: {input_tar_path}")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine output JSONL filename (same as tar but with .jsonl extension)
    output_jsonl = output_dir / f"{input_tar_path.stem}.jsonl"

    # Check if output already exists
    if output_jsonl.exists():
        if args.skip_existing:
            logging.info(f"Output file already exists, skipping: {output_jsonl}")
            sys.exit(0)
        elif args.force:
            logging.warning(f"Output file already exists, overwriting: {output_jsonl}")
        else:
            logging.warning(f"Output file already exists: {output_jsonl}")
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

    # Process tar file
    try:
        results = encoder.process_tar_file(
            args.input_tar,
            output_jsonl,
            extensions=tuple(args.extensions)
        )

        # Summary
        successful = sum(1 for r in results if "error" not in r)
        failed = len(results) - successful

        elapsed = datetime.now() - start_time
        logging.info("=" * 80)
        logging.info("Processing complete!")
        logging.info(f"Total files: {len(results)}")
        logging.info(f"Successful: {successful}")
        logging.info(f"Failed: {failed}")
        logging.info(f"Elapsed time: {elapsed}")
        logging.info(f"Output: {output_jsonl}")
        logging.info("=" * 80)

    except Exception as e:
        logging.error(f"Processing failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
