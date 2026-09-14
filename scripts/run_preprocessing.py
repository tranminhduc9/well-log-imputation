"""Command-line entry point for the well-log preprocessing pipeline.

By default, this script reads LAS files from ``data/raw`` and writes the
processed train/validation/test artifacts to ``data/processed``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_RAW_DIRECTORY = PROJECT_ROOT / "data" / "raw"
DEFAULT_PROCESSED_DIRECTORY = PROJECT_ROOT / "data" / "processed"


def parse_arguments() -> argparse.Namespace:
    """Parse preprocessing options from the command line."""
    parser = argparse.ArgumentParser(
        description=(
            "Read raw LAS files, split by well, remove outliers, normalize "
            "the logs, create fixed-length segments and save the results."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_RAW_DIRECTORY,
        help=f"Directory containing LAS files (default: {DEFAULT_RAW_DIRECTORY}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIRECTORY,
        help=(
            "Directory in which processed artifacts are saved "
            f"(default: {DEFAULT_PROCESSED_DIRECTORY})."
        ),
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--lower-quantile", type=float, default=0.01)
    parser.add_argument("--upper-quantile", type=float, default=0.99)
    parser.add_argument("--segment-length", type=int, default=256)
    parser.add_argument("--random-state", type=int, default=912)
    return parser.parse_args()


def main() -> None:
    """Run preprocessing and print a compact summary of the saved datasets."""
    arguments = parse_arguments()
    input_directory = arguments.input_dir.resolve()
    output_directory = arguments.output_dir.resolve()

    from src.preprocessing.pipeline import run_pipeline_from_directory

    result = run_pipeline_from_directory(
        directory_path=input_directory,
        output_directory=output_directory,
        train_ratio=arguments.train_ratio,
        validation_ratio=arguments.validation_ratio,
        test_ratio=arguments.test_ratio,
        lower_quantile=arguments.lower_quantile,
        upper_quantile=arguments.upper_quantile,
        segment_length=arguments.segment_length,
        random_state=arguments.random_state,
    )

    print(f"Input directory:  {input_directory}")
    print(f"Output directory: {output_directory}")
    for split_name in ("train", "validation", "test"):
        split = result[split_name]
        segment_count = split["segments"].shape[0]
        well_count = split["metadata"]["WELL"].nunique()
        print(
            f"{split_name:>10}: {segment_count:>5} segments, "
            f"{well_count:>3} wells, shape={split['segments'].shape}"
        )


if __name__ == "__main__":
    main()
