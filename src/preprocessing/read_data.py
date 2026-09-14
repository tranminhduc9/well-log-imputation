"""Read raw well-log data from LAS files."""

from pathlib import Path
from typing import Any

import lasio
import pandas as pd


def read_data(file_path: str | Path) -> pd.DataFrame:
    """
    Read raw well-log data from a las file.

    Args:
        file_path: Path to the LAS file containing well-log data.

    Returns:
        DataFrame containing the depth and well-log curves. LAS null values are
        represented as ``NaN``.

    Raises:
        FileNotFoundError: If ``file_path`` does not point to a file.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"LAS file not found: {path}")

    las = lasio.read(path)
    return las.df().reset_index()


def extract_metadata(file_path: str | Path) -> dict[str, Any]:
    """Extract all header metadata from a LAS file.

    Args:
        file_path: Path to the LAS file containing well-log data.

    Returns:
        Metadata from the Version, Well, Curve, Parameter and Other sections.
        Each header item retains its value, unit and description.

    Raises:
        FileNotFoundError: If ``file_path`` does not point to a file.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"LAS file not found: {path}")

    las = lasio.read(path, ignore_data=True)

    metadata = {}
    for section_name, section in (
        ("version", las.version),
        ("well", las.well),
        ("curves", las.curves),
        ("parameters", las.params),
    ):
        metadata[section_name] = {
            item.mnemonic: {
                "value": item.value,
                "unit": item.unit,
                "description": item.descr,
            }
            for item in section
        }

    metadata["other"] = las.other
    return metadata


def read_data_directory(directory_path: str | Path) -> pd.DataFrame:
    """Read and combine every LAS file in a directory.

    The files may contain different curves. Missing curves in an individual
    well are represented as ``NaN`` in the combined DataFrame.

    Args:
        directory_path: Directory containing LAS files.

    Returns:
        Combined well-log data with ``WELL`` and ``SOURCE_FILE`` columns.

    Raises:
        NotADirectoryError: If ``directory_path`` is not a directory.
        FileNotFoundError: If the directory contains no LAS files.
    """
    directory = Path(directory_path)
    if not directory.is_dir():
        raise NotADirectoryError(f"LAS directory not found: {directory}")

    las_files = sorted(
        path for path in directory.iterdir() if path.suffix.lower() == ".las"
    )
    if not las_files:
        raise FileNotFoundError(f"No LAS files found in: {directory}")

    frames = []
    for path in las_files:
        las = lasio.read(path)
        frame = las.df().reset_index()
        well_item = next(
            (item for item in las.well if item.mnemonic.upper() == "WELL"), None
        )
        well_name = well_item.value if well_item and well_item.value else path.stem
        frame["WELL"] = well_name
        frame["SOURCE_FILE"] = path.name
        frames.append(frame)

    return pd.concat(frames, ignore_index=True, sort=False)
