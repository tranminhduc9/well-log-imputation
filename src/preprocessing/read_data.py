"""Read raw well-log data from LAS files."""

from pathlib import Path

import lasio
import pandas as pd


def _decoded_frame(las: lasio.LASFile) -> pd.DataFrame:
    """Decode GeoLink's additional -999 curve sentinel after LAS NULL parsing.

    The first column is the depth index and is deliberately left untouched.
    Other negative values (for example apparent neutron porosity) are preserved.
    """
    frame = las.df().reset_index()
    curve_columns = frame.columns[1:]
    frame[curve_columns] = frame[curve_columns].replace(-999.0, float("nan"))
    return frame


def read_data(file_path: str | Path) -> pd.DataFrame:
    """
    Read raw well-log data from a las file.

    Args:
        file_path: Path to the LAS file containing well-log data.

    Returns:
        DataFrame containing the depth and well-log curves. LAS null values
        and GeoLink's additional ``-999.0`` curve sentinel are represented as
        ``NaN``. Depth values and other negative measurements are retained.

    Raises:
        FileNotFoundError: If ``file_path`` does not point to a file.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"LAS file not found: {path}")

    las = lasio.read(path)
    return _decoded_frame(las)


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
        frame = _decoded_frame(las)
        well_item = next(
            (item for item in las.well if item.mnemonic.upper() == "WELL"), None
        )
        well_name = well_item.value if well_item and well_item.value else path.stem
        frame["WELL"] = well_name
        frame["SOURCE_FILE"] = path.name
        frames.append(frame)

    return pd.concat(frames, ignore_index=True, sort=False)
