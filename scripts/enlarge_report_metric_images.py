"""Crop excess whitespace and upscale report metric charts uniformly."""

from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
METRICS = ("cc", "mae", "mse", "r2")

# All four notebook exports share this geometry. The crop retains every
# non-white pixel plus a 15 px safety margin, while removing 435 px of unused
# horizontal canvas that made the chart look small in a two-column report.
CROP_BOX = (204, 0, 1015, 521)
SCALE = 3
TARGET_SIZE = ((CROP_BOX[2] - CROP_BOX[0]) * SCALE,
               (CROP_BOX[3] - CROP_BOX[1]) * SCALE)


def enlarge(path: Path):
    image = Image.open(path).convert("RGB")
    if image.size == TARGET_SIZE:
        return
    if image.size != (1246, 521):
        raise ValueError(f"Unexpected source size for {path}: {image.size}")
    image = image.crop(CROP_BOX)
    image = image.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
    image.save(path, dpi=(300, 300), optimize=True)


def main():
    for metric in METRICS:
        enlarge(RESULTS / f"{metric}.png")


if __name__ == "__main__":
    main()
