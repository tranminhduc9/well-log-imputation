"""Update legacy ANP labels in exported result PNGs.

The plots are edited only in their title/legend regions so the plotted data
pixels remain unchanged.  This is safer than reconstructing values from the
rasterized bar heights or curves.
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FONT_PATH = Path(
    r"C:\Users\Admin\.cache\codex-runtimes\codex-primary-runtime\dependencies"
    r"\native\poppler\Library\share\fonts\DejaVuSans.ttf"
)

WHITE = (255, 255, 255)
TEXT = (38, 38, 38)
GRID = (204, 204, 204)


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_PATH), size=size)


def centered_text(draw: ImageDraw.ImageDraw, box, text: str, text_font, fill=TEXT):
    left, top, right, bottom = box
    bounds = draw.textbbox((0, 0), text, font=text_font)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    x = left + (right - left - width) / 2
    y = top + (bottom - top - height) / 2 - bounds[1]
    draw.text((x, y), text, font=text_font, fill=fill)


def dashed_line(draw, xy, fill, width=3, dash=10, gap=6):
    x1, y, x2, _ = xy
    x = x1
    while x < x2:
        draw.line((x, y, min(x + dash, x2), y), fill=fill, width=width)
        x += dash + gap


def dashdot_line(draw, xy, fill, width=3):
    x1, y, x2, _ = xy
    x = x1
    while x < x2:
        draw.line((x, y, min(x + 12, x2), y), fill=fill, width=width)
        x += 17
        if x < x2:
            draw.ellipse((x, y - 1, x + 2, y + 1), fill=fill)
        x += 7


def update_metric_legend(path: Path):
    image = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(image)

    # The chart itself starts at y=90. Only replace the header/legend strip.
    draw.rectangle((0, 0, image.width, 89), fill=WHITE)
    centered_text(draw, (0, 10, image.width, 43), "Model", font(25))

    labels = [
        "LOCF",
        "QRF",
        "BayesNN",
        "ANP + Depth Aware",
        "XGBoost",
        "U-Net",
        "SAITS",
    ]
    colors = [
        (23, 109, 156),
        (195, 136, 32),
        (21, 139, 106),
        (186, 97, 27),
        (194, 130, 181),
        (189, 146, 110),
        (242, 184, 224),
    ]
    label_font = font(18)
    patch_width, patch_height = 34, 13
    patch_gap, item_gap = 10, 22
    widths = []
    for label in labels:
        bbox = draw.textbbox((0, 0), label, font=label_font)
        widths.append(patch_width + patch_gap + bbox[2] - bbox[0])
    total_width = sum(widths) + item_gap * (len(labels) - 1)
    x = (image.width - total_width) / 2
    patch_y = 61
    for label, color, item_width in zip(labels, colors, widths):
        draw.rectangle(
            (x, patch_y - patch_height / 2, x + patch_width, patch_y + patch_height / 2),
            fill=color,
        )
        draw.text((x + patch_width + patch_gap, 49), label, font=label_font, fill=TEXT)
        x += item_width + item_gap

    image.save(path, optimize=True)


def update_comparison_curve(path: Path):
    image = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(image)

    draw.rectangle((0, 0, image.width, 36), fill=WHITE)
    centered_text(
        draw,
        (0, 1, image.width, 35),
        "U-Net vs ANP + Depth Aware — sequence 0, feature 3",
        font(25),
    )

    legend = (123, 48, 879, 94)
    draw.rounded_rectangle(legend, radius=5, fill=WHITE, outline=GRID, width=2)
    legend_font = font(22)
    y = 70
    draw.line((134, y, 179, y), fill=(0, 0, 0), width=3)
    draw.text((197, 55), "Ground truth", font=legend_font, fill=TEXT)
    dashed_line(draw, (392, y, 437, y), fill=(0, 114, 178), width=3)
    draw.text((455, 55), "U-Net", font=legend_font, fill=TEXT)
    dashdot_line(draw, (568, y, 613, y), fill=(0, 158, 115), width=3)
    draw.text((631, 55), "ANP + Depth Aware", font=legend_font, fill=TEXT)

    image.save(path, optimize=True)


def update_anp_uncertainty(path: Path):
    image = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(image)

    draw.rectangle((0, 0, image.width, 40), fill=WHITE)
    centered_text(
        draw,
        (0, 1, image.width, 37),
        "ANP + Depth Aware — sequence 0, feature 3",
        font(25),
    )

    # Remove the complete legacy legend (it extended lower than its visible
    # border after the first label pass), then draw a clean two-row legend.
    draw.rectangle((625, 41, 1347, 126), fill=WHITE)
    legend = (626, 44, 1346, 122)
    draw.rounded_rectangle(legend, radius=5, fill=WHITE, outline=GRID, width=2)
    legend_font = font(19)

    # Predictive interval swatch.
    draw.rectangle((638, 55, 684, 72), fill=(210, 238, 231), outline=(146, 215, 199))
    draw.text((701, 51), "90% predictive interval", font=legend_font, fill=TEXT)

    # Depth-aware ANP point-prediction line.
    dashdot_line(draw, (1004, 64, 1050, 64), fill=(0, 158, 115), width=3)
    draw.text((1065, 51), "ANP + Depth Aware", font=legend_font, fill=TEXT)

    draw.line((638, 103, 684, 103), fill=(0, 0, 0), width=3)
    draw.text((701, 90), "Ground truth", font=legend_font, fill=TEXT)

    image.save(path, optimize=True)


def main():
    for metric in ("cc", "mae", "mse", "r2", "rmse"):
        update_metric_legend(RESULTS / f"{metric}.png")
    update_comparison_curve(RESULTS / "unet_anp_log_comparison.png")
    update_anp_uncertainty(RESULTS / "anp_uncertainty_comparison.png")


if __name__ == "__main__":
    main()
