"""Localize exported well-log result charts to Vietnamese without changing data pixels."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
NP_DIR = ROOT / "np_family"
RESULTS_DIR = ROOT / "results"
REGULAR = Path(r"C:\Windows\Fonts\arial.ttf")
BOLD = Path(r"C:\Windows\Fonts\arialbd.ttf")
WHITE = (255, 255, 255)
TEXT = (38, 38, 38)
GRID = (204, 204, 204)


def font(size, bold=False):
    return ImageFont.truetype(str(BOLD if bold else REGULAR), size=size)


def centered(draw, box, text, text_font, fill=TEXT):
    left, top, right, bottom = box
    bounds = draw.textbbox((0, 0), text, font=text_font)
    width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
    draw.text(
        (left + (right - left - width) / 2, top + (bottom - top - height) / 2 - bounds[1]),
        text, font=text_font, fill=fill,
    )


def vertical_text(image, box, text, text_font):
    left, top, right, bottom = box
    bounds = text_font.getbbox(text)
    layer = Image.new("RGBA", (bounds[2] - bounds[0] + 20, bounds[3] - bounds[1] + 20), (0, 0, 0, 0))
    layer_draw = ImageDraw.Draw(layer)
    layer_draw.text((10 - bounds[0], 10 - bounds[1]), text, font=text_font, fill=TEXT)
    layer = layer.rotate(90, expand=True, resample=Image.Resampling.BICUBIC)
    x = left + (right - left - layer.width) // 2
    y = top + (bottom - top - layer.height) // 2
    image.paste(layer, (x, y), layer)


def dashed_line(draw, x1, x2, y, color, width=3):
    x = x1
    while x < x2:
        draw.line((x, y, min(x + 10, x2), y), fill=color, width=width)
        x += 16


def localize_np_metric(path: Path, metric: str):
    image = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(image)

    draw.rectangle((0, 0, image.width, 147), fill=WHITE)
    draw.rectangle((0, 148, 54, 600), fill=WHITE)
    draw.rectangle((139, 600, 786, image.height), fill=WHITE)
    draw.rectangle((790, 245, image.width, 475), fill=WHITE)

    centered(draw, (0, 4, image.width, 72), f"Họ Neural Process — {metric.upper()}", font(43))
    centered(draw, (139, 92, 786, 145), "Geolink", font(36))
    vertical_text(image, (0, 160, 55, 590), f"Mean {metric.upper()}", font(31))

    mode_font = font(29)
    for x, label in zip((220, 382, 543, 705), ("Single", "Block 20", "Block 100", "Profile")):
        centered(draw, (x - 90, 610, x + 90, 660), label, mode_font)
    centered(draw, (139, 661, 786, 710), "Missing pattern", font(31))

    centered(draw, (790, 252, 1248, 304), "Model", font(34))
    entries = [
        ((23, 109, 156), "NP"),
        ((195, 136, 32), "ANP"),
        ((21, 139, 106), "ANP + depth-aware"),
    ]
    for y, (color, label) in zip((327, 378, 429), entries):
        draw.rectangle((805, y - 10, 867, y + 10), fill=color)
        draw.text((890, y - 20), label, font=font(28), fill=TEXT)
    image.save(path, optimize=True)


def localize_result_metric(path: Path, metric: str):
    image = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(image)

    draw.rectangle((0, 0, image.width, 89), fill=WHITE)
    draw.rectangle((205, 95, 250, 438), fill=WHITE)
    draw.rectangle((305, 443, 890, image.height), fill=WHITE)

    centered(draw, (0, 5, image.width, 40), "Model", font(24))
    labels = ["LOCF", "QRF", "BayesNN", "ANP + Depth-aware", "XGBoost", "U-Net", "SAITS"]
    colors = [(23,109,156),(195,136,32),(21,139,106),(186,97,27),(194,130,181),(189,146,110),(242,184,224)]
    legend_font = font(15)
    patch_w, patch_h, gap, item_gap = 28, 11, 7, 16
    widths = []
    for label in labels:
        b = draw.textbbox((0, 0), label, font=legend_font)
        widths.append(patch_w + gap + b[2] - b[0])
    x = (image.width - sum(widths) - item_gap * 6) / 2
    for label, color, width in zip(labels, colors, widths):
        draw.rectangle((x, 57, x + patch_w, 57 + patch_h), fill=color)
        draw.text((x + patch_w + gap, 52), label, font=legend_font, fill=TEXT)
        x += width + item_gap

    vertical_text(image, (205, 105, 250, 430), f"Mean {metric.upper()}", font(24))
    for x, label in zip((379, 523, 668, 812), ("Single", "Block 20", "Block 100", "Profile")):
        centered(draw, (x - 72, 453, x + 72, 484), label, font(22))
    centered(draw, (307, 484, 884, 518), "Missing pattern", font(24))
    image.save(path, optimize=True)


def localize_np_uncertainty(path: Path):
    image = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(image)

    draw.rectangle((0, 0, image.width, 200), fill=WHITE)
    draw.rectangle((0, 285, image.width, 345), fill=WHITE)
    draw.rectangle((0, 350, 70, 800), fill=WHITE)
    draw.rectangle((1190, 340, 1383, 812), fill=WHITE)
    draw.rectangle((190, 800, 1190, image.height), fill=WHITE)
    draw.rectangle((1380, 800, 2380, image.height), fill=WHITE)

    centered(draw, (0, 5, image.width, 82), "Chất lượng độ bất định — Geolink", font(49, bold=True))
    centered(draw, (0, 88, image.width, 124), "Model / reference", font(27))
    entries = [
        ((23,109,156), "NP"), ((195,136,32), "ANP"),
        ((21,139,106), "ANP + depth-aware"), (None, "Nominal coverage (90%)"),
    ]
    legend_font = font(25)
    widths = []
    for _, label in entries:
        b = draw.textbbox((0, 0), label, font=legend_font)
        widths.append(75 + b[2] - b[0])
    x = (image.width - sum(widths) - 45 * 3) / 2
    for (color, label), width in zip(entries, widths):
        if color is None:
            dashed_line(draw, x, x + 55, 160, (213,94,0), width=3)
        else:
            draw.rectangle((x, 148, x + 55, 172), fill=color)
        draw.text((x + 70, 143), label, font=legend_font, fill=TEXT)
        x += width + 45

    centered(draw, (195, 286, 1188, 342), "Độ bao phủ khoảng dự báo", font(34))
    centered(draw, (1385, 286, 2378, 342), "Độ rộng trung bình của khoảng dự báo", font(34))
    vertical_text(image, (0, 360, 70, 790), "Coverage (PICP)", font(31))
    vertical_text(image, (1192, 350, 1298, 790), "Interval width (MPIW)", font(31))
    draw.rectangle((1285, 788, 1325, 815), fill=WHITE)
    for y, label in ((798, "0.0"), (628, "0.5"), (458, "1.0")):
        bounds = draw.textbbox((0, 0), label, font=font(31))
        draw.text((1365 - (bounds[2] - bounds[0]), y - 18), label, font=font(31), fill=TEXT)

    for centers in ((319,567,815,1063), (1508,1756,2004,2252)):
        for x, label in zip(centers, ("Single", "Block 20", "Block 100", "Profile")):
            centered(draw, (x - 112, 816, x + 112, 866), label, font(28))
    centered(draw, (195, 870, 1188, 930), "Missing pattern", font(31))
    centered(draw, (1385, 870, 2378, 930), "Missing pattern", font(31))
    image.save(path, optimize=True)


def localize_unet_anp(path: Path):
    image = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 37), fill=WHITE)
    centered(draw, (0, 0, image.width, 36), "U-Net và ANP + Depth-aware — chuỗi 0, đặc trưng 3", font(22))
    draw.rounded_rectangle((123, 48, 980, 94), radius=5, fill=WHITE, outline=GRID, width=2)
    draw.line((134,70,179,70), fill=(0,0,0), width=3)
    draw.text((196,55), "Ground truth", font=font(19), fill=TEXT)
    dashed_line(draw, 365, 410, 70, (0,114,178), width=3)
    draw.text((427,55), "U-Net", font=font(19), fill=TEXT)
    dashed_line(draw, 533, 578, 70, (0,158,115), width=3)
    draw.text((595,55), "ANP + Depth-aware", font=font(19), fill=TEXT)
    draw.rectangle((0, 95, 51, 470), fill=WHITE)
    vertical_text(image, (0, 105, 51, 460), "Normalized log value", font(22))
    draw.rectangle((390, 507, 1000, image.height), fill=WHITE)
    centered(draw, (390, 507, 1000, 545), "Depth/sample index within sequence", font(22))
    image.save(path, optimize=True)


def localize_interval_chart(path: Path, model: str):
    image = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(image)
    title = f"{model} — chuỗi 0, đặc trưng 3"
    draw.rectangle((0, 0, image.width, 40), fill=WHITE)
    centered(draw, (0, 0, image.width, 38), title, font(23))
    draw.rectangle((0, 95, 52, 490), fill=WHITE)
    vertical_text(image, (0, 105, 52, 480), "Normalized log value", font(22))
    draw.rectangle((390, 527, 1000, image.height), fill=WHITE)
    centered(draw, (390, 527, 1000, 565), "Depth/sample index within sequence", font(22))

    if model.startswith("BayesNN"):
        draw.rounded_rectangle((123, 399, 1040, 477), radius=5, fill=WHITE, outline=GRID, width=2)
        draw.rectangle((133, 410, 179, 426), fill=(210,229,239), outline=(140,194,222))
        draw.text((197,402), "90% predictive interval", font=font(18), fill=TEXT)
        dashed_line(draw, 510, 555, 418, (0,114,178), width=3)
        draw.text((571,402), "BayesNN point prediction", font=font(18), fill=TEXT)
        draw.line((133,454,179,454), fill=(0,0,0), width=3)
        draw.text((197,438), "Ground truth", font=font(18), fill=TEXT)
    else:
        draw.rectangle((625, 41, 1347, 126), fill=WHITE)
        draw.rounded_rectangle((626, 44, 1346, 122), radius=5, fill=WHITE, outline=GRID, width=2)
        draw.rectangle((638,55,684,72), fill=(210,238,231), outline=(146,215,199))
        draw.text((701,51), "90% predictive interval", font=font(18), fill=TEXT)
        dashed_line(draw, 1000, 1045, 64, (0,158,115), width=3)
        draw.text((1060,51), "Point prediction", font=font(18), fill=TEXT)
        draw.line((638,103,684,103), fill=(0,0,0), width=3)
        draw.text((701,90), "Ground truth", font=font(18), fill=TEXT)
    image.save(path, optimize=True)


def main():
    for metric in ("cc", "mae", "mse", "r2"):
        localize_np_metric(NP_DIR / f"np_family_{metric}.png", metric)
    localize_np_uncertainty(NP_DIR / "np_family_uncertainty_geolink.png")

    for metric in ("cc", "mae", "mse", "r2", "rmse"):
        localize_result_metric(RESULTS_DIR / f"{metric}.png", metric)
    localize_unet_anp(RESULTS_DIR / "unet_anp_log.png")
    localize_interval_chart(RESULTS_DIR / "bayesnn_uncertainty_comparison.png", "BayesNN")
    localize_interval_chart(RESULTS_DIR / "anp_uncertainty_comparison.png", "ANP + Depth-aware")


if __name__ == "__main__":
    main()
