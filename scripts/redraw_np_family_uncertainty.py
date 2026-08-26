"""Redraw NP-family uncertainty metrics using ANP results for depth-aware ANP."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "np_family" / "np_family_uncertainty_geolink.png"
REGULAR = Path(r"C:\Windows\Fonts\arial.ttf")
BOLD = Path(r"C:\Windows\Fonts\arialbd.ttf")

WIDTH, HEIGHT = 2400, 1000
WHITE = (255, 255, 255)
TEXT = (38, 38, 38)
GRID = (229, 229, 229)
SPINE = (195, 195, 195)
ERROR = (66, 66, 66)
COLORS = {
    "NP": (23, 109, 156),
    "ANP": (195, 136, 32),
    "ANP + depth-aware": (21, 139, 106),
}
MODES = ["Single", "Block 20", "Block 100", "Profile"]
KEYS = ["single", "block20", "block100", "profile"]

# NP and standard-ANP values are retained from the existing chart. The
# depth-aware values and error bars are replaced with the exact ANP run shown
# in the supplied reference figure.
PICP = {
    "NP": {
        "single": (0.91, 0.03), "block20": (0.90, 0.03),
        "block100": (0.89, 0.04), "profile": (0.75, 0.04),
    },
    "ANP": {
        "single": (0.91, 0.02), "block20": (0.91, 0.03),
        "block100": (0.89, 0.04), "profile": (0.79, 0.05),
    },
    "ANP + depth-aware": {
        "single": (0.94990, 0.012396), "block20": (0.94668, 0.014859),
        "block100": (0.92656, 0.020830), "profile": (0.79854, 0.079279),
    },
}

MPIW = {
    "NP": {
        "single": (0.70, 0.07), "block20": (0.72, 0.07),
        "block100": (0.77, 0.07), "profile": (1.04, 0.08),
    },
    "ANP": {
        "single": (0.50, 0.05), "block20": (0.59, 0.06),
        "block100": (0.69, 0.08), "profile": (1.08, 0.09),
    },
    "ANP + depth-aware": {
        "single": (0.40, 0.03), "block20": (0.55, 0.06),
        "block100": (0.64, 0.07), "profile": (1.07, 0.20),
    },
}


def font(size, bold=False):
    return ImageFont.truetype(str(BOLD if bold else REGULAR), size=size)


def centered(draw, xy, text, text_font, fill=TEXT):
    x, y = xy
    box = draw.textbbox((0, 0), text, font=text_font)
    draw.text((x - (box[2] - box[0]) / 2,
               y - (box[3] - box[1]) / 2 - box[1]),
              text, font=text_font, fill=fill)


def right_aligned(draw, xy, text, text_font):
    x, y = xy
    box = draw.textbbox((0, 0), text, font=text_font)
    draw.text((x - (box[2] - box[0]),
               y - (box[3] - box[1]) / 2 - box[1]),
              text, font=text_font, fill=TEXT)


def vertical_text(image, center, text, text_font):
    box = text_font.getbbox(text)
    layer = Image.new("RGBA", (box[2] - box[0] + 20, box[3] - box[1] + 20), (0, 0, 0, 0))
    layer_draw = ImageDraw.Draw(layer)
    layer_draw.text((10 - box[0], 10 - box[1]), text, font=text_font, fill=TEXT)
    layer = layer.rotate(90, expand=True, resample=Image.Resampling.BICUBIC)
    image.paste(layer, (int(center[0] - layer.width / 2), int(center[1] - layer.height / 2)), layer)


def draw_legend(draw):
    centered(draw, (WIDTH / 2, 105), "Model / reference", font(31))
    labels = ["NP", "ANP", "ANP + depth-aware", "Nominal coverage (90%)"]
    legend_font = font(27)
    swatch, gap, between = 58, 15, 45
    widths = []
    for label in labels:
        box = draw.textbbox((0, 0), label, font=legend_font)
        widths.append(swatch + gap + box[2] - box[0])
    x = (WIDTH - sum(widths) - between * 3) / 2
    y = 163
    for label, width in zip(labels, widths):
        if label.startswith("Nominal"):
            dash_x = x
            while dash_x < x + swatch:
                draw.line((dash_x, y, min(dash_x + 10, x + swatch), y), fill=(213, 94, 0), width=4)
                dash_x += 17
        else:
            draw.rectangle((x, y - 12, x + swatch, y + 12), fill=COLORS[label])
        draw.text((x + swatch + gap, y - 17), label, font=legend_font, fill=TEXT)
        x += width + between


def draw_panel(image, bounds, title, ylabel, data, ymax, ticks, nominal=False):
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = bounds
    panel_w, panel_h = right - left, bottom - top

    centered(draw, ((left + right) / 2, top - 60), title, font(34))
    tick_font, mode_font, axis_font, value_font = font(28), font(27), font(31), font(22)

    def y_of(value):
        return bottom - value / ymax * panel_h

    for value in ticks:
        y = y_of(value)
        draw.line((left, y, right, y), fill=GRID, width=3)
        decimals = 2 if value in (0.25, 0.75) else 1
        right_aligned(draw, (left - 25, y), f"{value:.{decimals}f}", tick_font)
    draw.line((left, top, left, bottom), fill=SPINE, width=5)
    draw.line((left, bottom, right, bottom), fill=SPINE, width=5)

    if nominal:
        y = y_of(0.90)
        x = left
        while x < right:
            draw.line((x, y, min(x + 16, right), y), fill=(213, 94, 0), width=4)
            x += 27

    centers = [left + panel_w * (i + 0.5) / 4 for i in range(4)]
    bar_width = 62
    offsets = (-bar_width, 0, bar_width)
    for mode, key, center_x in zip(MODES, KEYS, centers):
        for model, offset in zip(COLORS, offsets):
            mean, std = data[model][key]
            x0, x1 = center_x + offset - bar_width / 2, center_x + offset + bar_width / 2
            y_mean = y_of(mean)
            draw.rectangle((x0, y_mean, x1, bottom), fill=COLORS[model])
            x_mid = center_x + offset
            y_high, y_low = y_of(min(ymax, mean + std)), y_of(max(0, mean - std))
            draw.line((x_mid, y_high, x_mid, y_low), fill=ERROR, width=7)
            draw.line((x_mid - 8, y_high, x_mid + 8, y_high), fill=ERROR, width=7)
            draw.line((x_mid - 8, y_low, x_mid + 8, y_low), fill=ERROR, width=7)
            centered(draw, (x_mid, y_mean - 17), f"{mean:.2f}", value_font)
        centered(draw, (center_x, bottom + 43), mode, mode_font)

    centered(draw, ((left + right) / 2, bottom + 95), "Missing pattern", axis_font)
    vertical_text(image, (left - 120, (top + bottom) / 2), ylabel, axis_font)


def main():
    image = Image.new("RGB", (WIDTH, HEIGHT), WHITE)
    draw = ImageDraw.Draw(image)
    centered(draw, (WIDTH / 2, 45), "Chất lượng độ bất định — Geolink", font(48, bold=True))
    draw_legend(draw)
    draw_panel(
        image, (195, 355, 1185, 815),
        "Độ bao phủ khoảng dự báo", "Coverage (PICP)",
        PICP, ymax=1.05, ticks=(0.0, 0.25, 0.50, 0.75, 1.0), nominal=True,
    )
    draw_panel(
        image, (1395, 355, 2385, 815),
        "Độ rộng trung bình của khoảng dự báo", "Interval width (MPIW)",
        MPIW, ymax=1.65, ticks=(0.0, 0.5, 1.0, 1.5),
    )
    image.save(OUTPUT, dpi=(180, 180), optimize=True)


if __name__ == "__main__":
    main()
