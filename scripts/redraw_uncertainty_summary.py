"""Redraw the Geolink uncertainty summary with a non-overlapping legend."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "uncertainty_summary.png"
REGULAR = Path(r"C:\Windows\Fonts\arial.ttf")
BOLD = Path(r"C:\Windows\Fonts\arialbd.ttf")

WIDTH, HEIGHT = 2800, 1700
WHITE = (255, 255, 255)
TEXT = (38, 38, 38)
GRID = (229, 229, 229)
SPINE = (195, 195, 195)
ERROR = (66, 66, 66)
COLORS = {
    "ANP + Depth-aware": (20, 138, 106),
    "BayesNN": (22, 108, 156),
    "QRF": (201, 148, 29),
}

MODES = ["Single", "Block 20", "Block 100", "Profile"]
MODE_KEYS = ["single", "block20", "block100", "profile"]

# Exact mean/std values used by the notebook output.
PICP = {
    "ANP + Depth-aware": {
        "single": (0.94990, 0.012396), "block20": (0.94668, 0.014859),
        "block100": (0.92656, 0.020830), "profile": (0.79854, 0.079279),
    },
    "BayesNN": {
        "single": (0.89870, 0.007496), "block20": (0.89700, 0.018644),
        "block100": (0.89260, 0.020373), "profile": (0.89502, 0.020086),
    },
    "QRF": {
        "single": (0.54652, 0.023216), "block20": (0.54362, 0.035885),
        "block100": (0.54642, 0.040969), "profile": (0.55352, 0.033122),
    },
}

MPIW = {
    "ANP + Depth-aware": {
        "single": (0.40, 0.03), "block20": (0.55, 0.06),
        "block100": (0.64, 0.07), "profile": (1.07, 0.20),
    },
    "BayesNN": {
        "single": (1.67592, 0.044386), "block20": (1.68420, 0.059152),
        "block100": (1.66664, 0.032244), "profile": (1.66528, 0.046356),
    },
    "QRF": {
        "single": (0.71794, 0.038649), "block20": (0.72776, 0.033439),
        "block100": (0.71804, 0.025835), "profile": (0.72650, 0.026805),
    },
}


def font(size, bold=False):
    return ImageFont.truetype(str(BOLD if bold else REGULAR), size=size)


def centered(draw, xy, text, text_font, fill=TEXT):
    x, y = xy
    box = draw.textbbox((0, 0), text, font=text_font)
    draw.text((x - (box[2] - box[0]) / 2, y - (box[3] - box[1]) / 2 - box[1]),
              text, font=text_font, fill=fill)


def right_aligned(draw, xy, text, text_font, fill=TEXT):
    x, y = xy
    box = draw.textbbox((0, 0), text, font=text_font)
    draw.text((x - (box[2] - box[0]), y - (box[3] - box[1]) / 2 - box[1]),
              text, font=text_font, fill=fill)


def vertical_text(image, center, text, text_font):
    box = text_font.getbbox(text)
    layer = Image.new("RGBA", (box[2] - box[0] + 20, box[3] - box[1] + 20), (0, 0, 0, 0))
    layer_draw = ImageDraw.Draw(layer)
    layer_draw.text((10 - box[0], 10 - box[1]), text, font=text_font, fill=TEXT)
    layer = layer.rotate(90, expand=True, resample=Image.Resampling.BICUBIC)
    image.paste(layer, (int(center[0] - layer.width / 2), int(center[1] - layer.height / 2)), layer)


def draw_legend(draw):
    centered(draw, (WIDTH / 2, 105), "Model / reference", font(34))
    entries = ["ANP + Depth-aware", "BayesNN", "QRF", "Nominal coverage (90%)"]
    legend_font = font(31)
    swatch_width = 64
    item_gap = 62
    internal_gap = 18
    widths = []
    for label in entries:
        box = draw.textbbox((0, 0), label, font=legend_font)
        widths.append(swatch_width + internal_gap + box[2] - box[0])
    total = sum(widths) + item_gap * (len(entries) - 1)
    x = (WIDTH - total) / 2
    y = 174
    for label, item_width in zip(entries, widths):
        if label.startswith("Nominal"):
            for dash_x in range(int(x), int(x + swatch_width), 18):
                draw.line((dash_x, y, min(dash_x + 11, x + swatch_width), y),
                          fill=(213, 94, 0), width=4)
        else:
            draw.rectangle((x, y - 13, x + swatch_width, y + 13), fill=COLORS[label])
        draw.text((x + swatch_width + internal_gap, y - 18), label,
                  font=legend_font, fill=TEXT)
        x += item_width + item_gap


def draw_panel(image, bounds, title, ylabel, data, ymax, tick_step, nominal=False):
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = bounds
    panel_width = right - left
    panel_height = bottom - top

    centered(draw, ((left + right) / 2, top - 90), title, font(39, bold=True))

    tick_font = font(30)
    label_font = font(38)
    value_font = font(25)
    mode_font = font(34)

    def y_of(value):
        return bottom - value / ymax * panel_height

    ticks = []
    value = 0.0
    while value <= ymax + 1e-9:
        ticks.append(round(value, 10))
        value += tick_step
    for value in ticks:
        y = y_of(value)
        draw.line((left, y, right, y), fill=GRID, width=3)
        decimals = 1 if tick_step >= 0.1 and tick_step != 0.25 else 2
        right_aligned(draw, (left - 30, y), f"{value:.{decimals}f}", tick_font)

    draw.line((left, top, left, bottom), fill=SPINE, width=5)
    draw.line((left, bottom, right, bottom), fill=SPINE, width=5)

    if nominal:
        y = y_of(0.90)
        x = left
        while x < right:
            draw.line((x, y, min(x + 18, right), y), fill=(213, 94, 0), width=4)
            x += 30

    models = list(COLORS)
    group_centers = [left + panel_width * (i + 0.5) / 4 for i in range(4)]
    bar_width = 68
    offsets = (-bar_width, 0, bar_width)
    for mode_label, mode_key, center_x in zip(MODES, MODE_KEYS, group_centers):
        for model, offset in zip(models, offsets):
            mean, std = data[model][mode_key]
            x0 = center_x + offset - bar_width / 2
            x1 = center_x + offset + bar_width / 2
            y_mean = y_of(mean)
            draw.rectangle((x0, y_mean, x1, bottom), fill=COLORS[model])

            x_mid = center_x + offset
            y_low = y_of(max(0, mean - std))
            y_high = y_of(min(ymax, mean + std))
            draw.line((x_mid, y_high, x_mid, y_low), fill=ERROR, width=8)
            draw.line((x_mid - 9, y_high, x_mid + 9, y_high), fill=ERROR, width=8)
            draw.line((x_mid - 9, y_low, x_mid + 9, y_low), fill=ERROR, width=8)

            label_y = max(top + 18, y_high - 23)
            centered(draw, (x_mid, label_y), f"{mean:.2f}", value_font)

        centered(draw, (center_x, bottom + 53), mode_label, mode_font)

    centered(draw, ((left + right) / 2, bottom + 120), "Missing pattern", label_font)
    vertical_text(image, (left - 125, (top + bottom) / 2), ylabel, label_font)


def main():
    image = Image.new("RGB", (WIDTH, HEIGHT), WHITE)
    draw = ImageDraw.Draw(image)
    centered(draw, (WIDTH / 2, 46), "Chất lượng độ bất định — Geolink", font(49, bold=True))
    draw_legend(draw)

    draw_panel(
        image, (165, 360, 1325, 1450),
        "Độ bao phủ khoảng dự báo", "Coverage (PICP)",
        PICP, ymax=1.08, tick_step=0.2, nominal=True,
    )
    draw_panel(
        image, (1585, 360, 2745, 1450),
        "Độ rộng trung bình của khoảng dự báo", "Interval width (MPIW)",
        MPIW, ymax=2.10, tick_step=0.25,
    )
    image.save(OUTPUT, dpi=(180, 180), optimize=True)


if __name__ == "__main__":
    main()
