"""Render a presentation GIF from the saved physical CSV states, without interpolation."""

from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "friction_heat_road.csv"
OUTPUT = ROOT / "friction_and_heat.gif"
WIDTH, HEIGHT = 1280, 800
BG = "#101923"
CARD = "#172630"
GRID = "#334754"
TEXT = "#ecf2f5"
MUTED = "#a9bbc5"
BLUE = "#5ac8fa"
AMBER = "#ffb45e"
GREEN = "#75d6a3"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "segoeuib.ttf" if bold else "segoeui.ttf"
    path = Path("C:/Windows/Fonts") / name
    if path.exists():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def draw_chart(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    times: list[float],
    values: list[float],
    max_value: float,
    color: str,
    unit: str,
) -> None:
    x0, y0, x1, y1 = box
    for i in range(5):
        y = y0 + (y1 - y0) * i / 4
        draw.line((x0, y, x1, y), fill=GRID, width=1)
    draw.line((x0, y0, x0, y1), fill=MUTED, width=1)
    draw.line((x0, y1, x1, y1), fill=MUTED, width=1)
    points = [
        (x0 + t / max(times[-1], 1e-9) * (x1 - x0), y1 - max(v, 0.0) / max_value * (y1 - y0))
        for t, v in zip(times, values)
    ]
    if len(points) > 1:
        draw.line(points, fill=color, width=4, joint="curve")
        px, py = points[-1]
        draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=color)
    draw.text((x0, y1 + 10), "0 s", font=font(15), fill=MUTED)
    draw.text((x1 - 70, y1 + 10), f"{times[-1]:.1f} s", font=font(15), fill=MUTED)
    draw.text((x0 + 8, y0 + 5), unit, font=font(15, True), fill=MUTED)


def main() -> None:
    with SOURCE.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) < 2:
        raise SystemExit("Need at least two measured physical states")

    t = [float(r["time_s"]) for r in rows]
    x = [float(r["center_x_m"]) for r in rows]
    v = [float(r["center_vx_m_s"]) for r in rows]
    heat = [float(r["cumulative_friction_heat_j"]) * 1000.0 for r in rows]
    rise = [float(r["temperature_rise_mK"]) for r in rows]
    broken = [int(r["broken_bonds"]) for r in rows]
    road_start, road_end = -1.2, 1.2
    speed_max = max(max(v), 0.02)
    heat_max = max(max(heat) * 1.1, 0.001)

    frames: list[Image.Image] = []
    for i in range(len(rows)):
        row = rows[i]
        image = Image.new("RGB", (WIDTH, HEIGHT), BG)
        d = ImageDraw.Draw(image)
        d.text((48, 30), "FRICTION → HEAT", font=font(38, True), fill=TEXT)
        d.text((50, 81), "Oak 3 × 3 × 3 cm cube · concrete · recorded solver states", font=font(20), fill=MUTED)
        d.rounded_rectangle((980, 34, 1230, 74), radius=18, fill="#203540", outline=GRID, width=1)
        d.text((1000, 43), f"{len(rows)} MEASURED STATES", font=font(16, True), fill=GREEN)

        d.rounded_rectangle((40, 128, 1240, 354), radius=18, fill=CARD, outline=GRID, width=1)
        d.text((66, 147), "ROLLING ON A 2.40 m CONCRETE TRACK", font=font(18, True), fill=TEXT)
        d.text((66, 178), f"Physical time  {t[i]:.3f} s", font=font(19), fill=BLUE)
        left, right, y_track = 130, 1145, 275
        d.rounded_rectangle((left, y_track - 18, right, y_track + 18), radius=8, fill="#697982")
        for m in range(25):
            xx = left + m * (right - left) / 24
            d.line((xx, y_track - 16, xx, y_track + 16), fill="#91a0a8", width=1)
        cx = left + (x[i] - road_start) / (road_end - road_start) * (right - left)
        # The cube marker follows its recorded center position; its display size is schematic.
        d.polygon([(cx, y_track - 13), (cx + 13, y_track), (cx, y_track + 13), (cx - 13, y_track)], fill=AMBER)
        d.text((left, 315), "−1.2 m", font=font(15), fill=MUTED)
        d.text((right - 58, 315), "+1.2 m", font=font(15), fill=MUTED)

        cards = [(40, 382, 425, 520), (447, 382, 832, 520), (854, 382, 1240, 520)]
        metrics = [
            ("SPEED", f"{v[i]:.3f} m/s", BLUE),
            ("FRICTION HEAT", f"{heat[i]:.3f} mJ", AMBER),
            ("MEAN TEMPERATURE RISE", f"{rise[i]:.4f} mK", GREEN),
        ]
        for box, (label, value, color) in zip(cards, metrics):
            d.rounded_rectangle(box, radius=16, fill=CARD, outline=GRID, width=1)
            d.text((box[0] + 20, box[1] + 20), label, font=font(15, True), fill=MUTED)
            d.text((box[0] + 20, box[1] + 66), value, font=font(29, True), fill=color)

        d.rounded_rectangle((40, 548, 1240, 758), radius=18, fill=CARD, outline=GRID, width=1)
        d.text((66, 563), "MEASURED THROUGH THE RUN", font=font(17, True), fill=TEXT)
        draw_chart(d, (92, 602, 620, 713), t[: i + 1], v[: i + 1], speed_max, BLUE, "m/s")
        draw_chart(d, (685, 602, 1190, 713), t[: i + 1], heat[: i + 1], heat_max, AMBER, "mJ")
        d.text((68, 730), f"Broken bonds: {broken[i]}  ·  Playback: 8× slower than physical time  ·  Every point is a saved state", font=font(14), fill=MUTED)
        frames.append(image)

    frames[0].save(
        OUTPUT,
        save_all=True,
        append_images=frames[1:],
        duration=80,
        loop=0,
        optimize=True,
        disposal=2,
    )
    print(f"wrote {OUTPUT} from {len(rows)} saved states ({t[-1]:.3f} physical seconds)")


if __name__ == "__main__":
    main()
