#!/usr/bin/env python3
"""
Rebuild the widescreen feature ridge plot from the published figure.

data/dataset.csv lives only on the machine that ran the pipeline, but the
paper's ridge plot (paper/plots/feature_violins.pdf) is a vector figure: the
17 filled KDE polygons drawn by plot_feature_violins are stored in it as
paths. This script lifts those polygons back out, turns each into the
peak-normalised density curve plot_feature_violins started from, and hands
the curves to the same layout code that plot_feature_violins_wide uses. The
result is the paper's exact curves in the 16:9 two-panel layout.

How the polygons are identified: plot_feature_violins draws fill i (feature
FEATURE_COLUMNS[i]) with zorder n-i and colour PALETTE[i % 6], so the fills
come out of the PDF in order i = 16, 15, ..., 0 and their fill colour must
match PALETTE[i % 6]. Both facts are checked; the script aborts if either
does not hold, rather than guess.

Reads:
    paper/plots/feature_violins.pdf   (converted to SVG with pdftocairo)

Outputs:
    paper/plots/feature_violins_wide.pdf

Usage:
    python code/violins_from_pdf.py
"""

from __future__ import annotations

import argparse
import logging
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate as ev  # noqa: E402

LOG = logging.getLogger("violins_from_pdf")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)-5s  %(message)s",
)

DEFAULT_SRC = ev.DEFAULT_PLOTS_DIR / "feature_violins.pdf"
DEFAULT_OUT = ev.DEFAULT_PLOTS_DIR / "feature_violins_wide.pdf"

# The fills are the only paths drawn with this opacity (see plot_feature_violins).
FILL_ALPHA = "fill-opacity:0.6"


def _hex_to_pct(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(round(int(h[k:k + 2], 16) / 255 * 100) for k in (0, 2, 4))


def _svg_paths(svg: str):
    """Yield (fill colour as rounded % triple, Nx2 point array) per fill."""
    body = svg[svg.index("</defs>"):]  # skip glyph outlines
    for m in re.finditer(r"<path([^>]*)>", body):
        attrs = m.group(1)
        style = re.search(r'style="([^"]*)"', attrs)
        d = re.search(r'\bd="([^"]*)"', attrs)
        if not (style and d) or FILL_ALPHA not in style.group(1):
            continue
        rgb = re.search(r"fill:rgb\(([\d.]+)%,([\d.]+)%,([\d.]+)%\)", style.group(1))
        colour = tuple(round(float(v)) for v in rgb.groups())
        pts = np.array(re.findall(r"[ML] ([\d.\-]+) ([\d.\-]+)", d.group(1)), dtype=float)
        yield colour, pts


def _curve_from_polygon(pts: np.ndarray) -> np.ndarray:
    """Peak-normalised density on ev.RIDGE_GRID from a fill_between polygon.

    The polygon is the KDE curve plus its flat baseline. Taking the highest y
    at each x isolates the curve; x is rescaled from the axes' pixel range to
    the data range [0, 1] (the curve spans exactly that range), and y from
    [baseline, peak] to [0, 1]. pdftocairo drops collinear vertices, which
    linear interpolation restores exactly.
    """
    xs = np.unique(pts[:, 0])
    top = np.array([pts[pts[:, 0] == x, 1].max() for x in xs])
    base, peak = pts[:, 1].min(), top.max()
    x_norm = (xs - xs.min()) / (xs.max() - xs.min())
    return np.interp(ev.RIDGE_GRID, x_norm, (top - base) / (peak - base))


def recover_curves(pdf: Path) -> dict[str, np.ndarray]:
    with tempfile.TemporaryDirectory() as tmp:
        svg = Path(tmp) / "violins.svg"
        subprocess.run(["pdftocairo", "-svg", str(pdf), str(svg)], check=True)
        fills = list(_svg_paths(svg.read_text(encoding="utf-8")))

    feats = ev.FEATURE_COLUMNS
    if len(fills) != len(feats):
        sys.exit(f"expected {len(feats)} fill polygons, found {len(fills)}")

    palette = [_hex_to_pct(c) for c in ev.PALETTE]
    curves: dict[str, np.ndarray] = {}
    for k, (colour, pts) in enumerate(fills):
        i = len(feats) - 1 - k  # drawn back-to-front
        if colour != palette[i % len(palette)]:
            sys.exit(f"polygon {k}: colour {colour} is not PALETTE[{i % 6}] "
                     f"{palette[i % 6]}; refusing to guess the feature order")
        curves[feats[i]] = _curve_from_polygon(pts)
        LOG.info("%-28s %4d vertices", feats[i], len(pts))
    return curves


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--src", type=Path, default=DEFAULT_SRC)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    ev._apply_theme()
    curves = recover_curves(args.src)
    ev.draw_feature_violins_wide(curves, args.out)


if __name__ == "__main__":
    main()
