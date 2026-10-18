# -*- coding: utf-8 -*-
"""Drawing vocabulary shared by the two result figures.

``palette.py`` owns colour and ``figstyle.py`` owns the type scale; this module
adds only what the *result* figures need on top of them:

* the canvas width that matches how the figures are actually placed in the
  manuscript (0.90 x the journal text block), so the declared point sizes land
  1:1 instead of being scaled away;
* the mapping from a scientific role (warm-start, scratch, DAGMA ...) to a
  palette entry -- no script types a hex value;
* a monochrome twin of that mapping, because the journal lets us avoid the
  colour charge by printing black and white, and a greyscale photograph of a
  colour figure does not survive the trip (mist and violet collide in luminance);
* grouped bars resting on a light track, dumbbells and lollipops;
* the rasteriser that turns the vector PDF into the CMYK / 600 dpi greyscale
  TIFFs the production office asks for.

Nothing here changes a number.  Data are read by the figure scripts themselves.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from palette import (BOXEC, BOXFC, CARD_EC, CARD_FC, FRAME, GRID, GREY, INK,
                     PAL, RAMP, SHADOW, TICK, TRACK)

RAMP_SRC = RAMP  # light -> dark; ordinal encodings must use this, never a rainbow
from figstyle import card, dots, haloed_text, panel_letters, tidy  # noqa: F401

MM = 1.0 / 25.4

# The journal's text block is 126.9 mm and both result figures are included at
# 0.90\textwidth, i.e. 114.2 mm.  Drawing to that width means an 8 pt label is
# printed at 8 pt -- the single largest legibility gain available here.  (The
# old scripts drew on a 372 mm canvas that was then shrunk to 114 mm, so every
# label arrived at roughly 2.5 pt.)
W_TARGET_MM = 114.2
W = W_TARGET_MM * MM

# ---------------------------------------------------------------------------
# Role -> colour.  One encoding per panel; a colour never means two things
# inside one figure.
# ---------------------------------------------------------------------------
ROLE = {
    "warm":     PAL["mist"],     # the method under test
    "notears":  PAL["mist"],
    "dagma":    PAL["moss"],
    "golem":    PAL["violet"],
    "scratch":  PAL["lilac"],    # baseline
    "shuffled": GREY,            # negative control
    "depmap":   PAL["orchid"],
    "tcga":     PAL["peri"],
}

# Greyscale ladder used for the black-and-white submission version: lit by eye
# to stay ordered (dark = the method under test) and separated by >= 0.12 in
# luminance, with hatches so the two mid-tones cannot be confused after print.
MONO = {
    "warm":     ("0.28", None),
    "notears":  ("0.28", None),
    "dagma":    ("0.60", "///"),
    "golem":    ("0.44", "..."),
    "scratch":  ("0.82", None),
    "shuffled": ("0.62", "\\\\\\"),
    "depmap":   ("0.50", None),
    "tcga":     ("0.72", "///"),
}

MONO_MODE = os.environ.get("FIG_MONO", "0") == "1"

# Math is typeset from the same face as the labels: the journal asks for a
# single easy-to-read typeface, and the default DejaVu math glyphs would
# otherwise sit next to Arial labels at a visibly different weight.
plt.rcParams.update({
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "mathtext.default": "regular",
})


def C(role):
    """Colour for a role, in the palette or in the print ladder."""
    if MONO_MODE:
        return MONO[role][0]
    return ROLE[role]


def hatch(role):
    return MONO[role][1] if MONO_MODE else None


def ink():
    return "black" if MONO_MODE else INK


# ---------------------------------------------------------------------------
# Axis furniture
# ---------------------------------------------------------------------------
def frame(ax, grid="y"):
    tidy(ax, grid=grid)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("0.55" if MONO_MODE else FRAME)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=7.5)
    if MONO_MODE:
        ax.grid(color="0.90", lw=0.5)


def bar(ax, x, v, w, role, track=0.0, z=3, lw=0.0):
    """A grouped bar optionaly standing on a light track of the same width.

    The track is what gives a group of bars a common baseline; without it the
    eye compares bar tops across categories of very different size.
    """
    if track:
        ax.add_patch(Rectangle((x - w / 2, 0), w, track, fc=TRACK, ec="none", zorder=z - 1))
    ax.add_patch(Rectangle((x - w / 2, 0), w, v, fc=C(role), ec="white" if lw else "none",
                           lw=lw, hatch=hatch(role), zorder=z))
    return x


def dumbbell(ax, y, x0, x1, role0, role1, lw=1.2):
    """Two dots joined: the gap *is* the effect, so it is drawn as a segment."""
    ax.plot([x0, x1], [y, y], color="0.82" if MONO_MODE else FRAME, lw=lw,
            solid_capstyle="round", zorder=2)
    dots(ax, [x0], [y], 26, C(role0), z=4, ec="white", lw=0.6)
    dots(ax, [x1], [y], 34, C(role1), z=4, ec="white", lw=0.6)


def lollipop(ax, x, v, role, y0=0.0, marker="o"):
    """Stem plus dot, for signed quantities.  Bars cannot carry a sign.

    The marker defaults to a circle; the black-and-white ladder passes shapes so
    that series whose greys sit close together still separate in print.
    """
    ax.vlines(x, y0, v, color=C(role), lw=1.1, alpha=0.85, zorder=2)
    dots(ax, [x], [v], 30, C(role), z=4, ec="white", lw=0.6, marker=marker)


def annot(ax, x, y, s, ha="left", va="top", size=7.5, trans=None, **kw):
    return ax.text(x, y, s, transform=trans or ax.transAxes, ha=ha, va=va,
                   fontsize=size, color=ink(), zorder=7, linespacing=1.45,
                   bbox=dict(boxstyle="round,pad=0.34", fc=BOXFC, ec=BOXEC, lw=0.5,
                             alpha=0.96), **kw)


def letters(fig, pairs):
    panel_letters(fig, pairs, dx=-0.042, dy=0.008, size=9.5)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def save(fig, figures_dir, stem):
    pdf = os.path.join(figures_dir, stem + ".pdf")
    png = os.path.join(figures_dir, stem + ".png")
    fig.savefig(pdf, metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(png, dpi=400)
    print("wrote %s  and  %s" % (pdf, png))
    return pdf


def to_submission(pdf_path, out_dir, stem, dpi_color=300, dpi_bw=600):
    """CMYK TIFF for colour print, greyscale TIFF for the free black-and-white
    route.  Both are rasterised by MuPDF's own colour management (no naive
    channel arithmetic) and stored LZW-compressed, lossless."""
    import fitz
    from PIL import Image

    os.makedirs(out_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    pg = doc[0]

    pix = pg.get_pixmap(dpi=dpi_color, colorspace=fitz.csCMYK)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 4)
    cmyk = Image.fromarray(arr, "CMYK")
    p1 = os.path.join(out_dir, stem + "_cmyk.tif")
    cmyk.save(p1, compression="tiff_lzw", dpi=(dpi_color, dpi_color))

    pixg = pg.get_pixmap(dpi=dpi_bw, colorspace=fitz.csGRAY)
    gray = Image.frombytes("L", (pixg.width, pixg.height), pixg.samples)
    p2 = os.path.join(out_dir, stem + "_bw.tif")
    gray.save(p2, compression="tiff_lzw", dpi=(dpi_bw, dpi_bw))
    doc.close()

    print("   submission: %s (%d x %d, CMYK @%ddpi, %.1f MB)"
          % (os.path.basename(p1), cmyk.width, cmyk.height, dpi_color,
             os.path.getsize(p1) / 2 ** 20))
    print("   submission: %s (%d x %d, gray @%ddpi, %.1f MB)"
          % (os.path.basename(p2), gray.width, gray.height, dpi_bw,
             os.path.getsize(p2) / 2 ** 20))
    return p1, p2
