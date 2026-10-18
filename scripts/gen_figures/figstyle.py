# -*- coding: utf-8 -*-
"""Layout and figure-element vocabulary for the house style.

Colour lives in ``palette.py``.  This module holds everything else that makes a
figure recognisably ours: the type scale, the faint rounded card behind each
panel, capsules resting on a very light track, markers carrying a soft drop
shadow rather than a hard outline, rounded annotation containers, and panel
letters placed in figure coordinates so the left column is never clipped.

Typical use::

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from figstyle import (W, MM, PAL, INK, BOXFC, BOXEC, tidy, card, dots,
                          capsule, haloed_text, panel_letters)

"""
import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.transforms import offset_copy

from palette import (BOXEC, BOXFC, CARD_EC, CARD_FC, FRAME, GRID, GREY, INK,
                     PAL, SHADOW, TICK, TRACK, make_cmap)

MM = 1.0 / 25.4  # mm -> inch

# Canvas width.  174 mm is the width the manuscript is finally typeset at, so a
# figure drawn at this width lands 1:1 and its declared point sizes survive.
# (The author template's own text block is 130.77 mm wide -- do not draw to that.)
W = 174.0 * MM

# The house sequential ramp lives in palette.py; re-exported for convenience.
CMAP = make_cmap()

# Font family for the library's own documentation sheets, which carry Chinese.
# Chinese does not fall back from Arial (see the note on font.sans-serif below),
# so any figure with non-Latin text has to name this family explicitly --
# usually by switching the whole sans-serif list, e.g.
#     plt.rcParams["font.sans-serif"] = [CJK, "SimHei", "DejaVu Sans", "Arial"]
CJK = "Microsoft YaHei"

# ---------------------------------------------------------------------------
# One type scale for the whole library.  Arial; 8 pt base; 7.5 pt for anything
# that annotates a panel; 11 pt bold for panel letters.  **Nothing below 7.5 pt
# except super/subscripts** (log-axis exponents).
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    # NOTE: this list is a *first-match* chain, NOT a per-glyph fallback.  A CJK
    # face appended after Arial does nothing -- matplotlib picks Arial for the
    # whole string and draws the Chinese as empty boxes (verified on 3.10.9).
    # So: Latin text in a figure meant for publication gets Arial, via this
    # list; a documentation sheet carrying Chinese must switch to CJK itself.
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 7.5, "axes.linewidth": 0.6,
    "axes.edgecolor": FRAME, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": TICK, "ytick.color": TICK,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.4, "ytick.major.size": 2.4,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 400, "savefig.dpi": 400, "text.usetex": False,
})


# ---------------------------------------------------------------------------
# Axis language
# ---------------------------------------------------------------------------
def tidy(ax, grid=None):
    """One axis language for every panel: open frame, thin outward ticks, faint grid.

    Also relies on the rcParams above, which already remove the top and right
    spines -- the single cheapest change that makes a panel stop looking default.
    """
    ax.tick_params(direction="out", length=2.4, width=0.6)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.5)
        ax.set_axisbelow(True)


def card(fig, ax, pad=0.011, radius=0.016, z=-4):
    """A faint rounded container behind a panel -- the reference artwork's habit.

    Call after the axes are created and before the final draw; it reads the
    axes' figure-space bounding box, so the layout must already be final.
    """
    bb = ax.get_position()
    fig.add_artist(FancyBboxPatch(
        (bb.x0 - pad, bb.y0 - pad), bb.width + 2 * pad, bb.height + 2 * pad,
        boxstyle="round,pad=0,rounding_size=%.4f" % radius,
        transform=fig.transFigure, fc=CARD_FC, ec=CARD_EC, lw=0.7, zorder=z))


# ---------------------------------------------------------------------------
# Figure elements
# ---------------------------------------------------------------------------
def px_per_data(ax):
    """Display pixels per data unit in x and y, for sizing pixel-derived shapes.

    **Must be called only after ``set_xlim``/``set_ylim`` are final.** Before
    that the axes still carry their default 0-1 ranges and the ratio comes out
    wrong by more than an order of magnitude -- that mistake once turned every
    capsule into a string of giant balls.
    """
    p0 = ax.transData.transform((0.0, 0.0))
    p1 = ax.transData.transform((1.0, 1.0))
    return (p1[0] - p0[0]), (p1[1] - p0[1])


def capsule(ax, x0, x1, y, h, color, z=3):
    """A horizontal capsule (rounded-end bar) in data coordinates.

    The cap radius is derived so the ends stay circular on screen regardless of
    how the axes are scaled; a rounded rectangle drawn straight in data units
    would come out elliptical.
    """
    sx, sy = px_per_data(ax)
    d_px = abs(h) * sy
    rad = (d_px / 2.0) / sx
    w = x1 - x0
    if w <= 2.0 * rad:
        rad = w / 2.0
    ax.add_patch(Rectangle((x0 + rad, y - h / 2.0), max(w - 2.0 * rad, 1e-9), h,
                           fc=color, ec="none", zorder=z))
    if w > 1e-9:
        d_pt = d_px * 72.0 / ax.figure.dpi
        ax.scatter([x0 + rad, x1 - rad], [y, y], s=d_pt * d_pt,
                   c=color, edgecolors="none", zorder=z)


def dots(ax, x, y, s, color, z=4, shadow=True, ec="white", lw=0.6,
         marker="o", alpha=1.0):
    """Markers with a soft drop shadow instead of a heavy outline.

    The shadow is what makes points read as *floating above* the sheet rather
    than stuck to it -- the single highest-value trick in the reference artwork.
    """
    if shadow:
        tr = offset_copy(ax.transData, fig=ax.figure, x=0.9, y=-0.9, units="points")
        ax.scatter(x, y, s=s * 1.04, c=SHADOW, alpha=0.32, edgecolors="none",
                   marker=marker, zorder=z - 0.5, transform=tr, clip_on=True)
    return ax.scatter(x, y, s=s, c=color, edgecolors=ec, linewidths=lw,
                      marker=marker, zorder=z, alpha=alpha)


def haloed_text(ax, x, y, s, size=7.5, color=INK, halo="white", hw=2.2, z=5, **kw):
    """Text with a soft halo, drawn as two stacked objects.

    Path effects on text rasterise the glyphs to outlines, which would break any
    later text extraction from the PDF (and with it the word/number diff that
    guards against silently dropping data).  So only the underlying halo copy is
    stroked -- that copy is what stops a track or a bar showing through the
    letters -- while the copy on top is ordinary text and stays extractable.
    """
    ax.text(x, y, s, fontsize=size, color=halo, zorder=z,
            path_effects=[pe.withStroke(linewidth=hw, foreground=halo)], **kw)
    return ax.text(x, y, s, fontsize=size, color=color, zorder=z + 0.1, **kw)


def panel_letters(fig, pairs, dx=-0.027, dy=0.007, size=11):
    """Panel letters a) b) c) ... in figure coordinates.

    Axes-fraction offsets push the left-hand column's letters off the canvas, so
    the position is taken from each axes' figure-space bounding box.
    """
    for ax, s in pairs:
        bb = ax.get_position()
        fig.text(bb.x0 + dx, bb.y1 + dy, s + ")", fontsize=size,
                 fontweight="bold", ha="left", va="bottom", color=INK)
