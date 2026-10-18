# -*- coding: utf-8 -*-
"""Figure 1 -- causal transfer learning, drawn as a fan-in / fan-out schematic.

Layout borrowed from the top-venue Figure-1 vocabulary (node-link glyphs instead
of text boxes, pale block arrows, one shared module rather than the same module
repeated in every row):

    source data  ->  W_S  --\
    random init  ->  W_0  ----> [ causal discovery on BRCA ] ---> W_T, W_T^scr, W_T^shuf ---> read-out
    shuffled     ->  pi(W_S)-/

Every glyph uses the *same* node positions, so the only thing that changes
between them is the edge set -- which is exactly the quantity the method is
about.  The warm-start target graph keeps four of the seven source edges, loses
three and gains one, so "retention" is visible rather than asserted.

Colours come from palette.py through housefig's role map; nothing here types a
hex value.  The black-and-white twin (FIG_MONO=1) uses the ordered grey ladder.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from housefig import FRAME, MM, C, ink, save, to_submission
from palette import CARD_FC, GREY, TRACK

ROOT = os.path.dirname(os.path.dirname(HERE))
FIGURES = os.path.join(ROOT, "figures")
MONO = os.environ.get("FIG_MONO", "0") == "1"
STEM = "fig1_overview_bw" if MONO else "fig1_overview"

W_MM, H_MM = 114.2, 74.0
fig = plt.figure(figsize=(W_MM * MM, H_MM * MM))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W_MM)
ax.set_ylim(0, H_MM)
ax.set_aspect("equal")
ax.axis("off")

FS = 7.0        # smallest type on the figure
FS_B = 7.5      # box headings
LW_BOX = 0.6


# ---------------------------------------------------------------------------
def rbox(x0, y0, x1, y1, fc, ec, r=1.4, lw=LW_BOX, z=2):
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle="round,pad=0,rounding_size=%.2f" % r,
        fc=fc, ec=ec, lw=lw, zorder=z, mutation_aspect=1.0))


def block_arrow(p0, p1, color, tail=1.15, head=3.1, hw=2.6, z=1.5, alpha=1.0):
    """Chunky pale arrow -- the connective tissue of the reference artwork."""
    ax.add_patch(FancyArrowPatch(
        p0, p1,
        arrowstyle="simple,head_length=%.2f,head_width=%.2f,tail_width=%.2f"
                   % (head, hw, tail),
        mutation_scale=1.0, fc=color, ec="none", alpha=alpha,
        shrinkA=0, shrinkB=0, zorder=z, joinstyle="miter"))


def chip(x0, y0, x1, y1, head, sub, z=3):
    rbox(x0, y0, x1, y1, TRACK, FRAME, r=1.6, z=z)
    cy = (y0 + y1) / 2.0
    ax.text((x0 + x1) / 2.0, cy + 1.8, head, fontsize=FS_B, ha="center",
            va="center", color=ink(), zorder=z + 1)
    ax.text((x0 + x1) / 2.0, cy - 1.6, sub, fontsize=FS, ha="center",
            va="center", color=ink(), zorder=z + 1)


# ---------------------------------------------------------------------------
# node layout shared by every glyph
R_RING = 4.05
ANG = np.deg2rad(90 + np.arange(7) * 360.0 / 7.0)
NODE = np.vstack([[0.0, 0.0], np.c_[R_RING * np.cos(ANG), R_RING * np.sin(ANG)]])

E_SRC = [(0, 1), (0, 2), (1, 3), (3, 4), (2, 5), (5, 6), (6, 7)]
E_WARM = [(0, 1), (1, 3), (3, 4), (5, 6)]
E_NEW = [(4, 7)]
E_SCR = [(0, 3), (0, 6), (1, 5), (2, 4), (3, 7), (4, 1), (5, 2), (6, 7), (2, 7)]
E_SHUF = [(0, 3), (1, 4), (2, 6), (3, 5), (4, 7), (5, 1), (6, 2)]


def glyph(cx, cy, edges, color, lw=1.0, new=None, nw=0.55, z=4):
    """A small directed graph.  Nodes are identical everywhere; edges carry meaning."""
    for (i, j) in edges:
        p, q = NODE[i], NODE[j]
        ax.add_patch(FancyArrowPatch(
            (cx + p[0], cy + p[1]), (cx + q[0], cy + q[1]),
            arrowstyle="-|>,head_length=1.5,head_width=0.9",
            mutation_scale=1.0, lw=lw, color=color, zorder=z,
            shrinkA=2.5, shrinkB=2.7, joinstyle="round", capstyle="round"))
    if new:
        for (i, j) in new:
            p, q = NODE[i], NODE[j]
            ax.add_patch(FancyArrowPatch(
                (cx + p[0], cy + p[1]), (cx + q[0], cy + q[1]),
                arrowstyle="-|>,head_length=1.4,head_width=0.8",
                mutation_scale=1.0, lw=nw, color=GREY, zorder=z,
                shrinkA=2.5, shrinkB=2.7, linestyle=(0, (1.6, 1.4))))
    ax.scatter(cx + NODE[:, 0], cy + NODE[:, 1], s=11, c="white",
               edgecolors=color, linewidths=0.6, zorder=z + 1)


def sym(x, y, s, size=FS + 0.5):
    ax.text(x, y, s, fontsize=size, ha="center", va="center", color=ink(),
            zorder=8)


# ---------------------------------------------------------------------------
# geometry
Y1, Y2, Y3 = 56.0, 34.0, 13.0
CH_L, CH_R = 1.2, 25.2
GX = 32.0
BX0, BX1, BY0, BY1 = 43.0, 73.0, 20.0, 47.0
OX = 84.0
RX0, RX1 = 94.0, 113.0
FAN = 2.0

# --- chips ---------------------------------------------------------------
chip(CH_L, Y1 + 3.6, CH_R, Y1 + 11.2, "TCGA (no BRCA)", "9,241 samples")
chip(CH_L, Y1 - 11.2, CH_R, Y1 - 3.6, "DepMap", "1,208 cell lines")
chip(CH_L, Y2 - 3.8, CH_R, Y2 + 3.8, "random init",
     "$W_0\\sim\\mathcal{N}(0,\\sigma^{2}I)$")
chip(CH_L, Y3 - 3.8, CH_R, Y3 + 3.8, "shuffled", "$W_0=\\pi(W_S)$")

# --- source data -> W_S --------------------------------------------------
block_arrow((CH_R + 0.3, Y1 + 7.4), (GX - 5.4, Y1 + 1.5), C("tcga"))
block_arrow((CH_R + 0.3, Y1 - 7.4), (GX - 5.4, Y1 - 1.5), C("tcga"))
ax.text(GX + 1.5, Y1 + 9.0, "learn $W_S$", fontsize=FS, ha="center",
        va="bottom", color=GREY, style="italic", zorder=8)

# --- initialiser glyphs --------------------------------------------------
glyph(GX, Y1, E_SRC, C("warm"), lw=1.0)
glyph(GX, Y2, E_SCR, C("scratch"), lw=0.6, nw=0.5)
glyph(GX, Y3, E_SHUF, C("shuffled"), lw=0.75)

sym(GX, Y1 - 7.8, "$W_S$")
sym(GX, Y2 - 7.8, "$W_0^{\\mathrm{scr}}$")
sym(GX, Y3 - 7.8, "$W_0^{\\mathrm{shuf}}$")

# --- fan-in to the shared learner ---------------------------------------
block_arrow((GX + 5.6, Y1 - 1.0), (BX0 - 0.2, BY1 - FAN), C("warm"), tail=1.0)
block_arrow((GX + 5.6, Y2), (BX0 - 0.2, (BY0 + BY1) / 2.0), C("scratch"),
            tail=1.0, alpha=0.85)
block_arrow((GX + 5.6, Y3 + 1.0), (BX0 - 0.2, BY0 + FAN), C("shuffled"), tail=1.0)

# --- the learner, drawn once --------------------------------------------
rbox(BX0, BY0, BX1, BY1, CARD_FC, FRAME, r=2.0, lw=0.8, z=3)
bx, by = (BX0 + BX1) / 2.0, (BY0 + BY1) / 2.0
ax.text(bx, by + 7.6, "causal discovery", fontsize=FS_B + 0.5, ha="center",
        va="center", fontweight="bold", color=ink(), zorder=4)
ax.text(bx, by + 2.6, "target: BRCA", fontsize=FS, ha="center", va="center",
        color=ink(), zorder=4)
ax.text(bx, by - 2.4, "1,218 samples", fontsize=FS, ha="center", va="center",
        color=ink(), zorder=4)
ax.text(bx, by - 7.4, "$d$ genes, $|w|>\\tau=0.3$", fontsize=FS, ha="center",
        va="center", color=ink(), zorder=4)
# The three estimator names do not fit the 30 mm box at 7 pt (31.6 mm), so they
# sit just below it, where the two fan arrows leave 2.9 mm of clearance.
ax.text(bx, 16.6, "NOTEARS·DAGMA·GOLEM", fontsize=FS, ha="center",
        va="center", color=GREY, zorder=4)

# --- fan-out to the three target graphs ---------------------------------
block_arrow((BX1 + 0.2, BY1 - FAN), (OX - 5.6, Y1 - 1.0), C("warm"), tail=1.0)
block_arrow((BX1 + 0.2, (BY0 + BY1) / 2.0), (OX - 5.6, Y2), C("scratch"),
            tail=1.0, alpha=0.85)
block_arrow((BX1 + 0.2, BY0 + FAN), (OX - 5.6, Y3 + 1.0), C("shuffled"), tail=1.0)

glyph(OX, Y1, E_WARM, C("warm"), lw=1.0, new=E_NEW)
glyph(OX, Y2, E_SCR, C("scratch"), lw=0.6, nw=0.5)
glyph(OX, Y3, E_SHUF, C("shuffled"), lw=0.75)

sym(OX, Y1 - 7.8, "$W_T$")
sym(OX, Y2 - 7.8, "$W_T^{\\mathrm{scr}}$")
sym(OX, Y3 - 7.8, "$W_T^{\\mathrm{shuf}}$")

# --- read-out -----------------------------------------------------------
block_arrow((OX + 5.6, Y1 - 1.0), (RX0 - 0.2, BY1 - FAN), C("warm"), tail=1.0)
block_arrow((OX + 5.6, Y2), (RX0 - 0.2, (BY0 + BY1) / 2.0), C("scratch"),
            tail=1.0, alpha=0.85)
block_arrow((OX + 5.6, Y3 + 1.0), (RX0 - 0.2, BY0 + FAN), C("shuffled"), tail=1.0)

rbox(RX0, BY0, RX1, BY1, TRACK, FRAME, r=2.0, lw=0.7, z=3)
cx, cy = (RX0 + RX1) / 2.0, (BY0 + BY1) / 2.0
ax.text(cx, cy + 10.6, "read-out", fontsize=FS_B + 0.5, ha="center",
        va="center", fontweight="bold", color=ink(), zorder=4)
ax.text(cx, cy + 6.2, "retention", fontsize=FS, ha="center", va="center",
        color=ink(), zorder=4)
ax.text(cx, cy + 1.6, "$|E_T\\cap E_S|/|E_S|$", fontsize=FS,
        ha="center", va="center", color=ink(), zorder=4)
ax.text(cx, cy - 5.0, "agreement", fontsize=FS, ha="center", va="center",
        color=ink(), zorder=4)
ax.text(cx, cy - 9.4, "$r\\,(W_S,\\,W_T)$", fontsize=FS, ha="center",
        va="center", color=ink(), zorder=4)

pdf = save(fig, FIGURES, STEM)
print("   glyph nodes %d, source edges %d, kept %d, new %d"
      % (len(NODE), len(E_SRC), len(E_WARM), len(E_NEW)))
if "--submit" in sys.argv:
    to_submission(pdf, os.path.join(FIGURES, "submission"), STEM)
