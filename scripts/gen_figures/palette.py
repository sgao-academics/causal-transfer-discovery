# -*- coding: utf-8 -*-
"""THE palette -- the single source of colour for every figure in this library.

The seven codes come from the reference strip "Scientific Color Inspiration"
(Nature, Complex System cover palette) and are the exact colours Fig. 1 of the
MultiBatch paper was drawn in.  They are canonical: a figure script imports
them from here and never re-types a hex value.

    from palette import PAL, RAMP, make_cmap, INK, TICK, GREY, GRID, TRACK

**House rule: change the drawing, never the palette.**  A new figure type gets a
new layout, a new encoding, a new geometry -- it does not get new colours.
"""
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------------------
# The seven, with the job each one is expected to do.  Naming the job matters:
# it is what stops two figures from disagreeing about what a colour means.
# ---------------------------------------------------------------------------
PAL = {
    'mist':   '#4D6EAF',   # Pale Mist Blue      -- primary subject ("this work")
    'peri':   '#7A9AC3',   # Periwinkle Blue     -- secondary subject
    'lilac':  '#B0B9CB',   # Lilac Lavender      -- muted / de-emphasised / baseline
    'orchid': '#D0A0B6',   # Orchid Pink         -- contrast against the blues
    'moss':   '#8FB578',   # Soft Moss Green     -- second condition / control arm
    'violet': '#9A5A9F',   # Muted Violet Purple -- third category
    'indigo': '#3F3A5B',   # Indigo Plum         -- text, and the dark end of the ramp
}

# ---------------------------------------------------------------------------
# Sequential / continuous.  Light -> dark, monotone cool on purpose: a rainbow
# ramp reads as noise and dies in greyscale print.  Use for matrices, heatmaps,
# any ordinal encoding.
# ---------------------------------------------------------------------------
RAMP = ['#E9EEF5', '#C6D3E5', '#A9BFD9', '#7A9AC3', '#5A6A94', '#3F3A5B']


def make_cmap(name='house_ramp', ramp=None):
    """The house sequential colourmap (light -> dark)."""
    return LinearSegmentedColormap.from_list(name, ramp or RAMP)


# ---------------------------------------------------------------------------
# Supporting neutrals.  These are not "more colours" -- they are the fixed
# furniture of the style, kept here so no figure can invent its own grid grey.
# ---------------------------------------------------------------------------
INK     = PAL['indigo']   # all text; the deepest palette colour, never pure black
TICK    = '#5B6478'       # tick labels, one step lighter than INK
GREY    = '#8A93A6'       # de-emphasised labels
GRID    = '#EDF1F7'       # grid lines
TRACK   = '#EFF3F8'       # track lying behind capsule bars
FRAME   = '#C9D2E0'       # spines / axes frame
BOXFC   = '#FBFCFE'       # rounded annotation container fill
BOXEC   = '#DDE4EF'       # rounded annotation container edge
CARD_FC = '#FAFBFE'       # faint rounded card behind a panel
CARD_EC = '#E7EDF6'
SHADOW  = '#AEB8CC'       # drop shadow under markers
