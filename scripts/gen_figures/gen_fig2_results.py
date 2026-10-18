# -*- coding: utf-8 -*-
"""Figure 2 -- NOTEARS warm-start transfer on BRCA (five gene-set sizes).

Panels
  (a) retention of source edges per dimension, three arms (full width)
  (b) power-law scaling of retention with dimension
  (c) retention as a function of the edge-weight threshold at d = 454
  (d) source-versus-target edge weights, warm-start and scratch overlaid
  (e) warm-start and shuffled-control advantage in percentage points

Data: results/benchmark_v6_checkpoint.json (3 seeds per cell) and the three
d = 454 weight matrices archived in results/cache/.  No number is typed here --
everything is computed from the checkpoint, exactly as in the manuscript.

Fig 2 is drawn twice: colour (house palette) and, with FIG_MONO=1, a
black-and-white twin built on an ordered grey ladder.  The journal charges
US$200 per colour illustration, so the B/W twin is what makes the paper
free to print.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

import matplotlib.pyplot as plt
from matplotlib import gridspec

from housefig import (FRAME, INK, MM, TICK, W, bar, C, dots, dumbbell,
                      frame, ink, letters, lollipop, save, to_submission)

ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")
CACHE = os.path.join(RESULTS, "cache")

MONO = os.environ.get("FIG_MONO", "0") == "1"
STEM = "fig2_results_bw" if MONO else "fig2_results"

DIMS = [200, 454, 600, 750, 889]
SEEDS = (42, 123, 456)

comp = json.load(open(os.path.join(RESULTS, "benchmark_v6_checkpoint.json"),
                      encoding="utf-8"))["completed"]


def cell(d, arm):
    vals = [comp["d%d_s%d_%s" % (d, s, arm)]["retention_pct"]
            for s in SEEDS if "d%d_s%d_%s" % (d, s, arm) in comp]
    # sample SD (n - 1), the convention stated in the table captions
    return np.mean(vals), (np.std(vals, ddof=1) if len(vals) > 1 else 0.0)


R = {arm: [cell(d, arm) for d in DIMS] for arm in ("scratch", "warm", "shuffled")}

# ── canvas ---------------------------------------------------------------
# 131.2 mm = 0.90 x the 145.8 mm text block, so 8 pt labels print at 8 pt.
fig = plt.figure(figsize=(W, 84 * MM))
gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[0.86, 1.0, 1.0],
                       left=0.125, right=0.988, top=0.930, bottom=0.100,
                       wspace=0.40, hspace=0.92)

# ═══════════════════════════════════════════════════════════ (a) dumbbell
axa = fig.add_subplot(gs[0, :])
frame(axa, grid="x")
ys = np.arange(len(DIMS))[::-1]
for y, i, d in zip(ys, range(len(DIMS)), DIMS):
    ws, wsd = R["warm"][i]
    ss, ssd = R["scratch"][i]
    sh, shd = R["shuffled"][i]
    dots(axa, [sh], [y], 16, C("shuffled"), z=3, ec="white", lw=0.5)
    dumbbell(axa, y, ss, ws, "scratch", "warm")
    axa.text(ws + 1.6, y, "%+.0f" % (ws - ss), va="center", ha="left",
             fontsize=7.5, color=C("warm"), zorder=6)
axa.set_yticks(ys)
axa.set_yticklabels(["$d$ = %d" % d for d in DIMS])
axa.set_ylim(-0.6, len(DIMS) - 0.4)
axa.set_xlim(0, 80)
axa.set_xlabel("Source edges retained (%)")
for lab, role in (("Warm-start", "warm"), ("Scratch", "scratch"),
                  ("Shuffled control", "shuffled")):
    axa.plot([], [], "o", color=C(role), ms=5, mew=0, ls="none", label=lab)
axa.legend(loc="lower left", bbox_to_anchor=(0.005, 0.985), ncol=3, frameon=False,
           handletextpad=0.3, columnspacing=1.4, fontsize=7.5, borderpad=0.1)

# ═══════════════════════════════════════════════════════════ (b) scaling
axb = fig.add_subplot(gs[1, 0])
frame(axb)
ds = np.array(DIMS, float)


def pl(x, a, b):
    return a * x ** b


fits = {}
for arm in ("scratch", "warm"):
    y = np.array([v[0] for v in R[arm]])
    e = np.array([v[1] for v in R[arm]])
    popt, _ = curve_fit(pl, ds, y, p0=[2, 0.4], maxfev=20000)
    ss_res = np.sum((y - pl(ds, *popt)) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    fits[arm] = (popt, 1 - ss_res / ss_tot)
    dots(axb, ds, y, 22, C(arm), z=4, ec="white", lw=0.6)
    axb.errorbar(ds, y, yerr=e, fmt="none", ecolor="0.55" if MONO else FRAME,
                 elinewidth=0.8, capsize=2.2, capthick=0.8, zorder=3)
    xf = np.linspace(170, 930, 120)
    axb.plot(xf, pl(xf, *popt), lw=1.3, color=C(arm), alpha=0.9, zorder=2,
             ls="-" if arm == "warm" else (0, (4, 2)))
axb.set_xlabel("Gene-set size $d$")
axb.set_ylabel("Retention (%)")
# Top raised to 100: a single-row legend needs the band above 84, which is
# clear of the tallest whisker (d = 600, about 76 %).
axb.set_ylim(10, 100)
axb.set_yticks([25, 50, 75])
axb.set_xticks(DIMS)
axb.set_xticklabels(["200", "454", "600", "750", "889"])
for lab, role, ls in (("Warm-start", "warm", "-"), ("Scratch", "scratch", (0, (4, 2)))):
    axb.plot([], [], ls=ls, lw=1.3, color=C(role), label=lab)
axb.legend(loc="upper left", bbox_to_anchor=(0.0, 1.005), frameon=False,
           fontsize=7, handlelength=1.3, handletextpad=0.35, borderpad=0.0,
           ncol=2, columnspacing=0.9, labelcolor=ink())

# ═════════════════════════════════════════════════════ (c) threshold sweep
axc = fig.add_subplot(gs[1, 1])
frame(axc)
TAU = [0.1, 0.2, 0.3, 0.4, 0.5]
SRC = [1156, 168, 73, 48, 30]
SCR_O = [122, 38, 28, 20, 13]
WAR_O = [158, 57, 41, 28, 20]
x = np.arange(len(TAU))
bw = 0.34
for i, t in enumerate(TAU):
    bar(axc, x[i] - bw / 2 - 0.01, 100.0 * SCR_O[i] / SRC[i], bw, "scratch")
    bar(axc, x[i] + bw / 2 + 0.01, 100.0 * WAR_O[i] / SRC[i], bw, "warm")
    axc.text(x[i] + bw / 2 + 0.01, 100.0 * WAR_O[i] / SRC[i] + 1.5, "%.0f"
             % (100.0 * WAR_O[i] / SRC[i]), ha="center", fontsize=7, color=C("warm"))
axc.set_xticks(x)
axc.set_xticklabels(["%.1f" % t for t in TAU])
axc.set_xlabel("Edge-weight threshold $\\tau$")
axc.set_ylabel("Retained (%)")
axc.set_ylim(0, 82)
axc.set_xlim(-0.55, len(TAU) - 0.45)

# ══════════════════════════════════════════════════════════ (d) weights
axd = fig.add_subplot(gs[2, 0])
frame(axd, grid="both")
# The seed-42 pair from the benchmark itself: these are the matrices whose
# Pearson r is reported in Table 2 (0.2926 warm, 0.2051 scratch), so the panel
# and the table can no longer disagree.
W_s = np.load(os.path.join(CACHE, "src_d454_s42.npy"))
W_w = np.load(os.path.join(CACHE, "warm_d454_s42.npy"))
W_c = np.load(os.path.join(CACHE, "scratch_d454_s42.npy"))
d_act = min(W_s.shape[0], W_w.shape[0], W_c.shape[0])
W_s, W_w, W_c = W_s[:d_act, :d_act], W_w[:d_act, :d_act], W_c[:d_act, :d_act]
mask = np.ones((d_act, d_act), bool)
np.fill_diagonal(mask, False)
r_w = pearsonr(W_s[mask], W_w[mask])[0]
r_c = pearsonr(W_s[mask], W_c[mask])[0]
n_plot = min(5000, mask.sum())
idx = np.random.RandomState(42).choice(mask.sum(), n_plot, replace=False)
lim = 0.42
for M, role, a in ((W_c, "scratch", 0.30), (W_w, "warm", 0.34)):
    axd.scatter(W_s[mask][idx], M[mask][idx], s=2.2, c=C(role), alpha=a,
                edgecolors="none", rasterized=True, zorder=3)
axd.axhline(0, color="0.80" if MONO else FRAME, lw=0.8, zorder=1)
axd.axvline(0, color="0.80" if MONO else FRAME, lw=0.8, zorder=1)
axd.set_xlim(-lim, lim)
axd.set_ylim(-lim, lim)
axd.set_xlabel("Source weight $W_S$")
axd.set_ylabel("Target weight $W_T$")
# the two arms are identified by the colour of their own line, so no legend
from figstyle import haloed_text
haloed_text(axd, 0.035, 0.945, "Warm-start  $r$ = %.2f" % r_w,
            size=7.5, color=C("warm"), va="top", transform=axd.transAxes, z=7)
haloed_text(axd, 0.035, 0.800, "Scratch  $r$ = %.2f" % r_c,
            size=7.5, color=TICK, va="top", transform=axd.transAxes, z=7)

# ═══════════════════════════════════════════════════════════════ (e) delta
axe = fig.add_subplot(gs[2, 1])
frame(axe)
d_warm = np.array([R["warm"][i][0] - R["scratch"][i][0] for i in range(len(DIMS))])
d_shuf = np.array([R["shuffled"][i][0] - R["scratch"][i][0] for i in range(len(DIMS))])
xe = np.arange(len(DIMS))
for i in range(len(DIMS)):
    lollipop(axe, xe[i] - 0.16, d_warm[i], "warm")
    lollipop(axe, xe[i] + 0.16, d_shuf[i], "shuffled")
    axe.text(xe[i] - 0.16, d_warm[i] + 1.6, "%+.0f" % d_warm[i], ha="center",
             fontsize=7.5, color=C("warm"))
axe.axhline(0, color="0.55" if MONO else FRAME, lw=0.9, zorder=1)
axe.set_xticks(xe)
axe.set_xticklabels([str(d) for d in DIMS], fontsize=7.5)
axe.set_xlabel("Gene-set size $d$")
axe.set_xlim(-0.6, len(DIMS) - 0.4)
axe.set_ylim(-8, 64)
axe.set_ylabel("$\\Delta$ retention (pp)")
# The mean advantage of the control is folded into the legend label: a
# floating note in this 41 mm panel collided with the lollipop stems and the
# grey control markers at d = 200, 454 and 600.
for lab, role in (("Warm-start", "warm"), ("Shuffled", "shuffled")):
    axe.plot([], [], "o", color=C(role), ms=4.5, ls="none", label=lab)
axe.legend(loc="upper left", bbox_to_anchor=(0.015, 0.995), frameon=False,
           fontsize=7, handletextpad=0.3, borderpad=0.0, ncol=2,
           columnspacing=0.8, labelcolor=ink())

letters(fig, ((axa, "a"), (axb, "b"), (axc, "c"), (axd, "d"), (axe, "e")))

pdf = save(fig, FIGURES, STEM)
print("   warm R2 %.3f b=%.3f | scratch R2 %.3f b=%.3f"
      % (fits["warm"][1], fits["warm"][0][1], fits["scratch"][1], fits["scratch"][0][1]))
print("   r warm %.4f  r scratch %.4f" % (r_w, r_c))
print("   delta warm %s" % np.round(d_warm, 1))
print("   delta shuffled %s" % np.round(d_shuf, 1))
if "--submit" in sys.argv:
    to_submission(pdf, os.path.join(FIGURES, "submission"), STEM)
