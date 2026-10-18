# -*- coding: utf-8 -*-
"""Figure 3 -- algorithm generality, cross-domain transfer and external checks.

Panels
  (a) DAGMA transfer across five gene-set sizes
  (b) GOLEM transfer across five gene-set sizes
  (c) cross-domain DepMap -> BRCA transfer (7 of the 20 source edges recovered)
  (d) warm-start advantage by algorithm and dimension
  (e) STRING v12 precision at three combined-score cut-offs
  (f) KEGG annotation of the top-200 edges

Data come from results/benchmark_v6_checkpoint.json, results/full_experiment.json
and results/external_validation.json; error bars are sample SDs over the three
seeds, matching the tables.  Colour encodes the algorithm (NOTEARS = mist,
DAGMA = moss, GOLEM = violet) and the baseline is always the same muted lilac,
so no colour carries two meanings inside the figure.  FIG_MONO=1 renders the
grey ladder used for black-and-white print.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import gridspec

from housefig import (FRAME, INK, MM, RAMP_SRC, TICK, W, annot, bar, C, dots,
                      frame, hatch, ink, letters, lollipop, save, to_submission)
from figstyle import capsule, haloed_text

ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")

MONO = os.environ.get("FIG_MONO", "0") == "1"
STEM = "fig3_generality_bw" if MONO else "fig3_generality"

DIMS = [200, 454, 600, 750, 889]
SEEDS = (42, 123, 456)

comp = json.load(open(os.path.join(RESULTS, "benchmark_v6_checkpoint.json"),
                      encoding="utf-8"))["completed"]
full = json.load(open(os.path.join(RESULTS, "full_experiment.json"), encoding="utf-8"))
ext = json.load(open(os.path.join(RESULTS, "external_validation.json"), encoding="utf-8"))


def arm(algo, which):
    out = []
    for d in DIMS:
        vals = []
        for s in SEEDS:
            k = ("d%d_s%d_%s_%s" % (d, s, algo, which)) if algo else ("d%d_s%d_%s" % (d, s, which))
            if k in comp and "retention_pct" in comp[k]:
                vals.append(comp[k]["retention_pct"])
        out.append((np.mean(vals), np.std(vals, ddof=1) if len(vals) > 1 else 0.0))
    return out


DAG = {w: arm("dagma", w) for w in ("scratch", "warm")}
GOL = {w: arm("golem", w) for w in ("scratch", "warm")}

if MONO:
    SHADE = [("0.86", None), ("0.62", "///"), ("0.30", None)]
else:
    SHADE = [(RAMP_SRC[2], None), (RAMP_SRC[3], None), (RAMP_SRC[5], None)]

fig = plt.figure(figsize=(W, 84 * MM))
gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[0.95, 0.95, 1.14],
                       left=0.128, right=0.986, top=0.930, bottom=0.105,
                       wspace=0.46, hspace=0.80)

# ═══════════════════════════════════════════════════ (a) DAGMA  (b) GOLEM
panels = {}
for pos, data, role, lab in ((gs[0, 0], DAG, "dagma", "DAGMA"),
                             (gs[0, 1], GOL, "golem", "GOLEM")):
    ax = fig.add_subplot(pos)
    panels[lab] = ax
    frame(ax)
    x = np.arange(len(DIMS))
    bw = 0.36
    for off, arm_name, rl in ((-bw / 2 - 0.01, "scratch", "scratch"),
                              (bw / 2 + 0.01, "warm", role)):
        mu = np.array([v[0] for v in data[arm_name]])
        sd = np.array([v[1] for v in data[arm_name]])
        for i in range(len(DIMS)):
            bar(ax, x[i] + off, mu[i], bw, rl)
        ax.errorbar(x + off, mu, yerr=sd, fmt="none", ecolor="0.45" if MONO else "0.62",
                    elinewidth=0.8, capsize=2.2, capthick=0.8, zorder=5)
        if arm_name == "warm":
            # In the grey ladder the fill is mid-grey, so a value label painted in
            # the fill colour fades into the bar top; black keeps it legible.
            vcol = ink() if MONO else C(rl)
            for i in range(len(DIMS)):
                ax.text(x[i] + off, mu[i] + sd[i] + 2.4, "%.0f" % mu[i], ha="center",
                        fontsize=7.5, color=vcol)
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in DIMS], fontsize=7.5)
    ax.set_xlabel("Gene-set size $d$")
    ax.set_ylim(0, 112)
    ax.set_xlim(-0.6, len(DIMS) - 0.4)
    ax.set_ylabel("Retention (%%)\n%s" % lab)
    # the algorithm name rides on the axis label: at 41 mm of panel width a
    # corner tag lands on the last value label, and a three-word legend does
    # not fit at all
    ax.plot([], [], "s", color=C("scratch"), ms=5, ls="none", label="Scratch")
    ax.plot([], [], "s", color=C(role), ms=5, ls="none", label="Warm")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.008), frameon=False,
              fontsize=7, handletextpad=0.3, borderpad=0.0, ncol=2,
              columnspacing=0.9, labelcolor=ink())

# ══════════════════════════════════════════ (c) cross-domain DepMap -> BRCA
axc = fig.add_subplot(gs[1, 0])
frame(axc)
dep_src = full["depmap"]["edges_03"]
rows = [("Scratch", full["depmap_to_scratch"]["overlap"], "scratch"),
        ("TCGA\nwarm", full["depmap_to_tcga_warm"]["overlap"], "tcga"),
        ("DepMap\nwarm", full["depmap_to_dm_warm"]["overlap"], "depmap")]
xc = np.arange(len(rows))
for i, (name, v, rl) in enumerate(rows):
    axc.add_patch(plt.Rectangle((xc[i] - 0.26, 0), 0.52, dep_src,
                                fc="#EFF3F8" if not MONO else "0.92", ec="none", zorder=1))
    axc.add_patch(plt.Rectangle((xc[i] - 0.26, 0), 0.52, v, fc=C(rl), ec="none",
                                hatch=hatch(rl), zorder=3))
    axc.text(xc[i], v + 1.0, "%d / %d" % (v, dep_src), ha="center", fontsize=7.5,
             color=ink(), zorder=6)
axc.set_xticks(xc)
axc.set_xticklabels([r[0] for r in rows], fontsize=7.5)
axc.set_xlim(-0.6, len(rows) - 0.4)
axc.set_ylim(0, dep_src * 1.20)
axc.set_ylabel("Edges recovered\n(of %d)" % dep_src)

# ══════════════════════════════════════════ (d) advantage by algorithm
axd = fig.add_subplot(gs[1, 1])
frame(axd)
xd = np.arange(len(DIMS))
series = [("NOTEARS", "notears", arm("", "scratch"), arm("", "warm")),
          ("DAGMA", "dagma", DAG["scratch"], DAG["warm"]),
          ("GOLEM", "golem", GOL["scratch"], GOL["warm"])]
# NOTEARS 0.28 and GOLEM 0.44 are only 0.16 apart in the grey ladder, which is
# too close once the dots shrink to print size -- shape is the second encoding.
MARK = {"notears": "o", "dagma": "s", "golem": "D"}


def mk(role):
    return MARK[role] if MONO else "o"


for j, (lab, role, scr, wrm) in enumerate(series):
    off = (j - 1) * 0.20
    for i in range(len(DIMS)):
        lollipop(axd, xd[i] + off, wrm[i][0] - scr[i][0], role, marker=mk(role))
axd.axhline(0, color="0.55" if MONO else FRAME, lw=0.9, zorder=1)
axd.set_xticks(xd)
axd.set_xticklabels([str(d) for d in DIMS], fontsize=7.5)
axd.set_xlabel("Gene-set size $d$")
axd.set_xlim(-0.6, len(DIMS) - 0.4)
axd.set_ylim(0, 92)
axd.set_ylabel("$\\Delta$ retention (pp)")
for lab, role, _, _ in series:
    axd.plot([], [], marker=mk(role), color=C(role), ms=4.2, ls="none", label=lab)
# At 7 pt the three names ran 107 pt inside a 113 pt panel and the final "M"
# crossed the right spine (x1 = 324.0 vs 319.2); 6.5 pt with tighter handles
# brings it back inside without dropping a series from the key.
axd.legend(loc="upper left", frameon=False, fontsize=7, handletextpad=0.50,
           borderpad=0.02, ncol=3, columnspacing=0.16, handlelength=0.3,
           labelcolor=ink())

# ══════════════════════════════════════════════════ (e) STRING precision
axe = fig.add_subplot(gs[2, 0])
frame(axe)
methods = ext["string_v12"]["methods"]
prec = ext["string_v12"]["precision_pct"]
thr = ext["string_v12"]["thresholds"]
xe = np.arange(len(methods))
w3 = 0.24
for k in range(len(thr)):
    off = (k - 1) * w3
    col, ht = SHADE[k]
    for i in range(len(methods)):
        axe.add_patch(plt.Rectangle((xe[i] + off - w3 / 2, 0), w3, prec[k][i],
                                    fc=col, ec="white" if MONO else "none",
                                    hatch=ht, lw=0.6 if MONO else 0, zorder=3))
axe.set_xticks(xe)
axe.set_xticklabels(["TCGA\nsource", "Scratch", "TCGA\nwarm", "DepMap\nwarm"], fontsize=7)
axe.set_ylabel("Precision (%)\nby STRING score")
axe.set_ylim(0, 7.4)
for k, t in enumerate(thr):
    col, ht = SHADE[k]
    axe.bar([0], [0], color=col, hatch=ht, label="$\\geq$%d" % t,
            edgecolor="white" if MONO else "none")
axe.legend(loc="upper left", frameon=False, fontsize=7, ncol=3,
           handlelength=0.9, columnspacing=0.7, borderpad=0.05,
           handletextpad=0.3, labelcolor=ink())


# ══════════════════════════════════════════════════════ (f) KEGG summary
axf = fig.add_subplot(gs[2, 1])
axf.axis("off")
axf.set_xlim(0, 1)
axf.set_ylim(0, 1)
axf.text(0.02, 0.985, "Chemokine signalling", transform=axf.transAxes,
         va="top", ha="left", fontsize=8, color=ink(), fontweight="bold")
axf.text(0.02, 0.845,
         "hsa04062 \u00b7 FDR 0.036\n"
         "enriched in scratch and\n"
         "DepMap-warm only\n"
         "CCL13  CCL11  CCL19\n"
         "XCR1  XCL1  VAV3  CCL3L3",
         transform=axf.transAxes, va="top", ha="left", fontsize=7, color=ink(),
         linespacing=1.62)

letters(fig, ((panels["DAGMA"], "a"), (panels["GOLEM"], "b"), (axc, "c"),
              (axd, "d"), (axe, "e"), (axf, "f")))
pdf = save(fig, FIGURES, STEM)
print("   DAGMA d454 %.1f/%.1f  GOLEM d454 %.1f/%.1f"
      % (DAG["scratch"][1][0], DAG["warm"][1][0], GOL["scratch"][1][0], GOL["warm"][1][0]))
print("   depmap %d/%d/%d of %d" % (rows[0][1], rows[1][1], rows[2][1], dep_src))
if "--submit" in sys.argv:
    to_submission(pdf, os.path.join(FIGURES, "submission"), STEM)
