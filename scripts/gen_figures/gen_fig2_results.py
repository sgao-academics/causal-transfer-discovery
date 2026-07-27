"""
Fig 2: NOTEARS transfer results — 3x2 panels.
(a) Overlap bar chart (b) Scaling law (c) Threshold sensitivity
(d) Warm scatter (e) Scratch scatter (f) Delta advantage across dims
"""
import json, os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.optimize import curve_fit

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(ROOT, 'results')
FIGURES = os.path.join(ROOT, 'manuscript', 'figures')
os.makedirs(FIGURES, exist_ok=True)
CACHE = os.path.join(RESULTS, 'cache')
CKPT = os.path.join(RESULTS, 'benchmark_v6_checkpoint.json')

WARM = '#D62728'; SCRATCH = '#1F77B4'; SHUFFLED = '#7F7F7F'
GRID = '#E0E0E0'
plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
    'axes.grid': True, 'grid.alpha': 0.25, 'grid.color': GRID,
    'axes.facecolor': '#FAFAFA', 'figure.facecolor': 'white',
    'legend.fontsize': 7.5, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.framealpha': 0.85
})

comp = json.load(open(CKPT, encoding='utf-8'))['completed']
dims = [200, 454, 600, 750, 889]

data = {}
for d in dims:
    data[d] = {}
    for method, key_name in [('scratch','scratch'), ('warm','warm'), ('shuffled','shuffled')]:
        vals = []
        for seed in [42, 123, 456]:
            k = f'd{d}_s{seed}_{key_name}'
            if k in comp and 'retention_pct' in comp[k]:
                vals.append(comp[k])
        if vals:
            data[d][method] = {
                'mean_ret': np.mean([v['retention_pct'] for v in vals]),
                'std_ret': np.std([v['retention_pct'] for v in vals]),
                'mean_r': np.mean([v['pearson_r'] for v in vals])
            }

fig = plt.figure(figsize=(15, 9))

# ===== (a) Overlap bar chart =====
ax = fig.add_subplot(2, 3, 1)
x = np.arange(len(dims))
w = 0.22
positions = [-w, 0, w]
for meth, col, pos, name in [('scratch', SCRATCH, -w, 'Scratch'),
                               ('warm', WARM, 0, 'Warm-start'),
                               ('shuffled', SHUFFLED, w, 'Shuffled')]:
    vals = np.array([data[d][meth]['mean_ret'] if meth in data[d] else 0 for d in dims])
    bars = ax.bar(x + pos, vals, w * 0.95, color=col, alpha=0.9, zorder=3, label=name)
    if meth == 'warm':
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1.2,
                   f'{v:.0f}', ha='center', fontsize=7, fontweight='bold', color=WARM)
ax.set_xticks(x); ax.set_xticklabels([f'$d={d}$' for d in dims])
ax.set_ylabel('Source Edge Retention (%)')
ax.set_title('(a) NOTEARS Edge Retention', fontweight='bold')
ax.legend(loc='upper left')
ax.set_ylim(0, 78)

# ===== (b) Scaling law =====
ax2 = fig.add_subplot(2, 3, 2)
ds = np.array(dims)
scr_vals = np.array([data[d]['scratch']['mean_ret'] for d in dims])
warm_vals = np.array([data[d]['warm']['mean_ret'] for d in dims])
def pl(x, a, b): return a * x**b
popt_s, _ = curve_fit(pl, ds, scr_vals, p0=[2, 0.4], maxfev=5000)
popt_w, _ = curve_fit(pl, ds, warm_vals, p0=[2, 0.5], maxfev=5000)
xf = np.linspace(150, 950, 100)
r2_s = 1 - np.sum((scr_vals - pl(ds, *popt_s))**2) / np.sum((scr_vals - np.mean(scr_vals))**2)
r2_w = 1 - np.sum((warm_vals - pl(ds, *popt_w))**2) / np.sum((warm_vals - np.mean(warm_vals))**2)
ax2.errorbar(ds, scr_vals, yerr=[data[d]['scratch']['std_ret'] for d in dims],
             fmt='o', c=SCRATCH, capsize=3, zorder=4, markersize=7,
             label=f'Scratch ($R^2={r2_s:.2f}$)')
ax2.errorbar(ds, warm_vals, yerr=[data[d]['warm']['std_ret'] for d in dims],
             fmt='s', c=WARM, capsize=3, zorder=4, markersize=7,
             label=f'Warm-start ($R^2={r2_w:.2f}$)')
ax2.plot(xf, pl(xf, *popt_s), '--', c=SCRATCH, alpha=0.5, lw=1.5)
ax2.plot(xf, pl(xf, *popt_w), '-', c=WARM, alpha=0.7, lw=2)
ax2.set_xlabel('Gene set size $d$'); ax2.set_ylabel('Retention Rate (%)')
ax2.set_title('(b) Power-Law Scaling', fontweight='bold')
ax2.legend()

# ===== (c) Threshold sensitivity =====
ax3 = fig.add_subplot(2, 3, 3)
taus = [0.1, 0.2, 0.3, 0.4, 0.5]
src_e = [1156, 168, 73, 48, 30]
scr_o = [122, 38, 28, 20, 13]
warm_o = [158, 57, 41, 28, 20]
x3 = np.arange(len(taus)); w3 = 0.32
ax3.bar(x3 - w3/2, scr_o, w3, color=SCRATCH, alpha=0.85, label='Scratch')
ax3.bar(x3 + w3/2, warm_o, w3, color=WARM, alpha=0.85, label='Warm-start')
for i, (s, w_) in enumerate(zip(scr_o, warm_o)):
    ax3.text(i - w3/2, s + 0.5, str(s), ha='center', fontsize=6.5, color=SCRATCH)
    ax3.text(i + w3/2, w_ + 0.5, str(w_), ha='center', fontsize=6.5, fontweight='bold', color=WARM)
ax3.set_xticks(x3)
ax3.set_xticklabels([f'$\\tau={t}$\n({e})' for t, e in zip(taus, src_e)], fontsize=7)
ax3.set_ylabel('Overlapping Edges')
ax3.set_title('(c) Threshold Robustness ($d=454$)', fontweight='bold')
ax3.legend()

# ===== (d) Warm weight scatter =====
ax4 = fig.add_subplot(2, 3, 4)
W_src = np.load(os.path.join(CACHE, 'src_d454.npy'))
W_warm = np.load(os.path.join(CACHE, 'warm_d454_s42.npy'))
d_act = min(W_src.shape[0], W_warm.shape[0])
W_src, W_warm = W_src[:d_act,:d_act], W_warm[:d_act,:d_act]
mask = np.ones((d_act, d_act), bool); np.fill_diagonal(mask, False)
r_w, _ = pearsonr(W_src[mask], W_warm[mask])
# Subsample for visualization (full d^2 points are over 200k)
n_plot = min(4000, d_act*d_act - d_act)
idx = np.random.RandomState(42).choice(d_act*d_act - d_act, n_plot, replace=False)
ax4.scatter(W_src[mask][idx], W_warm[mask][idx], c=WARM, alpha=0.22, s=1.8,
            edgecolors='none', rasterized=True)
ax4.axhline(0, color='gray', lw=0.5, alpha=0.35); ax4.axvline(0, color='gray', lw=0.5, alpha=0.35)
ax4.set_xlabel('Source Edge Weight $W_S$')
ax4.set_ylabel('Target Edge Weight $W_T$')
ax4.set_title(f'(d) Warm-start: $r = {r_w:.3f}$', fontweight='bold')

# ===== (e) Scratch weight scatter =====
ax5 = fig.add_subplot(2, 3, 5)
W_scr = np.load(os.path.join(CACHE, 'scratch_d454_s42.npy'))[:d_act,:d_act]
r_s, _ = pearsonr(W_src[mask], W_scr[mask])
ax5.scatter(W_src[mask][idx], W_scr[mask][idx], c=SCRATCH, alpha=0.22, s=1.8,
            edgecolors='none', rasterized=True)
ax5.axhline(0, color='gray', lw=0.5, alpha=0.35); ax5.axvline(0, color='gray', lw=0.5, alpha=0.35)
ax5.set_xlabel('Source Edge Weight $W_S$')
ax5.set_ylabel('Target Edge Weight $W_T$')
ax5.set_title(f'(e) Scratch: $r = {r_s:.3f}$', fontweight='bold')

# ===== (f) Delta advantage =====
ax6 = fig.add_subplot(2, 3, 6)
x6 = np.arange(len(dims))
deltas = [data[d]['warm']['mean_ret'] - data[d]['scratch']['mean_ret'] for d in dims]
colors = ['#F1948A' if v < 25 else '#E74C3C' if v < 32 else '#C0392B' for v in deltas]
bars6 = ax6.bar(x6, deltas, 0.55, color=colors, alpha=0.9, zorder=3, edgecolor='white', linewidth=0.5)
for b, v in zip(bars6, deltas):
    ax6.text(b.get_x() + b.get_width()/2, b.get_height() + 0.6, f'+{v:.0f}',
            ha='center', fontsize=9, fontweight='bold', color='#C0392B')
ax6.set_xticks(x6); ax6.set_xticklabels([f'$d={d}$' for d in dims])
ax6.set_ylabel('$\\Delta$ Retention (pp)')
ax6.set_title('(f) Warm-Start Advantage ($\\Delta$) ', fontweight='bold')
ax6.axhline(y=0, color='gray', lw=0.5, alpha=0.3)
# Annotate total
total_delta = sum(deltas)
ax6.text(0.02, 0.95, f'$\\Sigma\\Delta = +{total_delta:.0f}$ pp\nacross all dimensions',
        transform=ax6.transAxes, ha='left', va='top', fontsize=7.5,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF3E0', edgecolor='#FF9800', alpha=0.7))

plt.tight_layout(pad=2.0)
fig.savefig(os.path.join(FIGURES, 'fig2_results.pdf'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(FIGURES, 'fig2_results.png'), dpi=300, bbox_inches='tight')
print(f'Fig 2 saved: {os.path.join(FIGURES, "fig2_results.pdf")}')
