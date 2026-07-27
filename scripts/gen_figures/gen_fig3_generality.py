"""
Fig 3: Algorithm generality + cross-domain transfer + external validation — 2x3 panels.
(a) DAGMA 5-dim (b) GOLEM 5-dim (c) Cross-domain DepMap
(d) Multi-algo delta summary (e) STRING validation (f) KEGG enrichment
"""
import json, os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIGURES = os.path.join(ROOT, 'manuscript', 'figures')
CKPT = os.path.join(ROOT, 'results', 'benchmark_v6_checkpoint.json')
FULL_EXP = os.path.join(ROOT, 'results', 'full_experiment.json')
EXT_VAL = os.path.join(ROOT, 'results', 'external_validation.json')

WARM = '#D62728'; SCRATCH = '#1F77B4'; SHUFFLED = '#7F7F7F'
DAGMA_C = '#E67E22'; GOLEM_C = '#27AE60'; DM_C = '#9467BD'
GRID = '#E0E0E0'
plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
    'axes.grid': True, 'grid.alpha': 0.3, 'grid.color': GRID,
    'axes.facecolor': '#FAFAFA', 'figure.facecolor': 'white',
    'legend.fontsize': 7.5, 'xtick.labelsize': 8, 'ytick.labelsize': 8
})

# ─── Load data ───
comp = json.load(open(CKPT, encoding='utf-8'))['completed']
full = json.load(open(FULL_EXP, encoding='utf-8'))
ext = json.load(open(EXT_VAL, encoding='utf-8'))
dims = [200, 454, 600, 750, 889]

def get_algo_data(algo):
    """Pull mean+std from checkpoint for a given algorithm."""
    res = {}
    for d in dims:
        res[d] = {}
        for m in ['scratch', 'warm']:
            vals = []
            for s in [42, 123, 456]:
                k = f'd{d}_s{s}_{algo}_{m}'
                if k in comp and 'retention_pct' in comp[k]:
                    vals.append(comp[k])
            if vals:
                res[d][m] = {
                    'mean': np.mean([v['retention_pct'] for v in vals]),
                    'std': np.std([v['retention_pct'] for v in vals])
                }
    return res

dagma_data = get_algo_data('dagma')
golem_data = get_algo_data('golem')

# NOTEARS from checkpoint (bare keys, no prefix)
notears_data = {}
for d in dims:
    notears_data[d] = {}
    for m_name in ['scratch', 'warm']:
        vals = []
        for s in [42, 123, 456]:
            k = f'd{d}_s{s}_{m_name}'
            if k in comp and 'retention_pct' in comp[k]:
                vals.append(comp[k])
        if vals:
            notears_data[d][m_name] = {
                'mean': np.mean([v['retention_pct'] for v in vals]),
                'std': np.std([v['retention_pct'] for v in vals])
            }

# DepMap cross-domain (from full_experiment.json)
depmap_scr = full['depmap_to_scratch']['overlap']       # 0
depmap_tcw = full['depmap_to_tcga_warm']['overlap']     # 2
depmap_dmw = full['depmap_to_dm_warm']['overlap']       # 7
depmap_src_edges = full['depmap']['edges_03']             # 20

# STRING validation (from external_validation.json)
string_thresh = ext['string_v12']['thresholds']
string_methods = ext['string_v12']['methods']
string_prec = ext['string_v12']['precision_pct']  # shape: [3 thresholds x 4 methods]

# ─── Build figure ───
fig = plt.figure(figsize=(15, 9))
x = np.arange(len(dims))

# ═══════════ (a) DAGMA 5-dim with error bars ═══════════
ax = fig.add_subplot(2, 3, 1)
w = 0.32
scr_vals_d = [dagma_data[d]['scratch']['mean'] if d in dagma_data and 'scratch' in dagma_data[d] else np.nan for d in dims]
warm_vals_d = [dagma_data[d]['warm']['mean'] if d in dagma_data and 'warm' in dagma_data[d] else np.nan for d in dims]
scr_std_d = [dagma_data[d]['scratch']['std'] if d in dagma_data and 'scratch' in dagma_data[d] else 0 for d in dims]
warm_std_d = [dagma_data[d]['warm']['std'] if d in dagma_data and 'warm' in dagma_data[d] else 0 for d in dims]

ax.bar(x - w/2, scr_vals_d, w, color=SCRATCH, alpha=0.85, label='Scratch',
       yerr=scr_std_d, error_kw={'capsize': 3, 'linewidth': 0.8, 'ecolor': '#555'})
bars_d = ax.bar(x + w/2, warm_vals_d, w, color=DAGMA_C, alpha=0.85, label='Warm-start',
                yerr=warm_std_d, error_kw={'capsize': 3, 'linewidth': 0.8, 'ecolor': '#555'})
for b, v in zip(bars_d, warm_vals_d):
    if not np.isnan(v):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 2.0, f'{v:.0f}',
                ha='center', fontsize=7, fontweight='bold', color=DAGMA_C)
ax.set_xticks(x); ax.set_xticklabels([f'$d={d}$' for d in dims]); ax.set_ylim(0, 108)
ax.set_ylabel('Retention (%)'); ax.set_title('(a) DAGMA Transfer Results', fontweight='bold')
ax.legend(framealpha=0.9, fontsize=7)

# ═══════════ (b) GOLEM 5-dim with error bars ═══════════
ax2 = fig.add_subplot(2, 3, 2)
scr_vals_g = [golem_data[d]['scratch']['mean'] if d in golem_data and 'scratch' in golem_data[d] else np.nan for d in dims]
warm_vals_g = [golem_data[d]['warm']['mean'] if d in golem_data and 'warm' in golem_data[d] else np.nan for d in dims]
scr_std_g = [golem_data[d]['scratch']['std'] if d in golem_data and 'scratch' in golem_data[d] else 0 for d in dims]
warm_std_g = [golem_data[d]['warm']['std'] if d in golem_data and 'warm' in golem_data[d] else 0 for d in dims]

ax2.bar(x - w/2, scr_vals_g, w, color=SCRATCH, alpha=0.85, label='Scratch',
        yerr=scr_std_g, error_kw={'capsize': 3, 'linewidth': 0.8, 'ecolor': '#555'})
bars_g = ax2.bar(x + w/2, warm_vals_g, w, color=GOLEM_C, alpha=0.85, label='Warm-start',
                 yerr=warm_std_g, error_kw={'capsize': 3, 'linewidth': 0.8, 'ecolor': '#555'})
for b, v in zip(bars_g, warm_vals_g):
    if not np.isnan(v):
        ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 2.0, f'{v:.0f}',
                ha='center', fontsize=7, fontweight='bold', color=GOLEM_C)
ax2.set_xticks(x); ax2.set_xticklabels([f'$d={d}$' for d in dims]); ax2.set_ylim(0, 108)
ax2.set_ylabel('Retention (%)'); ax2.set_title('(b) GOLEM Transfer Results', fontweight='bold')
ax2.legend(framealpha=0.9, fontsize=7)

# ═══════════ (c) Cross-domain DepMap → loaded from full_experiment.json ═══════════
ax3 = fig.add_subplot(2, 3, 3)
methods_c = ['Scratch', 'TCGA-warm', 'DepMap-\nwarm']
vals_c = [depmap_scr, depmap_tcw, depmap_dmw]
colors_c = [SCRATCH, WARM, DM_C]
bars3 = ax3.bar(methods_c, vals_c, color=colors_c, alpha=0.85, width=0.5)
for b, v in zip(bars3, vals_c):
    ax3.text(b.get_x() + b.get_width()/2, b.get_height() + 0.3, str(v),
             ha='center', fontweight='bold', fontsize=13, color='#333')
ax3.set_ylabel(f'Overlap with DepMap Source\n({depmap_src_edges} source edges, $d=454$)')
ax3.set_title('(c) Cross-Domain: DepMap $\\to$ BRCA', fontweight='bold')
ax3.set_ylim(0, max(vals_c) * 1.4)

# ═══════════ (d) Multi-algo delta summary ═══════════
ax4 = fig.add_subplot(2, 3, 4)
w4 = 0.22
delta_nt = [notears_data[d]['warm']['mean'] - notears_data[d]['scratch']['mean']
            if d in notears_data and 'warm' in notears_data[d] else 0 for d in dims]
delta_dg = [dagma_data[d]['warm']['mean'] - dagma_data[d]['scratch']['mean']
            if d in dagma_data and 'warm' in dagma_data[d] else 0 for d in dims]
delta_gl = [golem_data[d]['warm']['mean'] - golem_data[d]['scratch']['mean']
            if d in golem_data and 'warm' in golem_data[d] else 0 for d in dims]

ax4.bar(x - w4, delta_nt, w4, color=WARM, alpha=0.85, label='NOTEARS')
ax4.bar(x, delta_dg, w4, color=DAGMA_C, alpha=0.85, label='DAGMA')
ax4.bar(x + w4, delta_gl, w4, color=GOLEM_C, alpha=0.85, label='GOLEM')
ax4.set_xticks(x); ax4.set_xticklabels([f'$d={d}$' for d in dims])
ax4.set_ylabel('$\\Delta$ Retention (pp)')
ax4.set_title('(d) Warm-Start Advantage by Algorithm', fontweight='bold')
ax4.legend(framealpha=0.9, ncol=3, fontsize=7)
ax4.axhline(y=0, color='gray', lw=0.5, alpha=0.3)

# ═══════════ (e) STRING validation → loaded from external_validation.json ═══════════
ax5 = fig.add_subplot(2, 3, 5)
x5 = np.arange(len(string_methods)); w5 = 0.22
colors_st = ['#3498DB', '#F39C12', '#E74C3C']
labels_st = [f'score $\\geq {t}$' for t in string_thresh]

for i in range(3):
    offset = (i - 1) * w5
    ax5.bar(x5 + offset, string_prec[i], w5, color=colors_st[i], alpha=0.85, label=labels_st[i])

ax5.set_xticks(x5); ax5.set_xticklabels(string_methods, fontsize=7.5)
ax5.set_ylabel('STRING Precision (%)'); ax5.set_ylim(0, 6.5)
ax5.set_title('(e) STRING v12 Validation', fontweight='bold')
ax5.legend(framealpha=0.9, fontsize=6.5, ncol=3, loc='upper right')

# ═══════════ (f) KEGG enrichment ═══════════
ax6 = fig.add_subplot(2, 3, 6)
ax6.axis('off')
kegg_text = (
    'KEGG Pathway Enrichment ($d=454$, top-200 edges)\n'
    '\n'
    'Scratch & DepMap-warm:\n'
    '  Chemokine signaling (hsa04062)\n'
    '  FDR = 0.036 | 7 genes\n'
    '  CCL13, CCL11, CCL19, XCR1,\n'
    '  XCL1, VAV3, CCL3L3\n'
    '\n'
    'TCGA source & TCGA-warm:\n'
    '  No significant enrichment\n'
    '  (FDR < 0.05)\n'
    '\n'
    'Chemokine signaling mediates tumor-\n'
    'immune microenvironment interactions\n'
    'across cancer types \u2014 a conserved\n'
    'biological signal detectable by both\n'
    'scratch and transfer-learned graphs.'
)
ax6.text(0.05, 0.95, kegg_text, transform=ax6.transAxes, fontsize=8,
        va='top', linespacing=1.6,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#F5F7FA', edgecolor='#BDC3C7', alpha=0.9))

plt.tight_layout(pad=1.5)
fig.savefig(os.path.join(FIGURES, 'fig3_generality.pdf'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(FIGURES, 'fig3_generality.png'), dpi=300, bbox_inches='tight')
print(f'Fig 3 saved: {os.path.join(FIGURES, "fig3_generality.pdf")}')
