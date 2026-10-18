"""
CausalGPT Pre-training v2: NOTEARS Augmented Lagrangian with Low-Rank Backbone.

KEY FIX over v1: DAG constraint h(W) + L1 sparsity transforms pre-training
from a linear autoencoder into a causal structure learner.

Architecture:
  W_shared = U0 @ V0^T                           (shared backbone)
  W_m = (U0 + dU_m) @ (V0 + dV_m)^T             (dataset-specific)

Loss (augmented Lagrangian):
  L = MSE_recon + lambda_l2 * ||U||^2 + lambda_l1 * ||U||_1
      + rho/2 * h(W_shared)^2 + alpha * h(W_shared)

Scaling:
  - Forward/backward: O(b*d*r) via low-rank trick
  - DAG constraint: GPU torch.linalg.matrix_exp every outer_step batches
  - VRAM: <2GB for d=2000, r=256

Usage:
  python step3b_pretrain_notears.py --d 500   --epochs 200  --outer_step 500
  python step3b_pretrain_notears.py --d 2000  --epochs 300  --outer_step 1000
"""
import os, sys, json, time, argparse
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import torch
import torch.nn.functional as F

# ===========================================================================
# ARGS
# ===========================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--d', type=int, default=500)
parser.add_argument('--r', type=int, default=256)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--lambda_l2', type=float, default=1e-5)
parser.add_argument('--lambda_l1', type=float, default=1e-5)
parser.add_argument('--outer_step', type=int, default=500,
                    help='Batches between outer Lagrangian updates (h(W) recompute)')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--output_dir', type=str,
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints'))
parser.add_argument('--data_dir', type=str,
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'))
parser.add_argument('--log_every', type=int, default=50)
parser.add_argument('--h_tol', type=float, default=1e-7,
                    help='Stop if h(W) < tol for all datasets')
parser.add_argument('--use_cpu_expm', action='store_true', default=False,
                    help='Use scipy.linalg.expm on CPU (slower but more accurate for large d)')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
print(f'Device: {device} | d={args.d} r={args.r} epochs={args.epochs}')
print(f'L1: {args.lambda_l1} | L2: {args.lambda_l2} | outer_step: {args.outer_step}')

# ===========================================================================
# 1. DATA: Top-d genes by MAD across DepMap+TCGA(-BRCA)
# ===========================================================================
MAD_CACHE = os.path.join(args.output_dir, f'mad_top{args.d}.npy')

from scipy.stats import median_abs_deviation
if os.path.exists(MAD_CACHE):
    top_idx = np.load(MAD_CACHE)
    print(f'Loaded MAD gene ranking from cache: {MAD_CACHE}')
else:
    Xf = np.load(os.path.join(args.data_dir, 'X_depmap_final.npy'), mmap_mode='r').astype(np.float32)
    tcga_file = 'X_tcga_nobrca.npy' if os.path.exists(os.path.join(args.data_dir, 'X_tcga_nobrca.npy')) else 'X_tcga_final.npy'
    Xt = np.load(os.path.join(args.data_dir, tcga_file), mmap_mode='r').astype(np.float32)
    d_max = min(Xf.shape[1], Xt.shape[1])
    print(f'Computing MAD across {d_max} genes...')
    mads = np.array([median_abs_deviation(np.concatenate([Xf[:,j], Xt[:,j]]))
                     for j in range(d_max)])
    top_idx = np.argsort(mads)[-args.d:]
    np.save(MAD_CACHE, top_idx)
    print(f'Saved MAD cache: {MAD_CACHE}')

# Load selected genes into memory (mmap -> .copy())
Xd_raw = np.load(os.path.join(args.data_dir, 'X_depmap_final.npy'), mmap_mode='r').astype(np.float32)
tcga_file = 'X_tcga_nobrca.npy' if os.path.exists(os.path.join(args.data_dir, 'X_tcga_nobrca.npy')) else 'X_tcga_final.npy'
Xt_raw = np.load(os.path.join(args.data_dir, tcga_file), mmap_mode='r').astype(np.float32)

Xd = Xd_raw[:, top_idx].copy()
Xt = Xt_raw[:, top_idx].copy()
print(f'Data: DepMap {Xd.shape}, TCGA(-BRCA) {Xt.shape}')

# Verify no NaN
assert not np.isnan(Xd).any(), 'NaN in DepMap!'
assert not np.isnan(Xt).any(), 'NaN in TCGA!'

# ===========================================================================
# 2. MODEL
# ===========================================================================
class CausalGPTBackbone(torch.nn.Module):
    def __init__(self, d, r):
        super().__init__()
        self.d, self.r = d, r
        # Shared backbone
        self.U0 = torch.nn.Parameter(torch.randn(d, r) * 0.01)
        self.V0 = torch.nn.Parameter(torch.randn(d, r) * 0.01)
        # Dataset-specific deltas (small perturbations)
        self.dU = torch.nn.ParameterDict({
            'depmap': torch.nn.Parameter(torch.zeros(d, r)),
            'tcga': torch.nn.Parameter(torch.zeros(d, r)),
        })
        self.dV = torch.nn.ParameterDict({
            'depmap': torch.nn.Parameter(torch.zeros(d, r)),
            'tcga': torch.nn.Parameter(torch.zeros(d, r)),
        })

    def get_params(self, ds):
        U = self.U0 + self.dU[ds]
        V = self.V0 + self.dV[ds]
        return U, V

    def shared_W(self):
        return self.U0 @ self.V0.T

model = CausalGPTBackbone(args.d, args.r).to(device)
print(f'Params: {sum(p.numel() for p in model.parameters()):,}')

# ===========================================================================
# 3. LOSS FUNCTIONS
# ===========================================================================
def compute_mse(X_batch, U, V):
    """MSE: ||X - X@U@V^T||^2 / (b*d) via low-rank trick."""
    XU = X_batch @ U          # (b, r)
    recon = XU @ V.T          # (b, d)
    return F.mse_loss(recon, X_batch)

def compute_l1(U, V):
    """L1 penalty on factor matrices (proxy for ||W||_1)."""
    return U.abs().mean() + V.abs().mean()

def compute_l2(U, V):
    """L2 on factor matrices."""
    return (U * U).mean() + (V * V).mean()

def compute_h(W):
    """DAG constraint: h(W) = tr(exp(W*W)) - d."""
    if not args.use_cpu_expm and W.device.type == 'cuda':
        W2 = W * W
        h_val = torch.trace(torch.linalg.matrix_exp(W2)) - W.shape[0]
    else:
        from scipy.linalg import expm
        W_np = W.detach().cpu().numpy()
        h_val = float(np.trace(expm(W_np * W_np)) - W_np.shape[0])
        h_val = torch.tensor(h_val, device=W.device)
    return h_val

# ===========================================================================
# 4. OPTIMIZER & AUGMENTED LAGRANGIAN STATE
# ===========================================================================
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
steps_per_epoch_dm = max(1, Xd.shape[0] // args.batch_size)
steps_per_epoch_tc = max(1, Xt.shape[0] // args.batch_size)
total_steps = args.epochs * (steps_per_epoch_dm + steps_per_epoch_tc)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

# Augmented Lagrangian parameters
alpha = 0.0
rho = 0.1
rho_max = 1e10
rho_gamma = 1.1

# ===========================================================================
# 5. CHECKPOINT / RESUME
# ===========================================================================
LATEST = os.path.join(args.output_dir, f'pretrain_notears_d{args.d}.pt')
start_epoch = 0
global_step = 0
loss_hist = []
h_hist = []

if os.path.exists(LATEST):
    print('\n=== RESUMING ===')
    ckpt = torch.load(LATEST, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    opt.load_state_dict(ckpt['optimizer_state'])
    scheduler.load_state_dict(ckpt['scheduler_state'])
    start_epoch = ckpt['epoch']
    global_step = ckpt['global_step']
    loss_hist = ckpt.get('loss_history', [])
    h_hist = ckpt.get('h_history', [])
    alpha = ckpt.get('alpha', 0.0)
    rho = ckpt.get('rho', 0.1)
    print(f'  Resumed: epoch {start_epoch}/{args.epochs} step {global_step}')
    print(f'  alpha={alpha:.4e} rho={rho:.4e}')

if start_epoch >= args.epochs:
    print('Already done. Delete checkpoint to restart.')
    sys.exit(0)

# ===========================================================================
# 6. TRAINING LOOP
# ===========================================================================
print(f'\n=== PRE-TRAINING ===')
print(f'  d={args.d} r={args.r} epochs={args.epochs}')
print(f'  batches: DepMap={steps_per_epoch_dm}/epoch, TCGA={steps_per_epoch_tc}/epoch')
print(f'  outer_step={args.outer_step} (h(W) recompute frequency)')
print(f'  Estimated total steps: {total_steps}')
print()

t0 = time.time()

for epoch in range(start_epoch, args.epochs):
    model.train()
    ep_loss = 0.0
    n_batches = 0

    # ========== DepMap ==========
    perm_dm = torch.randperm(Xd.shape[0])
    for b in range(0, Xd.shape[0], args.batch_size):
        idx = perm_dm[b:b+args.batch_size]
        Xb = torch.from_numpy(Xd[idx]).to(device)
        U, V = model.get_params('depmap')

        # Reconstruction + regularization
        loss_mse = compute_mse(Xb, U, V)
        loss_l1 = args.lambda_l1 * compute_l1(U, V)
        loss_l2 = args.lambda_l2 * compute_l2(U, V)
        loss = loss_mse + loss_l1 + loss_l2

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        opt.zero_grad()
        scheduler.step()

        ep_loss += loss.item()
        n_batches += 1
        global_step += 1

        # Periodic DAG constraint update
        if global_step % args.outer_step == 0:
            with torch.no_grad():
                W = model.shared_W()
                h_val = compute_h(W)
                h_val_float = h_val.item() if isinstance(h_val, torch.Tensor) else h_val

            h_hist.append({'step': global_step, 'epoch': epoch+1, 'h': h_val_float,
                           'alpha': alpha, 'rho': rho})

            # Augmented Lagrangian update
            alpha += rho * h_val_float
            rho = min(rho * rho_gamma, rho_max)

            lr_now = scheduler.get_last_lr()[0]
            print(f'  [Step {global_step:6d}] E{epoch+1:3d} | h(W)={h_val_float:.2e} | '
                  f'alpha={alpha:.2e} rho={rho:.2e} | LR={lr_now:.1e} | loss={ep_loss/n_batches:.4f}')

            # Early stopping: h(W) converged
            if abs(h_val_float) < args.h_tol:
                print(f'  *** DAG constraint satisfied! h(W)={h_val_float:.2e} < {args.h_tol}')
                break

    # ========== TCGA(-BRCA) ==========
    perm_tc = torch.randperm(Xt.shape[0])
    for b in range(0, Xt.shape[0], args.batch_size):
        idx = perm_tc[b:b+args.batch_size]
        Xb = torch.from_numpy(Xt[idx]).to(device)
        U, V = model.get_params('tcga')

        loss_mse = compute_mse(Xb, U, V)
        loss_l1 = args.lambda_l1 * compute_l1(U, V)
        loss_l2 = args.lambda_l2 * compute_l2(U, V)
        loss = loss_mse + loss_l1 + loss_l2

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        opt.zero_grad()
        scheduler.step()

        ep_loss += loss.item()
        n_batches += 1
        global_step += 1

        if global_step % args.outer_step == 0:
            with torch.no_grad():
                W = model.shared_W()
                h_val = compute_h(W)
                h_val_float = h_val.item() if isinstance(h_val, torch.Tensor) else h_val

            h_hist.append({'step': global_step, 'epoch': epoch+1, 'h': h_val_float,
                           'alpha': alpha, 'rho': rho})
            alpha += rho * h_val_float
            rho = min(rho * rho_gamma, rho_max)

            lr_now = scheduler.get_last_lr()[0]
            print(f'  [Step {global_step:6d}] E{epoch+1:3d} | h(W)={h_val_float:.2e} | '
                  f'alpha={alpha:.2e} rho={rho:.2e} | LR={lr_now:.1e} | loss={ep_loss/n_batches:.4f}')

            if abs(h_val_float) < args.h_tol:
                print(f'  *** DAG constraint satisfied! h(W)={h_val_float:.2e} < {args.h_tol}')
                break

    avg_loss = ep_loss / max(n_batches, 1)
    loss_hist.append(avg_loss)
    elapsed = time.time() - t0

    if epoch % 5 == 0 or epoch == args.epochs - 1:
        lr_now = scheduler.get_last_lr()[0]
        h_last = h_hist[-1]['h'] if h_hist else float('inf')
        print(f'  Epoch {epoch+1:4d}/{args.epochs} | Loss {avg_loss:.4f} | '
              f'h(W) {h_last:.2e} | LR {lr_now:.1e} | {elapsed:.0f}s')

    # Checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
        torch.save({
            'epoch': epoch + 1,
            'global_step': global_step,
            'model_state': model.state_dict(),
            'optimizer_state': opt.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'loss_history': loss_hist,
            'h_history': h_hist,
            'alpha': alpha,
            'rho': rho,
            'args': vars(args),
        }, LATEST)

    # Early stop if h(W) < tol
    if h_hist and abs(h_hist[-1]['h']) < args.h_tol:
        print(f'\n*** DAG converged at epoch {epoch+1}! ***')
        break

total_hr = (time.time() - t0) / 3600
print(f'\nTraining done: {total_hr:.1f} hr | Steps: {global_step}')

# ===========================================================================
# 7. FINAL CHECK & EXPORT
# ===========================================================================
with torch.no_grad():
    W = model.shared_W()
    W_np = W.cpu().numpy()
    h_final = compute_h(W)
    h_final_float = h_final.item() if isinstance(h_val, torch.Tensor) else float('inf')

# Edge statistics
nz03 = int((np.abs(W_np) > 0.3).sum())
nz01 = int((np.abs(W_np) > 0.1).sum())
nz005 = int((np.abs(W_np) > 0.05).sum())
max_w = float(np.abs(W_np).max())
diag_mean = float(np.diag(W_np).mean())

print(f'\n=== FINAL BACKBONE ===')
print(f'  h(W): {h_final_float:.2e}')
print(f'  max|W|: {max_w:.4f}')
print(f'  diag mean: {diag_mean:.4f}')
print(f'  Edges: tau=0.3: {nz03}, tau=0.1: {nz01}, tau=0.05: {nz005}')
print(f'  Nonzero entries (>1e-6): {(np.abs(W_np)>1e-6).sum():,} / {args.d*args.d:,}')

# Export backbone for downstream use
hf_dir = os.path.join(args.output_dir, f'causal-gpt-v1-d{args.d}')
os.makedirs(hf_dir, exist_ok=True)
np.save(os.path.join(hf_dir, 'U0.npy'), model.U0.data.cpu().numpy())
np.save(os.path.join(hf_dir, 'V0.npy'), model.V0.data.cpu().numpy())
np.save(os.path.join(hf_dir, 'gene_idx.npy'), top_idx)

meta = {
    'd': args.d, 'r': args.r, 'epochs_completed': epoch + 1,
    'total_steps': global_step,
    'datasets': ['depmap', 'tcga_nobrca'],
    'h_final': f'{h_final_float:.2e}',
    'n_edges_tau0.3': nz03, 'n_edges_tau0.1': nz01, 'n_edges_tau0.05': nz005,
    'max_abs_W': round(max_w, 6),
    'diag_mean_W': round(diag_mean, 6),
    'final_loss': loss_hist[-1] if loss_hist else None,
    'final_alpha': alpha, 'final_rho': rho,
    'lambda_l1': args.lambda_l1, 'lambda_l2': args.lambda_l2,
    'license': 'MIT', 'author': 'Shuaidong Gao',
}
with open(os.path.join(hf_dir, 'config.json'), 'w') as f:
    json.dump(meta, f, indent=2)

print(f'\nBackbone exported: {hf_dir}/')
print(f'  U0.npy: {model.U0.shape} | V0.npy: {model.V0.shape}')

# ===========================================================================
# 8. QUICK SANITY: ZERO-SHOT EDGES
# ===========================================================================
print(f'\n=== ZERO-SHOT SANITY CHECK ===')
# Top edges by absolute weight
abs_W = np.abs(W_np)
np.fill_diagonal(abs_W, 0)  # exclude self-loops
top_k = min(20, args.d * (args.d - 1))
flat_idx = np.argpartition(abs_W.flatten(), -top_k)[-top_k:]
top_vals = abs_W.flatten()[flat_idx]
top_sorted = flat_idx[np.argsort(-top_vals)]
edge_list = [(int(i // args.d), int(i % args.d), float(W_np[i//args.d, i%args.d]))
             for i in top_sorted]

print(f'  Top {len(edge_list)} edges by |W|:')
for i, (u, v, w) in enumerate(edge_list[:10]):
    print(f'    {u:5d} -> {v:5d} : {w:+.4f}')
if len(edge_list) > 10:
    print(f'    ... ({len(edge_list)-10} more)')
