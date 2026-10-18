# --- paths resolved relative to this file (replication package) ---
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
PKG_ROOT = _HERE
for _ in range(2):
    PKG_ROOT = _os.path.dirname(PKG_ROOT)
PKG_RESULTS = _os.path.join(PKG_ROOT, "results")
DATA_DIR = _os.environ.get("TCGA_DATA_DIR", _os.path.join(PKG_ROOT, "data"))
# -------------------------------------------------------------------
"""
CausalGPT Pre-training: Multi-task low-rank factorization backbone.
No DAG constraint — pre-training learns gene embeddings (U,V rows).
DAG is enforced during fine-tuning via NOTEARS augmented Lagrangian.

Architecture:
  W_m = (U0 + dU_m) @ (V0 + dV_m)^T   for dataset m
  Loss = ||X_m - X_m @ W_m||^2 / n_m  +  lambda * (||U0||^2 + ||V0||^2)

Scaling: O(bdr) per batch, fully GPU-friendly at d=16078.

Usage:
  python step3_pretrain.py --d 500   --r 256 --epochs 100   # Phase 1: pilot
  python step3_pretrain.py --d 2000  --r 256 --epochs 100   # Phase 2: scale
  python step3_pretrain.py --d 16078 --r 256 --epochs 100   # Phase 3: full
"""
import os, sys, json, time, argparse, shutil
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import torch
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--d', type=int, default=500)
parser.add_argument('--r', type=int, default=256)
parser.add_argument('--epochs', type=int, default=100, help='Passes over each dataset')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--lambda_l2', type=float, default=1e-5)
parser.add_argument('--lambda_l1', type=float, default=1e-6, help='L1 on deltas')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--output_dir', type=str,
                    default=os.path.join(PKG_ROOT, "checkpoints"))
parser.add_argument('--log_every', type=int, default=50)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
print(f'Device: {device} | d={args.d} r={args.r} | epochs={args.epochs} batch={args.batch_size}')

# ===========================================================================
# 1. DATA
# ===========================================================================
DATA = os.path.join(PKG_ROOT, "data")
MAD_CACHE = os.path.join(args.output_dir, f'mad_top{args.d}.npy')

from scipy.stats import median_abs_deviation
if os.path.exists(MAD_CACHE):
    top_idx = np.load(MAD_CACHE)
else:
    Xf = np.load(os.path.join(DATA, 'X_depmap_final.npy'), mmap_mode='r').astype(np.float32)
    Xt = np.load(os.path.join(DATA, 'X_tcga_final.npy'), mmap_mode='r').astype(np.float32)
    mads = np.array([median_abs_deviation(np.concatenate([Xf[:,j], Xt[:,j]]))
                     for j in range(min(Xf.shape[1], Xt.shape[1]))])
    top_idx = np.argsort(mads)[-args.d:]
    np.save(MAD_CACHE, top_idx)

Xd = np.load(os.path.join(DATA, 'X_depmap_final.npy'), mmap_mode='r').astype(np.float32)[:, top_idx].copy()
tcga_file = 'X_tcga_nobrca.npy' if os.path.exists(os.path.join(DATA, 'X_tcga_nobrca.npy')) else 'X_tcga_final.npy'
Xt = np.load(os.path.join(DATA, tcga_file), mmap_mode='r').astype(np.float32)[:, top_idx].copy()
print(f'Data: DepMap {Xd.shape}, TCGA(-BRCA) {Xt.shape}')

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
        # Dataset-specific deltas
        self.dU = torch.nn.ParameterDict({
            'depmap': torch.nn.Parameter(torch.zeros(d, r)),
            'tcga': torch.nn.Parameter(torch.zeros(d, r)),
        })
        self.dV = torch.nn.ParameterDict({
            'depmap': torch.nn.Parameter(torch.zeros(d, r)),
            'tcga': torch.nn.Parameter(torch.zeros(d, r)),
        })

    def get_W(self, ds):
        U = self.U0 + self.dU[ds]
        V = self.V0 + self.dV[ds]
        return U @ V.T  # (d, d) — use low-rank trick in forward

    def shared_W(self):
        return self.U0 @ self.V0.T

model = CausalGPTBackbone(args.d, args.r).to(device)
print(f'Params: {sum(p.numel() for p in model.parameters()):,}')

# ===========================================================================
# 3. OPTIMIZER
# ===========================================================================
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
steps_per_epoch_dm = max(1, Xd.shape[0] // args.batch_size)
steps_per_epoch_tc = max(1, Xt.shape[0] // args.batch_size)
total_steps = args.epochs * (steps_per_epoch_dm + steps_per_epoch_tc)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

# ===========================================================================
# 4. LOSS (no DAG constraint)
# ===========================================================================
def compute_loss(X_batch, U, V, lambda_l2, lambda_l1):
    """||X - X@U@V^T||^2 / n"""
    # Low-rank trick: X@U@V^T = (X@U)@V^T
    # X: (b,d), U: (d,r), V: (d,r)
    # XU = X @ U: (b,r)     ←  small!
    # XUV^T = XU @ V^T: (b,d) ← reconstruction
    XU = X_batch @ U         # (b, r)
    recon = XU @ V.T         # (b, d)
    loss_recon = F.mse_loss(recon, X_batch)  # mean over all elements
    loss_l2 = lambda_l2 * (U.norm()**2 + V.norm()**2)
    return loss_recon + loss_l2

# ===========================================================================
# 5. TRAINING LOOP
# ===========================================================================
LATEST = os.path.join(args.output_dir, 'pretrain_latest.pt')
start_epoch = 0
loss_hist = []

if os.path.exists(LATEST):
    print('\n=== RESUMING ===')
    ckpt = torch.load(LATEST, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    opt.load_state_dict(ckpt['optimizer_state'])
    scheduler.load_state_dict(ckpt['scheduler_state'])
    start_epoch = ckpt['epoch']
    loss_hist = ckpt.get('loss_history', [])
    print(f'  Resumed at epoch {start_epoch}/{args.epochs}')

if start_epoch >= args.epochs:
    print('Already done.')
else:
    t0 = time.time()
    global_step = start_epoch * (steps_per_epoch_dm + steps_per_epoch_tc)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        ep_loss = 0.0
        n_batches = 0

        # DepMap
        perm_dm = torch.randperm(Xd.shape[0])
        for b in range(0, Xd.shape[0], args.batch_size):
            idx = perm_dm[b:b+args.batch_size]
            Xb = torch.from_numpy(Xd[idx]).to(device)
            U = model.U0 + model.dU['depmap']
            V = model.V0 + model.dV['depmap']
            loss = compute_loss(Xb, U, V, args.lambda_l2, args.lambda_l1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            opt.zero_grad()
            scheduler.step()
            ep_loss += loss.item()
            n_batches += 1
            global_step += 1

        # TCGA
        perm_tc = torch.randperm(Xt.shape[0])
        for b in range(0, Xt.shape[0], args.batch_size):
            idx = perm_tc[b:b+args.batch_size]
            Xb = torch.from_numpy(Xt[idx]).to(device)
            U = model.U0 + model.dU['tcga']
            V = model.V0 + model.dV['tcga']
            loss = compute_loss(Xb, U, V, args.lambda_l2, args.lambda_l1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            opt.zero_grad()
            scheduler.step()
            ep_loss += loss.item()
            n_batches += 1
            global_step += 1

        avg_loss = ep_loss / max(n_batches, 1)
        loss_hist.append(avg_loss)
        elapsed = time.time() - t0

        if epoch % 2 == 0 or epoch == args.epochs - 1:
            lr_now = scheduler.get_last_lr()[0]
            print(f'  Epoch {epoch+1:4d}/{args.epochs} | Loss {avg_loss:.4f} | '
                  f'LR {lr_now:.1e} | {elapsed:.0f}s')

        # Checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            torch.save({
                'epoch': epoch + 1,
                'model_state': model.state_dict(),
                'optimizer_state': opt.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'loss_history': loss_hist,
                'args': vars(args),
            }, LATEST)

    total_min = (time.time() - t0) / 60
    print(f'\nTraining done: {total_min:.1f} min')

# ===========================================================================
# 6. EXPORT BACKBONE
# ===========================================================================
hf_dir = os.path.join(args.output_dir, 'causal-gpt-v1')
os.makedirs(hf_dir, exist_ok=True)
np.save(os.path.join(hf_dir, 'U0.npy'), model.U0.data.cpu().numpy())
np.save(os.path.join(hf_dir, 'V0.npy'), model.V0.data.cpu().numpy())

W = model.shared_W()
W_np = W.detach().cpu().numpy()
nz = int((np.abs(W_np) > 0.3).sum())
max_w = float(np.abs(W_np).max())
from scipy.linalg import expm
h_val = float(np.trace(expm(W_np * W_np)) - args.d)

meta = {
    'd': args.d, 'r': args.r, 'epochs': args.epochs,
    'datasets': ['depmap', 'tcga'],
    'n_edges_tau0.3': nz, 'max_abs_W': round(max_w, 4),
    'h_W': f'{h_val:.2e}',
    'final_loss': loss_hist[-1] if loss_hist else None,
    'license': 'MIT', 'author': 'Shuaidong Gao',
}
with open(os.path.join(hf_dir, 'config.json'), 'w') as f:
    json.dump(meta, f, indent=2)

print(f'Backbone: {hf_dir}/')
print(f'  Edges(tau=0.3): {nz} | max|W|: {max_w:.4f} | h(W): {h_val:.2e}')
