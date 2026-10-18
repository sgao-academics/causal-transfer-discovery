"""
CausalGPT Pre-training v3: NOTEARS Augmented Lagrangian with DAG Gradient.

KEY FIX over v2: DAG penalty (rho/2*h^2 + alpha*h) now flows through autograd
to push U and V toward a DAG, not just computed for logging.

Architecture:
  W_shared = U0 @ V0^T                           (shared backbone)
  W_m = (U0 + dU_m) @ (V0 + dV_m)^T             (dataset-specific)

Loss at regular step:  MSE_recon + L1 + L2
Loss at outer step:    MSE_recon + L1 + L2 + rho/2*h(W)^2 + alpha*h(W)
                        ↑ DAG penalty gradient flows through autograd

Usage:
  python step3c_pretrain_hybrid.py --d 500  --epochs 200  --outer_step 100
  python step3c_pretrain_hybrid.py --d 2000 --epochs 300  --outer_step 50
"""
import os, sys, json, time, argparse
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import torch
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--d', type=int, default=500)
parser.add_argument('--r', type=int, default=256)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--lambda_l2', type=float, default=1e-5)
parser.add_argument('--lambda_l1', type=float, default=1e-5)
parser.add_argument('--outer_step', type=int, default=100,
                    help='Batches between DAG penalty backwards (injects DAG gradient)')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--data_dir', type=str, default=None)
parser.add_argument('--output_dir', type=str, default=None)
parser.add_argument('--h_tol', type=float, default=5e-3,
                    help='Stop if h(W) < tol')
args = parser.parse_args()

# Paths
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = args.data_dir or os.path.join(ROOT, 'data')
OUT_DIR = args.output_dir or os.path.join(ROOT, 'checkpoints')
os.makedirs(OUT_DIR, exist_ok=True)
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

print(f'Device: {device} | d={args.d} r={args.r} epochs={args.epochs} outer_step={args.outer_step}')
print(f'L1={args.lambda_l1} L2={args.lambda_l2}')

# ===========================================================================
# 1. DATA
# ===========================================================================
from scipy.stats import median_abs_deviation
MAD_CACHE = os.path.join(OUT_DIR, f'mad_top{args.d}.npy')

if os.path.exists(MAD_CACHE):
    top_idx = np.load(MAD_CACHE)
else:
    Xf = np.load(os.path.join(DATA_DIR, 'X_depmap_final.npy'), mmap_mode='r').astype(np.float32)
    tcga_src = 'X_tcga_nobrca.npy' if os.path.exists(os.path.join(DATA_DIR, 'X_tcga_nobrca.npy')) else 'X_tcga_final.npy'
    Xt = np.load(os.path.join(DATA_DIR, tcga_src), mmap_mode='r').astype(np.float32)
    d_max = min(Xf.shape[1], Xt.shape[1])
    mads = np.array([median_abs_deviation(np.concatenate([Xf[:,j], Xt[:,j]])) for j in range(d_max)])
    top_idx = np.argsort(mads)[-args.d:]
    np.save(MAD_CACHE, top_idx)

tcga_file = 'X_tcga_nobrca.npy' if os.path.exists(os.path.join(DATA_DIR, 'X_tcga_nobrca.npy')) else 'X_tcga_final.npy'
Xd = np.load(os.path.join(DATA_DIR, 'X_depmap_final.npy'), mmap_mode='r').astype(np.float32)[:, top_idx].copy()
Xt = np.load(os.path.join(DATA_DIR, tcga_file), mmap_mode='r').astype(np.float32)[:, top_idx].copy()
print(f'Data: DepMap {Xd.shape}, TCGA {Xt.shape}')

# ===========================================================================
# 2. MODEL
# ===========================================================================
class CausalGPTBackbone(torch.nn.Module):
    def __init__(self, d, r):
        super().__init__()
        self.d, self.r = d, r
        self.U0 = torch.nn.Parameter(torch.randn(d, r) * 0.01)
        self.V0 = torch.nn.Parameter(torch.randn(d, r) * 0.01)
        self.dU = torch.nn.ParameterDict({
            'depmap': torch.nn.Parameter(torch.zeros(d, r)),
            'tcga': torch.nn.Parameter(torch.zeros(d, r)),
        })
        self.dV = torch.nn.ParameterDict({
            'depmap': torch.nn.Parameter(torch.zeros(d, r)),
            'tcga': torch.nn.Parameter(torch.zeros(d, r)),
        })

    def get_params(self, ds):
        return self.U0 + self.dU[ds], self.V0 + self.dV[ds]

    def shared_W(self):
        return self.U0 @ self.V0.T

model = CausalGPTBackbone(args.d, args.r).to(device)
print(f'Params: {sum(p.numel() for p in model.parameters()):,}')

# ===========================================================================
# 3. OPTIMIZER & LAGRANGIAN STATE
# ===========================================================================
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
steps_per_epoch_dm = max(1, Xd.shape[0] // args.batch_size)
steps_per_epoch_tc = max(1, Xt.shape[0] // args.batch_size)
total_steps = args.epochs * (steps_per_epoch_dm + steps_per_epoch_tc)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

alpha = 0.0
rho = 0.001           # Start small to avoid DAG penalty crushing W to zero
dag_warmup_epochs = 50  # First N epochs: pure MSE (no DAG), build up weights
dag_ramp_epochs = 30    # Next M epochs: linearly ramp DAG weight 0 -> 1

# ===========================================================================
# 4. CHECKPOINT / RESUME
# ===========================================================================
LATEST = os.path.join(OUT_DIR, f'pretrain_d{args.d}.pt')
start_epoch = 0
global_step = 0
loss_hist, h_hist = [], []

if os.path.exists(LATEST):
    print('\n=== RESUMING ===')
    ckpt = torch.load(LATEST, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    opt.load_state_dict(ckpt['optimizer_state'])
    scheduler.load_state_dict(ckpt['scheduler_state'])
    start_epoch = ckpt['epoch']; global_step = ckpt['global_step']
    loss_hist = ckpt.get('loss_history', []); h_hist = ckpt.get('h_history', [])
    alpha = ckpt.get('alpha', 0.0); rho = ckpt.get('rho', 0.1)
    print(f'  Resumed: epoch {start_epoch} step {global_step} alpha={alpha:.2e} rho={rho:.2e}')

# ===========================================================================
# 5. TRAINING LOOP
# ===========================================================================
print(f'\n=== TRAINING ===')
print(f'  Batches per epoch: DM={steps_per_epoch_dm} TC={steps_per_epoch_tc}')
print(f'  DAG gradient injected every {args.outer_step} batches')
t0 = time.time()
dag_penalty_applied = 0

for epoch in range(start_epoch, args.epochs):
    model.train()
    ep_loss = 0.0; n_batches = 0

    # --- DepMap ---
    perm_dm = torch.randperm(Xd.shape[0])
    for b in range(0, Xd.shape[0], args.batch_size):
        idx = perm_dm[b:b+args.batch_size]
        Xb = torch.from_numpy(Xd[idx]).to(device)
        U, V = model.get_params('depmap')

        # Base loss: MSE + L1 + L2
        XU = Xb @ U
        recon = XU @ V.T
        loss_mse = F.mse_loss(recon, Xb)
        loss_l1 = args.lambda_l1 * (U.abs().mean() + V.abs().mean())
        loss_l2 = args.lambda_l2 * ((U*U).mean() + (V*V).mean())
        loss = loss_mse + loss_l1 + loss_l2

        # ---- DAG PENALTY INJECTION (with warmup) ----
        is_outer = (global_step > 0 and global_step % args.outer_step == 0)
        if is_outer:
            # Warmup: linearly ramp DAG weight from 0 to 1
            dag_weight = 0.0
            if epoch >= dag_warmup_epochs:
                dag_weight = min(1.0, (epoch - dag_warmup_epochs) / dag_ramp_epochs)
            W = model.shared_W()                    # (d,d) with grad
            h_val = torch.trace(torch.linalg.matrix_exp(W * W)) - args.d
            dag_penalty = dag_weight * ((rho / 2) * h_val**2 + alpha * h_val)
            loss = loss + dag_penalty
            dag_penalty_applied += 1

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step(); opt.zero_grad(); scheduler.step()

        ep_loss += loss.item(); n_batches += 1; global_step += 1

        if is_outer:
            h_float = h_val.item()
            h_hist.append({'step': global_step, 'epoch': epoch+1, 'h': h_float,
                           'alpha': alpha, 'rho': rho, 'dag_weight': dag_weight,
                           'dag_penalty': dag_penalty.item()})
            # Only update Lagrangian if DAG is active
            if dag_weight > 0:
                alpha += rho * h_float
                rho = min(rho * 1.1, 1e10)
            lr_now = scheduler.get_last_lr()[0]
            print(f'  [Step {global_step:5d}] E{epoch+1:3d} | h={h_float:.3e} '
                  f'a={alpha:.2e} r={rho:.2e} w={dag_weight:.2f} | loss={loss.item():.4f} | LR={lr_now:.1e}')

            if abs(h_float) < args.h_tol:
                print(f'  *** DAG SATISFIED! h={h_float:.3e} ***')
                break

    # --- TCGA(-BRCA) ---
    perm_tc = torch.randperm(Xt.shape[0])
    for b in range(0, Xt.shape[0], args.batch_size):
        idx = perm_tc[b:b+args.batch_size]
        Xb = torch.from_numpy(Xt[idx]).to(device)
        U, V = model.get_params('tcga')

        XU = Xb @ U
        recon = XU @ V.T
        loss_mse = F.mse_loss(recon, Xb)
        loss_l1 = args.lambda_l1 * (U.abs().mean() + V.abs().mean())
        loss_l2 = args.lambda_l2 * ((U*U).mean() + (V*V).mean())
        loss = loss_mse + loss_l1 + loss_l2

        is_outer = (global_step > 0 and global_step % args.outer_step == 0)
        if is_outer:
            dag_weight = 0.0
            if epoch >= dag_warmup_epochs:
                dag_weight = min(1.0, (epoch - dag_warmup_epochs) / dag_ramp_epochs)
            W = model.shared_W()
            h_val = torch.trace(torch.linalg.matrix_exp(W * W)) - args.d
            dag_penalty = dag_weight * ((rho / 2) * h_val**2 + alpha * h_val)
            loss = loss + dag_penalty
            dag_penalty_applied += 1

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step(); opt.zero_grad(); scheduler.step()

        ep_loss += loss.item(); n_batches += 1; global_step += 1

        if is_outer:
            h_float = h_val.item()
            h_hist.append({'step': global_step, 'epoch': epoch+1, 'h': h_float,
                           'alpha': alpha, 'rho': rho, 'dag_weight': dag_weight,
                           'dag_penalty': dag_penalty.item()})
            if dag_weight > 0:
                alpha += rho * h_float
                rho = min(rho * 1.1, 1e10)
            lr_now = scheduler.get_last_lr()[0]
            print(f'  [Step {global_step:5d}] E{epoch+1:3d} | h={h_float:.3e} '
                  f'a={alpha:.2e} r={rho:.2e} w={dag_weight:.2f} | loss={loss.item():.4f} | LR={lr_now:.1e}')

            if abs(h_float) < args.h_tol:
                break

    avg_loss = ep_loss / max(n_batches, 1)
    loss_hist.append(avg_loss)
    elapsed = time.time() - t0

    if epoch % 10 == 0 or epoch == args.epochs - 1:
        h_last = h_hist[-1]['h'] if h_hist else float('inf')
        lr_now = scheduler.get_last_lr()[0]
        print(f'  Epoch {epoch+1:4d}/{args.epochs} | Loss {avg_loss:.4f} | '
              f'h={h_last:.3e} | DAG={dag_penalty_applied} | LR={lr_now:.1e} | {elapsed:.0f}s')

    if (epoch + 1) % 20 == 0:
        torch.save({
            'epoch': epoch+1, 'global_step': global_step,
            'model_state': model.state_dict(),
            'optimizer_state': opt.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'loss_history': loss_hist, 'h_history': h_hist,
            'alpha': alpha, 'rho': rho, 'args': vars(args),
        }, LATEST)

    if h_hist and abs(h_hist[-1]['h']) < args.h_tol:
        print(f'\n*** DAG converged at epoch {epoch+1}! ***')
        break

total_hr = (time.time() - t0) / 3600
print(f'\nTraining done: {total_hr:.1f} hr | {global_step} steps | {dag_penalty_applied} DAG injections')

# ===========================================================================
# 6. FINAL EVAL + EXPORT
# ===========================================================================
with torch.no_grad():
    W = model.shared_W()
    W_np = W.cpu().numpy()
    h_final = torch.trace(torch.linalg.matrix_exp(W * W)) - args.d
    h_final_f = h_final.item()

nz03 = int((np.abs(W_np) > 0.3).sum())
nz01 = int((np.abs(W_np) > 0.1).sum())
nz005 = int((np.abs(W_np) > 0.05).sum())
max_w = float(np.abs(W_np).max())

print(f'\n=== FINAL BACKBONE ===')
print(f'  h(W): {h_final_f:.4e}  |  alpha={alpha:.2e}  rho={rho:.2e}')
print(f'  max|W|={max_w:.4f}  |  diag_mean={float(np.diag(W_np).mean()):.4f}')
print(f'  Edges: tau0.3={nz03}  tau0.1={nz01}  tau0.05={nz005}')

# Export
hf_dir = os.path.join(OUT_DIR, f'causal-gpt-v1-d{args.d}')
os.makedirs(hf_dir, exist_ok=True)
np.save(os.path.join(hf_dir, 'U0.npy'), model.U0.data.cpu().numpy())
np.save(os.path.join(hf_dir, 'V0.npy'), model.V0.data.cpu().numpy())
np.save(os.path.join(hf_dir, 'gene_idx.npy'), top_idx)

with open(os.path.join(hf_dir, 'config.json'), 'w') as f:
    json.dump({
        'd': args.d, 'r': args.r, 'h_final': f'{h_final_f:.4e}',
        'n_edges_0.3': nz03, 'n_edges_0.1': nz01, 'n_edges_0.05': nz005,
        'max_abs_W': round(max_w, 6),
        'alpha': alpha, 'rho': rho, 'dag_injections': dag_penalty_applied,
        'epochs': epoch+1, 'total_steps': global_step,
    }, f, indent=2)

print(f'Backbone -> {hf_dir}/')

# Final save
torch.save({
    'epoch': epoch+1, 'global_step': global_step,
    'model_state': model.state_dict(),
    'optimizer_state': opt.state_dict(),
    'scheduler_state': scheduler.state_dict(),
    'loss_history': loss_hist, 'h_history': h_hist,
    'alpha': alpha, 'rho': rho, 'args': vars(args),
}, LATEST)
