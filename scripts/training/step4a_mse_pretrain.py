"""
CausalGPT v4: Pure MSE gene embedding backbone.
DAG constraint deferred to fine-tuning (NOTEARS).
"""
import os, sys, json, time, argparse, numpy as np
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch, torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--d', type=int, default=500)
parser.add_argument('--r', type=int, default=128)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--lambda_l2', type=float, default=1e-6)
parser.add_argument('--lambda_l1', type=float, default=1e-6)
parser.add_argument('--device', type=str, default='cuda:0')
args = parser.parse_args()

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, 'data'); OUT = os.path.join(ROOT, 'checkpoints')
os.makedirs(OUT, exist_ok=True)
dev = torch.device(args.device if torch.cuda.is_available() else 'cpu')
print(f'd={args.d} r={args.r} epochs={args.epochs}')

from scipy.stats import median_abs_deviation
MC = os.path.join(OUT, f'mad_top{args.d}.npy')
if os.path.exists(MC):
    top = np.load(MC)
else:
    Xf = np.load(os.path.join(DATA,'X_depmap_final.npy'),mmap_mode='r').astype(np.float32)
    tn = 'X_tcga_nobrca.npy' if os.path.exists(os.path.join(DATA,'X_tcga_nobrca.npy')) else 'X_tcga_final.npy'
    Xt = np.load(os.path.join(DATA,tn),mmap_mode='r').astype(np.float32)
    dm = min(Xf.shape[1],Xt.shape[1])
    mads = np.array([median_abs_deviation(np.concatenate([Xf[:,j],Xt[:,j]])) for j in range(dm)])
    top = np.argsort(mads)[-args.d:]; np.save(MC,top)

tf = 'X_tcga_nobrca.npy' if os.path.exists(os.path.join(DATA,'X_tcga_nobrca.npy')) else 'X_tcga_final.npy'
Xd = np.load(os.path.join(DATA,'X_depmap_final.npy'),mmap_mode='r').astype(np.float32)[:,top].copy()
Xt = np.load(os.path.join(DATA,tf),mmap_mode='r').astype(np.float32)[:,top].copy()
print(f'DepMap {Xd.shape} TCGA {Xt.shape}')

class GeneEmb(torch.nn.Module):
    def __init__(self,d,r):
        super().__init__(); self.d,self.r = d,r
        self.U0 = torch.nn.Parameter(torch.randn(d,r)*0.01)
        self.V0 = torch.nn.Parameter(torch.randn(d,r)*0.01)
        self.dU = torch.nn.ParameterDict({
            'dm':torch.nn.Parameter(torch.zeros(d,r)),
            'tc':torch.nn.Parameter(torch.zeros(d,r))})
        self.dV = torch.nn.ParameterDict({
            'dm':torch.nn.Parameter(torch.zeros(d,r)),
            'tc':torch.nn.Parameter(torch.zeros(d,r))})
    def get(self,ds):
        return self.U0+self.dU[ds], self.V0+self.dV[ds]
    def W(self):
        return self.U0 @ self.V0.T

m = GeneEmb(args.d,args.r).to(dev)
opt = torch.optim.Adam(m.parameters(),lr=args.lr)
se = max(1,Xd.shape[0]//args.batch_size)+max(1,Xt.shape[0]//args.batch_size)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs*se)

CK = os.path.join(OUT,f'mse_d{args.d}_r{args.r}.pt')
se_epoch=0; lh=[]; gs=0
if os.path.exists(CK):
    ck=torch.load(CK,map_location=dev,weights_only=False)
    m.load_state_dict(ck['ms']); opt.load_state_dict(ck['os'])
    sch.load_state_dict(ck['ss']); se_epoch=ck['ep']; lh=ck.get('lh',[]); gs=ck.get('gs',0)
    print(f'Resume: epoch {se_epoch} step {gs}')

t0=time.time()
for ep in range(se_epoch,args.epochs):
    m.train(); el=0.0; nb=0
    for name,X in [('dm',Xd),('tc',Xt)]:
        perm=torch.randperm(X.shape[0])
        for b in range(0,X.shape[0],args.batch_size):
            Xb=torch.from_numpy(X[perm[b:b+args.batch_size]]).to(dev)
            U,V=m.get(name)
            rcn=(Xb@U)@V.T
            loss=F.mse_loss(rcn,Xb)
            loss+=args.lambda_l1*(U.abs().mean()+V.abs().mean())
            loss+=args.lambda_l2*((U*U).mean()+(V*V).mean())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),10.0)
            opt.step();opt.zero_grad();sch.step()
            el+=loss.item();nb+=1;gs+=1
    lh.append(el/nb)
    if ep%30==0 or ep==args.epochs-1:
        print(f'  E{ep+1:4d}/{args.epochs} loss={lh[-1]:.4f} | {time.time()-t0:.0f}s')
    if (ep+1)%50==0:
        torch.save({'ep':ep+1,'gs':gs,'ms':m.state_dict(),'os':opt.state_dict(),'ss':sch.state_dict(),'lh':lh},CK)

print(f'Done: {(time.time()-t0)/60:.1f}min steps={gs}')

# Eval
with torch.no_grad():
    W=m.W().cpu().numpy()
from scipy.linalg import expm
h=float(np.trace(expm(W*W))-args.d)
nz03=int((np.abs(W)>0.3).sum()); nz01=int((np.abs(W)>0.1).sum()); nz005=int((np.abs(W)>0.05).sum())
mw=float(np.abs(W).max()); print(f'\nmax|W|={mw:.4f} h={h:.2e} edges(tau0.3/0.1/0.05)={nz03}/{nz01}/{nz005}')

# Gene embedding quality
U=m.U0.data.cpu().numpy(); Un=U/(np.linalg.norm(U,axis=1,keepdims=True)+1e-12)
ns=min(1000,args.d); ix=np.random.choice(args.d,ns,replace=False)
cs=Un[ix]@Un[ix].T; ut=np.triu_indices(ns,k=1); co=cs[ut]
print(f'Gene emb cos-sim: mean={co.mean():.4f} std={co.std():.4f} p99={np.percentile(co,99):.4f}')

# Export
HD=os.path.join(OUT,f'causal-gpt-v1-d{args.d}'); os.makedirs(HD,exist_ok=True)
np.save(os.path.join(HD,'U0.npy'),m.U0.data.cpu().numpy())
np.save(os.path.join(HD,'V0.npy'),m.V0.data.cpu().numpy())
np.save(os.path.join(HD,'gene_idx.npy'),top)
json.dump({'d':args.d,'r':args.r,'h':f'{h:.2e}','max_w':round(mw,6),
           'nz03':nz03,'nz01':nz01,'nz005':nz005,'loss':lh[-1],'eps':args.epochs},
          open(os.path.join(HD,'config.json'),'w'),indent=2)
print(f'Exported: {HD}/')
