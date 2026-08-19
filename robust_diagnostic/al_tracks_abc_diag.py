"""al_tracks_abc_diag.py: Tracks A/B/C on COV-SHIFT (anchor) + fallback budget curve.

Covers the promising directions from C19/C20, with the fallback you noted:
if none of the cheap-budget tracks show promise, the budget curve is extended
to high budgets to see WHERE it becomes positive and WHAT information the added
labels provide (so a better selection can compress it).

Tracks (eval-only, COV-SHIFT ep10 as anchor):
  A1  Budget curve: k per class in [8,16,32,64,128] (total 136-2176 labels) on
      the current best recipe (centroid k means, source counts, control variate
      rho=0.5, fractional-residual beta=0.6). Reports t_cos, w_cos, delta.
      This is the information- vs estimator- vs decoder-limited split you asked to track.
  A2  Adaptive allocation: for a fixed total budget B = 8*C_active (~136), allocate
      k_c propto N_c^alpha with alpha in {0, 0.25, 0.5, 1.0}. Uniform (alpha=0) vs
      mass-proportional (alpha=1) and the two intermediates. Reports per-class delta
      vs k_c to see where labels matter.
  B1/B2 Residual correction: W = W_cov + Delta, Delta = U_r C (r=4,8) with
      (a) oracle U_r = SVD(R), R=W*-W0 (ceiling) and (b) pool-cov U_r (deployable).
      C fit from labels on the residual Y - XW0. Compare to full-probe on same budget.
      If oracle U_r works but pool U_r does not, the premise holds but U must come
      from elsewhere (feat-shift).
  C   Feature-space residual branch (diagnostic only, no training): does the
      per-class code-mean shift M (17x10000) live in a small subspace? Already C20
      feat_shift eff-rank 1.2 on wet, but re-measured here per k to see if a
      small AL branch z=[z_cov, eps*z_AL] could capture it.

Fallback: if A/B/C are flat-negative at k=8-16, the A1 curve to 128 shows the
knee (where delta turns positive) and the per-k t_cos/w_cos + per-class coverage
reveal what the added labels bought (mean quality vs mass vs rare-class filling),
so the next step is a selection mechanism that gets that information at lower k.

Usage:
  uv run python robust_diagnostic/al_tracks_abc_diag.py \
    --path_b <ckpt_dir> --method_b <method> --label <label> \
    --conds fog,crosstalk,snow,wet_ground \
    --out robust_diagnostic/logs/al_tracks_abc_<label>.json
"""
import os, sys, time, argparse, json, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn.functional as F
from dataset.kitti.parser import Parser
from modules.gen_trainers import GenTrainer
from modules.oracle_core import get_hdc_projection, compute_miou

NUM_CLASSES=17; SKETCH_SEED=11

def build_parser(root,data,arch):
    return Parser(root=root,train_sequences=['08'],valid_sequences=['08'],test_sequences=None,
                  labels=data["labels"],color_map=data["color_map"],learning_map=data["learning_map"],
                  learning_map_inv=data["learning_map_inv"],sensor=arch["dataset"]["sensor"],
                  max_points=arch["dataset"]["max_points"],batch_size=1,workers=4,gt=True,shuffle_train=False)
def extract_features(model,parser,device,num_frames=100):
    feats,lbls=[],[]
    model.eval()
    with torch.no_grad():
        for i,batch in enumerate(parser.get_train_set()):
            if i>=num_frames: break
            in_vol=batch[0].to(device); labels=batch[2].to(device).view(-1); mask=(batch[1].to(device)>0).view(-1)
            out=model(in_vol); z8=out[2] if len(out)==3 else out[1]
            zf=z8.permute(0,2,3,1).reshape(-1,z8.shape[1])[mask]
            feats.append(zf.cpu()); lbls.append(labels[mask].cpu())
    return torch.cat(feats),torch.cat(lbls)
def hdc_codes(feats,proj,device,chunk=100000):
    out=[]
    for s in range(0,len(feats),chunk): out.append(torch.sign(feats[s:s+chunk].to(device)@proj).cpu())
    return torch.cat(out)
def onehot(lbls,nc):
    y=torch.zeros(len(lbls),nc); y[torch.arange(len(lbls)),lbls.long()]=1; return y
def decode(W,codes,chunk=100000):
    W=W.detach().cpu(); p=[]
    for s in range(0,len(codes),chunk): p.append((codes[s:s+chunk].float()@W).argmax(1))
    return torch.cat(p)
def mw(W,Xv,vl): return compute_miou(decode(W,Xv),vl)
def cos_sim(a,b):
    a=a.detach().cpu().float().reshape(-1); b=b.detach().cpu().float().reshape(-1)
    return float((a*b).sum()/(a.norm()*b.norm()+1e-30))
def ridge_fit_soft(X,Y,lam,iters,m,device):
    X=X.to(device); torch.manual_seed(SKETCH_SEED); m=min(m,X.shape[1])
    P=(torch.rand(X.shape[1],m,device=device)>0.5).float()*2-1; XP=X@P; Yd=Y.float().to(device)
    Shat=XP.t()@XP+lam*torch.eye(m,device=device); That=XP.t()@Yd
    x=P@torch.linalg.solve(Shat,That); b=X.t()@Yd
    def A(v): return X.t()@(X@v)
    r=b-A(x); p=r.clone(); rs=(r*r).sum(0)
    for _ in range(iters):
        Ap=A(p); a=rs/((p*Ap).sum(0)+1e-30); x=x+a.unsqueeze(0)*p; r=r-a.unsqueeze(0)*Ap
        rsn=(r*r).sum(0); be=rsn/(rs+1e-30); p=r+be.unsqueeze(0)*p; rs=rsn
    return x.float()
def lsq_residual(X_lab,Y_lab,W0,U,device):
    Xd=X_lab.to(device).float(); Yd=Y_lab.to(device).float(); U_d=U.to(device)
    r=U_d.shape[1]; XU=Xd@U_d
    A=XU.t()@XU+1e-6*torch.eye(r,device=device)
    b=XU.t()@(Yd - Xd@W0.to(device))
    return torch.linalg.solve(A,b).cpu()
def allocate_counts(total_B, freq, alpha):
    classes=list(freq.keys()); f=np.array([freq[c] for c in classes],dtype=float)
    w=np.power(f, alpha) if alpha>0 else np.ones_like(f)
    w=w/w.sum()*total_B
    k=np.floor(w).astype(int); k=np.maximum(k,1)
    rem=w - k; order=np.argsort(-rem); cur=k.sum()
    for idx in order:
        if cur>=total_B: break
        if k[idx] < freq[classes[idx]]:
            k[idx]+=1; cur+=1
    order=np.argsort(rem)
    for idx in order:
        if cur<=total_B: break
        if k[idx]>1:
            k[idx]-=1; cur-=1
    return {c:int(k[i]) for i,c in enumerate(classes)}
def sync():
    if torch.cuda.is_available(): torch.cuda.synchronize()
def tic():
    sync(); return time.time()
def toc(t0):
    sync(); return time.time()-t0

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--kitti_dir",type=str,default="/mnt/alpha/jmfleming/KITTI")
    ap.add_argument("--kittic_dir",type=str,default="/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C")
    ap.add_argument("--config",type=str,default="config/labels/semantic-kitti-all.yaml")
    ap.add_argument("--arch",type=str,default="config/arch/senet-2048p.yml")
    ap.add_argument("--frames",type=int,default=100)
    ap.add_argument("--pool_size",type=int,default=50000)
    ap.add_argument("--val_size",type=int,default=100000)
    ap.add_argument("--lam",type=float,default=1e-3)
    ap.add_argument("--max_clean",type=int,default=200000)
    ap.add_argument("--nystrom_m",type=int,default=1000)
    ap.add_argument("--cg_iters",type=int,default=8)
    ap.add_argument("--conds",type=str,default="fog,crosstalk,snow,wet_ground")
    ap.add_argument("--path_b",type=str,required=True)
    ap.add_argument("--method_b",type=str,required=True)
    ap.add_argument("--label",type=str,default="covshift_ep10")
    ap.add_argument("--out",type=str,required=True)
    args=ap.parse_args()
    DATA=yaml.safe_load(open(args.config)); ARCH=yaml.safe_load(open(args.arch))
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    conds=[c.strip() for c in args.conds.split(',') if c.strip()]
    trainer=GenTrainer(ARCH,DATA,args.kitti_dir,args.path_b,path=args.path_b,method=args.method_b)
    model=trainer.model
    clean_parser=build_parser(args.kitti_dir,DATA,ARCH)
    fa,la=extract_features(model,clean_parser,device,args.frames)
    results={'label':args.label,'method':args.method_b,'conds':{}}
    for cond in conds:
        t0=tic()
        cdir=os.path.join(args.kittic_dir,cond,'heavy')
        if not os.path.exists(cdir): cdir=os.path.join(args.kittic_dir,cond,'moderate')
        f,l=extract_features(model,build_parser(cdir,DATA,ARCH),device,args.frames)
        torch.manual_seed(42); perm=torch.randperm(len(f))
        pool,pl=f[perm[:args.pool_size]],l[perm[:args.pool_size]]
        val,vl=f[perm[-args.val_size:]],l[perm[-args.val_size:]]
        mc=min(args.max_clean,len(fa)); ci=torch.randperm(len(fa))[:mc]
        proj=get_hdc_projection(dim_in=fa.shape[1],dim_out=10000,device=device)
        Xc=hdc_codes(fa[ci],proj,device).float()
        Xp=hdc_codes(pool,proj,device).float()
        Xv=hdc_codes(val,proj,device).float()
        Xd=Xp.to(device); N=Xp.shape[0]
        W0=ridge_fit_soft(Xc,onehot(la[ci],NUM_CLASSES),args.lam,args.cg_iters,args.nystrom_m,device)
        Ws=ridge_fit_soft(Xp,onehot(pl,NUM_CLASSES),args.lam,args.cg_iters,args.nystrom_m,device)
        classes=sorted(set(pl.tolist()) & set(range(1,NUM_CLASSES)))
        cls_idx={c:(pl==c).nonzero().squeeze(1) for c in classes}
        S=(Xd.double().t()@Xd.double())/N; eigS,U=torch.linalg.eigh(S); eigS=eigS.float(); U=U.float()
        lam_hat=args.lam/N; sig=(eigS+lam_hat).clamp(min=lam_hat)
        T_or=torch.zeros(10000,NUM_CLASSES)
        for c in classes: T_or[:,c]=Xp[cls_idx[c]].sum(0)
        m0=pl==0
        if int(m0.sum().item())>0: T_or[:,0]=Xp[m0].sum(0)
        T_or=T_or/N; Uc=U.to(device); sig_d=sig.to(device)
        UtT_or=Uc.t()@T_or.to(device); W_or_spec=(Uc@((1.0/sig_d).unsqueeze(1)*UtT_or)).cpu().float()
        R=(Ws - W0).detach().cpu().float()
        U_full,_ ,_=torch.linalg.svd(R.double(),full_matrices=False); U_full=U_full.float()
        S2=(Xd.double().t()@Xd.double())/N; _, U_S=torch.linalg.eigh(S2); U_S=U_S.float()
        clean_classes=sorted(set(la[ci].tolist()) & set(range(1,NUM_CLASSES)))
        clean_idx={c:(la[ci]==c).nonzero().squeeze(1) for c in clean_classes}
        mu_pool={c: Xp[cls_idx[c]].mean(0) for c in classes if len(cls_idx[c])>0}
        mu_clean={c: Xc[clean_idx[c]].mean(0) for c in classes if c in clean_idx and len(clean_idx[c])>0}
        common=[c for c in classes if c in mu_clean]
        if len(common)>=2:
            M=torch.stack([mu_pool[c]-mu_clean[c] for c in common])
            _,_,Vh=torch.linalg.svd(M.double(),full_matrices=False); U_shift=Vh.t().float()
        else:
            U_shift=None
        r={'refs':{},'A1':{},'A2':{},'B':{},'residual':{}}
        r['refs']['frozen']=mw(W0,Xv,vl); r['refs']['oracle']=mw(Ws,Xv,vl); r['refs']['oracle_spec']=mw(W_or_spec,Xv,vl)
        r['residual']['w_cos']=cos_sim(W0,Ws); r['residual']['rel_norm']=float(R.norm()/(Ws.norm()+1e-30))
        s=torch.linalg.svdvals(R.double()); s2=s**2; tot=s2.sum().item()
        r['residual']['eff_rank']=float(s2.sum()**2/(s2**2).sum().item()) if tot>0 else 0
        tot_clean=len(la[ci]); clean_freq={c:int((la[ci]==c).sum().item())/tot_clean for c in classes}
        for k in [8,16,32,64,128]:
            lab_idx=[]
            for c in classes:
                idx=cls_idx[c]
                if len(idx)<max(50,k): continue
                torch.manual_seed(2); lab_idx.append(idx[torch.randperm(len(idx))[:k]])
            if not lab_idx: continue
            lab_idx=torch.cat(lab_idx); X_lab=Xp[lab_idx]; Y_lab=onehot(pl[lab_idx],NUM_CLASSES)
            T_hat=torch.zeros(10000,NUM_CLASSES)
            for c in classes:
                idx=cls_idx[c]
                if len(idx)<max(50,k): continue
                torch.manual_seed(2); T_hat[:,c]=len(idx)*Xp[idx[torch.randperm(len(idx))[:k]]].mean(0)
            T_hat=T_hat/N
            t_cos=cos_sim(T_hat, T_or)
            W_full=ridge_fit_soft(X_lab, Y_lab, args.lam, args.cg_iters, args.nystrom_m, device)
            w_cos_full=cos_sim(W_full, Ws)
            U8=U_full[:,:8]
            C=lsq_residual(X_lab, Y_lab, W0, U8, device) if U8.shape[1]>0 else torch.zeros(0,NUM_CLASSES)
            W_res=W0.detach().cpu()+(U8.cpu()@C) if U8.shape[1]>0 else W0.detach().cpu()
            r['A1'][str(k)]={'n_labels':len(lab_idx),'t_cos':t_cos,'w_cos_full':w_cos_full,
                             'delta_full':mw(W_full,Xv,vl)-r['refs']['frozen'],
                             'delta_res64':mw(W_res,Xv,vl)-r['refs']['frozen']}
        C_active=len([c for c in classes if len(cls_idx[c])>=50])
        B_total=8*C_active
        freq_pool={c:len(cls_idx[c]) for c in classes}
        for alpha in [0,0.25,0.5,1.0]:
            alloc=allocate_counts(B_total, freq_pool, alpha)
            lab_idx=[]
            for c in classes:
                kc=alloc[c]
                idx=cls_idx[c]
                if len(idx)<1: continue
                kc=min(kc, len(idx))
                torch.manual_seed(2); lab_idx.append(idx[torch.randperm(len(idx))[:kc]])
            lab_idx=torch.cat(lab_idx)
            X_lab=Xp[lab_idx]; Y_lab=onehot(pl[lab_idx], NUM_CLASSES)
            W_full=ridge_fit_soft(X_lab, Y_lab, args.lam, args.cg_iters, args.nystrom_m, device)
            r['A2'][f"alpha_{alpha}"]={'n_labels':len(lab_idx),'delta_full':mw(W_full,Xv,vl)-r['refs']['frozen'],
                                       'alloc':{str(c):alloc[c] for c in classes}}
        for rnk in [4,8]:
            for k in [8,32]:
                lab_idx=[]
                for c in classes:
                    idx=cls_idx[c]
                    if len(idx)<max(50,k): continue
                    torch.manual_seed(2); lab_idx.append(idx[torch.randperm(len(idx))[:k]])
                if not lab_idx: continue
                lab_idx=torch.cat(lab_idx); X_lab=Xp[lab_idx]; Y_lab=onehot(pl[lab_idx], NUM_CLASSES)
                Uo=U_full[:,:rnk]
                Co=lsq_residual(X_lab, Y_lab, W0, Uo, device)
                Wo=W0.detach().cpu()+(Uo.cpu()@Co)
                Up=U_S[:,-rnk:].detach().cpu()
                Cp=lsq_residual(X_lab, Y_lab, W0, Up, device)
                Wp=W0.detach().cpu()+(Up@Cp)
                key=f"r{rnk}_k{k}"
                r['B'][key]={'oracle_delta':mw(Wo,Xv,vl)-r['refs']['frozen'],
                             'pool_delta':mw(Wp,Xv,vl)-r['refs']['frozen']}
        results['conds'][cond]=r
        del Xc,Xp,Xv,W0,Ws,R,U_full,S,U_S
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        print(f"\n=== {cond} ({toc(t0):.0f}s) ===")
        print(f"  frozen {r['refs']['frozen']:.3f} / oracle {r['refs']['oracle']:.3f} gap {r['refs']['oracle']-r['refs']['frozen']:+.3f} eff-rank {r['residual']['eff_rank']:.1f}")
        print(f"  A1 budget curve (delta_full): " + " ".join(f"k{k}:{r['A1'][k]['delta_full']:+.3f}" for k in sorted(r['A1'],key=int)))
        print(f"  A2 adaptive (delta_full): " + " ".join(f"a{a}:{r['A2'][a]['delta_full']:+.3f}" for a in sorted(r['A2'])))
        print(f"  B pool vs oracle (k=8): " + " ".join(f"{k}:{r['B'][k]['pool_delta']:+.3f}/{r['B'][k]['oracle_delta']:+.3f}" for k in sorted(r['B']) if 'k8' in k))
    os.makedirs(os.path.dirname(args.out),exist_ok=True)
    with open(args.out,'w') as fh: json.dump(results,fh,indent=2,default=float)
    print(f"\nSaved to {args.out}")
    print("\nFallback: if A1 flat-negative to k=32, the knee is beyond 32 and the per-k t_cos/w_cos +")
    print("per-class coverage in A1 reveal what the added labels bought (mean quality vs mass).")

if __name__=="__main__": main()
