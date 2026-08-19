"""al_lowbonus_bank_diag.py: why snow/crosstalk bonuses low, U without oracle, tiny memory bank.

Keeps COV-SHIFT extractor frozen (no extractor change, no large banks):

  Q1 Low bonuses: for snow/crosstalk vs fog/wet, what is missing? Tracks per
     condition at k=32: gap, t_cos, w_cos, whitened error, per-class delta,
     residual eff-rank and R_r curve. Distinguishes small gap vs poor T vs
     decoder amplification. Reports the fraction of gap recovered.

  Q2 U without oracle: can we get U without W*? Tests on same k=32 labels:
     U_T      = SVD(T_hat) where T_hat = N_c * mu_hat_c (k points, source counts)
     U_Rreg   = SVD(W_sub_reg - W0) with ridge lambda*10 on W_sub
     U_pool   = top eigenvectors of S (pool cov, already C22)
     U_shift  = SVD(M) code-mean shift (already C22)
     Each measured by align(U_hat, U_oracle) and B1 delta with that U.

  Q3 Tiny memory bank (no stream, just labelled + small unlabelled subset):
     Bank sizes: 56 (k=8 labels only), 556 (56+500 random unlabelled),
     1056 (56+1000), 5056 (56+5000). For each bank, 1-NN accuracy on val
     (cosine, chunked, no large bank). Does adding 500-1000 unlabelled points
     help, and is 500 enough vs 5000?

All eval-only on COV-SHIFT ep10, 4 conditions.

Usage:
  uv run python robust_diagnostic/al_lowbonus_bank_diag.py \
    --path_b <ckpt_dir> --method_b <method> --label <label> \
    --conds fog,crosstalk,snow,wet_ground \
    --out robust_diagnostic/logs/al_lowbonus_bank_<label>.json
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
    W=W.detach().cpu()
    p=[]
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
def knn_acc(val_feats, val_labels, bank_feats, bank_labels, k=1, device='cuda', chunk=4096):
    val_n=F.normalize(val_feats.float(),dim=1); bank_n=F.normalize(bank_feats.float(),dim=1)
    bank_d=bank_n.to(device)
    correct=0
    for s in range(0,len(val_n),chunk):
        e=min(s+chunk,len(val_n))
        sim=val_n[s:e].to(device) @ bank_d.t()
        if k==1:
            pred=bank_labels[sim.argmax(1).cpu()]
            correct+=int((pred==val_labels[s:e]).sum().item())
        else:
            topk=sim.topk(k,dim=1).indices.cpu()
            for i,idx in enumerate(topk):
                vals,counts=np.unique(bank_labels[idx].numpy(), return_counts=True)
                pred=vals[counts.argmax()]
                if pred==val_labels[s+i].item(): correct+=1
    return correct/len(val_labels)
def align_U(U_hat, U_oracle, r=8):
    # subspace alignment: r^{-1} || U_oracle^T U_hat ||_F^2
    rr=min(r, U_hat.shape[1], U_oracle.shape[1])
    Uh=U_hat[:,:rr]; Uo=U_oracle[:,:rr]
    return float((Uo.t() @ Uh).pow(2).sum() / rr)
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
        # spectral exact
        S=(Xd.double().t()@Xd.double())/N; eigS,U=torch.linalg.eigh(S); eigS=eigS.float(); U=U.float()
        lam_hat=args.lam/N; sig=(eigS+lam_hat).clamp(min=lam_hat)
        T_or=torch.zeros(10000,NUM_CLASSES)
        for c in classes: T_or[:,c]=Xp[cls_idx[c]].sum(0)
        m0=pl==0
        if int(m0.sum().item())>0: T_or[:,0]=Xp[m0].sum(0)
        T_or=T_or/N
        R=(Ws - W0).detach().cpu().float()
        U_full,_ ,_=torch.linalg.svd(R.double(),full_matrices=False); U_full=U_full.float()
        _, U_S=torch.linalg.eigh(S); U_S=U_S.float()
        # code-shift basis
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
        r={'refs':{},'Q1':{},'Q2':{},'Q3':{}}
        r['refs']['frozen']=mw(W0,Xv,vl); r['refs']['oracle']=mw(Ws,Xv,vl)
        gap=r['refs']['oracle']-r['refs']['frozen']
        # ---- Q1: low bonuses diagnosis (k=32, the budget where B1 is positive) ----
        for k in [32]:
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
            t_cos=np.nan; w_cos=np.nan
            try:
                t_cos=cos_sim(T_hat, T_or)
                W_tmp=ridge_fit_soft(X_lab, Y_lab, args.lam, args.cg_iters, args.nystrom_m, device)
                w_cos=cos_sim(W_tmp, Ws)
            except: pass
            # whitened error
            S_cpu=S.cpu().float() if 'S' in locals() else None
            r['Q1'][str(k)]={'t_cos':t_cos,'w_cos':w_cos,'gap':gap,'n_labels':len(lab_idx)}
            # per-class delta for B1 at this k
            Uo=U_full[:,:8]
            Co=lsq_residual(X_lab, Y_lab, W0, Uo, device)
            Wo=W0.detach().cpu()+(Uo.cpu()@Co)
            r['Q1'][str(k)]['B1_r8_delta']=mw(Wo,Xv,vl)-r['refs']['frozen']
        # ---- Q2: U without oracle ----
        for k in [32]:
            lab_idx=[]
            for c in classes:
                idx=cls_idx[c]
                if len(idx)<max(50,k): continue
                torch.manual_seed(2); lab_idx.append(idx[torch.randperm(len(idx))[:k]])
            if not lab_idx: continue
            lab_idx=torch.cat(lab_idx); X_lab=Xp[lab_idx]; Y_lab=onehot(pl[lab_idx],NUM_CLASSES)
            # U_T = SVD(T_hat) where T_hat = N_c * mu_hat_c
            T_hat2=torch.zeros(10000,NUM_CLASSES)
            for c in classes:
                idx=cls_idx[c]
                if len(idx)<max(50,k): continue
                torch.manual_seed(2); T_hat2[:,c]=len(idx)*Xp[idx[torch.randperm(len(idx))[:k]]].mean(0)
            T_hat2=T_hat2/N
            U_T,_ ,_=torch.linalg.svd(T_hat2.double(),full_matrices=False); U_T=U_T.float()
            # U_Rreg = SVD with ridge-regularized W_sub
            W_sub_reg=ridge_fit_soft(X_lab, Y_lab, args.lam*10, args.cg_iters, args.nystrom_m, device)
            R_reg=(W_sub_reg - W0).detach().cpu().float()
            U_reg,_ ,_=torch.linalg.svd(R_reg.double(),full_matrices=False); U_reg=U_reg.float()
            for name,U_hat in [('U_T',U_T),('U_Rreg',U_reg),('U_pool',U_S[:,-8:].detach().cpu()),('U_shift',U_shift[:,:8] if U_shift is not None else None)]:
                if U_hat is None: continue
                rr=min(8, U_hat.shape[1])
                Uh=U_hat[:,:rr]
                align=align_U(Uh, U_full[:,:rr])
                C=lsq_residual(X_lab, Y_lab, W0, Uh, device)
                W=W0.detach().cpu()+(Uh.cpu()@C)
                r['Q2'].setdefault(str(k), {})[name]={'align':align,'delta':mw(W,Xv,vl)-r['refs']['frozen'],'r':rr}
        # ---- Q3: tiny memory bank (labeled + small unlabelled subset) ----
        for k in [8,32]:
            lab_idx=[]
            for c in classes:
                idx=cls_idx[c]
                if len(idx)<max(50,k): continue
                torch.manual_seed(2); lab_idx.append(idx[torch.randperm(len(idx))[:k]])
            if not lab_idx: continue
            lab_idx=torch.cat(lab_idx)
            # bank sizes: labeled + 0,500,1000,5000 random unlabelled
            for add in [0,500,1000,5000]:
                if add==0:
                    bank_idx=lab_idx
                else:
                    # random unlabelled from pool excluding lab_idx
                    pool_all=torch.arange(len(pool))
                    mask=torch.ones(len(pool),dtype=torch.bool); mask[lab_idx]=False
                    avail=pool_all[mask]
                    torch.manual_seed(3)
                    extra=avail[torch.randperm(len(avail))[:min(add,len(avail))]]
                    bank_idx=torch.cat([lab_idx, extra])
                bank_feats=Xp[bank_idx]; bank_labels=pl[bank_idx]
                # 1-NN via bank
                acc=knn_acc(val, vl, bank_feats, bank_labels, k=1, device=device)
                r['Q3'].setdefault(str(k), {})[f"bank_{len(bank_idx)}"]={'acc':acc,'n_bank':len(bank_idx)}
        results['conds'][cond]=r
        del Xc,Xp,Xv,W0,Ws,R,U_full,S,U_S
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        print(f"\n=== {cond} ({toc(t0):.0f}s) ===")
        print(f"  frozen {r['refs']['frozen']:.3f} / oracle {r['refs']['oracle']:.3f} gap {gap:+.3f}")
        if '32' in r['Q1']:
            print(f"  Q1 k32: t_cos {r['Q1']['32']['t_cos']:.3f} w_cos {r['Q1']['32']['w_cos']:.3f} B1_r8 {r['Q1']['32']['B1_r8_delta']:+.3f}")
        if '32' in r['Q2']:
            print(f"  Q2 k32: " + " ".join(f"{n}:{v['delta']:+.3f}(a{v['align']:.2f})" for n,v in r['Q2']['32'].items()))
        if '8' in r['Q3']:
            print(f"  Q3 k8 bank: " + " ".join(f"{k}:{v['acc']:.3f}" for k,v in r['Q3']['8'].items()))
    os.makedirs(os.path.dirname(args.out),exist_ok=True)
    with open(args.out,'w') as fh: json.dump(results,fh,indent=2,default=float)
    print(f"\nSaved to {args.out}")

def align_U(U_hat, U_oracle, r=8):
    rr=min(r, U_hat.shape[1], U_oracle.shape[1])
    Uh=U_hat[:,:rr]; Uo=U_oracle[:,:rr]
    return float(((Uo.t() @ Uh).pow(2).sum() / rr).item())

if __name__=="__main__": main()
