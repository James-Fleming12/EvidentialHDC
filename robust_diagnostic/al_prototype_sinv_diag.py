"""al_prototype_sinv_diag.py: prototypes + S^{-1}T-derived U + B1 budget (no memory banks).

Keeps the COV-SHIFT extractor frozen (your constraint) and tests three
decoder-side levers that need no kNN memory bank:

  1. Prototype at k: P_c = mean of k code vectors per class (centroid rule,
     cosine to nearest prototype). k in [8,16,32,64,128] (56 to 896 labels).
     Does more labels make the cheap prototype viable on wet/fog?

  2. S^{-1}T-derived U: U_r = top-r SVD of (S + lI)^{-1} T_hat, where T_hat is
     N_c * mu_hat_c from k labels. C fit from labels on residual Y - XW0:
     W = W0 + U_r C. r=4,8, k=8,32. This is the WHITENED residual basis: the
     directions S^{-1} amplifies are exactly the low-variance directions from
     C22. If this U captures R, pool-cov's high-variance failure is explained.

  3. B1 budget for current linear separator: W = W0 + U_oracle_r * C with
     U_oracle = SVD(R) (ceiling) at k=8,16,32,64,128. How much does raising the
     budget help B1 in the linear paradigm you asked about?

All eval-only on COV-SHIFT ep10.

Usage:
  uv run python robust_diagnostic/al_prototype_sinv_diag.py \
    --path_b <ckpt_dir> --method_b <method> --label <label> \
    --conds fog,crosstalk,snow,wet_ground \
    --out robust_diagnostic/logs/al_prototype_sinv_<label>.json
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
def proto_miou(protos, Xv, vl):
    # protos: C_active x D (already normalized), Xv: N x D
    # assign val point to nearest prototype (cosine)
    if len(protos)==0: return 0
    protos_n=F.normalize(protos,dim=1)
    Xv_n=F.normalize(Xv.float(),dim=1)
    preds=[]
    for s in range(0,len(Xv_n),4096):
        e=min(s+4096,len(Xv_n))
        sim=Xv_n[s:e] @ protos_n.t()
        # need to map prototype index to class label
        preds.append(sim.argmax(1))
    # preds are indices into protos list, need to map to actual class ids
    return None  # handled per-call with class map
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
        # spectral exact for S and oracle
        S=(Xd.double().t()@Xd.double())/N; eigS,U=torch.linalg.eigh(S); eigS=eigS.float(); U=U.float()
        lam_hat=args.lam/N; sig=(eigS+lam_hat).clamp(min=lam_hat)
        R=(Ws - W0).detach().cpu().float()
        U_full,_ ,_=torch.linalg.svd(R.double(),full_matrices=False); U_full=U_full.float()
        S2=(Xd.double().t()@Xd.double())/N; _, U_S=torch.linalg.eigh(S2); U_S=U_S.float()
        r={'refs':{},'proto':{},'sinv':{},'B1':{}}
        r['refs']['frozen']=mw(W0,Xv,vl); r['refs']['oracle']=mw(Ws,Xv,vl)
        # ---- 1. Prototype at k sweep ----
        for k in [8,16,32,64,128]:
            lab_idx=[]
            for c in classes:
                idx=cls_idx[c]
                if len(idx)<max(50,k): continue
                torch.manual_seed(2); lab_idx.append(idx[torch.randperm(len(idx))[:k]])
            if not lab_idx: continue
            # prototype = mean of k code vectors per class
            protos=[]
            for c in classes:
                idx=cls_idx[c]
                if len(idx)<max(50,k): continue
                torch.manual_seed(2)
                sel=idx[torch.randperm(len(idx))[:k]]
                protos.append(Xp[sel].mean(0))
            if not protos: continue
            protos_n=F.normalize(torch.stack(protos),dim=1)
            # map prototype index to class label for this k (only classes with >=k points)
            cs_k=[c for c in classes if len(cls_idx[c])>=max(50,k)]
            # val decode via prototype (cosine)
            Xv_n=F.normalize(Xv.float(),dim=1)
            preds=[]
            for s in range(0,len(Xv_n),4096):
                e=min(s+4096,len(Xv_n))
                sim=Xv_n[s:e] @ protos_n.t()
                preds.append(torch.tensor([cs_k[i] for i in sim.argmax(1).tolist()]))
            pred=torch.cat(preds)
            r['proto'][str(k)]={'miou':compute_miou(pred,vl),'n_labels':len(lab_idx)*1}
        # ---- 2. S^{-1}T-derived U ----
        for k in [8,32]:
            lab_idx=[]
            for c in classes:
                idx=cls_idx[c]
                if len(idx)<max(50,k): continue
                torch.manual_seed(2); lab_idx.append(idx[torch.randperm(len(idx))[:k]])
            if not lab_idx: continue
            lab_idx=torch.cat(lab_idx); X_lab=Xp[lab_idx]; Y_lab=onehot(pl[lab_idx],NUM_CLASSES)
            # T_hat with oracle counts (same as before)
            T_hat=torch.zeros(10000,NUM_CLASSES)
            for c in classes:
                idx=cls_idx[c]
                if len(idx)<max(50,k): continue
                torch.manual_seed(2); T_hat[:,c]=len(idx)*Xp[idx[torch.randperm(len(idx))[:k]]].mean(0)
            T_hat=T_hat/N
            # S^{-1}T_hat via spectral on CPU: S^{-1}T = U diag(1/sig) U^T T
            U_cpu = U.cpu(); sig_cpu = sig.cpu()
            SinvT = (U_cpu * (1.0/sig_cpu).unsqueeze(0)) @ (U_cpu.t() @ T_hat)
            SinvT = SinvT.float()
            U_sinv,_ ,_=torch.linalg.svd(SinvT.double(),full_matrices=False); U_sinv=U_sinv.float()
            for rr in [4,8]:
                U=U_sinv[:,:rr]
                C=lsq_residual(X_lab, Y_lab, W0, U, device)
                W=W0.detach().cpu() + (U.cpu() @ C)
                r['sinv'].setdefault(str(k), {})[str(rr)]={'miou':mw(W,Xv,vl),'delta':mw(W,Xv,vl)-r['refs']['frozen']}
        # ---- 3. B1 budget for current linear separator (oracle U, full rank) ----
        for k in [8,16,32,64,128]:
            lab_idx=[]
            for c in classes:
                idx=cls_idx[c]
                if len(idx)<max(50,k): continue
                torch.manual_seed(2); lab_idx.append(idx[torch.randperm(len(idx))[:k]])
            if not lab_idx: continue
            lab_idx=torch.cat(lab_idx); X_lab=Xp[lab_idx]; Y_lab=onehot(pl[lab_idx],NUM_CLASSES)
            Uo=U_full  # full 10k x 17, r=17
            Co=lsq_residual(X_lab, Y_lab, W0, Uo, device)
            Wo=W0.detach().cpu()+(Uo.cpu()@Co)
            r['B1'][str(k)]={'miou':mw(Wo,Xv,vl),'delta':mw(Wo,Xv,vl)-r['refs']['frozen'],'n_labels':len(lab_idx)}
        results['conds'][cond]=r
        del Xc,Xp,Xv,W0,Ws,R,U_full,S,U_S
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        print(f"\n=== {cond} ({toc(t0):.0f}s) ===")
        print(f"  frozen {r['refs']['frozen']:.3f} / oracle {r['refs']['oracle']:.3f} gap {r['refs']['oracle']-r['refs']['frozen']:+.3f}")
        print(f"  proto k: " + " ".join(f"k{k}:{r['proto'][k]['miou']:.3f}" for k in sorted(r['proto'],key=int)))
        print(f"  S^-1T k=8 r4:{r['sinv']['8']['4']['delta']:+.3f} r8:{r['sinv']['8']['8']['delta']:+.3f} | k=32 r4:{r['sinv']['32']['4']['delta']:+.3f}")
        print(f"  B1 k: " + " ".join(f"k{k}:{r['B1'][k]['delta']:+.3f}" for k in sorted(r['B1'],key=int)))
    os.makedirs(os.path.dirname(args.out),exist_ok=True)
    with open(args.out,'w') as fh: json.dump(results,fh,indent=2,default=float)
    print(f"\nSaved to {args.out}")
    print("\n=== READ ===")
    print("Proto: does k=8 prototype beat linear frozen? Does k=32/64 close the gap?")
    print("S^-1T: does whitened T basis beat pool-cov? Compare to C22 pool ~0.")
    print("B1: how much does raising budget help the current linear residual paradigm?")

if __name__=="__main__": main()
