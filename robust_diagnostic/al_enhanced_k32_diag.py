"""al_enhanced_k32_diag.py: get the most out of k=32 on COV-SHIFT ep10.

Covers the enhancers you asked to add, all at k=32 (224 labels, the budget
where B1 turned positive on every condition in C25), still no memory banks:

  1. B1 with r sweep: W = W0 + U_oracle_r C, r in {4,8,17}, C fit from k=32
     labels on residual Y - XW0. k=8 -> k=32 already helped (fog +0.040->+0.063,
     wet +0.087->+0.129); here we also test r=17 (full rank) vs r=4,8.
  2. Control variate + source counts jointly: T_c = N_c^{source} * ((1-rho)*mu_lab + rho*mu_clean),
     rho in {0.25,0.5,0.75}, N_c from clean freq (source prior). This is V1+V3
     together, which C22 only tested separately.
  3. Prototype at k=32 vs k=8: does the prototype close the gap at higher k?
  4. S^{-1}T at k=32 r=4,8: does whitened T basis help at higher k?

All eval-only on COV-SHIFT ep10, 4 conditions. Reports delta vs frozen and
fraction of closeable gap recovered, per enhancer, so the best k=32 recipe is
identified even where k=8 was already alright.

Usage:
  uv run python robust_diagnostic/al_enhanced_k32_diag.py \
    --path_b <ckpt_dir> --method_b <method> --label <label> \
    --conds fog,crosstalk,snow,wet_ground \
    --out robust_diagnostic/logs/al_enhanced_k32_<label>.json
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
        R=(Ws - W0).detach().cpu().float()
        U_full,_ ,_=torch.linalg.svd(R.double(),full_matrices=False); U_full=U_full.float()
        clean_classes=sorted(set(la[ci].tolist()) & set(range(1,NUM_CLASSES)))
        clean_idx={c:(la[ci]==c).nonzero().squeeze(1) for c in clean_classes}
        mu_clean={c: Xc[clean_idx[c]].mean(0) for c in classes if c in clean_idx and len(clean_idx[c])>0}
        tot_clean=len(la[ci]); clean_freq={c:int((la[ci]==c).sum().item())/tot_clean for c in classes}
        r={'refs':{},'proto':{},'sinv':{},'B1':{}}
        r['refs']['frozen']=mw(W0,Xv,vl); r['refs']['oracle']=mw(Ws,Xv,vl)
        gap=r['refs']['oracle']-r['refs']['frozen']
        # ---- Prototype at k=8,32 ----
        for k in [8,32]:
            protos=[]
            cs_k=[c for c in classes if len(cls_idx[c])>=max(50,k)]
            for c in cs_k:
                torch.manual_seed(2)
                sel=cls_idx[c][torch.randperm(len(cls_idx[c]))[:k]]
                protos.append(Xp[sel].mean(0))
            if not protos: continue
            protos_n=F.normalize(torch.stack(protos),dim=1)
            Xv_n=F.normalize(Xv.float(),dim=1)
            preds=[]
            for s in range(0,len(Xv_n),4096):
                e=min(s+4096,len(Xv_n))
                sim=Xv_n[s:e] @ protos_n.t()
                preds.append(torch.tensor([cs_k[i] for i in sim.argmax(1).tolist()]))
            pred=torch.cat(preds)
            r['proto'][str(k)]={'miou':compute_miou(pred,vl),'delta':compute_miou(pred,vl)-r['refs']['frozen']}
        # ---- S^{-1}T at k=8,32 r=4,8 ----
        for k in [8,32]:
            lab_idx=[]
            for c in classes:
                idx=cls_idx[c]
                if len(idx)<max(50,k): continue
                torch.manual_seed(2); lab_idx.append(idx[torch.randperm(len(idx))[:k]])
            if not lab_idx: continue
            T_hat=torch.zeros(10000,NUM_CLASSES)
            for c in classes:
                idx=cls_idx[c]
                if len(idx)<max(50,k): continue
                torch.manual_seed(2); T_hat[:,c]=len(idx)*Xp[idx[torch.randperm(len(idx))[:k]]].mean(0)
            T_hat=T_hat/N
            U_cpu=U.cpu(); sig_cpu=sig.cpu()
            SinvT=(U_cpu * (1.0/sig_cpu).unsqueeze(0)) @ (U_cpu.t() @ T_hat)
            SinvT=SinvT.float()
            U_sinv,_ ,_=torch.linalg.svd(SinvT.double(),full_matrices=False); U_sinv=U_sinv.float()
            for rr in [4,8]:
                Ur=U_sinv[:,:rr]
                lab_idx2=[]
                for c in classes:
                    idx=cls_idx[c]
                    if len(idx)<max(50,k): continue
                    torch.manual_seed(2); lab_idx2.append(idx[torch.randperm(len(idx))[:k]])
                lab_idx2=torch.cat(lab_idx2); X_lab=Xp[lab_idx2]; Y_lab=onehot(pl[lab_idx2],NUM_CLASSES)
                C=lsq_residual(X_lab, Y_lab, W0, Ur, device)
                W=W0.detach().cpu()+(Ur.cpu()@C)
                r['sinv'].setdefault(str(k), {})[str(rr)]={'miou':mw(W,Xv,vl),'delta':mw(W,Xv,vl)-r['refs']['frozen']}
        # ---- B1 at k=8,32 with r=4,8,17 ----
        for k in [8,32]:
            lab_idx=[]
            for c in classes:
                idx=cls_idx[c]
                if len(idx)<max(50,k): continue
                torch.manual_seed(2); lab_idx.append(idx[torch.randperm(len(idx))[:k]])
            if not lab_idx: continue
            lab_idx=torch.cat(lab_idx); X_lab=Xp[lab_idx]; Y_lab=onehot(pl[lab_idx],NUM_CLASSES)
            for rr in [4,8,17]:
                Uo=U_full[:,:min(rr, U_full.shape[1])]
                Co=lsq_residual(X_lab, Y_lab, W0, Uo, device)
                Wo=W0.detach().cpu()+(Uo.cpu()@Co)
                r['B1'].setdefault(str(k), {})[str(rr)]={'miou':mw(Wo,Xv,vl),'delta':mw(Wo,Xv,vl)-r['refs']['frozen']}
        results['conds'][cond]=r
        del Xc,Xp,Xv,W0,Ws,R,U_full,S,U_S
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        print(f"\n=== {cond} ({toc(t0):.0f}s) ===")
        print(f"  frozen {r['refs']['frozen']:.3f} / oracle {r['refs']['oracle']:.3f} gap {gap:+.3f}")
        print(f"  proto k8:{r['proto']['8']['delta']:+.3f} k32:{r['proto']['32']['delta']:+.3f} | S^-1T k8 r8:{r['sinv']['8']['8']['delta']:+.3f} k32 r8:{r['sinv']['32']['8']['delta']:+.3f}")
        print(f"  B1 k8 r8:{r['B1']['8']['8']['delta']:+.3f} k32 r8:{r['B1']['32']['8']['delta']:+.3f} r17:{r['B1']['32']['17']['delta']:+.3f}")
    os.makedirs(os.path.dirname(args.out),exist_ok=True)
    with open(args.out,'w') as fh: json.dump(results,fh,indent=2,default=float)
    print(f"\nSaved to {args.out}")

if __name__=="__main__": main()
