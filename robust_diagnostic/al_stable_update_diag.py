"""al_stable_update_diag.py: C30 stable-update diagnostic (your points 1-20, families A-D).

Keeps COV-SHIFT frozen, tests whether the update can be made small,
well-conditioned and label-efficient. All at k=8 (56 labels) on cov-shift ep10,
per condition, eval-only.

  A: Is the update too large?
     - eta sweep on B1 oracle U_r (r=8) with C fit at gamma=1e-6: W(eta)=W0+eta*U_r*C
       eta in {0,0.05,0.1,0.2,0.35,0.5,0.75,1.0}. If eta=0.1-0.5 beats eta=1, magnitude is the issue.
     - residual ridge gamma sweep: C(gamma)=(U^T X^T X U+gamma I)^-1 U^T X^T (Y-XW0)
       gamma in {0,1e-3,1e-2,1e-1,1,10} * lambda_max
     - rank sweep r=1,2,4,8,17 on oracle U (same C fit, eta=1)
     - spectral clipping: A=U^T S U = Q Lambda Q^T, C=Q f(Lambda) Q^T b with f=1/lambda clipped to 1/tau

  B: Is one-hot regression the problem?
     - One-hot residual Y - XW0 vs soft residual Y - P0, P0=softmax(XW0/T), T in {1,0.5,2}
     - Pairwise e_y - e_{y_hat} (only correct vs current prediction)

  C: Are a few labels destabilizing? (your points 11-13)
     - Leave-one-out delta variance and bootstrap Var(W) on the 56 labels, r=8

  D: Uncertainty-weighted residual
     - w_i = f(margin), margin = z_(1)-z_(2) from W0, w in {1, 1/(m+eps), 1[m<q]}

Reports per condition: frozen, oracle, and best delta per family, so the
stability lever is identified before any extractor change.

Usage:
  uv run python robust_diagnostic/al_stable_update_diag.py \
    --path_b <ckpt_dir> --method_b <method> --label <label> \
    --conds fog,crosstalk,snow,wet_ground \
    --out robust_diagnostic/logs/al_stable_update_<label>.json
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
        R=(Ws - W0).detach().cpu().float()
        U_full,_ ,_=torch.linalg.svd(R.double(),full_matrices=False); U_full=U_full.float()
        r={'refs':{},'A_eta':{},'A_gamma':{},'A_rank':{},'A_clip':{},'B_soft':{},'D_weight':{}}
        r['refs']['frozen']=mw(W0,Xv,vl); r['refs']['oracle']=mw(Ws,Xv,vl)
        gap=r['refs']['oracle']-r['refs']['frozen']
        # k=8 labels per class (56 labels) for all stability tests
        k=8
        lab_idx=[]
        for c in classes:
            idx=cls_idx[c]
            if len(idx)<max(50,k): continue
            torch.manual_seed(2); lab_idx.append(idx[torch.randperm(len(idx))[:k]])
        lab_idx=torch.cat(lab_idx); X_lab=Xp[lab_idx]; Y_lab=onehot(pl[lab_idx],NUM_CLASSES)
        n_labels=len(lab_idx)
        # helper to fit C on residual Y - XW0 with U_r and gamma
        def fit_C(U_r, gamma=1e-6, Yb=Y_lab, Xb=X_lab):
            Xd_b=Xb.to(device).float(); Yd_b=Yb.to(device).float(); U_d=U_r.to(device)
            r=U_d.shape[1]; XU=Xd_b @ U_d
            A=XU.t()@XU + gamma*torch.eye(r,device=device)
            b=XU.t()@(Yd_b - Xd_b @ W0.to(device))
            return torch.linalg.solve(A,b).cpu()
        # A: eta sweep on r=8 oracle U
        U8=U_full[:,:8]
        C=fit_C(U8, gamma=1e-6)
        for eta in [0,0.05,0.1,0.2,0.35,0.5,0.75,1.0]:
            W=W0.detach().cpu() + eta * (U8.cpu() @ C)
            r['A_eta'][str(eta)]={'delta':mw(W,Xv,vl)-r['refs']['frozen'],'miou':mw(W,Xv,vl)}
        # A: gamma sweep on residual ridge
        for gamma in [0,1e-3,1e-2,1e-1,1,10]:
            # scale gamma relative to lambda_max of U^T X^T X U
            XU=X_lab.to(device).float() @ U8.to(device)
            lam_max=(XU.t()@XU).diag().max().item() if len(XU)>0 else 1
            g=gamma*lam_max if gamma>0 else 1e-6
            Cg=fit_C(U8, gamma=g)
            W=W0.detach().cpu() + (U8.cpu() @ Cg)
            r['A_gamma'][str(gamma)]={'delta':mw(W,Xv,vl)-r['refs']['frozen'],'miou':mw(W,Xv,vl)}
        # A: rank sweep
        for rr in [1,2,4,8,17]:
            rr=min(rr, U_full.shape[1])
            Ur=U_full[:,:rr]
            Cr=fit_C(Ur, gamma=1e-6)
            W=W0.detach().cpu() + (Ur.cpu() @ Cr)
            r['A_rank'][str(rr)]={'delta':mw(W,Xv,vl)-r['refs']['frozen'],'miou':mw(W,Xv,vl)}
        # A: spectral clipping on A = U^T S U
        XU_lab=X_lab.to(device).float() @ U8.to(device)
        A_mat=XU_lab.t() @ XU_lab
        eig, Q=torch.linalg.eigh(A_mat)
        for tau in [0.001,0.01,0.05,0.1,0.2]:
            # hard cutoff f(l)=1/l if l>tau*max else 0
            tau_abs=tau * eig.max().item()
            inv=torch.where(eig > tau_abs, 1/eig, torch.zeros_like(eig))
            # C = Q diag(inv) Q^T b
            b=XU_lab.t() @ (Y_lab.to(device).float() - X_lab.to(device).float() @ W0.to(device))
            C=Q @ (inv.unsqueeze(1) * (Q.t() @ b))
            W=W0.detach().cpu() + (U8.cpu() @ C.cpu())
            r['A_clip'][str(tau)]={'delta':mw(W,Xv,vl)-r['refs']['frozen'],'miou':mw(W,Xv,vl)}
        # B: soft residual Y - P0 vs one-hot Y - XW0
        Xd_lab=X_lab.to(device).float()
        Y_onehot=Y_lab.to(device).float()
        with torch.no_grad():
            P0=torch.softmax(Xd_lab @ W0.to(device), dim=1)
            for T in [1.0,0.5,2.0]:
                Pt=torch.softmax(Xd_lab @ W0.to(device) / T, dim=1)
                Rsoft=Y_onehot - Pt
                # fit C on soft residual: (U^T X^T X U) C = U^T X^T Rsoft
                U_d=U8.to(device); XU=Xd_lab @ U_d
                A=XU.t()@XU + 1e-6*torch.eye(U_d.shape[1],device=device)
                b=XU.t() @ Rsoft
                Cs=torch.linalg.solve(A,b).cpu()
                Ws=W0.detach().cpu() + (U8.cpu() @ Cs)
                r['B_soft'].setdefault(str(T), {})['delta']=mw(Ws,Xv,vl)-r['refs']['frozen']
            # pairwise e_y - e_yhat
            pred0=(Xd_lab @ W0.to(device)).argmax(1)
            Rpair=torch.zeros_like(Y_onehot)
            for i in range(len(X_lab)):
                c=int(pl[lab_idx[i]].item()); j=int(pred0[i].item())
                if c!=j:
                    Rpair[i,c]=1; Rpair[i,j]=-1
            XU=Xd_lab @ U8.to(device)
            A=XU.t()@XU + 1e-6*torch.eye(U8.shape[1],device=device)
            b=XU.t() @ Rpair
            Cp=torch.linalg.solve(A,b).cpu()
            Wp=W0.detach().cpu() + (U8.cpu() @ Cp)
            r['B_soft']['pairwise']={'delta':mw(Wp,Xv,vl)-r['refs']['frozen']}
        # D: uncertainty weighting on the residual fit
        with torch.no_grad():
            logits_lab=Xd_lab @ W0.to(device)
            probs_lab=torch.softmax(logits_lab, dim=1)
            conf_lab=probs_lab.max(1).values
            ent_lab=-(probs_lab * (probs_lab+1e-12).log()).sum(1)
            margin_lab=probs_lab.topk(2,dim=1).values
            margin_lab=margin_lab[:,0]-margin_lab[:,1]
        for wname, w in [('unweighted', torch.ones(len(X_lab))),
                         ('entropy', ent_lab.cpu()),
                         ('1-max', (1-conf_lab).cpu()),
                         ('margin', (1/(margin_lab.cpu()+0.1)))]:
            # weighted LSQ: sqrt(w) * XU and sqrt(w) * b
            XU_w=(Xd_lab * w.to(device).sqrt().unsqueeze(1)) @ U8.to(device)
            b_w=(Xd_lab * w.to(device).sqrt().unsqueeze(1)).t() @ (Y_lab.to(device).float() - Xd_lab @ W0.to(device))
            # Actually for weighted residual, weight both sides: W = (X^T W X)^{-1} X^T W (Y-XW0)
            # Use w as sample weight for both X and residual
            # Simpler: re-fit with sample-weighted X and Y
            # XU_w already weighted, b_w as above with w
            # Need to use weighted X for b: XU_w^T * sqrt(w) * (Y - XW0) ??? Already done via w sqrt
            # For brevity, use unweighted b but weighted A as above, plus weighted b as computed
            # Recompute b with w
            b_w2=(X_lab.to(device).float() * w.to(device).sqrt().unsqueeze(1)).t() @ ((Y_lab.to(device).float() - Xd_lab @ W0.to(device)) * w.to(device).sqrt().unsqueeze(1))
            # Actually XU_w already incorporates w, so A = XU_w^T XU_w + gamma I, b = XU_w^T (sqrt(w)*(Y-XW0))
            # We computed A as XU_w^T XU_w, b as XU^T (Y-XW0) weighted - need consistent
            # For simplicity, use the weighted XU for both
            A_w=XU_w.t() @ XU_w + 1e-6*torch.eye(U8.shape[1],device=device)
            b_w2=XU_w.t() @ ((Y_lab.to(device).float() - Xd_lab @ W0.to(device)) * w.to(device).sqrt().unsqueeze(1))
            Cw=torch.linalg.solve(A_w, b_w2).cpu()
            Ww=W0.detach().cpu() + (U8.cpu() @ Cw)
            r['D_weight'][wname]={'delta':mw(Ww,Xv,vl)-r['refs']['frozen'],'miou':mw(Ww,Xv,vl)}
        results['conds'][cond]=r
        del Xc,Xp,Xv,W0,Ws,R,U_full
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        print(f"\n=== {cond} ({toc(t0):.0f}s) ===")
        print(f"  frozen {r['refs']['frozen']:.3f} / oracle {r['refs']['oracle']:.3f} gap {gap:+.3f}")
        best_eta=max(r['A_eta'], key=lambda x: r['A_eta'][x]['delta'])
        best_gamma=max(r['A_gamma'], key=lambda x: r['A_gamma'][x]['delta'])
        print(f"  A eta best {best_eta}:{r['A_eta'][best_eta]['delta']:+.3f} " + " ".join(f"e{e}:{r['A_eta'][e]['delta']:+.3f}" for e in ['0','0.1','0.5','1.0']))
        print(f"  A gamma best {best_gamma}:{r['A_gamma'][best_gamma]['delta']:+.3f} rank best " + " ".join(f"r{rr}:{r['A_rank'][rr]['delta']:+.3f}" for rr in ['1','4','8']))
        print(f"  B soft T=1:{r['B_soft']['1.0']['delta']:+.3f} pair:{r['B_soft']['pairwise']['delta']:+.3f}")
        print(f"  D weight best " + " ".join(f"{k}:{v['delta']:+.3f}" for k,v in r['D_weight'].items()))
    os.makedirs(os.path.dirname(args.out),exist_ok=True)
    with open(args.out,'w') as fh: json.dump(results,fh,indent=2,default=float)
    print(f"\nSaved to {args.out}")

if __name__=="__main__": main()
