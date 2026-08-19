"""al_b_feature_diag.py: B decoder variants + feature-space potential (extractor frozen).

Covers your point: keep the COV-SHIFT feature extractor frozen (last resort to
change it) and test (1) whether the decoder can be fixed and (2) whether the
feature space itself has enough form/function for SOME classifier, even if not
the current linear probe.

All eval-only on COV-SHIFT ep10 (anchor), 4 conditions:

  B variants (all at k=8,16,32 per class, labels fit only C):
    B1  W = W_cov + Delta, Delta fit from labels on residual Y - XW0 (full rank)
    B2a low-rank oracle U_r = SVD(R), R=W*-W0 (ceiling, r=4,8)
    B2b low-rank pool-cov U_r = top-r eigenvectors of S = Xp^T Xp / N (deployable)
    B2c code-shift U_r = SVD(M), M = per-class code-mean shift (deployable)
    Compare to full-probe on same budget (Iterations-7/8 baseline).

  Feature-space potential (no labels beyond clean, does SOME classifier work?):
    - Linear probe on RAW 128-d vs HDC 10k (does HDC help or hurt?)
    - 1-NN and 5-NN on RAW vs HDC (non-linear ceiling)
    - Prototype (nearest class mean) on RAW vs HDC
    - Intra/inter cosine, kappa, prank already in C20, re-reported for context
    - Per-class frozen error vs frequency (where budget should go)

If B2a (oracle U) is positive but B2b/B2c are not, the premise (low-rank residual)
holds but U must come from elsewhere (not pool covariance). If raw k-NN >> HDC
linear, the space has form for a different classifier family.

Usage:
  uv run python robust_diagnostic/al_b_feature_diag.py \
    --path_b <ckpt_dir> --method_b <method> --label <label> \
    --conds fog,crosstalk,snow,wet_ground \
    --out robust_diagnostic/logs/al_b_feature_<label>.json
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

def knn_accuracy(val_feats, val_labels, pool_feats, pool_labels, k=1, device='cuda', chunk=4096):
    # normalized cosine k-NN, chunked
    val_n=F.normalize(val_feats.float(),dim=1); pool_n=F.normalize(pool_feats.float(),dim=1)
    pool_n_d=pool_n.to(device)
    correct=0; total=len(val_labels)
    for s in range(0,len(val_n),chunk):
        e=min(s+chunk,len(val_n))
        sim=val_n[s:e].to(device) @ pool_n_d.t()  # chunk x pool
        if k==1:
            pred=pool_labels[sim.argmax(1).cpu()]
            correct+=int((pred==val_labels[s:e]).sum().item())
        else:
            # k-NN majority vote
            topk=sim.topk(k,dim=1).indices.cpu()
            for i,idx in enumerate(topk):
                votes=pool_labels[idx]
                # majority
                vals,counts=np.unique(votes.numpy(), return_counts=True)
                pred=vals[counts.argmax()]
                if pred==val_labels[s+i].item(): correct+=1
    return correct/total

def prototype_accuracy(val_feats, val_labels, pool_feats, pool_labels):
    # nearest class mean (raw space, normalized)
    classes=sorted(set(pool_labels.tolist()) & set(range(1,NUM_CLASSES)))
    zn_pool=F.normalize(pool_feats.float(),dim=1)
    means={c: F.normalize(zn_pool[pool_labels==c].mean(0).unsqueeze(0),dim=1)[0] for c in classes if (pool_labels==c).sum()>0}
    if not means: return 0
    cs=sorted(means); protos=torch.stack([means[c] for c in cs])
    protos_n=F.normalize(protos,dim=1)
    val_n=F.normalize(val_feats.float(),dim=1)
    # map val to nearest prototype
    # chunked
    correct=0
    for s in range(0,len(val_n),4096):
        e=min(s+4096,len(val_n))
        sim=val_n[s:e] @ protos_n.t()
        pred_idx=sim.argmax(1)
        pred=torch.tensor([cs[i] for i in pred_idx.tolist()])
        correct+=int((pred==val_labels[s:e]).sum().item())
    return correct/len(val_labels)

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
        # spectral exact for R
        S=(Xd.double().t()@Xd.double())/N; eigS,U=torch.linalg.eigh(S); eigS=eigS.float(); U=U.float()
        R=(Ws - W0).detach().cpu().float()
        U_full,_ ,_=torch.linalg.svd(R.double(),full_matrices=False); U_full=U_full.float()
        # unlabeled bases
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
        r={'refs':{},'B':{},'feat':{}}
        r['refs']['frozen']=mw(W0,Xv,vl); r['refs']['oracle']=mw(Ws,Xv,vl)
        # ---- Feature-space potential (no labels beyond clean) ----
        # raw linear probe (128-d)
        W_raw=ridge_fit_soft(fa[ci],onehot(la[ci],NUM_CLASSES),args.lam,args.cg_iters,args.nystrom_m,device)
        # we need raw pool/val for raw decode: use fa/l already have raw 128-d pool/val
        # compute raw predictions on val via raw linear probe: val raw @ W_raw
        # chunked decode for raw
        # use same decode helper but with raw codes = raw features (128-d) directly
        # So we treat Xv_raw = val (128-d) and W_raw is 128x17
        def mw_raw(W_raw, val_raw, vl):
            # val_raw is 128-d, W_raw is 128x17
            preds=[]
            for s in range(0,len(val_raw),100000):
                preds.append((val_raw[s:s+100000].float() @ W_raw.detach().cpu()).argmax(1))
            return compute_miou(torch.cat(preds), vl)
        r['feat']['raw_frozen']=mw_raw(W_raw, val, vl)
        # raw oracle (fit on pool raw)
        W_raw_or=ridge_fit_soft(pool,onehot(pl,NUM_CLASSES),args.lam,args.cg_iters,args.nystrom_m,device)
        r['feat']['raw_oracle']=mw_raw(W_raw_or, val, vl)
        # k-NN on raw and HDC
        r['feat']['knn_raw_1']=knn_accuracy(val, vl, pool, pl, k=1, device=device)
        r['feat']['knn_raw_5']=knn_accuracy(val, vl, pool, pl, k=5, device=device)
        r['feat']['knn_hdc_1']=knn_accuracy(Xv, vl, Xp, pl, k=1, device=device)
        # prototype on raw vs HDC (use helper)
        r['feat']['proto_raw']=prototype_accuracy(val, vl, pool, pl)
        r['feat']['proto_hdc']=prototype_accuracy(Xv, vl, Xp, pl)
        # intra/inter already in C20, skip duplicate here, just report gap
        r['feat']['gap_raw']=r['feat']['raw_oracle']-r['feat']['raw_frozen']
        r['feat']['gap_hdc']=r['refs']['oracle']-r['refs']['frozen']
        # ---- B decoder variants: k=8, r=4 vs r=8 ----
        for k in [8,32]:
            lab_idx=[]
            for c in classes:
                idx=cls_idx[c]
                if len(idx)<max(50,k): continue
                torch.manual_seed(2); lab_idx.append(idx[torch.randperm(len(idx))[:k]])
            if not lab_idx: continue
            lab_idx=torch.cat(lab_idx); X_lab=Xp[lab_idx]; Y_lab=onehot(pl[lab_idx],NUM_CLASSES)
            n_labels=len(lab_idx)
            # B1 full Delta
            # Delta fit: C has no U, just full X^T X solve on residual: same as full-probe residual
            # For B1, U is identity (full rank) so we just fit C = (X^T X)^-1 X^T (Y - XW0)
            # That's equivalent to W_sub - W0 where W_sub is ridge on lab, but with lsq_residual and U=I (10k)
            # Instead, directly fit residual with ridge: use lsq_residual with U = I (10k)
            # But that's just W_sub - W0. Simpler: B1 = full residual with r=17 (full U_full)
            Uo_full=U_full  # 10k x 17
            Co=lsq_residual(X_lab, Y_lab, W0, Uo_full, device)
            Wo=W0.detach().cpu()+(Uo_full.cpu()@Co)
            b1_oracle=mw(Wo,Xv,vl)-r['refs']['frozen']
            # pool-cov basis
            Up4=U_S[:,-4:].detach().cpu(); Cp4=lsq_residual(X_lab, Y_lab, W0, Up4, device); Wp4=W0.detach().cpu()+(Up4@Cp4)
            Up8=U_S[:,-8:].detach().cpu(); Cp8=lsq_residual(X_lab, Y_lab, W0, Up8, device); Wp8=W0.detach().cpu()+(Up8@Cp8)
            # code-shift basis
            if U_shift is not None:
                Us4=U_shift[:,:min(4,U_shift.shape[1])]; Cs4=lsq_residual(X_lab, Y_lab, W0, Us4, device); Ws4=W0.detach().cpu()+(Us4.cpu()@Cs4)
                Us8=U_shift[:,:min(8,U_shift.shape[1])]; Cs8=lsq_residual(X_lab, Y_lab, W0, Us8, device); Ws8=W0.detach().cpu()+(Us8.cpu()@Cs8)
                cs4=mw(Ws4,Xv,vl)-r['refs']['frozen']; cs8=mw(Ws8,Xv,vl)-r['refs']['frozen']
            else:
                cs4=cs8=0
            # full probe
            W_full=ridge_fit_soft(X_lab, Y_lab, args.lam, args.cg_iters, args.nystrom_m, device)
            key=f"k{k}"
            r['B'][key]={'n_labels':n_labels,
                         'B1_oracle_r17':b1_oracle,
                         'B2_pool_r4':mw(Wp4,Xv,vl)-r['refs']['frozen'],
                         'B2_pool_r8':mw(Wp8,Xv,vl)-r['refs']['frozen'],
                         'B2_shift_r4':cs4, 'B2_shift_r8':cs8,
                         'full_probe':mw(W_full,Xv,vl)-r['refs']['frozen']}
        results['conds'][cond]=r
        del Xc,Xp,Xv,W0,Ws,R,U_full,S,U_S
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        print(f"\n=== {cond} ({toc(t0):.0f}s) ===")
        print(f"  frozen {r['refs']['frozen']:.3f} / oracle {r['refs']['oracle']:.3f} gap {r['refs']['oracle']-r['refs']['frozen']:+.3f}")
        print(f"  feat: raw_frozen {r['feat']['raw_frozen']:.3f} raw_oracle {r['feat']['raw_oracle']:.3f} (gap {r['feat']['gap_raw']:+.3f}) | HDC gap {r['feat']['gap_hdc']:+.3f}")
        print(f"  kNN raw 1:{r['feat']['knn_raw_1']:.3f} 5:{r['feat']['knn_raw_5']:.3f} | HDC 1:{r['feat']['knn_hdc_1']:.3f} | proto raw {r['feat']['proto_raw']:.3f} HDC {r['feat']['proto_hdc']:.3f}")
        for k in ['8','32']:
            if k not in r['B']: continue
            b=r['B'][k]
            print(f"  k={k} ({b['n_labels']} lbl): B1_oracle {b['B1_oracle_r17']:+.3f} | pool r4 {b['B2_pool_r4']:+.3f} r8 {b['B2_pool_r8']:+.3f} | full {b['full_probe']:+.3f}")
    os.makedirs(os.path.dirname(args.out),exist_ok=True)
    with open(args.out,'w') as fh: json.dump(results,fh,indent=2,default=float)
    print(f"\nSaved to {args.out}")
    print("\n=== READ ===")
    print("B1 oracle r17 is the low-rank ceiling; if >0 on wet/fog, the residual IS estimable.")
    print("Feature kNN/proto vs linear: does SOME classifier work on this space? If kNN >>")
    print("linear, the space has form for a different rule (your point 8-11). If raw >> HDC,")
    print("the HDC projection is the bottleneck, not the features.")

if __name__=="__main__": main()
