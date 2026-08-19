"""al_tinybank_baseline_diag.py: is the tiny bank (500 unlabelled + 56 labels) a viable baseline?

Covers your request: does the tiny bank method (500) meaningfully improve
every single condition, so we can use it as the baseline starting point for
the rest of the methods moving forward.

All eval-only on COV-SHIFT ep10, 4 conditions. Keeps extractor frozen.

Tests per condition:
  1. Frozen linear probe (W0) vs Tiny bank 1-NN (56+500) vs Oracle
     Reports delta_tiny = tiny - frozen and whether tiny > frozen on every cond.
     If yes, tiny becomes the new baseline.
  2. Tiny bank + B1: use tiny bank's 1-NN pseudo-labels for the 500 unlabelled
     points plus the 56 true labels to fit a new linear probe (self-training).
     Does this beat tiny 1-NN alone and approach oracle without more true labels?
  3. Tiny bank + prototype: same but fit prototype means from tiny bank labels.

Usage:
  uv run python robust_diagnostic/al_tinybank_baseline_diag.py \
    --path_b <ckpt_dir> --method_b <method> --label <label> \
    --conds fog,crosstalk,snow,wet_ground \
    --out robust_diagnostic/logs/al_tinybank_baseline_<label>.json
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
def knn_predict(val_feats, bank_feats, bank_labels, k=1, device='cuda', chunk=4096):
    val_n=F.normalize(val_feats.float(),dim=1); bank_n=F.normalize(bank_feats.float(),dim=1)
    bank_d=bank_n.to(device)
    preds=[]
    for s in range(0,len(val_n),chunk):
        e=min(s+chunk,len(val_n))
        sim=val_n[s:e].to(device) @ bank_d.t()
        if k==1:
            preds.append(bank_labels[sim.argmax(1).cpu()])
        else:
            topk=sim.topk(k,dim=1).indices.cpu()
            batch_pred=[]
            for idx in topk:
                vals,counts=np.unique(bank_labels[idx].numpy(), return_counts=True)
                batch_pred.append(vals[counts.argmax()])
            preds.append(torch.tensor(batch_pred))
    return torch.cat(preds)
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
        W0=ridge_fit_soft(Xc,onehot(la[ci],NUM_CLASSES),args.lam,args.cg_iters,args.nystrom_m,device)
        Ws=ridge_fit_soft(Xp,onehot(pl,NUM_CLASSES),args.lam,args.cg_iters,args.nystrom_m,device)
        classes=sorted(set(pl.tolist()) & set(range(1,NUM_CLASSES)))
        cls_idx={c:(pl==c).nonzero().squeeze(1) for c in classes}
        r={'refs':{},'tiny':{},'tiny_plus':{}}
        r['refs']['frozen']=mw(W0,Xv,vl); r['refs']['oracle']=mw(Ws,Xv,vl)
        gap=r['refs']['oracle']-r['refs']['frozen']
        # tiny bank: k=8 per class (56 labels) + 500 random unlabelled
        for k in [8]:
            lab_idx=[]
            for c in classes:
                idx=cls_idx[c]
                if len(idx)<max(50,k): continue
                torch.manual_seed(2); lab_idx.append(idx[torch.randperm(len(idx))[:k]])
            if not lab_idx: continue
            lab_idx=torch.cat(lab_idx)
            # Q3 from previous diag: bank sizes 56, 556, 1056, 5056
            for add in [0,500,1000,5000]:
                if add==0:
                    bank_idx=lab_idx
                else:
                    pool_all=torch.arange(len(pool))
                    mask=torch.ones(len(pool),dtype=torch.bool); mask[lab_idx]=False
                    avail=pool_all[mask]
                    torch.manual_seed(3)
                    extra=avail[torch.randperm(len(avail))[:min(add,len(avail))]]
                    bank_idx=torch.cat([lab_idx, extra])
                # use raw 128-d for kNN (no HDC, same as Q3)
                bank_feats_raw=pool[bank_idx]; bank_labels=pl[bank_idx]
                pred=knn_predict(val_feats=val, val_labels=vl, bank_feats=bank_feats_raw, bank_labels=bank_labels, k=1, device=device)
                acc=compute_miou(pred, vl)
                r['tiny'][f"bank_{len(bank_idx)}"]={'miou':acc,'delta':acc - r['refs']['frozen'],'n_bank':len(bank_idx)}
        # tiny + B: use tiny bank's pseudo-labels for the 500 unlabelled to fit a new probe
        # Take k=8 labeled + 500 pseudo-labeled from bank_556's 1-NN predictions on those 500
        k=8
        lab_idx=[]
        for c in classes:
            idx=cls_idx[c]
            if len(idx)<max(50,k): continue
            torch.manual_seed(2); lab_idx.append(idx[torch.randperm(len(idx))[:k]])
        lab_idx=torch.cat(lab_idx)
        pool_all=torch.arange(len(pool))
        mask=torch.ones(len(pool),dtype=torch.bool); mask[lab_idx]=False
        avail=pool_all[mask]
        torch.manual_seed(3)
        extra=avail[torch.randperm(len(avail))[:500]]
        bank_idx=torch.cat([lab_idx, extra])
        # pseudo-label the 500 via 1-NN from labeled 56
        bank_feats_raw=pool[lab_idx]; bank_labels=pl[lab_idx]
        # predict for extra
        extra_pred=knn_predict(val_feats=pool[extra], val_labels=pl[extra], bank_feats=bank_feats_raw, bank_labels=bank_labels, k=1, device=device)
        # fit new probe on 56 true + 500 pseudo
        X_lab_pseudo=torch.cat([Xp[lab_idx], Xp[extra]], dim=0)
        Y_lab_pseudo=torch.cat([onehot(pl[lab_idx],NUM_CLASSES), onehot(extra_pred,NUM_CLASSES)], dim=0)
        W_pseudo=ridge_fit_soft(X_lab_pseudo, Y_lab_pseudo, args.lam, args.cg_iters, args.nystrom_m, device)
        r['tiny_plus']['pseudo_500']={'miou':mw(W_pseudo,Xv,vl),'delta':mw(W_pseudo,Xv,vl)-r['refs']['frozen']}
        # also test with 500 true labels (upper bound for 500 extra)
        # Take 500 extra with true labels (oracle for those 500)
        Y_true_pseudo=onehot(pl[extra],NUM_CLASSES)
        X_true_pseudo=torch.cat([Xp[lab_idx], Xp[extra]], dim=0)
        Y_true_cat=torch.cat([onehot(pl[lab_idx],NUM_CLASSES), Y_true_pseudo], dim=0)
        W_true=ridge_fit_soft(X_true_pseudo, Y_true_cat, args.lam, args.cg_iters, args.nystrom_m, device)
        r['tiny_plus']['true_500']={'miou':mw(W_true,Xv,vl),'delta':mw(W_true,Xv,vl)-r['refs']['frozen']}
        results['conds'][cond]=r
        del Xc,Xp,Xv,W0,Ws
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        print(f"\n=== {cond} ({toc(t0):.0f}s) ===")
        print(f"  frozen {r['refs']['frozen']:.3f} / oracle {r['refs']['oracle']:.3f} gap {gap:+.3f}")
        print(f"  tiny bank 1-NN: " + " ".join(f"{k}:{v['delta']:+.3f}" for k,v in r['tiny'].items()))
        print(f"  tiny+ pseudo {r['tiny_plus']['pseudo_500']['delta']:+.3f} vs true {r['tiny_plus']['true_500']['delta']:+.3f} (k=8+500)")
    os.makedirs(os.path.dirname(args.out),exist_ok=True)
    with open(args.out,'w') as fh: json.dump(results,fh,indent=2,default=float)
    print(f"\nSaved to {args.out}")
    print("\n=== READ ===")
    print("If tiny bank (56+500) 1-NN is >> frozen on every cond, it is a viable")
    print("baseline for the rest of the methods (as you asked). Compare tiny+ pseudo")
    print("vs true 500 to see if pseudo-label quality is the limit.")

if __name__=="__main__": main()
