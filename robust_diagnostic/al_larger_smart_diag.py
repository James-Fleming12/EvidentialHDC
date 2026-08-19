"""al_larger_smart_diag.py: larger dataset + smarter 500 allocation (no extractor change).

Keeps COV-SHIFT extractor frozen (your constraint) and tests two levers you asked:

  1. Larger dataset: does a larger pool/val (more frames, more varied points per
     class) raise the closeable gap and the ceiling? Compares 100 frames (50k pool
     / 100k val, current) vs 200 frames (100k pool / 200k val). Reports frozen,
     oracle and gap per condition, and the per-class frequency of rare classes
     (does the larger pool give the 500 bank more minority points to choose from).

  2. Smarter 500 allocation for the tiny bank (56 true + 500 pseudo, k=8 labels
     per class as the 56, then 500 extra). Four strategies for the 500:
     - random: uniform random from pool \\ lab_idx (baseline, current)
     - uniform: 500/17 per class (class-balanced to not starve rare)
     - diversity: farthest-point / k-means diverse in 128-d raw space
     - uncertainty: high entropy under W0 (max information gain, low confidence)
     For each, the 500 are pseudo-labeled by 1-NN from the 56, then
     W_pseudo fit on 56 true + 500 pseudo is evaluated (mIoU vs frozen, vs
     500 true oracle). If a smarter 500 beats random 500, the bank becomes a
     viable low-cost scale estimator without starving minority classes.

All eval-only on COV-SHIFT ep10, 4 conditions.

Usage:
  uv run python robust_diagnostic/al_larger_smart_diag.py \
    --path_b <ckpt_dir> --method_b <method> --label <label> \
    --conds fog,crosstalk,snow,wet_ground \
    --out robust_diagnostic/logs/al_larger_smart_<label>.json
"""
import os, sys, time, argparse, json, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn.functional as F
from dataset.kitti.parser import Parser
from modules.gen_trainers import GenTrainer
from modules.oracle_core import get_hdc_projection, compute_miou

NUM_CLASSES=17; SKETCH_SEED=11

def build_parser(root,data,arch,frames=100):
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

def select_diverse(pool_feats, avail_idx, n_select, device='cuda'):
    # farthest-point diverse in 128-d raw space, seeded by first avail
    avail_feats=pool_feats[avail_idx]
    # start with random
    torch.manual_seed(3)
    selected=[avail_idx[torch.randperm(len(avail_idx))[0].item()]]
    # iteratively add farthest from current set
    pool_n=F.normalize(pool_feats.float(),dim=1)
    for _ in range(1, n_select):
        cur=torch.stack([pool_feats[i] for i in selected])
        cur_n=F.normalize(cur.float(),dim=1).to(device)
        # max min distance to selected
        # chunk avail to avoid OOM
        best_idx=None; best_dist=-1
        for s in range(0,len(avail_idx),4096):
            e=min(s+4096,len(avail_idx))
            chunk=avail_idx[s:e]
            chunk_n=F.normalize(pool_feats[chunk].float(),dim=1).to(device)
            # dist = 1 - max sim to selected
            sim=chunk_n @ cur_n.t()  # chunk x selected
            max_sim=sim.max(1).values
            min_idx=(1-max_sim).argmax()
            if (1-max_sim).max().item() > best_dist:
                best_dist=(1-max_sim).max().item()
                best_idx=chunk[min_idx].item()
        if best_idx is None: break
        selected.append(best_idx)
        if len(selected)>=n_select: break
    return torch.tensor(selected)

def select_uncertainty(pool_feats, pool_labels, avail_idx, n_select, W0, proj, device='cuda'):
    # high entropy under W0 (HDC code space, need HDC codes for val pool)
    # compute HDC codes for avail pool points and W0 logits
    # chunked to avoid OOM
    avail_feats=pool_feats[avail_idx]
    # need HDC codes for these avail points
    # we have proj already, compute on the fly
    scores=[]
    for s in range(0,len(avail_idx),10000):
        e=min(s+10000,len(avail_idx))
        chunk_codes=torch.sign(avail_feats[s:e].to(device) @ proj).float()
        logits=chunk_codes @ W0.to(device)
        probs=torch.softmax(logits,dim=1)
        ent=-(probs * (probs+1e-12).log()).sum(1)
        scores.append(ent.cpu())
    scores=torch.cat(scores)
    top=torch.argsort(scores, descending=True)[:n_select]
    return avail_idx[top]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--kitti_dir",type=str,default="/mnt/alpha/jmfleming/KITTI")
    ap.add_argument("--kittic_dir",type=str,default="/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C")
    ap.add_argument("--config",type=str,default="config/labels/semantic-kitti-all.yaml")
    ap.add_argument("--arch",type=str,default="config/arch/senet-2048p.yml")
    ap.add_argument("--frames_small",type=int,default=100)
    ap.add_argument("--frames_large",type=int,default=200)
    ap.add_argument("--pool_small",type=int,default=50000)
    ap.add_argument("--pool_large",type=int,default=100000)
    ap.add_argument("--val_small",type=int,default=100000)
    ap.add_argument("--val_large",type=int,default=200000)
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
    clean_parser_small=build_parser(args.kitti_dir,DATA,ARCH,frames=args.frames_small)
    clean_parser_large=build_parser(args.kitti_dir,DATA,ARCH,frames=args.frames_large)
    # extract both clean sizes at start (for larger val later, reuse small clean 200k)
    fa_small,la_small=extract_features(model,clean_parser_small,device,args.frames_small)
    # for large, reuse small + extra frames if needed, but simpler: extract large separately if frames_large > frames_small
    if args.frames_large>args.frames_small:
        fa_large,la_large=extract_features(model,clean_parser_large,device,args.frames_large)
    else:
        fa_large,la_large=fa_small,la_small
    results={'label':args.label,'method':args.method_b,'conds':{}}
    for cond in conds:
        t0=tic()
        cdir=os.path.join(args.kittic_dir,cond,'heavy')
        if not os.path.exists(cdir): cdir=os.path.join(args.kittic_dir,cond,'moderate')
        # small and large pools
        f_small,l_small=extract_features(model,build_parser(args.kittic_dir,DATA,ARCH,frames=args.frames_small),device,args.frames_small)
        f_large,l_large=extract_features(model,build_parser(args.kittic_dir,DATA,ARCH,frames=args.frames_large),device,args.frames_large) if args.frames_large>args.frames_small else (f_small,l_small)
        # small pool/val
        torch.manual_seed(42); perm_small=torch.randperm(len(f_small))
        pool_s,pl_s=f_small[perm_small[:args.pool_small]],l_small[perm_small[:args.pool_small]]
        val_s,vl_s=f_small[perm_small[-args.val_small:]],l_small[perm_small[-args.val_small:]]
        # large pool/val
        torch.manual_seed(42); perm_large=torch.randperm(len(f_large))
        pool_l,pl_l=f_large[perm_large[:args.pool_large]],l_large[perm_large[:args.pool_large]]
        val_l,vl_l=f_large[perm_large[-args.val_large:]],l_large[perm_large[-args.val_large:]]
        # proj and clean for each size
        mc_small=min(args.max_clean,len(fa_small)); ci_small=torch.randperm(len(fa_small))[:mc_small]
        mc_large=min(args.max_clean,len(fa_large)); ci_large=torch.randperm(len(fa_large))[:mc_large]
        proj_small=get_hdc_projection(dim_in=fa_small.shape[1],dim_out=10000,device=device)
        proj_large=get_hdc_projection(dim_in=fa_large.shape[1],dim_out=10000,device=device)
        # use same proj dim, but fa dim same (128), so proj_small==proj_large in dim, reuse one for large
        proj=proj_small
        Xc_s=hdc_codes(fa_small[ci_small],proj,device).float()
        Xc_l=hdc_codes(fa_large[ci_large],proj,device).float()
        Xp_s=hdc_codes(pool_s,proj,device).float()
        Xp_l=hdc_codes(pool_l,proj,device).float()
        Xv_s=hdc_codes(val_s,proj,device).float()
        Xv_l=hdc_codes(val_l,proj,device).float()
        W0_s=ridge_fit_soft(Xc_s,onehot(la_small[ci_small],NUM_CLASSES),args.lam,args.cg_iters,args.nystrom_m,device)
        W0_l=ridge_fit_soft(Xc_l,onehot(la_large[ci_large],NUM_CLASSES),args.lam,args.cg_iters,args.nystrom_m,device)
        Ws_s=ridge_fit_soft(Xp_s,onehot(pl_s,NUM_CLASSES),args.lam,args.cg_iters,args.nystrom_m,device)
        Ws_l=ridge_fit_soft(Xp_l,onehot(pl_l,NUM_CLASSES),args.lam,args.cg_iters,args.nystrom_m,device)
        r={'refs':{},'larger':{},'smart500':{}}
        r['refs']['frozen_small']=mw(W0_s,Xv_s,vl_s); r['refs']['oracle_small']=mw(Ws_s,Xv_s,vl_s)
        r['refs']['frozen_large']=mw(W0_l,Xv_l,vl_l); r['refs']['oracle_large']=mw(Ws_l,Xv_l,vl_l)
        r['refs']['gap_small']=r['refs']['oracle_small']-r['refs']['frozen_small']
        r['refs']['gap_large']=r['refs']['oracle_large']-r['refs']['frozen_large']
        # larger dataset: per-class freq of rare classes
        classes_small=sorted(set(pl_s.tolist()) & set(range(1,NUM_CLASSES)))
        classes_large=sorted(set(pl_l.tolist()) & set(range(1,NUM_CLASSES)))
        freq_small={c:int((pl_s==c).sum().item()) for c in classes_small}
        freq_large={c:int((pl_l==c).sum().item()) for c in classes_large}
        r['larger']['freq_small']=freq_small; r['larger']['freq_large']=freq_large
        r['larger']['gap_small']=r['refs']['gap_small']; r['larger']['gap_large']=r['refs']['gap_large']
        # smarter 500 allocation: compare 4 strategies for the 500 extra on top of 56 true (k=8)
        k=8
        cls_idx_s={c:(pl_s==c).nonzero().squeeze(1) for c in classes_small}
        # k=8 lab set
        lab_idx=[]
        for c in classes_small:
            idx=cls_idx_s[c]
            if len(idx)<max(50,k): continue
            torch.manual_seed(2); lab_idx.append(idx[torch.randperm(len(idx))[:k]])
        lab_idx=torch.cat(lab_idx) if lab_idx else torch.tensor([],dtype=torch.long)
        # bank sizes for Q3 already showed 500 is the knee, test allocation of that 500
        pool_all=torch.arange(len(pool_s))
        mask=torch.ones(len(pool_s),dtype=torch.bool); mask[lab_idx]=False
        avail=pool_all[mask]
        for name, sel_fn in [
            ('random', lambda avail, n: avail[torch.randperm(len(avail))[:n]]),
            ('uniform', None),  # special: need per-class uniform
            ('diverse', lambda avail, n: select_diverse(pool_s, avail, n, device)),
            ('uncertainty', lambda avail, n: select_uncertainty(pool_s, pl_s, avail, n, W0_s, proj, device)),
        ]:
            if name=='uniform':
                # 500/17 per class, cap by avail per class
                per_c=500//len(classes_small)
                extra=[]
                for c in classes_small:
                    idx=(pl_s==c).nonzero().squeeze(1)
                    # avail for this class
                    avail_c=torch.tensor([i for i in avail.tolist() if pl_s[i].item()==c])
                    if len(avail_c)==0: continue
                    torch.manual_seed(3)
                    take=min(per_c, len(avail_c))
                    extra.append(avail_c[torch.randperm(len(avail_c))[:take]])
                # fill remainder randomly if needed
                if extra:
                    extra=torch.cat(extra)
                    if len(extra)<500:
                        remain=avail[~torch.isin(avail, extra)]
                        torch.manual_seed(3)
                        extra=torch.cat([extra, remain[torch.randperm(len(remain))[:500-len(extra)]]])
                else:
                    continue
            else:
                torch.manual_seed(3)
                extra=sel_fn(avail, 500)
            bank_idx=torch.cat([lab_idx, extra])
            # 1-NN via bank (raw 128-d)
            bank_feats=pool_s[bank_idx]; bank_labels=pl_s[bank_idx]
            pred=knn_predict(val_s, bank_feats, bank_labels, k=1, device=device)
            acc=compute_miou(pred, vl_s)
            # pseudo-label the 500 for W fit
            extra_pred=knn_predict(pool_s[extra], bank_feats[torch.arange(len(lab_idx))], pl_s[lab_idx], k=1, device=device) if len(extra)>0 else torch.tensor([],dtype=torch.long)
            # fit W on 56 true + 500 pseudo (HDC)
            # need HDC codes for lab + extra
            X_lab_pseudo=torch.cat([Xp_s[lab_idx], Xp_s[extra]], dim=0) if len(extra)>0 else Xp_s[lab_idx]
            # pseudo labels for extra are extra_pred, true for lab are pl[lab_idx]
            Y_pseudo=torch.cat([onehot(pl_s[lab_idx],NUM_CLASSES), onehot(extra_pred,NUM_CLASSES)], dim=0) if len(extra)>0 else onehot(pl_s[lab_idx],NUM_CLASSES)
            W_pseudo=ridge_fit_soft(X_lab_pseudo, Y_pseudo, args.lam, args.cg_iters, args.nystrom_m, device)
            r['smart500'][name]={'bank_miou':acc,'bank_delta':acc - r['refs']['frozen_small'],
                                 'W_pseudo_delta':mw(W_pseudo,Xv_s,vl_s)-r['refs']['frozen_small'],
                                 'n_bank':len(bank_idx)}
        results['conds'][cond]=r
        del Xc_s,Xp_s,Xv_s,Xc_l,Xp_l,Xv_l,W0_s,Ws_s,W0_l,Ws_l
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        print(f"\n=== {cond} ({toc(t0):.0f}s) ===")
        print(f"  small {args.frames_small}fr {args.pool_small}/{args.val_small}: frozen {r['refs']['frozen_small']:.3f} / oracle {r['refs']['oracle_small']:.3f} gap {r['refs']['gap_small']:+.3f}")
        print(f"  large {args.frames_large}fr {args.pool_large}/{args.val_large}: frozen {r['refs']['frozen_large']:.3f} / oracle {r['refs']['oracle_large']:.3f} gap {r['refs']['gap_large']:+.3f}")
        print(f"  smart500 bank mIoU: " + " ".join(f"{n}:{v['bank_delta']:+.3f}" for n,v in r['smart500'].items()))
        print(f"  smart500 W_pseudo delta: " + " ".join(f"{n}:{v['W_pseudo_delta']:+.3f}" for n,v in r['smart500'].items()))
    os.makedirs(os.path.dirname(args.out),exist_ok=True)
    with open(args.out,'w') as fh: json.dump(results,fh,indent=2,default=float)
    print(f"\nSaved to {args.out}")

if __name__=="__main__": main()
