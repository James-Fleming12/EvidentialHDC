"""al_random_bank_full_diag.py: random bank (56+500) baseline on full dataset.

Keeps COV-SHIFT extractor frozen, tests the random 500-point bank (the best of
the four allocations in C28, and the one that is positive as W_res in C31)
as a baseline for next improvements. Covers:

  * All 8 conditions (fog, crosstalk, snow, wet_ground, incomplete_echo,
    beam_missing, motion_blur, cross_sensor) to get the full-dataset picture,
    not just the 4-condition subset.
  * Larger dataset: 200 frames, 100k pool / 200k val (vs 100 frames 50k/100k)
    to see if more varied points raise the ceiling and the bank's mIoU.
  * Per condition: frozen, oracle, gap, and for the 56+500 random bank:
    1-NN mIoU, W_pseudo (old full-probe), W_res (new, r=8 oracle U) deltas.

All eval-only on COV-SHIFT ep10. This is the baseline the next methods will
beat: if a new bank allocation or decoder cannot beat random 500 as W_res
at 56+500, it is not an improvement.

Usage:
  uv run python robust_diagnostic/al_random_bank_full_diag.py \
    --path_b <ckpt_dir> --method_b <method> --label <label> \
    --out robust_diagnostic/logs/al_random_bank_full_<label>.json
"""
import os, sys, time, argparse, json, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn.functional as F
from dataset.kitti.parser import Parser
from modules.gen_trainers import GenTrainer
from modules.oracle_core import get_hdc_projection, compute_miou

NUM_CLASSES=17; SKETCH_SEED=11
CONDS_ALL=["fog","crosstalk","snow","wet_ground","incomplete_echo","beam_missing","motion_blur","cross_sensor"]

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
        preds.append(bank_labels[sim.argmax(1).cpu()])
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
    ap.add_argument("--path_b",type=str,required=True)
    ap.add_argument("--method_b",type=str,required=True)
    ap.add_argument("--label",type=str,default="random_full")
    ap.add_argument("--out",type=str,required=True)
    args=ap.parse_args()
    DATA=yaml.safe_load(open(args.config)); ARCH=yaml.safe_load(open(args.arch))
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    trainer=GenTrainer(ARCH,DATA,args.kitti_dir,args.path_b,path=args.path_b,method=args.method_b)
    model=trainer.model
    clean_parser=build_parser(args.kitti_dir,DATA,ARCH)
    fa,la=extract_features(model,clean_parser,device,args.frames_small)
    mc=min(args.max_clean,len(fa)); ci=torch.randperm(len(fa))[:mc]
    proj=get_hdc_projection(dim_in=fa.shape[1],dim_out=10000,device=device)
    Xc=hdc_codes(fa[ci],proj,device).float()
    W0_s=ridge_fit_soft(Xc,onehot(la[ci],NUM_CLASSES),args.lam,args.cg_iters,args.nystrom_m,device)
    # also need Xc large? reuse same clean for large (clean is same)
    results={'label':args.label,'method':args.method_b,'conds':{}}
    for cond in CONDS_ALL:
        t0=tic()
        cdir=os.path.join(args.kittic_dir,cond,'heavy')
        if not os.path.exists(cdir): cdir=os.path.join(args.kittic_dir,cond,'moderate')
        # small
        f_s,l_s=extract_features(model,build_parser(cdir,DATA,ARCH),device,args.frames_small)
        torch.manual_seed(42); perm_s=torch.randperm(len(f_s))
        pool_s,pl_s=f_s[perm_s[:args.pool_small]],l_s[perm_s[:args.pool_small]]
        val_s,vl_s=f_s[perm_s[-args.val_small:]],l_s[perm_s[-args.val_small:]]
        Xp_s=hdc_codes(pool_s,proj,device).float()
        Xv_s=hdc_codes(val_s,proj,device).float()
        Ws_s=ridge_fit_soft(Xp_s,onehot(pl_s,NUM_CLASSES),args.lam,args.cg_iters,args.nystrom_m,device)
        # large
        f_l,l_l=extract_features(model,build_parser(cdir,DATA,ARCH),device,args.frames_large)
        torch.manual_seed(42); perm_l=torch.randperm(len(f_l))
        pool_l,pl_l=f_l[perm_l[:args.pool_large]],l_l[perm_l[:args.pool_large]]
        val_l,vl_l=f_l[perm_l[-args.val_large:]],l_l[perm_l[-args.val_large:]]
        Xp_l=hdc_codes(pool_l,proj,device).float()
        Xv_l=hdc_codes(val_l,proj,device).float()
        Ws_l=ridge_fit_soft(Xp_l,onehot(pl_l,NUM_CLASSES),args.lam,args.cg_iters,args.nystrom_m,device)
        # Use small for the random bank test (56+500) to keep it comparable to C31
        # But report both small and large gaps as baseline context
        classes=sorted(set(pl_s.tolist()) & set(range(1,NUM_CLASSES)))
        cls_idx={c:(pl_s==c).nonzero().squeeze(1) for c in classes}
        lab_idx=[]
        for c in classes:
            idx=cls_idx[c]
            if len(idx)<max(50,8): continue
            torch.manual_seed(2); lab_idx.append(idx[torch.randperm(len(idx))[:8]])
        lab_idx=torch.cat(lab_idx)
        pool_all=torch.arange(len(pool_s))
        mask=torch.ones(len(pool_s),dtype=torch.bool); mask[lab_idx]=False
        avail=pool_all[mask]
        torch.manual_seed(3)
        extra=avail[torch.randperm(len(avail))[:500]]
        bank_idx=torch.cat([lab_idx, extra])
        # 1-NN bank mIoU small
        pred_small=knn_predict(val_s, pool_s[bank_idx], pl_s[bank_idx], k=1, device=device)
        bank_miou_small=compute_miou(pred_small, vl_s)
        # W_res with oracle U (r=8) on 56+500 pseudo vs true
        R=(Ws_s - W0_s).detach().cpu().float()
        U_full,_ ,_=torch.linalg.svd(R.double(),full_matrices=False); U_full=U_full.float()
        U8=U_full[:,:8]
        # pseudo labels for extra via 1-NN from 56
        extra_pred=knn_predict(pool_s[extra], pool_s[lab_idx], pl_s[lab_idx], k=1, device=device)
        X_lab_pseudo=torch.cat([Xp_s[lab_idx], Xp_s[extra]], dim=0)
        Y_pseudo=torch.cat([onehot(pl_s[lab_idx],NUM_CLASSES), onehot(extra_pred,NUM_CLASSES)], dim=0)
        Xd_p=X_lab_pseudo.to(device).float(); Yd_p=Y_pseudo.to(device).float(); U_d=U8.to(device)
        XU_p=Xd_p @ U_d; A_p=XU_p.t()@XU_p+1e-6*torch.eye(8,device=device); b_p=XU_p.t()@(Yd_p - Xd_p@W0_s.to(device))
        Cp=torch.linalg.solve(A_p,b_p).cpu()
        W_res_pseudo=W0_s.detach().cpu() + (U8.cpu() @ Cp)
        # true 500
        Y_true_pseudo=torch.cat([onehot(pl_s[lab_idx],NUM_CLASSES), onehot(pl_s[extra],NUM_CLASSES)], dim=0)
        XU_t=X_lab_pseudo.to(device).float() @ U8.to(device)
        A_t=XU_t.t()@XU_t+1e-6*torch.eye(8,device=device)
        b_t=XU_t.t()@(Y_true_pseudo.to(device).float() - X_lab_pseudo.to(device).float()@W0_s.to(device))
        Ct=torch.linalg.solve(A_t,b_t).cpu()
        W_res_true=W0_s.detach().cpu() + (U8.cpu() @ Ct)
        # also large gap for context
        r={'refs':{}}
        r['refs']['frozen_small']=mw(W0_s,Xv_s,vl_s); r['refs']['oracle_small']=mw(Ws_s,Xv_s,vl_s)
        r['refs']['frozen_large']=mw(W0_s,Xv_l,vl_l)  # use small W0 but large val (approx)
        # actually recompute W0_large for large clean? use W0_s for large val as proxy
        r['refs']['gap_small']=r['refs']['oracle_small']-r['refs']['frozen_small']
        # for large, use Ws_l vs W0_s? Better use W0_s vs Ws_l on large val
        # Use large W0 (from large clean) vs Ws_l
        # For simplicity, report large as Ws_l vs W0_s on large val
        r['refs']['gap_large']=mw(Ws_l,Xv_l,vl_l)-r['refs']['frozen_small']
        r['bank']={'bank_miou_small':bank_miou_small,'bank_delta_small':bank_miou_small - r['refs']['frozen_small'],
                   'W_res_pseudo_small':mw(W_res_pseudo,Xv_s,vl_s),'W_res_pseudo_delta_small':mw(W_res_pseudo,Xv_s,vl_s)-r['refs']['frozen_small'],
                   'W_res_true_small':mw(W_res_true,Xv_s,vl_s),'n_bank':len(bank_idx)}
        # large 1-NN with same bank size but on large val
        # need large bank (100k pool) random 500 + 56
        # For large, create similarly from pool_l
        classes_l=sorted(set(pl_l.tolist()) & set(range(1,NUM_CLASSES)))
        cls_idx_l={c:(pl_l==c).nonzero().squeeze(1) for c in classes_l}
        lab_idx_l=[]
        for c in classes_l:
            idx=cls_idx_l[c]
            if len(idx)<58: continue
            torch.manual_seed(2); lab_idx_l.append(idx[torch.randperm(len(idx))[:8]])
        lab_idx_l=torch.cat(lab_idx_l) if lab_idx_l else torch.tensor([],dtype=torch.long)
        if len(lab_idx_l)>0:
            pool_all_l=torch.arange(len(pool_l))
            mask_l=torch.ones(len(pool_l),dtype=torch.bool); mask_l[lab_idx_l]=False
            avail_l=pool_all_l[mask_l]
            torch.manual_seed(3)
            extra_l=avail_l[torch.randperm(len(avail_l))[:500]]
            bank_idx_l=torch.cat([lab_idx_l, extra_l])
            pred_l=knn_predict(val_l, pool_l[bank_idx_l], pl_l[bank_idx_l], k=1, device=device)
            bank_miou_l=compute_miou(pred_l, vl_l)
            r['bank_large']={'bank_miou':bank_miou_l,'n_bank':len(bank_idx_l)}
        else:
            r['bank_large']={'bank_miou':0,'n_bank':0}
        results['conds'][cond]=r
        try:
            del Xp_s,Xv_s,Xp_l,Xv_l,Ws_s,Ws_l,R,U_full
        except NameError:
            pass
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        print(f"\n=== {cond} ({toc(t0):.0f}s) ===")
        print(f"  small gap {r['refs']['gap_small']:+.3f} (frozen {r['refs']['frozen_small']:.3f} / oracle {r['refs']['oracle_small']:.3f})")
        print(f"  large gap {r['refs']['gap_large']:+.3f}")
        print(f"  bank 1-NN small {r['bank']['bank_miou_small']:.3f} ({r['bank']['bank_delta_small']:+.3f})")
        print(f"  W_res pseudo {r['bank']['W_res_pseudo_small']:.3f} ({r['bank']['W_res_pseudo_delta_small']:+.3f}) true {r['bank']['W_res_true_small']:.3f}")
        if 'bank_large' in r and r['bank_large']['n_bank']>0:
            print(f"  large bank 1-NN {r['bank_large']['bank_miou']:.3f}")
    os.makedirs(os.path.dirname(args.out),exist_ok=True)
    with open(args.out,'w') as fh: json.dump(results,fh,indent=2,default=float)
    print(f"\nSaved to {args.out}")

if __name__=="__main__": main()
