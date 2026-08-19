"""al_bank_residual_diag.py: old tiny bank (500) on the new stable residual update.

Covers your request: do the 4 old bank allocations (random, uniform 29/class,
diverse farthest-point, uncertainty H(p)) immediately improve when the update
is the new stable form W = W0 + U_r C (r=8, oracle U, eta=1) instead of the
old full-probe W_pseudo on 56+500 pseudo?

All eval-only on COV-SHIFT ep10, 4 conditions. Keeps extractor frozen, no large
bank. For each bank (500 extra, k=8 per class as the 56 true):

  * 1-NN bank mIoU (as before, point predictor)
  * Full-probe W_pseudo on 56 true + 500 pseudo (old update, ridge on X_pseudo)
  * Residual W_res = W0 + U_r C on 56+500 pseudo, r=8, U_r = SVD(R) oracle (new)
  * Same three with 500 true labels (oracle for those 500, upper bound)

If W_res with the old bank beats W_pseudo and beats frozen at 56+500, the
new update is an immediate improvement on the existing bank.

Usage:
  uv run python robust_diagnostic/al_bank_residual_diag.py \
    --path_b <ckpt_dir> --method_b <method> --label <label> \
    --conds fog,crosstalk,snow,wet_ground \
    --out robust_diagnostic/logs/al_bank_residual_<label>.json
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
def select_diverse(pool_feats, avail_idx, n_select, device='cuda'):
    torch.manual_seed(3)
    selected=[avail_idx[torch.randperm(len(avail_idx))[0].item()]]
    pool_n=F.normalize(pool_feats.float(),dim=1)
    for _ in range(1, n_select):
        cur=torch.stack([pool_feats[i] for i in selected])
        cur_n=F.normalize(cur.float(),dim=1).to(device)
        best_idx=None; best_dist=-1
        for s in range(0,len(avail_idx),4096):
            e=min(s+4096,len(avail_idx))
            chunk=avail_idx[s:e]
            chunk_n=F.normalize(pool_feats[chunk].float(),dim=1).to(device)
            sim=chunk_n @ cur_n.t()
            max_sim=sim.max(1).values
            min_idx=(1-max_sim).argmax()
            if (1-max_sim).max().item() > best_dist:
                best_dist=(1-max_sim).max().item()
                best_idx=chunk[min_idx].item()
        if best_idx is None: break
        selected.append(best_idx)
        if len(selected)>=n_select: break
    return torch.tensor(selected)
def select_uncertainty(pool_feats, avail_idx, n_select, W0, proj, device='cuda'):
    avail_feats=pool_feats[avail_idx]
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
        Xd=Xp.to(device)
        W0=ridge_fit_soft(Xc,onehot(la[ci],NUM_CLASSES),args.lam,args.cg_iters,args.nystrom_m,device)
        Ws=ridge_fit_soft(Xp,onehot(pl,NUM_CLASSES),args.lam,args.cg_iters,args.nystrom_m,device)
        R=(Ws - W0).detach().cpu().float()
        U_full,_ ,_=torch.linalg.svd(R.double(),full_matrices=False); U_full=U_full.float()
        r={'refs':{},'bank':{}}
        r['refs']['frozen']=mw(W0,Xv,vl); r['refs']['oracle']=mw(Ws,Xv,vl)
        gap=r['refs']['oracle']-r['refs']['frozen']
        classes=sorted(set(pl.tolist()) & set(range(1,NUM_CLASSES)))
        cls_idx={c:(pl==c).nonzero().squeeze(1) for c in classes}
        # 56 true labels (k=8 per class)
        lab_idx=[]
        for c in classes:
            idx=cls_idx[c]
            if len(idx)<max(50,8): continue
            torch.manual_seed(2); lab_idx.append(idx[torch.randperm(len(idx))[:8]])
        lab_idx=torch.cat(lab_idx)
        # 4 bank allocations for the 500 extra
        pool_all=torch.arange(len(pool))
        mask=torch.ones(len(pool),dtype=torch.bool); mask[lab_idx]=False
        avail=pool_all[mask]
        banks={}
        banks['random']=avail[torch.randperm(len(avail))[:500]]
        # uniform 29/class
        per_c=500//len(classes)
        extra_u=[]
        for c in classes:
            idx=(pl==c).nonzero().squeeze(1)
            avail_c=torch.tensor([i for i in avail.tolist() if pl[i].item()==c])
            if len(avail_c)==0: continue
            take=min(per_c, len(avail_c))
            extra_u.append(avail_c[torch.randperm(len(avail_c))[:take]])
        if extra_u:
            extra_u=torch.cat(extra_u)
            if len(extra_u)<500:
                remain=avail[~torch.isin(avail, extra_u)]
                extra_u=torch.cat([extra_u, remain[torch.randperm(len(remain))[:500-len(extra_u)]]])
            banks['uniform']=extra_u[:500]
        else:
            banks['uniform']=banks['random']
        banks['diverse']=select_diverse(pool, avail, 500, device)
        banks['uncertainty']=select_uncertainty(pool, pl, avail, 500, W0, proj, device)
        for bname, extra in banks.items():
            bank_idx=torch.cat([lab_idx, extra])
            # 1-NN bank mIoU (as before)
            bank_feats=pool[bank_idx]; bank_labels=pl[bank_idx]
            pred=knn_predict(val, bank_feats, bank_labels, k=1, device=device)
            bank_miou=compute_miou(pred, vl)
            # pseudo-label the 500 via 1-NN from 56, fit W_pseudo (old update) and W_res (new)
            extra_pred=knn_predict(pool[extra], pool[lab_idx], pl[lab_idx], k=1, device=device)
            X_lab_pseudo=torch.cat([Xp[lab_idx], Xp[extra]], dim=0)
            Y_pseudo=torch.cat([onehot(pl[lab_idx],NUM_CLASSES), onehot(extra_pred,NUM_CLASSES)], dim=0)
            W_pseudo=ridge_fit_soft(X_lab_pseudo, Y_pseudo, args.lam, args.cg_iters, args.nystrom_m, device)
            # new: W = W0 + U_r C, r=8, oracle U, C fit from 56+500 pseudo
            U8=U_full[:,:8]
            # C fit from pseudo labels on residual
            X_pseudo=X_lab_pseudo; Y_pseudo_t=Y_pseudo
            # lsq_residual expects X_lab, Y_lab, W0, U
            # inline lsq
            Xd_p=X_pseudo.to(device).float(); Yd_p=Y_pseudo.to(device).float(); U_d=U8.to(device)
            rnk=U_d.shape[1]; XU=Xd_p @ U_d
            A=XU.t()@XU+1e-6*torch.eye(rnk,device=device)
            b=XU.t()@(Yd_p - Xd_p @ W0.to(device))
            C=torch.linalg.solve(A,b).cpu()
            W_res=W0.detach().cpu() + (U8.cpu() @ C)
            # also true 500 upper bound for both
            Y_true_pseudo=torch.cat([onehot(pl[lab_idx],NUM_CLASSES), onehot(pl[extra],NUM_CLASSES)], dim=0)
            W_true_pseudo=ridge_fit_soft(X_lab_pseudo, Y_true_pseudo, args.lam, args.cg_iters, args.nystrom_m, device)
            X_true_lab=torch.cat([Xp[lab_idx], Xp[extra]], dim=0); Y_true_lab=Y_true_pseudo
            # true residual
            Xd_t=X_true_lab.to(device).float(); Yd_t=Y_true_lab.to(device).float()
            XU_t=Xd_t @ U8.to(device)
            A_t=XU_t.t()@XU_t+1e-6*torch.eye(8,device=device)
            b_t=XU_t.t()@(Yd_t - Xd_t @ W0.to(device))
            Ct=torch.linalg.solve(A_t,b_t).cpu()
            W_res_true=W0.detach().cpu() + (U8.cpu() @ Ct)
            r['bank'][bname]={'bank_miou':bank_miou,'bank_delta':bank_miou - r['refs']['frozen'],
                              'W_pseudo_delta':mw(W_pseudo,Xv,vl)-r['refs']['frozen'],
                              'W_res_pseudo_delta':mw(W_res,Xv,vl)-r['refs']['frozen'],
                              'W_res_true_delta':mw(W_res_true,Xv,vl)-r['refs']['frozen'],
                              'W_true_pseudo_delta':mw(W_true_pseudo,Xv,vl)-r['refs']['frozen'],
                              'n_bank':len(bank_idx)}
        results['conds'][cond]=r
        del Xc,Xp,Xv,W0,Ws,R,U_full
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        print(f"\n=== {cond} ({toc(t0):.0f}s) ===")
        print(f"  frozen {r['refs']['frozen']:.3f} / oracle {r['refs']['oracle']:.3f} gap {gap:+.3f}")
        for bname in ['random','uniform','diverse','uncertainty']:
            if bname not in r['bank']: continue
            b=r['bank'][bname]
            print(f"  {bname:12s} bank {b['bank_delta']:+.3f} | W_pseudo {b['W_pseudo_delta']:+.3f} | W_res_pseudo {b['W_res_pseudo_delta']:+.3f} (true {b['W_res_true_delta']:+.3f})")
    os.makedirs(os.path.dirname(args.out),exist_ok=True)
    with open(args.out,'w') as fh: json.dump(results,fh,indent=2,default=float)
    print(f"\nSaved to {args.out}")

if __name__=="__main__": main()
