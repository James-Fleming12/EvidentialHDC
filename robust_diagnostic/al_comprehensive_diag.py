"""al_comprehensive_diag.py: is there cheap AL headroom in the new feature spaces?

Two questions, one eval-only script (minutes per condition), on the ball/spec 21ep
checkpoints (or any extractor):

  Part 1 — slight method variations: does the Iteration-10 fractional-residual
  recipe (W = W_frozen + eta*(W_beta - W_frozen), beta~0.6) become positive on
  snow/wet_ground with a CHEAP tweak?  All at k=8 means/class (64-72 labels) to
  keep the "cheap" constraint.  Variants are the Iteration-11 deployment gaps:

    V0  baseline:      oracle counts x random-k means, threshold max(50,k) [the
                        21ep C14 baseline, beta=0.6 eta per-condition best]
    V1  source-count:  source prior counts (clean class frequencies projected to
                        pool size) x random-k means  -- the 8F test
    V2  all-class:     same as V0 but threshold k only (rare classes included)
    V3  control-var:   source clean mean as control variate: mu_hat = rho*mu_clean
                        + (1-rho)*mu_sample, rho=0.5, oracle counts
    V4  source+all:    V1 + V2 together (fully deployable: no oracle pool stats)

  Part 2 — feature-space properties that could enable a cheap AL framework:

    - intra/inter cosine + separation (blob tightness; the ball objective)
    - 1-NN purity (packing)
    - kappa + participation rank (spectrum flatness; the spec objective)
    - mean-k curve (sample complexity of class means, k=2,8,32)
    - whitened T error at V0 (ridge sensitivity; the 8-10 smoking gun)
    - leverage vs confidence ranking (which points to label; Spearman)
    - per-class frozen error vs frequency (which classes need the budget)
    - prototype (R1) vs linear probe at k=8: does the ball space make
      prototype decode viable, i.e. could we drop the linear classifier?

All sections share one feature extraction (clean + pool + val) per condition.

Usage:
  uv run python robust_diagnostic/al_comprehensive_diag.py \
    --path_b <ckpt_dir> --method_b <method> --label <label> \
    --conds fog,crosstalk,snow,wet_ground \
    --out robust_diagnostic/logs/al_comprehensive_<label>.json
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
            out=model(in_vol)
            z8=out[2] if len(out)==3 else out[1]
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
    ap.add_argument("--mean_k",type=int,default=8)
    ap.add_argument("--betas",type=str,default="0.6,0.75")
    ap.add_argument("--etas",type=str,default="0.05,0.1,0.2,0.3,0.5")
    ap.add_argument("--conds",type=str,default="fog,crosstalk,snow,wet_ground")
    ap.add_argument("--path_b",type=str,required=True)
    ap.add_argument("--method_b",type=str,required=True)
    ap.add_argument("--label",type=str,default="med")
    ap.add_argument("--out",type=str,required=True)
    args=ap.parse_args()
    DATA=yaml.safe_load(open(args.config)); ARCH=yaml.safe_load(open(args.arch))
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    conds=[c.strip() for c in args.conds.split(',') if c.strip()]
    betas=[float(x) for x in args.betas.split(',')]; etas=[float(x) for x in args.etas.split(',')]
    k=args.mean_k
    trainer=GenTrainer(ARCH,DATA,args.kitti_dir,args.path_b,path=args.path_b,method=args.method_b)
    model=trainer.model
    clean_parser=build_parser(args.kitti_dir,DATA,ARCH)
    fa,la=extract_features(model,clean_parser,device,args.frames)
    results={'label':args.label,'method':args.method_b,'mean_k':k,'betas':betas,'etas':etas,'conds':{}}
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
        # CHUNKED HDC: unchunked X.to(device)@proj with 50-200k x 10k overflows
        # GPU memory (200k*10k float = 8GB per matmul). The defined hdc_codes()
        # chunks at 100k rows to keep peak < 4GB.
        Xc=hdc_codes(fa[ci],proj,device,chunk=100000).float()
        Xp=hdc_codes(pool,proj,device,chunk=100000).float()
        Xv=hdc_codes(val,proj,device,chunk=100000).float()
        Xd=Xp.to(device); N=Xp.shape[0]
        W_clean=ridge_fit_soft(Xc,onehot(la[ci],NUM_CLASSES),args.lam,args.cg_iters,args.nystrom_m,device)
        W_oracle=ridge_fit_soft(Xp,onehot(pl,NUM_CLASSES),args.lam,args.cg_iters,args.nystrom_m,device)
        r={'refs':{},'props':{},'slight':{},'prototype':{}}
        r['refs']['frozen']=mw(W_clean,Xv,vl); r['refs']['oracle']=mw(W_oracle,Xv,vl)
        classes=sorted(set(pl.tolist()) & set(range(1,NUM_CLASSES)))
        cls_idx={c:(pl==c).nonzero().squeeze(1) for c in classes}
        # spectral-exact ceiling + per-condition best beta/eta
        S=(Xd.t()@Xd).double()/N; eigS,U=torch.linalg.eigh(S); eigS=eigS.float(); U=U.float()
        lam_hat=args.lam/N; sig=(eigS+lam_hat).clamp(min=lam_hat)
        T_or=torch.zeros(10000,NUM_CLASSES)
        for c in classes: T_or[:,c]=Xp[cls_idx[c]].sum(0)
        m0=pl==0
        if int(m0.sum().item())>0: T_or[:,0]=Xp[m0].sum(0)
        T_or=T_or/N; Uc=U.to(device); sig_d=sig.to(device)
        UtT_or=Uc.t()@T_or.to(device)
        W_or_spec=(Uc@((1.0/sig_d).unsqueeze(1)*UtT_or)).cpu().float()
        r['refs']['oracle_spec']=mw(W_or_spec,Xv,vl)
        # ---- Part 2: feature-space properties ----
        pr=r['props']
        # raw 128-d pool features for geometry
        # 1-NN purity + intra/inter
        zn=F.normalize(pool.float(),dim=1)
        means={}
        for c in classes:
            idx=cls_idx[c]
            if len(idx)>=50: means[c]=zn[idx].mean(0)
        cs=sorted(means); cmat=torch.stack([means[c] for c in cs]) if cs else torch.zeros(0,zn.shape[1])
        if len(cs)>1:
            cmat=F.normalize(cmat,dim=1)
        # intra/inter
        intra=[]; inter=[]
        for i,c in enumerate(cs):
            m=zn[cls_idx[c]]
            if len(m)==0: continue
            mu=F.normalize(means[c].unsqueeze(0),dim=1)[0]
            intra.append(float((m@mu).mean().item()))
            for j in range(i+1,len(cs)):
                inter.append(float((m@cmat[j]).mean().item()))
        pr['intra_cos']=float(np.mean(intra)) if intra else None
        pr['inter_cos']=float(np.mean(inter)) if inter else None
        pr['separation']=pr['intra_cos']-pr['inter_cos'] if intra and inter else None
        # 1-NN purity — hoist zn_d to avoid re-uploading each class/chunk;
        # chunked matmul keeps peak at ~820 MB (4096*50k) instead of 50k*50k.
        nn1=0; den=0
        zn_d=zn.to(device)
        for c in classes:
            idx=cls_idx[c]
            if len(idx)<50: continue
            sub=zn[idx].to(device); den+=len(idx)
            for s in range(0,len(idx),4096):
                e=min(s+4096,len(idx))
                sim=sub[s:e]@zn_d.t()
                sim[torch.arange(e-s,device=device),idx[s:e]]=-1e9
                nn=sim.argmax(1)
                nn1+=int((pl[nn.cpu()]==c).sum().item())
        del zn_d
        pr['nn1_purity']=nn1/den if den else None
        # kappa / prank
        Sf=pool.float()-pool.float().mean(0); cov=(Sf.t()@Sf)/(len(pool)-1)
        eig=torch.linalg.eigvalsh(cov.float()).clamp(min=1e-8)
        pr['kappa']=float((eig[-1]/eig[0]).item()); pr['participation_rank']=float((eig.sum()**2/(eig**2).sum()).item())
        # mean-k curve
        pr['mean_k']={}
        for kk in [2,8,32]:
            vals=[]
            for c in classes:
                idx=cls_idx[c]
                if len(idx)<max(50,kk): continue
                mu_true=F.normalize(zn[idx].mean(0).unsqueeze(0),dim=1)[0]
                acc=[]
                for rep in range(5):
                    torch.manual_seed(rep)
                    sub=F.normalize(zn[idx[torch.randperm(len(idx))[:kk]]].mean(0).unsqueeze(0),dim=1)[0]
                    acc.append(float((sub*mu_true).sum().item()))
                vals.append(float(np.mean(acc)))
            pr['mean_k'][str(kk)]=float(np.mean(vals)) if vals else None
        # whitened T error at V0 (sensitivity)
        # leverage/conf ranking
        Xp_d=Xp.to(device); Wf=W_clean.to(device)
        scores=(Xp_d.float()@Wf).max(1).values.cpu(); conf=scores
        # influence approx: || X * residual ||  (cheap proxy for per-point ||dW||)
        # residual per point = onehot - softmax(scores)
        probs=torch.softmax(Xp_d.float()@Wf,dim=1).cpu()
        Yoh=onehot(pl,NUM_CLASSES).float()
        resid=(Yoh-probs).norm(dim=1)
        # leverage proxy: row norm of X (code norm is constant = sqrt(d), so use feature norm)
        lev=pool.float().norm(dim=1)
        def spearman(a,b):
            a=np.array(a); b=np.array(b); 
            ra=a.argsort().argsort().astype(float); rb=b.argsort().argsort().astype(float)
            return float(np.corrcoef(ra,rb)[0,1]) if len(a)>10 else None
        pr['lev_conf_spearman']=spearman(lev.numpy(),conf.numpy())
        pr['resid_conf_spearman']=spearman(resid.numpy(),conf.numpy())
        # per-class frozen error vs freq
        Wf_cpu=W_clean.detach().cpu()
        preds=decode(Wf_cpu,Xv)
        # quick per-class IoU on val
        per_c={}
        for c in classes:
            m=(vl==c)
            if int(m.sum())==0: continue
            inter=int(((preds==c)&m).sum().item()); uni=int(((preds==c)|m).sum().item())
            per_c[str(c)]={'freq':int((pl==c).sum().item()),'iou':inter/max(1,uni)}
        pr['per_class_val']=per_c
        # prototype (R1) at k=8 vs linear probe (cheap-classifier question)
        # R1: nearest class mean in 128-d
        protos=torch.stack([F.normalize(zn[cls_idx[c]].mean(0).unsqueeze(0),dim=1)[0] for c in cs]) if cs else torch.zeros(0,zn.shape[1])
        # val decode by prototype
        if len(cs)>0:
            zn_val=F.normalize(val.float(),dim=1)
            sim_p=zn_val@F.normalize(protos,dim=1).t()
            pred_p=torch.tensor([cs[i] for i in sim_p.argmax(1).tolist()])
            # map to 0..16 via direct
            # compute IoU for prototype decode
            inter_p=(pred_p==vl).float().mean().item()  # accuracy proxy
            # use compute_miou with mapped predictions (pred_p already in 1..16)
            # need full NUM_CLASSES mapping: prototype only has `cs` classes, others never predicted -> iou 0
            pred_p_full=pred_p
            pr['prototype_acc']=float(inter_p)
            pr['prototype_miou']=compute_miou(pred_p_full,vl)
        else:
            pr['prototype_acc']=None; pr['prototype_miou']=None
        pr['linear_frozen_miou']=r['refs']['frozen']
        # ---- Part 1: slight method variations (all at k=8) ----
        sl=r['slight']
        # helpers for T_hat variants
        clean_classes=sorted(set(la[ci].tolist()) & set(range(1,NUM_CLASSES)))
        clean_idx={c:(la[ci]==c).nonzero().squeeze(1) for c in clean_classes}
        clean_code_means={c:Xc[idx].mean(0) for c in clean_classes if len((idx:=clean_idx[c]))>0}
        tot_clean=len(la[ci]); clean_freq={c:int((la[ci]==c).sum().item())/tot_clean for c in classes}
        def make_T(counts, thresh, rho=None):
            # counts: dict c->count, thresh: min pool points to include class, rho: control-var weight
            Th=torch.zeros(10000,NUM_CLASSES)
            for c in classes:
                idx=cls_idx[c]
                if len(idx)<thresh: continue
                torch.manual_seed(2)
                mu_sample=Xp[idx[torch.randperm(len(idx))[:k]]].mean(0)
                if rho is not None and c in clean_code_means:
                    mu_sample=rho*clean_code_means[c] + (1-rho)*mu_sample
                Th[:,c]=counts[c]*mu_sample
            return Th/N
        # counts
        oracle_counts={c:len(cls_idx[c]) for c in classes}
        source_counts={c:int(clean_freq.get(c,0)*N) for c in classes}
        # grid for each variant: sweep beta/eta, keep best delta
        def best_combo(Th, betas=betas, etas=etas):
            UtTh=Uc.t()@Th.to(device)
            Wf=W_clean.detach().cpu()
            best=None; best_cfg=None
            for beta in betas:
                Wb=(Uc@ (sig_d.pow(-beta).unsqueeze(1)*UtTh)).cpu().float()
                for eta in etas:
                    W=Wf+eta*(Wb-Wf)
                    d=mw(W,Xv,vl)-r['refs']['frozen']
                    mi=mw(W,Xv,vl)
                    if best is None or d>best['delta']:
                        best={'beta':beta,'eta':eta,'delta':d,'miou':mi}
            # also report the single (0.6,0.05) that was best at medium
            key='0.6' if 0.6 in betas else str(betas[0])
            single=None
            for beta in betas:
                for eta in etas:
                    if abs(beta-0.6)<1e-6 and abs(eta-0.05)<1e-6:
                        Wb=(Uc@ (sig_d.pow(-0.6).unsqueeze(1)*UtTh)).cpu().float()
                        W=Wf+0.05*(Wb-Wf)
                        single={'delta':mw(W,Xv,vl)-r['refs']['frozen'],'miou':mw(W,Xv,vl)}
            return best, single
        # V0 baseline
        b0,s0=best_combo(make_T(oracle_counts, max(50,k), None))
        sl['V0_baseline']={'desc':'oracle counts, thresh 50, random-k means','best':b0,'at_06_005':s0}
        # V1 source-count prior
        b1,s1=best_combo(make_T(source_counts, max(50,k), None))
        sl['V1_source_counts']={'desc':'source prior counts, thresh 50','best':b1,'at_06_005':s1}
        # V2 all-class inclusive
        b2,s2=best_combo(make_T(oracle_counts, k, None))
        sl['V2_all_class']={'desc':'oracle counts, thresh k (rare included)','best':b2,'at_06_005':s2}
        # V3 control variate
        b3,s3=best_combo(make_T(oracle_counts, max(50,k), rho=0.5))
        sl['V3_control_var']={'desc':'oracle counts, rho=0.5 control variate','best':b3,'at_06_005':s3}
        # V4 fully deployable (source + all-class)
        b4,s4=best_combo(make_T(source_counts, k, None))
        sl['V4_source_all']={'desc':'source counts + all-class (deployable)','best':b4,'at_06_005':s4}
        results['conds'][cond]=r
        # free per-condition GPU tensors before next condition
        del Xc, Xp, Xv, Xd, S, eigS, U, Uc, sig, sig_d, UtT_or, W_or_spec, W_clean, W_oracle
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        print(f"\n=== {cond} ({toc(t0):.0f}s) ===")
        print(f"  frozen {r['refs']['frozen']:.3f} / oracle {r['refs']['oracle']:.3f} / spec-ceil {r['refs']['oracle_spec']:.3f}")
        print(f"  props: intra {pr['intra_cos']:.3f} inter {pr['inter_cos']:.3f} sep {pr['separation']:.3f} | nn1 {pr['nn1_purity']:.3f} kappa {pr['kappa']:.0f} prank {pr['participation_rank']:.0f} | R1 {pr['prototype_miou']:.3f} vs lin {pr['linear_frozen_miou']:.3f}")
        print(f"  slight variations (best delta at k=8, beta in {betas} eta in {etas}):")
        for tag in ['V0_baseline','V1_source_counts','V2_all_class','V3_control_var','V4_source_all']:
            b=sl[tag]['best']
            print(f"    {tag}: best b={b['beta']} e={b['eta']} -> {b['delta']:+.3f} (miou {b['miou']:.3f})")
    os.makedirs(os.path.dirname(args.out),exist_ok=True)
    with open(args.out,'w') as fh: json.dump(results,fh,indent=2,default=float)
    print(f"\nSaved to {args.out}")
    print("\n=== READ ===")
    print("Part 1 (slight): is there a V1-V4 whose best delta is POSITIVE on snow/wet")
    print("or materially better than V0? If V1 (source counts) holds V0, the method")
    print("is deployable (no oracle pool stats). V2 tests the rare-class exclusion.")
    print("Part 2 (props): does R1 ~= linear (prototype viable -> drop the classifier)?")
    print("Do intra/nn1/kappa/prank/mean-k/leverage cues suggest a cheaper query or")
    print("a lighter spectral filter (Lanczos top-k)?")

if __name__=="__main__": main()
