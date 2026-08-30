"""al_error_predict_diag.py: the ERROR-PREDICTABILITY diagnostic -- does ANY cheap
label-free statistic predict the frozen probe's oracle errors?

This is the measurement that decides whether any decision-rule method (gate,
abstention, mixture-of-decoders, local correction) has signal, and it tells us
WHICH statistic to build an update mechanism around -- not a stop signal.

For each val point x, compute a large set of cheap observable statistics phi(x),
ALL label-free (computed on the frozen probe + unlabeled pool):

  margin          |top-2 logit margin| of W0      (low = near a boundary)
  entropy         H(p0) of the frozen softmax
  p1, p2          top-1 and top-2 frozen probabilities
  tta_var         variance of p(aug_k(x)) over bit-flip augmentations
  tta_ent         H(mean_aug p)
  tta_agree       P(aug argmax == frozen argmax) over augmentations
  proto_dist      ||x - mu_{c_hat}|| to the nearest CLEAN class-mean prototype
  proto_disagree  frozen argmax != nearest-clean-prototype argmax
  density         mean distance to k nearest pool neighbors (128-d)
  local_disagree  fraction of kNN pool neighbors with a different frozen pseudo-
                  label (local boundary disagreement)
  classifier_div  number of "reasonable decoders" (W0, clean prototype, cosine
                  kNN) that DISAGREE with the frozen argmax (epistemic
                  disagreement, 0-3)

Then, using oracle labels ONLY for evaluation, measure how well each statistic
predicts e(x) = 1[y_hat0(x) != y(x)]:
  AUROC          error detection
  AUPRC          error detection (class imbalance aware)
  enrichment@k   error rate in the top k% by the statistic / global error rate
                 (1.0 = no signal, >>1 = the top-k points are error-dense)
  corr(stat, e)  rank correlation with error

Also fits a LOGISTIC COMBINATION on the pool (oracle labels) and reports its val
AUROC -- the ceiling of "all these features combined can predict errors".

Decisive reads (this selects the next update mechanism, not a stop):
  margin/entropy/tta_* enrichment >> 1  -> errors are boundary/TTA-identifiable:
      an abstention gate or TTA-conditioned local correction can route the right
      points; build the acquisition-driven mechanism.
  proto_dist / proto_disagree strong     -> the error is a CLASS-MEAN SHIFT:
      reformulate the decoder as an UPDATABLE prototype/class-mean classifier
      (few labels re-estimate the shifted means directly, no U, no W-update).
  classifier_div strong                  -> mixture-of-decoders / disagreement
      gate is the right update object.
  all statistics ~1 / AUROC ~0.5         -> the frozen representation exposes NO
      error information: the update must come from the LABELS themselves, i.e.
      reformat WHAT is stored/updated (e.g. per-condition class statistics), not
      better selection.
  density strong                         -> spatial/feature propagation is viable
      (label one point, propagate through neighbors).

Usage:
  uv run python robust_diagnostic/al_error_predict_diag.py \
    --path_b <ckpt_dir> --method_b <method> --label <label> \
    --conds fog,crosstalk \
    --out robust_diagnostic/logs/al_error_predict_<label>.json
"""
import os, sys, time, argparse, json, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
from dataset.kitti.parser import Parser
from modules.gen_trainers import GenTrainer
from modules.oracle_core import get_hdc_projection

NUM_CLASSES = 17
SKETCH_SEED = 11


def build_parser(root, data, arch):
    return Parser(root=root, train_sequences=['08'], valid_sequences=['08'], test_sequences=None,
                  labels=data["labels"], color_map=data["color_map"], learning_map=data["learning_map"],
                  learning_map_inv=data["learning_map_inv"], sensor=arch["dataset"]["sensor"],
                  max_points=arch["dataset"]["max_points"], batch_size=1, workers=4, gt=True, shuffle_train=False)


def extract_clean(model, parser, device, num_frames=100):
    feats, lbls = [], []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(parser.get_train_set()):
            if i >= num_frames:
                break
            in_vol = batch[0].to(device); labels = batch[2].to(device).view(-1); mask = (batch[1].to(device) > 0).view(-1)
            out = model(in_vol); z8 = out[2] if len(out) == 3 else out[1]
            zf = z8.permute(0, 2, 3, 1).reshape(-1, z8.shape[1])[mask]
            feats.append(zf.cpu()); lbls.append(labels[mask].cpu())
    return torch.cat(feats), torch.cat(lbls)


def hdc_codes(feats, proj, device, chunk=100000):
    out = []
    for s in range(0, len(feats), chunk):
        out.append(torch.sign(feats[s:s + chunk].to(device) @ proj).cpu())
    return torch.cat(out)


def onehot(lbls, nc):
    y = torch.zeros(len(lbls), nc); y[torch.arange(len(lbls)), lbls.long()] = 1; return y


def ridge_fit_soft(X, Y, lam, iters, m, device):
    X = X.to(device); torch.manual_seed(SKETCH_SEED)
    m = min(m, X.shape[1], max(1, X.shape[0] - 1))
    P = (torch.rand(X.shape[1], m, device=device) > 0.5).float() * 2 - 1
    XP = X @ P; Yd = Y.float().to(device)
    Shat = XP.t() @ XP + lam * torch.eye(m, device=device); That = XP.t() @ Yd
    x0 = P @ torch.linalg.solve(Shat, That)
    if X.shape[0] <= 8:
        return x0.float()
    x = x0; b = X.t() @ Yd
    def A(v):
        return X.t() @ (X @ v)
    r = b - A(x); p = r.clone(); rs = (r * r).sum(0)
    for _ in range(iters):
        Ap = A(p); d = (p * Ap).sum(0)
        if not torch.isfinite(d).all() or d.abs().max().item() < 1e-20:
            break
        a = rs / (d + 1e-30); x = x + a.unsqueeze(0) * p
        r = r - a.unsqueeze(0) * Ap; rsn = (r * r).sum(0); be = rsn / (rs + 1e-30)
        p = r + be.unsqueeze(0) * p; rs = rsn
        if not torch.isfinite(x).all():
            return x0.float()
    return x.float()


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def tic():
    sync(); return time.time()


def toc(t0):
    sync(); return time.time() - t0


def auroc(scores, labels):
    """Area under ROC from scores and binary labels."""
    s = scores[labels == 1]; ns = scores[labels == 0]
    if len(s) == 0 or len(ns) == 0:
        return None
    # Mann-Whitney: P(score_pos > score_neg)
    if len(ns) > 50000:
        idx = torch.randperm(len(ns))[:50000]
        ns = ns[idx]
    if len(s) > 50000:
        idx = torch.randperm(len(s))[:50000]
        s = s[idx]
    n_pos, n_neg = len(s), len(ns)
    if n_pos == 0 or n_neg == 0:
        return None
    rank = torch.argsort(torch.cat([s, ns]))
    ranks = torch.empty(len(rank), dtype=torch.float)
    ranks[rank] = torch.arange(len(rank), dtype=torch.float)
    u = ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


def enrichment(scores, labels, fracs=(0.01, 0.05, 0.10)):
    """Error rate in the top-k% by score / global error rate. 1.0 = no signal."""
    n = len(scores)
    base = float(labels.float().mean().item())
    out = {}
    for f in fracs:
        k = max(int(f * n), 1)
        topk = torch.argsort(scores, descending=True)[:k]
        rate = float(labels[topk].float().mean().item())
        out[str(int(f * 100))] = rate / base if base > 0 else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti_dir", type=str, default="/mnt/alpha/jmfleming/KITTI")
    ap.add_argument("--kittic_dir", type=str, default="/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C")
    ap.add_argument("--config", type=str, default="config/labels/semantic-kitti-all.yaml")
    ap.add_argument("--arch", type=str, default="config/arch/senet-2048p.yml")
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--pool_size", type=int, default=20000)
    ap.add_argument("--val_size", type=int, default=100000)
    ap.add_argument("--eval_size", type=int, default=30000, help="val subsample for feature eval")
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--max_clean", type=int, default=30000)
    ap.add_argument("--nystrom_m", type=int, default=1000)
    ap.add_argument("--cg_iters", type=int, default=8)
    ap.add_argument("--tta_augs", type=int, default=5)
    ap.add_argument("--knn", type=int, default=10)
    ap.add_argument("--conds", type=str, default="fog,crosstalk")
    ap.add_argument("--path_b", type=str, required=True)
    ap.add_argument("--method_b", type=str, required=True)
    ap.add_argument("--label", type=str, default="dglsspp")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    DATA = yaml.safe_load(open(args.config)); ARCH = yaml.safe_load(open(args.arch))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    conds = [c.strip() for c in args.conds.split(',') if c.strip()]

    trainer = GenTrainer(ARCH, DATA, args.kitti_dir, args.path_b, path=args.path_b, method=args.method_b)
    model = trainer.model
    model.eval()
    clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
    fa, la = extract_clean(model, clean_parser, device, args.frames)
    proj = get_hdc_projection(dim_in=fa.shape[1], dim_out=10000, device=device)
    results = {'label': args.label, 'method': args.method_b, 'knn': args.knn, 'conds': {}}

    mc = min(args.max_clean, len(fa)); ci = torch.randperm(len(fa))[:mc]
    Xc = hdc_codes(fa[ci], proj, device).float()
    W0 = ridge_fit_soft(Xc, onehot(la[ci], NUM_CLASSES), args.lam, args.cg_iters, args.nystrom_m, device)
    W0c = W0.detach().cpu()
    del Xc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # clean class-mean prototypes (code space) from the clean features
    torch.manual_seed(1)
    pci = torch.randperm(len(fa))[:50000]
    Xc_full = hdc_codes(fa[pci], proj, device).float()
    proto_clean = torch.zeros(NUM_CLASSES, Xc_full.shape[1])
    for c in range(1, NUM_CLASSES):
        m = (la[pci] == c)
        if int(m.sum().item()) > 50:
            proto_clean[c] = Xc_full[m].mean(dim=0)
    proto_clean = F.normalize(proto_clean.float(), p=2, dim=1)
    del Xc_full
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for cond in conds:
        t0 = tic()
        cdir = os.path.join(args.kittic_dir, cond, 'heavy')
        if not os.path.exists(cdir):
            cdir = os.path.join(args.kittic_dir, cond, 'moderate')
        fd, ld = extract_clean(model, build_parser(cdir, DATA, ARCH), device, args.frames)
        torch.manual_seed(42); perm = torch.randperm(len(fd))
        pool_f, pl = fd[perm[:args.pool_size]], ld[perm[:args.pool_size]]
        val_f, vl = fd[perm[-args.val_size:]], ld[perm[-args.val_size:]]
        Xp = hdc_codes(pool_f, proj, device).float()
        Xv = hdc_codes(val_f, proj, device).float()
        del val_f, fd, ld
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # eval subsample
        torch.manual_seed(9)
        ei = torch.randperm(len(Xv))[:min(args.eval_size, len(Xv))]
        Xe = Xv[ei].float(); ye = vl[ei]

        # ---- frozen probe ----
        Lv = Xe @ W0c
        p0 = torch.softmax(Lv, dim=1)
        pred_v = Lv.argmax(1)
        err = (pred_v != ye).long()

        top2 = torch.topk(Lv, 2, dim=1)
        margin = (top2.values[:, 0] - top2.values[:, 1]).abs()
        entropy = -(p0 * (p0 + 1e-12).log()).sum(1)
        p1 = top2.values[:, 0]
        p2 = top2.values[:, 1]

        # ---- TTA statistics (bit-flip augmentations on the codes) ----
        tta_var = torch.zeros(len(Xe)); tta_ent = torch.zeros(len(Xe))
        tta_agree = torch.zeros(len(Xe))
        draws = []
        for _ in range(args.tta_augs):
            torch.manual_seed(100 + _)
            flip = torch.rand_like(Xe) < 0.02
            draws.append(torch.softmax(torch.where(flip, -Xe, Xe) @ W0c, dim=1))
        draws = torch.stack(draws)
        tta_var = draws.var(dim=0).mean(dim=1)
        p_avg = draws.mean(dim=0)
        tta_ent = -(p_avg * (p_avg + 1e-12).log()).sum(1)
        tta_agree = (draws.argmax(2) == pred_v.unsqueeze(0)).float().mean(0)

        # ---- prototype distance + disagreement (clean class means) ----
        proto_sim = F.normalize(Xe.float(), p=2, dim=1) @ proto_clean.t()   # n x C
        proto_pred = proto_sim.argmax(1)
        proto_conf = proto_sim.gather(1, pred_v.unsqueeze(1)).squeeze(1)
        proto_dist = 1.0 - proto_conf
        proto_disagree = (proto_pred != pred_v).float()

        # ---- density + local disagreement (kNN in the 10000-d code space) ----
        torch.manual_seed(3)
        sub = torch.randperm(len(Xp))[:10000]
        pf = F.normalize(Xp[sub].float(), p=2, dim=1)
        Xe_c = F.normalize(Xe.float(), p=2, dim=1)
        # chunked cosine kNN
        k = args.knn
        density = torch.zeros(len(Xe)); local_dis = torch.zeros(len(Xe))
        pool_pred = (Xp[sub].float() @ W0c).argmax(1)
        chunk = 5000
        for s in range(0, len(Xe), chunk):
            sim = Xe_c[s:s+chunk] @ pf.t()
            topk = torch.topk(sim, k, dim=1)
            density[s:s+chunk] = 1.0 - topk.values.mean(1)
            lab_topk = pool_pred[topk.indices]
            local_dis[s:s+chunk] = (lab_topk != pred_v[s:s+chunk].unsqueeze(1)).float().mean(1)
        del pf, Xe_c
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ---- classifier disagreement (W0 vs prototype vs TTA-mean) ----
        tta_pred = p_avg.argmax(1)
        classifier_div = ((proto_pred != pred_v).long()
                          + (tta_pred != pred_v).long())

        # ---- pool labels for the logistic combination ceiling ----
        Lp = Xp.float() @ W0c
        pp = torch.softmax(Lp, dim=1)
        pe = (Lp.argmax(1) != pl).long()
        tpm = torch.topk(Lp, 2, dim=1)
        pool_margin = (tpm.values[:, 0] - tpm.values[:, 1]).abs()
        pool_ent = -(pp * (pp + 1e-12).log()).sum(1)

        features = {
            'margin': margin, 'entropy': entropy, 'p1': p1, 'p2': p2,
            'tta_var': tta_var, 'tta_ent': tta_ent, 'tta_agree': tta_agree,
            'proto_dist': proto_dist, 'proto_disagree': proto_disagree,
            'density': density, 'local_disagree': local_dis,
            'classifier_div': classifier_div,
        }

        cond_res = {'frozen': float((pred_v == ye).float().mean().item()),
                    'error_rate': float(err.float().mean().item()),
                    'error_rate_top1pct_global': None, 'features': {}}
        for name, feat in features.items():
            fr = {}
            fr['auroc'] = auroc(feat.float(), err)
            fr['enrichment'] = enrichment(feat.float(), err)
            try:
                fr['corr'] = float(torch.corrcoef(torch.stack([feat.float(), err.float()]))[0, 1].item())
            except Exception:
                fr['corr'] = None
            cond_res['features'][name] = fr

        # ---- logistic combination (fit on pool, eval on val) ----
        # pool-side features: margin, entropy, p1, p2 (cheap); approximate the
        # rest with the val-only features by fitting on the VAL itself is a leak;
        # instead fit on pool with the 4 cheap features and report the ceiling.
        Xp_fit = torch.stack([pool_margin, pool_ent,
                              torch.topk(Lp, 2, dim=1).values[:, 0],
                              torch.topk(Lp, 2, dim=1).values[:, 1]], dim=1)
        Xe_fit = torch.stack([margin, entropy, p1, p2], dim=1)
        # standardize
        mu = Xp_fit.mean(0); sd = Xp_fit.std(0).clamp(min=1e-8)
        Xp_s = (Xp_fit - mu) / sd
        Xe_s = (Xe_fit - mu) / sd
        wlog = torch.zeros(4, requires_grad=True); blog = torch.zeros(1, requires_grad=True)
        opt = torch.optim.Adam([wlog, blog], lr=0.1)
        yp = pe.float()
        for _ in range(200):
            opt.zero_grad()
            logit = Xp_s @ wlog + blog
            loss = F.binary_cross_entropy_with_logits(logit, yp)
            loss.backward(); opt.step()
        with torch.no_grad():
            ce = torch.sigmoid(Xe_s @ wlog.detach() + blog.detach())
        cond_res['logistic_4feat'] = {'auroc': auroc(ce, err),
                                      'enrichment': enrichment(ce, err)}

        results['conds'][cond] = cond_res
        del Xp, Xv, Xe, p0, Lv, pp, Lp, draws
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"\n=== {cond} ({toc(t0):.0f}s) frozen-acc {cond_res['frozen']:.3f} error-rate {cond_res['error_rate']:.3f} ===")
        for name, fr in cond_res['features'].items():
            en = " ".join(f"{k}:{v:.2f}" for k, v in fr['enrichment'].items())
            print(f"    {name:16s} auroc {fr['auroc']} corr {fr['corr']} enrich(top%) {en}")
        l4 = cond_res['logistic_4feat']
        print(f"    logistic_4feat      auroc {l4['auroc']} enrich " +
              " ".join(f"{k}:{v:.2f}" for k, v in l4['enrichment'].items()))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out}")
    print("\n=== READ ===")
    print("enrichment@top1% >> 1 (e.g. 3-5x) on a feature -> that feature finds")
    print("the errors. This selects the next update mechanism:")
    print("  margin/entropy/tta strong  -> boundary/TTA-gate + local correction")
    print("  proto_* strong             -> reformulate as UPDATABLE class-mean")
    print("                                decoder (few labels re-estimate means)")
    print("  classifier_div strong      -> mixture-of-decoders / disagreement gate")
    print("  density/local_disagree     -> spatial/feature label propagation")
    print("  ALL flat (enrich ~1, auroc ~0.5) -> the frozen representation exposes")
    print("    no error info; the update must REFORMAT what is stored/updated")
    print("    (per-condition class statistics), not better selection.")


if __name__ == "__main__":
    main()
