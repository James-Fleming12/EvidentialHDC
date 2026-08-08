"""D3CTTA test-time adaptation mechanism, adapted to our 128D feature pipeline.

Extracts the mechanism-only core of D3CTTA (Liu et al., T-ITS 2023) that does NOT
require MinkowskiEngine or the Synth4D pretraining:
  - per-class entropy + probability pseudo-label selection (select_pseudo),
  - kNN-consistency filtering of the selected points (in 128D feature space),
  - per-domain random-projection ridge-classifier adaptation (the T3A-style Q/G
    accumulation and cross-validated ridge solve), accumulated online over the
    target frames.

Used by d3ctta_diag.py to diagnose whether D3CTTA's fog/crosstalk robustness comes
from its mechanism (transferable to our features) or from its backbone/pretraining
(not replicable). If the mechanism's confident pseudo-labels are wrong on our
features, that localizes the difference to the feature extractor.
"""
import math
import numpy as np
import torch
import torch.nn.functional as F


def softmax_entropy(x):
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


class D3CTTA_Decoder:
    """Online per-domain ridge-classifier adaptation over random-projected 128D features."""

    def __init__(self, num_classes, feat_dim=128, w_dim=256, alpha=0.95, tau_c=0.85,
                 select_ratio=0.05, k_consistency=20, use_consistency=True, seed=42):
        self.num_classes = num_classes
        torch.manual_seed(seed)
        self.w_rand = torch.randn(feat_dim, w_dim)
        self.alpha = alpha
        self.tau_c = tau_c
        self.select_ratio = select_ratio
        self.k_consistency = k_consistency
        self.use_consistency = use_consistency
        self.Q = []                 # per-domain [w_dim, num_classes]
        self.G = []                 # per-domain [w_dim, w_dim]
        self.domain_params = []     # {'mu', 'sigma', 't'}
        self.prev_domain = -1
        self.prev_mean = None
        self.n_adapted = 0

    def _select_pseudo(self, logits):
        ent = softmax_entropy(logits)
        prob = 1.0 - logits.softmax(1).max(1).values
        pred = logits.argmax(1)
        indices = []
        for label in range(self.num_classes):
            li = torch.nonzero(pred == label).squeeze(1)
            if li.numel() < 2:
                continue
            for sig in (ent, prob):
                vals = sig[li]
                k = math.ceil(self.select_ratio * len(li))
                indices.append(li[torch.argsort(vals)[:k]])
        return torch.cat(indices) if indices else torch.zeros(0, dtype=torch.long, device=logits.device)

    def _detect_domain(self, z):
        mu = z.mean(0)
        if self.prev_mean is None:
            self.prev_mean = mu
            self.domain_params.append({'mu': mu.clone(), 'sigma': z.var(0), 't': 1})
            self.prev_domain = 0
            return 0
        c = F.cosine_similarity(mu, self.prev_mean, dim=0)
        self.prev_mean = mu
        if c > self.tau_c:
            p = self.domain_params[self.prev_domain]
            p['mu'] = p['mu'] + (mu - p['mu']) / p['t']
            p['sigma'] = p['sigma'] + (z.var(0) - p['sigma']) / p['t']
            p['t'] += 1
            return self.prev_domain
        best, best_d = None, float('inf')
        for i, p in enumerate(self.domain_params):
            d = (torch.abs(mu - p['mu']) + torch.abs(torch.sqrt(z.var(0)) - torch.sqrt(p['sigma']))).sum()
            if d < best_d:
                best_d, best = d, i
        if best_d < self.tau_c:
            self.prev_domain = best
            return best
        self.domain_params.append({'mu': mu.clone(), 'sigma': z.var(0), 't': 1})
        self.prev_domain = len(self.domain_params) - 1
        return self.prev_domain

    def _knn_consistency(self, z, pred):
        n = len(z)
        kn = min(self.k_consistency + 1, n)
        zz = F.normalize(z, p=2, dim=1)
        # Chunked so the (anchor x all) similarity never materializes on the full
        # frame: the full-frame sim is ~19GB at 68k points and OOMs near-full GPUs.
        ok = torch.zeros(n, dtype=torch.bool, device=z.device)
        chunk = 4096
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            sim = zz[s:e] @ zz.T
            nb = torch.topk(sim, kn, dim=1).indices[:, 1:]
            ok[s:e] = (pred[nb] == pred[s:e].unsqueeze(1)).float().mean(1) > 0.8
        return ok

    def _optimise_ridge(self, F_, Y_):
        n = F_.shape[0]
        nval = int(n * 0.8)
        if nval < 1 or n - nval < 1:
            return 1e-4
        Qv = F_[:nval].T @ Y_[:nval]
        Gv = F_[:nval].T @ F_[:nval]
        best_r, best_l = 1e-4, float('inf')
        for r in 10.0 ** np.arange(-8, 9):
            rt = torch.tensor(r, dtype=torch.float64)
            W = torch.linalg.solve(Gv.double() + rt * torch.eye(Gv.size(0), dtype=torch.float64),
                                   Qv.double()).T
            loss = F.mse_loss(F_[nval:].double() @ W.T, Y_[nval:].double())
            if loss.item() < best_l:
                best_l, best_r = loss.item(), r
        return best_r

    @torch.no_grad()
    def fit_predict(self, z, logits, true_lbls=None):
        """Online step: detect domain, select confident pseudo-labels, accumulate the
        ridge classifier, and predict the current chunk.

        Returns (predictions, n_selected, selected_accuracy). selected_accuracy is the
        fraction of selected points whose pseudo-label is CORRECT (the key diagnostic:
        if D3CTTA's confident selection is wrong on our features, the difference is the
        feature extractor, not the mechanism)."""
        device = z.device
        if self.w_rand.device != device:
            self.w_rand = self.w_rand.to(device)
        domain = self._detect_domain(z)
        if len(self.Q) <= domain:
            self.Q.append(torch.zeros(self.w_rand.shape[1], self.num_classes, device=device))
            self.G.append(torch.zeros(self.w_rand.shape[1], self.w_rand.shape[1], device=device))
        Q, G = self.Q[domain], self.G[domain]
        sel = self._select_pseudo(logits)
        pred = logits.argmax(1)
        if self.use_consistency and len(sel) > 1:
            sel = sel[self._knn_consistency(z, pred)[sel]]
        feat_h = F.relu(z @ self.w_rand)
        if len(sel) > 1:
            yhat = F.one_hot(pred[sel], self.num_classes).float()
            Q.add_(feat_h[sel].T @ yhat)
            G.add_(feat_h[sel].T @ feat_h[sel])
            self.n_adapted += len(sel)
            ridge = self._optimise_ridge(feat_h[sel].double().cpu(), yhat.double().cpu())
            wo = torch.linalg.solve(G + ridge * torch.eye(G.size(0), device=device), Q).T
            sel_acc = None
            if true_lbls is not None:
                sel_acc = float((pred[sel] == true_lbls[sel]).float().mean().item())
            return (feat_h @ wo.T).argmax(1), len(sel), sel_acc
        return pred, 0, None

    def predict_adapted(self, z):
        """Predict with the current ridge classifier (no further adaptation)."""
        if self.w_rand.device != z.device:
            self.w_rand = self.w_rand.to(z.device)
        feat_h = F.relu(z @ self.w_rand)
        domain = max(self.prev_domain, 0)
        if domain < len(self.Q):
            wo = torch.linalg.solve(self.G[domain] + 1e-4 * torch.eye(self.G[domain].size(0),
                                                                      device=z.device),
                                    self.Q[domain]).T
            return (feat_h @ wo.T).argmax(1)
        return None
