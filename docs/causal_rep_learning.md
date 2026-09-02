# Causal representation learning: what we can use from CorrelationVSCausal-empirical

Status: assessment. Source:
`/home/james/Research/Theory/CorrelationVSCausal-empirical` (an empirical
validation of PAC/correlation vs causal learning theory on synthetic SCMs, with
four sections: in-distribution correlation bounds, causal effect bounds, the
head-to-head, and distribution-shift theory).

## Context

### What the repository shows

Every experiment runs on synthetic SCMs with known ground-truth causal effects and
counterfactuals, so each theory is tested against the exact quantity it claims to
bound. The four sections establish a clean division of labour:

1. **Correlation (PAC) theory** bounds in-distribution *prediction* error
   (Rademacher). It holds on i.i.d. data but is **silent under shift**.
2. **Causal theory** (Shalit-Johansson-Sontag ITE bound) adds an IPM term between the
   treated and control distributions that is exactly the confounding cost the
   Rademacher bound structurally ignores. The IPM grows monotonically with
   confounding and predicts the causal estimator's degradation.
3. **Head-to-head**: the factual (correlation) model predicts well and its Rademacher
   bound holds, yet its reading of the causal effect is badly biased (+0.386 ATE
   bias under confounding), and only the IPM flags it. The two theories certify
   different quantities; they are not in conflict.
4. **Under shift**, three theories operate and each has documented boundaries:
   the **Ben-David divergence bound** quantifies the damage
   (`R_T <= R_S + 1/2 d_HΔH(S,T) + lambda*`, holds with coverage 1.00);
   **transportability** recovers a causal effect whose *correlation flips sign*
   across environments (Simpson's paradox; transported do-effect 0.785 vs true
   0.800);
   **invariance** (ICP, IRM) identifies the causal feature or recovers causal-only
   performance when the spurious association flips (IRMv1 0.498 vs ERM 0.299 on
   tabular ColoredMNIST, causal-only ceiling 0.616).

### The three documented limits (directly relevant to us)

- **S4.T5: invariance fails when no observed environment carries causal signal.**
  With the causal feature uninformative in all training environments, the only
  stable predictor is the spurious one; IRMv1 does not crash (0.490 vs ERM 0.143)
  but does not recover either. Invariance is a heuristic, not a guarantee.
- **S4.T6: invariance alone cannot distinguish causal from spurious.** When the
  spurious association is stable across the *observed* environments, the invariance
  test certifies the spurious feature, and a new environment where it flips breaks
  the model (RMSE 2.389 vs causal oracle 1.404).
- **S4.T7: unmeasured confounding biases every adjustment set.** Adjusting for the
  observed confounder reduces but does not remove the bias; only the oracle set
  recovers the truth, and no estimator can detect the hidden confounder from data.

## Takeaways for our robust feature extractor (fog/crosstalk)

### 1. Causal/invariance learning is the right frame, and it is more robust under shift

The clean/clean-vs-corrupt setting is a distribution-shift problem, and the
repository's Section 4 is the empirical backing for the "invariance under the
corruption transformation" direction (DIRT-GAN in `docs/inv_rep_learning.md`, and
what DGLSS++ already approximates with GMSIFC/LSCC/SupCon). The invariant predictor
survives a flipped spurious association where ERM collapses (S4.T4). For us, the
"causal" (transportable) features are the geometry/semantics that survive
fog/crosstalk; the "spurious" features are the corruption-specific statistics (fog's
deleted remission channel, crosstalk's wrong returns). Enforcing invariance under the
corruption (aligned pairs, or the synthetic proxies) is exactly the theory-backed
objective: keep the features that are transportable, drop those that are not.

### 2. The domain-shift theory we could ADD to make it better

Four concrete additions, in increasing order of effort:

1. **Ben-David divergence as a diagnostic.** Compute `d_HΔH(clean, corrupt)` on the
   DGLSS++ features (a clean-vs-fog discriminator: `d ~ 2(1 - 2 eps*)`) per
   condition, plus `lambda*` (the minimal joint risk). This QUANTIFIES the collapse
   the way the Rademacher bound cannot, and predicts the zero-shot drop before we
   train the decoder. It directly complements the isotropy / covariance-gate
   diagnostics: high divergence on fog/crosstalk = the features are domain-separable
   = the collapse is real and measurable. We could report it alongside the map19
   numbers.
2. **IPM balancing (Shalit et al.) as a training objective.** The ITE bound says the
   clean-corrupt distribution gap is the error source. A balancing objective that
   minimizes the IPM (e.g., Gaussian MMD) between the clean and corrupt feature
   distributions is a principled robustness regularizer. This is closely related to
   what GMSIFC does indirectly; a direct MMD term between `z8(clean)` and
   `z8(corrupt)` would be the clean version (and a more defensible variant of the
   DIRT `||z8_clean - z8_corrupt||^2` idea, since the MMD is scale-robust where the
   squared error's norm-shrinking is not).
3. **Transportability re-weighting.** S4.T2 is the clean demonstration that the
   correlation flips under selection while the causal effect is recovered by
   re-weighting with the target marginal. This is exactly the propagated-mean /
   active-learning idea already measured in the AL arc (re-estimate class statistics
   under the target). The theory here certifies WHY it works and states its limit:
   it needs the target class marginal, and it cannot recover a destroyed signal.
4. **IRM-style invariance regularizer.** Penalize environment-dependence of the loss
   gradient across clean/corrupt views. This is the most explicit form of "learn the
   invariant mechanism" and would be a direct addition to the DGLSS++ trainer
   (the repo's S4.T4 shows it recovers causal-only performance where ERM collapses).

### 3. The honest limits that PREDICT our fog/crosstalk failure

The repository's S4.T5-S4.T7 map onto our measured results almost exactly, and they
set expectations before more runs:

- **S4.T5 predicts fog heavy cannot be fixed by invariance.** The recoverability
  check already showed the true class of a misclassified fog point is in the top-3
  clean prototypes only 8-13% of the time (at or below the ~19% random baseline):
  the causal (semantic) signal is effectively ABSENT in the corrupted features. When
  no environment carries causal signal, the repo shows invariance methods do not
  recover. This is the theoretical explanation for why DGLSS++/DIRT are unlikely to
  lift fog heavy's ~0.08 mIoU: the information is gone, not just re-arranged.
- **S4.T6 predicts the synthetic-proxy DIRT weakness.** If our synthetic corruption
  proxies are stable-but-proxy-specific, the invariance learned under them can
  certify the wrong (proxy-specific) features, and the real corruption flips them.
  This is the concrete reason to prefer the aligned-real-pair or learned-operator
  version (DIRT-GAN) over the hand-crafted proxy.
- **S4.T7 frames the corruption as an unmeasured confounder.** The corruption
  distortion is an unmodeled factor; a sensitivity-style analysis would quantify how
  much corruption-specific shift would explain away a claimed robustness gain. This
  is a useful framing for the paper's honesty, not a method.

## Suggested next steps

1. Add the **Ben-David divergence** (clean vs each corruption, on the DGLSS++-19
   features) as a diagnostic column next to the map19 zero-shot/ceiling numbers in
   `twenty_cls_iterations.md`. It quantifies the collapse and predicts the zero-shot
   drop.
2. Test the **IPM/MMD balancing** term (or the transportability re-weighting) as the
   robustness regularizer instead of the squared-error DIRT loss, since the theory
   says the IPM is the scale-robust version of the same objective.
3. Use **S4.T5** to set expectations: healthy conditions and crosstalk have surviving
   semantic signal (invariance can help there); fog heavy does not, and no invariance
   method will recover it.
