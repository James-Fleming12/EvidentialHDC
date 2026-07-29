# Method Details: Domain-Adaptive Prototype Control for LiDAR Test-Time Adaptation

**Location:** `EvidentialHDC/docs/method_details.md`
**Last Updated:** July 29, 2026

This document specifies a test-time adaptation (TTA) framework for 3D LiDAR semantic segmentation under sensor corruption. The method rests on a single empirical principle established by our ceiling analysis: **the correct amount of prototype adaptation is domain-dependent, and applying a uniform adaptation policy across corruptions is strictly worse than not adapting at all.** We formalize a controller that, per input, decides *how much* to adapt and *how far* any prototype is permitted to move, and that applies an inference-time class-prior correction independently of the update.

---

## 1. Previous Methods and Motivation

Prototype-based TTA for LiDAR segmentation replaces a trained classifier head with class prototypes and updates them online from self-generated pseudo-labels. Two families dominate: **hard-gating** methods that admit only high-confidence points into the update (conformal set cardinality, top-percentage entropy ranking), and **uniform-adaptation** methods that apply a fixed learning rate to all admitted points. Both share an implicit assumption — that more confident selection and more adaptation monotonically improve robustness. Our analysis shows this assumption is false under corruption, in three specific ways that this method is designed to correct.

**Failure 1 — Uniform adaptation is net-negative because its sign is domain-dependent.** Measured across an eight-corruption panel (SemanticKITTI-C, severity 3), a uniformly adapting prototype method scores **33.28 mIoU** against a **33.68** frozen baseline: adaptation *loses* 0.40 mIoU on average. The mean hides the mechanism. On corruptions where the frozen model is already strong (wet ground 42.0, incomplete echo 43.9), sustained adaptation drags performance down (wet ground → 34.8, a −7.2 collapse); on corruptions where the frozen model is weak (crosstalk 10.7), the same adaptation *helps* (→ 15.7). The correlation between frozen quality and adaptation gain is positive: the better the starting point, the more adaptation hurts. No global learning rate, schedule, or gate threshold can satisfy both regimes simultaneously, because they require opposite actions. **This method replaces the global policy with a per-input controller.**

**Failure 2 — The damage is catastrophic prototype drift on a small number of classes, which confidence gating cannot see.** On wet ground, the −7.2 collapse is not diffuse. It is concentrated almost entirely in the majority classes whose geometry is corrupted by specular reflection: the road and building prototypes rotate on the order of tens of degrees from their source orientation, while tail-class prototypes barely move. A confidence gate does not prevent this — the reflected points are *confidently* wrong, so they pass any uncertainty threshold and drive the majority prototype into the reflection geometry. Selection quality is the wrong control surface for this failure; **the correct surface is a direct bound on prototype displacement**, which this method imposes.

**Failure 3 — Confidence gating is mis-calibrated exactly where adaptation is needed.** On the corruptions where adaptation *should* help (high feature-space uncertainty, e.g. crosstalk), a fixed-threshold gate rejects the overwhelming majority of correct points along with the incorrect ones, starving the update of the very signal that would improve the prototype. The gate is too tight precisely in the high-uncertainty regime where it needs to be loose. **This method makes gate admission a function of the domain's uncertainty rather than a fixed threshold.**

**The headroom is real and it is a control problem, not a representation problem.** An oracle that selects, per corruption, the single best action among {freeze, adapt, prior-correct} scores **36.30 mIoU (+2.62 over frozen)** — nearly double the ceiling of any single fixed policy (perfect gating alone: +1.75; perfect prior alone: +1.78). Critically, the perfect-gating and perfect-prior ceilings recover *disjoint* corruptions: prior correction recovers the reflection/echo failures without touching a prototype, while gated adaptation recovers the sparse/noisy failures. The gap between +2.62 and the −0.40 of uniform adaptation is entirely a control gap. The remainder of this document specifies a controller that closes it using only quantities already computed in the forward pass.

---

## 2. Problem Setting

We address **unsupervised online test-time adaptation** for 3D LiDAR semantic segmentation. A source-pretrained encoder $f_\theta$ and a set of class prototypes $\{\mathbf{w}_c\}_{c=1}^{C}$ are deployed against a stream of corrupted point clouds. For each frame the model must (i) predict per-point labels and (ii) optionally update its prototypes, subject to three constraints:

1. **No ground truth and no source data.** Adaptation is driven only by streaming inputs and self-generated pseudo-labels, creating a standing risk of confirmation bias.
2. **Streaming compute and memory bounds.** No replay buffers, no multi-pass optimization; updates must be $O(1)$ per frame in stored state.
3. **Severe class imbalance under structural degradation.** Majority classes outnumber tail classes by orders of magnitude, and corruption blurs the boundaries that separate them.

We evaluate under a strict one-pass protocol on SemanticKITTI-C across eight corruption axes (fog, wet ground, snow, motion blur, beam missing, crosstalk, incomplete echo, cross sensor) at severity 3, reporting final-frozen mIoU under a three-pass (frozen → adapt → frozen) measurement protocol so that reported gains reflect the adapted model state rather than transient online statistics.

---

## 3. Method Overview

For each frame, a point $x$ is embedded on the unit hypersphere, $\mathbf{z} = f_\theta(x)/\|f_\theta(x)\|_2$, and scored against normalized prototypes $\tilde{\mathbf{w}}_c$. The controller then applies four mechanisms, each keyed to quantities available in the forward pass:

1. **Inference-time prior correction** ($\S 4$): a class-prior logit adjustment applied to *predictions*, independent of any prototype update. This is the mechanism that recovers the high-frozen-quality corruptions without adaptation.
2. **Domain-gap gain control** ($\S 5$): a scalar, estimated online from mean epistemic uncertainty, that scales the prototype learning rate — high for far-from-source domains, near zero for near-source domains.
3. **Uncertainty-conditioned admission** ($\S 6$): a gate whose admission threshold loosens as domain uncertainty rises, so high-uncertainty domains contribute more update signal rather than less.
4. **Per-class rotation budget** ($\S 7$): a hard bound on the angular displacement of any prototype from its source orientation, catching catastrophic majority-class drift that the soft controls miss.

Mechanisms 2–4 govern the *update*; mechanism 1 governs *prediction* and is active even when the update is fully suppressed. This separation is deliberate: the largest single source of recoverable performance (reflection-type corruptions) is claimed at inference, with no prototype movement at all.

The pseudocode for one frame:

```
z         = normalize(f_theta(x))                     # embedding
S         = z @ W_norm.T                              # cosine similarities
# --- prediction path ---
pi_hat    = update_prior_estimate(prev_predictions)   # sec 4
logits    = kappa * S - tau * log(pi_hat)             # sec 4
y_hat     = argmax(logits)
# --- update path (optional) ---
u         = epistemic_uncertainty(S)                  # Dirichlet, sec 5
g         = gain(mean(u))                             # sec 5, in [0,1]
thresh    = fire_th * (1 - beta * u_norm)             # sec 6
admit     = pseudo_label_weight(logits) > thresh      # sec 6
W        += g * eta_0 * prototype_step(z[admit], y_hat[admit])
enforce_rotation_budget(W, W_0, cap_degrees)          # sec 7
```

---

## 4. Inference-Time Prior Correction

### 4.1 Motivation

Under corruption, feature distortion scatters points across the hypersphere. Because majority classes dominate the scene, an uncalibrated $\arg\max$ assigns a disproportionate share of that scatter to majority classes, drowning rare classes in false positives. This is a **precision** failure, not a recall failure: tail classes are over-predicted, not under-predicted. Boosting tail logits — the standard supervised long-tail remedy — amplifies the false positives and is counterproductive. The correct operation is the opposite: a prior-weighted *penalty* that raises the evidentiary bar for assigning a point to a rare class.

### 4.2 Formulation

Let $\pi_c$ be an estimate of the class prior and $\kappa$ a cosine scaling factor. By Bayes' rule the log-posterior decomposes as
$$ \log P(y=c \mid \mathbf{z}) = \log P(\mathbf{z} \mid y=c) + \log P(y=c) - \log P(\mathbf{z}). $$
Modeling the scaled cosine $\kappa\, \mathbf{z}^\top \tilde{\mathbf{w}}_c$ as the log-likelihood term and introducing a calibration coefficient $\tau$, the adjusted logit is
$$ \mathcal{L}_{c} = \kappa\, \mathbf{z}^\top \tilde{\mathbf{w}}_c + \tau \log \pi_c. $$
With $\tau > 0$ and $\pi_c \in (0,1)$ (so $\log\pi_c < 0$), rare classes receive a negative offset (we effectively subtract $|\log \pi_c|$). Because the $\arg\max$ boundary depends only on the ratio $\tau/\kappa$, this defines a scale-invariant similarity margin: a point is assigned to a rarer class only if its cosine advantage exceeds
$$ \Delta S \ge \frac{\tau}{\kappa}\,\log\!\left(\frac{\pi_{\text{common}}}{\pi_{\text{rare}}}\right), $$
which removes diffuse false-positive clouds without lowering true-positive recall. In our implementation, to match the theoretical formulation $S - \tau \log \pi_c$ with $\tau = -1.0$, this explicitly adds $\log \pi_c$. We consistently use an effective $\tau = +1.0$ in the additive formulation and $\kappa = 15.0$.

### 4.3 Online Prior Estimation

The ceiling analysis shows that substituting the *true per-domain* class prior for the source prior is the single largest recoverable gain on reflection- and echo-type corruptions (worth up to +11 mIoU on wet ground alone), and that this gain requires no prototype update — it is purely a decision-boundary correction. We therefore estimate $\pi_c$ online rather than fixing it to the source frequency:
$$ \hat{\pi}_c^{(t)} = (1-\rho)\,\hat{\pi}_c^{(t-1)} + \rho\, \frac{1}{N_t}\sum_{i} \mathbb{1}[\hat{y}_i^{(t)} = c], $$
an exponential moving average of the predicted label distribution over a sliding window, initialized at the source prior. Where the target distribution matches the source, the estimate converges back to it and the correction is unchanged; where it drifts (e.g. a scene with no rare classes present), the estimate tracks it. This estimator is bounded above by the true-prior oracle, which we report as the achievable ceiling for the mechanism.

---

## 5. Domain-Gap Gain Control

### 5.1 Motivation

The sign of the adaptation gain is predicted by how far the current domain sits from the source. A single scalar — the mean epistemic uncertainty of the frame — proxies this distance: it is low when features remain source-like (adaptation unnecessary and harmful) and high when features are broadly distorted (adaptation beneficial). Gain control converts this scalar into a global learning-rate multiplier.

### 5.2 Formulation

We derive per-point epistemic uncertainty from Dirichlet evidence. Mapping source-anchored cosine similarities through a scaled Softplus,
$$ e_c = \text{Softplus}\!\big(\gamma\,(\mathbf{z}^\top\tilde{\mathbf{w}}_c - \mu_c)/\sigma_c\big), \qquad E = \sum_c (e_c + 1), \qquad u = \frac{C}{E}, $$
where $\mu_c, \sigma_c$ are source-domain similarity statistics for class $c$. The frame-level domain-gap estimate is a running mean of $u$, and the learning-rate multiplier is a clipped linear ramp:
$$ \bar{u}^{(t)} = (1-\rho)\bar{u}^{(t-1)} + \rho\,\text{mean}_i(u_i), \qquad g^{(t)} = \text{clip}\!\left(\frac{\bar{u}^{(t)} - u_{\text{lo}}}{u_{\text{hi}} - u_{\text{lo}}},\, 0,\, 1\right). $$
The effective learning rate is $\eta = g^{(t)}\,\eta_0$. The floor $u_{\text{lo}}$ is calibrated once on clean source data as the mean uncertainty of the frozen model on in-distribution inputs; $u_{\text{hi}}$ is set at a fixed multiple of $u_{\text{lo}}$. Near-source domains yield $g \to 0$ (freeze); far-source domains yield $g \to 1$ (full adaptation).

---

## 6. Uncertainty-Conditioned Admission

### 6.1 Motivation

A fixed admission threshold is mis-matched to the corruption regime: in high-uncertainty domains it rejects almost all points, including correct ones, starving adaptation exactly where it is needed. Because gain control ($\S 5$) already suppresses adaptation in low-uncertainty domains, the admission gate can be made *looser* as uncertainty rises without risking over-adaptation in the near-source regime — the two mechanisms share a signal and act in complementary directions.

### 6.2 Formulation

Let $b_i$ be the softmax confidence of point $i$ under the prior-corrected logits, and $u_{\text{norm}} \in [0,1]$ the normalized frame uncertainty. A point is admitted to the update when
$$ b_i > \theta_{\text{eff}}, \qquad \theta_{\text{eff}} = \theta_0\,(1 - \beta\, u_{\text{norm}}), $$
so the effective threshold relaxes monotonically with domain uncertainty. $\beta = 0$ recovers a fixed threshold; larger $\beta$ admits more signal in corrupted domains. Admitted points contribute to a normalized prototype-momentum step; the step magnitude is computed over admitted points only, so admission tightness does not implicitly rescale the learning rate.

---

## 7. Per-Class Rotation Budget

### 7.1 Motivation

Gain control and admission are *soft*, frame-level controls; neither prevents a specific prototype from accumulating a catastrophic displacement over many frames when the corruption confidently distorts one class's geometry (the reflection-driven majority-class collapse of Failure 2). The rotation budget is a *hard*, per-class safety constraint that bounds cumulative angular displacement directly — the control surface that matches the failure.

### 7.2 Formulation

Let $\mathbf{w}_c^{0}$ be the source prototype for class $c$ and $\mathbf{w}_c^{(t)}$ its current value. After each update step we measure the angular displacement
$$ \phi_c^{(t)} = \arccos\!\left(\frac{\mathbf{w}_c^{0} \cdot \mathbf{w}_c^{(t)}}{\|\mathbf{w}_c^{0}\|\,\|\mathbf{w}_c^{(t)}\|}\right), $$
and if $\phi_c^{(t)}$ exceeds a budget $\Phi$, we revert that class to its pre-step value:
$$ \mathbf{w}_c^{(t)} \leftarrow \mathbf{w}_c^{(t-1)} \quad \text{if } \phi_c^{(t)} > \Phi. $$
The constraint is enforced per class, so a domain that corrupts only majority geometry (wet ground) freezes those prototypes while tail prototypes continue to adapt — the "adapt tail, freeze head" behavior that a domain-level freeze cannot express. Because reverting is strictly a projection back onto the feasible set, the constraint can never make a class worse than its last feasible state; in the limit $\Phi \to 0$ the update reduces to the frozen model. $\Phi$ is the single most important hyperparameter of the update path and is swept directly.

---

## 8. Unified Update

Combining the four mechanisms, the per-frame prototype update for class $c$ is
$$ \mathbf{w}_c^{(t+1)} = \Pi_{\Phi}\!\left[\,\mathbf{w}_c^{(t)} + g^{(t)}\,\eta_0 \sum_{i \in \mathcal{A}_c} \mathbf{z}_i\,\right], \qquad \mathcal{A}_c = \{ i : \hat{y}_i = c,\ b_i > \theta_{\text{eff}} \}, $$
where $g^{(t)}$ is the domain-gap gain ($\S 5$), $\theta_{\text{eff}}$ the uncertainty-conditioned admission threshold ($\S 6$), and $\Pi_{\Phi}$ the per-class rotation-budget projection ($\S 7$). Prediction uses the prior-corrected logits of $\S 4$ throughout, including when $g^{(t)} \to 0$ and the update is inactive.

The design is deliberately hierarchical in strength: prior correction acts always and without risk; gain control softly modulates the global rate; admission softly modulates the input set; and the rotation budget is the hard backstop. Each mechanism is keyed to a quantity already computed for prediction, so the controller adds no forward passes and no learned parameters.

---

## 9. Evaluation Protocol and Ceilings

We evaluate on SemanticKITTI-C, eight corruptions, severity 3, one-pass, reporting final-frozen mIoU (three-pass measurement) with head/mid/tail breakdowns and three-seed variance. Every claim is measured against three reference points:

| Reference | Mean mIoU | Meaning |
| :--- | :---: | :--- |
| Frozen (no adaptation) | 33.68 | the model must beat this to justify adapting at all |
| Uniform adaptation | 33.28 | the prior-art policy this method replaces (net −0.40) |
| Perfect-gating oracle | 35.43 | ceiling for admission/selection alone (+1.75) |
| Perfect-prior oracle | 35.46 | ceiling for prior correction alone (+1.78) |
| **Per-domain action oracle** | **36.30** | **ceiling for any per-input controller (+2.62)** |

The per-domain oracle is the target: it is the score of an oracle that chooses, per corruption, the single best action among {freeze, adapt, prior-correct}. The method is validated to the extent it approaches 36.30 while never falling below the 33.68 frozen baseline on any individual corruption. Each mechanism is ablated as an additive ladder (prior correction → +gain control → +admission → +rotation budget) at three seeds, so that any component contributing less than the seed-variance floor can be dropped in favor of the simpler configuration. A design goal, not an afterthought, is that if the rotation budget alone captures the majority of the recoverable headroom, the reported method is the rotation budget alone.

---

## 10. Software

The controller is implemented in `modules/HDC_utils.py`. Prediction-path prior correction and the online prior estimator are frame-local and stateless beyond the prior EMA; gain control maintains a single scalar EMA; the rotation budget maintains the source prototype matrix $\mathbf{w}^0$ and the pre-step matrix for reversion. No component requires gradients, replay buffers, or additional forward passes beyond the single embedding computed for prediction.