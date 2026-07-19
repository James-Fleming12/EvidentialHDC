# Preliminary Methods (Mathematical Formalization)

This document formalizes the continuous soft-gating weights $W(x)$ assigned to incoming target samples $x$ during the test-time adaptation (TTA) update phase. Let $\hat{y}$ be the predicted pseudo-label, and let $W_{base}(x)$ be the baseline soft-weight (typically derived from the softmax confidence or cosine similarity).

---

### 1. Prototype Cosine (Baseline)
The standard pseudo-labeling baseline. It relies solely on the initial similarity between the latent feature $x$ and the HDC prototypes $\mathcal{P}$.
$$ W(x) = \text{Softmax}\left( \frac{\cos(x, \mathcal{P})}{T} \right)_{\hat{y}} $$
*Flaw: High susceptibility to confirmation bias and outlier shattering.*

---

### 2. Epistemic Density
Gates updates based on the Euclidean distance from $x$ to its predicted class's source latent mean $\mu_{\hat{y}}$, scaled by the source density's standard deviation $\sigma_{density}$.
$$ D(x, \mu_{\hat{y}}) = \|x - \mu_{\hat{y}}\|_2 $$
$$ W(x) = W_{base}(x) \cdot \exp\left( -\ln(2) \frac{D(x, \mu_{\hat{y}})}{\sigma_{density}} \right) $$
*Mechanism: Imposes an exponential decay on points that fall outside the dense "core" of the source distribution. Highly robust against out-of-distribution (OOD) noise.*

---

### 3. Balanced Margin
A class-level regularization technique to prevent common classes (e.g., Road) from receiving vastly more updates than rare classes (e.g., Bicyclist), which causes Voronoi shattering.
Let $M(x) = P_{top1}(x) - P_{top2}(x)$ be the prediction margin.
$$ W(x) = 
\begin{cases} 
0 & \text{if } M(x) > \tau \text{ with probability } p \\
W_{base}(x) & \text{otherwise}
\end{cases} $$
*Mechanism: Probabilistically drops highly confident points (which are overwhelmingly from common classes), effectively throttling the update rate of majority classes to allow minority classes to maintain their geometric volume.*

---

### 4. Balanced Epistemic Density
The sequential application of **Balanced Margin** followed by **Epistemic Density**.
$$ W'(x) = \text{BalancedMargin}(W_{base}(x)) $$
$$ W(x) = W'(x) \cdot \exp\left( -\ln(2) \frac{D(x, \mu_{\hat{y}})}{\sigma_{density}} \right) $$
*Mechanism: Prevents inter-class washout (via balancing) while simultaneously preventing intra-class outlier degradation (via density decay).*

---

### 5. Temporal Veto
Ensures structural consistency by requiring the current prediction $\hat{y}_t(x)$ to match the prediction of its nearest spatial neighbor from the previous frame $\hat{y}_{t-1}(x_{nn})$.
$$ W(x) = W_{base}(x) \cdot \mathbb{I}\left[ \hat{y}_t(x) = \hat{y}_{t-1}(x_{nn}) \right] $$
*Mechanism: Binary veto that completely zeros out the update weight if the point is structurally transient (e.g., a snowflake passing through the air). Highly effective against chaotic frame-by-frame noise.*

---

### 6. Subcluster Ledger (Geometric Rebalancing) - *Deprecated*
Attempts to perform intra-class balancing by defining $K$ subclusters $\mu_{\hat{y}, k}$ via K-Means. It tracks the number of admitted updates $N_{\hat{y}, k}$ for each subcluster.
Let $k^*$ be the nearest subcluster to $x$.
$$ W(x) = W_{base}(x) \cdot \mathbb{I}\left[ (N_{\hat{y}, k^*} - \min_j(N_{\hat{y}, j})) \leq \text{margin} \right] $$
*Flaw: Acts as an outlier amplifier. Because dense "core" subclusters instantly hit the budget margin, they are frozen ($\mathbb{I}[\cdot] = 0$). The network is then forced to adapt using only the sparse, noisy, outlier subclusters.*

---

### 7. Epistemic Multi-RP (Ensemble Consensus)
Projects the 128D latent space $x$ into 10,000D HDC space using $M$ different random projection matrices $\Phi_m$. 
$$ \hat{y}_m = \text{argmax}_c \cos(\Phi_m(x), \mathcal{P}_{m, c}) $$
$$ W(x) = W_{base}(x) \cdot \left( \frac{1}{M} \sum_{m=1}^M \mathbb{I}[ \hat{y}_m = \hat{y} ] \right) $$
*Mechanism: The update weight is scaled proportionally to the consensus of the HDC ensemble. High-variance OOD points will scatter across the projections, resulting in low consensus and heavily penalized weights.*
