# Preliminary Methods (Mathematical Formalization)

## Problem Setting
Let $x \in \mathbb{R}^{128}$ be the continuous latent feature vector extracted by the neural bottleneck for a given point. Let $\mathcal{P} \in \mathbb{R}^{K \times 10,000}$ be the fixed HDC prototype matrix mapping to $K$ semantic classes. Let $f(x)$ be the encoding function that projects $x$ into the 10,000D symbolic hyperdimensional space.

During unsupervised Test-Time Adaptation (TTA), the network is optimized via pseudo-labeling. We seek to generate a continuous confidence weight $W(x) \in [0, 1]$ that controls the magnitude of the gradient step applied to the network's latent classification boundary. 

**The Objective:** Maximize the gradient step on true geometric domain shifts while mathematically vetoing Out-of-Distribution (OOD) noise, structurally corrupted artifacts, and semantic collisions.

---

## The Final Preliminary Architecture (The 4-Tier Pipeline)

Following extensive ablations, we established that Spatial and Energy-based OOD metrics are fundamentally incompatible with structural point cloud adaptation (they falsely veto missing geometry). The definitive, mathematically pure pipeline relies on explicitly calibrated dual-space anchoring (128D + 10,000D), frequency dampening, and prototype drift tracking.

The final test-time update weight is defined as:
$$ W_{final}(x) = W_{freq}(x) \cdot W_{net}(x) \cdot W_{HDC}(x) $$

### 1. Frequency Protection (Deterministic Soft-Dampening)
**Problem:** High-frequency classes (like Road or Vegetation) generate overwhelming confident pseudo-labels, structurally crushing the feature space of rare classes (Majority Class Paralysis).
**Formulation:** Let $f_k$ be the exponential moving average of the frequency of class $k$. We scale updates using an inverse-frequency dampener:
$$ W_{freq}(x) = \left( \frac{\min_{k} f_k}{f_{\hat{y}}} \right)^\gamma $$
where $\gamma = 0.1$ is the dampening factor. This mathematically throttles majority classes without freezing them completely.

### 2. Network Uncertainty (128D Euclidean Manifold Density)
**Problem:** Standard pseudo-labeling assumes all confident points belong to the target class, ignoring the underlying manifold geometry and admitting severe structural outliers.
**Formulation:** We compute the $Z$-score of the latent feature $x \in \mathbb{R}^{128}$ relative to the source distribution's pre-computed centroid $\mu_{128}^{(c)}$ and standard deviation $\sigma_{128}^{(c)}$ for the predicted class $c = \hat{y}$.
$$ Z_{128}(x) = \frac{\|x - \mu_{128}^{(c)}\|_2}{\sigma_{128}^{(c)}} $$
$$ W_{net}(x) = \exp\left( -\lambda_{net} \cdot \text{ReLU}(Z_{128}(x) - \tau_{net}) \right) $$
This Tier 1 gate guarantees the point resides on the continuous Euclidean semantic manifold before updating.

### 3. Hypervector Uncertainty (10,000D Calibrated Dirichlet Epistemic Density)
**Problem:** Cosine similarities in 10,000D space occupy a highly concentrated, narrow band. Raw Dirichlet evidence generation squashes all inputs to $\approx 1.0$, creating representation shrinkage where safe samples are over-rejected.
**Formulation:** We standardize the cosine similarities $s_c = \cos(f(x), \mathcal{P}_c)$ using the pre-computed source statistics $\mu_{\cos}^{(c)}$ and $\sigma_{\cos}^{(c)}$:
$$ Z_{10k}(x, c) = \frac{s_c - \mu_{\cos}^{(c)}}{\sigma_{\cos}^{(c)}} $$
This $Z$-score is passed through a $\text{Softplus}$ function to generate raw Dirichlet evidence, native to the HDC symbolic space:
$$ e_c = \text{Softplus}(\kappa \cdot Z_{10k}(x, c)) $$
$$ S = \sum_{c=1}^K (e_c + 1), \quad u(x) = \frac{K}{S} $$
$$ W_{HDC}(x) = \exp\left( -\lambda_{HDC} \cdot \text{ReLU}(u(x) - \tau_{u}) \right) $$
If a point lacks strong similarity to any prototype, total evidence $S$ is low, driving epistemic uncertainty $u(x)$ up and instantly vetoing the update.

### 4. Temporal Uncertainty (Latent Prototype Drift Tracking)
**Problem:** A magnitude throttle slows down the accumulation of persistent wrong-label bias (like specular reflections on wet ground), but eventually, the bias still wins. 
**Formulation:** We physically anchor the optimization trajectory by tracking the angular drift of the target prototype $w_t^{(c)}$ away from its original frozen source initialization $w_0^{(c)}$.
$$ \Delta \theta_c = \arccos\left( \frac{w_t^{(c)} \cdot w_0^{(c)}}{\|w_t^{(c)}\| \|w_0^{(c)}\|} \right) $$
If $\Delta \theta_c$ exceeds a hard rotation budget (e.g., $40^\circ$), the step magnitude is zeroed. Additionally, a continuous anchor spring $k_{spring}$ gently pulls the prototype back to its origin after every update step:
$$ w_{t+1}^{(c)} = \text{Normalize}\left( (1 - k_{spring}) w_t^{(c)} + k_{spring} w_0^{(c)} \right) $$
This acts as a definitive backstop against the slow, high-mass accumulation of false-positive reflection noise.

---

## Deprecated / Legacy Methods
*These methods were rigorously tested but ultimately failed due to specific mathematical or structural vulnerabilities.*

### HDC-Energy Gating
Calculates the Free Energy natively in the 10,000D HDC space.
$$ E(x) = -T \cdot \log \sum_{c=1}^K \exp\left( \frac{\cos(f(x), \mathcal{P}_c)}{T} \right) $$
*Flaw:* Categorizes structural corruptions (like missing LiDAR scan lines) as severe OOD noise, preventing the network from adapting its prototypes to recognize the new, sparser geometry.

### Orthogonal Noise Detection (ViM / Spatial Gating)
Measures spatial and structural noise by computing the orthogonal residual norm of a point relative to the principal semantic subspace.
$$ \mathbf{Q}, R = \text{qr}(\mathcal{P}^T), \quad x_{\parallel} = x \mathbf{Q} \mathbf{Q}^T, \quad \text{Residual}(x) = \|x - x_{\parallel}\|_2 $$
*Flaw:* Instantly vetoes any geometry that falls into the 10,000D null space. Like Energy, this perfectly isolates geometric noise but completely paralyzes adaptation to true structural shifts like Beam Missing or Cross-Sensor density changes.

### Subcluster Ledger (Geometric Rebalancing)
Attempts to perform intra-class balancing by defining $K$ subclusters and freezing dense core clusters to wait for sparse fringe clusters.
*Flaw:* Acts as a massive outlier amplifier. By freezing the core updates, it forced the network to adapt its prototypes toward heavily penalized fringe noise.
