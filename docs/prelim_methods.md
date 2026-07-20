# Preliminary Methods (Mathematical Formalization)

This document formalizes the continuous soft-gating weights $W(x)$ assigned to incoming target samples $x$ during the test-time adaptation (TTA) update phase. Let $\hat{y}$ be the predicted pseudo-label, and let $W_{base}(x)$ be the baseline soft-weight (typically derived from the softmax confidence or cosine similarity).

---

### Baseline: Prototype Cosine
The standard pseudo-labeling baseline. It relies solely on the initial similarity between the latent feature $x$ and the HDC prototypes $\mathcal{P}$.
$$ W(x) = \text{Softmax}\left( \frac{\cos(x, \mathcal{P})}{T} \right)_{\hat{y}} $$
*Flaw: High susceptibility to confirmation bias and outlier shattering.*

---

## 1. HDC Uncertainty (OOD & Geometric Density)
These methods aim to gate updates by modeling the density and geometric layout of the high-dimensional latent space.

### Epistemic Density (128D Bottleneck) - *Legacy*
Gates updates based on the Euclidean distance from $x$ to its predicted class's source latent mean $\mu_{\hat{y}}$, scaled by the source density's standard deviation $\sigma_{density}$.
$$ D(x, \mu_{\hat{y}}) = \|x - \mu_{\hat{y}}\|_2 $$
$$ W(x) = W_{base}(x) \cdot \exp\left( -\ln(2) \frac{D(x, \mu_{\hat{y}})}{\sigma_{density}} \right) $$
*Mechanism: Imposes an exponential decay on points outside the dense "core" of the source distribution.*

### HDC-Energy Gating (`energy_density`) - *Current*
Replaces Euclidean density with Free Energy calculated natively in the 10,000D HDC space.
$$ E(x) = -T \cdot \log \sum_{c=1}^K \exp\left( \frac{\cos(x, \mathcal{P}_c)}{T} \right) $$
$$ W(x) = W_{base}(x) \cdot \exp\left( - \lambda \cdot \text{ReLU}(E(x) - \mu_{E}) \right) $$
*Mechanism: LogSumExp preserves the geometric magnitude of the 10,000D vectors. Samples with higher (less negative) energy than the moving average $\mu_{E}$ are aggressively throttled.*

### Balanced Margin - *Inter-class Balancing*
A class-level regularization technique to prevent common classes from shattering rare classes.
Let $M(x) = P_{top1}(x) - P_{top2}(x)$ be the prediction margin.
$$ W(x) = 
\begin{cases} 
0 & \text{if } M(x) > \tau \text{ with probability } p \\
W_{base}(x) & \text{otherwise}
\end{cases} $$
*Mechanism: Probabilistically drops highly confident points (which are overwhelmingly from common classes).*

### Subcluster Ledger (Geometric Rebalancing) - *Deprecated*
Attempts to perform intra-class balancing by defining $K$ subclusters.
*Flaw: Acts as an outlier amplifier, freezing the core and forcing the network to adapt using only sparse, noisy subclusters.*

---

## 2. Network Uncertainty (Epistemic / Predictive)
These methods measure the network's predictive confidence to natively formulate an "I don't know" threshold.

### Epistemic Multi-RP (Ensemble Consensus) - *Legacy*
Projects the latent space into $M$ different random HDC spaces to measure consensus.
$$ W(x) = W_{base}(x) \cdot \left( \frac{1}{M} \sum_{m=1}^M \mathbb{I}[ \hat{y}_m = \hat{y} ] \right) $$
*Flaw: Multi-RP projections slightly over-regularize the signal, dropping optimal accuracy.*

### Dirichlet Evidential Gating (`dirichlet_density`) - *Current*
Treats HDC cosine similarities as raw evidence to parameterize a Dirichlet distribution (True Evidential Deep Learning).
$$ e_c = \text{Softplus}(\tau \cdot \cos(x, \mathcal{P}_c)) $$
$$ \alpha_c = e_c + 1, \quad S = \sum_{c=1}^K \alpha_c $$
$$ u(x) = \frac{K}{S} $$
$$ W(x) = W_{base}(x) \cdot \exp\left( -\lambda \cdot \text{ReLU}(u(x) - \tau_u) \right) $$
*Mechanism: If a point lacks strong similarity to any prototype, total evidence $S$ is low, driving epistemic uncertainty $u(x)$ up and instantly vetoing the update.*

---

## 3. Temporal Uncertainty (Optimization Trajectory)
These methods track structural consistency or gradient flow over time to filter out chaotic noise.

### Temporal Veto - *Legacy*
A hard binary mask requiring spatial neighbors from frame $t-1$ to match frame $t$.
$$ W(x) = W_{base}(x) \cdot \mathbb{I}\left[ \hat{y}_t(x) = \hat{y}_{t-1}(x_{nn}) \right] $$
*Flaw: Point-to-point structural overlap is brittle across dynamic LiDAR sweeps.*

### Central Flow Momentum Veto (`momentum_veto`) - *Current*
A continuous formulation that tracks the Exponential Moving Average (EMA) of the optimization trajectory (the "central flow").
$$ L_{EMA}^{(t)} = \beta L_{EMA}^{(t-1)} + (1-\beta) L_{curr}^{(t)} $$
$$ \text{Div}(x) = 1 - \cos(L_{curr}(x), L_{EMA}(x)) $$
$$ W(x) = W_{base}(x) \cdot \exp\left( -\lambda \cdot \text{Div}(x) \right) $$
*Mechanism: Sudden, chaotic gradient spikes (e.g., from snow reflections) sharply diverge from the EMA central flow, causing exponential learning rate decay.*

---

## 4. Spatial Uncertainty (Geometric Integrity)
These methods evaluate the structural integrity of the point cloud. Traditionally, this requires computationally prohibitive physical geometry checks (like 3D KD-Tree searches for surface normal estimation). Our method completely replaces this overhead by performing spatial integrity checks natively within the latent HDC space.

### Orthogonal Noise Detection / ViM (`orthogonal_spatial_veto`) - *Current*
Measures spatial and structural noise by computing the orthogonal residual norm of a point relative to the principal semantic subspace (Virtual-logit Matching).
$$ \mathbf{Q}, R = \text{qr}(\mathcal{P}^T) $$
$$ x_{\parallel} = x \mathbf{Q} \mathbf{Q}^T $$
$$ \text{Residual}(x) = \|x - x_{\parallel}\|_2 $$
$$ W(x) = W_{base}(x) \cdot \exp\left( -\lambda \cdot \text{Residual}(x) \right) $$
*Mechanism: In a 10,000D space, the 17 prototypes span a tiny semantic subspace. Structurally corrupted points (like LiDAR crosstalk) project massively into the orthogonal null space. This replaces a multi-million-node KD-Tree search with a single matrix multiplication, instantly vetoing physical artifacts even if their Softmax confidence is high.*

---

## 5. Paper Organization & Ensembling Strategy

The structure of the study systematically builds upon the failure of purely spatial gating mechanisms. Our framework is organized into a primary lightweight method and two targeted augmentations.

### The Basic Core Method
* **Composition:** Network Uncertainty (`dirichlet_density`) + HDC Uncertainty (`energy_density`)
* **Rationale:** Purely Spatial/Geometric gating (as seen in baseline models like D3CTTA) catastrophically fails by itself under structurally destructive corruptions like fog and motion blur. By replacing physical space with the latent hyperdimensional space, we construct a "Very Basic Method" that massively improves performance over the baseline. Because Network and HDC uncertainty are computed instantly from the initial forward pass, this core method incurs **zero additional storage and zero additional computational overhead**.

### Variant 1: Temporal Augmentation (+ Storage)
* **Composition:** Core Method + Temporal Uncertainty (`momentum_veto`)
* **Rationale:** Adds the continuous optimization trajectory (EMA). This variant provides critical resistance to chaotic, frame-by-frame volatility (e.g., snow reflections) but requires additional GPU memory/storage to maintain the central flow tensor across time.

### Variant 2: Spatial Augmentation (Latent Geometry)
* **Composition:** Core Method + Spatial Uncertainty (`orthogonal_spatial_veto`)
* **Rationale:** Traditionally, geometric integrity requires expensive physical KD-Trees. This variant introduces Orthogonal Noise Detection (ViM), completely replacing physical operations with a single highly-parallelized linear projection. It detects structural artifacts that fall into the 10,000D null space, adding robust geometric filtering with practically zero computational overhead.

### Variant 3: The Full Ensemble
* **Composition:** Network + HDC + Temporal + Spatial
* **Rationale:** The complete unified pipeline, fusing all four gates for maximum robustness across every corruption modality.
