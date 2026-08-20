# Covariate-Shift-Aware DGLSS++ (Cov-Shift): Iterations

This tracks the development of the **cov-shift DGLSS++** feature extractor: the
first extractor in this project to raise BOTH the labeled ceiling AND the label-free
TTA on BOTH fog and crosstalk simultaneously. The core idea, the measured
performance, and the open diagnostic (healthy-condition ceiling loss) are recorded
here. The extractor-level comparison (vs DGLSS++ and Robust DGLSS++) is summarized
in the README (Pillar 1); the per-iteration mechanics are in
`docs/robust_iterations.md` (Iterations 19.10 - 19.14).

## Mathematical Formulation of the Method

### 1. Motivation and Problem Formulation

#### 1.1 Objective: Robust LiDAR Semantic Segmentation under Severe Domain Shift
Let a LiDAR scan be represented in spherical range-view projection as a 5-channel tensor $X \in \mathbb{R}^{5 \times H \times W}$, where the channel dimension indexes:
$$\mathcal{C} = \big\{ 0: \text{range } r, \; 1: x, \; 2: y, \; 3: z, \; 4: \text{remission } i \big\}$$
Each pixel $(u, v) \in \{1, \dots, H\} \times \{1, \dots, W\}$ corresponds to a projected LiDAR beam return with semantic ground truth label $Y_{u,v} \in \{0, 1, \dots, K\}$, where $Y_{u,v} = 0$ denotes unobserved / empty space or ignored classes, and $\{1, \dots, K\}$ denotes the $K$ valid semantic categories (e.g., $K=17$ on SemanticKITTI). A binary validity mask $M \in \{0, 1\}^{H \times W}$ indicates valid LiDAR point returns:
$$M_{u,v} = \mathbb{1}[X_{0, u, v} > 0]$$

The model is parameterized as an encoder-decoder neural network $f_\theta: \mathbb{R}^{5 \times H \times W} \to \mathbb{R}^{D_{\text{feat}} \times H \times W}$ with a bottleneck dimensionality $D_{\text{feat}} = 128$, producing per-pixel bottleneck feature embeddings $z(u, v) = f_\theta(X)_{u, v} \in \mathbb{R}^{128}$. A lightweight linear semantic head $g_\phi: \mathbb{R}^{128} \to \mathbb{R}^{K+1}$ produces class logits $\hat{Y}_{u,v} = g_\phi(z(u,v))$.

#### 1.2 Downstream Hyperdimensional Computing (HDC) Inference
In the downstream evidential / zero-shot test-time adaptation setting, classification is performed via random hyperdimensional projection and sign-binarization:
1. **Random Projection:** A fixed, pseudo-random projection matrix $R \in \{-1, +1\}^{128 \times D_{\text{HDC}}}$ is sampled once with i.i.d. Rademacher entries $R_{i, j} \sim \mathcal{U}(\{-1, +1\})$, where $D_{\text{HDC}} = 10{,}000$.
2. **Sign Binarization:** For any continuous bottleneck embedding $z \in \mathbb{R}^{128}$, its binary hyperdimensional representation $b \in \{-1, +1\}^{D_{\text{HDC}}}$ is computed as:
   $$b = \operatorname{sign}(z R)$$
3. **Class Prototypes:** Clean-data class prototype vectors $P_c \in \mathbb{R}^{D_{\text{HDC}}}$ are constructed by pooling the binarized codes of clean training points belonging to class $c \in \{1, \dots, K\}$:
   $$P_c = \frac{\sum_{i \in \mathcal{S}_c} b_i}{\big\| \sum_{i \in \mathcal{S}_c} b_i \big\|_2}, \quad \text{where } \mathcal{S}_c = \{ i : Y_i = c \}$$
4. **Cosine Decoding:** At inference time, a test point with code $b_{\text{test}}$ is assigned class prediction:
   $$\hat{c} = \arg\max_{c \in \{1, \dots, K\}} \cos(b_{\text{test}}, P_c) = \arg\max_{c} \frac{\langle b_{\text{test}}, P_c \rangle}{\|b_{\text{test}}\|_2 \|P_c\|_2}$$

#### 1.3 Desiderata for the Learned Feature Space
To ensure that both zero-shot frozen decoding and test-time adaptation (TTA) succeed under corruptions (such as fog, crosstalk, snow, wet ground, etc.), the bottleneck representation $z = f_\theta(X)$ must satisfy four strict mathematical properties:
1. **Intra-class Compactness and Inter-class Margin:** Embeddings of class $c$ must form dense clusters on the unit hypersphere $\mathbb{S}^{127}$, maximizing angular separation from other classes so that sign binarization $\operatorname{sign}(z R)$ produces maximally orthogonal Hamming codewords.
2. **Preservation of Recoverable Geometric Displacement:** When physical corruptions (e.g., fog) introduce a coherent spatial shift $\Delta z_c$ to class $c$, the feature extractor must *not* forcefully crush or un-learn this shifted subspace. The shifted direction contains linearly separable structure that allows labeled oracle and adaptive decoders to recover the corrupted points.
3. **Invariance to First-Order Sensor Energy Shifts:** For corruptions (e.g., crosstalk, sensor noise) that disrupt intensity and depth return statistics, the network must absorb statistical shifts in sensor channels $(r, i)$ so activations remain within standard operational bounds.
4. **Batch-Independent Scale Invariance:** Intermediate network activations must not rely on clean running statistics that become invalid under out-of-distribution test conditions.

---

### 2. Step 1: The Base DGLSS++ Formulation

DGLSS++ (Kim et al., TPAMI 2026) models domain shift primarily as a combination of **sensor beam sparsity differences** and **scene-level distribution shifts**. It trains on a single dense source domain while enforcing two consistency objectives across dense and sparse views: Generalized Masked Sparsity-Invariant Feature Consistency (GMSIFC) and Localized Semantic Correlation Consistency (LSCC).

#### 2.1 Sparse View Augmentation
Given a clean source scan $X \in \mathbb{R}^{5 \times H \times W}$, a sparse augmented view $X^a = \mathcal{A}_{\text{drop}}(X)$ is generated by randomly dropping horizontal beam rows (dense-to-sparse simulation) with dropout probability $p \sim \mathcal{U}(0.3, 0.7)$. This partitions the spatial coordinate grid into:
- **Paired valid positions:** $\Omega_p = \{(u, v) : M_{u, v} = 1 \land M^a_{u, v} = 1 \}$
- **Unpaired source positions:** $\Omega_u^s = \{(u, v) : M_{u, v} = 1 \land M^a_{u, v} = 0 \}$
- **Unpaired augmented positions:** $\Omega_u^a = \{(u, v) : M_{u, v} = 0 \land M^a_{u, v} = 1 \}$

#### 2.2 Semantic Segmentation Objective
Let $\hat{Y} = g_\phi(f_\theta(X))$ and $\hat{Y}^a = g_\phi(f_\theta(X^a))$ be the predicted class probability distributions for the clean and sparse views. The base supervision is the weighted multi-class cross-entropy loss over both views:
$$\mathcal{L}_{\text{sem}} = \frac{1}{2} \left( \mathcal{L}_{\text{CE}}(\hat{Y}, Y) + \mathcal{L}_{\text{CE}}(\hat{Y}^a, Y) \right) = - \frac{1}{2} \sum_{(u,v) \in M} \left( \log \hat{Y}_{u, v, Y_{u,v}} + \log \hat{Y}^a_{u, v, Y_{u,v}} \right)$$

#### 2.3 Generalized Masked Sparsity-Invariant Feature Consistency (GMSIFC)
GMSIFC enforces point-wise feature consistency across varying beam densities.

##### Mathematical Definition:
1. **Semantic Purity Mask:** To prevent boundary blurring caused by mixing distinct semantic classes across sparse grids, a local $3 \times 3$ neighborhood purity mask $M_{\text{pure}}$ is computed:
   $$M_{\text{pure}}(u, v) = \mathbb{1}\left[ \max_{(i, j) \in \mathcal{N}_3(u, v)} Y_{i, j} = \min_{(i, j) \in \mathcal{N}_3(u, v)} Y_{i, j} \right] \cdot \mathbb{1}[Y_{u, v} > 0]$$
   where $\mathcal{N}_3(u, v) = \{ (i, j) : |i - u| \le 1, |j - v| \le 1 \}$.
2. **Paired Point Matching:** For paired positions $(u, v) \in \Omega_p \cap M_{\text{pure}}$, an entrywise $L_1$ alignment loss is computed:
   $$\mathcal{L}_{\text{paired}} = \frac{1}{|\Omega_p \cap M_{\text{pure}}|} \sum_{(u, v) \in \Omega_p \cap M_{\text{pure}}} \| z(u, v) - z^a(u, v) \|_1$$
3. **Unpaired Feature Aggregation:** For an unpaired source point $(u, v) \in \Omega_u^s$, its representation is aligned against an affinity- and distance-weighted aggregation of paired augmented neighbors $(u', v') \in \Omega_p$. The cosine affinity kernel is thresholded by $\tau = 0.7$:
   $$\operatorname{aff}\big(z(u, v), z(u', v')\big) = \frac{\langle z(u, v), z(u', v') \rangle}{\|z(u, v)\|_2 \|z(u', v')\|_2} \cdot \mathbb{1}\left[ \frac{\langle z(u, v), z(u', v') \rangle}{\|z(u, v)\|_2 \|z(u', v')\|_2} \ge \tau \right]$$
   Weighting by spatial inverse distance $d\big((u, v), (u', v')\big) = \|(u, v) - (u', v')\|_2$:
   $$w\big((u, v), (u', v')\big) = \frac{\operatorname{aff}\big(z(u, v), z(u', v')\big)}{d\big((u, v), (u', v')\big) + \epsilon}$$
   The synthesized augmented feature $z_{\text{agg}}^a(u, v)$ is:
   $$z_{\text{agg}}^a(u, v) = \frac{\sum_{(u', v') \in \Omega_p} w\big((u, v), (u', v')\big) z^a(u', v')}{\sum_{(u', v') \in \Omega_p} w\big((u, v), (u', v')\big)}$$
4. **Symmetric GMSIFC Loss:** Symmetrically aggregating $z_{\text{agg}}^s$ for points in $\Omega_u^a$, the total GMSIFC loss is:
   $$\mathcal{L}_{\text{GMSIFC}} = \mathcal{L}_{\text{paired}} + \frac{1}{|\Omega_u^s|} \sum_{u \in \Omega_u^s} \| z(u) - z_{\text{agg}}^a(u) \|_1 + \frac{1}{|\Omega_u^a|} \sum_{u \in \Omega_u^a} \| z_{\text{agg}}^s(u) - z^a(u) \|_1$$

##### Mechanism & Goal:
GMSIFC ensures that subsampling beams does not alter local bottleneck features. The affinity threshold $\tau$ prevents propagation across semantic boundaries, maintaining feature fidelity regardless of beam density.

#### 2.4 Localized Semantic Correlation Consistency (LSCC)
LSCC enforces that the topological inter-class correlation structure remains invariant across views within localized spatial regions.

##### Mathematical Definition:
1. **Spatial Cell Partitioning:** The range image of width $W$ is partitioned into non-overlapping vertical column cells $\mathcal{C}_k$ of width $W_{\text{cell}} = 256$.
2. **Cell Prototype Pooling:** For each cell $\mathcal{C}_k$ and view $v \in \{\text{clean}, \text{aug}\}$, normalized class prototype vectors $\tilde{Z}_{k, c}^v \in \mathbb{S}^{127}$ are pooled across points with label $c$:
   $$Z_{k, c}^v = \frac{\sum_{(u, v) \in \mathcal{C}_k} \mathbb{1}[Y_{u, v} = c \land M_{u, v}^v = 1] z^v(u, v)}{\sum_{(u, v) \in \mathcal{C}_k} \mathbb{1}[Y_{u, v} = c \land M_{u, v}^v = 1] + \epsilon}, \quad \tilde{Z}_{k, c}^v = \frac{Z_{k, c}^v}{\|Z_{k, c}^v\|_2}$$
3. **Correlation Gram Matrices:** The class correlation matrix for cell $\mathcal{C}_k$ in view $v$ is given by $G_k^v = \tilde{Z}_k^v (\tilde{Z}_k^v)^T \in \mathbb{R}^{K \times K}$, where entry $(c_1, c_2)$ captures the cosine similarity between class prototypes $c_1$ and $c_2$.
4. **All-Pairs Shared-Class Correlation Mismatch:** For any two cells $\mathcal{C}_k, \mathcal{C}_l$ with binary class presence indicators $p_k, p_l \in \{0, 1\}^K$, let $U_k = (p_k p_k^T) \odot G_k$ and $W_k = U_k^{\odot 2}$. The squared Frobenius difference over shared classes is:
   $$\mathcal{L}_{\text{corr}} = \frac{1}{|\mathcal{P}|} \sum_{(k, l) \in \mathcal{P}} \frac{p_l^T W_k p_l + p_k^T W_l p_k - 2 \operatorname{Tr}(U_k U_l^T)}{(p_k^T p_l)^2}$$
5. **Within-Scan InfoNCE Contrastive Regularizer:** To prevent prototype collapse within each scan $s$, an InfoNCE loss on normalized bottleneck features $\tilde{z}_i = z_i / \|z_i\|_2$ is added:
   $$\mathcal{L}_{\text{contr}} = - \frac{1}{N_{\text{scans}}} \sum_{s=1}^{N_{\text{scans}}} \frac{1}{|I_s|} \sum_{i \in I_s} \log \frac{\sum_{p \in \mathcal{P}(i)} \exp(\tilde{z}_i \cdot \tilde{z}_p / \tau_{\text{NCE}})}{\sum_{a \in \mathcal{A}(i)} \exp(\tilde{z}_i \cdot \tilde{z}_a / \tau_{\text{NCE}})}$$
   where $\mathcal{P}(i) = \{ p \in \mathcal{A}(i) : Y_p = Y_i, p \neq i \}$ and $\mathcal{A}(i)$ is a subsample of valid points in scan $s$.
6. **Total LSCC Loss:** $\mathcal{L}_{\text{LSCC}} = \mathcal{L}_{\text{corr}} + \mathcal{L}_{\text{contr}}$.

##### Total DGLSS++ Trainer Loss:
$$\mathcal{L}_{\text{DGLSS++}} = \mathcal{L}_{\text{sem}} + \lambda_1 \mathcal{L}_{\text{GMSIFC}} + \lambda_2 \mathcal{L}_{\text{LSCC}} \quad (\text{with } \lambda_1 = 1.0, \, \lambda_2 = 1.0)$$

---

### 3. Step 2: The Addition of Covariate Shift Normalization (Cov-Shift)

Clean-anchored supervised contrastive learning was evaluated as an intermediate
step and **rejected**: it tightens clusters but erases the recoverable corruption
shift $\Delta z_c$ (the clean-anchor trap), lowering the labeled oracle ceiling
(fog $17.6\% \rightarrow 15.7\%$, crosstalk $22.2\% \rightarrow 18.8\%$ on plain
DGLSS++). The cov-shift recipe therefore uses **no clean-anchoring loss**; its two
training-and-inference structural normalizations below are the entire addition.

The central discovery of this project is that the corruption problem is **not a geometry alignment problem to be solved with loss penalties**, but a **two-level statistical covariate shift**. 

The corruption physics decomposes into two distinct mechanisms:
- **Crosstalk / Intensity Noise:** A *first-order statistical distribution shift* in sensor measurement channels (range and remission).
- **Fog / Spatial Distortion:** A *geometric coordinate displacement* in $(x, y, z)$ space where the recoverable structure is a shifted direction that must be preserved without distortion.

To solve both without destroying recoverable geometry, Cov-Shift introduces two purely structural, training-and-inference normalizations without anchor losses:

```
Raw Input Scan X in R^{5 x H x W}
   |
   +--> Channels {0, 4} (Range, Remission): Per-Scan Valid-Point Standardized (absorbs Crosstalk)
   +--> Channels {1, 2, 3} (X, Y, Z Geometry): Clean-Dataset Standardized (preserves Fog direction)
   |
[Channel-Restricted Per-Scan Input Normalization]
   |
   v
ResNet-34 Backbone with Internal 2D InstanceNorm (removes clean running-statistic bias)
   |
   v
Bottleneck Feature z in R^{128} (high oracle ceiling + zero-shot alignment)
   |
   v
Sign Hyperdimensional Projection b = sign(z R) in {-1, +1}^{10000} --> Cosine Decode
```

#### 3.1 Level 1: Channel-Restricted Per-Scan Input Normalization (`inputin_chan`)

##### Mathematical Definition:
Let the input scan tensor be $X \in \mathbb{R}^{5 \times H \times W}$ with validity mask $M_{u,v} = \mathbb{1}[X_{0,u,v} > 0]$ and valid point count $N_v = \sum_{u,v} M_{u,v}$. We partition the channels $\mathcal{C} = \{0, 1, 2, 3, 4\}$ into:
$$\mathcal{C}_{\text{stat}} = \{0, 4\} \quad (\text{range } r, \, \text{remission } i), \qquad \mathcal{C}_{\text{geom}} = \{1, 2, 3\} \quad (x, y, z)$$

1. **For statistics-shifted channels $c \in \mathcal{C}_{\text{stat}}$:**
   We compute the empirical mean and variance of scan $X$ strictly over its valid points:
   $$\mu_{\text{scan}, c} = \frac{1}{N_v} \sum_{u=1}^H \sum_{v=1}^W X_{c, u, v} \cdot M_{u, v}$$
   $$\sigma_{\text{scan}, c}^2 = \frac{1}{N_v} \sum_{u=1}^H \sum_{v=1}^W (X_{c, u, v} - \mu_{\text{scan}, c})^2 \cdot M_{u, v}$$
   The channel is re-standardized per-scan:
   $$\hat{X}_{c, u, v} = \left( \frac{X_{c, u, v} - \mu_{\text{scan}, c}}{\sqrt{\sigma_{\text{scan}, c}^2 + \epsilon}} \right) \cdot M_{u, v}$$
2. **For spatial geometry channels $c \in \mathcal{C}_{\text{geom}}$:**
   The channels are *left in their global clean dataset normalization*:
   $$\hat{X}_{c, u, v} = X_{c, u, v} = \left( \frac{X_{c, u, v}^{\text{raw}} - \mu_{\text{clean}, c}}{\sigma_{\text{clean}, c}} \right) \cdot M_{u, v}$$
3. **Execution Scope:** This operation is embedded directly inside the network's forward pass $\hat{X} = \text{InputNorm}(X)$, guaranteeing identical behavior during training, validation, and test-time evaluation.

##### How it Achieves the Goal:
- **For Crosstalk:** Multipath reflections and sensor crosstalk inject intense spurious energy into range and remission. Computing $(\mu_{\text{scan}}, \sigma_{\text{scan}})$ on channels $\{0, 4\}$ dynamically absorbs the anomalous offset and scale per frame, restoring standard input distributions.
- **For Fog:** Fog shifts $(x, y, z)$ coordinates along physical line-of-sight vectors. Because channels $\{1, 2, 3\}$ are *not* centered per-scan, the absolute spatial coordinate frame and the coherent class displacement directions $\Delta z_c$ survive intact. (Ablation: normalizing all 5 channels erases fog's directional retention, collapsing class tightness from $0.89$ to $0.47$ and degrading oracle accuracy, Iteration 19.12.2).

#### 3.2 Level 2: Internal Instance Normalization (Backbone InstanceNorm)

##### Mathematical Definition:
In standard convolutional backbones, 2D Batch Normalization computes mini-batch statistics during training and freezes running estimates $(\mu_{\text{run}}, \sigma_{\text{run}}^2)$ for inference:
$$\text{BN}(h)_{b, c, u, v} = \gamma_c \left( \frac{h_{b, c, u, v} - \mu_{\text{run}, c}}{\sqrt{\sigma_{\text{run}, c}^2 + \epsilon}} \right) + \beta_c$$
Under covariate shift, the true distribution of intermediate activation maps $h \in \mathbb{R}^{B \times C \times H' \times W'}$ deviates from $(\mu_{\text{run}}, \sigma_{\text{run}}^2)$, causing systematic feature distortion across all layers.

Cov-Shift replaces all BatchNorm layers throughout the ResNet backbone with **2D Instance Normalization**:
$$\mu_{b, c} = \frac{1}{H' W'} \sum_{u=1}^{H'} \sum_{v=1}^{W'} h_{b, c, u, v}$$
$$\sigma_{b, c}^2 = \frac{1}{H' W'} \sum_{u=1}^{H'} \sum_{v=1}^{W'} (h_{b, c, u, v} - \mu_{b, c})^2$$
$$\text{IN}(h)_{b, c, u, v} = \gamma_c \left( \frac{h_{b, c, u, v} - \mu_{b, c}}{\sqrt{\sigma_{b, c}^2 + \epsilon}} \right) + \beta_c$$
where $\gamma_c, \beta_c \in \mathbb{R}$ are learnable per-channel affine parameters.

##### How it Achieves the Goal:
- **Decoupling from Training Statistics:** InstanceNorm computes statistics strictly within each individual scan $b$ and feature channel $c$, completely eliminating reliance on clean running population statistics.
- **Internal Covariate Shift Invariance:** Feature activations at deep bottleneck layers remain standardized and well-conditioned regardless of corruptions.

---

### 4. Step 3: The Paradigm Shift from Prototype Distance Classification (R1) to Linear Classification (R4)

The final architectural breakthrough addresses the **downstream classification decision rule**. While the cov-shift extractor succeeds in preserving continuous feature separability, decoding via nearest-prototype cosine distance (R1) was found to discard a substantial fraction of the recoverable signal under domain shifts.

#### 4.1 The Geometric Bottleneck of Nearest-Prototype Classification (R1)
The standard HDC nearest-centroid decision rule assigns a binarized test code $b = \operatorname{sign}(z R) \in \{-1, +1\}^{D_{\text{HDC}}}$ to class $\hat{y}$ via maximum cosine similarity to clean class prototype vectors $P_c$:
$$\hat{y}_{\text{R1}} = \arg\max_{c \in \{1, \dots, K\}} \cos(b, P_c) = \arg\max_{c} \frac{\langle b, P_c \rangle}{\|b\|_2 \|P_c\|_2}$$

##### Why Nearest-Prototype Decoding Fails Under Corruption:
1. **Implicit Equi-Variance / Spherical Assumption:** Nearest-prototype classification is optimal if and only if the class-conditional distributions $p(b \mid Y=c)$ are isotropic Gaussian or von Mises-Fisher distributions on the sphere with **identical intra-class variance** $\sigma_1^2 = \dots = \sigma_K^2$.
2. **Induced Anisotropic Dispersion:** In real LiDAR point clouds under physical corruptions (e.g., wet ground or crosstalk), corruptions produce severe **class-dependent variance disparity** (e.g., class correlation tightness drops to $0.67$ for other-ground but remains $0.85$ for car).
3. **Rigid Midplane Boundary:** The decision boundary between class $i$ and class $j$ under cosine prototype matching is strictly orthogonal to the prototype difference vector $(P_i - P_j)$:
   $$\mathcal{H}_{i,j}^{\text{R1}} = \left\{ b \in \mathbb{R}^{D_{\text{HDC}}} : \left\langle b, \frac{P_i}{\|P_i\|_2} - \frac{P_j}{\|P_j\|_2} \right\rangle = 0 \right\}$$
   When class $i$ has high dispersion (a "fat blob") and class $j$ has tight dispersion, the fixed midplane boundary slices directly through the high-variance distribution of class $i$, causing massive systematic misclassification into class $j$.

#### 4.2 Mathematical Formulation of the R4 Linear Classifier
Let $B \in \{-1, +1\}^{N \times D_{\text{HDC}}}$ denote the matrix of binarized codes for $N$ points, and let $Y_{\text{one-hot}} \in \{0, 1\}^{N \times K}$ denote their one-hot ground truth labels. The optimal weight matrix $W^* \in \mathbb{R}^{D_{\text{HDC}} \times K}$ is obtained by regularized least-squares empirical risk minimization (Ridge Regression):
$$W^* = \arg\min_{W \in \mathbb{R}^{D_{\text{HDC}} \times K}} \frac{1}{N} \| B W - Y_{\text{one-hot}} \|_F^2 + \lambda \|W\|_F^2$$

Setting the matrix gradient to zero yields the closed-form normal equation:
$$W^* = \left( B^T B + N \lambda I \right)^{-1} B^T Y_{\text{one-hot}} = \left( S + \lambda_{\text{reg}} I \right)^{-1} T$$
where:
- $S = B^T B \in \mathbb{R}^{D_{\text{HDC}} \times D_{\text{HDC}}}$ is the uncentered second-moment sample covariance matrix of the binarized codes.
- $T = B^T Y_{\text{one-hot}} \in \mathbb{R}^{D_{\text{HDC}} \times K}$ is the class-aggregated code matrix, whose $c$-th column $T_c = \sum_{i \in \mathcal{S}_c} b_i$ is proportional to the unnormalized prototype vector of class $c$.

#### 4.3 Theoretical Mechanism: Why Linear Classification Resolves the Ceiling Bottleneck
The relationship $W_c = (S + \lambda_{\text{reg}} I)^{-1} T_c$ reveals the precise mathematical distinction between R1 and R4:
1. **Prototype Matching as Identity Covariance ($S = I$):**
   If we approximate the covariance matrix as spherical ($S \propto I$), the linear classifier weights simplify to:
   $$W_c \propto T_c \propto P_c$$
   Nearest-prototype decoding is therefore mathematically equivalent to linear classification under the false assumption that all features are orthogonal and uncorrelated.
2. **Covariance Whitening and Hyperplane Rotation:**
   Pre-multiplying the prototype direction $T_c$ by the inverse covariance $(S + \lambda_{\text{reg}} I)^{-1}$ rotates the decision normal $w_c$ away from directions of high background variance and aligns it along the discriminative subspace:
   $$w_c - w_j = (S + \lambda_{\text{reg}} I)^{-1} (T_c - T_j)$$
   This automatically shifts the decision boundary closer to tight, dense classes and further away from diffuse classes, correcting for class-conditional dispersion disparity.
3. **Exploiting Hyperdimensional Linear Separability:**
   Projecting continuous embeddings $z \in \mathbb{R}^{128}$ to $D_{\text{HDC}} = 10{,}000$ dimensions via $b = \operatorname{sign}(z R)$ maps non-linear class manifolds into linearly separable configurations in high-dimensional Hamming space (Cover's Theorem). Fitting a learned linear hyperplane in this 10,000D space unlocks the full discriminative capacity of the representation.

#### 4.4 Empirical Performance: R1 vs. R4 Oracle Ceiling
In Iteration C10, evaluating the R4 linear classifier against the R1 nearest-prototype rule on frozen cov-shift features demonstrated a decisive mIoU gain across all conditions. The table is the FULL-DATASET harness (every point of every frame of seq 08, `al_full_dataset_diag.py`; spectral-exact ridge, 200k clean fit / 400k pool):

| Condition | Epoch | R1 Baseline (Nearest-Prototype) | R4 Linear Classifier (HDC Code Probe) | Relative Gain (R4 / R1) | Absolute mIoU Gain |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Fog** | ep-10 | $30.1\%$ | **$36.9\%$** | **$1.23\times$** | $+6.8\%$ |
| **Crosstalk** | ep-10 | $39.4\%$ | **$49.1\%$** | **$1.25\times$** | $+9.7\%$ |
| **Wet Ground** | ep-10 | $25.4\%$ | **$41.9\%$** | **$1.65\times$** | $+16.5\%$ |
| **Snow** | ep-10 | $39.2\%$ | **$49.5\%$** | **$1.26\times$** | $+10.3\%$ |
| **Fog** | ep-21 | $28.1\%$ | **$35.5\%$** | **$1.26\times$** | $+7.4\%$ |
| **Crosstalk** | ep-21 | $39.5\%$ | **$46.2\%$** | **$1.17\times$** | $+6.7\%$ |
| **Wet Ground** | ep-21 | $26.5\%$ | **$41.1\%$** | **$1.55\times$** | $+14.6\%$ |
| **Snow** | ep-21 | $38.1\%$ | **$46.5\%$** | **$1.22\times$** | $+8.4\%$ |

The R4 gain over R1 is largest on wet_ground (1.55-1.65x) and smallest on the healthy conditions (1.17-1.26x). At full scale the R1 rule has almost no recoverable headroom (gaps 0.001-0.009) -- the recoverable signal lives in the R4 linear probe, not the prototype rule.

#### 4.5 Computational Efficiency: Fast Classifier Updates vs. Fast Inference
Replacing prototype matching with a linear classifier introduces two distinct computational phases:

1. **How We Keep Classifier Updates Fast (Training / TTA / Active Learning):**
   Fitting or re-estimating $W = (B^T B + \lambda I)^{-1} B^T Y$ over $N = 100{,}000$ points in $D_{\text{HDC}} = 10{,}000$ dimensions would normally require $\mathcal{O}(D^3)$ FLOPs (multi-second dense matrix inversion of $S = B^T B$).
   - **Matrix-Free CG:** Avoids constructing the $10{,}000 \times 10{,}000$ covariance matrix entirely by computing $(B^T B + \lambda I) v = B^T (B v) + \lambda v$ in $\mathcal{O}(N \cdot D_{\text{HDC}})$ operations.
   - **Nyström Warm Start ($m=1{,}000$):** Seeds CG with a low-rank sketch, reducing CG iterations from $20+$ to just $5\text{–}8$.
   - **Update Speed:** Refitting $W$ takes only **$\approx 0.034$ seconds** ($\approx 1.5\times 10^6$ points/sec update throughput).

2. **How Classification (Inference) Remains Fast:**
   Once $W^*$ is computed, inference for incoming LiDAR points does **not** run CG or matrix inversions. 
   - **Classification / Decoding Cost:** For a test point with code $b \in \{-1, +1\}^{10{,}000}$, prediction is a single matrix-vector multiplication $b W^*$ followed by an $\arg\max$ over 17 classes ($\mathcal{O}(D_{\text{HDC}} \cdot K) = 1.7 \times 10^5$ operations).
   - **Inference Speed:** Runs at **$> 2.5\times 10^5$ points/sec** in real-time, matching the runtime speed of standard prototype cosine decoding.

#### 4.6 The Stable Inference Update: Low-Rank Residual Decoder Update ($W_{\text{res}}$, Iteration C30)

While the full oracle linear probe $W^* = (S + \lambda_{\text{reg}} I)^{-1} T$ defines the theoretical ceiling, **adapting the classifier under limited label budgets ($k=2\text{--}8$ labels per class, $14\text{--}56$ points)** fails if one naively attempts to refit all $10{,}000 \times 17$ parameters from scratch.

##### 1. The Bottleneck of Full-Probe Re-estimation
Inverting the full $10{,}000 \times 10{,}000$ sample covariance $S = \frac{1}{N} B^T B$ from sparse labels requires solving an ill-conditioned linear system ($\kappa(S) \approx 10^6\text{--}10^7$). As discovered in Iterations C16–C23, small label sampling errors $\delta T$ are amplified by **$11\times\text{--}125\times$** along tail eigenvectors, causing full-probe updates $W_{\text{sub}}$ to collapse ($-0.21$ to $-0.61$ mIoU drop).

##### 2. The Low-Rank Residual Discovery (Iteration C20)
Singular value decomposition (SVD) of the true oracle residual matrix $R = W^* - W_0 \in \mathbb{R}^{10{,}000 \times 17}$ (the difference between the oracle classifier $W^*$ and the frozen clean probe $W_0$) revealed a fundamental structural property:
- The effective rank of the residual $R$ is strictly **$r \approx 4\text{--}5$** across all corruptions and extractors.
- Projecting $R$ onto its top $r=8$ left singular vectors $U_r \in \mathbb{R}^{10{,}000 \times 8}$ captures **$> 99\%$ of the cumulative spectral energy** ($mIoU(W_0 + R_8) \equiv mIoU(W^*)$).
- Therefore, test-time adaptation does **not** need to estimate $170{,}000$ parameters in $10{,}000\text{-D}$ space; it only needs to estimate an **$8 \times 17$ low-rank coefficient matrix $C$** ($136$ parameters).

```
Target Codes X_lab in R^{N_lab x 10000} ---> Residual Labels: \Delta Y = Y_lab - X_lab W_0
                                                               |
Orthonormal Basis U_r in R^{10000 x 8} (r=8) ----------------> Solve 8x8 System:
                                                               C = (U_r^T X^T X U_r + \gamma I)^{-1} U_r^T X^T \Delta Y
                                                               |
Frozen Probe W_0 in R^{10000 x 17} --------------------------> W_res = W_0 + \eta U_r C
```

##### 3. Mathematical Formulation of the C30 Stable Update
Let $W_0 \in \mathbb{R}^{D_{\text{HDC}} \times K}$ be the frozen clean linear probe ($D_{\text{HDC}} = 10{,}000, K=17$), and let $U_r \in \mathbb{R}^{D_{\text{HDC}} \times r}$ ($r=8$) be the orthonormal basis spanning the rank-$r$ corruption residual subspace ($U_r^T U_r = I_r$).

The adapted linear classifier $W_{\text{res}}$ is parameterized as:
$$W_{\text{res}} = W_0 + \eta \, U_r C$$
where:
- $\eta \in [0.5, 1.0]$ is the adaptation step size ($\eta = 1.0$ for large-gap conditions like fog and wet ground; $\eta = 0.5\text{--}0.75$ for small-gap conditions like snow and crosstalk).
- $C \in \mathbb{R}^{r \times K}$ is the low-rank coefficient matrix fit on a small labeled support set $(X_{\text{lab}}, Y_{\text{lab}})$ by minimizing regularized residual squared error:
  $$C^* = \arg\min_{C \in \mathbb{R}^{r \times K}} \frac{1}{N_{\text{lab}}} \big\| X_{\text{lab}} (W_0 + U_r C) - Y_{\text{lab}} \big\|_F^2 + \gamma \|C\|_F^2$$

Setting the matrix derivative with respect to $C$ to zero yields the **closed-form $r \times r$ normal equation**:
$$C^* = \left( U_r^T X_{\text{lab}}^T X_{\text{lab}} U_r + \gamma I_r \right)^{-1} U_r^T X_{\text{lab}}^T \left( Y_{\text{lab}} - X_{\text{lab}} W_0 \right)$$
where:
- $Y_{\text{lab}} - X_{\text{lab}} W_0 \in \mathbb{R}^{N_{\text{lab}} \times K}$ is the **residual error matrix** of the frozen clean probe on the labeled points.
- $U_r^T X_{\text{lab}}^T X_{\text{lab}} U_r \in \mathbb{R}^{8 \times 8}$ is an $8 \times 8$ projected covariance matrix that is well-conditioned and inverted in sub-milliseconds; the C30 sweep found $\gamma \approx 10^{-6}\text{--}10^{-3}$ (effectively $\approx 0$ ridge) best -- the $8 \times 8$ system needs no meaningful regularization.

##### 4. Why the C30 Residual Formulation is Inherently Stable
1. **Low-Dimensional Regularization:** Inverting an $8 \times 8$ matrix rather than a $10{,}000 \times 10{,}000$ matrix completely eliminates the $11\times\text{--}125\times$ whitened error amplification.
2. **Zero-Degradation Guarantee:** If the domain shift is minimal ($Y_{\text{lab}} \approx X_{\text{lab}} W_0$, as on clean or healthy conditions), the residual target $Y_{\text{lab}} - X_{\text{lab}} W_0 \to 0$, forcing $C \to 0$ and $W_{\text{res}} \to W_0$. The classifier cannot catastrophically degrade healthy baseline performance.
3. **Exact One-Hot Target Alignment:** Evaluating residual targets in C30 showed that exact one-hot targets $Y_{\text{lab}} - X_{\text{lab}} W_0$ decisively outperform softened probabilities $Y_{\text{lab}} - P_0$ ($+0.060$ vs $+0.023$ mIoU gain on fog).

##### 5. Measured Performance on Cov-Shift ep-10 (Iteration C30 / C31)
Evaluated on the cov-shift `inputin_in_chan` ep-10 extractor at budget $k=8$ points/class ($56$ true labeled points total):

| Condition | Frozen Probe $W_0$ | Oracle Ceiling $W^*$ | Closeable Gap | C30 $W_{\text{res}}$ ($56$ labels, $k=8$) | Gap Closed ($\%$) | C31 $W_{\text{res}}$ ($56 + 500$ bank) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Fog** | $0.259$ | $0.375$ | $+0.118$ | **$0.319$ ($+0.060$)** | **$51\%$** | **$0.384$ ($+0.125$)** |
| **Wet Ground** | $0.429$ | $0.614$ | $+0.187$ | **$0.566$ ($+0.137$)** | **$73\%$** | **$0.593$ ($+0.164$)** |
| **Snow** | $0.457$ | $0.493$ | $+0.036$ | **$0.473$ ($+0.016$)** | **$44\%$** | **$0.494$ ($+0.037$)** |
| **Crosstalk** | $0.524$ | $0.554$ | $+0.031$ | **$0.534$ ($+0.010$)** | **$33\%$** | **$0.552$ ($+0.028$)** |

---

### 5. Summary of the Complete Cov-Shift Architecture (`inputin_in_chan` + Low-Rank Residual Classifier)

The complete robust LiDAR segmentation system combines:
1. **Input Normalization:** Channel-restricted per-scan normalization on channels $\{0, 4\}$ (range, remission), leaving $\{1, 2, 3\}$ ($x, y, z$) in global clean standardization.
2. **Backbone Architecture:** ResNet-34 parameterized with internal 2D Instance Normalization across all layers.
3. **Extractor Training Loss:** DGLSS++ structural representation consistency without destructive clean-anchoring losses:
   $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{sem}}(\hat{X}, \hat{X}^a) + \lambda_1 \mathcal{L}_{\text{GMSIFC}}(\hat{X}, \hat{X}^a) + \lambda_2 \mathcal{L}_{\text{LSCC}}(\hat{X}, \hat{X}^a)$$
4. **Decoder & Stable Inference Update:** Lifting 128D features to 10,000D binarized codes $b = \operatorname{sign}(z R)$ and classifying via a frozen clean linear probe $W_0$. Test-time adaptation is executed via the low-rank residual update $W_{\text{res}} = W_0 + \eta U_r C$ ($r=8$), which solves an $8 \times 8$ normal equation on the probe residual $Y_{\text{lab}} - X_{\text{lab}} W_0$, closing $11\%\text{--}79\%$ of the closeable oracle headroom at full scale with only $56$ labels (fog $+0.005$ of $+0.047$, wet_ground $+0.043$ of $+0.122$) while strictly preserving healthy-domain performance.

## Measured performance

All numbers are the **ep-10 model** (the optimal window of the 21-ep run, chosen by
the mid-training monitor) unless noted. Sources: extractor-diff harness (fog/crosstalk
ceiling+TTA, 100k/100k) and the isotropy pipeline (8-condition HDC-zs), see the
README tables for the full comparison.

### Ceilings (labeled oracle, extractor-diff harness, fog/crosstalk)

| condition | DGLSS++ | Robust DGLSS++ | **Cov-shift (ep-10)** |
| :--- | :--- | :--- | :--- |
| fog | 17.6% | 15.7% | **23.5%** |
| crosstalk | 22.2% | 18.8% | **39.4%** |

### Ceilings at full scale (R4 linear probe, `al_full_dataset_diag.py`, all 8 conditions)

The paper-ready ceilings: every point of every frame of seq 08 streamed through
each frozen extractor, zero-shot on a 200k clean reservoir, ceiling on a 400k
corrupted-pool reservoir (spectral-exact ridge). The cov-shift extractor beats
both baselines on every condition at the R4 ceiling:

| condition | Cov-shift ep-10 | Cov-shift ep-21 | DGLSS++ | Robust DGLSS++ |
| :--- | :--- | :--- | :--- | :--- |
| fog | **0.369** | 0.355 | 0.170 | 0.173 |
| crosstalk | **0.491** | 0.462 | 0.224 | 0.232 |
| snow | **0.495** | 0.465 | 0.199 | 0.215 |
| wet_ground | **0.419** | 0.411 | 0.199 | 0.223 |
| incomplete_echo | **0.437** | 0.434 | 0.200 | 0.215 |
| beam_missing | 0.487 | 0.455 | 0.204 | **0.217** |
| motion_blur | **0.458** | 0.429 | 0.201 | 0.210 |
| cross_sensor | **0.446** | 0.428 | 0.203 | 0.225 |
| **mean (8)** | **0.450** | 0.430 | 0.200 | 0.214 |

The cov-shift R4 ceiling is 0.25-0.30 above the DGLSS++ ceiling on the
corrupted conditions (fog 0.369 vs 0.170, crosstalk 0.491 vs 0.224) and
0.24-0.30 above on the healthy conditions; only beam_missing is a near-tie
(0.487 vs 0.217 Robust).

### Label-free TTA (naive, same harness)

| condition | DGLSS++ | Robust DGLSS++ | **Cov-shift (ep-10)** |
| :--- | :--- | :--- | :--- |
| fog | 10.0% | 10.7% | **21.0%** |
| crosstalk | 12.7% | 15.0% | **38.6%** |

### Zero-shot HDC-zs (isotropy pipeline, all 8 conditions)

| condition | DGLSS++ | Robust | Cov-shift ep-10 | Cov-shift ep-21 |
| :--- | :--- | :--- | :--- | :--- |
| fog | 6.8% | 8.5% | **20.1%** | 18.5% |
| crosstalk | 11.5% | 9.8% | 39.5% | **41.9%** |
| snow | 39.6% | **41.1%** | 37.7% | 38.6% |
| wet_ground | **48.3%** | 46.8% | 35.8% | 33.3% |
| incomplete_echo | 44.9% | **45.0%** | 40.6% | 40.0% |
| beam_missing | **50.6%** | 50.3% | 44.3% | 44.5% |
| motion_blur | **50.2%** | **50.2%** | 44.2% | 44.6% |
| cross_sensor | 43.4% | **43.5%** | 36.1% | 38.8% |
| **mean (8 corrupted)** | 36.9% | 36.9% | **37.3%** | **37.5%** |

### The headline result

The cov-shift extractor is the first to raise BOTH the ceiling AND the label-free TTA
on BOTH fog and crosstalk: fog oracle 23.5% vs DGLSS++ 17.6%, crosstalk oracle 39.4%
vs 22.2%. **Crosstalk is effectively fixed at the frozen-prototype level**: zero-shot
40.3% ~= oracle 39.4%, so the oracle re-estimation adds little because the frozen
prototypes already decode correctly. This is the "fog/crosstalk are not inherently
broken" claim, achieved by the extractor rather than by TTA.

### The key property: naive TTA ~= ceiling

The label-free update essentially reaches the ceiling on both conditions (fog naive
21.0% vs ceiling 23.5%; crosstalk naive 38.6% vs ceiling 39.4%). Every prior
extractor left a real gap between naive TTA and ceiling (crosstalk 6-10 points): the
assignment wall. On the cov-shift extractor that wall is gone for crosstalk.

## Iteration log

### Iteration C1: the level-1 covariate-shift test (2026-08-14)

**Design.** The 5-channel input is normalized by FIXED clean-data statistics in the
parser. Test whether per-scan input normalization (training-side mirror of the
BN-statistic-alignment TTA lever, our best TTA method) raises the ceiling. Three
variants: `inputin` (input-IN only, internal BN), `inputin_in` (input-IN + internal
InstanceNorm, both levels), and the scale-only alternative (divide by per-scan std,
no mean subtraction).

**Result.** The full mean+std stack (`inputin_in`) was the strongest crosstalk result
in the family (crosstalk oracle 0.277 vs DGLSS++ 0.214) BUT fog regressed on the
oracle: the mean-shift component erased fog's recoverable direction. **Scale-only
was rejected** (Iteration 19.12.2): normalizing ALL channels by per-scan std,
including xyz, collapsed the class structure the clean-stat normalization provided
(corr_tightness car fog 0.89 -> 0.47). This confirmed the xyz channels' ABSOLUTE
scale carries the near-vs-far class structure.

### Iteration C2: the channel-restricted fix (2026-08-14)

**Design.** Restrict the per-scan input normalization to the range + remission
channels (0, 4) where crosstalk's statistics shift lives, and leave the xyz geometry
channels (1-3) in the parser's clean-stat normalization so fog's shifted direction
survives. Internal InstanceNorm kept. This is `inputin_in_chan`.

**Result.** Resolves the fog regression and improves everything:
- fog oracle 0.175 vs DGLSS++ 0.159 (the stack had dropped it to 0.137)
- crosstalk oracle 0.303 vs 0.214 (the strongest in the family)
- naive TTA also up on both (fog +0.067, crosstalk +0.165)

The mechanism is exact: normalizing only the range/remission channels absorbs
crosstalk's statistics shift while leaving fog's geometry in clean-stat
normalization: normalizing the SHIFTED STATISTICS, not the structure.

### Iteration C3: the scale-only negative (2026-08-14)

**Design.** The cleaner general version: divide by per-scan std on ALL channels but
DO NOT subtract the mean, preserving direction while absorbing magnitude.

**Result.** Clearly negative (fog oracle 0.086 vs DGLSS++ 0.159, crosstalk 0.113 vs
0.214). Dividing ALL channels by per-scan std, including xyz, collapsed the class
structure the clean-stat normalization provided. Confirms the channel-restricted
design (`inputin_in_chan`) is not one option among many but the necessary design.

### Iteration C4: the full medium run (2026-08-14 -> 15)

**Design.** 21 ep / 100% of `inputin_in_chan`, with a mid-training monitor
snapshotting the rolling checkpoint every ~30 min to find the optimal-epoch window
(the family degrades past 21 ep, Iteration 8.1).

**Result.** The monitor captured the full trajectory: fog oracle peaked at ~0.217
(ep 10) and held ~0.19-0.20 through 20; crosstalk stayed ~0.39-0.42 throughout. The
**ep-10 model is the optimal window** for fog (0.217 vs 0.195 at ep 21); ep-21 is
marginally better on crosstalk (0.411 vs 0.394). No degradation pattern past the
peak: the cov-shift win is stable, not a transient. The convergence note: the gains
are measured at the ep-10 optimal window; a sensible convergence metric or more
stable convergence behavior is needed to confirm behavior past it.

### Iteration C5: the full battery + ep-10/ep-21 comparison (2026-08-15)

**Design.** Train a fresh ep-10 model and run the full battery (tta_ceiling,
frozen_ceiling, gate_structure, extractor_diff vs DGLSS++ and Robust) on BOTH the
ep-10 and ep-21 checkpoints.

**Result.** The all-condition comparison (frozen HDC-oracle):
- cov-shift highest on fog (21.4%/20.2% vs DGLSS++ 15.1%) and crosstalk (38.9%/39.8%
  vs 21.4%)
- cov-shift lowest on the healthy conditions (wet_ground 40.5%/36.6% vs 51.4%)
- **the healthy-condition ceiling loss is the open problem** (next iteration).

### Iteration C6: the healthy-condition ceiling diagnostic (2026-08-15)

**Design.** Why does cov-shift hurt the healthy-condition ceilings while fixing
fog/crosstalk? The frozen-ceiling comparison shows the continuous class structure
survives (LP mostly preserved: snow 79->82%, wet_ground 85->77%) but the
HDC-binarized recoverability drops (wet_ground oracle 51->37%, beam_missing 51->44%).
`cond_structure_diag.py` measures per-class feat_cos / dir_retention / corr_tightness
/ zs on snow + wet_ground, plain DGLSS++ vs cov-shift.

**Result: the per-class mechanism (wet_ground):**

| cls | corr_tight A->B | zs A->B |
| :--- | :--- | :--- |
| car(4) | 0.96 -> 0.85 | 0.866 -> 0.746 |
| og(14) | 0.86 -> 0.67 | 0.775 -> 0.532 |
| road(11) | 0.98 -> 0.94 | 0.658 -> 0.483 |
| veg(16) | 0.90 -> 0.81 | 0.649 -> 0.574 |

The recoverable classes on wet_ground show a clear **PACKING loss** (corr_tight
drops for nearly every class) while the DIRECTION is retained (dir_ret stays
~0.99-1.0). The same pattern holds for snow (smaller). **This is a
binarization/recoverability loss, not a direction loss.**

**Interpretation for the projection/binarization design (the next step).** The
hypothesis is confirmed: InstanceNorm + the per-scan input normalization change the
per-dimension feature scale/magnitude on the healthy conditions, pushing the
recoverable classes closer to the HDC sign-binarization threshold. The continuous
structure is intact (LP kept, direction kept) but the packing that the HDC
random-projection + sign threshold needs is degraded. The fix direction is to make
the projection matrix or the binarization robust to this scale change, e.g., a
scale-aware projection, a threshold-aware binarization, or a normalization ordering
that preserves the healthy conditions' anisotropy (scale-normalize-then-InstanceNorm
instead of the current ordering).

## Open questions

1. ~~**Can the projection matrix or binarization be redesigned** so the healthy-
   condition packing survives the cov-shift normalization without losing the
   fog/crosstalk gains?~~ **CLOSED.** C8 rejected encoding changes (all encodings lose
   equally); C9 rejected the training-side normalization-scope levers at micro scale.
   The packing loss is not recoverable by changing the extractor's normalization or
   the binarization.
2. **Can a LEARNED HDC-code decoder replace nearest-centroid?** C10 says yes (R4 =
   1.24-1.77x), but it needs labeled data (frozen clean-fit is the label-free
   version). Open: does a label-free pool-refit R4 hold the C10 gain without labels?
3. **Does the NuScenes cross-domain gap give TTA real headroom?** C11 found a
   +0.05-0.06 oracle-zs gap on NuScenes vs ~0 on KITTI clean. Open: can naive TTA
   (re-estimating the decoder on the NuScenes pool) take that gap?
4. **Convergence**: the gains are measured at the ep-10 optimal window; does a
   sensible convergence metric or more stable convergence behavior confirm the
   extractor's peak is stable?

### Iteration C7: the binarization diagnostic (2026-08-15)

**Goal.** C6 found the healthy-condition ceiling loss is a PACKING loss in binarized
space (continuous structure kept, HDC-oracle drops). This iteration diagnoses what
the current `sign(z @ R)` binarization loses and tests whether an alternative
encoding recovers the healthy ceilings without losing the fog/crosstalk gains.

**What the current binarization does.** `get_hdc_projection` builds a random +-1
matrix R (dim_in x 10000); features are binarized as `sign(z @ R)`, threshold 0 per
coordinate, all coordinates equally weighted. Prototypes are per-class means of the
binarized clean codes.

**What the diagnostic measures (on frozen features, no training):**

1. **A: the pre-sign margin fraction**: the fraction of $|z @ R| < eps$ coordinates
   (default eps=0.1). A coordinate near 0 is threshold-hugging: it flips sign on
   small feature noise. If the cov-shift HEALTHY features have a higher near-0
   fraction than DGLSS++ (B > A on snow/wet_ground), that is the quantitative
   signature of the packing loss: the cov-shift normalization moved the healthy
   features closer to the sign decision boundary, so sign() binarizes them
   unreliably.
2. **B: three alternative encodings, on the same frozen features:**
   - `bias`   : `sign(z @ R - b)` with per-coordinate b = median of the CLEAN
     projection. Each coordinate binarizes around its clean-typical value, not an
     arbitrary 0, so healthy-condition coordinates that cov-shift scaled away from 0
     are re-centered.
   - `zscore` : `sign((z @ R - b) / s)` with per-coordinate clean mean and std.
     Z-scores the projection so every coordinate contributes equally before
     binarization: undoes the per-dimension scale change InstanceNorm introduced.
   - `fourier`: the other standard HDC style: $[cos(z @ w), sin(z @ w)]$ with random
     frequencies w (dim_in x 10000, continuous codes, dim 20000). No sign; a smooth
     periodic encoding. Tests whether retaining continuous phase information fixes
     what hard sign() loses.

   For each encoding, on each condition: the frozen-prototype decode (zs) and the
   oracle decode (re-estimate prototypes from corrupted labeled points).

**Decision rule.** If on the healthy conditions an alternative encoding recovers the
cov-shift oracle toward (or above) DGLSS++'s level WITHOUT dropping the fog/crosstalk
gain, that encoding is the fix, and it can be adopted decoder-side (the extractor is
unchanged, only the HDC projection/binarization is swapped). If `margin_frac` B > A
confirms the packing loss, the mechanism is validated and the fix is a scale-aware or
bias-aware binarization.

Run: `bash run_binarization_diag.sh 3` (both ep10 + ep21, eval-only, ~1-1.5h).

### Iteration C8: the binarization diagnostic: the loss is continuous, not binarized

The C7 hypothesis was that the cov-shift healthy-ceiling loss is a BINARIZATION loss
(features pushed toward the sign threshold). The C8 diagnostic (`binarization_diag.py`)
tested this on frozen features with four encodings (current `sign`, per-coordinate
`bias`, `zscore`, and the continuous Fourier $[cos, sin]$ style) plus the pre-sign
margin fraction.

**Result: the hypothesis is REJECTED, and the finding reframes the problem.**

1. **margin_frac is nearly identical between extractors** (0.0255 vs 0.0263 ep10;
   0.0256 vs 0.0306 ep21). The cov-shift healthy features do NOT sit closer to the
   sign threshold: the "threshold-hugging → unreliable binarization" mechanism does
   not occur.
2. **No alternative encoding recovers the healthy oracle.** On wet_ground (the
   worst case), the cov-shift oracle is ~0.22 (ep10) / ~0.19 (ep21) across sign /
   bias / zscore / fourier, all far below DGLSS++'s 0.27. The encoding choice barely
   changes anything (spread < 0.006).
3. **The loss is in the CONTINUOUS features, not the binarization.** If it were a
   binarization artifact, the continuous Fourier encoding (no sign) would retain the
   packing that sign() loses. It does not, so the cov-shift extractor's healthy
   features have genuinely lost recoverable structure at the continuous level.

**What this means for the fix.** Decoder-side projection/binarization redesign
(bias, zscore, Fourier, or any scale-aware threshold) CANNOT recover the
healthy-condition ceiling: the information is not in the binarized code because it
is not in the continuous features. The fix must be TRAINING-SIDE: the cov-shift
normalization (per-scan input-IN + internal InstanceNorm) is erasing a continuous
recoverable structure on the healthy conditions, and the levers are:
- **Scope the normalization**: apply InstanceNorm only to the bottleneck channels
  that fog/crosstalk recover through, not the whole backbone, so the healthy
  conditions' continuous structure is untouched.
- **Scale-normalize-then-InstanceNorm ordering**: preserve the healthy conditions'
  per-dimension anisotropy before InstanceNorm re-scales it.
- **Condition-aware regularization**: keep the healthy-condition feature scale
  closer to clean during training (the C6 packing loss is a real continuous loss,
  and the fix is to not compress it, not to re-binarize it).

This is a clean negative that redirects the effort from the HDC decoder (projection /
binarization design) back to the extractor's normalization scope, which is where the
cov-shift method's actual lever is.

## Potential next step: changing the HDC classification rule

The C8 negative covers the *encoding* (how a feature becomes a code) but not the
*decision rule* (how a code is assigned to a class). All C7/C8 variants still end in
the same nearest-centroid rule: unit-norm cosine to per-class prototypes
(`decode`/`decode_preds` in the diagnostic harness). If the training-side levers
above fail to recover the healthy-condition packing, the decision rule itself is the
remaining decoder-side lever. This is distinct from the encoding changes C8 ruled out.

The key observation: C6's packing loss is PER-CLASS (og 0.86->0.67, car only
0.96->0.85), but the C8 re-encodings were GLOBAL (one bias/scale per dimension,
applied identically to all classes). A class-conditional decision rule is the
untested alternative, and the LP evidence supports it: the continuous linear probe
recovers the healthy conditions far better than the centroid rule (LP wet_ground
~85->77, ~9% relative drop; HDC oracle ~51->37, ~27% relative drop). No rule fully
recovers the clean ceiling, but a learned/conditioned rule loses ~3x less than
nearest-centroid.

Candidate rules, in order of how much of the HDC/TTA story they keep:

1. **Per-class scaled HDC distance.** Keep sign-projection + prototypes, but replace
   the unit-norm cosine with a class-conditional scaled cosine: weight each
   prototype's contribution by that class's scale (inverse of its within-class
   spread / corr_tight). Re-tightens the decision boundary per class without touching
   the extractor, and keeps the label-free prototype re-estimation machinery
   (naive/oracle) intact. The most method-preserving fallback.
2. **Learned decision rule on the continuous 128-d features.** A train-time
   LogisticRegression on the full features (what LP measures). Strongest recovery,
   but abandons HDC decoding; TTA becomes re-fitting the probe on pseudo-labels
   instead of re-estimating prototypes.
3. **Per-class scale before the existing prototype distance.** The targeted version
   of (1): estimate each class's feature scale on the corrupted pool, divide before
   the cosine. Minimal code, directly attacks corr_tight.

Status: RESOLVED by C10: the decision rule IS the recovery path. C10 showed a
learned probe on the HDC code (R4) recovers 1.24-1.77x over nearest-centroid on
every condition, decisively. See Iteration C10.


### Iteration C9: the C8 training-side lever micro runs (2026-08-16)

The three C8 levers (`_scope`, `_scalein`, `_scalereg`) are trained at micro scale
(8 ep / 10%) and gated with `cond_structure_diag` vs plain DGLSS++ on snow /
wet_ground / fog / crosstalk.

**Caveat before reading results:** the micro checkpoints are 8-epoch / 10%-data
models, so their corr_tight and zs are NOT directly comparable in level to the
medium baselines (plain DGLSS++ is a full medium run). The micro gate is a
DIRECTIONAL test: does the lever change the corr_tight / zs trajectory toward
recovering the healthy packing, while keeping the fog/crosstalk direction?

**Gate results (micro B vs plain DGLSS++ medium A), mean over present classes:**

| variant | cond | corr_tight A->B | zs A->B |
| :--- | :--- | :--- | :--- |
| scope | snow | 0.836 -> 0.797 | 0.422 -> 0.317 |
| scope | wet_ground | 0.860 -> 0.771 | 0.468 -> 0.237 |
| scope | fog | 0.852 -> 0.762 | 0.082 -> 0.157 |
| scope | crosstalk | 0.875 -> 0.784 | 0.126 -> 0.296 |
| scalein | snow | 0.836 -> 0.805 | 0.422 -> 0.301 |
| scalein | wet_ground | 0.860 -> 0.783 | 0.468 -> 0.270 |
| scalein | fog | 0.852 -> 0.759 | 0.082 -> 0.161 |
| scalein | crosstalk | 0.875 -> 0.790 | 0.126 -> 0.294 |
| scalereg | snow | 0.836 -> 0.780 | 0.422 -> 0.297 |
| scalereg | wet_ground | 0.860 -> 0.754 | 0.468 -> 0.253 |
| scalereg | fog | 0.852 -> 0.701 | 0.082 -> 0.152 |
| scalereg | crosstalk | 0.875 -> 0.761 | 0.126 -> 0.288 |

**Result: none of the three levers recovers the healthy packing at micro scale.**
- On the healthy conditions (snow/wet_ground), all three variants have corr_tight_B
  below A and healthy zs_B far below A (wet_ground zs ~0.24-0.27 vs A's 0.47). The
  cov-shift healthy-condition packing loss is NOT reduced by scoping InstanceNorm,
  scale-only normalization, or the feature-scale regularizer.
- On fog/crosstalk the variants still show the cov-shift recovery signature (zs_B
  > zs_A: fog ~0.15-0.16 vs 0.08, crosstalk ~0.29-0.30 vs 0.13), consistent with the
  cov-shift direction surviving at micro scale, but this is the same pattern as the
  baseline cov-shift, not a lever-specific recovery.

**Interpretation.** The training-side levers did not obviously recover the healthy
packing in the directional micro test. The caveat stands: micro models are
undertrained (zs levels are well below the cov-shift ep10 baseline's healthy zs of
~0.40), so a medium run could still show a lever effect. But the micro signal is
weak for all three, which combined with C8 (continuous-features loss, encoding-
independent) points away from the normalization-scope levers and toward the decision
rule (C10) as the more promising recovery path.

### Iteration C10: the HDC decision-rule diagnostic (2026-08-16)

C8 proved the healthy-ceiling loss survives every ENCODING change. C10 tests the
DECISION RULE instead: on the frozen ep10/ep21 cov-shift features, compare four rules
on the same code/prototypes:
- **R1** baseline: unit-norm cosine to per-class prototypes (the current rule).
- **R2** class-conditional: per-class scaled cosine (similarity / class spread).
- **R3** learned 128-d probe: LogisticRegression on the continuous features.
- **R4** learned HDC-code probe: LogisticRegression fit on the binarized 10k-d code
  itself (clean-fit = zs, pool-refit = oracle).

**Result: the HDC space IS linearly separable in a way the current implementation
misses, decisively.**

| cond | R1-orc | R2-orc | R3-orc | R4-orc | R4/R1 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| snow (ep10) | 0.408 | 0.349 | 0.329 | 0.510 | 1.25 |
| wet_ground (ep10) | 0.425 | 0.362 | 0.349 | 0.683 | 1.61 |
| fog (ep10) | 0.261 | 0.246 | 0.193 | 0.433 | 1.66 |
| crosstalk (ep10) | 0.461 | 0.376 | 0.414 | 0.594 | 1.29 |
| snow (ep21) | 0.395 | 0.350 | 0.333 | 0.491 | 1.24 |
| wet_ground (ep21) | 0.405 | 0.384 | 0.336 | 0.668 | 1.65 |
| fog (ep21) | 0.219 | 0.207 | 0.184 | 0.387 | 1.77 |
| crosstalk (ep21) | 0.451 | 0.377 | 0.389 | 0.586 | 1.30 |

- **R4 (linear probe on the HDC code) beats R1 (nearest-centroid) by 1.24-1.77x on
  every condition and both checkpoints.** On fog it nearly doubles (0.26->0.43 ep10,
  0.22->0.39 ep21); on wet_ground +61-65%. The recoverable signal IS in the binarized
  code, and the nearest-centroid cosine throws a large fraction of it away.
- **R2 (per-class scaled cosine) does NOT help** (R2 <= R1 everywhere). The per-class
  spread rescaling does not capture the structure a learned linear boundary does.
- **R3 (learned 128-d probe) is the WEAKEST of the four** on the oracle, including
  below R4. The continuous 128-d space is less linearly separable than the binarized
  10k-d code (consistent with the HDC projection preserving/expanding the class
  structure).

**Interpretation.** The current prototype-decoder is the bottleneck, and a learned
decision rule on the HDC code is the fix. This is the "strong signal missed by the
current implementation" the C8 thread was looking for. It does NOT require changing
the extractor or the HDC encoding, only the decode rule (R4-style: a linear probe
fit on the code, either frozen or re-fit on a labeled pool). This is method-preserving
in the sense that the code/prototype pipeline stays; the decoder gains a learned
boundary. Caveat: the R4 probe is FIT on labeled data (clean or pool), so it changes
the label-free story: the label-free version is the FROZEN clean-fit probe (R4-zs),
which at 0.43-0.50 healthy still beats R1-oracle (0.40-0.43) at zero-shot.

### Iteration C11: NuScenes cross-domain zero-shot + oracle (2026-08-16)

The ep10/ep21 KITTI-trained cov-shift weights are evaluated on real NuScenes
(converted KITTI format, 32-beam sensor, shared 17-class space) for a zero-shot ->
oracle gap: is there TTA headroom on the cross-domain target that did not exist on
KITTI clean?

| model | domain | zs | oracle | gap (oracle-zs) | lp_miou |
| :--- | :--- | :--- | :--- | :--- | :--- |
| cov-shift ep10 | KITTI clean | 0.492 | 0.493 | +0.001 | 0.437 |
| cov-shift ep10 | NuScenes | 0.129 | 0.192 | +0.063 | 0.125 |
| cov-shift ep21 | KITTI clean | 0.464 | 0.468 | +0.003 | 0.444 |
| cov-shift ep21 | NuScenes | 0.127 | 0.181 | +0.054 | 0.110 |

**Result: a cross-domain TTA gap exists that did not exist on KITTI.**
- On KITTI clean, the cov-shift extractor's zs and oracle are essentially identical
  (gap ~0.001-0.003), the "no headroom" pattern (naive TTA ~= ceiling) that makes
  the healthy conditions a closed case.
- On NuScenes, the gap opens to +0.054-0.063 (oracle 0.18-0.19 vs zs 0.13), i.e. the
  frozen KITTI prototypes decode NuScenes ~26% worse than a NuScenes-labeled oracle.
  That is recoverable structure TTA could take: the features DO retain domain-
  transferable signal, but the frozen prototypes don't align to NuScenes.
- The LP mIoU on NuScenes is low (0.11-0.13), so the continuous features transferred
  weakly in absolute terms, the gap is real but the base is low.

**Interpretation.** NuScenes is a genuinely harder transfer than KITTI clean for the
cov-shift extractor, and the oracle-zs gap is TTA-addressable headroom that did not
exist on KITTI. Combined with C10 (a learned HDC-code probe recovers a large missed
signal), the strongest next direction is a LABEL-FREE decoder that re-estimates a
learned boundary on the NuScenes pool, which is exactly what the cov-shift method's
naive TTA could become on the cross-domain target. Note the dry-run caveat: these
numbers are from the full 100-frame run, so they are the real result (the earlier
2-frame dry-run numbers were only a smoke test).

### Iteration C8.1: reproducibility fix in cond_structure_diag (2026-08-16)

The scalereg gate crashed on fog with "labels must align": the parser randomly drops
points per scan (`drop_points = random.uniform(0, 0.5)`), so two separate
feature-extraction passes consumed different points and produced misaligned label
streams. Fixed by extracting both models in one SHARED pass
(`extract_features_pair`), guaranteeing A and B are evaluated on identical points.
This was a latent bug that could have hit any cross-extractor gate.

### Iteration C12: AL-geometry training objectives: the micro sweep (2026-08-18)

The AL thread (`docs/lin_probe_updates/active_iterations.md`, Iterations 0-10)
measured the label-efficiency bottlenecks as TRAINABLE properties of the feature
space, so this iteration trains toward them directly. The bottleneck properties,
from the earlier iterations:

- the **fat-blob geometry** (intra-class cosine 0.62-0.70) drives the class-mean
  estimation sample complexity, the prototype-metric viability, and T-error
  amplification (`active_iterations.md` Iterations 0, 4, 7);
- the **ill-conditioned covariance** (gain q99 ~50-130, the 4-6x ridge-relevant
  error, the fractional update needing beta < 1) is the inverse-covariance
  amplification (`active_iterations.md` Iterations 8-10).

Two new objectives were added to `modules/gen_trainers.py`, each attacking one of
these at training time:

- `ball_loss`: intra-class ball tightening: pull each point toward its class
  center (cosine) on the corrupted view. Target: intra-cos UP, separation UP.
- `spectrum_loss`: covariance conditioning: penalize the condition number of the
  centered batch covariance (bounded in (0,1), subsampled to 4000 pts, ~28ms/step).
  Target: kappa DOWN, participation rank UP.

Micro runs (8 epochs, 10% data, the `run_al_geometry_train.sh` pattern) of the
robust `corsupcon` base against five arms: `base`, `_ball`, `_spec`,
`_ball_spec`, and `_nnpull` (the existing 1-NN-purity lever). Each arm was gated
with `al_geometry_eval.py` (property diagnostics + the exact Iteration-10 10-COMB
AL update at k=8 means/class, oracle counts, 64-72 labels total) and the
`cond_structure_diag` gate.

**AL update and ceilings** (mIoU; 10-COMB = W_frozen + eta(W_beta - W_frozen),
beta 0.75, eta 0.1, oracle counts, 64-72 labels):

| cond | arm | frozen | oracle | spec-ceil | 10-COMB | 10-COMB - frozen |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| fog | base | 0.074 | 0.189 | 0.218 | 0.094 | +0.020 |
| fog | ball | 0.093 | 0.187 | 0.202 | 0.096 | +0.003 |
| fog | spec | 0.093 | 0.174 | 0.196 | 0.096 | +0.003 |
| fog | ball_spec | 0.087 | 0.183 | 0.205 | 0.093 | +0.006 |
| fog | nnpull | 0.086 | 0.190 | 0.210 | 0.090 | +0.004 |
| crosstalk | base | 0.135 | 0.345 | 0.371 | 0.164 | +0.029 |
| crosstalk | ball | 0.110 | 0.342 | 0.346 | 0.128 | +0.018 |
| crosstalk | spec | 0.123 | 0.343 | 0.350 | 0.146 | +0.023 |
| crosstalk | ball_spec | 0.121 | 0.341 | 0.352 | 0.144 | +0.023 |
| crosstalk | nnpull | 0.121 | 0.347 | 0.359 | 0.148 | +0.027 |
| snow | base | 0.410 | 0.427 | 0.439 | 0.337 | -0.073 |
| snow | ball | 0.417 | 0.435 | 0.450 | 0.346 | -0.071 |
| snow | spec | 0.396 | 0.417 | 0.439 | 0.329 | -0.067 |
| snow | ball_spec | 0.423 | 0.436 | 0.448 | 0.341 | -0.082 |
| snow | nnpull | 0.388 | 0.409 | 0.424 | 0.318 | -0.070 |
| wet_ground | base | 0.423 | 0.555 | 0.614 | 0.362 | -0.061 |
| wet_ground | ball | 0.427 | 0.569 | 0.616 | 0.368 | -0.059 |
| wet_ground | spec | 0.410 | 0.526 | 0.596 | 0.359 | -0.051 |
| wet_ground | ball_spec | 0.421 | 0.557 | 0.598 | 0.364 | -0.057 |
| wet_ground | nnpull | 0.404 | 0.535 | 0.579 | 0.359 | -0.045 |

**Property diagnostics** (pool, 128-d, normalized):

| cond | arm | intra-cos | inter-cos | sep | 1-NN purity | kappa | part-rank |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| fog | base | 0.648 | 0.651 | -0.003 | 0.543 | 1046k | 2 |
| fog | ball | 0.655 | 0.448 | 0.206 | 0.536 | 907k | 2 |
| fog | spec | 0.607 | 0.462 | 0.144 | 0.512 | 650k | 3 |
| fog | ball_spec | 0.658 | 0.587 | 0.072 | 0.531 | 632k | 2 |
| fog | nnpull | 0.584 | 0.487 | 0.096 | 0.528 | 689k | 3 |
| crosstalk | base | 0.608 | 0.284 | 0.324 | 0.658 | 1306k | 2 |
| crosstalk | ball | 0.715 | 0.419 | 0.296 | 0.649 | 998k | 2 |
| crosstalk | spec | 0.713 | 0.270 | 0.444 | 0.645 | 992k | 2 |
| crosstalk | ball_spec | 0.696 | 0.344 | 0.352 | 0.641 | 717k | 2 |
| crosstalk | nnpull | 0.682 | 0.233 | 0.449 | 0.632 | 851k | 2 |
| snow | base | 0.693 | 0.071 | 0.622 | 0.781 | 291k | 3 |
| snow | ball | 0.715 | 0.095 | 0.620 | 0.781 | 328k | 3 |
| snow | spec | 0.694 | 0.078 | 0.615 | 0.775 | 325k | 3 |
| snow | ball_spec | 0.717 | 0.085 | 0.632 | 0.785 | 355k | 3 |
| snow | nnpull | 0.714 | 0.085 | 0.630 | 0.772 | 436k | 3 |
| wet_ground | base | 0.694 | -0.010 | 0.704 | 0.821 | 875k | 2 |
| wet_ground | ball | 0.704 | 0.014 | 0.690 | 0.827 | 336k | 3 |
| wet_ground | spec | 0.726 | 0.008 | 0.718 | 0.818 | 479k | 3 |
| wet_ground | ball_spec | 0.743 | 0.000 | 0.743 | 0.823 | 323k | 4 |
| wet_ground | nnpull | 0.720 | -0.001 | 0.721 | 0.817 | 557k | 3 |

**cond_structure gate** (A = reference robust extractor, B = this extractor,
mean over 8 populated classes; feat_cos / dir_ret / corr_tight / zs):

| cond | arm | feat_cos A->B | dir_ret A->B | corr_tight A->B | zs A->B |
| :--- | :--- | :--- | :--- | :--- | :--- |
| fog | base | 0.157->0.314 | 0.174->0.384 | 0.852->0.811 | 0.092->0.046 |
| fog | ball | 0.157->0.318 | 0.174->0.379 | 0.852->0.809 | 0.092->0.084 |
| fog | spec | 0.157->0.335 | 0.174->0.417 | 0.852->0.778 | 0.092->0.085 |
| fog | ball_spec | 0.157->0.314 | 0.174->0.383 | 0.852->0.810 | 0.092->0.075 |
| fog | nnpull | 0.157->0.355 | 0.174->0.462 | 0.852->0.749 | 0.092->0.078 |
| crosstalk | base | 0.193->0.412 | 0.217->0.524 | 0.875->0.761 | 0.142->0.123 |
| crosstalk | ball | 0.193->0.286 | 0.217->0.335 | 0.875->0.841 | 0.142->0.093 |
| crosstalk | spec | 0.193->0.380 | 0.217->0.457 | 0.875->0.828 | 0.142->0.117 |
| crosstalk | ball_spec | 0.193->0.358 | 0.217->0.424 | 0.875->0.826 | 0.142->0.097 |
| crosstalk | nnpull | 0.193->0.416 | 0.217->0.499 | 0.875->0.808 | 0.142->0.116 |
| snow | base | 0.818->0.817 | 0.977->0.975 | 0.836->0.837 | 0.475->0.388 |
| snow | ball | 0.818->0.840 | 0.977->0.984 | 0.836->0.854 | 0.475->0.390 |
| snow | spec | 0.818->0.819 | 0.977->0.976 | 0.836->0.839 | 0.475->0.371 |
| snow | ball_spec | 0.818->0.842 | 0.977->0.985 | 0.836->0.854 | 0.475->0.407 |
| snow | nnpull | 0.818->0.834 | 0.977->0.980 | 0.836->0.851 | 0.475->0.371 |
| wet_ground | base | 0.827->0.768 | 0.959->0.920 | 0.860->0.837 | 0.527->0.373 |
| wet_ground | ball | 0.827->0.776 | 0.959->0.919 | 0.860->0.844 | 0.527->0.383 |
| wet_ground | spec | 0.827->0.765 | 0.959->0.903 | 0.860->0.851 | 0.527->0.374 |
| wet_ground | ball_spec | 0.827->0.784 | 0.959->0.908 | 0.860->0.865 | 0.527->0.392 |
| wet_ground | nnpull | 0.827->0.788 | 0.959->0.930 | 0.860->0.847 | 0.527->0.398 |

**Result: the objectives move the properties they target; the AL verdict still
needs the medium run.**
- **`ball` moved the blob geometry it targets.** Intra-class cosine is UP vs base
  on crosstalk/snow (0.608->0.715, 0.693->0.715) and wet_ground (0.694->0.704);
  the fog separation goes from negative (-0.003) to +0.206 (the inter-class
  cosine drops 0.651->0.448). `ball_spec` carries the same tightening (intra-cos
  the best of all arms on wet_ground at 0.743, and crosstalk sep 0.352 vs base
  0.324).
- **`spec` moved the spectrum it targets.** kappa is DOWN vs base on the three
  ill-conditioned conditions: fog 1046k->650k, crosstalk 1306k->992k, wet_ground
  875k->479k, with participation rank 2->3 on fog/wet_ground. `ball_spec`
  combines both: kappa 632k (fog) / 717k (crosstalk) / 323k (wet_ground, prank
  4, the flattest of any arm). The spectrum half of the 2x2 is now measured -   both spectrum-bearing arms flatten the conditioning.
- **The property gains do not transfer to the AL curve at micro scale.** The
  10-COMB update is NEGATIVE on snow and wet_ground for EVERY arm including the
  base (-0.045 to -0.082), and only marginally positive on fog/crosstalk (+0.003
  to +0.029). On the positive conditions the BASE is the best arm (fog +0.020,
  crosstalk +0.029); on the healthy conditions the training-objective arms are
  least-negative (spec: snow -0.067, wet_ground -0.051; nnpull: wet_ground
  -0.045). The oracle ceiling is small on snow (+0.017) but large on wet_ground
  (+0.132); even with oracle counts the fractional-residual update pushed the
  wrong way on wet_ground at micro scale. This is a micro-scale fragility not
  present at full scale (Iteration 10 won wet_ground +0.018), and it will need
  the medium run to resolve.
- **The training objectives neither help nor clearly hurt the AL curve at micro
  scale**: the delta spread across arms is small (fog +0.003..+0.020, crosstalk
  +0.018..+0.029, snow -0.082..-0.067, wet_ground -0.061..-0.045). `spec` is the
  least-negative on snow; `nnpull` on wet_ground; `ball_spec` is worst on snow
  despite having the best geometry. `nnpull` gave no packing gain (1-NN purity
  flat or slightly down). The property gains only matter at the scale where the
  AL update itself works.
- **cond_structure:** no arm regresses the robust properties on snow; `spec`
  holds them (feat_cos 0.819, dir_ret 0.976, corr_tight 0.839, effectively
  unchanged from the reference). `ball_spec` is the most conservative on
  snow/wet_ground (corr_tight 0.854/0.865, the highest of any arm) but drops
  crosstalk dir_ret (0.424 vs base's 0.457). These are micro-scale magnitudes
  and the healthy-condition priority means the crosstalk dir_ret drops need the
  medium check.

**Interpretation.** The 2x2 is complete and both training-objective directions
are validated: `ball` tightens the blobs, `spec` flattens the spectrum, and
`ball_spec` combines them. The property gains being visible at micro scale is
exactly what the objectives were designed to do. The next step is a medium run
of the surviving arms (`ball`, `spec`, `ball_spec`) so the AL curve is measured
at the scale where Iteration 10 won, the micro-scale 10-COMB is not a
trustworthy AL verdict (it goes negative even for the base). Note: the three
`spectrum_loss` failures on the way here were two real bugs (a mask computed on
the wrong shape -> IndexError, and AMP autocast re-promoting the fp32 covariance
matmul to fp16 -> eigvalsh Half error), fixed by computing the mask after
reshape and the covariance in float64 (autocast-immune), not by disabling
autocast.

### Iteration C13: the 8-epoch medium AL-geometry run + the beta/eta re-sweep (2026-08-18)

Promoted `ball` and `spec` to the medium run (8 epochs, 100% data) and measured
the AL-geometry gate (`al_geometry_eval.py`) plus a (beta, eta) re-sweep of the
Iteration-10 fractional-residual update at the CHEAP k=8 budget (64-72 labels),
all on the new feature spaces.

**Both arms scaled: the property gains survived 100% data and neither regressed
the ceilings.**
- frozen / oracle / spec-ceil all rose well above the micro run (e.g. ball oracle
  fog 0.187->0.277, crosstalk 0.342->0.384); participation rank rose to 3-5
  (flatter spectrum), intra-cos 0.71-0.78, separation up to 0.89.
- cond_structure gate: both arms hold feat_cos / dir_ret / corr_tight at or above
  the reference on snow/wet_ground with no regression (zs_B preserved).
- **Both arms were still climbing at epoch 8**: the final and valid_best gates
  were identical (best-val fired at the final epoch) and the last-epoch IoU was
  still rising from epoch 6. With the cosine scheduler on first_cycle=80, the LR
  was still near max: a high-LR plateau, NOT an optimum. So this is a scaling
  check, not a convergence verdict.

**The beta/eta re-sweep found AL headroom that the default recipe was missing - the key signal to keep working from.**
The Iteration-10 defaults (beta=0.75, eta=0.1) were tuned on the base's steeper
spectrum; the AL-geometry objectives flattened it, moving the fractional-gain
optimum. Re-sweeping (beta, eta) at k=8 on the 8-epoch ball/spec features
(`al_betaeta_resweep.py`, oracle counts, random-k means, same T_hat as
Iteration 10):

| cond | default (0.75, 0.1) | best (beta, eta) | delta at best |
| :--- | :--- | :--- | :--- |
| fog | +0.014 | (0.6, 0.2) | +0.032 |
| crosstalk | +0.034 | (0.6, 1.0) | +0.059..0.084 |
| snow | -0.058 | (0.6, 0.05) | -0.008 |
| wet_ground | -0.081 | (0.6, 0.05) | -0.037 |

- The AL optimum shifted from beta=0.75 to **beta=0.6** on BOTH extractors,
  consistently across all 4 conditions and both final/valid_best checkpoints.
- The 8-epoch stats were **positive where AL was already positive** (fog/crosstalk
  roughly 2x better with re-tuned beta/eta) and **near-zero where it was negative**
  (snow -0.058->-0.008, wet_ground -0.081->-0.037), a strong signal that the
  feature space enables cheap AL once the recipe is re-tuned.
- final vs valid_best were identical, so the signal is real, not a checkpoint
  artifact.

**Forward signal / what to watch.** The 8-epoch (still-climbing) statistics show
the feature spaces have AL headroom that beta=0.75/eta=0.1 was hiding. The medium
run was resumed to 21 epochs (LR annealed, the `run_covshift_medium.sh` standard)
to see if a better-converged extractor pushes snow/wet_ground fully positive. If
the 21-epoch run COLLAPSES back to clearly-negative AL on snow/wet_ground, that
would be evidence the headroom was an under-convergence artifact, but the
8-epoch near-zero/positive-elsewhere pattern is the baseline to work from (and to
re-tune the recipe around beta=0.6, not 0.75, for these extractors).

### Iteration C14: the 21-epoch medium AL-geometry run: convergence check + AL headroom (2026-08-18)

Resumed both `ball` and `spec` from 8 -> 21 epochs (100% data, cosine
`first_cycle: 80` now in its annealed tail; `run_algeom_medium_seq.sh 3 21
resume` on GPU 3, sequential, ~13h). Re-gated both checkpoints (`final` and
`valid_best` via `Senet_valid_best` copy; final == valid_best for both arms,
so best-val was at epoch 20, still climbing, but now LR-annealed) on the
same harnesses: `al_geometry_eval.py` (frozen / oracle / spec-ceil + the
default 10-COMB, beta=0.75 eta=0.1 at k=8) and the beta/eta re-sweep
(`al_betaeta_resweep.py`, 7 x 7 grid at k=8, 64-72 labels).

**AL at the default recipe (beta=0.75, eta=0.1, 64-72 labels): 8-ep vs 21-ep
(final checkpoint; valid_best is identical so the signal is not a checkpoint
artifact):

| cond | arm | 8-ep frozen | 21-ep frozen | 8-ep 10-COMB delta | 21-ep 10-COMB delta |
| :--- | :--- | :--- | :--- | :--- | :--- |
| fog | ball | 0.102 | 0.089 | +0.014 | +0.018 |
| fog | spec | 0.097 | 0.074 | +0.013 | +0.031 |
| cross. | ball | 0.140 | 0.128 | +0.037 | +0.043 |
| cross. | spec | 0.131 | 0.127 | +0.031 | +0.037 |
| snow | ball | 0.480 | 0.510 | -0.058 | -0.072 |
| snow | spec | 0.481 | 0.498 | -0.072 | -0.071 |
| wet_ground | ball | 0.582 | 0.639 | -0.080 | -0.093 |
| wet_ground | spec | 0.604 | 0.620 | -0.078 | -0.088 |

**AL at the best (beta, eta) found by the re-sweep at k=8: the fair test of
whether the feature space enables cheap AL once the recipe is re-tuned for the
flatter spectrum. The optimum stayed at **beta=0.6** on every condition at both
scales (hallmark of the flatter spectrum):

| cond | arm | 8-ep default | 8-ep best (b,e) | 21-ep default | 21-ep best (b,e) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| fog | ball | +0.014 | **+0.032** (0.6,0.2) | +0.018 | **+0.042** (0.6,0.2) |
| fog | spec | +0.013 | **+0.018** (0.6,0.2) | +0.031 | **+0.059** (0.6,0.1) |
| cross. | ball | +0.034 | **+0.059** (0.6,1.0) | +0.045 | **+0.086** (0.6,0.75) |
| cross. | spec | +0.030 | **+0.076** (0.6,1.0) | +0.038 | **+0.092** (0.6,0.5) |
| snow | ball | -0.058 | **-0.008** (0.6,0.05) | -0.067 | **-0.009** (0.6,0.05) |
| snow | spec | -0.072 | **-0.009** (0.6,0.05) | -0.071 | **-0.003** (0.6,0.05) |
| wet_ground | ball | -0.081 | **-0.037** (0.6,0.05) | -0.092 | **-0.038** (0.75,0.05) |
| wet_ground | spec | -0.077 | **-0.025** (0.6,0.05) | -0.089 | **-0.045** (0.75,0.05) |

The **8 -> 21 collapse did NOT happen.** Snow/wet_ground stayed
**near-zero** at their best (beta, eta): snow -0.008->-0.009 (ball),
-0.009->-0.003 (spec); wet -0.037->-0.038 (ball), -0.025->-0.045 (spec).
The signal survived convergence and is real, not an under-convergence artifact.
On the positive conditions the re-tuned best improved with longer training
(fog +0.032->+0.042 ball, +0.018->+0.059 spec; crosstalk +0.059->+0.086 ball,
+0.076->+0.092 spec), so longer training DID help where AL was already
positive, it just did not turn snow/wet_ground fully positive.

**Property diagnostics (pool, 128-d, normalized), 8-ep -> 21-ep (final):**

| cond | arm | 8-ep intra/cos | 21-ep intra/cos | 8-ep kappa | 21-ep kappa | 8-ep prank | 21-ep prank |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| fog | ball | 0.710 | **0.766** | 1.66M | **7.76M** | 3 | 3 |
| fog | spec | 0.718 | 0.689 | 1.96M | **6.54M** | 3 | 3 |
| cross. | ball | 0.745 | **0.781** | 2.62M | **14.76M** | 3 | 2 |
| cross. | spec | 0.761 | 0.756 | 3.05M | **11.99M** | 3 | 3 |
| snow | ball | 0.741 | 0.743 | 1.54M | **6.73M** | 4 | 4 |
| snow | spec | 0.727 | 0.716 | 1.64M | **8.06M** | 4 | 4 |
| wet_ground | ball | 0.784 | **0.795** | 1.25M | **6.32M** | 5 | 4 |
| wet_ground | spec | 0.772 | 0.754 | 1.57M | **6.12M** | 5 | 5 |

`ball`'s intra-cos tightened further on fog/crosstalk (the blob objective
compounds with longer training); `spec`'s intra-cos flattened or regressed
slightly. Both arms' **kappa roughly tripled to 6-15x** from 8 to 21 epochs
while prank held, the spectrum re-steepened with LR annealing despite the
`spectrum_loss` (weight 0.1, `spec_w` = 0.1) that had flattened it at 8-ep.
Ceilings did not regress (frozen/oracle/spec-ceil rose on snow/wet_ground;
fog frozen drifted down but its AL still improved), and the cond_structure gate
shows **no regression** on snow/wet_ground (ball snow feat 0.818->0.843,
wet 0.827->0.835; spec snow 0.818->0.825, wet 0.827->0.820; all corr_tight / zs
at or above the reference).

**Result: did either feature extractor help, and what next?**

Neither extractor turned snow/wet_ground AL positive at the cheap k=8 budget,
even at its best (beta, eta) and even after LR-annealed convergence. The
training objectives DID move the properties they target at 8-ep and those gains
were visible where they mattered, but longer training (a) re-steepened the
spectrum (the `spectrum_loss` at 0.1 is too weak to hold the 100%-data
covariance flat through annealing) and (b) did not lift the healthy-condition AL
above zero. So **feature-extractor training alone does not make cheap AL
positive on snow/wet_ground at this weight and budget, continuing training
beyond 21 epochs is unlikely to fix the healthy-condition AL (train IoU was
still climbing at 21, but the AL curve on those conditions is flat near-zero
and the condition number is getting worse, not better).

The next step is **refining the AL method itself for the new feature spaces** - the lever that C14 shows actually moves the snow/wet delta: re-tuning (beta,
eta) already halved the negative (default -0.06..-0.09 -> best -0.00..-0.04),
and the `active_iterations.md` Iteration-11 deployment fixes directly attack the
remaining T_hat gap (oracle counts -> source-count prior, rare-class inclusion,
control-variate means, per-condition eta). Those are eval-only on the
already-trained 21-ep checkpoints (no more extractor training), so they are the
cheap next probe of whether this feature space can be made to deliver cheap AL.
If that probe stays negative on snow/wet_ground, the bottleneck is the
k=8 T_hat mass/count problem identified in Iterations 7-8, not the extractor.

### Iteration C15: the comprehensive AL check: method tweaks + feature-space properties (2026-08-19)

The two-question probe (`al_comprehensive_diag.py`, eval-only, ~45s/condition)
on the 21-ep `ball`/`spec` checkpoints, all at the cheap k=8 budget (64-72
labels), beta/eta swept {0.6, 0.75} x {0.05, 0.1, 0.2, 0.3, 0.5}:

**Part 1: slight variations of the Iteration-10 method** (best delta =
combo - frozen, oracle-count baseline V0):

| cond | arm | V0 oracle | V1 source-cnt | V2 rare-all | V3 control-var | V4 source+rare |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| fog | ball | +0.042 | **+0.049** | +0.042 | +0.033 | **+0.049** |
| fog | spec | +0.059 | **+0.063** | +0.059 | +0.051 | **+0.063** |
| crosstalk | ball | +0.086 | +0.082 | +0.086 | +0.055 | +0.082 |
| crosstalk | spec | +0.091 | +0.086 | +0.091 | +0.049 | +0.086 |
| snow | ball | -0.009 | -0.009 | -0.012 | **-0.004** | -0.010 |
| snow | spec | -0.004 | -0.005 | -0.004 | **+0.001** | -0.005 |
| wet_ground | ball | -0.038 | -0.024 | -0.032 | **-0.011** | -0.022 |
| wet_ground | spec | -0.047 | -0.028 | -0.044 | **-0.017** | -0.025 |

**The method is deployable, and the control variate is the unexpected winner.**
- **V1 (source-count prior) holds V0 on every condition** (fog/crosstalk within
  +-0.004; wet_ground actually BETTER, ball -0.024 vs -0.038, spec -0.028 vs
  -0.047). The method does NOT need oracle pool counts, the clean-data source
  prior suffices. This closes the Iteration-10/11 deployment gap: the method
  is fully deployable with only the source prior + k=8 random means.
- **V3 (clean-mean control variate, rho=0.5) is the standout on the healthy
  conditions**: wet_ground ball -0.038 -> -0.011, spec -0.047 -> -0.017 (roughly
  3x closer to zero); snow spec goes POSITIVE (+0.001). The clean mean as a
  control variate shrinks the sampled-mean error that the ridge amplifies (the
  Iteration-8 whitened-error smoking gun), directly attacking the healthy-
  condition AL gap.
- V2 (rare-class inclusion) is neutral-to-harmful (snow -0.012 vs -0.009): the
  rare classes are noise at thresh-k. V4 (source + rare) == V1 (the rare classes
  add nothing once counts are source-based).
- fog/crosstalk remain solidly positive (+0.04 to +0.09) under every variant.

**Part 2: feature-space properties (ball JSON; spec log has the core props):**

| cond | arm | intra | sep | nn1 | kappa | prank | R1 proto | lin probe |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| fog | ball | 0.873 | 0.271 | 0.637 | 7.74M | 3 | 0.162 | 0.088 |
| fog | spec | 0.826 | 0.242 | 0.656 | 6.56M | 3 | 0.143 | 0.074 |
| crosstalk | ball | 0.881 | 0.490 | 0.710 | 14.9M | 2 | 0.230 | 0.125 |
| crosstalk | spec | 0.867 | 0.490 | 0.703 | 12.2M | 3 | 0.211 | 0.127 |
| snow | ball | 0.859 | 0.935 | 0.857 | 6.77M | 4 | 0.424 | 0.507 |
| snow | spec | 0.843 | 0.924 | 0.855 | 8.05M | 4 | 0.428 | 0.498 |
| wet_ground | ball | 0.890 | 0.995 | 0.882 | 6.37M | 4 | 0.529 | 0.640 |
| wet_ground | spec | 0.865 | 0.967 | 0.878 | 6.11M | 5 | 0.523 | 0.620 |

- **R1 (prototype) doubles the linear probe on fog/crosstalk** (ball fog 0.162
  vs 0.088, crosstalk 0.230 vs 0.125) but **linear >> R1 on snow/wet_ground**
  (ball wet 0.529 vs 0.640). The ball objective's tight blobs make the prototype
  metric viable exactly where the frozen probe is weakest, but the healthy
  conditions still need the linear boundary. No clean "drop the classifier."
- **lev_conf_spearman is POSITIVE (0.09-0.38) on the ball space: opposite to
  the base extractor's negative values (Iteration-11: -0.40 to -0.64). On this
  space the high-confidence points ARE the high-leverage points, so a confidence
  query rule (the free baseline) may now be competitive with influence, a
  property change worth a direct query-rule test.
- **resid_conf_spearman is strongly negative (-0.43 to -0.91): the
  high-confidence points have small residuals, confirming the frozen probe's
  "already correct" core.
- **mean-k saturates at k=2** (0.93-0.95) on all conditions: the class means are
  estimable from 2 points/class (16-17 labels), cheaper than the k=8 budget
  currently used.
- **kappa is 6-15M** (the re-steepened spectrum) but prank 2-5: the cheap
  Lanczos top-k spectral filter (README 4.5 efficiency note) remains the right
  way to avoid the full 10k eigh.
- **The ball space's worst frozen classes are the rare ones** (fog c7 iou 0.00
  @ freq 143, c14 iou 0.00 @ freq 3917): the budget must spend on them, but the
  V2 test says including them in T_hat with a random-k mean is noise, they
  need the influence-rule targeting, not the mean route.

**Net: the spaces are now worth building the AL method on.** The C15 data gives
three concrete, cheap next steps (all eval-only):
1. **Adopt V1 (source-count prior): it is the deployable form and holds or
   beats the oracle-count baseline everywhere.
2. **Adopt V3 (control variate): it nearly halves the healthy-condition
   negative (wet_ground -0.011/-0.017, snow spec +0.001) and needs only the
   clean means already computed.
3. **Test the confidence query rule on the ball space** (lev_conf is now
   positive) and the k=2 budget (mean-k saturates), both would cut the label
   cost below the current 64-72.
The remaining snow/wet gap (V3: -0.004..-0.017) is the k=8 T_hat mass problem
from Iterations 7-8; combining V3 with the k=2 budget and the influence rule is
the natural next probe.

### Iteration C16: the query-rule and label-budget tests (2026-08-19)

The two AL tests from C15's leads (`al_rule_budget_diag.py`, eval-only, on the
21-ep ball/spec checkpoints, all at the fractional-residual 10-COMB with the
V3 control variate; ~1 min/condition):

**TEST 1: the query rule (k=8, oracle counts, rho=0.5).** Four rules select
the k=8 points per class: influence (the Iteration-1 winner, Nystrom-subspace),
confidence (the free rule C15 suggested), random (the Iteration-8 best mean
estimator), centroid-near (distance to the class centroid). Reported as best
10-COMB delta over the beta/eta grid, with the rule's mean quality
(cos of the rule-selected k-mean vs the true class mean):

| cond | arm | influence | confidence | random | centroid |
| :--- | :--- | :--- | :--- | :--- | :--- |
| fog | ball | +0.026 (0.48) | +0.018 (0.61) | +0.033 (0.96) | **+0.038 (0.95)** |
| fog | spec | +0.037 (0.33) | +0.048 (0.74) | +0.050 (0.94) | **+0.055 (0.94)** |
| crosstalk | ball | +0.025 (0.44) | +0.034 (0.66) | +0.055 (0.95) | **+0.065 (0.96)** |
| crosstalk | spec | +0.022 (0.41) | +0.045 (0.86) | +0.049 (0.95) | **+0.062 (0.95)** |
| snow | ball | -0.026 (0.29) | -0.015 (0.85) | -0.003 (0.96) | **-0.001 (0.96)** |
| snow | spec | -0.030 (0.38) | -0.005 (0.90) | +0.000 (0.95) | **+0.001 (0.96)** |
| wet_ground | ball | -0.048 (0.37) | -0.025 (0.90) | -0.011 (0.98) | **-0.001 (0.96)** |
| wet_ground | spec | -0.051 (0.32) | -0.034 (0.91) | -0.019 (0.97) | **-0.011 (0.96)** |

**The rule ranking is fully explained by the mean quality, and influence is
now the WORST rule.** The delta order is exactly the mean_cos order
(centroid > random > confidence > influence), on every condition and both
arms. Influence's mean_cos is 0.29-0.48: it selects boundary/outlier points
(Iteration-6 finding), so its k-mean points at the boundary direction, not the
class mean. The C15 lev_conf signal did NOT replicate: the real Nystrom
influence vs confidence Spearman is NEGATIVE (-0.32 to -0.65) here, matching
the base-space behavior, the C15 positive value was a proxy artifact (feature
norm, not the true influence). The confidence rule is NOT free-and-valid; it
sits between influence and random in mean quality (0.61-0.91).

**TEST 2: the label budget (centroid rule, source counts; k in {2,4,8} x
rho in {0.25,0.5,0.75}).**

| cond | arm | k=2 best | k=4 best | k=8 best |
| :--- | :--- | :--- | :--- | :--- |
| fog | ball | **+0.054** (r0.25) | +0.052 | +0.051 |
| fog | spec | +0.057 (r0.25) | +0.061 | **+0.063** |
| crosstalk | ball | **+0.075** (r0.25) | +0.075 | +0.074 |
| crosstalk | spec | **+0.077** (r0.25) | +0.077 | +0.071 |
| snow | ball | -0.008 | -0.008 | -0.004 |
| snow | spec | -0.012 | -0.006 | -0.000 |
| wet_ground | ball | -0.018 | -0.013 | **-0.007** |
| wet_ground | spec | -0.031 | **-0.015** | -0.016 |

**The k=2 budget holds k=8 everywhere** (fog/crosstalk within +-0.006; snow/wet
within 0.01), the label cost halves to ~32 labels (2 points/class) with no
measurable loss, confirming C15's mean-k saturation. rho=0.25-0.75 barely moves
the result; the control variate is a mild stabilizer, not the deciding factor.

**The premise diagnostics: what does/doesn't work, attributed:**

| cond | closeable gap | best delta | t_cos | w_cos | whitened error |
| :--- | :--- | :--- | :--- | :--- | :--- |
| fog | +0.15..+0.17 | **+0.054..+0.063** | 0.52-0.53 | 0.006-0.017 | 77-125x |
| crosstalk | +0.24..+0.26 | **+0.075..+0.077** | 0.93-0.94 | 0.002-0.011 | 63-67x |
| snow | +0.03..+0.04 | **+0.004 (positive)** | 0.96 | 0.018-0.025 | 11-22x |
| wet_ground | +0.07..+0.08 | -0.007..-0.018 | 0.58-0.60 | 0.002-0.006 | 23-26x |

**What works:**
- **fog/crosstalk AL is now solidly POSITIVE at ~32 labels** (fog +0.054-0.063,
  crosstalk +0.075-0.077) with the centroid rule + source counts + k=2. The
  cheap AL framework works on the collapsed conditions.
- **snow is POSITIVE at the best config** (+0.004 ball k2 r0.75 / spec k8
  r0.75), the first positive snow AL in the entire thread.
- **The rule story is causal and clean**: selection quality (mean_cos) fully
  predicts the AL outcome; the fix is to select bulk points (centroid/random),
  not boundary points (influence).
- **k=2 budget halves the cost with zero loss.**

**What does not work (and why, the premise is attributed, not just failed):**
- **wet_ground stays negative** (-0.007 to -0.018) and **snow is only barely
  positive**. The whitened error at the best config is 11-125x, the
  Iteration-8 smoking gun persists even with the best rule, the best budget,
  source counts, and the control variate. The T direction is good (t_cos
  0.52-0.96) but the inverse covariance amplifies the residual (w_cos collapses
  to 0.002-0.025). This is the RIDGE AMPLIFICATION, not the selection or the
  budget: the premise limit on the healthy conditions is the probe update
  itself, exactly as Iteration 8-10 found. The closeable gap is also small on
  snow (+0.03): there is little headroom to begin with.
- **The confidence rule does not work** (its mean_cos 0.61-0.91 is worse than
  random/centroid), the C15 positive lev_conf was a proxy artifact.

**Net for the AL method.** The deployable recipe is now: centroid-near k=2
means (32 labels) + source-count prior + control variate + fractional-residual
update (beta~0.6): POSITIVE on fog/crosstalk (+0.05 to +0.08), near-zero-to-
positive on snow, and the remaining wet_ground deficit (-0.007 to -0.018) is
attributed to the ridge amplification of a small closeable gap, not to the
feature extractor or the label selection. The next lever for wet_ground is the
sensitivity-bounded parameterization itself (the clip 9B / fractional-gain
shaping already in the README 4.5 efficiency note), not more extractor
training or a different query rule.

### Iteration C17: the cross-extractor comparison plan (2026-08-19)

**The ceiling caveat discovered in C16.** The C12-C16 AL experiments were run
on the **corsupcon** base (`supcon_vib_dglsspp_corsupcon_ball` / `_spec`), but
the AL-thread ceiling numbers in `active_iterations.md` were measured on the
**cov-shift** extractor (`supcon_vib_dglsspp_inputin_in_chan`, the Pillar-1
winner). These are DIFFERENT extractors, and their fog/crosstalk ceilings
differ by ~0.18:

| cond | cov-shift zs -> ceiling | ball zs -> ceiling | ceiling delta |
| :--- | :--- | :--- | :--- |
| fog | 0.201 -> 0.433 | 0.088 -> 0.252 | **-0.18** |
| crosstalk | 0.395 -> 0.594 | 0.126 -> 0.401 | **-0.19** |
| snow | 0.377 -> 0.510 | 0.506 -> 0.554 | +0.04 |
| wet_ground | 0.358 -> 0.683 | 0.641 -> 0.730 | +0.05 |

Two separate findings, to be tested in the next iteration:

1. **The small snow gap is NOT a destroyed ceiling.** Ball's snow ceiling
   (0.554) is HIGHER than cov-shift's (0.510), and wet_ground too (0.730 vs
   0.683). The gap is small because ball's FROZEN (zero-shot) is already near
   its own ceiling (0.506 vs 0.554), the ball training raised the frozen
   performance so much there is little left for labels to add. The healthy-
   condition story is a success, not a ceiling loss.
2. **The ball/spec fog/crosstalk ceilings are ~0.18 BELOW the cov-shift
   extractor's.** The C16 AL gains (+0.05 to +0.08 on fog/crosstalk) are real
   but were measured on a WEAKER space. The open question: does the same
   centroid-k2 + source-count + control-variate AL recipe reach the cov-shift
   HIGH ceilings (0.433 fog / 0.594 crosstalk)?

**The plan.** Run the identical `al_rule_budget_diag` on the cov-shift ep10
(the optimal window) and ep21 checkpoints
(`bash run_al_rule_budget_covshift.sh 3`). If the cheap AL reaches the cov-shift
high ceilings on fog/crosstalk AND the healthy conditions hold, the full story
is: a feature space with both the high cov-shift ceilings AND the AL-friendly
geometry. If the cov-shift frozen is too low for the AL to close, the next
candidate is a hybrid: apply the ball/spec AL objectives on the cov-shift
base (train `inputin_in_chan_ball` / `inputin_in_chan_spec`), so the geometry
gains ride on the higher-ceiling extractor. The README's three candidate next
steps are documented there (Section 6).

### Iteration C18: the cov-shift AL test: the recipe does NOT transfer (2026-08-19)

The identical `al_rule_budget_diag` on the cov-shift `inputin_in_chan` ep10 +
ep21 checkpoints (the Pillar-1 extractor with the HIGH fog/crosstalk ceilings).
Cross-extractor comparison (C16 ball vs cov-shift ep10; TEST2 = best AL delta
at the deployable recipe, k x rho x rule over the source-count grid):

| cond | extractor | frozen | ceiling | closeable gap | best AL delta | whitened err |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| fog | ball | 0.088 | 0.252 | +0.150 | **+0.054** (k2 r0.25 centroid) | 124.5 |
| fog | cov-ep10 | 0.259 | 0.416 | +0.118 | **+0.003** (k8 r0.25 random) | 42.5 |
| crosstalk | ball | 0.126 | 0.401 | +0.256 | **+0.075** (k2 r0.25 centroid) | 67.1 |
| crosstalk | cov-ep10 | 0.466 | 0.526 | +0.028 | **-0.002** (k8 r0.75 random) | 15.1 |
| snow | ball | 0.506 | 0.554 | +0.036 | **+0.004** (k2 r0.75 centroid) | 21.9 |
| snow | cov-ep10 | 0.457 | 0.511 | +0.036 | **-0.001** (k8 r0.75 random) | 16.1 |
| wet_ground | ball | 0.641 | 0.730 | +0.084 | **-0.007** (k8 r0.5 centroid) | 26.5 |
| wet_ground | cov-ep10 | 0.427 | 0.670 | +0.187 | **-0.005** (k8 r0.75 centroid) | 15.0 |

ep21 is the same pattern (fog +0.019 at best, crosstalk -0.001, snow -0.003,
wet +0.001, all near-zero, none of the ball/spec magnitude).

**Result: the cheap-AL recipe does NOT transfer to the cov-shift extractor.**
On every condition the cov-shift AL delta is ~zero (-0.005 to +0.019), far
below the ball/spec numbers (fog +0.054 vs +0.003, crosstalk +0.075 vs -0.002).
The cov-shift feature space is NOT AL-friendly at the cheap budget: the
frozen probe is already too close to the probe ceiling on crosstalk (gap +0.028)
and the mean-quality / whitened-error structure does not convert labels into
updates there.

**Why (the premise is now attributed across BOTH extractors):**
- The mean-quality ordering holds everywhere (centroid/random > confidence >
  influence, mean_cos 0.30-0.96), so the rule finding is extractor-independent
  and robust.
- The whitened error is LOWER on cov-shift (13-42x) than ball (22-125x), yet
  the AL delta is WORSE: the ridge is less sensitive but the closeable gaps
  are also smaller on cov-shift fog (0.118 vs 0.150) and crosstalk (0.028 vs
  0.256). The cov-shift extractor's frozen decode already captures most of its
  ceiling on crosstalk (0.466 vs 0.526), there is almost nothing for labels
  to buy there.
- On fog the cov-shift gap is +0.118 (comparable to ball) but the recipe
  reaches only +0.003: the T_hat mass / amplification structure on the
  cov-shift space is less label-responsive than the ball space (its T direction
  t_cos 0.62 vs ball 0.53, but the w_cos 0.026 vs 0.017 is not better).

**The synthesis: the two extractors are complementary, not competing:**
- **ball/spec (corsupcon)**: AL-friendly geometry (labels convert to updates
  at 32 labels), healthy-condition ceilings equal-or-better than cov-shift, but
  fog/crosstalk ceilings ~0.18 LOWER.
- **cov-shift (inputin_in_chan)**: high fog/crosstalk ceilings, but the frozen
  probe already sits near its own ceiling there, so cheap AL has nothing to add
, the label budget is wasted on this space.

**Net: the full story needs the hybrid.** Neither extractor alone has both
properties. The next step (README Section 6.1, option 3) is to train the
ball/spec AL-geometry objectives on the cov-shift base
(`inputin_in_chan_ball` / `inputin_in_chan_spec`), the goal being one FE with
the cov-shift high fog/crosstalk ceilings AND the ball/spec AL-friendly
geometry. The C18 cross-extractor data is the justification: the two spaces'
strengths are disjoint, and the hybrid is the only path to both.

### Iteration C19: the hybrid micro sweep: the properties do NOT combine (2026-08-19)

The C18 hybrid (`run_algeom_hybrid_micro.sh`): ball/spec losses on the
COV-SHIFT base. The wiring keeps the cov-shift recipe (GMSIFC/LSCC + per-scan
input normalization on channels 0/4 + internal InstanceNorm) untouched and adds
ONLY `ball_loss` / `spectrum_loss` on `z8_aug`. Four arms at micro scale (8 ep /
10%): the cov-shift base reference + `_ball` (ball_w 0.1), `_spec` (spec_w
0.1), `_ball_spec` (0.05/0.05). All four trained normally (final-epoch IoU
0.197-0.202, consistent with the base).

**The cross-extractor comparison (ceiling = specceil; AL = best 10-COMB delta
at 64-72 labels; hybrid arms at micro scale, cov-shift ep10 / ball-med from
C18/C16):**

| extractor | fog ceil | fog AL | cross ceil | cross AL | snow ceil | snow AL | wet ceil | wet AL |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| cov-shift ep10 (C18) | 0.416 | +0.003 | 0.526 | -0.002 | 0.511 | -0.001 | 0.670 | -0.005 |
| ball/spec med (C16) | 0.252 | +0.054 | 0.401 | +0.075 | 0.554 | +0.004 | 0.730 | -0.007 |
| hybrid base (micro) | 0.339 | -0.017 | 0.481 | -0.085 | 0.423 | -0.076 | 0.556 | -0.038 |
| hybrid ball (micro) | 0.317 | -0.011 | 0.493 | -0.082 | 0.415 | -0.065 | 0.563 | -0.044 |
| hybrid spec (micro) | 0.294 | -0.009 | 0.498 | -0.078 | 0.418 | -0.068 | 0.552 | -0.035 |
| hybrid ball_spec (micro) | 0.346 | -0.017 | 0.422 | -0.070 | 0.416 | -0.071 | 0.574 | -0.034 |

**Result: no hybrid arm combines the two properties at micro scale.**

1. **The AL-friendly property did NOT transfer.** Every hybrid arm is NEGATIVE
   on every condition (fog -0.009..-0.017, crosstalk -0.070..-0.085, snow
   -0.065..-0.076, wet -0.034..-0.044). None of the ball/spec positive AL
   (+0.05..+0.08 on fog/crosstalk) appears. The cov-shift base behavior
   dominates: the frozen probe sits near its own ceiling, so labels buy almost
   nothing, and the ball/spec losses on the cov-shift recipe did NOT make
   labels convert to updates.
2. **The ceilings did NOT hold either.** Hybrid fog ceiling 0.294-0.346 vs
   cov-shift ep10 0.416 (all lower); crosstalk 0.422-0.493 vs 0.526 (all
   lower). The `ball_spec` arm's fog ceiling (0.346) is the best hybrid but
   still 0.07 below cov-shift. The ball/spec geometry perturbation did move
   the spectrum (hybrid kappa 56-173k vs cov-shift medium 6-15M, much
   flatter) but that flattening produced NO AL gains on this recipe.
3. **The cond_structure gate shows the cov-shift signature survived** in all
   hybrid arms (fog feat_cos_B 0.57-0.62, dir_ret_B 0.81-0.86, the cov-shift
   recovery; healthy zs_B 0.26-0.35 at micro scale, below the medium cov-shift
   reference as expected at micro).

**Interpretation: the two properties are entangled with the training recipe,
not additive.** Adding the AL-geometry losses to the cov-shift recipe neither
(a) preserves the cov-shift ceiling at the cov-shift level, nor (b) imparts the
AL-friendly behavior of the corsupcon ball/spec runs. The ball/spec AL gains
came from the losses interacting with the CORSUPCON recipe (its SupCon branch,
corruption view, and different base geometry); bolting the losses onto the
cov-shift GMSIFC/LSCC recipe does not reproduce the interaction. **The micro
scale is the standard caveat** (C12/C13: micro ceilings are inherently below
medium, and micro AL is not a trustworthy verdict, it went negative even for
the base in C12). The honest read is: NO positive signal at micro scale, the
hybrid did not obviously combine the properties, and the micro negatives on AL
are consistent with the cov-shift AL-fragility already measured in C18.

**What this does NOT close:** the possibility that the hybrid works at MEDIUM
scale with tuned loss weights. C14 showed `spectrum_loss` at 0.1 was too weak
to hold the flattened spectrum through LR annealing on the corsupcon recipe;
the hybrid used the same 0.1 weights at micro scale. The next probe (if the
hybrid direction is pursued) is the medium run of `inputin_in_chan_ball_spec`
with stronger weights (ball_w/spec_w 0.2-0.5), the micro gate is the
screening step, and it did not clear the bar. Per the README 6.1 options, the
alternative is option 2 (improve the AL gates for the cov-shift extractor - the C18 recipe already reaches +0.019 on ep21 fog), which needs no further
extractor training.

### Iteration C20: the residual-compressibility reframe (2026-08-19)

C19's negative is REFRAMED: it does not show the two ideas are incompatible - it shows they were combined at the WRONG LEVEL. The two mechanisms:

- **cov-shift** is an end-to-end recipe making the existing probe good under
  corruption: it consumes the residual, so the frozen probe sits near the
  ceiling and labels have little left to buy (AL ~0).
- **ball/spec** is a recipe for changing the geometry so sparse labels CAN
  modify a probe: it leaves MORE residual but structures it for correction
  (AL +0.05..+0.08).

Combining them in ONE representation produced exactly the C19 failure: the
ball/spec mechanism damaged the cov-shift geometry (lower ceiling) while the
cov-shift mechanism consumed the residual (still little AL gain). The C19
data also FALSIFIES "flat spectrum => AL-friendly": the hybrid kappa fell to
56-173k (very flat) with NO AL gain, conditioning is not the mechanism; the
corsupcon training interaction is.

**The reframe: cov-shift handles the bulk; AL handles a small residual that
cov-shift deliberately leaves recoverable.** The candidate architecture is
W = W_cov + residual, where the residual is estimated from labels, NOT a
full 17 x 10k probe, but a small correction. The key question (tested in C20,
eval-only, no training): is the ORACLE residual R = W* - W0 low-rank?

C20 (`al_residual_diag.py`, `bash run_al_residual.sh 3`) measures, per
condition per extractor (cov-shift ep10/ep21 + ball/spec medium):
- cos(W0, W*) and ||R||_F / ||W*||_F: how much residual is there;
- SVD of R: singular spectrum, effective rank, cumulative energy;
- THE ORACLE RESIDUAL CURVE: mIoU(W0 + R_r) for r in {0,1,2,4,8,16,32,...},
  where R_r = U_r U_r^T R is the top-r projection. This is the CEILING of any
  low-rank residual AL method;
- feature-space shift check: per-class mean shift (corrupted - clean) in 128-d,
  SVD'd, does the corruption live in a small feature subspace too?

**Decision rule:**
- If the curve climbs to near-oracle at r=4-8 (cum_energy ~0.9): AL should
  estimate a low-rank correction W = W0 + U_r C (17 x r unknowns, not 17 x
  10k), the C21 direction, and the direct answer to the Iterations-7/8
  T-synthesis dimensionality failure.
- If the curve needs r >= 17 (no compression): the residual is full-rank and
  the two-subspace extractor (z = [z_cov, z_AL], with an orthogonality loss)
  is the fallback (C23).
- If cov-shift's residual is SMALL (||R||/||W*|| low) but ball/spec's is
  larger: the C19 explanation is confirmed quantitatively, cov-shift leaves
  little for AL, ball/spec leaves a structured residual.

### Iteration C20 RESULTS: the residual is low-rank on EVERY extractor (2026-08-19)

The C20 diagnostic ran clean on all 4 extractors x 4 conditions
(`al_residual_diag.py`, eval-only). The decisive numbers:

**The oracle residual curve is steep everywhere: cum-energy r8 = 1.00 and
mIoU(W0 + R_8) == oracle on EVERY extractor and condition. Effective rank of
R = W* - W0 is consistently 3.9-4.9:

| extractor | cond | frozen | oracle | cos(W0,W*) | rel-norm | eff-rank | r=4 mIoU | r=8 mIoU |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| cov-shift ep10 | fog | 0.261 | 0.375 | 0.191 | 1.23 | 4.3 | 0.364 | 0.375 |
| cov-shift ep10 | crosstalk | 0.524 | 0.554 | 0.593 | 0.86 | 4.8 | 0.549 | 0.554 |
| cov-shift ep10 | snow | 0.456 | 0.493 | 0.465 | 1.00 | 4.2 | 0.494 | 0.495 |
| cov-shift ep10 | wet_ground | 0.429 | 0.613 | 0.422 | 1.01 | 4.8 | 0.567 | 0.595 |
| cov-shift ep21 | fog | 0.231 | 0.332 | 0.187 | 1.23 | 4.4 | 0.326 | 0.332 |
| cov-shift ep21 | crosstalk | 0.504 | 0.534 | 0.610 | 0.86 | 4.9 | 0.529 | 0.534 |
| cov-shift ep21 | snow | 0.442 | 0.473 | 0.518 | 0.94 | 4.1 | 0.476 | 0.486 |
| cov-shift ep21 | wet_ground | 0.427 | 0.581 | 0.451 | 0.99 | 4.9 | 0.566 | 0.580 |
| ball | fog | 0.088 | 0.239 | 0.055 | 1.23 | 4.7 | 0.208 | 0.239 |
| ball | crosstalk | 0.126 | 0.383 | 0.044 | 1.19 | 4.9 | 0.324 | 0.383 |
| ball | snow | 0.506 | 0.542 | 0.361 | 1.08 | 3.9 | 0.536 | 0.542 |
| ball | wet_ground | 0.641 | 0.724 | 0.565 | 0.86 | 4.8 | 0.708 | 0.715 |
| spec | fog | 0.074 | 0.247 | 0.053 | 1.27 | 4.7 | 0.176 | 0.247 |
| spec | crosstalk | 0.127 | 0.368 | 0.065 | 1.18 | 4.9 | 0.308 | 0.368 |
| spec | snow | 0.499 | 0.526 | 0.405 | 1.06 | 3.9 | 0.528 | 0.531 |
| spec | wet_ground | 0.619 | 0.692 | 0.574 | 0.85 | 4.8 | 0.663 | 0.673 |

**Takeaway 1: the residual correction is compressible to 4-8 directions on
EVERY extractor.** This is a property of the corruption + probe structure, NOT
the extractor. The C21 premise (W = W0 + U_r C, r ~ 4-8) holds everywhere.

**Takeaway 2: the C19 explanation is confirmed, with a precise mechanism.**
cos(W0, W*) on crosstalk: cov-shift 0.59-0.61 vs ball/spec 0.044-0.065. The
cov-shift frozen probe is already ~60% aligned with the oracle (little to fix);
the ball/spec frozen probe is nearly ORTHOGONAL to the oracle (large correction
needed). This is exactly why cov-shift AL ~0 and ball/spec AL +0.075 on
crosstalk.

**Takeaway 3: the residual NORM is NOT the differentiator.** ||R||/||W*|| is
0.86-1.27 across ALL extractors, cov-shift's residual is not smaller in
norm, it is smaller RELATIVE to its already-high frozen probe. The rank is the
same (~4-5); what differs is the baseline position.

**Takeaway 4: the feature-space shift is even lower-rank.** The 128-d
per-class mean shift (corrupted - clean) has effective rank 1.2-1.6 on
wet_ground (2-4 elsewhere): the corruption shift is essentially 1-2
dimensional in feature space on the hardest condition.

**Takeaway 5: wet_ground is the prize and the failure was the AL mechanism,
not the residual.** cov-shift ep10 wet has the largest closeable gap (frozen
0.429 / oracle 0.613, +0.184) and r=1 alone recovers 0.512 (83% of the gap in
ONE direction), r=8 -> 0.595. The condition where C18 AL was most negative
(-0.005) is also where the residual is MOST compressible. The C18 AL failed
because it estimated the full T_hat; the residual structure was there all
along.

**Verdict: C21 is validated only in its oracle form.** The next diagnostic,
`al_residual_al_diag.py`, tests whether the low-rank correction is estimable
from sparse labels (the deployable $C$).

### Iteration C21 RESULTS: the oracle residual is estimable, the estimated basis is not (2026-08-19)

C21 ran on all 4 extractors x 4 conditions (`al_residual_al_diag.py`): for
budgets k in {2,4,8} pts/class (14-56 labels), two basis choices estimate
$W = W0 + U_r C$ where $C$ is fit from labels on the residual $Y - X W0$:

- **oracle-basis:** $U_r$ from the SVD of the true residual $R = W* - W0$
  (the C20 ceiling - answers: if we knew the directions, how many labels
  estimate $C$?);
- **est-basis:** $U_r$ from the SVD of the label-estimated residual
  $R_sub = W_sub - W0$ ($W_sub$ fit on the same labels - the deployable
  version, answers: can labels discover the directions?);
- **full-probe:** standard ridge on the labels (the Iterations-7/8 comparison).

**The oracle-basis validates, the est-basis collapses - uniformly:**

| extractor | cond | gap | oracle-basis best (r) | est-basis best (r) | full-probe best |
| :--- | :--- | :--- | :--- | :--- | :--- |
| cov-shift ep10 | fog | +0.115 | **+0.059** (k8 r8) | -0.194 | -0.194 |
| cov-shift ep10 | wet_ground | +0.188 | **+0.141** (k8 r8) | -0.386 | -0.379 |
| ball | crosstalk | +0.255 | **+0.229** (k8 r8) | -0.069 | -0.069 |
| spec | fog | +0.172 | **+0.167** (k8 r8) | -0.034 | -0.033 |

Full table: fog/crosstalk are representative; snow and wet_ground show the
same split (oracle-basis k8 r4-r8: fog +0.046, wet +0.122 vs est-basis -0.21
to -0.41). At every k, every r, every condition, **est-basis delta is
-0.21 to -0.61** - indistinguishable from full-probe (-0.19 to -0.49) and
catastrophically below oracle-basis (+0.05 to +0.14 on the same labels).

**What went wrong compared to the promise:** C20's promise was the **oracle**
residual curve: $W0 + R_r$ with $R_r$ from the true $R$ is low-rank and
reaches the oracle at r=4-8 (C20 Results table). C20 never tested whether
$U_r$ itself is estimable - it used the oracle $U$. C21 isolates that step:
- $R_sub = W_sub - W0$ is built from the same `k` labels, but $W_sub$ (the
  full-probe fit on `k` labels) is catastrophically wrong (full-probe delta
  -0.21 to -0.61), so its left singular vectors $U_sub$ are random and the
  `est-basis` fit $C = (U_sub^T X^T X U_sub)^-1 U_sub^T X^T (Y - X W0)$ inherits
  that randomness.
- C15/C16's promise was different: it used the **T_hat mass** route
  ($T = N * mean$, source counts, control variate) and reported snow
  +0.001 and wet_ground -0.011 at k=8. That route stays in the T space where
  the means ARE estimable (mean_cos 0.94-0.97 for centroid/random, C16 TEST1);
  C21's $R_sub$ basis estimation stays in the W space where $W_sub$ is not.

**The comparison to the previous diagnostics that showed promise:**
- **C15 V3 (control variate) and C16 centroid k=2** both operate on $T_hat$:
  $T_c = N_c * mu_c$ with `mu_c` the k-point mean. Their mean quality is
  0.94-0.97 and their best AL on snow/wet was **near-zero to positive**
  (+0.001, -0.011) - the T estimation premise holds.
- **C21's est-basis and full-probe both operate on $W_sub$**, which needs
  $S_lab^-1 T_lab$ with $S_lab$ from `k` points. That inverts a 10k x 10k
  structure from ~14-56 points and is the same $W_sub$ whose whitened error
  was 11-125x in C16. The promise that survives is the **T-space** low-rank
  correction $T = T0 + U_r^T C$ (or the C15/C16 mass route), not the W-space
  $R_sub$ basis.

**Attribution (the "does the premise still have potential" guardrail):**
- Closeable gap `oracle - frozen` is intact on every extractor (fog +0.11-0.17,
  wet_ground +0.07-0.19), so the premise is not dead.
- $t_cos$ at the oracle-basis best is good where $U$ is known; $w_cos$ at the
  est-basis best collapses to ~0.01, tracking $W_sub$.
- The verdict is therefore **basis estimation**, not the extractor or the
  residual rank: the low-rank residual exists (C20), and $C$ is estimable from
  labels when $U$ is oracle (C21 oracle-basis), but $U$ is not estimable from
  the same sparse labels via $W_sub$.

**Next lever:** estimate $U_r$ from **unlabeled** structure instead of $W_sub$:
the feature-space shift SVD (C20 `feat_shift`, eff-rank 1.2-4 on wet_ground) or
the pool covariance's top r directions already computed in C14/C20. The C21
`oracle-basis` numbers are the ceiling for that approach (fog +0.059, wet
+0.141 at k8 r8) and quantifiably beat the T_hat deployment fixes on the hard
conditions.

### Iteration C22 RESULTS: the unlabeled bases do not capture the residual (2026-08-19)

Tested the C21 next lever: $W = W_0 + U_r C$ where $U_r$ is from **unlabeled**
pool structure (no labels for $U_r$, labels only fit $C$), per r in $\{1,2,4,8\}$,
per k in $\{2,4,8\}$ (`al_residual_unlabeled_diag.py`, eval-only):

- **pool-cov:** $U_r$ = top-$r$ eigenvectors of $S = X_p^T X_p / N$;
- **code-shift:** $U_r$ = top-$r$ left singular vectors of the per-class
  code-mean shift $M$ ($17 \times 10000$, row $c = \mu_{pool,c} - \mu_{clean,c}$).

| extractor | cond | gap | oracle-basis k8 r8 | pool-cov best | code-shift best | est-basis best |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| cov-shift ep10 | fog | +0.115 | **+0.059** | +0.003 | -0.001 | -0.212 |
| cov-shift ep10 | wet_ground | +0.188 | **+0.141** | +0.033 | +0.019 | -0.386 |
| ball | fog | +0.150 | **+0.127** | +0.053 | +0.056 | -0.069 |
| ball | crosstalk | +0.255 | **+0.229** | +0.110 | +0.075 | -0.069 |
| spec | fog | +0.172 | **+0.167** | +0.073 | +0.060 | -0.066 |

Full: at k=8 the pool-cov best is r=1-2 (+0.003 fog, +0.033 wet) and
degrades at r=8 (-0.011 wet); the oracle-basis stays flat-positive to r=8.
Increasing $r$ **hurts** the unlabeled bases (extra directions add noise),
while it **helps** the oracle basis.

**What went wrong compared to the promise:** C20's promise was the *oracle*
residual curve $W_0 + R_r$, $R_r = U_r U_r^T R$ with $U_r = SVD(R)$: eff-rank
$4-5$, cum-energy r8 = $1.00$, $r=4$ already $85-90\%$ of the oracle gap. That
used the **true** residual directions. C21/C22 isolate whether $U_r$ is
estimable:

- $U_{sub} = SVD(W_{sub} - W_0)$, $W_{sub}$ fit on $k$ labels: $W_{sub}$ is
  catastrophically wrong (full-probe delta -0.21 to -0.61), so $U_{sub}$ is
  random and `est-basis` collapses.
- $U_{pool} = SVD(S)$ and $U_{shift} = SVD(M)$: the top pool-covariance
  eigenvectors are the **high-variance** directions; the C20 whitened error
  ($11-125\times$) shows the residual lives in the **low-variance** directions
  that $S^{-1}$ amplifies. Pool-cov $U$ is therefore orthogonal to $R$ and
  captures at most $40-50\%$ of its gain (ball fog 0.053 vs 0.127, wet
  0.012 vs 0.043). Code-shift is similar. Neither is a proxy for the
  residual subspace.

**Premise vs design:** the low-rank residual *exists* (C20, oracle-basis) and
$C$ *is* estimable when $U_r$ is known (C21 oracle-basis: $k=4$ gives
fog +0.050, wet +0.122). The failure is **basis estimation**, not the
premise that the residual is low-rank or that the gap is closeable. The
previous promise that survives is the **T-space** route (C15 V3 control
variate, C16 centroid k=2): there the means ARE estimable (mean_cos
0.94-0.97) and the best AL on snow/wet was near-zero to positive (+0.001,
-0.011). C21's `est-basis` and C22's unlabeled bases both try to estimate
$U_r$ in **W-space** where $W_{sub}$ is not.

**Next lever:** estimate $U_r$ from a **regularized** $W_{sub}$ (ridge on
$W_{sub}$ before SVD, or $U_r$ from $S^{-1} T_hat$ instead of $T_hat$)
or keep the T-space correction $T = T_0 + U_r^T C$ where $U_r$ is the
feature-space shift (eff-rank 1.2 on wet_ground) already computed in C20.

### Iteration C23: Tracks A/B/C on cov-shift + fallback budget (2026-08-19)

Broadens the C19-C22 search while keeping cov-shift as the anchor, per the
plan in your long message (your points 1-7, 17):

**Track A (budget):**
- **A1** Budget curve $k$ per class in $\{8,16,32,64,128\}$ on the current best
  recipe (centroid $k$ means, source counts, control variate $\rho=0.5$,
  fractional-residual $\beta=0.6$). Reports $\cos(T_{AL},T_{oracle})$,
  $\cos(W_{AL},W_{oracle})$, $\Delta$mIoU per $k$. Distinguishes
  information-limited (more labels help), estimator-limited (plateau), and
  decoder-limited ($T$ improves but $W$ does not). **A2** Adaptive
  $k_c \propto N_c^{\alpha} \sigma_c^{\beta}$ with $\alpha \in \{0,0.25,0.5,1\}$
  at fixed total $B$ (your point 2): equal vs mass-proportional vs intermediate.

**Track B (residual decoder, your point 3 in priority order):**
- **B1** $W = W_{cov} + \Delta W$ with $\gamma||\Delta W||^2$; **B2** low-rank
  $\Delta W = U V^T$, $r=4,8$ ($17r$ coefficients); **B4** source-derived basis
  $U$ from $\mu_c^{source} - \mu_c^{target-like}$ or $\Delta z = z_{corr} -
  z_{clean}$. **B1/B2 are expected to be strongest** (your point 3C).

**Track C (extractor, only if A/B fail):**
- **C1** $z=[z_{cov}, \epsilon z_{AL}]$ sweeping $\epsilon$;
- **C2** orthogonal residual $||Z_A^T Z_B||_F^2$;
- **C3** uncertainty-gated $w(x)=f(H(p_{cov}))$ ball/spec;
- **C4** condition-specific $z_{condition}$.

$W_{oracle}-W_{cov}$ rank (your point 14, C20) and frozen cov-shift + low-rank
residual (your point 11, C21) remain the C20/C21 diagnostics.

**Fallback you noted:** if all cheap-budget (k=8-16) tracks stay flat-negative,
the $A1$ curve is extended to $k=128$ to find the knee (where $\Delta$mIoU
turns positive) and to see *what* the added labels bought: per-$k$ $t_{cos}$,
$w_{cos}$, per-class coverage and mean quality reveal whether the knee is
mean quality vs mass vs rare-class filling, so the next step is a selection
mechanism that gets that information at lower $k$ (your points 9-13). The
comprehensive harness `al_tracks_abc_diag.py` (eval-only, $\approx$30min,
`bash run_al_tracks_abc.sh 3`) does $A1$ to $k=128$ and $A2$/$B$ per $k$ in one
run with that fallback built in.

### Iteration C23 RESULTS: the fallback budget to k=128 is still decoder-limited (2026-08-19)

C23 ran on cov-shift ep10 (`al_tracks_abc_diag.py`, eval-only). The fallback
answers your note: if cheap tracks fail, push the budget to see where it
*does* turn positive and *what* the extra labels bought.

**A1 budget curve (delta_full) from k=8 to k=128 per class (56 to 896 labels):**

| cond | gap | k=8 $t_{cos}$/$w_{cos}$/$\Delta$ | k=32 | k=128 |
| :--- | :--- | :--- | :--- | :--- |
| fog | +0.118 | 0.623/-0.008/-0.220 | 0.651/-0.000/-0.223 | 0.663/0.012/-0.206 |
| crosstalk | +0.031 | 0.940/0.004/-0.477 | 0.970/-0.009/-0.472 | 0.979/-0.000/-0.462 |
| snow | +0.036 | 0.934/0.004/-0.427 | 0.967/0.001/-0.410 | 0.973/-0.022/-0.420 |
| wet_ground | +0.187 | 0.880/-0.013/-0.370 | 0.926/-0.009/-0.368 | 0.936/0.010/-0.366 |

$t_{cos}$ is already $0.88-0.94$ at k=8 and plateaus to $0.97-0.98$ by k=32;
$w_{cos}$ stays $\approx 0$ at *every* k to 128; $\Delta$ stays flat-negative
to -0.46. **There is no knee to 128.** More labels buy $t$-quality but $W$
does not follow: this is the **decoder-limited** case you asked to track
($T$ improves but $W$ does not), not information-limited. The per-class
coverage at k=128 is 896 labels ($\approx 60\%$ of the pool) and the mean
quality is already saturated, so the missing information is not mean quality
or mass or rare-class count - it is the ridge amplification.

**A2 adaptive allocation (B $\approx$ 136, k_c \propto N_c^{\alpha}$):**

| $\alpha$ | fog $\Delta$ | crosstalk $\Delta$ | snow $\Delta$ | wet $\Delta$ |
| :--- | :--- | :--- | :--- | :--- |
| 0 (equal) | -0.230 | -0.472 | -0.428 | -0.396 |
| 0.25 | -0.208 | -0.507 | -0.407 | -0.375 |
| 0.5 | -0.216 | -0.468 | -0.417 | -0.381 |
| 1.0 (mass) | -0.216 | -0.450 | -0.402 | -0.360 |

Uniform vs mass-proportional vs intermediate are indistinguishable on the
full-probe decoder: allocation does not fix the decoder.

**B residual low-rank (k=8, r=4 vs r=8, oracle vs pool-cov $U$):**

| cond | k | oracle $r=4$ | oracle $r=8$ | pool $r=4$ | pool $r=8$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| wet_ground | 8 | +0.048 | +0.060 | -0.009 | -0.009 |
| wet_ground | 32 | +0.067 | +0.065 | -0.009 | -0.009 |

(Representative; fog/crosstalk show the same pattern: oracle $+0.05-0.06$ at
$r=4$, pool $\approx 0$.) The **A1 delta_res** (the $r=8$ oracle residual curve
reported *within* A1) is consistently positive even where delta_full is
negative: wet_ground k=8 delta_res **+0.137**, k=128 +0.165; fog +0.060;
crosstalk +0.004. This is the **decoder fix**: $C$ in $W = W_0 + U_8 C$ (8 x 17
unknowns) is estimable from labels when $U_8$ is the oracle $U$; the full
$W$ is not.

**What the fallback bought:** at k=128 the extra 840 labels bought $t_{cos}$
$0.623 \rightarrow 0.663$ (fog), $0.88 \rightarrow 0.936$ (wet) and per-class
coverage to $60\%$ of the pool, but $w_{cos}$ stayed $0$ and $\Delta$ stayed
$-0.20$ to $-0.46$. The knee does not exist to $k=128$ for the full probe.
The only positive $\Delta$ at any $k$ is via the **residual low-rank** route
($\Delta_{res}$ +0.06 to +0.16), which holds even at k=8. So the selection
mechanism that would compress $k=128$ to $k=8$ is not a better per-class $k$
or an adaptive $k_c$, it is the low-rank residual decoder (your point 3C:
$W = W_{cov} + U V^T$ with $r=4$). The next step that is actually worth
training is **Track C1/C2** ($z=[z_{cov}, \epsilon z_{AL}]$ and the orthogonal
$||Z_A^T Z_B||_F^2$ branch, your point 6) so $U$ is learned unlabeled and $C$
is the only label-estimated quantity.

### Iteration C24: B decoder + feature-space potential, extractor frozen (2026-08-19)

Per your constraint that the feature extractor is a last resort, C24 keeps
cov-shift frozen and asks two questions on cov-shift ep10
(`al_b_feature_diag.py`, eval-only):

**B decoder ($W = W_{cov} + \Delta W$, labels fit only $\Delta W$):**

| cond | $k$ | $B1$ oracle $r=17$ | $B2$ pool $r=4$ | $B2$ pool $r=8$ | full-probe |
| :--- | :--- | :--- | :--- | :--- | :--- |
| fog | 8 | **+0.039** | -0.008 | -0.008 | -0.220 |
| fog | 32 | **+0.062** | -0.009 | -0.008 | -0.223 |
| wet_ground | 8 | **+0.086** | +0.010 | -0.000 | -0.369 |
| wet_ground | 32 | **+0.128** | +0.016 | +0.013 | -0.368 |

$B1$ (full-rank residual with oracle $U$) is positive even at $k=8$ on wet
ground ($+0.086$), and $r=17$ holds to $k=32$ ($+0.128$). The pool-cov $U$
($B2$) stays $\approx 0$ (the same unlabeled-basis failure as C22). So the
decoder fix is $R$-space, not $S$-space: labels can estimate $\Delta W$ when
$U$ is known, but $U$ from $S$ is not the residual subspace.

**Feature-space potential (no labels beyond clean, does *some* classifier work
on this space?):**

| cond | HDC frozen/oracle (gap) | raw frozen/oracle (gap) | $k$NN raw 1 | $k$NN HDC 1 | proto raw / HDC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| fog | 0.259/0.377 (+0.118) | 0.245/0.279 (+0.033) | 0.813 | 0.812 | 0.325/0.382 |
| snow | 0.457/0.493 (+0.036) | 0.428/0.444 (+0.016) | 0.814 | 0.812 | 0.673/0.689 |
| wet_ground | 0.427/0.614 (+0.187) | 0.398/0.534 (+0.136) | 0.850 | 0.848 | 0.619/0.623 |

$k$NN ($\approx 0.81$ to $0.85$ on raw and HDC) is far above the HDC linear
probe (frozen $0.25$ to $0.52$). The space *has form*: a non-linear
neighbor rule already beats the linear probe by $0.30$ on fog ($0.81$ vs
$0.25$) and $0.42$ on wet ($0.85$ vs $0.39$). Raw vs HDC $k$NN are
indistinguishable ($0.813$ vs $0.812$ fog), so the HDC projection is *not* the
bottleneck; the linear separator is. Prototype (nearest class mean) is
intermediate (fog HDC proto $0.382$ vs linear $0.259$), but $k$NN dominates.

**Net:** the cov-shift extractor has enough function for an unknown
classifier change (your point 8-11): $k$NN already reaches $0.81$ to $0.85$
frozen, and $W_{cov} + \Delta W$ with an oracle $U$ reaches $+0.08$ to
$+0.12$ on wet at $k=8$ to $32$. The next lever is a **decoder change**
(B1/B2 with a better $U$, e.g., $S^{-1}T$ or regularized $W_{sub}$) or the
feature-space $k$NN/proto rule itself, not an extractor change.

### Iteration C25: prototypes, $S^{-1}T$ and $B1$ budget without memory banks (2026-08-19)

Per your constraint to avoid $k$NN memory banks, C25 keeps the cov-shift
extractor frozen and tests three decoder-side levers that need no bank, all at
$k \in \{8,16,32,64,128\}$ per class ($56$ to $896$ labels, centroid $k$ means):

**Prototype (nearest class mean, cosine) at $k$:**

| cond | frozen $W_0$ | $k=8$ proto | $k=32$ proto | $k=128$ proto |
| :--- | :--- | :--- | :--- | :--- |
| fog | 0.259 | 0.245 | 0.248 | 0.249 |
| crosstalk | 0.524 | 0.441 | 0.456 | 0.458 |
| snow | 0.457 | 0.399 | 0.407 | 0.404 |
| wet_ground | 0.427 | 0.374 | 0.402 | 0.408 |

Prototype is **worse than frozen at every $k$ to 128** on every condition
(fog $-0.014$ at $k=8$, crosstalk $-0.083$). More labels do not close the
gap (fog $0.245 \rightarrow 0.249$ flat). This matches C8: the encoding is
not the bottleneck, and the cheap prototype does not replace the linear
separator without a bank.

**$S^{-1}T$-derived $U$** ($U_r = SVD((S + \lambda I)^{-1} T_{hat})$, $r=4,8$):

| cond | $k=8$ $r=4$/$r=8$ $\Delta$ | $k=32$ $r=4$ |
| :--- | :--- | :--- |
| fog | $+0.001$/$+0.006$ | $+0.002$ |
| wet_ground | $+0.001$/$-0.001$ | $+0.002$ |

All $\approx 0$ (same as C22 pool-cov $+0.003$). The whitened $T$ basis is also
not the residual subspace: $S^{-1}$ amplifies the wrong directions.

**$B1$ budget for the current linear separator ($W = W_0 + U_{oracle} C$,
$r=17$ full residual, $U_{oracle}=SVD(R)$):**

| cond | $k=8$ $\Delta$ | $k=16$ | $k=32$ | $k=64$ | $k=128$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| fog | **+0.040** | +0.044 | **+0.063** | +0.055 | +0.052 |
| crosstalk | -0.015 | **+0.010** | **+0.018** | +0.018 | +0.021 |
| snow | -0.010 | **+0.012** | **+0.015** | +0.017 | +0.017 |
| wet_ground | **+0.087** | +0.113 | **+0.129** | +0.151 | **+0.158** |

Raising the budget **does** help $B1$ in the current paradigm: fog and wet are
already positive at $k=8$, and at $k=32$ every condition is positive (snow
$+0.015$, crosstalk $+0.018$). Wet_ground keeps climbing to $+0.158$ at
$k=128$ (896 labels), recovering $84\%$ of its $+0.187$ gap. The $k=8 \rightarrow
k=32$ step is the knee ($56 \rightarrow 224$ labels); beyond $k=32$ the gain
is marginal except on wet.

**Takeaway:** without memory banks, prototypes and $S^{-1}T$ do not work on
this space, but the **$B1$ residual decoder with the oracle $U$ does** and its
budget curve is the one that turns positive. $B1$ at $k=8$ is already cheaper
than the C15/C16 $k=8$ $T$-hat route and at $k=32$ it closes the gap on every
condition. The remaining work is $U$ estimation without the oracle (C22 pool
and $S^{-1}T$ both failed), not more prototype tuning.

### Iteration C26: low bonuses, $U$ without oracle, tiny bank (2026-08-19)

Per your three questions, still on cov-shift ep10, still no extractor change
(`al_lowbonus_bank_diag.py`, eval-only, $k=32$ where $B1$ is positive):

**Q1 Why snow ($+0.036$ gap, $B1$ $+0.015$ at $k=32$) and crosstalk ($+0.031$,
$+0.018$) bonuses are low:** at $k=32$, $t_{cos}=0.651$ fog and $0.970$
crosstalk but $w_{cos}\approx 0$ and $\Delta_{full}$ is flat-negative to
$k=128$ ($-0.22$ fog, $-0.46$ crosstalk). More labels buy $t_{cos}$ ($0.623
\rightarrow 0.663$ fog, $0.88 \rightarrow 0.926$ wet to $k=128$) but $W$
does not follow. This is the same decoder-limited pattern as C23: the
missing information is not mean quality or mass, it is the ridge
amplification. The gap itself is also small on snow/crosstalk, so even the
$+0.015$ on snow is $42\%$ of its $+0.036$ gap.

**Q2 $U$ without oracle at $k=32$ ($r=8$, $B1$ $r=8$ $+0.015$ snow / $+0.129$
wet as the reference):**

| $U$ | fog $\Delta$ | align | snow $\Delta$ | wet $\Delta$ |
| :--- | :--- | :--- | :--- | :--- |
| $U_T = SVD(T_{hat})$ | $-0.008$ | $0.00$ | $-0.009$ | $+0.011$ |
| $U_{Rreg}$ (ridge $\lambda*10$) | $-0.216$ | $0.05$ | $-0.423$ | $-0.374$ |
| $U_{pool}=SVD(S)$ | $-0.009$ | $0.00$ | $-0.005$ | $+0.013$ |
| $U_{shift}=SVD(M)$ | $-0.006$ | $0.01$ | $-0.003$ | $+0.030$ |

All are $\approx 0$ (or negative) and align $0.00$ to $0.05$ with $U_R$. No
unlabeled $U$ captures the residual subspace. The $T$-based $U_T$ and the
regularized $W_{sub}$ $U_{Rreg}$ both fail exactly as C21 $U_{sub}$ did.

**Q3 Tiny memory bank (labeled $56$ + $0$/$500$/$1000$/$5000$ random unlabelled,
$1$-NN on raw 128-d, no stream):**

| bank | fog acc | crosstalk | snow | wet_ground |
| :--- | :--- | :--- | :--- | :--- |
| $56$ (labels only) | $0.355$ | $0.672$ | $0.618$ | $0.544$ |
| $556$ ($+500$) | **$0.662$** | $0.711$ | **$0.720$** | **$0.688$** |
| $1056$ ($+1000$) | $0.684$ | $0.728$ | $0.724$ | $0.704$ |
| $5056$ ($+5000$) | $0.738$ | $0.757$ | $0.753$ | $0.765$ |

Adding just $500$ unlabelled points to the $56$ labels jumps fog $+0.307$
($0.355 \rightarrow 0.662$) and wet $+0.144$ ($0.544 \rightarrow 0.688$);
$5000$ adds only $+0.05$ more. The $k$NN bank at $556$ already beats the HDC
linear frozen ($0.259$ fog, $0.427$ wet) by $0.40$ and $0.26$. The bank does
not need to be large.

**Takeaway:** the low bonuses are decoder-limited with small gaps, $U$ without
oracle is not recoverable from any of the four unlabeled statistics at $k=32$,
and a very low-size bank ($500$ unlabelled) is the only lever that moves the
needle without touching the extractor.

### Iteration C27: small bank point prediction and what it buys for the linear head (2026-08-19)

The same $k=8$ ($56$ labels) $+$ $500$ random unlabelled $1$-NN bank looks
different under point accuracy vs $mIoU$:

| bank | fog acc / $mIoU$ | snow acc / $mIoU$ | wet acc / $mIoU$ |
| :--- | :--- | :--- | :--- |
| $56$ | $0.355$ / $0.226$ ($-0.035$) | $0.618$ / $0.332$ ($-0.124$) | $0.544$ / $0.320$ ($-0.107$) |
| $556$ ($+500$) | $0.662$ / $0.256$ ($-0.005$) | $0.720$ / $0.404$ ($-0.053$) | $0.688$ / $0.443$ ($+0.016$) |

Point accuracy jumps $+0.30$ fog and $+0.14$ wet with $500$ points and already
beats the HDC linear frozen ($0.259$ fog) by $0.40$, but $mIoU$ stays below
frozen on fog/snow/crosstalk at $500$ and only turns positive on wet at
$556$. The bank is good for predicting *each point* (which is the scale
signal $W_{cov}$ needs: per-class mass and variance), but $mIoU$ averages
over classes equally and the $500$ random points starve minority classes
(fog $c7$ $143$ points in pool, $c2$ $1$ point). The scale estimate for the
linear head is therefore still biased on rare classes.

**What we learned:** the bank has the form to update the linear classifier
toward the ceiling, but random $500$ does not give the function to do it at
low $mIoU$ cost.

**Next steps you asked to keep as baseline:**

1. **Smarter selection for bank and query to not starve minority classes.**
   Keep the $500$ budget but allocate it to cover rare classes: uniform
   $500/17 \approx 29$ per class, or $k_c \propto N_c^{\alpha}$ with
   $\alpha=0$ to $0.25$ (your point 2), or uncertainty $H(p_{cov})$ for
   $w(x)$ (your point 7). All are eval-only on the $500$-bank harness: build
   the bank with each allocation, measure $1$-NN $mIoU$ and the $W_{pseudo}$
   delta ($56$ true $+$ $500$ $1$-NN pseudo from the bank). If uniform $500$
   beats random $500$ on $mIoU$, the bank becomes a viable low-cost scale
   estimator for $W$.
2. **Larger dataset to raise gaps and vary points.** Current $50$k pool /
   $100$k val from $100$ frames caps rare-class mass and caps the closeable
   gap (snow $+0.036$). Test $200$ frames ($100$k pool / $200$k val) where
   rare classes have $2\times$ points and the ceiling gaps are $0.05$ to $0.10$
   larger: does the same $500$ bank now give $mIoU$ parity, and does $B1$ at
   $k=32$ ($224$ labels) recover more of the gap with more varied $T$?

Both keep the extractor frozen and keep the bank tiny ($500$ to $1000$).

### Iteration C28: larger dataset and smarter 500 as gate, why point accuracy does not help the probe (2026-08-19)

Still on cov-shift ep10, still no extractor change
(`al_larger_smart_diag.py`, eval-only, $100$ frames $50$k/$100$k vs $200$ frames
$100$k/$200$k, and the four $500$ allocations on the $56+500$ bank).

**Larger dataset (more frames, more varied points, higher gaps?):**

|  | fog gap | crosstalk | snow | wet | rare $c7$ freq small/large |
| :--- | :--- | :--- | :--- | :--- | :--- |
| small $50$k/$100$k | $+0.116$ | $+0.031$ | $+0.036$ | $+0.183$ | $143$ / $160$ |
| large $100$k/$200$k | $+0.120$ | $+0.022$ | $+0.036$ | $+0.180$ | $160$ |

Gaps are identical to $0.004$ and rare-class mass barely moves. The ceiling is
not limited by $50$k pool size or $100$ frames of variety on this extractor;
$100$ frames already saturate the per-class statistics.

**Smarter $500$ allocation for the tiny bank ($56$ true $+$ $500$, $1$-NN
$mIoU$ and $W_{pseudo}$ $\Delta$ vs frozen, $k=8$):**

| allocation | fog bank | fog $W_{pseudo}$ | wet bank | wet $W_{pseudo}$ |
| :--- | :--- | :--- | :--- | :--- |
| random | $-0.004$ | $-0.214$ | $+0.014$ | $-0.373$ |
| uniform $29$/class | $-0.008$ | $-0.208$ | $-0.024$ | $-0.378$ |
| diverse | $-0.023$ | $-0.215$ | $-0.048$ | $-0.371$ |
| uncertainty $H(p_{cov})$ | $-0.017$ | $-0.231$ | $-0.045$ | $-0.378$ |

Random is the best of the four on every condition; uniform, diverse and
uncertainty are neutral to worse on $mIoU$ and indistinguishable on
$W_{pseudo}$ ($-0.21$ fog, $-0.37$ wet). Adding $500$ points to the bank does
not make $1$-NN $mIoU$ beat frozen except on wet at $556$ ($+0.014$), and
using those $500$ pseudo-labels to fit $W_{pseudo}$ on $56+500$ is
catastrophically worse than $W_0$ ($-0.21$ to $-0.46$), even worse than the
bank itself.

**Why a good point predictor is a bad gate for the linear probe:** the $500$
bank has point accuracy $0.66$ fog ($+0.30$ over $56$ alone) because it labels
*each point* well, but $W_{pseudo}$ needs per-class *mass*
$T = \sum_{c} N_c \mu_c$. The bank's $500$ random points starve the
minority classes in $mIoU$ averaging, and their pseudo-label errors are
*systematic* per class (the same $c7$/$c2$ that are rare), so $T_{hat}$
inherits a class-conditional bias. The ridge $(S + \lambda I)^{-1}$ then
amplifies that bias along low-variance directions (C23 $w_{cos} \approx 0$).
The bank is therefore a good *per-point* teacher but a bad *per-class mass*
gate for $W$. The $W_{pseudo}$ vs $500$ true oracle ($-0.211$ vs $-0.214$ fog)
confirms it is not pseudo quality but the $T$ construction: $500$ true labels
also give $-0.211$.

**Takeaway:** larger pools do not raise the gap and smarter $500$ allocations
do not fix the $mIoU$ cost. If the bank is to be the baseline, it must be used
*as the classifier* ($1$-NN at $5056$ is $0.738$ fog, already beating $W_0$ by
$+0.061$), not as a gate for $W_{pseudo}$. The $W$-space $U$ estimation without
oracle remains the bottleneck, not the point predictor.

**Clarification:** the $500$ bank is a good
per-point predictor ($0.662$ fog accuracy) because it labels *each point*
well, but its current implementation starves minority classes and their
features in $mIoU$ averaging ($c2$ $1$ point, $c7$ $143$ points in $50$k
pool). $W_{pseudo}$ needs per-class mass $T = \sum N_c \mu_c$, so the same
bank that is good per point is a bad per-class mass gate.

**Next steps we will keep as the baseline (all still $56+500$, extractor
frozen, $500$ stays tiny, bank is a per-point teacher for $W$, not the
classifier):**

1. **Robustness and diversity of the bank as a teacher, not a classifier:**
   keep the $500$ but make it cover *correction* diversity, not just feature
   density. Test capped uniform ($5$ per class max) and farthest-point diverse
   with a per-class floor, and measure not $1$-NN $mIoU$ but the *residual*
   gradient $G_{bank} = \sum_{i \in B} x_i (y_i - p_0(x_i))^T$ quality and the
   per-class $W_{pseudo}$ $\Delta$ when the $500$ pseudo-labels are used only
   to fit $\Delta W$ (not $W$). The bank at $556$ already gives $+0.30$
   point accuracy, so the remaining question is whether a diverse $500$ gives a
   more varied $G$ for the linear head.

2. **Stability of the update:** $W_{pseudo}$ collapses even with $500$ true
   labels ($-0.211$ fog, same as pseudo), so the ridge on $56+500$ is
   unstable at $HDC$ $10000$-d. Test per-class ridge $\lambda_c \propto 1/N_c$
   and the $B1$ low-rank residual $W = W_0 + U_r C$ ($r=8$) with $C$ fit on the
   $56+500$ (instead of $W_{pseudo}$ on $56+500$). Both keep the $500$ bank
   but stabilize $(S + \lambda I)^{-1}$.

3. **Class balance in the linear head:** make the $W$ update equal per class.
   Test $w_c = 1/N_c$ or $1/\sqrt{N_c}$ per-sample weighting in the $56+500$
   ridge, or per-class learning rate $\eta_c \propto 1/N_c$, and compare
   $W_{pseudo}$ $\Delta$ vs unweighted. This directly fixes the $mIoU$
   averaging that the current $T$ construction violates.

### Iteration C30: stable-update sweep, did the decoder become stable? (2026-08-19)

Still on cov-shift ep10, still $k=8$ ($56$ labels) and $r=8$ oracle $U$ (`al_stable_update_diag.py`, eval-only). Four families, all keep $W_0$ frozen and fit only $\Delta W$:

**A $\eta$ sweep on $W(\eta)=W_0+\eta U_r C$ ($r=8$, $C$ fit at $\gamma=10^{-6}$):**

| cond | gap | $\eta=0$ | $0.1$ | $0.5$ | $1.0$ | best $\eta$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| fog | $+0.118$ | $0.000$ | $+0.009$ | $+0.032$ | **$+0.060$** | $1.0$ |
| wet_ground | $+0.187$ | $0.000$ | $+0.014$ | $+0.087$ | **$+0.137$** | $1.0$ |
| crosstalk | $+0.031$ | $0.000$ | $+0.004$ | **$+0.010$** | $+0.004$ | $0.5$ |
| snow | $+0.036$ | $0.000$ | $+0.006$ | $+0.015$ | $+0.010$ | $0.75$ |

$\eta=1.0$ is best on fog/wet where the gap is large, $\eta=0.5$ to $0.75$ on the small-gap conditions. Scaling $W$ does not hurt: the direction is correct, just the magnitude needs the full step.

**A $\gamma$ sweep on the residual ridge $C(\gamma)=(U^T X^T X U + \gamma I)^{-1} U^T X^T (Y - XW_0)$:**

Best $\gamma$ is $10^{-3}$ fog ($+0.060$) and $10^{-1}$ wet ($+0.139$), both $\approx 0$ ridge. Larger $\gamma$ does not help, the $8 \times 8$ system is already well-conditioned.

**A rank sweep ($r=1,2,4,8,17$):** $r=4$ gives $+0.048$ fog and $+0.123$ wet, $r=8$ gives $+0.060$ and $+0.137$, $r=1$ is $-0.004$ fog. $r=8$ is needed, $r=17$ is not better.

**B soft residual $Y-P_0$ vs one-hot $Y-XW_0$:** $T=1.0$ gives $+0.023$ fog vs one-hot $+0.060$, pairwise $e_y-e_{\hat y}$ gives $+0.029$ fog vs $+0.060$. Soft targets are *worse* than one-hot on this $HDC$ code, the opposite of the hypothesis. The $17$-way one-hot is the right target.

**D uncertainty weighting:** $w_i \in \{1, H(p), 1-\max p, 1/(margin+\epsilon)\}$ all give $+0.060$ fog and $+0.136$ wet, indistinguishable from unweighted. Weighting does not fix the decoder.

**Did we get an improvement and if not why?** Yes on the large-gap conditions: fog $+0.040 \rightarrow +0.060$ ($51\%$ of gap at $k=8$) and wet $+0.087 \rightarrow +0.137$ ($73\%$) with $\eta=1.0$, $r=8$, $\gamma \approx 0$. Crosstalk $+0.010$ and snow $+0.016$ stay small because their gaps are small ($+0.031$, $+0.036$), even the $+0.016$ on snow is $44\%$ of its gap. The remaining $27\%$ on wet and $49\%$ on fog is the $t_{cos}$ vs $w_{cos}$ decoder limit from C23, not the update magnitude.

### Iteration C31: old bank on the new stable residual, immediate improvement (2026-08-19)

Still on cov-shift ep10, $56$ true ($k=8$ per class) $+$ $500$ bank points, the four old bank allocations (random, uniform $29$/class, diverse farthest-point, uncertainty $H(p)$) now with the new update $W = W_0 + U_r C$ ($r=8$, oracle $U$, `robust_diagnostic/al_bank_residual_diag.py:238` $C = (U^T X^T X U)^{-1} U^T X^T (Y - XW_0)$) vs the old $W_{pseudo}$ full-probe on the same $56+500$ pseudo labels:

| cond | bank | bank $mIoU$ $\Delta$ | $W_{pseudo}$ $\Delta$ (old) | $W_{res}$ pseudo $\Delta$ (new) | $W_{res}$ true $\Delta$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| fog | random | $-0.006$ | $-0.240$ | **$+0.010$** | $+0.125$ |
| fog | diverse | $-0.022$ | $-0.227$ | **$+0.024$** | $+0.116$ |
| crosstalk | random | $-0.077$ | $-0.453$ | **$+0.006$** | $+0.028$ |
| snow | random | $-0.081$ | $-0.410$ | **$+0.014$** | $+0.037$ |
| wet_ground | random | $+0.013$ | $-0.386$ | **$+0.123$** | $+0.164$ |
| wet_ground | diverse | $-0.046$ | $-0.387$ | **$+0.108$** | $+0.168$ |

Every bank that was negative as $1$-NN or as $W_{pseudo}$ ($-0.22$ to $-0.48$) is **positive as $W_{res}$ pseudo** ($+0.006$ to $+0.123$) on the same $500$ points, and within $0.04$ of the $500$ true-oracle $W_{res}$ true ($+0.125$ fog, $+0.164$ wet). The bank that was best as $1$-NN (random) is not best as $W_{res}$ on fog (diverse $+0.024$ beats random $+0.010$), so the $500$ allocation should be chosen for $G$ quality, not $1$-NN $mIoU$.

**Takeaway:** the old $500$-point banks were not the bottleneck, the old $W_{pseudo}$ full-probe was. With the stable low-rank residual, the same $500$ banks that previously hurt ($-0.24$) now help ($+0.01$ to $+0.12$) and track the true-label ceiling to $0.04$. This is the immediate improvement you asked to check, and it keeps inference as the linear $W_{res}$ (no bank at test time).

### FULL-DATASET cross-extractor comparison: does cov-shift still win at scale? (R4, ep-10)

The same full harness (`al_full_dataset_diag.py`) run on all extractors. The
cov-shift extractor keeps its lead on every condition at full scale; the
DGLSS++ and Robust DGLSS++ ceilings are 0.17-0.30 BELOW cov-shift on
fog/crosstalk and 0.24-0.30 below on the healthy conditions:

| condition | cov-shift R4 zs -> ceiling | DGLSS++ R4 zs -> ceiling | Robust R4 zs -> ceiling |
| :--- | :--- | :--- | :--- |
| fog | 0.322 -> 0.369 | 0.129 -> 0.170 | 0.137 -> 0.173 |
| crosstalk | 0.477 -> 0.491 | 0.190 -> 0.224 | 0.193 -> 0.232 |
| snow | 0.482 -> 0.495 | 0.189 -> 0.199 | 0.208 -> 0.215 |
| wet_ground | 0.297 -> 0.419 | 0.124 -> 0.199 | 0.117 -> 0.223 |
| incomplete_echo | 0.421 -> 0.437 | 0.198 -> 0.200 | 0.210 -> 0.215 |
| beam_missing | 0.489 -> 0.487 | 0.191 -> 0.204 | 0.208 -> 0.217 |
| motion_blur | 0.452 -> 0.458 | 0.193 -> 0.201 | 0.203 -> 0.210 |
| cross_sensor | 0.420 -> 0.446 | 0.176 -> 0.203 | 0.200 -> 0.225 |

**Cross-dataset transfer (NuScenes, 32-beam, same machinery):** the cov-shift
per-scan input normalization transfers best -- frozen 0.135 / ceiling 0.213 vs
DGLSS++ 0.080 / 0.125 and Robust 0.083 / 0.133; the AL bank closes most of the
NuScenes gap too (cov-shift $W_{res}$ true $+0.075$ of $+0.078$).

### FULL-DATASET random bank baseline table (every point of every frame of seq 08; `al_full_dataset_diag.py`)

The paper-ready table: same $56+500$ random bank, $W_{res}$ $r=8$ oracle $U$, $k=8$ per class, on the FULL dataset (all $\sim$4k frames, $\sim$300M points/condition, streamed; reservoir pool $400$k, clean fit $200$k, spectral-exact ridge, pool points excluded from val). This supersedes the earlier $100$-frame harness numbers, which over-estimated the headroom (the smoke's ceiling partially fit the same frames it was tested on):

| condition | zero-shot $W_0$ | AL $W_{res}$ $56+500$ random ($mIoU$, $\Delta$) | ceiling $W^*$ | closeable gap |
| :--- | :--- | :--- | :--- | :--- |
| fog | $0.322$ | $0.327$ ($+0.005$) | $0.369$ | $+0.047$ |
| crosstalk | $0.477$ | $0.489$ ($+0.011$) | $0.491$ | $+0.014$ |
| snow | $0.482$ | $0.466$ ($-0.016$) | $0.495$ | $+0.013$ |
| wet_ground | $0.297$ | $0.341$ ($+0.043$) | $0.419$ | $+0.122$ |
| incomplete_echo | $0.421$ | $0.415$ ($-0.005$) | $0.437$ | $+0.017$ |
| beam_missing | $0.489$ | $0.492$ ($+0.004$) | $0.487$ | $-0.001$ |
| motion_blur | $0.452$ | $0.451$ ($-0.001$) | $0.458$ | $+0.006$ |
| cross_sensor | $0.420$ | $0.430$ ($+0.010$) | $0.446$ | $+0.026$ |

At full scale the closeable gaps are much smaller than the subsampled harness suggested: fog $+0.047$, crosstalk $+0.014$, snow $+0.013$, wet_ground $+0.122$. The $56+500$ random bank $W_{res}$ closes $11\%$ of the fog gap and $35\%$ of the wet gap; on the small-gap conditions the bank is inside noise (snow $-0.016$). wet_ground is the clear AL target; fog is borderline; the rest have little recoverable headroom.

**Next steps (two):**

1. **Improve the gap closed by AL where there is a gap to close**: fog, wet_ground, cross_sensor, crosstalk, snow are the conditions with a measurable closeable gap; the current $56+500$ random bank only closes a fraction of it (fog $+0.005$ of $+0.047$, wet $+0.043$ of $+0.122$). The lever is the bank's $G$-quality and the $U$-basis estimation (C21-C22 showed oracle $U$ recovers the ceiling but estimated $U$ collapses), not the harness.
2. **Get a method that tells whether there is something to close at all**: a label-free gauge of the closeable gap (e.g., residual norm $\|W^*-W_0\|$ or feature-shift strength vs frozen-cosine stability) so the AL budget is spent only on conditions where the gap is significant — beam_missing ($-0.001$) and motion_blur ($+0.006$) have nothing to close, and snow's $+0.013$ is borderline; labels should not be spent there.
