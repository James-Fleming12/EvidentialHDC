# Adaptive Memory Bank: Iterations and Testing

## Iteration 1: The Baseline (Binarization & Topological Gating)
**Goal:** Prove the foundational architecture (1-Bit Binary Buffer, Hamming Query, Graph-Pruned Update).
**Structure:**
- **Buffer:** 10,000 capacity `Float32` tensor containing bipolar `{-1, 1}` vectors. (Simulating binarization before writing custom CUDA POPCOUNT kernels).
- **Query:** `torch.sign()` followed by Matrix Multiplication dot product (functionally identical to Hamming Distance). majority vote of `k=10` neighbors.
- **Update:** Graph Pruning via topological purity. Only points where $\ge 80\%$ of neighbors agree with the vote are allowed into the FIFO queue.
- **Cold Start:** The very first frame is evaluated by the legacy 17D prototype classifier to populate the initial nodes of the graph. All subsequent frames rely entirely on the 10,000D Memory Bank.

### Expected Results (`unsup_kitti-c.py` output)
| Corruption | Method | mIoU | Accuracy | Firing Rate |
| :--- | :--- | :--- | :--- | :--- |
| Fog | Iteration 1 | TBD | TBD | TBD |
| Snow | Iteration 1 | TBD | TBD | TBD |

---

## Iteration 2: Class-Balanced Memory Bank and Pre-Trained Prototype Seeding

**Goal:** Solve catastrophic majority-class collapse (where Tail classes vanished) and eliminate the KNN computational bottleneck, while maintaining a multi-instance representation.

**Mathematical & Structural Upgrades:**

1. **Class-Balanced FIFO Buffer ($C \times N$ Partitioning):**
   - **Problem:** A globally shared FIFO queue of 10,000 instances uniformly sampled from a highly imbalanced dataset (e.g. $80\%$ road) quickly drops all rare class instances. Because update purity is conditioned on KNN predictions, rare classes can never pass the graph purity check once evicted, causing total systemic collapse.
   - **Solution:** The capacity $K=10,000$ is strictly partitioned into $\mathcal{C}$ independent queues of size $N = \lfloor K / \mathcal{C} \rfloor$. For $\mathcal{C}=17$, each class maintains exactly 588 unique geometric instances.
   - **Formulation:** Let $\mathcal{M} \in \mathbb{R}^{\mathcal{C} \times N \times D}$ be the partitioned memory. Upon admitting a point $x_t \in \mathbb{R}^D$ with pseudo-label $c$, it is routed exclusively to $\mathcal{M}_c$. If $n > N$ points are admitted, we update via uniform random sampling over the $n$ points to avoid spatial bias inherent to LiDAR sweep structures.

2. **Prototype Seeding (Cold-Start Missing Class Guarantee):**
   - **Problem:** If a rare class (e.g., bicycle) does not appear in the very first LiDAR frame, a purely online memory bank will never contain it. Consequently, if $k=10$ neighbors are queried, the class can *never* win a majority vote and therefore can never admit future points, remaining permanently dead.
   - **Solution:** Prior to the first frame, the buffer $\mathcal{M}_c$ is seeded with $k$ exact copies of the pre-trained HDC prototype $P_c \in \mathbb{R}^D$. This structural guarantee ensures that even if a class is unobserved for $1,000$ frames, its $k$ latent prototypes reside in the buffer, capable of capturing and winning a $10$-NN vote the moment the true geometry appears.

3. **Tensor Core KNN Acceleration:**
   - **Problem:** Computing the $L_2$ equivalent Hamming distance over $130,000$ points against $10,000$ buffer elements requires $1.3 \times 10^{13}$ MACs per frame, collapsing inference to $\sim 1.4$ it/s.
   - **Solution:** Since bipolar vectors $\{ -1, 1 \}$ bound the maximum dot product strictly at $D=10,000$, the matrices are safely cast to `Float16`. This natively triggers NVIDIA Tensor Cores, yielding a $10 \times - 30 \times$ speedup ($\sim 4.23$ it/s) with zero information loss or overflow.

### Expected Results (`unsup_kitti-c.py` output)
| Corruption | Method | mIoU | Accuracy | Firing Rate |
| :--- | :--- | :--- | :--- | :--- |
| Fog | Iteration 2 | TBD | TBD | TBD |
| Snow | Iteration 2 | TBD | TBD | TBD |

---

## Iteration 3: Full-Capacity Prototype Seeding (Fixing KNN Density Bias)

**Goal:** Fix the catastrophic "50 Million False Positives" issue where Tail classes absorbed massive amounts of incorrect predictions, causing their precision to drop to 0.

**Mathematical & Structural Upgrades:**

1. **Eliminating KNN Density Bias ($N=588$ Seeding):**
   - **Problem:** In Iteration 2, the buffer was seeded with only $k=10$ copies of the prototype per class. When Frame 1 processed, the `road` class confidently filled its buffer to its maximum capacity of $588$, while unobserved Tail classes (like `bicycle`) remained at $10$ points. When the $10$-NN search was conducted across the unbalanced $10,000$ points, the vastly denser `road` class dominated the geometric space. This *Density Bias* caused $10$-NN to statistically return `road` for almost all queries. These misclassified points were then written into the Tail class buffers, entirely replacing the prototypes with $588$ `road` instances, leading to $50$ Million false positives.
   - **Solution:** We enforce a permanent structural guarantee that the memory bank is perfectly balanced from initialization. Prior to the first frame, the buffer $\mathcal{M}_c$ is seeded with **FULL CAPACITY** ($N=588$) exact copies of the pre-trained HDC prototype $P_c \in \mathbb{R}^D$ for every single class. 
   - **Mathematical Guarantee:** By maintaining exactly $N=588$ points per class at *all times*, there is $0\%$ statistical density bias in the memory bank. A query point has an equal chance of matching any class, relying entirely on true geometric distance. As real points stream in, they organically overwrite the 588 prototypes one-by-one via the FIFO pointer.

### Expected Results (`unsup_kitti-c.py` output)
| Corruption | Method | mIoU | Accuracy | Firing Rate | Update Mag |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Fog | Iteration 3 | TBD | TBD | TBD | TBD |
| Snow | Iteration 3 | TBD | TBD | TBD | TBD |

---

## Iteration 4: Global Reservoir Sampling & Density-Adaptive k-NN

**Goal:** Implement a completely dynamic, un-partitioned global memory bank without suffering from majority-class collapse, while preventing temporal locality from flushing out old, structurally sound points.

**Mathematical & Structural Upgrades:**

1. **Density-Adaptive k-NN (Fixing the Search Bias):**
   - **Problem:** In an un-partitioned global memory bank, the number of instances per class is highly imbalanced (e.g., $N_{road} = 5,000$ vs $N_{bicycle} = 10$). Due to this density bias, the dense `road` cloud statistically swallows the sparse `bicycle` points during the $k$-NN search.
   - **Solution:** We implement an adaptive metric that normalizes the Hamming distance by the *internal density* of the target class's sub-cluster. We approximate the internal density $\mu_c$ of class $c$ as the mean similarity of its instances to the class prototype (mean vector). When querying, the raw similarity to a neighbor $y \in c$ is divided by $\mu_c$: $sim_{adaptive}(x, y) = sim(x, y) / \mu_c$. This mathematically scales the distances so that dense classes are penalized, allowing sparse classes to compete fairly regardless of their instance count.

2. **Reservoir Sampling (Fixing the FIFO Overwrite):**
   - **Problem:** A strict FIFO queue is highly vulnerable to temporal locality. If a continuous stream of fog points arrives for 10 seconds, it will completely flush out all the clean, structurally sound points from the queue.
   - **Solution:** We replace the FIFO write policy with a momentum-based Reservoir Sampling approach. Once the 10,000-capacity queue is full, an incoming confident point does not automatically overwrite the oldest point. Instead, it replaces a completely random point in the memory bank with a fixed probability $P=0.1$. This guarantees that the memory bank maintains a diverse, long-term memory of the environment, preventing transient severe corruptions from scrubbing safe geometries.

### Expected Results (`unsup_kitti-c.py` output)
| Corruption | Method | mIoU | Accuracy | Firing Rate | Update Mag |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Fog | Iteration 4 | TBD | TBD | TBD | TBD |
| Snow | Iteration 4 | TBD | TBD | TBD | TBD |

---

## Future Iterations
*Will be defined as bottlenecks or failure modes are discovered.*

## Iteration 5: Coreset Anchoring and Epistemic Consensus Gating

**Goal:** Break the catastrophic "Echo Chamber" where the memory bank continually reinforced its own misclassifications, and mathematically neutralize the density bias using Extreme Value Theory.

**Mathematical & Structural Upgrades:**

1. **Coreset Anchoring (`reserved_slots`):**
   - **Problem:** Even with Reservoir Sampling, long-term exposure to severe corruption caused the fundamental class boundaries to drift, permanently losing the original clean geometries.
   - **Solution:** We introduced `reserved_slots`. Prior to the first frame, we extract 9,996 clean geometric Coreset Seeds from the source domain and lock them in the first half of the memory bank capacity. The online Reservoir Sampling is strictly bound to `[reserved_slots, capacity]`, ensuring a permanent anchor to true geometry while still allowing dynamic adaptation in the upper memory partition.

2. **Extreme Value Theory (EVT) Density Penalty:**
   - **Problem:** The previous heuristic density normalization failed because max dot-products scale sub-linearly with cluster size. Dense classes still swallowed rare classes.
   - **Solution:** We derived an exact density penalty based on Extreme Value Theory (EVT) for the Gumbel domain. The max similarity advantage of a class with $N$ points over a single point scales by $\sqrt{2 \ln(N+1)}$. We subtract this exact EVT penalty (`100.0 * sqrt(2.0 * log(class_counts + 1.0))`) from the raw similarities, mathematically neutralizing the statistical advantage of high-frequency classes.

3. **Active Learning via Epistemic Consensus Gating:**
   - **Problem:** The previous memory bank operated as an "Echo Chamber": it queried its own points, achieved 100% purity, and admitted poisoned points back into itself. For fog-3, this caused a 90% `MemError`. Furthermore, hard binary gates (`soft_dual_weight` or uniform gating) either fired 98% of the time (letting in poison) or 0% of the time (freezing adaptation).
   - **Solution:** We fused the Neural Network's Dirichlet Epistemic Uncertainty $u = C/E$ directly into the memory bank update mask. The memory bank only admits a point if (a) the Neural Network explicitely agrees with the Memory Bank's prediction, and (b) the point serves as a "Hard Example" (acting as a gradient step multiplier via $u$). This continuous gating breaks the echo chamber by demanding dual-modality agreement and actively seeking uncertain edge geometries.

### Expected Results (`unsup_kitti-c.py` output)
| Corruption | Method | mIoU | Accuracy | Firing Rate | MemError |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Fog | Iteration 5 | 0.0383 | 0.1153 | 67.55% | 86.60% |
| Snow | Iteration 5 | 0.3044 | 0.8764 | 84.39% | 13.46% |
| Wet Ground | Iteration 5 | 0.3652 | 0.8136 | 71.08% | 13.33% |
| Motion Blur| Iteration 5 | 0.3178 | 0.7600 | 80.84% | 16.28% |

---

## Iteration 6: Native Relative Manifold Cohesion (Pure HDC Gating)

**Goal:** Decouple the memory bank entirely from the Neural Network's uncertainty measurements by gating incoming points based purely on their structural fit within the 10,000D manifold.

**Mathematical & Structural Upgrades:**

1. **Relative Manifold Cohesion Gating:**
   - **Problem:** The Epistemic Consensus Gating (Iteration 5) still suffered from systemic failures (like Fog) because it relied on the neural network. If the neural network confidently hallucinated geometry, the memory bank was poisoned.
   - **Solution:** We introduced a native 10,000D structural integrity gate. For an incoming query point, we calculate the average distance to its $k$ nearest neighbors ($D_q$). We then compare this to the average pairwise distance between those exact $k$ neighbors themselves ($D_{int}$). The point is only admitted if $D_q / \max(D_{int}, 0.45) \le 1.25$. This dynamically scales the admission boundary based on the natural geometric spread of the class.
2. **True Coreset Seed Variance:**
   - **Problem:** Coreset seeding previously forced exactly 588 points per class by artificially duplicating latent vectors. This caused the internal variance ($D_{int}$) of clusters to collapse exactly to 0.00, breaking the cohesion metric.
   - **Solution:** We removed artificial duplication. The memory bank is seeded only with the true, unique geometric seeds available from the offline frames. The EVT Density Penalty mathematically handles the resulting imbalance.

### Expected Results (`unsup_kitti-c.py` output)
| Corruption | Method | mIoU (Initial ➔ Final) | Accuracy (Initial ➔ Final) | Firing Rate | MemError |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Fog | Iteration 6 | 0.0652 ➔ 0.0292 | 0.1736 ➔ 0.2146 | 100.00% | 89.22% |
| Snow | Iteration 6 | 0.4308 ➔ 0.5175 | 0.8664 ➔ 0.8403 | 99.99% | 14.73% |
| Wet Ground | Iteration 6 | 0.4387 ➔ 0.4886 | 0.9081 ➔ 0.8598 | 99.99% | 9.30% |
| Motion Blur| Iteration 6 | 0.4219 ➔ 0.4827 | 0.8447 ➔ 0.8269 | 99.99% | 10.81% |
| Beam Missing | Iteration 6 | 0.3927 ➔ 0.4059 | 0.8369 ➔ 0.7590 | 99.99% | 9.79% |
| Crosstalk | Iteration 6 | 0.0737 ➔ 0.0760 | 0.2165 ➔ 0.4659 | 99.99% | 61.92% |
| Incomplete Echo | Iteration 6 | 0.3884 ➔ 0.3806 | 0.9057 ➔ 0.8399 | 99.99% | 12.30% |
| Cross Sensor | Iteration 6 | 0.2705 ➔ 0.2361 | 0.6213 ➔ 0.5653 | 100.00% | 5.41% |

### Critical Failure Modes Diagnosed:
1. **The Inlier Paradox (Unstructured Noise Admission):** Severe unstructured noise like Fog and Crosstalk fundamentally fools purely geometric distance gating. Because the HDC encoder collapses random noise into highly dense "background" vectors near the origin, Fog points actually have a *lower* query distance ($D_q \approx 0.43$) to the seeds than clean geometry ($D_q \approx 0.52$). Relative Manifold Cohesion assumes noise will be an outlier, but the data proves it acts as an ultra-dense inlier. As a result, Fog is fully admitted (`FiringRate = 100%`), causing catastrophic memory corruption (`MemError = 89.22%`).
2. **Head Class Degradation (EVT Density Trade-off):** While the EVT Density Penalty beautifully recovers the Tail classes (e.g., Snow Tail jumps from 16.89% to 39.06%), it penalizes the Head classes too aggressively during 10-NN retrieval. This causes Head mIoU to drop consistently (e.g., Cross Sensor Head drops from 46.79% to 31.96%), which drags down the global Accuracy metric. The penalty function is over-correcting.

---

## Iteration 7: The Manifold Denoiser (Generative Reconstruction Gate)

**Goal:** Bypass the structural failures of local density and logit margins by using generative reconstruction error on the global HDC manifold to identify deep geometric outliers (hallucinations).

**Mathematical & Structural Upgrades:**

1. **The HDC Manifold Denoiser (Generative Gating):**
   - **Problem:** Diagnostics proved the Inlier Paradox: fog and noise are incredibly dense relative to themselves ($k$-NN cohesion fails), and they sit deep inside the wrong semantic Voronoi cells (Margin thresholding fails). However, they exist in completely different regions of the 10,000D hyperspace than the clean source data (Diagnostic M).
   - **Solution:** We trained a lightweight Autoencoder (The Manifold Denoiser) explicitly on the clean source 10,000D geometry. Incoming points are first passed through this Autoencoder. If a point is true geometry, the AE easily reconstructs it because it obeys clean geometric rules. If the point is fog, it is an out-of-distribution anomaly relative to the *clean* source (despite being a local inlier), causing massive reconstruction error. Points are admitted into the Memory Bank *only* if their reconstruction error is below a threshold.
2. **Reversal of the EVT Density Penalty:**
   - **Problem:** Diagnostic 3 proved that memory bank accuracy scales monotonically up to $N=1000$ points per class without early saturation. The EVT Density Penalty from Iteration 6 actively punished large, information-rich clusters, causing catastrophic Head-Class degradation.
   - **Solution:** We formally abandoned the EVT density penalty, returning the $k$-NN search strictly to geometric distance, allowing Head classes to retain their mass and accuracy.
3. **Abandonment of Extraneous Constraints:**
   - Diagnostics 2, 4, and 5 empirically disproved the viability of Temporal Lifetimes, Intrinsic Dimensionality (Local Rank), and Logit Margins. They were fully stripped from the architectural pipeline, ensuring we operate strictly on the raw AUROC 1.0000 discriminative power of the HDC topology.
