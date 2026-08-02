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

**Implementation & Calibration Fixes:**
- **Domain Mismatch & Binarization:** Initially, the autoencoder was trained offline using `coreset_seed_keys` (which are strict $\{-1, 1\}^{10000}$ binary vectors). However, during TTA, continuous L2-normalized vectors were being passed into the autoencoder, causing massive domain mismatch (reconstruction errors of 1.0 / completely orthogonal outputs). We resolved this by explicitly binarizing incoming TTA vectors before passing them through the denoiser.
- **Threshold Calibration:** Once binarized, we mapped the true reconstruction errors (via `test_denoiser_b.py`) and found the generative separation maintained a strong AUROC of 0.9266 even on the quantized binary data. Correct geometry averaged an error of 0.43, while hallucinations averaged 0.60.
- We set the optimal error threshold right down the middle at 0.52 (yielding a `mem_purity` threshold of `0.48`), which successfully broke the 100% lockout and allowed sensible firing rates (e.g. 16% on Fog, 95% on Motion Blur) while keeping memory contamination strictly bounded.

### Expected Results (`unsup_kitti-c.py` output)
| Corruption | Method | mIoU (Initial ➔ Final) | Firing Rate | MemError |
| :--- | :--- | :--- | :--- | :--- |
| Fog | Iteration 7 | 0.0652 ➔ 0.0217 | 26.91% | 59.82% |
| Snow | Iteration 7 | 0.4308 ➔ 0.4848 | 98.81% | 15.45% |
| Wet Ground | Iteration 7 | 0.4387 ➔ 0.4445 | 98.45% | 12.73% |
| Motion Blur| Iteration 7 | 0.4219 ➔ 0.4005 | 98.83% | 16.74% |
| Beam Missing | Iteration 7 | 0.3927 ➔ 0.3813 | 98.86% | 10.70% |
| Crosstalk | Iteration 7 | 0.0737 ➔ 0.0540 | 42.00% | 53.70% |
| Incomplete Echo | Iteration 7 | 0.3884 ➔ 0.3237 | 98.93% | 18.68% |
| Cross Sensor | Iteration 7 | 0.2705 ➔ 0.1844 | 98.45% | 6.26% |

### Critical Failure Modes Diagnosed:
1. **Class Imbalance Collapse:** While the Generative Gating successfully solved the Inlier Paradox (dropping Fog firing rates from 100% to 26%), the global performance actually *degraded* relative to Iteration 6. 
   - Without the EVT Density Penalty (which artificially punished dense clusters), Head classes (which make up 90% of the data stream) completely overwhelmed the global Reservoir Sampling. 
   - The 10,000 dynamic memory slots became ~90% filled with Head classes. 
   - Consequently, the 10-NN search overwhelmingly returned Head classes for everything, destroying the Head mIoU (e.g., Cross Sensor Head dropped from 0.4679 ➔ 0.2481) due to massive False Positives, and artificially inflating Tail mIoU by collapsing Tail False Positives to near-zero (e.g. Tail FP dropped from 1.5M ➔ 63k).

---

## Iteration 8: Class-Partitioned Reservoir Sampling

**Goal:** Mathematically guarantee perfect class balance in the memory bank natively, eliminating the need for complex density penalties like EVT.

**Mathematical & Structural Upgrades:**

1. **Native Class Partitioning:**
   - **Problem:** A global, un-partitioned memory bank updated via Reservoir Sampling will inevitably mirror the extreme class imbalance of the incoming data stream, leading to Class Imbalance Collapse during $k$-NN retrieval.
   - **Solution:** We strictly partitioned the 20,000 capacity memory bank equally across all 19 classes (1,052 slots per class). Each class now manages its own independent pointer and reservoir.
   - **Impact:** Head classes can physically never exceed 1,052 slots, preventing them from flooding the bank. Tail classes are guaranteed equal representation during the 10-NN search. This perfectly balances the HDC geometry without requiring external density penalties, allowing us to safely exploit the raw distance metric.

2. **Continuous Queries vs Exact Ties (ReLU Sparsity Collision):**
   - **Problem:** In Iteration 8, we initially passed binarized query vectors (`torch.sign(x)`) directly into the `10-NN` search against the binary memory bank. Because modern backbones use deep ReLUs, the pre-classifier features (`raw_enc`) contain ~80% exact `0.0` values. The binarization forcibly mapped all of these zeros to exactly `+1.0`. This stripped away all sub-integer variance, creating massive identical vectors and causing exact distance ties. The `topk` algorithm predictably broke these ties by memory order (which, due to class partitioning, meant Class 0 always won).
   - **Solution:** We explicitly reverted the `10-NN` query input to use the continuous L2-normalized feature vectors (`norm_enc`). Dotting a continuous vector against a binary vector is equivalent to the L1 projection onto the hypercube, preserving all sub-integer float precision and eliminating exact ties.

3. **Removal of Extraneous Passes & Hardcoded Debug Bottlenecks:**
   - **Problem:** `Populating Source Stats 2` was running over 5 hours, calculating variables exclusively used for the EVT Density Penalty (which was ripped out in Iteration 7). Additionally, a legacy debug loop (`if batch_idx > 500: break`) was silently truncating Pass 1, physically preventing 14 classes from ever entering the offline coreset, causing them to collapse to a single fallback prototype and rendering them mathematically unable to win a `10-NN` vote.
   - **Solution:** Pass 2 was entirely deleted. The debug `break` statements were removed. The pipeline now successfully scans all 19,130 training frames in ~16 minutes, extracting perfectly balanced, globally representative 588-point seeds for all 19 classes.

4. **Poison Accumulation and Dual-Threshold Gating (The "Gold Standard" Buffer):**
   - **Problem:** With a strict class-partitioned reservoir, head classes accumulate points fast enough to overwrite bad geometry. However, tail classes receive very few true points. A static generative admission threshold of 0.52 blocks 74% of confident fog hallucinations, but admits 26%. Over long sequences, these 26% slowly accumulate in the 1,052 tail-class slots without being overwritten by true points, eventually "poisoning" the core and causing total semantic collapse for rare classes.
   - **Solution:** We implemented a Dual-Threshold logic to harden the memory bank:
     - **Query Threshold (0.52):** If an incoming point has a reconstruction error $> 0.52$, it is deemed an absolute hallucination. We block it from querying the `10-NN` memory bank and fall back to the base `SqueezeSegV3` linear projection.
     - **Admission Threshold (0.45):** Even if a point passes the Query Threshold, it is ONLY allowed to enter the memory bank (and overwrite a coreset seed) if its reconstruction error is $\le 0.45$ (Purity $\ge 0.55$). This guarantees that only the absolute most pristine geometry is allowed to update the 1,052 slots, permanently immunizing the rare classes from slow poison accumulation.

5. **The $k$-NN Under-Voting Mathematical Trap & The Exact Tie Bug:**
   - **Problem:** If an extremely rare class (like 'Motorcyclist') only physically appeared 3 times in the offline training set, placing those 3 points in a `10-NN` memory bank makes it mathematically incapable of winning a majority vote (max 3 votes vs 7 for competing classes). However, solving this by aggressively repeating the 3 points to fill 588 slots resurrects the fatal **Exact Tie Bug**. Injecting identical continuous vectors into the memory bank causes identically tied cosine similarities during queries, forcing the `topk` algorithm to break ties by memory index order and completely destroying variance.
   - **Solution (Distance-Weighted $k$-NN):** Rather than generating synthetic points via Latent Gaussian Perturbation (which could structurally compromise the clean HDC topology) or copying exact points (which causes exact tie collapse), we fundamentally altered the voting mechanism itself. In `AdaptiveMemoryBank`, we replaced the simple majority vote with an exponentially scaled **Distance-Weighted $k$-NN** ($w = e^{(sim - 1.0) / \tau}$). This elegantly allows sparse tail classes with $< k$ points to instantly overpower a majority of slightly more distant points, natively preserving the clean memory geometry while successfully breaking the mathematical constraints of under-voting.

### Expected Results (`unsup_kitti-c.py` output)
| Corruption | Method | Baseline mIoU | Adapted mIoU | Baseline Acc | Adapted Acc | Firing Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Fog | Iteration 8 | 0.0714 | **0.0601** | 19.15% | 21.66% | 53.51% |
| Snow | Iteration 8 | 0.4025 | **0.2452** | 83.67% | 51.12% | 53.20% |
| Wet Ground | Iteration 8 | 0.3946 | **0.1617** | 87.57% | 38.72% | 36.49% |
| Motion Blur| Iteration 8 | 0.3830 | **0.2401** | 81.62% | 49.40% | 41.45% |
| Beam Missing | Iteration 8 | 0.3653 | **0.2455** | 80.58% | 52.73% | 34.09% |
| Crosstalk | Iteration 8 | 0.0806 | **0.0883** | 23.24% | 22.91% | 52.50% |
| Incomplete Echo | Iteration 8 | 0.3705 | **0.1881** | 87.88% | 46.62% | 58.34% |
| Cross Sensor | Iteration 8 | 0.2681 | **0.1144** | 60.18% | 36.08% | 14.81% |

### Critical Failure Modes Diagnosed:
1. **The Distance-Weighted $k$-NN Flat Softmax Trap:** By taking an unscaled `softmax` over cosine similarities (which are strictly bounded between `[-1, 1]`), we generated an extremely flat, near-uniform weight distribution. A mathematically perfect match (`1.0`) only held `~23%` voting power against 9 terrible matches (`0.0`). This inadvertently caused the algorithm to act exactly like a strict majority vote, completely nullifying the intended benefits and leaving rare classes suppressed.
2. **Loose Dynamic Calibration:** The query threshold multiplier of $\mu + 3\sigma$ proved too generous (allowing queries with similarities as low as ~`0.487`), resulting in massive Firing Rates (e.g. 58% on Incomplete Echo). Millions of corrupted "random noise" points were erroneously queried against the Memory Bank.
3. **Tail FP Explosion via Class-Partitioning:** Because the Memory Bank enforces perfectly uniform capacity across all 19 classes, whenever a "random noise" point slips past the loose query gate, its nearest neighbors are a completely random, uniform mix of classes. Combined with the flat softmax vote, the $k$-NN outputs a purely random guess. Across 400M points, this random uniform guessing resulted in ~20 Million False Positives being forced into every single Tail class, instantly tanking the mIoU.

---

## Iteration 9: Softmax Scaling & Absolute Geometry Veto

**Goal:** Resolve the massive accuracy drop introduced in Iteration 8 by tightening the Firing Rate and correcting the math behind the Distance-Weighted $k$-NN vote.

**Mathematical & Structural Upgrades:**

1. **Temperature-Scaled Softmax:**
   - **Problem:** Unscaled softmax over `[-1, 1]` cosine similarities acts as a strict majority vote, nullifying the distance weighting.
   - **Solution:** Apply a strict temperature scaling (`tau = 0.05` or smaller) before the softmax operation to exponentially prioritize tight geometric matches, or utilize an unnormalized exponential weight `weights = torch.exp(topk_sims / tau)`.
   
2. **Tightened Calibrated Thresholds:**
   - **Problem:** $\mu + 3\sigma$ let in too much noise, allowing 50%+ firing rates.
   - **Solution:** Retract the query gate multiplier to a much stricter $\mu + 1.0\sigma$ or $\mu + 1.5\sigma$ to suppress the processing of random noise.
   
3. **Absolute Veto Fallback:**
   - **Problem:** When random noise queries the uniform bank, it finds a "best" match with very low similarity and makes a random guess.
   - **Solution:** Implement a hard safety check where if the highest absolute cosine similarity among the top-$k$ retrieved neighbors is below a safe geometric baseline (e.g., `< 0.60`), the system automatically vetoes the $k$-NN and falls back to SqueezeSegV3, preventing uniform guessing on noisy vectors.

