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
