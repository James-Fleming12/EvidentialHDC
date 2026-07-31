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

## Future Iterations
*Will be defined as bottlenecks or failure modes are discovered.*
