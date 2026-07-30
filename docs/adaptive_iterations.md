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

## Future Iterations
*Will be defined as bottlenecks or failure modes are discovered.*
