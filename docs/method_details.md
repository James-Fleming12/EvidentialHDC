# Adaptive Memory Bank Architecture

Based on the empirical findings of the Representation and Adaptation Characterization Framework, we transition from prototype (centroid-based) adaptation to a high-dimensional nearest-neighbor (k-NN) Memory Bank.

## Core Directives

### 1. Operate Natively in HDC Space (10,000D)
**Justification:** Prototype projection linearly collapses the geometric rank of the space (from 13.3 to 2.8) and severely scrambles local neighborhoods (Spearman rank 0.3892). However, Phase I proves the full 10,000D space perfectly preserves semantic information. We must operate prior to the geometric bottleneck.

### 2. k-NN Consensus for Hallucination Gating
**Justification:** The mangled 17D geometry generates ~91,000 confident hallucinations per frame. Because relative distances in 17D are structurally invalid, standard `max(logit)` confidence gating triggers massive confirmation bias (Prototype Velocity 0.2355). We must use a dynamic Memory Bank and k-NN consensus in 10,000D to rigorously gate predictions based on topological consistency.

### 3. Memory Capacity Bounded to 10,000 Elements
**Justification:** Phase IV memory scaling proves that semantic accuracy and Hallucination Detection AUROC saturate at exactly 10,000 elements. A memory bank of 10,000 float32 points consumes only ~381 MB of VRAM, making it exceptionally efficient for embedded deployment.

### 4. Inference Acceleration (FAISS / Subsampling / Binarization)
**Justification:** A naive full-frame matrix multiplication query ($130,000 \times 10,000$) in PyTorch takes ~638 ms per frame (1.6 FPS). To hit the $>10$ FPS robotics constraint, the inference module must implement hardware acceleration (e.g., *Billion-scale similarity search with GPUs*) or aggressive point-cloud subsampling (e.g., down to 20,000 points) prior to query.

### 5. Robust Queue Hygiene (Graph-Pruning)
**Justification:** A blind FIFO queue will become instantly corrupted by fog hallucinations. Drawing on *Momentum Contrast (MoCo)* and *Robust Self-Training via Nearest Neighbor Graphs*, the memory bank must employ a structural verification step (e.g. density consensus) to strictly prune hallucinated outliers before they enter the buffer.

### 6. (Optional) Invertible Feature Recovery
**Justification:** If memory/compute dictates we *must* operate the classifier in a lower-dimensional space, Phase V proves the 17D projection bottleneck is mathematically reversible (0.9192 cosine similarity via MLP). We could employ an Invertible Neural Network (*Invertible Neural Networks for Feature Recovery*) to classify efficiently while preserving exact recovery paths for adaptation.
