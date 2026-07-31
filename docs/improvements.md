# AdaptMem Improvement Log

### Known Bottlenecks and Areas for Optimization

1. **KNN Search Efficiency (`4.23 it/s`)**
   - **Current State:** The method runs at ~4.23 it/s during adaptation, which is slower than legacy TTA methods (~14 it/s). 
   - **Root Cause:** The `torch.mm` matrix multiplication for querying `130,000` points against the `10,000` point memory bank requires roughly 13 Trillion MAC operations per frame. While `float16` Tensor Cores improved this significantly (from 1.45 it/s to 4.23 it/s), it still remains the primary bottleneck.
   - **Proposed Solutions:**
     1. **FAISS Integration:** Implement FAISS (`IndexBinaryFlat` or HNSW) for exact or approximate Hamming distance search natively in HDC space.
     2. **Custom CUDA Kernel:** Write a custom CUDA kernel optimized for binary XOR operations (bitwise Hamming distance) to leverage the 1-bit native format of the features.
     3. **Weighted Representative Vectors:** Compress the memory bank by clustering or finding weighted representative vectors (e.g. prototypes) to drastically lower the number of elements we need to compute against.

2. **Memory Bank Sampling**
   - **Current State:** The memory bank uses uniform random sampling via `torch.randperm` when evicting/admitting points to avoid spatial bias.
   - **Proposed Solution:** Evaluate if a stratified sampling approach based on semantic class labels or entropy (keeping points that represent rare classes rather than just uniformly sampling) would improve performance.