Sections

## Prior Issues

Our recent diagnostic runs have definitively proven why textbook Euclidean test-time adaptation methods (like `ConformalHDC` and `StandardT3A`) catastrophically fail when ported into high-dimensional computing (HDC) architectures. The failure stems from three profound structural incompatibilities:

### 1. The Inseparability of the 128D Latent Space
Standard prototype methods typically cluster and update prototypes using the backbone's low-dimensional latent space (e.g., 128D). However, in our EvidentialHDC network, the 128D space is merely an intermediate representation that is immediately passed through a massive, non-linear expansion (via fixed random projections and `hard_quantize` non-linearities) into 10,000D hypervectors. The 128D features are structurally not linearly separable via cosine similarity. When Euclidean baselines attempt to classify points using 128D centroids, the predictions overlap heavily, resulting in essentially random chance performance (e.g., 4.4% initial mIoU on `fog`, compared to 6% in the true HDC space).

### 2. The L2 Normalization Pitfall in 10,000D
When we adapt these methods to correctly operate in the 10,000D space (`ConformalHDC_10k`), they still fail due to prototype normalization. Textbook methods explicitly apply L2 normalization to the learned prototypes ($W = \text{F.normalize}(W, \text{dim=1})$). In an HDC network, however, the unnormalized magnitude of the `classify.weight` vectors is critical, as it acts as a learned class prior. Stripping this magnitude causes a massive, immediate drop in initial performance (dropping from the true frozen baseline of 42.02% down to 36.08% on `wet_ground`). 

### 3. The Softmax Temperature Blockade in High Dimensions
Euclidean baselines often rely on confidence thresholding (e.g., admitting pseudo-labels if $conf > 0.90$). To compute this, they scale the cosine similarities by a fixed temperature (e.g., $15.0$) before applying a softmax. However, the curse of dimensionality dictates that random vectors in a 10,000-dimensional space have a cosine similarity extremely close to zero ($\approx 0.01$). Multiplying by 15.0 yields tiny logits, which completely flattens the softmax distribution. The maximum confidence caps out around $\sim0.06$. As a result, a hard-coded $0.90$ threshold acts as a total blockade, rejecting *every single point* and preventing any adaptation from taking place.

### Conclusion for Adaptation
Test-time adaptation in HDC cannot be accomplished by naïvely porting standard Euclidean mechanisms. It requires HDC-native mechanics:
- Adapting the unnormalized prototypes directly in the 10,000D space.
- Utilizing Evidential epistemic uncertainty rather than softmax confidence, as softmax distributions flatten irrecoverably in high dimensions.
- Implementing careful, HDC-specific gating (like the M-series component ladder) to prevent prototype hallucination and runaway drift under severe domain shifts.