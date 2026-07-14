# Implementation Architecture for EvidentialHDC

This document outlines the software architecture and module breakdown for the `UQModel`, translating the pivot strategy (multi-source UQ and balanced allocation) into concrete methods. 

## Philosophy

The goal is to keep the adaptation logic completely modular. Instead of a single massive `inference_update` function that does everything, the gating decision is split into independently testable components. This allows us to ablate each uncertainty source exactly as described in the README.

---

## 1. UQModel Function Placeholders

To maintain cleanliness and organization in `modules/HDC_utils.py`, the `UQModel` class should implement the following method signatures. 

### Core Inference & Updating
```python
    @torch.no_grad()
    def get_confidence(self, enc, preds=None, method='hybrid'):
        """
        Master method to compute the confidence score for gating.
        Routes to the appropriate underlying uncertainty methods based on the requested string.
        """
        pass

    @torch.no_grad()
    def online_update(self, x, proj_xyz=None, update_method='hybrid_balanced'):
        """
        The primary entrypoint for test-time adaptation. 
        Calls `get_confidence`, applies thresholds, consults the subcluster ledger (if balanced), 
        and updates the prototype weights `self.classify.weight`.
        """
        pass
```

### Pillar 1: Independent Uncertainty Sources
To ablate each source of uncertainty, they must be cleanly separated:

```python
    @torch.no_grad()
    def _get_epistemic_uncertainty(self, x, enc):
        """
        Pillar 1(b): Network Uncertainty.
        Estimates uncertainty before the softmax.
        Candidates to implement here:
        - Multi-RP ensemble (multiple random projections)
        - Feature-space density
        - Evidential deep learning mass
        
        Returns a score in [0, 1] where 1 is highly reliable.
        """
        pass

    @torch.no_grad()
    def _get_spatial_consistency(self, enc, preds, proj_xyz):
        """
        Pillar 1(c): Spatial/Temporal Consistency.
        Uses `proj_xyz` (3D coordinates) to check if a point's predicted label 
        agrees with its physical neighbors. Acts as a hard veto against fog/noise artifacts.
        
        Returns a binary mask or soft score in [0, 1].
        """
        pass

    @torch.no_grad()
    def _get_geometric_confidence(self, enc, preds):
        """
        Pillar 1(a): HD Space Geometry.
        The baseline standard prototype cosine similarity.
        
        Returns similarity score in [-1, 1].
        """
        pass

    @torch.no_grad()
    def _fuse_uncertainties(self, epistemic, consistency, geometric):
        """
        Combines the independent uncertainty scores into a single gating metric.
        Must be calibrated so that a failure in one source (e.g. geometric drift) 
        can be overridden by another (e.g. epistemic rejection).
        """
        pass
```

### Pillar 3: Headroom-Based Update Allocation (The Ledger)

The subcluster ledger exists entirely outside the inference path. It is purely an accounting mechanism to throttle updates for saturated classes (like Road) while funneling updates to rare/high-headroom classes (like Car or Motorcycle).

```python
    @torch.no_grad()
    def _initialize_subcluster_ledger(self):
        """
        Initializes K subclusters per class using K-Means or source-domain density.
        Initializes a counter array `self.subcluster_update_counts = zeros(NUM_CLASSES, K)`.
        """
        pass

    @torch.no_grad()
    def _consult_budget_ledger(self, enc, preds, candidate_mask):
        """
        Takes the mask of points that passed the UQ gate (`candidate_mask`).
        1. Maps each candidate point to its nearest subcluster.
        2. Checks `self.subcluster_update_counts`. If a subcluster has exceeded its 
           relative budget (e.g., it has N more updates than its siblings), 
           it is temporarily frozen.
        3. Returns a refined mask where saturated points are dropped.
        4. Increments the ledger counts for the points that are ultimately admitted.
        """
        pass
```

---

## 2. Execution Flow

When `unsup_main.py` calls `model.online_update(x, proj_xyz, update_method='hybrid_balanced')`, the internal flow should be:

1. **Forward Pass**: Extract features `enc`.
2. **Classification**: Get initial `preds` using standard prototype distance.
3. **Uncertainty Estimation**: 
   - Call `get_confidence(enc, preds, method='hybrid')`.
   - Internally, this calculates epistemic, spatial, and geometric scores, and fuses them.
4. **Thresholding**: Apply the quantile or absolute threshold to the fused confidence score to get the `candidate_mask`.
5. **Ledger Allocation**: 
   - If `update_method == 'hybrid_balanced'`, pass `candidate_mask` to `_consult_budget_ledger()`.
   - Saturated high-frequency regions are dropped. The mask is finalized.
6. **Prototype Update**: Multiply the valid `enc` vectors by a learning rate and add them to the appropriate class prototypes in `self.classify.weight`.

---

## 3. Immediate Next Steps

1. **Implement `_get_epistemic_uncertainty`**: We need to decide on the specific math for the epistemic UQ. (Will it be temperature scaling on the HD logits, or a direct feature-space evidential loss during pre-training?).
2. **Design the Budget Condition**: How exactly do we define "saturated"? (e.g., `count(subcluster_i) > mean(counts_in_class) + margin`).
