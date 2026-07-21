# Phase 2: Advanced Class-Balancing and Multi-View Architectures
**Date:** July 21, 2026

## Objective
Following the establishment of our core 4-tier adaptation pipeline (Network, Hypervector, and Temporal uncertainties with basic frequency protection) in the preliminary phase, the objective of Phase 2 is to rigorously refine the class-balancing mechanisms and introduce multi-view architectures. We aim to identify and exploit the highest leverage areas for improvement, specifically targeting rare-class preservation (inter-class balance), sub-cluster geometric integrity (intra-class balance), and multi-perspective spatial consistency.

## Background & Context

### 1. The Class-Balancing Challenge
During preliminary testing, we discovered that standard threshold-based Test-Time Adaptation suffers from massive "Voronoi Shattering," where majority classes (like Road or Vegetation) confidently dominate updates and geometrically crush rare class prototypes. 
While our initial heuristic—a deterministic frequency soft-dampener ($\gamma=0.1$)—successfully mitigated the worst of this "Majority Class Paralysis" without completely freezing the network, it remains a blunt instrument. 

In this phase, we will explore:
* **Advanced Inter-Class Balancing:** Moving beyond simple inverse frequency to true dynamic semantic weighting that adapts to the shifting class distributions of varying environments.
* **Intra-Class Balancing:** Rare sub-clusters (e.g., a specific pose of a Pedestrian) within a single class are often vetoed as Out-of-Distribution (OOD) noise. We will investigate methods to protect rare geometries from being smoothed out by the dense "core" of the class manifold, revisiting concepts like the Subcluster Ledger with more refined, non-restrictive math.

### 2. Multi-View Architectures
Single-frame adaptation is inherently vulnerable to transient noise and occlusions. By leveraging multi-view architectures, we can enforce spatial and semantic consistency across multiple perspectives or temporal frames before committing to a permanent weight update. 

In this phase, we will investigate:
* **Multi-View Consistency Gating:** Requiring consensus across multiple spatial projections (or consecutive LiDAR sweeps) before allowing high-magnitude updates.
* **Feature Fusion:** Aggregating features from multiple views into the 10,000D hyperdimensional space to generate a more robust, physically grounded Dirichlet uncertainty prior.

## Preliminary Tests & Baselines
*(Test logs, ablations, and empirical results for multi-view and advanced balancing runs will be recorded here as the iteration progresses.)*
