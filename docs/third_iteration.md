# Phase 3: Continual Learning & Global Memory Dynamics
**Date:** [Pending - Drafted July 23, 2026]

## Objective
Following the completion of Phase 2 (Class-Balancing and Multi-View consistency), Phase 3 will transition the system from isolated, single-domain adaptation chunks to **true, continuous online learning**. We will evaluate the model's ability to seamlessly traverse sequential domain shifts (e.g., Clean $\rightarrow$ Snow $\rightarrow$ Rain $\rightarrow$ Night) without ever resetting the model state, focusing on preserving forward plasticity while mitigating catastrophic forgetting.

---

## Part A: The Global Spring (Physics-Based Plasticity)

Our formalized S2 schedule (Norm-Driven Dynamic LR) perfectly regulates learning rates by inflating a momentum vector $M_c$ when a class fires, which effectively freezes classes once they have confidently adapted. 

However, in a true continual learning environment, this creates a vulnerability:
* When a domain shifts (e.g., Snow to Night), the old frozen prototypes will start generating errors.
* The epistemic uncertainty gates will correctly veto these errors, preventing the class from firing.
* If the class doesn't fire, its momentum $M_c$ remains massive, and the class is permanently frozen in the old domain's geometry.

**The Implementation:** We will move the `anchor_spring` *outside* the firing loop. 
$$ M_c \leftarrow (1-k)M_c + k \cdot 1.0 $$
By applying the spring unconditionally to all classes every frame, the system gains a self-regulating domain-shift detector:
1. **Confident Classes:** Fire frequently. Their $M_c$ stays inflated, keeping them frozen and stable.
2. **Shifted Classes:** Hit a domain shift, generate errors, and are vetoed. They starve. As they starve, the global spring silently erodes their $M_c$ back to $1.0$. This automatically **"thaws"** the class, spiking its learning rate and allowing it to rapidly snap to the new domain.

---

## Part B: Sequential Multi-Domain Benchmarking

We will modify the core evaluation loop in `unsup_kitti-c.py` to support a `--continual` flag.
* Disable `model.load_state_dict(clean_state_dict)` between corruptions.
* Track the temporal evolution of mIoU across domain boundaries.
* **Metrics:** We will measure both *Forward Transfer* (how fast the model adapts to a new domain given its prior momentum) and *Catastrophic Forgetting* (how much performance degrades if we revisit an earlier domain).

---

## Part C: Long-Term Epistemic Drift Protection

As the model adapts indefinitely, there is a risk that the continuous incorporation of new geometric clusters will slowly degrade the absolute semantic boundaries of the feature space.
* **Continual Veto Purity:** Ensure that the hyperdimensional Dirichlet gates do not gradually widen and accept noise over millions of frames.
* **Dynamic Anchor Re-weighting:** Investigate if the $w_0$ (Clean pre-trained weights) anchor needs to be updated or if it should permanently anchor the system against infinite drift.

*(Note: Execution of Phase 3 is paused pending the full resolution of Phase 2 Inter/Intra-class balancing and Multi-View Architecture tests).*
