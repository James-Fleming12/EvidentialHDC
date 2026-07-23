# S2: Formalizing the Emergent `1/t` Per-Class Decay (Norm-Driven Dynamic LR)

The PyTorch `.data` bug accidentally created a mathematically beautiful **Dynamic Class-Balancing** learning rate schedule. We will now formalize this into a clean, explicit algorithm (S2).

## The Math Behind the Magic
In the Bug Reproduction run, the prototype vectors $w_c$ were not normalized after steps.
When a class fires, the un-normalized accumulation acts as:
$w_c \leftarrow w_c + \text{step\_vec}$

If the steps are roughly orthogonal to $w_c$, the norm $M_c = \|w_c\|$ grows linearly with the cumulative magnitude of all steps applied to that class:
$M_c \approx 1.0 + \sum \text{step\_mag}$

Because the forward pass always normalizes $w_c$, the effective angular step size on the unit hypersphere is exactly:
$\alpha_c = \frac{\text{step\_mag}}{M_c}$

1. **Majority Classes (e.g., Road):** Fire constantly. Their norm $M_c$ inflates rapidly (e.g., reaching 10.0 or 30.0). Their effective learning rate plummets to $1/30$, effectively freezing them and protecting their established geometry.
2. **Rare Classes (e.g., Bicycle):** Rarely fire. Their norm $M_c$ stays close to $1.0$. When they do fire, they get the full, maximum learning rate, allowing them to rapidly capture new rare geometries without being squashed.

This perfectly explains why my explicitly coded `1/t` schedule failed! My `beta = 0.05` was 10x to 100x too aggressive, freezing all classes instantly. The PyTorch bug was implicitly using a `beta` equal to the actual step magnitudes ($\approx 0.001$), resulting in a slow, elegant decay.

## The Implementation (S2)
Instead of relying on un-normalized tensors, we will explicitly track the cumulative step scalar $M_c$ for each class and use it to divide the step magnitude before applying it to the true normalized prototypes.

```python
# 1. Update the tracking scalar
model.class_M[c] += step_mag

# 2. Decay the current step
effective_step = step_mag / model.class_M[c]

# 3. Apply the step on the clean normalized vector
model.classify.weight[c].data += effective_step * c_update
```
