# Toy Model Experiments: Addressing R1 & R2 Critical Feedback

This document outlines three self-contained computational experiments that
directly respond to the P1 reviewer requests. Each experiment is designed to
be reproducible, minimal in scope, and directly mappable to a claim in the
paper.

---

## Overview of experiments

| # | Experiment | Addresses | Estimated effort |
|---|------------|-----------|-----------------|
| 1 | Fisher computation + natural gradient vs SGD on logistic regression | R1 (toy worked example) | Low |
| 2 | Empirical Fisher / K-FAC summary scalars on a small transformer | R1 (LLM-relevant approximation) | Medium |
| 3 | QFI toy computation on a parameterized qubit state | R1 + R2 (quantum claim grounding) | Low–Medium |

All experiments are implemented in Python using NumPy, PyTorch, and
PennyLane. PyTorch operations in Experiments 1 and 2 use the MPS backend
(`torch.device("mps")`) available on Apple Silicon. Target runtime on a
MacBook Pro with MPS: under 5 minutes per experiment.

---

## Experiment 1 — Fisher information and natural gradient on logistic regression

### Purpose
Provide a minimal, fully reproducible demonstration that information geometry
("curvature matters") changes the optimization trajectory in a measurable,
concrete way. This directly answers R1's request for "a worked example where
the Fisher information matrix is computed explicitly."

### Setup

- **Model**: binary logistic regression with $d = 2$ or $d = 4$ input features.
- **Dataset**: 2D Gaussian blobs (scikit-learn `make_classification`) or a
  small UCI dataset (e.g. Iris, two classes). Use $N = 200$ samples.
- **Parameters**: weight vector $\theta \in \mathbb{R}^d$, bias included.

### What to compute

**Step 1 — Exact Fisher information matrix**

For logistic regression with output probability $p = \sigma(\theta^\top x)$,
the Fisher matrix has the closed form:

$$F(\theta) = \frac{1}{N} \sum_{i=1}^{N} p_i (1 - p_i) \, x_i x_i^\top$$

Compute $F(\theta)$ at initialization and at several checkpoints during
training. Report:

- Condition number $\kappa(F) = \lambda_{\max} / \lambda_{\min}$
- Trace $\text{tr}(F)$
- Top two eigenvalues

**Step 2 — Compare update directions**

At a fixed $\theta$, compute and compare:

- **SGD step**: $\Delta\theta_{\text{SGD}} = -\eta \nabla_\theta \mathcal{L}$
- **Natural gradient step**: $\Delta\theta_{\text{NG}} = -\eta F(\theta)^{-1} \nabla_\theta \mathcal{L}$

Report the angle between the two update vectors:

$$\cos\alpha = \frac{\Delta\theta_{\text{SGD}} \cdot \Delta\theta_{\text{NG}}}{\|\Delta\theta_{\text{SGD}}\| \, \|\Delta\theta_{\text{NG}}\|}$$

A large deviation ($\alpha$ close to 90°) directly illustrates why Euclidean
gradient descent can be a poor choice on a curved manifold.

**Step 3 — Full training curves**

Train the same model from the same initialization under:

1. SGD with fixed learning rate
2. Natural gradient descent (exact $F^{-1}$)
3. Adam (as a practical baseline)

Record loss and accuracy at each step. Plot convergence curves side by side.

### Expected results and paper connection

Natural gradient descent should converge in fewer steps on ill-conditioned
problems (high $\kappa(F)$). This provides a concrete, reproducible instance
of the paper's core claim: "curvature-aware approaches clarify training
dynamics." Even if the difference is modest on this simple model, the
geometric interpretation is exact and falsifiable.

### Suggested figure

A 2×2 panel: (top-left) loss curves, (top-right) update direction angle
$\alpha$ vs. training step, (bottom-left) Fisher eigenvalue spectrum at
init vs. convergence, (bottom-right) decision boundary comparison.

---

## Experiment 2 — Empirical Fisher / K-FAC summary scalars on a small transformer

### Purpose
Provide an LLM-relevant grounding for the curvature claims. R1 specifically
asks for "an LLM-relevant approximation experiment (small transformer
enough)" with summary scalars correlated with training phase and
generalization.

### Setup

- **Model**: a 2-layer transformer encoder. Suggested config:
  - Vocabulary size: 256 (byte-level) or 1000 (small token vocab)
  - Embedding dim: 64
  - Attention heads: 2
  - FFN hidden dim: 128
  - Sequence length: 32
  - Total parameters: ~500K–1M
- **Dataset**: Penn Treebank (PTB) character-level, or the first 5MB of
  WikiText-2. Use a 90/10 train/validation split.
- **Training**: Adam, 20–50 epochs, batch size 64, on `device = torch.device("mps")`.

### What to compute

**Empirical Fisher (diagonal or block-diagonal approximation)**

At checkpoints $t \in \{0, 10\%, 30\%, 60\%, 100\%\}$ of training:

$$\hat{F}(\theta) \approx \frac{1}{B} \sum_{i \in \mathcal{B}} \nabla_\theta \log p(x_i; \theta) \, \nabla_\theta \log p(x_i; \theta)^\top$$

Computing the full $\hat{F}$ is infeasible. Instead use:

- **Diagonal approximation**: store only the $d$ diagonal entries. Fast and
  sufficient for summary statistics. Fully compatible with MPS tensors.
- **K-FAC block approximation**: use `BackPACK` (check MPS compatibility
  first; if unsupported, fall back to CPU for the Fisher accumulation step
  only and move tensors back to MPS for the forward/backward pass).

**Summary scalars to report** (per checkpoint, per layer):

| Scalar | Interpretation |
|--------|---------------|
| $\text{tr}(\hat{F})$ | Total curvature / gradient signal magnitude |
| $\lambda_{\max}(\hat{F})$ | Sharpest direction in parameter space |
| $\kappa(\hat{F}) = \lambda_{\max} / \lambda_{\min}$ | Condition number proxy (ill-conditioning) |
| $\|\hat{F}\|_F$ | Frobenius norm of curvature |

**Generalization proxy**

Record the train–validation loss gap $\Delta\mathcal{L} = \mathcal{L}_{\text{train}} - \mathcal{L}_{\text{val}}$
at each checkpoint. Test for correlation between $\kappa(\hat{F})$ or
$\lambda_{\max}$ and $\Delta\mathcal{L}$.

### Expected results and paper connection

Prior literature (Martens & Grosse 2015; Pennington & Bahri 2017) suggests
curvature concentrates in early training and flattens as the model
generalizes. If this pattern is reproduced — even qualitatively — it anchors
the paper's claim that "Fisher information is a key player in shaping" the
optimization manifold, with a direct LLM-architecture instance.

### Suggested figure

A 3-panel figure: (left) $\text{tr}(\hat{F})$ and $\lambda_{\max}$ over
training time, (centre) condition number $\kappa$ vs. train step, (right)
scatter of $\kappa$ vs. train–val gap across checkpoints.

---

## Experiment 3 — QFI computation on a parameterized qubit state

### Purpose
Ground the quantum geometry section in at least one explicit calculation,
as R1 requests: "define a parameterized state, compute the Fubini–Study
metric / QFI, and show how the induced update differs from a classical
natural-gradient update on the analogous classical probabilistic model."
This directly answers R2's observation that "there isn't any actual evidence
showing quantum systems provide more efficient optimization paths."

### Setup

Use a single-qubit pure state parameterized by two angles:

$$|\psi(\theta, \phi)\rangle = \cos\!\tfrac{\theta}{2} \, |0\rangle + e^{i\phi} \sin\!\tfrac{\theta}{2} \, |1\rangle$$

This is the Bloch sphere — a well-understood 2D manifold where the
Fubini–Study metric is exactly computable.

**Classical analogue**: a Bernoulli distribution $p(x; \theta)$ with a
single parameter, where the Fisher information is $F = 1/(p(1-p))$.

### What to compute

**Step 1 — Fubini–Study metric tensor**

The metric tensor components for $|\psi(\theta, \phi)\rangle$ are:

$$g_{\theta\theta} = \frac{1}{4}, \quad g_{\phi\phi} = \frac{1}{4}\sin^2\theta, \quad g_{\theta\phi} = 0$$

This gives the line element $ds^2 = \frac{1}{4}(d\theta^2 + \sin^2\theta \, d\phi^2)$,
the standard round metric on $S^2$ scaled by $\frac{1}{4}$.

**Step 2 — Quantum Fisher Information matrix**

For a pure state $|\psi(\theta)\rangle$ with parameter vector
$\boldsymbol{\lambda} = (\theta, \phi)$:

$$[\mathcal{F}_Q]_{jk} = 4 \, \text{Re}\!\left[\langle \partial_j \psi | \partial_k \psi \rangle - \langle \partial_j \psi | \psi \rangle \langle \psi | \partial_k \psi \rangle\right]$$

Compute $\mathcal{F}_Q$ numerically across a grid of $(\theta, \phi)$ values.
Verify it matches $4 \times g_{jk}$ from Step 1.

**Step 3 — Compare quantum natural gradient to classical gradient**

Define a simple loss: the expected value of $\sigma_z$ (Pauli-Z),
$\mathcal{L}(\theta, \phi) = \langle \psi | \sigma_z | \psi \rangle = \cos\theta$.

Compute and compare at several points on the Bloch sphere:

- **Euclidean gradient step**: $\Delta\boldsymbol{\lambda} = -\eta \nabla \mathcal{L}$
- **Quantum natural gradient step**: $\Delta\boldsymbol{\lambda} = -\eta \mathcal{F}_Q^{-1} \nabla \mathcal{L}$
- **Classical Fisher step** (Bernoulli analogue): $\Delta\theta = -\eta F^{-1} \partial_\theta \mathcal{L}$

Report the angular deviation between the Euclidean and quantum natural
gradient updates at each point. Show that near the poles ($\theta \approx 0$
or $\pi$), where $g_{\phi\phi} \to 0$ (the manifold pinches), the
corrections are largest.

**Step 4 — Optimization trajectory comparison**

Starting from $(\theta_0, \phi_0) = (0.5, 0.5)$, run 50 steps of each
optimizer toward the minimum at $\theta = \pi$. Plot the trajectory on the
Bloch sphere. The quantum natural gradient trajectory should follow the
geodesic more closely.

### Tooling

PennyLane is the primary library for this experiment. Use the `"default.qubit"`
device, which runs on CPU via NumPy and is unaffected by MPS. The MPS
backend is PyTorch-specific and does not apply to PennyLane's qubit
simulators.

```python
import pennylane as qml
import pennylane.numpy as pnp
import numpy as np
import matplotlib.pyplot as plt

dev = qml.device("default.qubit", wires=1)
```

Use `qml.qinfo.quantum_fisher` to compute $\mathcal{F}_Q$ and verify it
against the analytic Fubini–Study result. Use `qml.QNGOptimizer` for the
quantum natural gradient trajectory in Step 4, which directly implements
$\Delta\boldsymbol{\lambda} = -\eta \mathcal{F}_Q^{-1} \nabla \mathcal{L}$.

### Expected results and paper connection

The key result is not that the quantum optimizer is "better" — it is that
the update direction is demonstrably different, and that this difference is
the direct consequence of the non-Euclidean geometry induced by the
Fubini–Study metric. This transforms the paper's central quantum analogy
from metaphor to a computed, falsifiable instance.

---

## Recommended paper integration

### New subsection: Section 2.6 — Toy worked examples

Place all three experiments in a new subsection immediately following the
existing theoretical background, before the Related Work section. Structure
as:

1. **2.6.1** Logistic regression: exact Fisher and natural gradient (Experiment 1)
2. **2.6.2** Small transformer: empirical curvature scalars (Experiment 2)
3. **2.6.3** Qubit state: Fubini–Study metric and quantum natural gradient (Experiment 3)

Each sub-subsection should be brief (half a page): state the model, report
the key scalar results in a small table or figure, and close with one
sentence connecting the result back to the paper's claim.

### Claim recalibration

After adding the experiments, revisit the following passages and update
language from speculative implication to grounded hypothesis:

| Section | Current phrasing | Suggested revision |
|---------|-----------------|-------------------|
| §2.4 | "optimization over such a manifold may follow steeper, more directed gradients" | "as shown in §2.6.3, the quantum natural gradient update deviates from the Euclidean direction by X° at …" |
| §4.2 | "it is conceivable that quantum-enhanced models may deviate from classical scaling trends" | "this remains an open hypothesis; a necessary first step would be…" |
| §4.1 | "quantum systems bake in geometry" | cite §2.6.3 result as supporting instance |

---

## Environment and reproducibility checklist

- [ ] Python 3.10+, NumPy ≥ 1.24, PyTorch ≥ 2.1, Matplotlib ≥ 3.7
- [ ] PennyLane ≥ 0.38 (required for Experiment 3: `qml.qinfo.quantum_fisher`, `qml.QNGOptimizer`)
- [ ] BackPACK ≥ 1.6 (optional for K-FAC in Experiment 2; verify MPS support at runtime)
- [ ] PyTorch MPS backend enabled: `torch.backends.mps.is_available()` must return `True`
- [ ] Fix all random seeds: `np.random.seed(42)`, `torch.manual_seed(42)`
- [ ] Hardware to report: Apple Silicon MacBook Pro, MPS accelerator, RAM size
- [ ] PennyLane qubit simulations run on `"default.qubit"` (CPU); note this explicitly in the paper
- [ ] Release code as a supplementary Jupyter notebook
- [ ] All figures: 300 dpi, axis labels in LaTeX math mode, colorblind-safe palette
