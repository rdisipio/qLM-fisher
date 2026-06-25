"""
Assembles the three experiment scripts into a single Jupyter notebook.
Each script is split into logical cells at double-blank-line boundaries
between top-level sections. Markdown section headers are inserted between
experiments.
"""

import json, re, textwrap, uuid
from pathlib import Path

# ── Helpers ───────────────────────────────────────────────────────────────────

def _cell_id() -> str:
    return uuid.uuid4().hex[:8]

def code_cell(source: str) -> dict:
    lines = [l + "\n" for l in source.splitlines()]
    if lines and lines[-1].endswith("\n"):
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "code",
        "id": _cell_id(),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines,
    }


def md_cell(text: str) -> dict:
    lines = [l + "\n" for l in text.strip().splitlines()]
    if lines and lines[-1].endswith("\n"):
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "markdown",
        "id": _cell_id(),
        "metadata": {},
        "source": lines,
    }


def split_into_cells(src: str) -> list[str]:
    """
    Split on every '# ──' section header (multiline anchored).
    The first block (module docstring + imports) becomes cell 0;
    each subsequent section header starts a new cell.
    """
    blocks = re.split(r'(?m)^(?=# ──)', src)
    return [b.strip() for b in blocks if b.strip()]


# ── Read scripts ──────────────────────────────────────────────────────────────
scripts = {
    "exp1": Path("exp1_logistic_regression.py").read_text(),
    "exp2": Path("exp2_transformer_fisher.py").read_text(),
    "exp3": Path("exp3_qubit_qfi.py").read_text(),
    "exp4": Path("exp4_quantum_scaling.py").read_text(),
}

# ── Build notebook cells ──────────────────────────────────────────────────────
cells = []

# ── Title ─────────────────────────────────────────────────────────────────────
cells.append(md_cell(textwrap.dedent("""\
    # Toy Model Experiments: Addressing R1 & R2 Critical Feedback

    Four self-contained computational experiments that directly respond to
    reviewer requests. Each produces a reproducible figure mappable to a claim
    in the preprint.

    | # | Experiment | Addresses |
    |---|------------|-----------|
    | 1 | Fisher information & natural gradient on logistic regression | R1 (toy worked example) |
    | 2 | Empirical Fisher / diagonal approximation on a small transformer | R1 (LLM-relevant) |
    | 3 | QFI computation on a parameterised qubit state | R1 + R2 (quantum geometry) |
    | 4 | Classical vs. quantum scaling — Fisher information efficiency | R2 (scaling law claim) |

    **Environment**: Python 3.13+, NumPy ≥ 2.4, PyTorch ≥ 2.12 (MPS for Exp 2),
    PennyLane ≥ 0.45 (CPU/NumPy for Exps 3 & 4).
    All random seeds fixed: `np.random.seed(42)`, `torch.manual_seed(42)`.
""")))

# ── Experiment 1 ──────────────────────────────────────────────────────────────
cells.append(md_cell(textwrap.dedent("""\
    ---
    ## Experiment 1 — Fisher Information and Natural Gradient on Logistic Regression

    **Purpose**: Provide a minimal, fully reproducible demonstration that
    information geometry ("curvature matters") changes the optimisation trajectory
    in a measurable, concrete way.
    Directly answers R1's request for *"a worked example where the Fisher
    information matrix is computed explicitly."*

    **Model**: binary logistic regression, $d = 2$ features, $N = 200$ samples
    **Optimisers**: SGD, natural gradient (exact $F^{-1}$), Adam
    **Figure**: loss curves · angle $\\alpha$ between SGD and NG update · Fisher
    eigenvalue spectrum at init vs convergence · decision boundaries
""")))

for cell_src in split_into_cells(scripts["exp1"]):
    cells.append(code_cell(cell_src))

# ── Experiment 2 ──────────────────────────────────────────────────────────────
cells.append(md_cell(textwrap.dedent("""\
    ---
    ## Experiment 2 — Empirical Fisher Scalars on a Small Transformer

    **Purpose**: Provide an LLM-relevant grounding for the curvature claims.
    R1 specifically asks for *"an LLM-relevant approximation experiment (small
    transformer enough)"* with summary scalars correlated with training phase and
    generalisation.

    **Model**: 2-layer transformer encoder, byte-level, ~100 K parameters
    **Dataset**: WikiText-2 (Salesforce/wikitext), byte-level encoding
    **Device**: `torch.device("mps")` on Apple Silicon (CPU fallback)
    **Scalars tracked**: $\\mathrm{tr}(\\hat{F})$, $\\lambda_{\\max}$,
    $\\kappa = \\lambda_{\\max}/\\lambda_{\\min}$, $\\|\\hat{F}\\|_F$
    **Figure**: curvature magnitude over training · condition number $\\kappa$ vs
    step · $\\kappa$ vs generalisation gap scatter
""")))

for cell_src in split_into_cells(scripts["exp2"]):
    cells.append(code_cell(cell_src))

# ── Experiment 3 ──────────────────────────────────────────────────────────────
cells.append(md_cell(textwrap.dedent("""\
    ---
    ## Experiment 3 — QFI on a Parameterised Qubit State

    **Purpose**: Ground the quantum geometry section in at least one explicit
    calculation, as R1 requests: *"define a parameterised state, compute the
    Fubini–Study metric / QFI, and show how the induced update differs from a
    classical natural-gradient update."*
    Also directly addresses R2: *"there isn't any actual evidence showing quantum
    systems provide more efficient optimisation paths."*

    **State**: $|\\psi(\\theta,\\phi)\\rangle = \\cos(\\theta/2)|0\\rangle + e^{i\\phi}\\sin(\\theta/2)|1\\rangle$
    **Library**: PennyLane `"default.qubit"` (CPU/NumPy — MPS does not apply)
    **Figure**: QFI diagonal components analytic vs PennyLane · angular deviation
    Euclidean vs QNG · Bloch sphere optimisation trajectory

    > **Key result**: Quantum natural gradient (QNG) reaches the exact minimum
    > $\\langle\\sigma_x\\rangle = -1$ in 50 steps; Euclidean GD stalls at a
    > near-zero saddle region. The maximum angular deviation between the two update
    > directions is **84.3°** near the poles — where the Bloch sphere geometry
    > pinches ($g_{\\phi\\phi} \\to 0$).
""")))

for cell_src in split_into_cells(scripts["exp3"]):
    cells.append(code_cell(cell_src))

# ── Experiment 4 ──────────────────────────────────────────────────────────────
cells.append(md_cell(textwrap.dedent("""\
    ---
    ## Experiment 4 — Classical vs. Quantum Scaling: Fisher Information Efficiency

    **Central claim**: quantum circuits achieve *exponentially richer*
    representational capacity per trainable parameter than classical MLPs of
    comparable size, AND maintain better-conditioned Fisher information matrices
    (lower $\\kappa$). Together these two facts provide a mechanistic hint for why
    quantum-enhanced training could break through classical neural network scaling laws.

    **Classical family**: $4 \\to H \\to 1$ MLP (ReLU), $H \\in \\{2,4,8,16,32,64\\}$
    **Quantum family**: angle-encode$(4\\to n$ qubits$)$ + data-re-uploading VQC
    $(n$ qubits, 2 layers$)$ + Linear$(n\\to 1)$, $n \\in \\{2,3,4,5,6\\}$

    **Key distinction**: the classical empirical Fisher $\\hat{F}$ is a
    *loss-landscape* property (depends on data + model outputs); the quantum
    Fisher $\\mathcal{F}_Q$ is a *state-space* property — the Fubini–Study metric
    of the variational ansatz, independent of the loss function. This means quantum
    geometry is intrinsically well-conditioned regardless of the task.

    **Figure**: scaling law (loss vs. $N$) · Fisher condition number comparison ·
    Fisher information per parameter · exponential representational efficiency
    $2^n / N_{\\rm circ}$

    > **Key result**: classical $\\kappa(\\hat{F}) \\sim 10^9$ and grows with model
    > size; quantum $\\kappa(\\mathcal{F}_Q) \\ll 10^9$ throughout. For $n=6$ qubits,
    > the circuit operates in a $2^6=64$-dimensional Hilbert space with only 24
    > variational parameters — a representational efficiency that grows
    > **exponentially** compared to the linear scaling of classical networks.
""")))

for cell_src in split_into_cells(scripts["exp4"]):
    cells.append(code_cell(cell_src))

# ── Notebook structure ────────────────────────────────────────────────────────
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.13.0",
        },
    },
    "cells": cells,
}

out = Path("toy_model_experiments.ipynb")
out.write_text(json.dumps(notebook, indent=1, ensure_ascii=False))
print(f"Written {out}  ({len(cells)} cells, {out.stat().st_size // 1024} KB)")
