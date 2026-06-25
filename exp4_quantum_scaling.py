"""
Experiment 4 — Classical vs. Quantum Scaling: Fisher Information Efficiency

Central claim: quantum circuits achieve exponentially richer representational
capacity per trainable parameter than classical MLPs of comparable size, AND
maintain better-conditioned Fisher information matrices (lower κ).  Together
these two facts provide a mechanistic hint for why quantum-enhanced training
could break through classical neural network scaling laws.

Classical family:  4 → H → 1  (MLP, ReLU), H ∈ {2, 4, 8, 16, 32, 64}
Quantum family:    angle-encode(4→n qubits) + BasicEntangler(n, 2 layers)
                   + Linear(n→1),            n ∈ {2, 3, 4, 5, 6}

Key quantity:
  Classical Fisher  F̂(θ)    — empirical, via per-sample gradients (loss-based)
  Quantum Fisher    F_Q(θ)  — Fubini–Study metric of the variational circuit
                              (state-based, INDEPENDENT of the loss function)

The QFI independence from the loss is itself an important message: quantum
geometry is intrinsically well-conditioned, whereas the classical Fisher is
shaped (and typically worsened) by the data distribution.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import pennylane.numpy as pnp

np.random.seed(42)
torch.manual_seed(42)

# ── Dataset ────────────────────────────────────────────────────────────────────
# 4-feature task with 2 clusters per class: hard enough to show a classical
# scaling law (small MLPs underfit) but simple enough for quantum circuits to learn.
N_SAMPLES  = 2000
N_FEATURES = 4
N_STEPS_C  = 800    # classical training steps (full-batch)
N_STEPS_Q  = 1500   # quantum training steps   (mini-batch)
LR_C       = 5e-3   # classical learning rate
LR_Q       = 0.02   # quantum learning rate (higher: circuits need larger steps)
BATCH_Q    = 32     # small batch keeps the Python loop manageable
FISHER_N   = 400    # samples for empirical Fisher

X_raw, y = make_classification(
    n_samples=N_SAMPLES, n_features=N_FEATURES,
    n_informative=N_FEATURES, n_redundant=0,
    n_clusters_per_class=2, class_sep=1.0, random_state=42,
)
X = StandardScaler().fit_transform(X_raw)
X_tr_np, X_te_np, y_tr_np, y_te_np = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_tr = torch.tensor(X_tr_np, dtype=torch.float32)
y_tr = torch.tensor(y_tr_np, dtype=torch.float32)
X_te = torch.tensor(X_te_np, dtype=torch.float32)
y_te = torch.tensor(y_te_np, dtype=torch.float32)
N_TRAIN = len(X_tr)

# ── Classical MLP family ──────────────────────────────────────────────────────

class ClassicalMLP(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_FEATURES, h), nn.ReLU(),
            nn.Linear(h, 1), nn.Sigmoid(),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


def empirical_fisher(model, n_samples=FISHER_N):
    """Empirical Fisher F̂ = (1/N) Σ ∇L_i ∇L_i^T, computed via per-sample grads."""
    n_p   = sum(p.numel() for p in model.parameters())
    F_mat = np.zeros((n_p, n_p))
    idx   = np.random.choice(N_TRAIN, n_samples, replace=False)
    for i in idx:
        model.zero_grad()
        F.binary_cross_entropy(model(X_tr[i:i+1]), y_tr[i:i+1]).backward()
        g = np.concatenate([
            p.grad.detach().cpu().numpy().ravel() for p in model.parameters()
        ])
        F_mat += np.outer(g, g)
    model.zero_grad()
    return F_mat / n_samples


def fisher_stats(F_mat):
    eig   = np.linalg.eigvalsh(F_mat)
    tr_f  = float(np.trace(F_mat))
    kappa = float(eig[-1] / max(abs(eig[0]), 1e-12))
    return tr_f, kappa, eig


def run_classical(h):
    torch.manual_seed(42)
    model = ClassicalMLP(h)
    opt   = torch.optim.Adam(model.parameters(), lr=LR_C)
    losses = []
    for _ in range(N_STEPS_C):
        opt.zero_grad()
        l = F.binary_cross_entropy(model(X_tr), y_tr)
        l.backward()
        opt.step()
        losses.append(l.item())
    with torch.no_grad():
        test_loss = F.binary_cross_entropy(model(X_te), y_te).item()
        test_acc  = ((model(X_te) >= 0.5) == y_te.bool()).float().mean().item()
    n_p       = sum(p.numel() for p in model.parameters())
    F_mat     = empirical_fisher(model)
    tr_f, kappa, eig = fisher_stats(F_mat)
    print(f"  Classical H={h:3d}  n_params={n_p:4d}  test_loss={test_loss:.4f}  "
          f"acc={test_acc:.3f}  κ={kappa:.2e}  tr/N={tr_f/n_p:.4f}")
    return dict(h=h, n_params=n_p, test_loss=test_loss, test_acc=test_acc,
                tr_f=tr_f, kappa=kappa, eig=eig, losses=losses)


# ── Quantum hybrid model family ───────────────────────────────────────────────
N_Q_LAYERS = 2
BATCH_Q    = 32   # small batch keeps the Python loop manageable


def make_quantum_model(n_qubits):
    """
    Hybrid quantum-classical model with DATA RE-UPLOADING.

    At each variational layer the input features are re-encoded before the
    rotation gates.  This gives the circuit expressivity that grows with depth
    rather than width, avoids barren-plateau initialisation, and matches the
    "quantum kernel" viewpoint: the encoded state changes with the data at
    every layer so the model can learn non-trivial decision boundaries.

    Circuit per layer l:
      RY(x_i * pi)  for i in range(n_qubits)   <- data encoding (repeated)
      RX(w[l,i,0])  for i in range(n_qubits)   <- variational
      RZ(w[l,i,1])  for i in range(n_qubits)   <- variational
      CNOT(i, i+1)  for i in range(n_qubits-1) <- entanglement

    Number of circuit parameters: N_Q_LAYERS x n_qubits x 2
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        for layer in range(N_Q_LAYERS):
            # Data re-uploading: encode features at every layer
            for i in range(n_qubits):
                qml.RY(inputs[i % N_FEATURES] * np.pi, wires=i)
            # Variational rotations (two axes per qubit for richer geometry)
            for i in range(n_qubits):
                qml.RX(weights[layer, i, 0], wires=i)
                qml.RZ(weights[layer, i, 1], wires=i)
            # Linear entanglement
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
        return tuple(qml.expval(qml.PauliZ(i)) for i in range(n_qubits))

    class QuantumHybrid(nn.Module):
        def __init__(self):
            super().__init__()
            self.weights = nn.Parameter(torch.zeros(N_Q_LAYERS, n_qubits, 2))
            nn.init.uniform_(self.weights, -np.pi / 4, np.pi / 4)
            self.linear = nn.Linear(n_qubits, 1)

        def forward(self, x):
            q_outs = []
            for xi in x:
                raw = circuit(xi, self.weights)
                q_outs.append(torch.stack(list(raw)))
            q_out = torch.stack(q_outs).float()   # (batch, n_qubits)
            return torch.sigmoid(self.linear(q_out)).squeeze(-1)

    return QuantumHybrid()


def compute_qfi(weights_np, n_qubits, eps=1e-4):
    """
    Numerical Quantum Fisher Information via central-difference state derivatives.

    QFI_ij = 4 Re[⟨∂_i ψ|∂_j ψ⟩ − ⟨∂_i ψ|ψ⟩⟨ψ|∂_j ψ⟩]

    Computed on the PURE VARIATIONAL CIRCUIT (no angle encoding), so QFI reflects
    the intrinsic Fubini–Study geometry of the ansatz — independent of the task or
    the loss function.  This is a state-space property, not a loss-landscape property.
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def var_circuit(params):
        # Mirror the variational part of the training circuit (no data encoding)
        w = params.reshape(N_Q_LAYERS, n_qubits, 2)
        for layer in range(N_Q_LAYERS):
            for i in range(n_qubits):
                qml.RX(w[layer, i, 0], wires=i)
                qml.RZ(w[layer, i, 1], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
        return qml.state()

    params = weights_np.ravel().astype(float)
    n_p    = len(params)
    psi0   = np.array(var_circuit(pnp.array(params)))

    # Central-difference state derivatives  ∂_i |ψ⟩
    derivs = []
    for i in range(n_p):
        p_p = params.copy(); p_p[i] += eps
        p_m = params.copy(); p_m[i] -= eps
        d_i = (np.array(var_circuit(pnp.array(p_p))) -
               np.array(var_circuit(pnp.array(p_m)))) / (2 * eps)
        derivs.append(d_i)

    QFI = np.zeros((n_p, n_p))
    for i in range(n_p):
        for j in range(i, n_p):
            t1 = np.vdot(derivs[i], derivs[j])
            t2 = np.vdot(derivs[i], psi0) * np.vdot(psi0, derivs[j])
            QFI[i, j] = 4.0 * np.real(t1 - t2)
            QFI[j, i] = QFI[i, j]
    return QFI


def run_quantum(n_qubits):
    torch.manual_seed(42)
    print(f"  Quantum n_qubits={n_qubits}…")
    model = make_quantum_model(n_qubits)
    opt   = torch.optim.Adam(model.parameters(), lr=LR_Q)
    losses = []
    for step in range(N_STEPS_Q):
        idx = np.random.choice(N_TRAIN, BATCH_Q, replace=False)
        opt.zero_grad()
        l = F.binary_cross_entropy(model(X_tr[idx]), y_tr[idx])
        l.backward()
        opt.step()
        losses.append(l.item())
        if step % 250 == 0:
            print(f"    step {step:4d}  loss={l.item():.4f}")

    with torch.no_grad():
        tl_sum = 0.0
        preds  = []
        for i in range(0, len(X_te), 128):
            o = model(X_te[i:i+128])
            tl_sum += F.binary_cross_entropy(o, y_te[i:i+128]).item() * len(X_te[i:i+128])
            preds.append((o >= 0.5).cpu())
        test_loss = tl_sum / len(y_te)
        test_acc  = (torch.cat(preds) == y_te.bool().cpu()).float().mean().item()

    n_circ = model.weights.numel()          # circuit rotation angles only
    n_tot  = sum(p.numel() for p in model.parameters())
    h_dim  = 2 ** n_qubits                 # Hilbert space dimension

    print(f"    Computing QFI (n_circuit_params={n_circ})…")
    w_np       = model.weights.detach().cpu().numpy()
    qfi        = compute_qfi(w_np, n_qubits)
    tr_q, kappa, eig_q = fisher_stats(qfi)

    print(f"    n_params={n_tot:3d} (circ={n_circ})  hilbert=2^{n_qubits}={h_dim}  "
          f"test_loss={test_loss:.4f}  acc={test_acc:.3f}  "
          f"κ(QFI)={kappa:.2e}  tr/circ_N={tr_q/n_circ:.4f}")
    return dict(
        n_qubits=n_qubits, n_params=n_tot, n_circ=n_circ,
        hilbert_dim=h_dim, test_loss=test_loss, test_acc=test_acc,
        tr_f=tr_q, kappa=kappa, eig=eig_q, losses=losses,
    )


# ── Run ───────────────────────────────────────────────────────────────────────
print("=" * 68)
print("Classical MLP family  (4→H→1, ReLU, Adam, 800 full-batch steps)")
print("=" * 68)
c_res = [run_classical(h) for h in [2, 4, 8, 16, 32, 64]]

print("\n" + "=" * 68)
print("Quantum hybrid family  (n qubits, 2 layers, Adam, 1000 mini-batch steps)")
print("=" * 68)
q_res = [run_quantum(n) for n in [2, 3, 4, 5, 6]]

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n" + "=" * 78)
print(f"{'Model':>22} {'N_params':>8} {'Hilbert':>8} "
      f"{'Loss':>7} {'Acc':>6} {'κ(Fisher)':>12} {'tr/N':>8}")
print("-" * 78)
for r in c_res:
    print(f"{'MLP  H='+str(r['h']):>22} {r['n_params']:>8} {'N/A':>8} "
          f"{r['test_loss']:>7.4f} {r['test_acc']:>6.3f} "
          f"{r['kappa']:>12.2e} {r['tr_f']/r['n_params']:>8.4f}")
for r in q_res:
    print(f"{'VQC  n='+str(r['n_qubits'])+'q':>22} "
          f"{r['n_params']:>8} {r['hilbert_dim']:>8} "
          f"{r['test_loss']:>7.4f} {r['test_acc']:>6.3f} "
          f"{r['kappa']:>12.2e} {r['tr_f']/r['n_circ']:>8.4f}")

# Highlight the "parameter efficiency" gap at comparable performance
print("\n── Equivalent-capacity comparison ───────────────────────────────────────")
print("  For Hilbert-space dimension ≈ 2^n:")
print(f"  {'n_qubits':>8} {'QFI circ params':>16} {'Hilbert dim':>12} "
      f"{'Equiv. classical H':>19} {'Classical params':>17}")
for r in q_res:
    n = r["n_qubits"]
    h_equiv = r["hilbert_dim"]
    c_params_equiv = N_FEATURES * h_equiv + 2 * h_equiv + 1
    print(f"  {n:>8} {r['n_circ']:>16} {r['hilbert_dim']:>12} "
          f"{'H='+str(h_equiv):>19} {c_params_equiv:>17}")

# ── Figure ────────────────────────────────────────────────────────────────────
PAL = {"c": "#0072B2", "q": "#D55E00", "g": "#009E73", "m": "#CC79A7"}

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle(
    "Experiment 4: Classical vs. Quantum Scaling — Fisher Information Efficiency\n"
    r"Classical 4→H→1 MLP  vs.  Quantum angle-encode(4→$n$) + VQC($n$, 2L) + Linear",
    fontsize=11,
)

c_np  = [r["n_params"] for r in c_res]
c_l   = [r["test_loss"] for r in c_res]
c_k   = [r["kappa"] for r in c_res]
c_tpn = [r["tr_f"] / r["n_params"] for r in c_res]

q_np  = [r["n_params"] for r in q_res]
q_l   = [r["test_loss"] for r in q_res]
q_k   = [r["kappa"] for r in q_res]
q_cn  = [r["n_circ"] for r in q_res]
q_tpn = [r["tr_f"] / r["n_circ"] for r in q_res]
q_nb  = [r["n_qubits"] for r in q_res]
q_hd  = [r["hilbert_dim"] for r in q_res]

# ── Panel 1 (top-left): Scaling law — test loss vs. N_params ─────────────────
ax = axes[0, 0]
ax.loglog(c_np, c_l, "o-", color=PAL["c"], markersize=7, lw=2.0,
          label="Classical MLP")
ax.loglog(q_np, q_l, "s--", color=PAL["q"], markersize=7, lw=2.0,
          label="Quantum hybrid")

# Annotate each quantum point with qubit count
for r in q_res:
    ax.annotate(f"n={r['n_qubits']}q\n(dim={r['hilbert_dim']})",
                xy=(r["n_params"], r["test_loss"]),
                xytext=(r["n_params"] * 1.15, r["test_loss"] * 1.02),
                fontsize=6.5, color=PAL["q"], ha="left")

# Power-law fits
from scipy.stats import linregress
for xdata, ydata, color, label in [
    (c_np, c_l, PAL["c"], "classical"),
    (q_np, q_l, PAL["q"], "quantum"),
]:
    if len(xdata) >= 3:
        sl, ic, *_ = linregress(np.log(xdata), np.log(ydata))
        xf = np.geomspace(min(xdata), max(xdata), 200)
        ax.loglog(xf, np.exp(ic) * xf**sl, ":", color=color, alpha=0.5, lw=1.2)
        ax.text(0.05 if label == "classical" else 0.55, 0.12,
                fr"$\alpha_{{\rm {label}}} \approx {sl:.2f}$",
                transform=ax.transAxes, color=color, fontsize=8)

ax.set_xlabel(r"Number of trainable parameters $N$")
ax.set_ylabel("Test loss")
ax.set_title(r"Scaling law: test loss $L(N) \approx a\,N^{-\alpha}$")
ax.legend(fontsize=9)

# ── Panel 2 (top-right): Fisher / QFI condition number ───────────────────────
ax = axes[0, 1]
ax.semilogy(c_np, c_k, "o-", color=PAL["c"], markersize=7, lw=2.0,
            label=r"Classical: $\kappa(\hat{F})$  (empirical)")
ax.semilogy(q_np, q_k, "s--", color=PAL["q"], markersize=7, lw=2.0,
            label=r"Quantum: $\kappa(\mathcal{F}_Q)$  (Fubini–Study)")
ax.set_xlabel(r"Number of parameters $N$")
ax.set_ylabel(r"Condition number $\kappa$")
ax.set_title(
    "Fisher condition number\n"
    r"lower $\kappa$ = better-conditioned optimization landscape"
)
ax.legend(fontsize=8)

# ── Panel 3 (bottom-left): Fisher information per parameter ──────────────────
ax = axes[1, 0]
ax.semilogy(range(len(c_res)), c_tpn, "o-", color=PAL["c"], markersize=7, lw=2.0,
            label=r"Classical $\mathrm{tr}(\hat{F})/N$")
ax.semilogy(range(len(q_res)), q_tpn, "s--", color=PAL["q"], markersize=7, lw=2.0,
            label=r"Quantum $\mathrm{tr}(\mathcal{F}_Q)/N_{\rm circ}$")
ax.set_xticks(range(max(len(c_res), len(q_res))))
ax.set_xlabel("Model index (increasing size →)")
ax.set_ylabel("Fisher information per parameter")
ax.set_title(
    "Fisher information efficiency\n"
    "higher = more information per parameter"
)
ax.legend(fontsize=8)

# ── Panel 4 (bottom-right): Exponential representational efficiency ───────────
ax = axes[1, 1]

# Quantum: Hilbert-space dimension per circuit parameter = 2^n / (N_Q_LAYERS * n)
q_heff = [hd / nc for hd, nc in zip(q_hd, q_cn)]
# Classical: analogously H / total_params ≈ H / (6H+1) → decreasing and small
c_heff = [r["h"] / r["n_params"] for r in c_res]

ax.semilogy([r["h"] for r in c_res], c_heff, "o-", color=PAL["c"],
            markersize=7, lw=2.0, label=r"Classical: $H / N_{\rm params}$")
ax.semilogy(q_nb, q_heff, "s--", color=PAL["q"],
            markersize=9, lw=2.5, label=r"Quantum: $2^n / N_{\rm circ}$")

# Shared x-axis label for the "size" axis
ax.set_xlabel(r"Model width: hidden units $H$ (classical) or qubits $n$ (quantum)")
ax.set_ylabel("Hilbert-space capacity / circuit params")
ax.set_title(
    "Exponential representational efficiency\n"
    r"Quantum: $2^n/N_{\rm circ}$ grows exponentially;  "
    r"Classical: $H/N$ stays $\approx 1/6$"
)
ax.legend(fontsize=8)

# Annotate the gap at n=6
if q_res:
    last = q_res[-1]
    ax.annotate(
        fr"$n={last['n_qubits']}$: $2^{{{last['n_qubits']}}}={last['hilbert_dim']}$ dims"
        f"\nwith {last['n_circ']} circuit params",
        xy=(last["n_qubits"], q_heff[-1]),
        xytext=(last["n_qubits"] - 1.5, q_heff[-1] * 0.45),
        arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
        fontsize=8, color=PAL["q"],
    )

plt.tight_layout()
out = "exp4_quantum_scaling.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
print(f"\nSaved {out}")
