"""
Experiment 4 (unified) — Classical vs. Quantum: Scaling, Architecture, Optimization

Scenarios
---------
  A. Classical MLP family       2→H→1, H ∈ {2,4,8,16,32,64}
  B. QCNN + Adam                brick-wall local 2-qubit blocks, data re-uploading
  C. QCNN + QNG_pb              pullback metric G_eff = (1/B)Σ p_i(1-p_i)(WJ_i)ᵀ(WJ_i)

QCNN (Pesah et al. 2021) is used throughout: local gate structure prevents the
circuit from forming a global 2-design, avoiding barren plateaus by construction.
The pullback metric (C) corrects for the classical readout layer that QNG_FS ignores.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import linregress
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import pennylane.numpy as pnp

np.random.seed(42)
torch.manual_seed(42)

# ── Hyperparameters ────────────────────────────────────────────────────────────
N_SAMPLES  = 2000
N_FEATURES = 2
N_Q_LAYERS = 3
N_STEPS_C  = 800
N_STEPS_Q  = 2000
LR_C       = 5e-3
LR_Q       = 0.01
BATCH_Q    = 32
FISHER_N   = 400

# ── Dataset ────────────────────────────────────────────────────────────────────
X_raw, y = make_moons(n_samples=N_SAMPLES, noise=0.25, random_state=42)
X = StandardScaler().fit_transform(X_raw)
X_tr_np, X_te_np, y_tr_np, y_te_np = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_tr = torch.tensor(X_tr_np, dtype=torch.float32)
y_tr = torch.tensor(y_tr_np, dtype=torch.float32)
X_te = torch.tensor(X_te_np, dtype=torch.float32)
y_te = torch.tensor(y_te_np, dtype=torch.float32)
N_TRAIN = len(X_tr)

PAL = {"c": "#0072B2", "adam": "#D55E00", "pb": "#009E73"}


# ══════════════════════════════════════════════════════════════════════════════
# A. Classical MLP
# ══════════════════════════════════════════════════════════════════════════════

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
    n_p   = sum(p.numel() for p in model.parameters())
    F_mat = np.zeros((n_p, n_p))
    idx   = np.random.choice(N_TRAIN, n_samples, replace=False)
    for i in idx:
        model.zero_grad()
        F.binary_cross_entropy(model(X_tr[i:i+1]), y_tr[i:i+1]).backward()
        g = np.concatenate([p.grad.detach().cpu().numpy().ravel()
                            for p in model.parameters()])
        F_mat += np.outer(g, g)
    model.zero_grad()
    return F_mat / n_samples


def fisher_stats(F_mat):
    eig   = np.linalg.eigvalsh(F_mat)
    tr_f  = float(np.trace(F_mat))
    kappa = float(eig[-1] / max(abs(eig[0]), 1e-12))
    return tr_f, kappa


def run_classical(h):
    torch.manual_seed(42)
    model = ClassicalMLP(h)
    opt   = torch.optim.Adam(model.parameters(), lr=LR_C)
    losses = []
    for _ in range(N_STEPS_C):
        opt.zero_grad()
        l = F.binary_cross_entropy(model(X_tr), y_tr)
        l.backward(); opt.step()
        losses.append(l.item())
    with torch.no_grad():
        test_loss = F.binary_cross_entropy(model(X_te), y_te).item()
        test_acc  = ((model(X_te) >= 0.5) == y_te.bool()).float().mean().item()
    n_p        = sum(p.numel() for p in model.parameters())
    tr_f, kappa = fisher_stats(empirical_fisher(model))
    print(f"  MLP H={h:3d}  n_params={n_p:4d}  loss={test_loss:.2f}  "
          f"acc={test_acc:.0%}  tr/N={tr_f/n_p:.3f}", flush=True)
    return dict(h=h, n_params=n_p, test_loss=test_loss, test_acc=test_acc,
                tr_f=tr_f, kappa=kappa, losses=losses)


# ══════════════════════════════════════════════════════════════════════════════
# B/C. QCNN model factory
# ══════════════════════════════════════════════════════════════════════════════

def _qcnn_layout(n_qubits):
    n_even   = n_qubits // 2
    n_odd    = (n_qubits - 1) // 2
    n_blocks = n_even + n_odd
    all_pairs = ([(2*k, 2*k+1) for k in range(n_even)] +
                 [(2*k+1, 2*k+2) for k in range(n_odd)])
    return n_blocks, all_pairs


def make_qcnn(n_qubits):
    """
    Brick-wall QCNN: alternating even/odd local 2-qubit blocks, data re-uploading.

    Block on wires (i, j):  RY(α, i) — CNOT(i→j) — RY(β, i) — RY(γ, j)   [3 params]

    Local gate structure prevents 2-design formation → no barren plateaus
    (Pesah et al. 2021; Cerezo et al. 2021).
    """
    n_blocks, all_pairs = _qcnn_layout(n_qubits)
    N_BP      = 3
    n_readout = n_qubits + n_qubits * (n_qubits - 1) // 2
    dev       = qml.device("lightning.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="adjoint")
    def circuit(inputs, weights):
        for layer in range(N_Q_LAYERS):
            for i in range(n_qubits):
                qml.RY(inputs[i % N_FEATURES] * np.pi / 4, wires=i)
            for b, (i, j) in enumerate(all_pairs):
                qml.RY(weights[layer, b, 0], wires=i)
                qml.CNOT(wires=[i, j])
                qml.RY(weights[layer, b, 1], wires=i)
                qml.RY(weights[layer, b, 2], wires=j)
        singles = [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        pairs   = [qml.expval(qml.PauliZ(i) @ qml.PauliZ(j))
                   for i in range(n_qubits) for j in range(i + 1, n_qubits)]
        return tuple(singles + pairs)

    class QCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.weights = nn.Parameter(torch.zeros(N_Q_LAYERS, n_blocks, N_BP))
            nn.init.uniform_(self.weights, -np.pi / 4, np.pi / 4)
            self.linear  = nn.Linear(n_readout, 1)
        def forward(self, x):
            q_outs = []
            for xi in x:
                raw = circuit(xi, self.weights)
                q_outs.append(torch.stack(list(raw)))
            return torch.sigmoid(self.linear(torch.stack(q_outs).float())).squeeze(-1)

    return QCNN()


def compute_qfi_qcnn(weights_np, n_qubits, eps=1e-4):
    """QFI of the QCNN variational circuit via central-difference state derivatives."""
    n_blocks, all_pairs = _qcnn_layout(n_qubits)
    dev = qml.device("lightning.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def var_circuit(params):
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
        w = params.reshape(N_Q_LAYERS, n_blocks, 3)
        for layer in range(N_Q_LAYERS):
            for b, (i, j) in enumerate(all_pairs):
                qml.RY(w[layer, b, 0], wires=i)
                qml.CNOT(wires=[i, j])
                qml.RY(w[layer, b, 1], wires=i)
                qml.RY(w[layer, b, 2], wires=j)
        return qml.state()

    params = weights_np.ravel().astype(float)
    n_p    = len(params)
    psi0   = np.array(var_circuit(pnp.array(params)))
    derivs = []
    for i in range(n_p):
        p_p = params.copy(); p_p[i] += eps
        p_m = params.copy(); p_m[i] -= eps
        derivs.append(
            (np.array(var_circuit(pnp.array(p_p))) -
             np.array(var_circuit(pnp.array(p_m)))) / (2 * eps)
        )
    QFI = np.zeros((n_p, n_p))
    for i in range(n_p):
        for j in range(i, n_p):
            t1 = np.vdot(derivs[i], derivs[j])
            t2 = np.vdot(derivs[i], psi0) * np.vdot(psi0, derivs[j])
            QFI[i, j] = QFI[j, i] = 4.0 * np.real(t1 - t2)
    return QFI


def _eval(model):
    with torch.no_grad():
        tl_sum, preds = 0.0, []
        for i in range(0, len(X_te), 128):
            o = model(X_te[i:i+128])
            tl_sum += F.binary_cross_entropy(o, y_te[i:i+128]).item() * len(X_te[i:i+128])
            preds.append((o >= 0.5).cpu())
        test_loss = tl_sum / len(y_te)
        test_acc  = (torch.cat(preds) == y_te.bool().cpu()).float().mean().item()
    return test_loss, test_acc


# ══════════════════════════════════════════════════════════════════════════════
# B. QCNN + Adam
# ══════════════════════════════════════════════════════════════════════════════

def run_qcnn_adam(n_qubits):
    torch.manual_seed(42); np.random.seed(42)
    print(f"  [QCNN+Adam] n={n_qubits}…", flush=True)
    model  = make_qcnn(n_qubits)
    n_circ = model.weights.numel()
    opt    = torch.optim.Adam(model.parameters(), lr=LR_Q)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_STEPS_Q, eta_min=1e-4)
    losses = []
    for step in range(N_STEPS_Q):
        idx = np.random.choice(N_TRAIN, BATCH_Q, replace=False)
        opt.zero_grad()
        l = F.binary_cross_entropy(model(X_tr[idx]), y_tr[idx])
        l.backward(); opt.step(); sched.step()
        losses.append(l.item())
        if step % 100 == 0:
            print(f"    step {step:4d}  loss={l.item():.4f}  lr={sched.get_last_lr()[0]:.5f}",
                  flush=True)
    test_loss, test_acc = _eval(model)
    w_np = model.weights.detach().cpu().numpy()
    qfi  = compute_qfi_qcnn(w_np, n_qubits)
    tr_f, kappa = fisher_stats(qfi)
    n_tot = sum(p.numel() for p in model.parameters())
    print(f"    final  loss={test_loss:.2f}  acc={test_acc:.0%}  "
          f"tr/N={tr_f/n_circ:.3f}", flush=True)
    return dict(n_qubits=n_qubits, n_circ=n_circ, n_tot=n_tot,
                hilbert_dim=2**n_qubits, test_loss=test_loss, test_acc=test_acc,
                tr_f=tr_f, kappa=kappa, losses=losses)


# ══════════════════════════════════════════════════════════════════════════════
# C. QCNN + Pullback QNG
# ══════════════════════════════════════════════════════════════════════════════

def run_qcnn_qng_pb(n_qubits):
    """
    Pullback QNG on QCNN: combines local-gate geometry (no barren plateau)
    with the correct hybrid metric G_eff = (1/B) Σ p_i(1-p_i)(WJ_i)ᵀ(WJ_i).
    """
    torch.manual_seed(42); np.random.seed(42)
    print(f"  [QCNN+QNG_pb] n={n_qubits}…", flush=True)

    n_blocks, all_pairs = _qcnn_layout(n_qubits)
    N_BP      = 3
    n_circ    = N_Q_LAYERS * n_blocks * N_BP
    n_readout = n_qubits + n_qubits * (n_qubits - 1) // 2

    model  = make_qcnn(n_qubits)
    dev_ps = qml.device("lightning.qubit", wires=n_qubits)

    @qml.qnode(dev_ps, diff_method="parameter-shift")
    def circuit_ps(weights, inputs):
        for layer in range(N_Q_LAYERS):
            for i in range(n_qubits):
                qml.RY(inputs[i % N_FEATURES] * pnp.pi / 4, wires=i)
            for b, (i, j) in enumerate(all_pairs):
                qml.RY(weights[layer, b, 0], wires=i)
                qml.CNOT(wires=[i, j])
                qml.RY(weights[layer, b, 1], wires=i)
                qml.RY(weights[layer, b, 2], wires=j)
        singles = [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        pairs   = [qml.expval(qml.PauliZ(i) @ qml.PauliZ(j))
                   for i in range(n_qubits) for j in range(i + 1, n_qubits)]
        return tuple(singles + pairs)

    def circuit_jacobian(w_np, x_np):
        w_flat = w_np.ravel().astype(float)
        J = np.zeros((n_readout, n_circ))
        for i in range(n_circ):
            wp = w_flat.copy(); wp[i] += np.pi / 2
            wm = w_flat.copy(); wm[i] -= np.pi / 2
            J[:, i] = (
                np.array(circuit_ps(pnp.array(wp.reshape(w_np.shape), requires_grad=False),
                                    pnp.array(x_np.astype(float), requires_grad=False))) -
                np.array(circuit_ps(pnp.array(wm.reshape(w_np.shape), requires_grad=False),
                                    pnp.array(x_np.astype(float), requires_grad=False)))
            ) / 2
        return J

    opt_lin  = torch.optim.Adam(model.linear.parameters(), lr=LR_Q)
    sched_lin = torch.optim.lr_scheduler.CosineAnnealingLR(opt_lin, T_max=N_STEPS_Q, eta_min=1e-4)
    G_BATCH  = 32; G_K = 50; G_REG = 0.01
    LR_QNG_MAX = 0.005; LR_QNG_MIN = 1e-4
    mt_inv   = np.eye(n_circ)
    losses   = []
    for step in range(N_STEPS_Q):
        # cosine annealing for the circuit natural-gradient step size
        lr_qng = LR_QNG_MIN + 0.5 * (LR_QNG_MAX - LR_QNG_MIN) * (
            1 + np.cos(np.pi * step / N_STEPS_Q))
        if step % G_K == 0:
            w_np  = model.weights.detach().cpu().numpy()
            W_np  = model.linear.weight.detach().cpu().numpy()
            idx_g = np.random.choice(N_TRAIN, G_BATCH, replace=False)
            with torch.no_grad():
                p_g = model(torch.tensor(X_tr_np[idx_g], dtype=torch.float32)).cpu().numpy()
            G_eff = np.zeros((n_circ, n_circ))
            for x_i, p_i in zip(X_tr_np[idx_g], p_g):
                WJ_i  = W_np @ circuit_jacobian(w_np, x_i)
                G_eff += float(p_i * (1 - p_i)) * (WJ_i.T @ WJ_i)
            mt_inv = np.linalg.inv(G_eff / G_BATCH + G_REG * np.eye(n_circ))
        idx = np.random.choice(N_TRAIN, BATCH_Q, replace=False)
        model.zero_grad()
        l = F.binary_cross_entropy(model(X_tr[idx]), y_tr[idx])
        l.backward()
        if model.weights.grad is not None:
            g = model.weights.grad.detach().cpu().numpy().ravel()
            with torch.no_grad():
                model.weights -= lr_qng * torch.tensor(
                    (mt_inv @ g).reshape(model.weights.shape), dtype=torch.float32)
            model.weights.grad = None
        opt_lin.step()
        sched_lin.step()
        losses.append(l.item())
        if step % 100 == 0:
            print(f"    step {step:4d}  loss={l.item():.4f}  lr_qng={lr_qng:.5f}", flush=True)
    test_loss, test_acc = _eval(model)
    n_tot = sum(p.numel() for p in model.parameters())
    print(f"    final  loss={test_loss:.2f}  acc={test_acc:.0%}", flush=True)
    return dict(n_qubits=n_qubits, n_circ=n_circ, n_tot=n_tot,
                test_loss=test_loss, test_acc=test_acc, losses=losses)


# ══════════════════════════════════════════════════════════════════════════════
# Run
# ══════════════════════════════════════════════════════════════════════════════

Q_RANGE   = [2, 3, 4, 5, 6, 7, 8]
QNG_RANGE = [2, 3, 4, 5, 6, 7, 8]


def run_all():
    print("=" * 68)
    print("A. Classical MLP family")
    print("=" * 68)
    c_res = [run_classical(h) for h in [2, 4, 8, 16, 32, 64]]

    print("\n" + "=" * 68)
    print("B. QCNN + Adam")
    print("=" * 68)
    adam_res = [run_qcnn_adam(n) for n in Q_RANGE]

    print("\n" + "=" * 68)
    print("C. QCNN + QNG_pb (pullback through W)")
    print("=" * 68)
    pb_res = [run_qcnn_qng_pb(n) for n in QNG_RANGE]

    print("\n" + "=" * 72)
    print(f"{'Scenario':>20} {'n/H':>5} {'n_circ':>7} {'loss':>7} {'acc':>6}")
    print("-" * 72)
    for r in c_res:
        print(f"{'MLP':>20} {r['h']:>5}   {'–':>5} {r['test_loss']:>7.2f} {r['test_acc']:>6.0%}")
    for r in adam_res:
        print(f"{'QCNN+Adam':>20} {r['n_qubits']:>5} {r['n_circ']:>7} "
              f"{r['test_loss']:>7.2f} {r['test_acc']:>6.0%}")
    for r in pb_res:
        print(f"{'QCNN+QNG_pb':>20} {r['n_qubits']:>5} {r['n_circ']:>7} "
              f"{r['test_loss']:>7.2f} {r['test_acc']:>6.0%}")

    return c_res, adam_res, pb_res


if __name__ == "__main__":
    c_res, adam_res, pb_res = run_all()


# ══════════════════════════════════════════════════════════════════════════════
# Figures
# ══════════════════════════════════════════════════════════════════════════════

def smooth(v, w=20):
    return np.convolve(v, np.ones(w) / w, mode="valid")


def plot_scaling_fisher(c_res, adam_res, save=True):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(
        "Exp 4 — Classical scaling law & Fisher information efficiency\n"
        "Classical MLP (2→H→1)  vs.  QCNN brick-wall, data re-uploading, $\\mathrm{RY}(x\\cdot\\pi/4)$",
        fontsize=11,
    )
    c_np  = [r["n_params"] for r in c_res]
    c_l   = [r["test_loss"] for r in c_res]
    c_tpn = [r["tr_f"] / r["n_params"] for r in c_res]
    q_np  = [r["n_tot"] for r in adam_res]
    q_l   = [r["test_loss"] for r in adam_res]
    q_tpn = [r["tr_f"] / r["n_circ"] for r in adam_res]

    ax = axes[0]
    ax.loglog(c_np, c_l, "o-", color=PAL["c"], lw=2, label="Classical MLP")
    sl, ic, *_ = linregress(np.log(c_np), np.log(c_l))
    xf = np.geomspace(min(c_np), max(c_np), 200)
    ax.loglog(xf, np.exp(ic) * xf**sl, ":", color=PAL["c"], alpha=0.5)
    ax.text(0.55, 0.82, fr"$L \propto N^{{{sl:.2f}}}$",
            transform=ax.transAxes, color=PAL["c"], fontsize=10)
    ax.loglog(q_np, q_l, "s--", color=PAL["adam"], lw=2, label="QCNN + Adam")
    ax.set_xlabel("Parameters $N$"); ax.set_ylabel("Test loss")
    ax.set_title("Classical scaling law"); ax.legend(fontsize=9)

    ax = axes[1]
    ax.semilogy(range(len(c_res)), c_tpn, "o-", color=PAL["c"], lw=2,
                label=r"Classical $\mathrm{tr}(\hat{F})/N$")
    ax.semilogy(range(len(adam_res)), q_tpn, "s--", color=PAL["adam"], lw=2,
                label=r"QCNN $\mathrm{tr}(\mathcal{F}_Q)/N_\mathrm{circ}$")
    ax.set_xlabel("Model index"); ax.set_ylabel(r"$\mathrm{tr}(F)/N$")
    ax.set_title("Fisher information per parameter"); ax.legend(fontsize=9)

    plt.tight_layout()
    if save:
        plt.savefig("artifacts/exp4_scaling_fisher.png", dpi=300, bbox_inches="tight")
        print("\nSaved artifacts/exp4_scaling_fisher.png")
    return fig


def plot_qng_comparison(c_res, adam_res, pb_res, save=True):
    adam_by_n = {r["n_qubits"]: r for r in adam_res if r["n_qubits"] in QNG_RANGE}
    pb_by_n   = {r["n_qubits"]: r for r in pb_res}

    best_adam = min(adam_by_n.values(), key=lambda r: r["test_loss"])
    best_pb   = min(pb_by_n.values(),   key=lambda r: r["test_loss"])
    mlp_best  = min(c_res, key=lambda r: r["test_loss"])
    mlp_small = next(r for r in c_res if r["n_params"] >= 33)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_title(
        "QCNN: best Adam vs. best Pullback QNG  (1000 steps, two-moons)\n"
        r"Pullback metric $G_\mathrm{eff} = \frac{1}{B}\sum p_i(1{-}p_i)(WJ_i)^\top(WJ_i)$",
        fontsize=10,
    )
    ax.axhspan(mlp_best["test_loss"], mlp_small["test_loss"],
               color=PAL["c"], alpha=0.12,
               label=f"MLP plateau  ({mlp_best['test_loss']:.2f}–{mlp_small['test_loss']:.2f})")
    for res, col, lbl in [
        (best_adam, PAL["adam"],
         f"QCNN+Adam  n={best_adam['n_qubits']}  ({best_adam['test_loss']:.2f} / {best_adam['test_acc']:.0%})"),
        (best_pb,   PAL["pb"],
         f"QCNN+QNG$_{{\\rm pb}}$  n={best_pb['n_qubits']}  ({best_pb['test_loss']:.2f} / {best_pb['test_acc']:.0%})"),
    ]:
        s = smooth(res["losses"])
        ax.plot(s, color=col, lw=2, label=lbl)
        ax.axhline(res["test_loss"], color=col, lw=0.8, ls=":", alpha=0.5)
    ax.set_xlabel("Training step")
    ax.set_ylabel("BCE loss (smoothed)")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylim(bottom=0.1)
    plt.tight_layout()
    if save:
        plt.savefig("artifacts/exp4_qng.png", dpi=300, bbox_inches="tight")
        print("Saved artifacts/exp4_qng.png")
    return fig


if __name__ == "__main__":
    c_res, adam_res, pb_res = run_all()
    plot_scaling_fisher(c_res, adam_res)
    plt.close()
    plot_qng_comparison(c_res, adam_res, pb_res)
    plt.close()
