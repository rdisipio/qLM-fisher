"""
Experiment 5 — Two-class interleaved spiral: Classical vs. Quantum Scaling Law

A two-class Archimedean spiral (N_TURNS full rotations per arm) exposes a smooth
L ∝ N^α scaling law without the capacity threshold of two-moons.  The boundary
complexity grows continuously with N_TURNS, making it a better stress test for
parameter-efficiency comparisons.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import linregress
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import pennylane.numpy as pnp

# ── Shared circuit utilities from exp4 ────────────────────────────────────────
from exp4_mlp_vs_qcnn import (
    make_qcnn, _qcnn_layout, compute_qfi_qcnn, fisher_stats, smooth,
    N_Q_LAYERS, N_FEATURES,
    LR_C, LR_Q, LR_QNG_MIN, N_STEPS_C, N_STEPS_Q, BATCH_Q, FISHER_N,
    PAL, Q_RANGE, QNG_RANGE,
)

np.random.seed(42)
torch.manual_seed(42)

# ── Hyperparameters ────────────────────────────────────────────────────────────
N_SAMPLES = 2000
N_TURNS   = 2     # full rotations per arm — increase for harder task
NOISE    = 0.05   # Gaussian noise on the spiral
N_STEPS_C = 2000  # classical MLP training steps

# ── Dataset ───────────────────────────────────────────────────────────────────
def make_spiral(n_samples=N_SAMPLES, n_turns=N_TURNS, noise=NOISE, seed=42):
    """Two interleaved Archimedean spirals, binary labels."""
    rng   = np.random.default_rng(seed)
    n_per = n_samples // 2
    t     = np.linspace(0.5, n_turns * 2 * np.pi, n_per)
    r     = t / (n_turns * 2 * np.pi)
    x0    = np.column_stack([r * np.cos(t),           r * np.sin(t)])
    x1    = np.column_stack([r * np.cos(t + np.pi),   r * np.sin(t + np.pi)])
    X     = np.vstack([x0, x1]) + rng.normal(0, noise, (n_samples, 2))
    y     = np.array([0]*n_per + [1]*n_per, dtype=np.float32)
    idx   = rng.permutation(n_samples)
    return X[idx].astype(np.float32), y[idx]


X_raw, y = make_spiral()
X = StandardScaler().fit_transform(X_raw)
X_tr_np, X_te_np, y_tr_np, y_te_np = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_tr = torch.tensor(X_tr_np, dtype=torch.float32)
y_tr = torch.tensor(y_tr_np, dtype=torch.float32)
X_te = torch.tensor(X_te_np, dtype=torch.float32)
y_te = torch.tensor(y_te_np, dtype=torch.float32)
N_TRAIN = len(X_tr)


def load_dataset(X, y):
    """Replace the active dataset used by all run_* functions."""
    global X_tr, y_tr, X_te, y_te, X_tr_np, X_te_np, N_TRAIN
    X_tr_np, X_te_np, y_tr_np, y_te_np = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_tr    = torch.tensor(X_tr_np, dtype=torch.float32)
    y_tr    = torch.tensor(y_tr_np, dtype=torch.float32)
    X_te    = torch.tensor(X_te_np, dtype=torch.float32)
    y_te    = torch.tensor(y_te_np, dtype=torch.float32)
    N_TRAIN = len(X_tr)


# ── Classical MLP ─────────────────────────────────────────────────────────────
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


def run_classical(h, n_steps=N_STEPS_C):
    torch.manual_seed(42)
    model  = ClassicalMLP(h)
    opt    = torch.optim.Adam(model.parameters(), lr=LR_C)
    losses = []
    for _ in range(n_steps):
        opt.zero_grad()
        l = F.binary_cross_entropy(model(X_tr), y_tr)
        l.backward(); opt.step()
        losses.append(l.item())
    test_loss, test_acc = _eval(model)
    n_p         = sum(p.numel() for p in model.parameters())
    tr_f, kappa = fisher_stats(empirical_fisher(model))
    print(f"  MLP H={h:3d}  n_params={n_p:4d}  loss={test_loss:.2f}  "
          f"acc={test_acc:.0%}  tr/N={tr_f/n_p:.3f}", flush=True)
    return dict(h=h, n_params=n_p, test_loss=test_loss, test_acc=test_acc,
                tr_f=tr_f, kappa=kappa, losses=losses)


# ── QCNN + Adam ───────────────────────────────────────────────────────────────
def run_qcnn_adam(n_qubits, n_steps=None):
    n_steps = n_steps if n_steps is not None else N_STEPS_Q
    torch.manual_seed(42); np.random.seed(42)
    print(f"  [QCNN+Adam] n={n_qubits}…", flush=True)
    model  = make_qcnn(n_qubits)
    n_circ = model.weights.numel()
    opt    = torch.optim.Adam(model.parameters(), lr=LR_Q)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps, eta_min=1e-3)
    losses = []
    for step in range(n_steps):
        idx = np.random.choice(N_TRAIN, BATCH_Q, replace=False)
        opt.zero_grad()
        l = F.binary_cross_entropy(model(X_tr[idx]), y_tr[idx])
        l.backward(); opt.step(); sched.step()
        losses.append(l.item())
        if step % 100 == 0:
            print(f"    step {step:4d}  loss={l.item():.4f}  lr={sched.get_last_lr()[0]:.5f}",
                  flush=True)
    test_loss, test_acc = _eval(model)
    w_np        = model.weights.detach().cpu().numpy()
    qfi         = compute_qfi_qcnn(w_np, n_qubits)
    tr_f, kappa = fisher_stats(qfi)
    n_tot       = sum(p.numel() for p in model.parameters())
    print(f"    final  loss={test_loss:.2f}  acc={test_acc:.0%}  "
          f"tr/N={tr_f/n_circ:.3f}", flush=True)
    return dict(n_qubits=n_qubits, n_circ=n_circ, n_tot=n_tot,
                hilbert_dim=2**n_qubits, test_loss=test_loss, test_acc=test_acc,
                tr_f=tr_f, kappa=kappa, losses=losses)


# ── QCNN + Pullback QNG ───────────────────────────────────────────────────────
def run_qcnn_qng_pb(n_qubits, n_steps=None, lr_qng_min=None):
    """Pullback QNG on QCNN with spiral dataset."""
    n_steps    = n_steps    if n_steps    is not None else N_STEPS_Q
    lr_qng_min = lr_qng_min if lr_qng_min is not None else LR_QNG_MIN
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

    opt_lin   = torch.optim.Adam(model.linear.parameters(), lr=LR_Q)
    sched_lin = torch.optim.lr_scheduler.CosineAnnealingLR(opt_lin, T_max=n_steps, eta_min=1e-3)
    G_BATCH   = 32; G_K = 50; G_REG = 0.01
    LR_QNG_MAX = 0.005
    mt_inv    = np.eye(n_circ)
    losses    = []
    for step in range(n_steps):
        lr_qng = lr_qng_min + 0.5 * (LR_QNG_MAX - lr_qng_min) * (
            1 + np.cos(np.pi * step / n_steps))
        if step % G_K == 0:
            w_np  = model.weights.detach().cpu().numpy()
            W_np  = model.linear.weight.detach().cpu().numpy()
            idx_g = np.random.choice(N_TRAIN, G_BATCH, replace=False)
            with torch.no_grad():
                p_g = model(torch.tensor(X_tr_np[idx_g], dtype=torch.float32)).cpu().numpy()
            G_eff = np.zeros((n_circ, n_circ))
            for x_i, p_i in zip(X_tr_np[idx_g], p_g):
                WJ_i   = W_np @ circuit_jacobian(w_np, x_i)
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


if __name__ == "__main__":
    print("=" * 68)
    print("A. Classical MLP family")
    print("=" * 68)
    c_res = [run_classical(h) for h in [1, 2, 3, 4, 8, 16, 32, 64]]

    print("=" * 68)
    print("B. QCNN + Adam")
    print("=" * 68)
    adam_res = [run_qcnn_adam(n) for n in Q_RANGE]

    print("=" * 68)
    print("C. QCNN + QNG_pb")
    print("=" * 68)
    pb_res = [run_qcnn_qng_pb(n) for n in QNG_RANGE]
