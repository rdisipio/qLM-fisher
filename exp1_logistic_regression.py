"""
Experiment 1 — Fisher information and natural gradient on logistic regression.

Directly addresses R1's request for a worked example where the Fisher information
matrix is computed explicitly, and the natural gradient update is compared to SGD.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

# ── Hyper-parameters ─────────────────────────────────────────────────────────
N, D   = 200, 2
N_STEPS = 300
LR_SGD  = 0.5
LR_NG   = 0.5    # NG can reuse the same lr: F⁻¹ already rescales the step
LR_ADAM = 0.05
LAMBDA  = 1e-4   # Tikhonov regularisation when inverting F

# ── Data ─────────────────────────────────────────────────────────────────────
X_raw, y = make_classification(
    n_samples=N, n_features=D, n_redundant=0,
    n_informative=D, class_sep=1.0, random_state=42,
)
X = StandardScaler().fit_transform(X_raw)
X_aug = np.hstack([X, np.ones((N, 1))])   # augment with bias column → N×(D+1)

# ── Logistic-regression primitives (all NumPy) ────────────────────────────────
def _sigmoid(z: np.ndarray) -> np.ndarray:
    # numerically stable
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)),
                    np.exp(z) / (1.0 + np.exp(z)))

def prob(theta: np.ndarray) -> np.ndarray:
    return _sigmoid(X_aug @ theta)

def bce(theta: np.ndarray) -> float:
    p = np.clip(prob(theta), 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

def grad(theta: np.ndarray) -> np.ndarray:
    return X_aug.T @ (prob(theta) - y) / N

def fisher(theta: np.ndarray) -> np.ndarray:
    """Exact Fisher: F = (1/N) Σ p_i(1-p_i) x_i xᵢᵀ"""
    p = prob(theta)
    w = p * (1 - p)               # shape (N,)
    return (X_aug.T * w) @ X_aug / N

def accuracy(theta: np.ndarray) -> float:
    return float(np.mean((prob(theta) >= 0.5) == y))

def ng_angle_deg(g: np.ndarray, ng: np.ndarray) -> float:
    """Angle in degrees between the vanilla gradient and natural-gradient direction."""
    cos_a = np.dot(g, ng) / (np.linalg.norm(g) * np.linalg.norm(ng) + 1e-15)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))

# ── Trainers ─────────────────────────────────────────────────────────────────
def train_sgd(lr: float) -> tuple:
    theta = np.zeros(D + 1)
    losses, accs = [], []
    for _ in range(N_STEPS):
        losses.append(bce(theta))
        accs.append(accuracy(theta))
        theta -= lr * grad(theta)
    return theta, np.array(losses), np.array(accs)


def train_ng(lr: float) -> tuple:
    theta = np.zeros(D + 1)
    losses, accs, angles = [], [], []
    for _ in range(N_STEPS):
        losses.append(bce(theta))
        accs.append(accuracy(theta))
        g  = grad(theta)
        F  = fisher(theta) + LAMBDA * np.eye(D + 1)
        ng = np.linalg.solve(F, g)          # F⁻¹ g, avoids explicit inversion
        angles.append(ng_angle_deg(g, ng))
        theta -= lr * ng
    return theta, np.array(losses), np.array(accs), np.array(angles)


def train_adam(lr: float, beta1: float = 0.9, beta2: float = 0.999,
               eps: float = 1e-8) -> tuple:
    theta = np.zeros(D + 1)
    m, v  = np.zeros_like(theta), np.zeros_like(theta)
    losses, accs = [], []
    for t in range(1, N_STEPS + 1):
        losses.append(bce(theta))
        accs.append(accuracy(theta))
        g      = grad(theta)
        m      = beta1 * m + (1 - beta1) * g
        v      = beta2 * v + (1 - beta2) * g ** 2
        m_hat  = m / (1 - beta1 ** t)
        v_hat  = v / (1 - beta2 ** t)
        theta -= lr * m_hat / (np.sqrt(v_hat) + eps)
    return theta, np.array(losses), np.array(accs)

# ── Run ───────────────────────────────────────────────────────────────────────
print("Training SGD …")
theta_sgd,  losses_sgd,  accs_sgd            = train_sgd(LR_SGD)
print("Training natural gradient …")
theta_ng,   losses_ng,   accs_ng,  angles_ng = train_ng(LR_NG)
print("Training Adam …")
theta_adam, losses_adam, accs_adam            = train_adam(LR_ADAM)

# ── Fisher summary at init and convergence ────────────────────────────────────
theta_init = np.zeros(D + 1)
for label, theta in [("init", theta_init), ("SGD final", theta_sgd)]:
    F   = fisher(theta)
    eig = np.linalg.eigvalsh(F)          # ascending order
    kappa = eig[-1] / (eig[0] + 1e-15)
    print(f"Fisher @ {label}: tr={np.trace(F):.4f}, "
          f"κ={kappa:.2f}, "
          f"top-2 eigs={eig[-2]:.4f}, {eig[-1]:.4f}")

print(f"Final losses  — SGD: {losses_sgd[-1]:.4f}, "
      f"NG: {losses_ng[-1]:.4f}, Adam: {losses_adam[-1]:.4f}")
print(f"Final accuracy— SGD: {accs_sgd[-1]:.3f}, "
      f"NG: {accs_ng[-1]:.3f}, Adam: {accs_adam[-1]:.3f}")

# ── Figure ────────────────────────────────────────────────────────────────────
PALETTE = {
    "sgd":  "#0072B2",
    "ng":   "#D55E00",
    "adam": "#009E73",
    "misc": "#CC79A7",
}
steps = np.arange(N_STEPS)

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle("Experiment 1: Fisher information and natural gradient "
             "(logistic regression)", fontsize=12)

# ── Top-left: loss curves ──────────────────────────────────────────────────
ax = axes[0, 0]
ax.plot(steps, losses_sgd,  label="SGD",              color=PALETTE["sgd"])
ax.plot(steps, losses_ng,   label="Natural gradient",  color=PALETTE["ng"])
ax.plot(steps, losses_adam, label="Adam",              color=PALETTE["adam"],
        linestyle="--")
ax.set_xlabel("Step")
ax.set_ylabel(r"$\mathcal{L}$")
ax.set_title("Training loss")
ax.set_yscale("log")
ax.legend(fontsize=8)

# ── Top-right: angle α vs step ─────────────────────────────────────────────
ax = axes[0, 1]
ax.plot(steps, angles_ng, color=PALETTE["misc"])
ax.axhline(45, color="gray", linestyle=":", linewidth=0.8, label=r"$45°$")
ax.set_xlabel("Step")
ax.set_ylabel(r"$\alpha$ (degrees)")
ax.set_title(r"Angle between SGD and NG update, $\alpha$")
ax.legend(fontsize=8)

# ── Bottom-left: Fisher eigenvalue spectrum at init vs convergence ──────────
ax = axes[1, 0]
eig_init = np.linalg.eigvalsh(fisher(theta_init))
eig_conv = np.linalg.eigvalsh(fisher(theta_sgd))
x_pos    = np.arange(D + 1)
width    = 0.35
ax.bar(x_pos - width / 2, eig_init, width,
       label="Init",        color=PALETTE["sgd"], alpha=0.85)
ax.bar(x_pos + width / 2, eig_conv, width,
       label="Convergence", color=PALETTE["ng"],  alpha=0.85)
ax.set_xlabel("Eigenvalue index")
ax.set_ylabel(r"$\lambda$")
ax.set_title(r"Fisher eigenvalue spectrum $\lambda(F)$")
ax.set_xticks(x_pos)
ax.legend(fontsize=8)

# ── Bottom-right: decision boundaries ──────────────────────────────────────
ax = axes[1, 1]
margin = 0.6
x0_lo, x0_hi = X[:, 0].min() - margin, X[:, 0].max() + margin
x1_lo, x1_hi = X[:, 1].min() - margin, X[:, 1].max() + margin
xx, yy = np.meshgrid(np.linspace(x0_lo, x0_hi, 300),
                     np.linspace(x1_lo, x1_hi, 300))
grid = np.c_[xx.ravel(), yy.ravel(), np.ones(xx.size)]

for theta, label, color in [
    (theta_sgd,  "SGD",             PALETTE["sgd"]),
    (theta_ng,   "Natural gradient",PALETTE["ng"]),
    (theta_adam, "Adam",            PALETTE["adam"]),
]:
    zz = _sigmoid(grid @ theta).reshape(xx.shape)
    ax.contour(xx, yy, zz, levels=[0.5], colors=[color], linewidths=1.8)

ax.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr",
           alpha=0.4, s=14, edgecolors="none")
ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")
ax.set_title("Decision boundaries (contour = 0.5)")

from matplotlib.lines import Line2D
ax.legend(handles=[
    Line2D([0], [0], color=PALETTE["sgd"],  label="SGD"),
    Line2D([0], [0], color=PALETTE["ng"],   label="Natural gradient"),
    Line2D([0], [0], color=PALETTE["adam"], label="Adam"),
], fontsize=8)

plt.tight_layout()
out = "plots/exp1_logistic_regression.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
print(f"Saved {out}")

# =============================================================================
# Experiment 1b — MLP (2→16→1, ReLU): empirical Fisher via per-sample gradients
#
# Logistic regression admits a closed-form Fisher; neural networks do not.
# Here we compute the exact empirical Fisher
#   F̂(θ) = (1/N) Σ_i ∇_θ L_i · ∇_θ L_i^T
# via per-sample backward passes and compare natural gradient to SGD/Adam
# on the same dataset, showing that the Fisher geometry is richer (higher κ,
# heavier-tailed spectrum) in even the smallest MLP.
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as Fnn

torch.manual_seed(42)

D_HIDDEN    = 16
N_STEPS_MLP = 500
LR_SGD_MLP  = 0.10
LR_NG_MLP   = 0.10
LR_ADAM_MLP = 0.01
# Damping for (F̂ + λI)⁻¹. With κ ≈ 10⁹ and λ=1e-4 the regularised
# condition number is ~7700 → effective lr up to 1000 → divergence.
# λ=1e-2 caps κ_reg ≈ 77 (max amplification ×100, effective lr ≤ 10).
LAMBDA_MLP  = 1e-2

# 80/20 train/test split of the N=200 standardised samples.
N_TRAIN = int(0.8 * N)   # 160 training, 40 test
X_t     = torch.tensor(X[:N_TRAIN], dtype=torch.float32)
y_t     = torch.tensor(y[:N_TRAIN], dtype=torch.float32)
X_t_tst = torch.tensor(X[N_TRAIN:], dtype=torch.float32)
y_t_tst = torch.tensor(y[N_TRAIN:], dtype=torch.float32)
print(f"Train: {N_TRAIN} samples  |  Test: {N - N_TRAIN} samples")


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(D, D_HIDDEN)
        self.fc2 = nn.Linear(D_HIDDEN, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.fc2(Fnn.relu(self.fc1(x)))).squeeze(-1)


N_PARAMS_MLP = sum(p.numel() for p in MLP().parameters())
# For D=2, D_HIDDEN=16: (2×16+16) + (16×1+1) = 48 + 17 = 65
print(f"\nMLP (2→{D_HIDDEN}→1) parameters: {N_PARAMS_MLP}")


def _init_mlp() -> MLP:
    """Fixed initialisation so all three optimisers start identically."""
    torch.manual_seed(42)
    m = MLP()
    nn.init.xavier_uniform_(m.fc1.weight)
    nn.init.zeros_(m.fc1.bias)
    nn.init.xavier_uniform_(m.fc2.weight)
    nn.init.zeros_(m.fc2.bias)
    return m


_INIT_STATE = _init_mlp().state_dict()


def make_mlp() -> MLP:
    m = MLP()
    m.load_state_dict(_INIT_STATE)
    return m


def _flat_grad(model: MLP) -> np.ndarray:
    return np.concatenate([p.grad.detach().numpy().ravel()
                           for p in model.parameters()])


def bce_mlp(model: MLP) -> torch.Tensor:
    return Fnn.binary_cross_entropy(model(X_t), y_t)


def acc_mlp(model: MLP) -> float:
    with torch.no_grad():
        return float(((model(X_t) >= 0.5) == y_t.bool()).float().mean())


def bce_mlp_tst(model: MLP) -> float:
    with torch.no_grad():
        return Fnn.binary_cross_entropy(model(X_t_tst), y_t_tst).item()


def acc_mlp_tst(model: MLP) -> float:
    with torch.no_grad():
        return float(((model(X_t_tst) >= 0.5) == y_t_tst.bool()).float().mean())


def empirical_fisher_mlp(model: MLP) -> np.ndarray:
    """Exact empirical Fisher via per-sample gradient outer products."""
    F_mat = np.zeros((N_PARAMS_MLP, N_PARAMS_MLP))
    model.eval()
    for xi, yi in zip(X_t, y_t):
        model.zero_grad()
        Fnn.binary_cross_entropy(
            model(xi.unsqueeze(0)), yi.unsqueeze(0)
        ).backward()
        g = _flat_grad(model)
        F_mat += np.outer(g, g)
    model.train()
    return F_mat / N_TRAIN


def _apply_ng_step(model: MLP, ng: np.ndarray, lr: float):
    with torch.no_grad():
        off = 0
        for p in model.parameters():
            n = p.numel()
            p.data -= lr * torch.from_numpy(
                ng[off : off + n].reshape(p.shape)).to(dtype=p.dtype)
            off += n


# ── MLP trainers ──────────────────────────────────────────────────────────────

def train_mlp_sgd(lr: float) -> tuple:
    model = make_mlp()
    opt   = torch.optim.SGD(model.parameters(), lr=lr)
    losses, accs, tst_losses, tst_accs = [], [], [], []
    for _ in range(N_STEPS_MLP):
        model.zero_grad()
        loss = bce_mlp(model)
        losses.append(loss.item())
        accs.append(acc_mlp(model))
        tst_losses.append(bce_mlp_tst(model))
        tst_accs.append(acc_mlp_tst(model))
        loss.backward()
        opt.step()
    return model, np.array(losses), np.array(accs), np.array(tst_losses), np.array(tst_accs)


def train_mlp_ng(lr: float) -> tuple:
    model  = make_mlp()
    losses, accs, angles, tst_losses, tst_accs = [], [], [], [], []
    for _ in range(N_STEPS_MLP):
        model.zero_grad()
        loss = bce_mlp(model)
        losses.append(loss.item())
        accs.append(acc_mlp(model))
        tst_losses.append(bce_mlp_tst(model))
        tst_accs.append(acc_mlp_tst(model))
        loss.backward()
        g      = _flat_grad(model)
        F_mat  = empirical_fisher_mlp(model)
        F_reg  = F_mat + LAMBDA_MLP * np.eye(N_PARAMS_MLP)
        ng     = np.linalg.solve(F_reg, g)
        cos_a  = np.dot(g, ng) / (np.linalg.norm(g) * np.linalg.norm(ng) + 1e-15)
        angles.append(float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))))
        _apply_ng_step(model, ng, lr)
    return (model, np.array(losses), np.array(accs), np.array(angles),
            np.array(tst_losses), np.array(tst_accs))


def train_mlp_adam(lr: float) -> tuple:
    model = make_mlp()
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    losses, accs, tst_losses, tst_accs = [], [], [], []
    for _ in range(N_STEPS_MLP):
        model.zero_grad()
        loss = bce_mlp(model)
        losses.append(loss.item())
        accs.append(acc_mlp(model))
        tst_losses.append(bce_mlp_tst(model))
        tst_accs.append(acc_mlp_tst(model))
        loss.backward()
        opt.step()
    return model, np.array(losses), np.array(accs), np.array(tst_losses), np.array(tst_accs)


# ── Run ───────────────────────────────────────────────────────────────────────
print("Training MLP — SGD …")
mlp_sgd,  mlp_losses_sgd,  mlp_accs_sgd,  mlp_tst_losses_sgd,  mlp_tst_accs_sgd  = train_mlp_sgd(LR_SGD_MLP)
print("Training MLP — natural gradient …")
mlp_ng,   mlp_losses_ng,   mlp_accs_ng,   mlp_angles, mlp_tst_losses_ng,   mlp_tst_accs_ng   = train_mlp_ng(LR_NG_MLP)
print("Training MLP — Adam …")
mlp_adam, mlp_losses_adam, mlp_accs_adam, mlp_tst_losses_adam, mlp_tst_accs_adam = train_mlp_adam(LR_ADAM_MLP)

# ── Fisher summary at init and convergence ────────────────────────────────────
mlp_init_model = make_mlp()
F_init_mlp = empirical_fisher_mlp(mlp_init_model)
F_conv_mlp = empirical_fisher_mlp(mlp_sgd)

for label, F_np in [("init", F_init_mlp), ("SGD final", F_conv_mlp)]:
    eig   = np.linalg.eigvalsh(F_np)
    kappa = eig[-1] / (max(abs(eig[0]), 1e-15))
    print(f"MLP Fisher @ {label}: tr={np.trace(F_np):.4f}, "
          f"κ={kappa:.2e}, top-2 eigs={eig[-2]:.6f}, {eig[-1]:.6f}")

print(f"\nMLP train losses  — SGD: {mlp_losses_sgd[-1]:.4f}, "
      f"NG: {mlp_losses_ng[-1]:.4f}, Adam: {mlp_losses_adam[-1]:.4f}")
print(f"MLP test  losses  — SGD: {mlp_tst_losses_sgd[-1]:.4f}, "
      f"NG: {mlp_tst_losses_ng[-1]:.4f}, Adam: {mlp_tst_losses_adam[-1]:.4f}")
print(f"MLP train accuracy— SGD: {mlp_accs_sgd[-1]:.3f}, "
      f"NG: {mlp_accs_ng[-1]:.3f}, Adam: {mlp_accs_adam[-1]:.3f}")
print(f"MLP test  accuracy— SGD: {mlp_tst_accs_sgd[-1]:.3f}, "
      f"NG: {mlp_tst_accs_ng[-1]:.3f}, Adam: {mlp_tst_accs_adam[-1]:.3f}")

# ── Figure ────────────────────────────────────────────────────────────────────
steps_mlp = np.arange(N_STEPS_MLP)

fig_mlp, axes_mlp = plt.subplots(1, 3, figsize=(13, 4.5))
fig_mlp.suptitle(
    f"Experiment 1b: Fisher information and natural gradient "
    f"(MLP 2→{D_HIDDEN}→1, ReLU, 500 steps)",
    fontsize=12,
)

# Loss curves — solid=train, dashed=test, same colour per optimiser
ax = axes_mlp[0]
ax.plot(steps_mlp, mlp_losses_sgd,      color=PALETTE["sgd"],  label="SGD train")
ax.plot(steps_mlp, mlp_tst_losses_sgd,  color=PALETTE["sgd"],  linestyle="--", alpha=0.6, label="SGD test")
ax.plot(steps_mlp, mlp_losses_ng,       color=PALETTE["ng"],   label="NG train")
ax.plot(steps_mlp, mlp_tst_losses_ng,   color=PALETTE["ng"],   linestyle="--", alpha=0.6, label="NG test")
ax.plot(steps_mlp, mlp_losses_adam,     color=PALETTE["adam"], label="Adam train")
ax.plot(steps_mlp, mlp_tst_losses_adam, color=PALETTE["adam"], linestyle="--", alpha=0.6, label="Adam test")
ax.set_xlabel("Step")
ax.set_ylabel(r"$\mathcal{L}$")
ax.set_title("Training & test loss (solid / dashed)")
ax.set_yscale("log")
ax.legend(fontsize=7, ncol=2)

# Angle α between SGD and NG update
ax = axes_mlp[1]
ax.plot(steps_mlp, mlp_angles, color=PALETTE["misc"])
ax.axhline(45, color="gray", linestyle=":", linewidth=0.8, label=r"$45°$")
ax.set_xlabel("Step")
ax.set_ylabel(r"$\alpha$ (degrees)")
ax.set_title(r"Angle between SGD and NG update, $\alpha$")
ax.legend(fontsize=8)

# Full eigenvalue spectrum (sorted descending, log scale)
ax = axes_mlp[2]
eig_init_mlp = np.sort(np.linalg.eigvalsh(F_init_mlp))[::-1]
eig_conv_mlp = np.sort(np.linalg.eigvalsh(F_conv_mlp))[::-1]
idx = np.arange(1, N_PARAMS_MLP + 1)
ax.semilogy(idx, np.clip(eig_init_mlp, 1e-12, None), "o-", markersize=3,
            color=PALETTE["sgd"], label="Init")
ax.semilogy(idx, np.clip(eig_conv_mlp, 1e-12, None), "s-", markersize=3,
            color=PALETTE["ng"],  label="Convergence (SGD)")
ax.set_xlabel("Eigenvalue index (sorted descending)")
ax.set_ylabel(r"$\lambda$")
ax.set_title(r"Fisher eigenvalue spectrum $\lambda(\hat{F})$")
ax.legend(fontsize=8)

plt.tight_layout()
out_mlp = "plots/exp1_mlp.png"
plt.savefig(out_mlp, dpi=300, bbox_inches="tight")
print(f"Saved {out_mlp}")
