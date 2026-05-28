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
out = "exp1_logistic_regression.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
print(f"Saved {out}")
