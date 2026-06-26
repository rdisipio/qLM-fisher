"""
Experiment 2 — Empirical Fisher / K-FAC summary scalars on a small transformer.

Trains a 2-layer byte-level transformer encoder on WikiText-2 and tracks
diagonal empirical Fisher scalars at five checkpoints across training.
Directly addresses R1's request for an LLM-relevant approximation experiment.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset
from tqdm import tqdm

np.random.seed(42)
torch.manual_seed(42)

# ── Constants ─────────────────────────────────────────────────────────────────
VOCAB           = 256
D_MODEL         = 64
NHEAD           = 2
FFN_DIM         = 128
NUM_LAYERS      = 2
SEQ_LEN         = 32
BATCH_SIZE      = 64
N_EPOCHS        = 20
LR              = 1e-3
FISHER_SAMPLES  = 64     # per-sample gradients for diagonal Fisher estimate
# epochs at which to snapshot Fisher (≈ 0 %, 10 %, 30 %, 60 %, 100 % of training)
CHECKPOINT_EPOCHS = {0, 2, 6, 12, 20}

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── Dataset ───────────────────────────────────────────────────────────────────
print("Loading WikiText-2 …")
raw = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

def text_to_bytes(split: str) -> np.ndarray:
    text = "".join(raw[split]["text"])
    return np.frombuffer(text.encode("utf-8", errors="replace"), dtype=np.uint8).copy()

train_bytes = text_to_bytes("train")
val_bytes   = text_to_bytes("validation")
print(f"Train: {len(train_bytes):,} bytes  |  Val: {len(val_bytes):,} bytes")


class ByteSeqDataset(Dataset):
    """Sliding-window next-byte prediction dataset."""
    def __init__(self, data: np.ndarray, seq_len: int):
        n = (len(data) - 1) // seq_len
        self.x = torch.from_numpy(
            data[: n * seq_len].reshape(n, seq_len).astype(np.int64))
        self.y = torch.from_numpy(
            data[1: n * seq_len + 1].reshape(n, seq_len).astype(np.int64))

    def __len__(self):            return len(self.x)
    def __getitem__(self, i):     return self.x[i], self.y[i]


train_ds = ByteSeqDataset(train_bytes, SEQ_LEN)
val_ds   = ByteSeqDataset(val_bytes,   SEQ_LEN)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
# batch_size=1 loader used for per-sample Fisher gradients
fisher_dl = DataLoader(train_ds, batch_size=1, shuffle=True, drop_last=True)
print(f"Train batches: {len(train_dl)}  |  Val batches: {len(val_dl)}")

# ── Model ──────────────────────────────────────────────────────────────────────
class SmallTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB,   D_MODEL)
        self.pos_emb = nn.Embedding(SEQ_LEN, D_MODEL)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=NHEAD, dim_feedforward=FFN_DIM,
            batch_first=True, dropout=0.1, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=NUM_LAYERS)
        self.head = nn.Linear(D_MODEL, VOCAB, bias=False)
        self._init_weights()

    def _init_weights(self):
        for emb in (self.tok_emb, self.pos_emb):
            nn.init.normal_(emb.weight, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T   = x.shape[1]
        pos = torch.arange(T, device=x.device)
        h   = self.tok_emb(x) + self.pos_emb(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        h   = self.encoder(h, mask=mask)
        return self.head(h)                      # B × T × VOCAB


model = SmallTransformer().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Parameters: {n_params:,}")

# ── Fisher utilities ──────────────────────────────────────────────────────────
def diagonal_fisher(model: nn.Module, dl: DataLoader, n_samples: int) -> dict:
    """
    Diagonal empirical Fisher: F̂_diag ≈ (1/B) Σ_i (∇_θ L_i)²

    Uses per-sample gradients (batch_size=1 loader) so each term is the
    squared gradient of one sequence's cross-entropy.
    """
    model.eval()
    diag = {name: torch.zeros_like(p)
            for name, p in model.named_parameters() if p.requires_grad}
    count = 0
    for x, y in dl:
        if count >= n_samples:
            break
        x, y = x.to(DEVICE), y.to(DEVICE)
        model.zero_grad()
        logits = model(x)
        loss   = F.cross_entropy(logits.view(-1, VOCAB), y.view(-1))
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                diag[name] += p.grad.detach() ** 2
        count += 1
    for name in diag:
        diag[name] /= max(count, 1)
    model.train()
    return diag


def fisher_scalars(diag: dict) -> tuple:
    """Returns (trace, λ_max, κ, ‖F̂‖_F) from the diagonal approximation."""
    vals  = torch.cat([v.flatten().cpu() for v in diag.values()])
    tr    = vals.sum().item()
    lmax  = vals.max().item()
    pos   = vals[vals > 0]
    lmin  = pos.min().item() if pos.numel() > 0 else 1e-30
    kappa = lmax / (lmin + 1e-30)
    frob  = (vals ** 2).sum().sqrt().item()
    return tr, lmax, kappa, frob


# ── Evaluation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model: nn.Module, dl: DataLoader) -> float:
    model.eval()
    total_loss, total_tok = 0.0, 0
    for x, y in dl:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits      = model(x)
        total_loss += F.cross_entropy(
            logits.view(-1, VOCAB), y.view(-1), reduction="sum").item()
        total_tok  += y.numel()
    model.train()
    return total_loss / total_tok


# ── Storage ───────────────────────────────────────────────────────────────────
train_losses, val_losses                         = [], []
ckpt_steps, ckpt_tr, ckpt_lmax                   = [], [], []
ckpt_kappa, ckpt_frob, ckpt_gap                  = [], [], []

global_step = 0

# ── Checkpoint 0 (before any training) ───────────────────────────────────────
print("\nCheckpoint 0 % (before training) …")
t_loss_0 = evaluate(model, DataLoader(train_ds, batch_size=BATCH_SIZE,
                                       shuffle=False, drop_last=True))
v_loss_0 = evaluate(model, val_dl)
diag0 = diagonal_fisher(model, fisher_dl, FISHER_SAMPLES)
tr0, lmax0, kappa0, frob0 = fisher_scalars(diag0)
ckpt_steps.append(0)
ckpt_tr.append(tr0); ckpt_lmax.append(lmax0)
ckpt_kappa.append(kappa0); ckpt_frob.append(frob0)
ckpt_gap.append(v_loss_0 - t_loss_0)
print(f"  tr={tr0:.4e}  λ_max={lmax0:.4e}  κ={kappa0:.2e}  "
      f"gap={v_loss_0 - t_loss_0:.4f}")

# ── Training loop ─────────────────────────────────────────────────────────────
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(1, N_EPOCHS + 1):
    model.train()
    epoch_loss = epoch_tok = 0

    for x, y in tqdm(train_dl, desc=f"Epoch {epoch:2d}/{N_EPOCHS}", leave=False):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x)
        loss   = F.cross_entropy(logits.view(-1, VOCAB), y.view(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item() * y.numel()
        epoch_tok  += y.numel()
        global_step += 1

    t_loss = epoch_loss / epoch_tok
    v_loss = evaluate(model, val_dl)
    train_losses.append(t_loss)
    val_losses.append(v_loss)
    print(f"Epoch {epoch:2d} | train={t_loss:.4f} | val={v_loss:.4f}")

    if epoch in CHECKPOINT_EPOCHS:
        pct = round(epoch / N_EPOCHS * 100)
        print(f"  → Checkpoint {pct} % (epoch {epoch}) …")
        diag = diagonal_fisher(model, fisher_dl, FISHER_SAMPLES)
        tr, lmax, kappa, frob = fisher_scalars(diag)
        ckpt_steps.append(global_step)
        ckpt_tr.append(tr); ckpt_lmax.append(lmax)
        ckpt_kappa.append(kappa); ckpt_frob.append(frob)
        ckpt_gap.append(v_loss - t_loss)
        print(f"    tr={tr:.4e}  λ_max={lmax:.4e}  κ={kappa:.2e}  "
              f"gap={v_loss - t_loss:.4f}")

# ── Figure ────────────────────────────────────────────────────────────────────
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9"]

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
fig.suptitle(
    "Experiment 2: Empirical Fisher scalars — 2-layer transformer on WikiText-2 "
    "(byte-level, diagonal approximation)",
    fontsize=10,
)

# ── Left: tr(F̂) and λ_max over training ──────────────────────────────────
ax  = axes[0]
ax2 = ax.twinx()
ax.plot(ckpt_steps, ckpt_tr,   "o-",  color=PALETTE[0],
        label=r"$\mathrm{tr}(\hat{F})$")
ax2.plot(ckpt_steps, ckpt_lmax, "s--", color=PALETTE[1],
         label=r"$\lambda_{\max}(\hat{F})$")
ax.set_xlabel("Training step")
ax.set_ylabel(r"$\mathrm{tr}(\hat{F})$",      color=PALETTE[0])
ax2.set_ylabel(r"$\lambda_{\max}(\hat{F})$",  color=PALETTE[1])
ax.tick_params(axis="y", labelcolor=PALETTE[0])
ax2.tick_params(axis="y", labelcolor=PALETTE[1])
ax.set_title("Curvature magnitude over training")
lines  = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
labels = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
ax.legend(lines, labels, fontsize=8, loc="upper right")

# ── Centre: κ vs training step ──────────────────────────────────────────────
ax = axes[1]
ax.plot(ckpt_steps, ckpt_kappa, "D-", color=PALETTE[2])
ax.set_xlabel("Training step")
ax.set_ylabel(r"$\kappa(\hat{F}) = \lambda_{\max}/\lambda_{\min}$")
ax.set_title(r"Condition number $\kappa(\hat{F})$")
ax.set_yscale("log")

# ── Right: scatter κ vs generalisation gap ──────────────────────────────────
ax = axes[2]
sc = ax.scatter(ckpt_kappa, ckpt_gap, c=ckpt_steps,
                cmap="viridis", s=90, zorder=5)
plt.colorbar(sc, ax=ax, label="Training step")
for i, step in enumerate(ckpt_steps):
    ax.annotate(f"step {step}",
                (ckpt_kappa[i], ckpt_gap[i]),
                textcoords="offset points", xytext=(5, 3), fontsize=7)
ax.set_xlabel(r"$\kappa(\hat{F})$")
ax.set_ylabel(r"$\mathcal{L}_{\mathrm{val}} - \mathcal{L}_{\mathrm{train}}$")
ax.set_title("Condition number vs. generalisation gap")
ax.set_xscale("log")
ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)

plt.tight_layout()
out = "plots/exp2_transformer_fisher.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
print(f"\nSaved {out}")
