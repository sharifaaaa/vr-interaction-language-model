# train_finetune.py — unified head; TemporalEncoder vs VR (transformer_blocks) finetune
from __future__ import annotations
from typing import Dict, Tuple, List, Optional

import os
import time
import random
from collections import Counter
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

# Safe plotting in headless environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Data
from pretrain_dataset import labeled_collate_fn, LabeledWindowsDataset

# Temporal backbone (legacy, post-norm)
from train_pretrain import TemporalEncoder

# VR transformer encoder (a prenorm stack; encoder-only → returns Y or (scores,weights,Y))
from vr_transformer.transformer_blocks import TransformerEncoder

# Configs
import config_attention as CATTN   # ATTN_CONFIG, POOLING, USE_PRETRAINED, SAVE_BASELINE, ...
import config_pretrain as CFG      # finetune / paths / loss config

# Confusion tools
from confusion_tools import (
    compute_confusion_matrix,
    precision_recall_f1_from_cm,
    ConfusionTracker,accuracy_from_cm,
)

# ---- Embedding visualizations: PCA / t-SNE / UMAP ----
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
try:
    import umap
    _UMAP = umap.UMAP
except Exception:
    class _UMAP:
        def __init__(self, **kwargs): pass
        def fit_transform(self, X):
            raise ImportError("Install umap-learn to enable UMAP panel.")

# ----------------------------
# Small helpers
# ----------------------------

def _cfg(name: str, default=None):
    return getattr(CFG, name, default)

def _count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _train_val_split(n: int, val_ratio: float = 0.2, seed: int = 42) -> Tuple[List[int], List[int]]:
    idxs = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idxs)
    n_val = max(1, int(round(val_ratio * n)))
    val_idx = idxs[:n_val]
    train_idx = idxs[n_val:]
    return train_idx, val_idx

def _compute_class_weights(ds: LabeledWindowsDataset, class_to_idx: Dict[str, int]) -> torch.Tensor:
    counts = Counter(str(row) for row in ds.df_lab[ds.label_col].values)
    n_classes = len(class_to_idx)
    freqs = np.zeros(n_classes, dtype=np.float32)
    for cls, idx in class_to_idx.items():
        freqs[idx] = counts.get(cls, 0)
    freqs = np.where(freqs == 0, 1.0, freqs)  # avoid div-by-zero
    weights = (freqs.sum() / (len(freqs) * freqs)).astype(np.float32)
    return torch.tensor(weights, dtype=torch.float32)

def _ensure_parent(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

# ---- plots & logs ----

def _save_weights_vs_metrics_plot(
    class_names,
    class_weights,
    f1_per_cls,
    rec_per_cls,
    acc_per_cls,            # ✅ NEW
    out_png,
    out_csv,
):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    # ------- CSV -------
    rows = [["class", "weight", "F1", "recall", "accuracy"]]  # ✅ add accuracy column
    rows += [
        [c, float(w), float(f), float(r), float(a)]
        for c, w, f, r, a in zip(class_names, class_weights, f1_per_cls, rec_per_cls, acc_per_cls)
    ]
    with open(out_csv, "w", newline="") as f:
        csv.writer(f).writerows(rows)

    # ------- Plot -------
    x = np.arange(len(class_names))
    width = 0.24  # three bars → slightly narrower
    fig, ax1 = plt.subplots(figsize=(10, 4.2))

    b1 = ax1.bar(x - width, f1_per_cls,  width=width, label="F1")
    b2 = ax1.bar(x,          rec_per_cls, width=width, label="Recall")
    b3 = ax1.bar(x + width,  acc_per_cls, width=width, label="Accuracy")  # ✅ NEW

    ax1.set_ylim(0, 1.0)
    ax1.set_ylabel("Score (0–1)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(class_names, rotation=0)
    ax1.set_title("Class weight vs per-class F1 / Recall / Accuracy (final)")
    ax1.legend(loc="upper left")

    # Class weights on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(x, class_weights, marker="o", linewidth=2, label="Class weight")
    ax2.set_ylabel("Class weight")
    ax2.tick_params(axis="y")

    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

def _f1_per_class_from_cm(cm: np.ndarray) -> np.ndarray:
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = (2.0 * tp + fp + fn)
    denom[denom == 0.0] = 1.0
    return (2.0 * tp) / denom

def _plot_f1_per_class_over_epochs(
    history: dict,
    class_names: list[str],
    class_weights: np.ndarray,
    out_path: str = "./artifacts/finetune/f1_per_class_over_epochs.png",
):
    _ensure_parent(out_path)
    epochs = np.arange(1, len(history["train_loss"]) + 1, dtype=int)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    for cname in class_names:
        ys = history["per_class_f1"][cname]
        ax1.plot(epochs, ys, marker="o", linewidth=2, label=f"F1 – {cname}")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Per-class F1")
    ax1.set_ylim(0.0, 1.0)
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    for w in np.asarray(class_weights, dtype=float):
        ax2.plot(epochs, np.full_like(epochs, w, dtype=float), color="#8B0000", linewidth=2, alpha=0.35)
    ax2.set_ylabel("Class weight", color="#8B0000")
    ax2.tick_params(axis="y", colors="#8B0000")

    h1, l1 = ax1.get_legend_handles_labels()
    weight_handle = plt.Line2D([0], [0], color="#8B0000", lw=2, label="Class weight")
    ax1.legend(h1 + [weight_handle], l1 + ["Class weight"], loc="lower right")

    plt.title("Per-class F1 over epochs (validation) + class weights")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

def _plot_finetune_history(history: dict, out_path: str):
    _ensure_parent(out_path)
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure()
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Fine-tune Loss")
    plt.tight_layout()
    p = Path("./artifacts/finetune/finetune_loss.png")
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        p.unlink()
    plt.savefig(p)
    plt.savefig(out_path)
    plt.close()

def _write_history_csv(history: dict, csv_path: str):
    _ensure_parent(csv_path)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_loss", "macro_f1","macro_acc","overall_acc", "lr_head", "lr_enc"])
        for i in range(len(history["train_loss"])):
            w.writerow([
                i + 1,
                history["train_loss"][i],
                history["val_loss"][i],
                history["macro_f1"][i],
                history["macro_acc"][i],
                history["overall_acc"][i],
                history["lr_head"][i],
                history["lr_enc"][i],
            ])

# ---- Head spec helpers ----

def _linear_shapes(mod: nn.Sequential):
    shapes = []
    for m in mod:
        if isinstance(m, nn.Linear):
            shapes.append((m.in_features, m.out_features))
    return shapes

def _module_counts(mod: nn.Sequential):
    n_linear = n_norm = n_act = n_drop = n_other = 0
    for m in mod:
        if isinstance(m, nn.Linear):
            n_linear += 1
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.GroupNorm)):
            n_norm += 1
        elif isinstance(m, (nn.ReLU, nn.GELU, nn.SiLU, nn.Tanh)):
            n_act += 1
        elif isinstance(m, nn.Dropout):
            n_drop += 1
        else:
            n_other += 1
    total = n_linear + n_norm + n_act + n_drop + n_other
    return dict(total=total, linear=n_linear, norm=n_norm, act=n_act, drop=n_drop, other=n_other)

def _dropout_p(mod: nn.Sequential) -> list[float]:
    ps = []
    for m in mod:
        if isinstance(m, nn.Dropout):
            ps.append(m.p)
    return ps

def _count_params_trainable(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def _log_and_save_head_spec(head, d_model, num_classes, hidden, device, backbone_kind):
    head_shapes = _linear_shapes(head.net)
    head_drops  = _dropout_p(head.net)
    head_params = _count_params_trainable(head)
    counts      = _module_counts(head.net)

    spec_txt = [
        "=== ClassifierHead specification ===",
        f"Backbone kind           : {backbone_kind}",
        f"input dim               : {d_model}",
        f"output dim              : {num_classes}",
        f"hidden (None=>linear)   : {hidden}",
        f"dropout rate(s)         : {head_drops if head_drops else '—'}",
        f"Layers (total)          : {counts['total']}",
        f"  • linear/norm/act/drop/other : "
        f"{counts['linear']}/{counts['norm']}/{counts['act']}/{counts['drop']}/{counts['other']}",
        f"Linear layer shapes     : {head_shapes}",
        f"Trainable params (head) : {head_params:,}",
    ]
    with torch.no_grad():
        _z = torch.randn(1, d_model, device=device)
        _out = head(_z)
        spec_txt.append(f"Forward check            : input (1,{d_model}) -> logits {_out.shape}")

    spec_str = "\n".join(spec_txt)
    print(spec_str)
    spec_path = "./artifacts/finetune/head_spec.txt"
    os.makedirs(os.path.dirname(spec_path), exist_ok=True)
    with open(spec_path, "w", encoding="utf-8") as f:
        f.write(spec_str + "\n")
    print(f"📝 Head spec saved to: {spec_path}")

    return {
        "linear_shapes": head_shapes,
        "dropouts": head_drops,
        "trainable_params": int(head_params),
        "forward_check": [int(_out.shape[0]), int(_out.shape[1])] if isinstance(_out, torch.Tensor) and _out.ndim == 2 else str(_out.shape),
    }

# ----------------------------
# Embedding collection/viz
# ----------------------------

def _get_Y(enc_out):
    if isinstance(enc_out, (tuple, list)):
        return enc_out[-1]
    return enc_out

def _vr_pool(Y: torch.Tensor, pad_mask: torch.Tensor, pooling: str = "cls") -> torch.Tensor:
    B, Ly, D = Y.shape
    Lm = pad_mask.shape[1]
    if str(pooling).lower() == "cls" and Ly == Lm + 1:
        return Y[:, 0, :]
    tokens = Y[:, 1:, :] if Ly == Lm + 1 else Y
    valid = (~pad_mask).float()
    s = (tokens * valid.unsqueeze(-1)).sum(dim=1)
    denom = valid.sum(dim=1).clamp(min=1.0).unsqueeze(-1)
    return s / denom

def _collect_embeddings(enc, head, loader, device, pooling, backbone_kind):
    enc.eval(); head.eval()
    zs, logits_all, y_all = [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            m = batch["mask"].to(device)
            y = batch["y"].to(device)
            if backbone_kind == "temporal":
                H = enc(x, m)
                z = enc.pool(H, m, pooling=pooling)
            else:
                Y = _get_Y(enc(x, m))
                z = _vr_pool(Y, m, pooling=pooling)
            lg = head(z)
            zs.append(z.detach().cpu().numpy())
            logits_all.append(lg.detach().cpu().numpy())
            y_all.append(y.detach().cpu().numpy())
    Z = np.concatenate(zs, axis=0)
    LOG = np.concatenate(logits_all, axis=0)
    Y = np.concatenate(y_all, axis=0)
    return Z, LOG, Y

def plot_embedding_maps(Z, Y, class_names, seed=42, out_path="./artifacts/finetune/embedding_maps.png"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    X = Z
    pca2  = PCA(n_components=2, random_state=seed).fit_transform(X)
    tsne2 = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=30, random_state=seed).fit_transform(X)
    umap2 = _UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=seed).fit_transform(X)

    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(class_names))]
    label_to_color = {i: colors[i] for i in range(len(class_names))}

    def _panel(ax, pts, title):
        for i, cname in enumerate(class_names):
            m = (Y == i)
            ax.scatter(pts[m, 0], pts[m, 1], s=8, alpha=0.8, label=cname, c=[label_to_color[i]])
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([]); ax.legend(markerscale=2, fontsize=8, frameon=False)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    _panel(axs[0], pca2, "PCA")
    _panel(axs[1], tsne2, "t-SNE")
    try:
        _panel(axs[2], umap2, "UMAP")
    except Exception as e:
        axs[2].text(0.5, 0.5, f"UMAP unavailable:\n{e}", ha="center", va="center")
        axs[2].set_axis_off()
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"🖼️ Saved embedding maps to: {out_path}")

# ----------------------------
# Classifier head
# ----------------------------

class ClassifierHead(nn.Module):
    """
    Two cases:
      - hidden=None: linear head
      - hidden=int: 1-layer MLP with LayerNorm between Linear and ReLU
    """
    def __init__(self, d_model: int, num_classes: int, *, hidden: int | None = None,
                 dropout: float = _cfg("DROPOUT_HEAD", 0.3)):
        super().__init__()
        if hidden is None:
            self.net = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(d_model, num_classes),
            )
        else:
            fc1   = nn.Linear(d_model, hidden)
            norm1 = nn.LayerNorm(hidden)
            act1  = nn.ReLU()
            drop1 = nn.Dropout(dropout)
            fc2   = nn.Linear(hidden, num_classes)

            nn.init.xavier_uniform_(fc1.weight, gain=nn.init.calculate_gain('relu'));
            nn.init.zeros_(fc1.bias)
            nn.init.xavier_uniform_(fc2.weight, gain=1.0);
            nn.init.zeros_(fc2.bias)

            self.net = nn.Sequential(fc1, norm1, act1, drop1, fc2)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

# ----------------------------
# Focal loss (optional)
# ----------------------------

def _focal_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, gamma: float, alpha: torch.Tensor | float | None,
) -> torch.Tensor:
    ce = F.cross_entropy(logits, targets, weight=None, reduction="none")
    with torch.no_grad():
        pt = torch.exp(-ce)
    mod = (1.0 - pt) ** gamma
    if alpha is None:
        alpha_term = 1.0
    elif isinstance(alpha, float):
        alpha_term = alpha
    else:
        alpha = alpha.to(logits.device)
        alpha_term = alpha[targets]
    loss = mod * ce * alpha_term
    return loss.mean()

# ----------------------------
# One epoch (train or eval)
# ----------------------------

def _run_epoch(
    enc: nn.Module,
    head: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    class_weights: torch.Tensor | None,
    loss_type: str,
    use_class_weights_ce: bool,
    focal_gamma: float,
    focal_alpha: torch.Tensor | float | None,
    pooling: str,
    backbone_kind: str,
    on_step=None,
    grad_clip = None,
) -> Tuple[float, np.ndarray, np.ndarray]:
    train_mode = optimizer is not None
    enc.train(train_mode)
    head.train(train_mode)

    total_loss = 0.0
    n_batches = 0
    y_true_list: List[int] = []
    y_pred_list: List[int] = []

    for batch in loader:
        x = batch["x"].to(device)
        m = batch["mask"].to(device)
        y = batch["y"].to(device)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        if backbone_kind == "temporal":
            H = enc(x, m)
            z = enc.pool(H, m, pooling=pooling)
            logits = head(z)
        else:
            Y = _get_Y(enc(x, m))
            z = _vr_pool(Y, m, pooling=pooling)
            logits = head(z)

        if loss_type == "weighted_ce":
            if use_class_weights_ce and class_weights is not None:
                loss = F.cross_entropy(logits, y, weight=class_weights.to(device))
            else:
                loss = F.cross_entropy(logits, y)
        elif loss_type == "focal":
            loss = _focal_cross_entropy(logits, y, gamma=focal_gamma, alpha=focal_alpha)
        else:
            raise ValueError(f"Unknown LOSS_TYPE: {loss_type}")

        if train_mode:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(head.parameters()), grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            if on_step is not None:
                on_step()    # this advances the scheduler once per batch

        total_loss += float(loss.detach().cpu())
        n_batches += 1

        preds = logits.argmax(dim=-1).detach().cpu().numpy()
        y_true_list.extend(y.detach().cpu().numpy().tolist())
        y_pred_list.extend(preds.tolist())

    avg_loss = total_loss / max(1, n_batches)
    return avg_loss, np.array(y_true_list), np.array(y_pred_list)

# ----------------------------
# Main API
# ----------------------------

def run_supervised_finetune(
    dataset: LabeledWindowsDataset,
    class_to_idx: Dict[str, int],
    ds_val: LabeledWindowsDataset
):
    # ---- Config snapshot ----
    batch_size   = _cfg("BATCH_SIZE_FT", _cfg("BATCH_SIZE", 32))
    epochs       = _cfg("FINETUNE_EPOCHS", _cfg("EPOCHS", 30))
    lr_head      = _cfg("LR_FT_HEAD", _cfg("LR_FT", _cfg("LR", 5e-5)))
    lr_enc       = _cfg("LR_FT_ENC",  _cfg("LR_FT", _cfg("LR", 5e-5)))
    weight_decay = _cfg("WEIGHT_DECAY", _cfg("WD_PRE_FT", 0.01))
    grad_clip    = _cfg("GRAD_CLIP_NORM", 1.0)
    scheduler_kind = str(_cfg("SCHEDULER", "cosine")).lower()
    scheduler_enabled = scheduler_kind not in (None, "", "none", "off", False, 0)
    warmup_frac  = float(_cfg("WARMUP_FRAC", 0.06))
    pooling      = CATTN.POOLING
    val_ratio    = _cfg("VAL_RATIO", 0.2)
    seed         = _cfg("SEED", 42)
    freeze_epochs= _cfg("FREEZE_EPOCHS", 0)
    save_path    = _cfg("CLF_CKPT_PATH", "./emotion_classifier.pt")
    DROPOUT_HEAD = _cfg("DROPOUT_HEAD", 0.3)

    plot_path  = _cfg("FINETUNE_PLOT_PATH", "./artifacts/finetune/finetune_loss.png")
    csv_path   = _cfg("FINETUNE_HISTORY_CSV", "./artifacts/finetune/finetune_history.csv")
    ma_window  = _cfg("STABILIZE_MA_WINDOW", 3)
    ma_rtol    = _cfg("STABILIZE_REL_TOL", 1e-3)

    use_pretrained       = CATTN.USE_PRETRAINED
    loss_type            = _cfg("LOSS_TYPE", "weighted_ce")
    use_class_weights_ce = _cfg("USE_CLASS_WEIGHTS_CE", True)

    d_model  = CATTN.ATTN_CONFIG["d_model"]
    n_layers = CATTN.ATTN_CONFIG["num_layers"]
    n_heads  = CATTN.ATTN_CONFIG["n_heads"]
    max_len  = CATTN.ATTN_CONFIG["max_len"]

    # Backbone switch
    _raw_kind = str(_cfg("FINETUNE_BACKBONE", "temporal")).strip().lower()
    if _raw_kind in {"vrtransformer", "transformer", "tfm"}:
        backbone_kind = "vrtransformer"
    elif _raw_kind == "temporal":
        backbone_kind = "temporal"
    else:
        raise ValueError(f"Unknown FINETUNE_BACKBONE: {CFG.FINETUNE_BACKBONE}")

    _set_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ---- Build encoder
    if backbone_kind == "temporal":
        temporal_ckpt = _cfg("PRETRAIN_CKPT_PATH", "encoder_pretrained.pt")
        if use_pretrained:
            if not (temporal_ckpt and os.path.isfile(temporal_ckpt)):
                raise FileNotFoundError(
                    f"USE_PRETRAINED=True but checkpoint not found at: {temporal_ckpt}. "
                    f"Run MODE=pretrain first or set USE_PRETRAINED=False."
                )
            ckpt = torch.load(temporal_ckpt, map_location="cpu")
            conf = ckpt.get("config", {})
            d_model  = conf.get("d_model",  d_model)
            n_layers = conf.get("n_layers", conf.get("num_layers", n_layers))
            n_heads  = conf.get("n_heads",  n_heads)
            max_len  = conf.get("max_len",  max_len)

            enc = TemporalEncoder(d_model=d_model, n_layers=n_layers, n_heads=n_heads, max_len=max_len).to(device)
            enc.load_state_dict(ckpt["state_dict"], strict=True)
            print(f"📦 Loaded pretrained TemporalEncoder from {temporal_ckpt}")
        else:
            enc = TemporalEncoder(d_model=d_model, n_layers=n_layers, n_heads=n_heads, max_len=max_len).to(device)
            print("🆕 TemporalEncoder: starting from random init.")
    else:
        enc = TransformerEncoder().to(device)
        if use_pretrained:
            base = _cfg("PRETRAIN_CKPT_PATH", "encoder_pretrained.pt")
            vr_ckpt_path = base.replace(".pt", "_vr.pt") if base.endswith(".pt") else (base + "_vr.pt")
            if not os.path.isfile(vr_ckpt_path):
                raise FileNotFoundError(
                    f"USE_PRETRAINED=True but VR checkpoint not found at: {vr_ckpt_path}. "
                    f"Run MODE=pretrain (VR) first or set USE_PRETRAINED=False."
                )
            vr_ckpt = torch.load(vr_ckpt_path, map_location="cpu")
            missing, unexpected = enc.load_state_dict(vr_ckpt["state_dict"], strict=False)
            enc.to(device)
            print(f"📦 Loaded VR encoder from {vr_ckpt_path} (missing={len(missing)}, unexpected={len(unexpected)})")
        else:
            print("🆕 VRTransformer: starting from random init.")
        print("🧱 VRTransformer backbone ready for fine-tune.")

    # ---- Build head
    num_classes = len(class_to_idx)
    ratio  = _cfg("HEAD_HIDDEN_RATIO", None)
    hidden = None if (ratio is None or ratio == 0) else int(ratio * CATTN.ATTN_CONFIG["d_model"])
    head   = ClassifierHead(d_model, num_classes, hidden=hidden, dropout=DROPOUT_HEAD).to(device)

    # ---- Log & persist head spec
    head_spec_dict = _log_and_save_head_spec(
        head=head, d_model=d_model, num_classes=num_classes, hidden=hidden, device=device, backbone_kind=backbone_kind
    )

    # ---- Dataloaders
    if ds_val is None:
        train_idx, val_idx = _train_val_split(len(dataset), val_ratio=val_ratio, seed=seed)
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True,
                                  collate_fn=labeled_collate_fn, pin_memory=torch.cuda.is_available())
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False,
                                collate_fn=labeled_collate_fn, pin_memory=torch.cuda.is_available())
        train_subset = Subset(dataset, train_idx)
        class_weights = _compute_class_weights(train_subset.dataset, class_to_idx)
    else:
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                  collate_fn=labeled_collate_fn, pin_memory=torch.cuda.is_available())
        val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False,
                                collate_fn=labeled_collate_fn, pin_memory=torch.cuda.is_available())
        class_weights = _compute_class_weights(dataset, class_to_idx)

    print(f"⚖️  Class weights: {class_weights.tolist()}")
    #New code for Solving LR issue
    # ---- Optimizer + scheduler (two LRs with warmup & cosine) ----
    def _trainable(params):
        return [p for p in params if p.requires_grad]

    # 1) Freeze encoder params first
    for p in enc.parameters():
        p.requires_grad = False

    head_params = list(_trainable(head.parameters()))
    enc_params = list(enc.parameters())  # <-- DO NOT filter here; keep them in the optimizer!

    optimizer = torch.optim.AdamW(
        [
            {"params": head_params, "lr": lr_head, "weight_decay": weight_decay},
            {"params": enc_params, "lr": 0.0, "weight_decay": weight_decay},  # frozen but present
        ]
    )

    # Store per-group base LRs for warmup scaling
    optimizer.param_groups[0]["base_lr"] = lr_head
    optimizer.param_groups[1]["base_lr"] = 0.0  # while frozen; will change at unfreeze

    # 2) Scheduler setup
    total_steps = max(1, epochs * len(train_loader))
    warm_steps = int(max(1, warmup_frac * total_steps)) if scheduler_enabled else 0

    cos = (torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps - warm_steps)
    ) if scheduler_enabled else None)

    # We’ll start cosine *after* warmup by resetting last_epoch to -1 later.

    def step_scheduler(global_step: int):
        if not scheduler_enabled:
            return

        if global_step < warm_steps:
            # Linear warmup toward each group's base_lr
            scale = float(global_step + 1) / float(warm_steps)
            for g in optimizer.param_groups:
                g["lr"] = g["base_lr"] * scale
        else:
            # First time we cross warmup boundary, re-init cosine epoch
            if cos.last_epoch == -1:
                # Ensures cosine starts fresh from current LRs
                pass
            cos.step()

    # 3) Unfreeze at the start of the unfreeze epoch
    def unfreeze_encoder_now():
        for p in enc.parameters():
            p.requires_grad = True

        optimizer.param_groups[1]["base_lr"] = lr_enc
        optimizer.param_groups[1]["lr"] = lr_enc  # set unconditionally

        if cos is not None:
            cos.base_lrs = [optimizer.param_groups[0]["base_lr"],
                            optimizer.param_groups[1]["base_lr"]]
            # optional: restart cosine after unfreeze
            # cos.last_epoch = -1

    #End of Solving LR issue

    print(f"🧪 Fine-tuning ({backbone_kind}) | freeze {freeze_epochs} epoch(s) | "
          f"params(enc total)={_count_params(enc):,} | params(head)={_count_params(head):,}")
    print(f"   LRs — head: {lr_head} | enc: {lr_enc}")

    # ---- Confusion tracking
    class_names = [k for k, _ in sorted(class_to_idx.items(), key=lambda kv: kv[1])]
    tracker = ConfusionTracker(class_names=class_names, out_dir="./artifacts/finetune/cm_plots",
                               normalize=True, keep_history=False)

    # Baseline (epoch 0)
    if CATTN.SAVE_BASELINE:
        enc.eval(); head.eval()
        y_true0: List[int] = []
        y_pred0: List[int] = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                m = batch["mask"].to(device)
                y = batch["y"].to(device)
                if backbone_kind == "temporal":
                    H = enc(x, m); z = enc.pool(H, m, pooling=pooling); logits = head(z)
                else:
                    Y = _get_Y(enc(x, m)); z = _vr_pool(Y, m, pooling=pooling); logits = head(z)
                preds = logits.argmax(dim=-1).detach().cpu().tolist()
                y_true0.extend(y.detach().cpu().tolist()); y_pred0.extend(preds)
        cm0 = compute_confusion_matrix(np.array(y_true0), np.array(y_pred0), num_classes)
        tracker.save_baseline(cm0)
        print("📊 Baseline confusion matrix saved (epoch 0).")
    else:
        print("⏭️ Skipping baseline confusion matrix (SAVE_BASELINE=False).")

    # ---- History containers

    history = {
        "train_loss": [],
        "val_loss": [],
        "macro_f1": [],
        "macro_acc": [],
        "overall_acc": [],
        "lr_head": [],
        "lr_enc": [],
        "per_class_f1": {name: [] for name in class_names},
        "per_class_acc": {name: [] for name in class_names},  # ✅ added per-class accuracy
    }

    # --- Focal configuration ---
    alpha_mode = _cfg("FOCAL_ALPHA_MODE", None)
    alpha_norm = _cfg("FOCAL_ALPHA_NORMALIZE", True)
    focal_gamma = _cfg("FOCAL_GAMMA", 2.0)

    focal_alpha = None
    if loss_type == "focal":
        if alpha_mode == "class_weights":
            vec = class_weights.clone().detach().float()
            focal_alpha = vec / vec.sum() if alpha_norm and vec.sum() > 0 else vec
        elif alpha_mode == "scalar":
            focal_alpha = float(_cfg("FOCAL_ALPHA_SCALAR", 0.25))
        elif alpha_mode == "vector":
            user_vec = _cfg("FOCAL_ALPHA_VECTOR", None)
            if user_vec is None:
                raise ValueError("FOCAL_ALPHA_MODE='vector' requires FOCAL_ALPHA_VECTOR in config.")
            vec = torch.tensor(user_vec, dtype=torch.float32)
            if vec.numel() != len(class_to_idx):
                raise ValueError(f"FOCAL_ALPHA_VECTOR length {vec.numel()} != num_classes {len(class_to_idx)}.")
            focal_alpha = vec / vec.sum() if alpha_norm and vec.sum() > 0 else vec

    # Friendly loss print
    if loss_type == "focal":
        print(f"🔧 Loss=focal | gamma={focal_gamma} | alpha_mode={alpha_mode} | "
              f"alpha={'vector' if isinstance(focal_alpha, torch.Tensor) else focal_alpha}")
    else:
        print(f"🔧 Loss=weighted_ce | use_class_weights={use_class_weights_ce} | classes={len(class_to_idx)}")

    # ---- Training loop ----
    best_macro_f1 = -1.0
    global_step   = 0   # keep cumulative across all epochs
    t0 = time.time()
    last_cm = None

    for ep in range(1, epochs + 1):
        # Unfreeze encoder params at the chosen epoch (no optimizer rebuild)
        if ep == freeze_epochs + 1:
            unfreeze_encoder_now()
            print(f"After unfreeze LRs at epoch {ep}- NEW LRs:", [g["lr"] for g in optimizer.param_groups])
        # Train
        def on_step():
            nonlocal global_step
            step_scheduler(global_step)   # per-batch schedule
            global_step += 1

        train_loss, _, _ = _run_epoch(
            enc, head, train_loader, device, optimizer, class_weights,
            loss_type=loss_type, use_class_weights_ce=use_class_weights_ce,
            focal_gamma=focal_gamma, focal_alpha=focal_alpha,
            pooling=pooling, backbone_kind=backbone_kind, on_step=on_step,grad_clip=grad_clip
        )

        # Validate
        enc.eval(); head.eval()
        with torch.no_grad():
            val_loss, y_true, y_pred = _run_epoch(
                enc, head, val_loader, device, optimizer=None, class_weights=class_weights,
                loss_type=loss_type, use_class_weights_ce=use_class_weights_ce,
                focal_gamma=focal_gamma, focal_alpha=focal_alpha,
                pooling=pooling, backbone_kind=backbone_kind
            )

        cm = compute_confusion_matrix(y_true, y_pred, num_classes)
        last_cm = cm
        prec, rec, f1, macro_f1 = precision_recall_f1_from_cm(cm)
        acc_per_class, macro_acc,overall_acc = accuracy_from_cm(cm)

        # Track epoch-wise metrics + LRs of both groups
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["macro_f1"].append(float(macro_f1))
        history["macro_acc"].append(float(macro_acc))
        history["overall_acc"].append(float(overall_acc))
        head_lr = optimizer.param_groups[0]["lr"]
        enc_lr  = optimizer.param_groups[1]["lr"]
        history["lr_head"].append(head_lr)
        history["lr_enc"].append(enc_lr)

        f1_per_class = _f1_per_class_from_cm(cm)
        for i, cname in enumerate(class_names):
            history["per_class_f1"][cname].append(float(f1_per_class[i]))
            history["per_class_acc"][cname].append(float(acc_per_class[i]))

        dt = time.time() - t0
        print(
            f"Epoch {ep:03d}/{epochs} | "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"macroF1={macro_f1:.4f}  macroAcc={macro_acc:.4f} | "
            f"LRs(h,e)=({head_lr:.2e},{enc_lr:.2e}) | {dt / 60:.1f} min"
        )

        print("Per-class F1:", {name: f"{history['per_class_f1'][name][-1]:.3f}" for name in class_names})

        tracker.update_if_best(cm, macro_f1, ep)

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            _ensure_parent(save_path)
            torch.save(
                {
                    "encoder": enc.state_dict(),
                    "head": head.state_dict(),
                    "config": {
                        "backbone": backbone_kind,
                        "d_model": d_model,
                        "n_layers": n_layers,
                        "n_heads": n_heads,
                        "max_len": max_len,
                        "pooling": pooling,
                        "class_to_idx": class_to_idx,
                        "head_hidden": hidden,
                    },
                    "head_spec": head_spec_dict,
                    "metrics": {
                        "macro_f1": float(macro_f1),
                        "confusion_matrix": cm.tolist(),
                    }
                },
                save_path
            )
            print(f"💾 Saved best classifier to {save_path} (macroF1 {best_macro_f1:.4f})")

    # Final snapshot
    if last_cm is not None:
        tracker.save_final(last_cm)

    # ---- Class-weight vs per-class F1/Recall (final) ----
    if last_cm is not None:
        out_png = "./artifacts/finetune/class_weight_vs_f1_recall_accuracy(Final).png"
        out_csv = "./artifacts/finetune/class_weight_vs_f1_recall_accuracy(Final).csv"
        prec, rec, f1, _ = precision_recall_f1_from_cm(last_cm)
        acc, _ , _ = accuracy_from_cm(last_cm)
        _save_weights_vs_metrics_plot(
            class_names=class_names,
            class_weights=class_weights.tolist(),
            f1_per_cls=f1.tolist(),
            rec_per_cls=rec.tolist(),
            acc_per_cls=acc.tolist(),
            out_png=out_png,
            out_csv=out_csv,
        )
        print(f"📁 Saved: {out_png}")
        print(f"🧾 Saved: {out_csv}")

    # ---- Curves + CSV + stabilization heuristic ----
    _plot_finetune_history(history, plot_path)
    _write_history_csv(history, csv_path)

    _plot_f1_per_class_over_epochs(
        history=history,
        class_names=class_names,
        class_weights=np.asarray(class_weights.cpu() if torch.is_tensor(class_weights) else class_weights, dtype=float),
        out_path="./artifacts/finetune/f1_per_class_over_epochs.png",
    )

    # Embedding panels on VAL
    Z, LOG, Y = _collect_embeddings(enc, head, val_loader, device, pooling, backbone_kind)
    plot_embedding_maps(Z, Y, class_names, seed=seed, out_path="./artifacts/finetune/embedding_maps.png")

    # Stabilization heuristic
    def _moving_avg(xs, k=3):
        if k <= 1 or len(xs) < k: return xs[:]
        out = []
        for i in range(len(xs)):
            a = max(0, i - k + 1)
            out.append(sum(xs[a:i+1]) / (i - a + 1))
        return out

    def _detect_stabilization(losses: list, window: int, rel_tol: float) -> int:
        if len(losses) < 2: return len(losses)
        ma = _moving_avg(losses, k=window)
        for i in range(1, len(ma)):
            prev, cur = ma[i-1], ma[i]
            if prev != 0 and abs(cur - prev) / abs(prev) < rel_tol:
                return i + 1
        return len(losses)

    stab_epoch = _detect_stabilization(history["val_loss"], window=ma_window, rel_tol=ma_rtol)
    print(f"📉 Fine-tune loss plot saved to: {plot_path}")
    print(f"📁 Epoch history saved to: {csv_path}")
    print(f"🧭 Loss stabilization (heuristic): around epoch {stab_epoch} (MA window={ma_window}, rel tol={ma_rtol})")
    print(f"✅ Fine-tuning finished. Backbone: {backbone_kind} | Using pretrained: {'Yes' if CATTN.USE_PRETRAINED else 'No'}")
