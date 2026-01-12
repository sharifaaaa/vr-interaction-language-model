# train_pretrain.py
from __future__ import annotations
from typing import Dict, Tuple, Optional, Callable
import math
import time
import inspect
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
from pathlib import Path
import config_attention as CA
import config_pretrain as CP
from pretrain_dataset import pretrain_collate_fn
from vr_transformer.transformer_blocks import TransformerEncoder


# ============================================================
# Positional encoding + simple temporal encoder (baseline path)
# ============================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # [max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        L = x.size(1)
        return x + self.pe[:L]


class TemporalEncoder(nn.Module):
    """
    Transformer encoder with a learned CLS token and sinusoidal positional encoding.
    """
    def __init__(self, d_model: int, n_layers: int, n_heads: int, max_len: int, dropout: float = 0.1):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.pos = PositionalEncoding(d_model, max_len + 1)  # +1 for CLS

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, D]
        pad_mask: [B, L]  (True for PAD)
        returns: h [B, L+1, D]  (including CLS at index 0)
        """
        B, L, D = x.shape
        cls_tok = self.cls.expand(B, 1, D)
        x_in = torch.cat([cls_tok, x], dim=1)          # [B, L+1, D]
        x_in = self.pos(x_in)                           # add sin/cos pos enc

        # key padding mask for transformer: [B, L+1]
        pad = torch.ones((B, L + 1), dtype=torch.bool, device=x.device)
        pad[:, 0] = False                               # CLS never padded
        pad[:, 1:] = pad_mask

        h = self.encoder(x_in, src_key_padding_mask=pad)
        return h

    def pool(self, h: torch.Tensor, pad_mask: torch.Tensor, pooling: str = "cls") -> torch.Tensor:
        """
        h: [B, L+1, D]; pad_mask: [B, L]
        returns: [B, D]
        """
        if pooling == "cls":
            return h[:, 0, :]
        elif pooling == "mean":
            valid = (~pad_mask).float()                 # [B, L]
            s = (h[:, 1:, :] * valid.unsqueeze(-1)).sum(dim=1)
            denom = valid.sum(dim=1).clamp(min=1.0).unsqueeze(-1)
            return s / denom
        else:
            return h[:, 0, :]


# ===================
# Heads / loss pieces
# ===================

class MTMHead(nn.Module):
    """Predict original embedded token at masked positions (simple MLP)."""
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, h_tokens: torch.Tensor) -> torch.Tensor:
        return self.proj(h_tokens)


class ContrastiveHead(nn.Module):
    """Small projector + L2 normalization for InfoNCE."""
    def __init__(self, d_model: int, proj_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, proj_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.net(z)
        z = F.normalize(z, dim=-1)
        return z


def info_nce(z1: torch.Tensor, z2: torch.Tensor, tau: float) -> torch.Tensor:
    """
    Symmetric InfoNCE with cosine-sim via normalized z's.
    z1, z2: [B, D]
    """
    sim = (z1 @ z2.t()) / tau  # [B, B]
    labels = torch.arange(z1.size(0), device=z1.device)
    loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2
    return loss


# =========
# Utilities
# =========

def _count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _get_Y(enc_out):
    """Accept either Y or (scores, weights, Y). Returns Y."""
    if isinstance(enc_out, (tuple, list)):
        return enc_out[-1]
    return enc_out


def _pool_sequence(Y: torch.Tensor, pad_mask: torch.Tensor, pooling: str = "cls") -> torch.Tensor:
    """
    Y: [B, L or L+1, D], pad_mask: [B, L] (True=PAD).
    If Y has CLS (L+1), use CLS for 'cls' pooling; else mean over valid tokens.
    """
    B, L_y, D = Y.shape
    L_m = pad_mask.shape[1]
    if pooling == "cls" and L_y == L_m + 1:
        return Y[:, 0, :]

    # mean over tokens (drop CLS if present)
    Y_tokens = Y[:, 1:, :] if L_y == L_m + 1 else Y
    valid = (~pad_mask).float()
    s = (Y_tokens * valid.unsqueeze(-1)).sum(dim=1)
    denom = valid.sum(dim=1).clamp(min=1.0).unsqueeze(-1)
    return s / denom


def _encode_Y(enc: nn.Module, x: torch.Tensor, mask: torch.Tensor, vr_mode: bool) -> torch.Tensor:
    """Unify encoder output across temporal and VR paths."""
    if vr_mode:
        return _get_Y(enc(x, mask))
    else:
        return enc(x, mask)  # TemporalEncoder returns [B, L+1, D]


def _grad_norm(parameters, norm_type: float = 2.0) -> float:
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += float(param_norm**2)
    return float(total_norm**0.5)


# ==================================
# Shared training / validation steps
# ==================================

def train_one_epoch(
    enc: nn.Module,
    mtm_head: Optional[nn.Module],
    con_head: Optional[nn.Module],
    loader: DataLoader,
    device: torch.device,
    objectives: Dict[str, bool],
    lambdas: Dict[str, float],
    temperature: float,
    optimizer: torch.optim.Optimizer,
    scheduler_step_fn: Callable[[int], None],
    grad_clip: Optional[float],
    vr_mode: bool,
    pooling: str,
) -> Dict[str, float]:
    enc.train()
    if mtm_head: mtm_head.train()
    if con_head: con_head.train()

    totals = {"mtm": 0.0, "con": 0.0, "total": 0.0, "grad_norm": 0.0}
    n_batches = 0
    global_step = 0  # local per-epoch step for scheduler stepping

    for batch in loader:
        optimizer.zero_grad(set_to_none=True)

        # Move to device
        mtm_input     = batch["mtm_input"].to(device)      # [B,L,D]
        mtm_target    = batch["mtm_target"].to(device)     # [B,L,D]
        mtm_mask      = batch["mtm_mask"].to(device)       # [B,L] bool
        attn_mask_mtm = batch["attn_mask_mtm"].to(device)  # [B,L] bool

        v1_x   = batch["v1_x"].to(device)
        v1_mask= batch["v1_mask"].to(device)
        v2_x   = batch["v2_x"].to(device)
        v2_mask= batch["v2_mask"].to(device)

        total_loss = torch.zeros((), device=device)

        for key in ("mtm_input", "mtm_target", "v1_x", "v2_x"):
            batch[key] = torch.nan_to_num(batch[key], nan=0.0, posinf=0.0, neginf=0.0)

        # --- MTM objective ---
        if objectives.get("mtm", False):
            Y_mtm = _encode_Y(enc, mtm_input, attn_mask_mtm, vr_mode=vr_mode)  # [B,L or L+1,D]
            B = Y_mtm.size(0)
            L = mtm_mask.size(1)
            D = Y_mtm.size(-1)

            # Align to token length (drop CLS if present)
            if Y_mtm.size(1) == L + 1:
                Y_tokens = Y_mtm[:, 1:, :]
            else:
                Y_tokens = Y_mtm

            if mtm_mask.view(B * L).any():
                pred = mtm_head(Y_tokens).view(B * L, D)[mtm_mask.view(B * L)]
                targ = mtm_target.view(B * L, D)[mtm_mask.view(B * L)]
                mtm_loss = F.mse_loss(pred, targ)
                total_loss = total_loss + lambdas.get("mtm", 1.0) * mtm_loss
                totals["mtm"] += float(mtm_loss.detach().cpu())

        # --- Contrastive objective ---
        if objectives.get("contrastive", False):
            Y1 = _encode_Y(enc, v1_x, v1_mask, vr_mode=vr_mode)
            Y2 = _encode_Y(enc, v2_x, v2_mask, vr_mode=vr_mode)
            z1 = _pool_sequence(Y1, v1_mask, pooling=pooling)
            z2 = _pool_sequence(Y2, v2_mask, pooling=pooling)
            z1 = con_head(z1)
            z2 = con_head(z2)
            con_loss = info_nce(z1, z2, temperature)
            total_loss = total_loss + lambdas.get("contrastive", 1.0) * con_loss
            totals["con"] += float(con_loss.detach().cpu())

        # Backward + step
        total_loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(enc.parameters(), max_norm=grad_clip)
            if mtm_head:
                torch.nn.utils.clip_grad_norm_(mtm_head.parameters(), max_norm=grad_clip)
            if con_head:
                torch.nn.utils.clip_grad_norm_(con_head.parameters(), max_norm=grad_clip)

        # grad norm (after clipping for stability)
        all_params = list(enc.parameters())
        if mtm_head: all_params += list(mtm_head.parameters())
        if con_head: all_params += list(con_head.parameters())
        totals["grad_norm"] += _grad_norm(all_params)

        optimizer.step()
        scheduler_step_fn(global_step)
        global_step += 1

        totals["total"] += float(total_loss.detach().cpu())
        n_batches += 1

    for k in ("mtm", "con", "total", "grad_norm"):
        totals[k] /= max(1, n_batches)
    return totals


@torch.no_grad()
def evaluate_one_epoch(
    enc: nn.Module,
    mtm_head: Optional[nn.Module],
    con_head: Optional[nn.Module],
    loader: DataLoader,
    device: torch.device,
    objectives: Dict[str, bool],
    lambdas: Dict[str, float],
    temperature: float,
    vr_mode: bool,
    pooling: str,
) -> Dict[str, float]:
    enc.eval()
    if mtm_head: mtm_head.eval()
    if con_head: con_head.eval()

    totals = {"mtm": 0.0, "con": 0.0, "total": 0.0}
    n_batches = 0

    for batch in loader:
        # Move to device
        mtm_input     = batch["mtm_input"].to(device)
        mtm_target    = batch["mtm_target"].to(device)
        mtm_mask      = batch["mtm_mask"].to(device)
        attn_mask_mtm = batch["attn_mask_mtm"].to(device)

        v1_x   = batch["v1_x"].to(device)
        v1_mask= batch["v1_mask"].to(device)
        v2_x   = batch["v2_x"].to(device)
        v2_mask= batch["v2_mask"].to(device)

        total_loss = torch.zeros((), device=device)
        # train_pretrain.py, in evalaute_one_epoch(...) right after you move to device:
        for key in ("mtm_input", "mtm_target", "v1_x", "v2_x"):
            batch[key] = torch.nan_to_num(batch[key], nan=0.0, posinf=0.0, neginf=0.0)

        # MTM
        if objectives.get("mtm", False):
            Y = _encode_Y(enc, mtm_input, attn_mask_mtm, vr_mode=vr_mode)
            B = Y.size(0); L = mtm_mask.size(1); D = Y.size(-1)
            if Y.size(1) == L + 1:
                Y_tokens = Y[:, 1:, :]
            else:
                Y_tokens = Y

            if mtm_mask.view(B * L).any():
                pred = mtm_head(Y_tokens).view(B * L, D)[mtm_mask.view(B * L)]
                targ = mtm_target.view(B * L, D)[mtm_mask.view(B * L)]
                mtm_loss = F.mse_loss(pred, targ)
                total_loss = total_loss + lambdas.get("mtm", 1.0) * mtm_loss
                totals["mtm"] += float(mtm_loss.detach().cpu())

        # Contrastive
        if objectives.get("contrastive", False):
            Y1 = _encode_Y(enc, v1_x, v1_mask, vr_mode=vr_mode)
            Y2 = _encode_Y(enc, v2_x, v2_mask, vr_mode=vr_mode)
            z1 = _pool_sequence(Y1, v1_mask, pooling=pooling)
            z2 = _pool_sequence(Y2, v2_mask, pooling=pooling)
            z1, z2 = con_head(z1), con_head(z2)
            con_loss = info_nce(z1, z2, temperature)
            total_loss = total_loss + lambdas.get("contrastive", 1.0) * con_loss
            totals["con"] += float(con_loss.detach().cpu())

        totals["total"] += float(total_loss.detach().cpu())
        n_batches += 1

    for k in totals:
        totals[k] /= max(1, n_batches)
    return totals


# ===========================
# Helpers: plots & heuristics
# ===========================

def _plot_losses(history: Dict[str, list], out_path: str):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    epochs = range(1, len(history["train_total"]) + 1)

    plt.figure()
    plt.plot(epochs, history["train_total"], label="train_total")
    if "val_total" in history:
        plt.plot(epochs, history["val_total"], label="val_total")
    if "ema_total" in history:
        plt.plot(epochs, history["ema_total"], label="ema_total")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Pretraining Loss")
    plt.tight_layout()
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        p.unlink()
    plt.savefig(out_path)
    plt.close()


def _moving_avg(xs, k=3):
    if k <= 1 or len(xs) < k:
        return xs[:]
    out = []
    for i in range(len(xs)):
        a = max(0, i - k + 1)
        out.append(sum(xs[a:i+1]) / (i - a + 1))
    return out


def _detect_stabilization(losses: list, window: int, rel_tol: float) -> int:
    """
    Heuristic: first epoch t where |MA_t - MA_{t-1}| / MA_{t-1} < rel_tol
    using moving average with given 'window'. Returns 1-based epoch index, or len(losses) if none.
    """
    if len(losses) < 2:
        return len(losses)
    ma = _moving_avg(losses, k=window)
    for i in range(1, len(ma)):
        prev = ma[i-1]
        cur  = ma[i]
        if prev != 0 and abs(cur - prev) / abs(prev) < rel_tol:
            return i + 1  # epochs are 1-based
    return len(losses)

# =====================================
# Temporal pretraining (DB-aware splits)
# =====================================

def run_temporal_pretraining(
    train_dataset,
    val_dataset=None,   # <- pass your DB 'val' dataset here to disable random_split
    used_stratified: bool = False,
):
    """
    Self-supervised pretraining with MTM and/or contrastive InfoNCE.

    If `val_dataset` is provided, it's used directly (DB-driven split).
    Otherwise we fall back to an internal random split of `train_dataset`.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # --- build loaders (DB-driven if given) ---
    if val_dataset is None:
        val_fraction = getattr(CP, "VAL_FRACTION", 0.1)
        n_total = len(train_dataset)
        n_val   = max(1, int(n_total * val_fraction))
        n_train = n_total - n_val
        train_ds, val_ds = random_split(train_dataset, [n_train, n_val])
    else:
        train_ds, val_ds = train_dataset, val_dataset

    train_loader = DataLoader(
        train_ds, batch_size=CP.BATCH_SIZE_PRE, shuffle=True,
        collate_fn=pretrain_collate_fn, num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_ds, batch_size=CP.BATCH_SIZE_PRE, shuffle=False,
        collate_fn=pretrain_collate_fn, num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    # --- configs ---
    objectives   = CP.OBJECTIVES
    lambdas      = CP.LAMBDA
    temperature  = CP.TEMPERATURE
    epochs       = CP.PRETRAIN_EPOCHS
    lr           = CP.LR_PRE
    weight_decay = CP.WEIGHT_DECAY
    grad_clip    = CP.GRAD_CLIP_NORM
    warmup_frac  = CP.WARMUP_FRAC
    patience     = CP.EARLY_STOP_PATIENCE   # 0 disables
    ema_alpha    = CP.EMA_ALPHA            # 0 disables
    plot_path    = CP.PRETRAIN_PLOT_PATH
    stab_window  = CP.STABILIZE_MA_WINDOW
    stab_tol     = CP.STABILIZE_REL_TOL

    d_model   = CA.ATTN_CONFIG["d_model"]
    n_layers  = CA.ATTN_CONFIG["num_layers"]
    n_heads   = CA.ATTN_CONFIG["n_heads"]
    max_len   = CA.ATTN_CONFIG["max_len"]
    dropout   = CA.ATTN_CONFIG.get("dropout", 0.1)
    save_path = CP.PRETRAIN_CKPT_PATH
    pooling   = CA.POOLING

    # --- model + heads ---
    enc = TemporalEncoder(d_model=d_model, n_layers=n_layers, n_heads=n_heads, max_len=max_len, dropout=dropout).to(device)
    mtm_head = MTMHead(d_model).to(device) if objectives.get("mtm", False) else None
    con_head = ContrastiveHead(d_model).to(device) if objectives.get("contrastive", False) else None

    params = list(enc.parameters())
    if mtm_head is not None: params += list(mtm_head.parameters())
    if con_head is not None: params += list(con_head.parameters())

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    total_steps = max(1, epochs * len(train_loader))
    warm_steps  = int(max(1, warmup_frac * total_steps))
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total_steps - warm_steps))

    def step_scheduler(global_step: int):
        if global_step < warm_steps:
            scale = float(global_step + 1) / float(warm_steps)
            for g in opt.param_groups:
                g["lr"] = lr * scale
        else:
            cos.step()

    split_type = (
        "stratified in-memory split" if used_stratified
        else ("DB-driven splits" if val_dataset is not None else "random split")
    )
    print(
        f"🧠 Pretraining (Temporal): params(enc)={_count_params(enc):,}"
        + (f", params(MTM)={_count_params(mtm_head):,}" if mtm_head else "")
        + (f", params(Con)={_count_params(con_head):,}" if con_head else "")
        + f" | {split_type}"
    )

    # --- training state ---
    history = {
        "train_total": [], "val_total": [],
        "train_mtm": [], "train_con": [],
        "val_mtm": [], "val_con": [],
        "ema_total": [], "lr": [], "grad_norm": []
    }
    best_val = float("inf")
    best_epoch = 0
    no_improve = 0
    ema = None
    t0 = time.time()

    # --- epochs ---
    for ep in range(1, epochs + 1):
        history["lr"].append(opt.param_groups[0]["lr"])

        train_metrics = train_one_epoch(
            enc, mtm_head, con_head, train_loader, device,
            objectives, lambdas, temperature, opt, step_scheduler,
            grad_clip, vr_mode=False, pooling=pooling
        )
        val_metrics = evaluate_one_epoch(
            enc, mtm_head, con_head, val_loader, device,
            objectives, lambdas, temperature, vr_mode=False, pooling=pooling
        )

        if ema_alpha > 0.0:
            ema = val_metrics["total"] if ema is None else (ema_alpha * val_metrics["total"] + (1 - ema_alpha) * ema)
        else:
            ema = val_metrics["total"]

        history["train_total"].append(train_metrics["total"])
        history["val_total"].append(val_metrics["total"])
        history["train_mtm"].append(train_metrics["mtm"])
        history["train_con"].append(train_metrics["con"])
        history["val_mtm"].append(val_metrics["mtm"])
        history["val_con"].append(val_metrics["con"])
        history["ema_total"].append(ema)
        history["grad_norm"].append(train_metrics["grad_norm"])

        dt = time.time() - t0
        print(
            f"Epoch {ep:03d}/{epochs} | "
            f"train_total={train_metrics['total']:.4f}  "
            f"val_total={val_metrics['total']:.4f}  "
            f"(mtm tr/val {train_metrics['mtm']:.4f}/{val_metrics['mtm']:.4f}  "
            f"con tr/val {train_metrics['con']:.4f}/{val_metrics['con']:.4f})  "
            f"EMA={ema:.4f}  grad={train_metrics['grad_norm']:.3f}"
        )

        # save last
        last_path = save_path.replace(".pt", "_last.pt") if save_path.endswith(".pt") else (save_path + "_last")
        torch.save(
            {
                "state_dict": enc.state_dict(),
                "config": {
                    "d_model": d_model,
                    "n_layers": n_layers,
                    "n_heads": n_heads,
                    "max_len": max_len,
                    "pooling": pooling,
                    "encoder_type": "temporal",
                },
                "epoch": ep,
                "metrics": {"train": train_metrics, "val": val_metrics},
            },
            last_path,
        )

        # best-by-val
        if val_metrics["total"] < best_val:
            best_val = val_metrics["total"]
            best_epoch = ep
            no_improve = 0
            torch.save(
                {
                    "state_dict": enc.state_dict(),
                    "config": {
                        "d_model": d_model,
                        "n_layers": n_layers,
                        "n_heads": n_heads,
                        "max_len": max_len,
                        "pooling": pooling,
                        "encoder_type": "temporal",
                    },
                    "epoch": ep,
                    "metrics": {"train": train_metrics, "val": val_metrics},
                },
                save_path,
            )
            print(f"💾 Saved BEST (temporal) to {save_path} (val {best_val:.4f})")
        else:
            no_improve += 1

        if patience > 0 and no_improve >= patience:
            print(f"⏹️ Early stopping at epoch {ep} (no val improvement for {patience} epochs).")
            break

    _plot_losses(history, plot_path)
    stab_epoch = _detect_stabilization(history["val_total"], window=stab_window, rel_tol=stab_tol)
    print(f"📉 Loss plot saved to: {plot_path}")
    print(f"🧭 Stabilization (heuristic): around epoch {stab_epoch}")
    print(f"🏆 Best epoch: {best_epoch} (val={best_val:.4f})")
    print("✅ Temporal pretraining finished.")


# ==========================================
# VR-Transformer pretraining (DB-aware splits)
# ==========================================

def run_vr_pretraining(
    train_dataset,
    val_dataset=None,              # <- pass your DB 'val' dataset here to disable random_split
    encoder_ctor: Optional[Callable[[], nn.Module]] = None,
    save_path: Optional[str] = None,
    used_stratified: bool = False,
):
    """
    Self-supervised pretraining for the VR Transformer.
    Uses DB-driven validation if `val_dataset` is provided, otherwise random split.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # --- build loaders (DB-driven if given) ---
    if val_dataset is None:
        val_fraction = getattr(CP, "VAL_FRACTION", 0.1)
        n_total = len(train_dataset)
        n_val   = max(1, int(n_total * val_fraction))
        n_train = n_total - n_val
        train_ds, val_ds = random_split(train_dataset, [n_train, n_val])
    else:
        train_ds, val_ds = train_dataset, val_dataset

    train_loader = DataLoader(
        train_ds, batch_size=CP.BATCH_SIZE_PRE, shuffle=True,
        collate_fn=pretrain_collate_fn, num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_ds, batch_size=CP.BATCH_SIZE_PRE, shuffle=False,
        collate_fn=pretrain_collate_fn, num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    # --- configs ---
    objectives   = CP.OBJECTIVES
    lambdas      = CP.LAMBDA
    temperature  = CP.TEMPERATURE
    epochs       = CP.PRETRAIN_EPOCHS
    lr           = CP.LR_PRE
    weight_decay = CP.WEIGHT_DECAY
    grad_clip    = CP.GRAD_CLIP_NORM
    warmup_frac  = CP.WARMUP_FRAC
    patience     = getattr(CP, "EARLY_STOP_PATIENCE", 0)
    ema_alpha    = getattr(CP, "EMA_ALPHA", 0.0)
    base_ckpt    = save_path or CP.PRETRAIN_CKPT_PATH
    best_ckpt    = base_ckpt.replace(".pt", "_vr.pt") if base_ckpt.endswith(".pt") else (base_ckpt + "_vr")
    last_ckpt    = base_ckpt.replace(".pt", "_vr_last.pt") if base_ckpt.endswith(".pt") else (base_ckpt + "_vr_last")
    plot_path    = getattr(CP, "PRETRAIN_PLOT_PATH_VR", "./artifacts/pretrain/pretrain_vr_loss.png")
    stab_window  = getattr(CP, "STABILIZE_MA_WINDOW", 3)
    stab_tol     = getattr(CP, "STABILIZE_REL_TOL", 1e-3)

    d_model = CA.ATTN_CONFIG["d_model"]
    max_len = CA.ATTN_CONFIG["max_len"]
    pooling = CA.POOLING

    # --- model + heads ---
    enc = (encoder_ctor() if encoder_ctor is not None else TransformerEncoder()).to(device)
    mtm_head = MTMHead(d_model).to(device) if objectives.get("mtm", False) else None
    con_head = ContrastiveHead(d_model).to(device) if objectives.get("contrastive", False) else None

    params = list(enc.parameters())
    if mtm_head is not None: params += list(mtm_head.parameters())
    if con_head is not None: params += list(con_head.parameters())

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    total_steps = max(1, epochs * len(train_loader))
    warm_steps  = int(max(1, warmup_frac * total_steps))
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total_steps - warm_steps))

    def step_scheduler(global_step: int):
        if global_step < warm_steps:
            scale = float(global_step + 1) / float(warm_steps)
            for g in opt.param_groups:
                g["lr"] = lr * scale
        else:
            cos.step()

    split_type = (
        "stratified in-memory split" if used_stratified
        else ("DB-driven splits" if val_dataset is not None else "random split")
    )
    print(
        f"🧠 Pretraining (VR): params(enc)={_count_params(enc):,}"
        + (f", params(MTM)={_count_params(mtm_head):,}" if mtm_head else "")
        + (f", params(Con)={_count_params(con_head):,}" if con_head else "")
        + f" | {split_type}"
    )

    # --- training state ---
    history = {
        "train_total": [], "val_total": [],
        "train_mtm": [], "train_con": [],
        "val_mtm": [], "val_con": [],
        "ema_total": [], "lr": [], "grad_norm": []
    }
    best_val = float("inf")
    best_epoch = 0
    no_improve = 0
    ema = None
    t0 = time.time()

    # --- epochs ---
    for ep in range(1, epochs + 1):
        history["lr"].append(opt.param_groups[0]["lr"])

        train_metrics = train_one_epoch(
            enc, mtm_head, con_head, train_loader, device,
            objectives, lambdas, temperature, opt, step_scheduler,
            grad_clip, vr_mode=True, pooling=pooling
        )
        val_metrics = evaluate_one_epoch(
            enc, mtm_head, con_head, val_loader, device,
            objectives, lambdas, temperature, vr_mode=True, pooling=pooling
        )

        if ema_alpha > 0.0:
            ema = val_metrics["total"] if ema is None else (ema_alpha * val_metrics["total"] + (1 - ema_alpha) * ema)
        else:
            ema = val_metrics["total"]

        history["train_total"].append(train_metrics["total"])
        history["val_total"].append(val_metrics["total"])
        history["train_mtm"].append(train_metrics["mtm"])
        history["train_con"].append(train_metrics["con"])
        history["val_mtm"].append(val_metrics["mtm"])
        history["val_con"].append(val_metrics["con"])
        history["ema_total"].append(ema)
        history["grad_norm"].append(train_metrics["grad_norm"])

        dt = time.time() - t0
        print(
            f"[VR] Epoch {ep:03d}/{epochs} | "
            f"train_total={train_metrics['total']:.4f}  "
            f"val_total={val_metrics['total']:.4f}  "
            f"(mtm tr/val {train_metrics['mtm']:.4f}/{val_metrics['mtm']:.4f}  "
            f"con tr/val {train_metrics['con']:.4f}/{val_metrics['con']:.4f})  "
            f"EMA={ema:.4f}  grad={train_metrics['grad_norm']:.3f}"
        )

        torch.save(
            {
                "state_dict": enc.state_dict(),
                "config": {
                    "d_model": d_model,
                    "max_len": max_len,
                    "pooling": pooling,
                    "encoder_type": "vr",
                },
                "epoch": ep,
                "metrics": {"train": train_metrics, "val": val_metrics},
            },
            last_ckpt,
        )

        if ema < best_val:  # <-- Change condition to use EMA
            best_val = ema  # <-- Update best_val with the new best EMA value
        #if val_metrics["total"] < best_val:
        #    best_val = val_metrics["total"]
            best_epoch = ep
            no_improve = 0
            torch.save(
                {
                    "state_dict": enc.state_dict(),
                    "config": {
                        "d_model": d_model,
                        "max_len": max_len,
                        "pooling": pooling,
                        "encoder_type": "vr",
                    },
                    "epoch": ep,
                    "metrics": {"train": train_metrics, "val": val_metrics},
                },
                best_ckpt,
            )
            print(f"💾 Saved BEST (vr) to {best_ckpt} (val {best_val:.4f})")
        else:
            no_improve += 1

        if patience > 0 and no_improve >= patience:
            print(f"⏹️ Early stopping at epoch {ep} (no val improvement for {patience} epochs).")
            break

    _plot_losses(history, plot_path)
    stab_epoch = _detect_stabilization(history["val_total"], window=stab_window, rel_tol=stab_tol)
    print(f"📉 Loss plot saved to: {plot_path}")
    print(f"🧭 Stabilization (heuristic): around epoch {stab_epoch}")
    print(f"🏆 Best epoch: {best_epoch} (val={best_val:.4f})")
    print("✅ VR pretraining finished.")



