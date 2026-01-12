# trainer_supervised.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple
import math
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ---------- Losses ----------

class FocalLoss(nn.Module):
    """
    Standard focal loss for multi-class CE.
    alpha can be: float scalar (same for all classes) or a 1D tensor of per-class weights.
    """
    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor | float] = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        # logits: [B, C], targets: [B]
        logp = torch.log_softmax(logits, dim=-1)            # [B, C]
        p = torch.softmax(logits, dim=-1)                   # [B, C]
        pt = p[torch.arange(p.size(0), device=p.device), targets]            # [B]
        logpt = logp[torch.arange(logp.size(0), device=logp.device), targets]# [B]
        loss = -(1 - pt).pow(self.gamma) * logpt

        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                a = self.alpha[targets]  # per-sample
            else:
                a = torch.tensor(float(self.alpha), device=logits.device)
            loss = a * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def make_loss(loss_type: str,
             class_weights: Optional[torch.Tensor] = None,
             focal_gamma: float = 2.0,
             focal_alpha_scalar: Optional[float] = None) -> nn.Module:
    loss_type = loss_type.lower()
    if loss_type in ("ce", "cross_entropy"):
        return nn.CrossEntropyLoss(weight=class_weights)
    if loss_type in ("weighted_ce",):
        return nn.CrossEntropyLoss(weight=class_weights)
    if loss_type in ("focal", "focal_loss"):
        alpha = None
        if focal_alpha_scalar is not None:
            # If class_weights provided, combine them by multiplying with scalar
            if class_weights is not None:
                alpha = class_weights * float(focal_alpha_scalar)
            else:
                alpha = float(focal_alpha_scalar)
        else:
            alpha = class_weights
        return FocalLoss(gamma=focal_gamma, alpha=alpha)
    raise ValueError(f"Unknown loss_type: {loss_type}")


# ---------- Metrics ----------

@torch.no_grad()
def confusion_matrix(pred: torch.Tensor, targ: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    pred, targ: [N] int64 in [0, num_classes)
    returns: [C, C] where rows=true class, cols=predicted class
    """
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long, device=pred.device)
    for t, p in zip(targ, pred):
        cm[t, p] += 1
    return cm


def f1_from_confusion(cm: torch.Tensor) -> Tuple[float, np.ndarray]:
    """
    cm: [C, C]
    returns macro_f1, per_class_f1 (numpy)
    """
    cm = cm.to(torch.float32)
    tp = torch.diag(cm)
    fp = cm.sum(dim=0) - tp
    fn = cm.sum(dim=1) - tp
    denom = (2 * tp + fp + fn).clamp_min(1e-12)
    f1_per_class = (2 * tp) / denom
    macro_f1 = f1_per_class.mean().item()
    return macro_f1, f1_per_class.cpu().numpy()


# ---------- Scheduler (warmup + cosine) ----------

class WarmupCosine:
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int, base_lr: float):
        self.opt = optimizer
        self.warmup_steps = max(0, warmup_steps)
        self.total_steps = max(1, total_steps)
        self.base_lr = base_lr
        self.step_idx = 0

    def _lr_mult(self, step: int) -> float:
        if step < self.warmup_steps and self.warmup_steps > 0:
            return (step + 1) / self.warmup_steps
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def step(self):
        mult = self._lr_mult(self.step_idx)
        for g in self.opt.param_groups:
            g["lr"] = self.base_lr * mult
        self.step_idx += 1


# ---------- Config ----------

@dataclass
class SupervisedTrainArgs:
    epochs: int
    lr: float
    weight_decay: float
    grad_clip: float = 1.0

    # freezing the encoder
    freeze_epochs: int = 0

    # scheduler
    use_warmup_cosine: bool = True
    warmup_steps: int = 0
    total_steps: Optional[int] = None  # if None, computed as epochs * len(train_loader)

    # loss
    loss_type: str = "ce"  # "ce" | "weighted_ce" | "focal"
    class_weights: Optional[torch.Tensor] = None
    focal_gamma: float = 2.0
    focal_alpha_scalar: Optional[float] = None

    # device & misc
    device: Optional[torch.device] = None
    log_every: int = 100  # steps


# ---------- Core loops ----------

def _set_requires_grad(module: nn.Module, enabled: bool):
    for p in module.parameters():
        p.requires_grad = enabled


def _step(
    encoder: nn.Module,
    head: nn.Module,
    batch,
    device: torch.device,
    forward_fn: Callable[[nn.Module, nn.Module, object, torch.device], Tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    grad_clip: Optional[float] = None,
) -> Tuple[float, torch.Tensor, torch.Tensor]:
    """
    One forward (and optional backward) step.
    Returns: loss_value, preds (int64), targets (int64)
    """
    logits, targets = forward_fn(encoder, head, batch, device)  # [B,C], [B]
    loss = criterion(logits, targets)

    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(head.parameters()), grad_clip)
        optimizer.step()

    preds = logits.argmax(dim=-1)
    return float(loss.item()), preds.detach(), targets.detach()


def _run_epoch(
    encoder: nn.Module,
    head: nn.Module,
    loader: DataLoader,
    device: torch.device,
    forward_fn: Callable[[nn.Module, nn.Module, object, torch.device], Tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    grad_clip: Optional[float],
    num_classes: int,
    train: bool,
    scheduler: Optional[WarmupCosine] = None,
    log_every: int = 100,
) -> Dict[str, object]:
    mode = "train" if train else "val"
    if train:
        encoder.train(); head.train()
    else:
        encoder.eval(); head.eval()

    total_loss = 0.0
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long, device=device)
    n_batches = len(loader)
    t0 = time.time()

    for step, batch in enumerate(loader):
        with torch.set_grad_enabled(train):
            loss, preds, targets = _step(
                encoder, head, batch, device, forward_fn, criterion,
                optimizer=optimizer if train else None,
                grad_clip=grad_clip if train else None,
            )

        total_loss += loss
        cm += confusion_matrix(preds, targets, num_classes=num_classes)

        if train and scheduler is not None:
            scheduler.step()

        if (step + 1) % max(1, log_every) == 0:
            macro_f1, _ = f1_from_confusion(cm)
            print(f"[{mode}] step {step+1}/{n_batches} | loss={total_loss/(step+1):.4f} | macro_f1={macro_f1:.4f}")

    macro_f1, per_class_f1 = f1_from_confusion(cm)
    out = {
        "loss": total_loss / max(1, n_batches),
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "confusion": cm.detach().cpu().numpy(),
        "time_sec": time.time() - t0,
    }
    return out


def run_supervised_trainval(
    *,
    encoder: nn.Module,
    head: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    args: SupervisedTrainArgs,
    forward_fn: Callable[[nn.Module, nn.Module, object, torch.device], Tuple[torch.Tensor, torch.Tensor]],
    make_optimizer: Optional[Callable[[torch.nn.ParameterList], torch.optim.Optimizer]] = None,
) -> Dict[str, object]:
    """
    High-level loop managing freeze→unfreeze, scheduler, metrics.
    Returns a dict with best and final metrics and (optionally) best state_dicts.
    """
    device = args.device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    encoder.to(device); head.to(device)

    # ---- freeze if requested ----
    if args.freeze_epochs > 0:
        print(f"Freezing encoder for {args.freeze_epochs} epoch(s).")
        _set_requires_grad(encoder, False)

    params = list(filter(lambda p: p.requires_grad, list(encoder.parameters()) + list(head.parameters())))
    if make_optimizer is None:
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = make_optimizer(params)

    if args.use_warmup_cosine:
        total_steps = args.total_steps or (args.epochs * len(train_loader))
        scheduler = WarmupCosine(optimizer, warmup_steps=args.warmup_steps, total_steps=total_steps, base_lr=args.lr)
    else:
        scheduler = None

    # criterion
    class_weights = args.class_weights.to(device) if isinstance(args.class_weights, torch.Tensor) else None
    criterion = make_loss(args.loss_type, class_weights, args.focal_gamma, args.focal_alpha_scalar)

    best = {"epoch": -1, "macro_f1": -1.0, "state_dict_encoder": None, "state_dict_head": None, "val": None}
    history = {"train": [], "val": []}

    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")

        # unfreeze boundary
        if epoch == args.freeze_epochs + 1 and args.freeze_epochs > 0:
            print("Unfreezing encoder.")
            _set_requires_grad(encoder, True)
            # Rebuild optimizer with encoder params now trainable
            params = list(encoder.parameters()) + list(head.parameters())
            optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

            if args.use_warmup_cosine:
                total_steps = (args.epochs - (epoch - 1)) * len(train_loader)
                scheduler = WarmupCosine(optimizer, warmup_steps=args.warmup_steps, total_steps=total_steps, base_lr=args.lr)

        # train
        tr = _run_epoch(
            encoder, head, train_loader, device, forward_fn, criterion,
            optimizer, args.grad_clip, num_classes, train=True, scheduler=scheduler, log_every=args.log_every
        )
        # val
        with torch.no_grad():
            va = _run_epoch(
                encoder, head, val_loader, device, forward_fn, criterion,
                optimizer=None, grad_clip=None, num_classes=num_classes, train=False, scheduler=None
            )

        history["train"].append(tr); history["val"].append(va)
        print(f"→ train: loss {tr['loss']:.4f} | macro_f1 {tr['macro_f1']:.4f}")
        print(f"→   val: loss {va['loss']:.4f} | macro_f1 {va['macro_f1']:.4f}")

        if va["macro_f1"] > best["macro_f1"]:
            best["macro_f1"] = va["macro_f1"]
            best["epoch"] = epoch
            best["state_dict_encoder"] = {k: v.cpu() for k, v in encoder.state_dict().items()}
            best["state_dict_head"] = {k: v.cpu() for k, v in head.state_dict().items()}
            best["val"] = va
            print(f"✓ New best macro-F1: {best['macro_f1']:.4f} (epoch {epoch})")

    return {
        "best": best,
        "final": {"epoch": args.epochs, "val": history["val"][-1], "train": history["train"][-1]},
        "history": history,
    }
