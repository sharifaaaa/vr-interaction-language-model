# train_classifier.py (uses trainer_supervised.py; 10 Sep)
from __future__ import annotations
from typing import Tuple, Dict, List
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

import config_attention as CA
import config_pretrain as CP

from confusion_tools import (
    compute_confusion_matrix,
    precision_recall_f1_from_cm,
    ConfusionTracker,
)

# Unified trainer
from trainer_supervised import (
    SupervisedTrainArgs,
    run_supervised_trainval,
)

# -----------------------------
# Data helpers
# -----------------------------
def make_dataloaders(
    X: torch.Tensor,
    M: torch.Tensor,
    y: torch.Tensor,
    batch_size: int | None = None,
    val_ratio: float | None = None,
    seed: int | None = None
) -> Tuple[DataLoader, DataLoader]:
    batch_size = batch_size or BATCH_SIZE
    val_ratio  = val_ratio  or VAL_RATIO
    seed       = seed       or SEED

    dataset = TensorDataset(X, M, y)
    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = max(1, len(dataset) - val_size)
    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=g)

    tr_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=False)
    va_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    return tr_loader, va_loader


# -----------------------------
# Forward adapter for the unified trainer
# -----------------------------
def _forward_fn_tensors(enc, head, batch, device: torch.device):
    """
    Adapter to fit trainer_supervised's (encoder, head, batch, device) -> (logits, targets).
    Here, 'head' IS your classifier model with signature: model(X, M) -> (logits, pooled).
    The 'enc' argument is unused (we pass nn.Identity()).
    """
    X, M, y = batch
    X, M, y = X.to(device), M.to(device), y.to(device).long()
    logits, _ = head(X, M)
    return logits, y


# -----------------------------
# Main training API (with confusion tracking)
# -----------------------------
def train_transformer_classifier(
    model,
    X: torch.Tensor, M: torch.Tensor, y: torch.Tensor,
    *,
    class_to_idx: dict[str, int] | None = None,   # optional mapping for nicer CM labels
    tracker_out_dir: str | None = None,           # e.g., "./cm_plots"
    batch_size: int | None = None,
    epochs: int | None = None,
    lr: float | None = None,
    weight_decay: float | None = None,
    val_ratio: float | None = None,
    seed: int | None = None,
    device: str | None = None,
) -> Dict:
    # defaults
    batch_size   = batch_size   or CP.BATCH_SIZE
    epochs       = epochs       or CP.EPOCHS
    lr           = lr           or CP.LR
    weight_decay = weight_decay or CP.WEIGHT_DECAY
    val_ratio    = val_ratio    or CP.VAL_RATIO
    seed         = seed         or CP.SEED
    #device       = torch.device(device or DEVICE or ("cuda" if torch.cuda.is_available() else "cpu"))

    # class names for tracker (ordered by index)
    if class_to_idx is None:
        unique_ids = sorted(set(int(t) for t in y.cpu().tolist()))
        class_names = [str(i) for i in unique_ids]
    else:
        class_names = [k for k, _ in sorted(class_to_idx.items(), key=lambda kv: kv[1])]
    n_classes = len(class_names)

    # data loaders
    tr_loader, va_loader = make_dataloaders(X, M, y, batch_size=batch_size, val_ratio=val_ratio, seed=seed)

    # Confusion tracker
    tracker = ConfusionTracker(
        class_names=class_names,
        out_dir=tracker_out_dir,
        normalize=True,
        keep_history=False
    )

    # Baseline BEFORE training (optional)
    if CA.SAVE_BASELINE:
        model.to(device).eval()
        y_true0: List[int] = []
        y_pred0: List[int] = []
        with torch.no_grad():
            for batch in va_loader:
                logits, yb = _forward_fn_tensors(None, model, batch, device)
                preds = logits.argmax(dim=-1).detach().cpu().tolist()
                y_true0.extend(yb.detach().cpu().tolist())
                y_pred0.extend(preds)
        cm0 = compute_confusion_matrix(y_true0, y_pred0, n_classes)   # np.ndarray
        _, _, _, macro_f1_0 = precision_recall_f1_from_cm(cm0)
        tracker.save_baseline(cm0)
        print(f"🧪 Baseline (no training) -> macroF1={macro_f1_0:.3f}")
    else:
        print("⏭️ Skipping baseline confusion matrix (SAVE_BASELINE=False).")

    # ---- Unified trainer (no freezing/scheduler here; plain CE like your original) ----
    args = SupervisedTrainArgs(
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        grad_clip=1.0,
        freeze_epochs=0,                 # no freeze/unfreeze in train-clf
        use_warmup_cosine=False,         # keep behavior similar to original simple loop
        warmup_steps=0,
        total_steps=None,
        loss_type="ce",                  # standard CE
        class_weights=None,              # add if you want imbalance handling here
        focal_gamma=2.0,
        focal_alpha_scalar=None,
        device=device,
        log_every=100,
    )

    # Dummy encoder; your model is the "head"
    enc = torch.nn.Identity()
    head = model.to(device)

    results = run_supervised_trainval(
        encoder=enc,
        head=head,
        train_loader=tr_loader,
        val_loader=va_loader,
        num_classes=n_classes,
        args=args,
        forward_fn=_forward_fn_tensors,
    )

    # Pull best/final CMs from trainer results and feed tracker
    best = results["best"]
    final = results["final"]

    if best.get("val") and best["val"].get("confusion") is not None:
        cm_best = np.array(best["val"]["confusion"])  # ensure numpy
        tracker.update_if_best(cm_best, best.get("macro_f1", -1.0), best.get("epoch", -1))

    macro_f1_final = None
    if final.get("val") and final["val"].get("confusion") is not None:
        cm_final = np.array(final["val"]["confusion"])  # ensure numpy
        tracker.save_final(cm_final)
        _, _, _, macro_f1_final = precision_recall_f1_from_cm(cm_final)

    # Restore best weights returned by the trainer (it stores the head's state_dict)
    if best.get("state_dict_head") is not None:
        head.load_state_dict(best["state_dict_head"])

    # Derive accuracy from final CM if available
    val_acc = None
    if final.get("val") and final["val"].get("confusion") is not None:
        cm = np.array(final["val"]["confusion"])
        total = cm.sum()
        correct = np.trace(cm)
        val_acc = float(correct) / float(total) if total > 0 else None

    print(f"✅ Best-by-tracker macroF1: {tracker.best_macro_f1:.3f} @ epoch {tracker.best_epoch}")
    if macro_f1_final is not None:
        print(f"✅ Final macroF1: {macro_f1_final:.3f}")

    return {
        "model": head,  # same model object, loaded with best weights
        "val_loss": final["val"]["loss"] if final.get("val") else None,
        "val_acc": val_acc,
        "macro_f1_final": macro_f1_final if macro_f1_final is not None else float(best.get("macro_f1", -1.0)),
        "macro_f1_best": tracker.best_macro_f1,
        "best_epoch": tracker.best_epoch,
        "paths": getattr(tracker, "paths", None),
    }
