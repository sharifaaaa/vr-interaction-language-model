# evaluate_only.py
from __future__ import annotations
from typing import Dict, Optional, List
import os, csv
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pretrain_dataset import LabeledWindowsDataset, labeled_collate_fn
from train_finetune import ClassifierHead, _get_Y, _vr_pool
from train_pretrain import TemporalEncoder
from vr_transformer.transformer_blocks import TransformerEncoder

import config_attention as CATTN
import config_pretrain as CFG
from confusion_tools import compute_confusion_matrix, precision_recall_f1_from_cm,accuracy_from_cm,ConfusionTracker


def _cfg(name: str, default=None):
    return getattr(CFG, name, default)


def _infer_head_spec(head_sd: Dict[str, torch.Tensor],
                     ckpt_class_to_idx: Dict[str, int] | None) -> tuple[int, Optional[int]]:
    """
    Infer (num_classes, hidden) from the stored head state_dict.

    num_classes:
        - Prefer len(ckpt_class_to_idx) if provided.
        - Otherwise, take out_features of the LAST linear layer.

    hidden:
        - If we detect two linear layers (MLP head), we use the out_features
          of the FIRST linear as hidden (when different from num_classes).
        - Otherwise None (simple linear head).
    """
    # 1) num_classes
    if ckpt_class_to_idx:
        num_classes = len(ckpt_class_to_idx)
    else:
        # Try to identify "last" linear layer by common keys.
        last_linear_key = None
        for k in ("net.4.weight", "net.3.weight", "net.1.weight", "net.0.weight"):
            if k in head_sd and head_sd[k].dim() == 2:
                last_linear_key = k
        if last_linear_key is None:
            raise RuntimeError("Cannot infer num_classes: no linear weights found in head checkpoint.")
        num_classes = int(head_sd[last_linear_key].shape[0])

    # 2) hidden size (if any)
    hidden = None
    first_linear_key = None
    for k in ("net.0.weight", "net.1.weight"):
        if k in head_sd and head_sd[k].dim() == 2:
            first_linear_key = k
            break

    if first_linear_key is not None:
        first_out = int(head_sd[first_linear_key].shape[0])
        if first_out != num_classes:
            hidden = first_out

    return num_classes, hidden


@torch.no_grad()
def evaluate_dataset(
    dataset: LabeledWindowsDataset,
    class_to_idx: Dict[str, int] | None,
    *,
    ckpt_path: str,
    out_dir: str = "./artifacts/test_eval",
    batch_size: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:

    if batch_size is None:
        batch_size = _cfg("BATCH_SIZE_FT", _cfg("BATCH_SIZE", 32))
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=labeled_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Finetuned checkpoint not found at '{ckpt_path}'")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    conf = ckpt.get("config", {})

    # === Backbone + encoder config from checkpoint (with safe defaults) ===
    backbone_kind = conf.get("backbone", str(_cfg("FINETUNE_BACKBONE", "temporal")).lower())
    d_model = conf.get("d_model", CATTN.ATTN_CONFIG["d_model"])
    n_layers = conf.get("n_layers", CATTN.ATTN_CONFIG["num_layers"])
    n_heads = conf.get("n_heads", CATTN.ATTN_CONFIG["n_heads"])
    max_len = conf.get("max_len", CATTN.ATTN_CONFIG["max_len"])
    pooling = conf.get("pooling", CATTN.POOLING)
    dropout_head = conf.get("dropout_head", _cfg("DROPOUT_HEAD", 0.1))
    ckpt_map: Dict[str, int] = conf.get("class_to_idx", {})  # name -> index used in training

    # === Rebuild head with matching architecture ===
    head_sd: Dict[str, torch.Tensor] = ckpt["head"]
    num_classes, hidden = _infer_head_spec(head_sd, ckpt_map)

    head = ClassifierHead(
        d_model,
        num_classes,
        hidden=hidden,
        dropout=dropout_head,
    ).to(device)
    head.load_state_dict(head_sd, strict=True)

    # === Rebuild encoder and load weights ===
    enc_sd: Dict[str, torch.Tensor] = ckpt["encoder"]
    if backbone_kind == "temporal":
        enc = TemporalEncoder(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            max_len=max_len,
        ).to(device)
        enc.load_state_dict(enc_sd, strict=True)
    else:
        # vrtransformer-style backbone
        enc = TransformerEncoder().to(device)
        enc.load_state_dict(enc_sd, strict=False)

    enc.eval()
    head.eval()

    # === Label-name remapping: dataset indices -> ckpt indices ===
    ds_inv = None  # dataset idx -> name
    if class_to_idx:
        ds_inv = {v: k for k, v in class_to_idx.items()}
        if ckpt_map:
            missing = [name for name in ds_inv.values() if name not in ckpt_map]
            if missing:
                raise ValueError(f"Dataset has classes not in checkpoint: {missing}")

    # For reporting columns / pretty printing
    if ckpt_map:
        inv_idx_to_class = {idx: name for name, idx in ckpt_map.items()}
    else:
        inv_idx_to_class = {
            i: (ds_inv[i] if ds_inv and i in ds_inv else f"class_{i}")
            for i in range(num_classes)
        }

    # === Inference loop ===
    all_true_ckpt: List[int] = []
    all_pred: List[int] = []
    all_prob: List[np.ndarray] = []
    all_ids: List[str] = []
    all_sids: List[str] = []

    for batch in loader:
        x = batch["x"].to(device)
        m = batch["mask"].to(device)
        y = batch["y"].to(device)  # indices in dataset label space

        ids = batch.get("chunk_id")
        sids = batch.get("db_session_id")

        # forward pass
        if backbone_kind == "temporal":
            H = enc(x, m)
            z = enc.pool(H, m, pooling=pooling)
        else:
            Y = _get_Y(enc(x, m))
            z = _vr_pool(Y, m, pooling=pooling)

        logits = head(z)
        probs = F.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)

        # remap y -> ckpt indices via class names (if needed)
        if ckpt_map and ds_inv is not None:
            y_names = [ds_inv[int(t)] for t in y.detach().cpu().tolist()]
            y_ckpt = [ckpt_map[name] for name in y_names]
            all_true_ckpt.extend(y_ckpt)
        else:
            all_true_ckpt.extend(y.detach().cpu().tolist())

        all_pred.extend(preds.detach().cpu().tolist())
        all_prob.extend(probs.detach().cpu().numpy())
        if ids is not None:
            all_ids.extend(list(ids))
        if sids is not None:
            all_sids.extend(list(sids))

    # === Metrics (in ckpt label space) ===
    y_true = np.array(all_true_ckpt, dtype=int)
    y_pred = np.array(all_pred, dtype=int)

    print(f"num_classes: {num_classes}")
    cm = compute_confusion_matrix(y_true, y_pred, num_classes)
    prec, rec, f1, macro_f1 = precision_recall_f1_from_cm(cm)
    acc_per_class, macro_acc, overall_acc = accuracy_from_cm(cm)

    # ---- Confusion tracking
    class_names = [k for k, _ in sorted(class_to_idx.items(), key=lambda kv: kv[1])]
    tracker = ConfusionTracker(class_names=class_names, out_dir="./artifacts/test_eval/cm_plots",
                               normalize=True, keep_history=False)
    if cm is not None:
        tracker.save_final(cm)

    print("\n===== TEST EVALUATION =====")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"Macro-Acc: {macro_acc:.4f}")
    print(f"Overall-Acc: {overall_acc:.4f}")
    for i in range(num_classes):
        cname = inv_idx_to_class.get(i, f"class_{i}")
        print(f"{cname:>12s} | accuracy={acc_per_class[i]:.3f} precision={prec[i]:.3f}  Recall={rec[i]:.3f}  F1={f1[i]:.3f}")

    # === Save per-chunk predictions ===
    prob_cols = [f"prob_{inv_idx_to_class.get(i, f'class_{i}')}" for i in range(num_classes)]
    csv_path = os.path.join(out_dir, "test_predictions.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["chunk_id", "db_session_id", "y_true", "y_pred"] + prob_cols)
        for i in range(len(y_true)):
            w.writerow([
                (all_ids[i] if i < len(all_ids) else ""),
                (all_sids[i] if i < len(all_sids) else ""),
                int(y_true[i]),
                int(y_pred[i]),
                *[float(p) for p in all_prob[i]],
            ])
    print(f"💾 Wrote per-chunk predictions: {csv_path}")

    # === Save metrics to NPZ ===
    np.savez(
        os.path.join(out_dir, "test_metrics.npz"),
        confusion_matrix=cm,
        precision=prec,
        recall=rec,
        f1=f1,
        macro_f1=np.array([macro_f1], dtype=float),
        class_names=np.array(
            [inv_idx_to_class.get(i, f"class_{i}") for i in range(num_classes)],
            dtype=object,
        ),
    )
    print(f"💾 Saved metrics: {os.path.join(out_dir, 'test_metrics.npz')}")

    return {
        "macro_f1": float(macro_f1),
        "precision": prec.tolist(),
        "recall": rec.tolist(),
        "f1": f1.tolist(),
    }
