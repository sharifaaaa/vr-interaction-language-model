# interpretability_captum.py
from __future__ import annotations

import os
import json
from typing import Dict, Tuple, Optional, List, Union

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Captum (optional guard)
try:
    from captum.attr import IntegratedGradients
    _CAPTUM_OK = True
except Exception:
    _CAPTUM_OK = False

# Project imports
import config_attention as CA
from train_pretrain import TemporalEncoder
from pretrain_dataset import make_labeled_windows_dataset, labeled_collate_fn
from pooling import use_cls_from_pooling, _safe_pool

# Optional VR backbone
try:
    from vr_transformer import TransformerEncoder
    _HAS_VR = True
except Exception:
    TransformerEncoder = None
    _HAS_VR = False


# =========================
# Internal helpers
# =========================

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _merge_attn_conf(base: Dict, ck_conf: Dict) -> Dict:
    """Merge checkpoint config into ATTN_CONFIG names."""
    conf = dict(base)
    if "d_model"   in ck_conf: conf["d_model"]   = ck_conf["d_model"]
    if "num_layers" in ck_conf: conf["num_layers"] = ck_conf["num_layers"]
    if "n_layers"  in ck_conf: conf["num_layers"] = ck_conf["n_layers"]
    if "num_heads" in ck_conf: conf["n_heads"]    = ck_conf["num_heads"]
    if "n_heads"   in ck_conf: conf["n_heads"]    = ck_conf["n_heads"]
    if "max_len"   in ck_conf: conf["max_len"]    = ck_conf["max_len"]
    return conf


class _VRWrapper(nn.Module):
    """Wrap (VR encoder) + (classification head) -> forward(x, m) => logits [B, C]."""
    def __init__(self, enc: "TransformerEncoder", head: nn.Module, pooling: str):
        super().__init__()
        self.enc = enc
        self.head = head
        self.pooling = pooling
        self.add_cls = bool(getattr(enc, "use_cls", False)) and use_cls_from_pooling(pooling)

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        out = self.enc(x, m)                               # [B,L,D] or (... last=[B,L,D])
        Y = out[-1] if isinstance(out, (tuple, list)) else out
        z = _safe_pool(Y, m, self.pooling, self.add_cls)   # [B,D]
        return self.head(z)                                # [B,C]


def _build_temporal_from_ckpt(ckpt: Dict, num_classes: int, attn_conf: Dict, pooling: str):
    d_model  = ckpt.get("config", {}).get("d_model",  attn_conf["d_model"])
    n_layers = ckpt.get("config", {}).get("n_layers", attn_conf["num_layers"])
    n_heads  = ckpt.get("config", {}).get("n_heads",  attn_conf["n_heads"])
    max_len  = ckpt.get("config", {}).get("max_len",  attn_conf["max_len"])

    enc = TemporalEncoder(d_model, n_layers, n_heads, max_len)
    enc.load_state_dict(ckpt["encoder"], strict=False)

    head = nn.Sequential(
        nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(p=0.1),
        nn.Linear(d_model, num_classes)
    )
    head.load_state_dict(ckpt["head"], strict=False)

    enc.eval(); head.eval()

    class _F(nn.Module):
        def __init__(self, enc, head, pooling):
            super().__init__(); self.enc, self.head, self.pooling = enc, head, pooling
        def forward(self, x, m):
            H = self.enc(x, m)
            z = self.enc.pool(H, m, pooling=self.pooling)
            return self.head(z)
    return _F(enc, head, pooling)


def _build_vr_from_ckpt(ckpt: Dict, num_classes: int, attn_conf: Dict, pooling: str):
    assert _HAS_VR, "VR transformer package not found; cannot build VR backbone."
    vr_enc = TransformerEncoder()
    try: vr_enc.load_state_dict(ckpt["encoder"], strict=False)
    except Exception: pass
    vr_enc.eval()

    d_model = CA.ATTN_CONFIG["d_model"]
    head = nn.Sequential(
        nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(p=0.1),
        nn.Linear(d_model, num_classes)
    )
    try: head.load_state_dict(ckpt["head"], strict=False)
    except Exception: pass
    head.eval()

    return _VRWrapper(vr_enc, head, pooling)


# =========================
# Visualization helpers
# =========================

def _maybe_crop_time(A: np.ndarray, t_window: Optional[Tuple[int, int]]) -> np.ndarray:
    if t_window is None:
        return A
    s, e = t_window
    s = max(0, int(s)); e = min(A.shape[0], int(e))
    if e <= s:
        return A
    return A[s:e]


def _normalize_heat(Aabs: np.ndarray,
                    mode: str = "global",
                    log_norm: bool = False) -> tuple[np.ndarray, Optional[mcolors.Normalize]]:
    """
    Aabs: [L,D] absolute attributions.
    mode: 'global' (min/max over all), 'per_feature' (row-wise after transpose), 'none'
    Returns (A_norm, norm_obj) usable by imshow.
    """
    if log_norm:
        vmin = max(1e-6, float(Aabs[Aabs > 0].min()) if (Aabs > 0).any() else 1e-6)
        norm_obj = mcolors.LogNorm(vmin=vmin, vmax=float(Aabs.max() + 1e-8))
        return Aabs, norm_obj

    if mode == "per_feature":
        B = Aabs.T                           # [D,L]
        mn = B.min(axis=1, keepdims=True)
        mx = B.max(axis=1, keepdims=True)
        den = (mx - mn + 1e-8)
        Bn = (B - mn) / den
        return Bn.T, None
    elif mode == "none":
        return Aabs, None
    else:  # global
        mn, mx = Aabs.min(), Aabs.max()
        if mx <= mn:
            return np.zeros_like(Aabs), None
        return (Aabs - mn) / (mx - mn + 1e-8), None


def _topk_feature_indices(Aabs: np.ndarray, k: int) -> np.ndarray:
    feat_score = Aabs.sum(axis=0)  # [D]
    k = int(min(max(1, k), feat_score.size))
    return np.argsort(feat_score)[-k:]


def _plot_timestep_line(ts_imp: np.ndarray, title: str, out_path: str):
    plt.figure(figsize=(10, 3))
    plt.plot(ts_imp)
    plt.ylim(bottom=0)
    plt.xlabel("time step"); plt.ylabel("normalized |attr|")
    plt.title(title)
    plt.tight_layout(); plt.savefig(out_path, dpi=140); plt.close()


def _plot_heatmap(A: np.ndarray, title: str, out_path: str,
                  norm_mode: str = "global", log_norm: bool = False):
    Aabs = np.abs(A)
    A_norm, norm_obj = _normalize_heat(Aabs, mode=norm_mode, log_norm=log_norm)
    plt.figure(figsize=(10, 4))
    if norm_obj is None:
        plt.imshow(A_norm.T, aspect="auto", interpolation="nearest")
    else:
        plt.imshow(A_norm.T, aspect="auto", interpolation="nearest", norm=norm_obj)
    plt.colorbar(); plt.xlabel("time"); plt.ylabel("feature dim")
    plt.title(title)
    plt.tight_layout(); plt.savefig(out_path, dpi=140); plt.close()


def _plot_topk_heatmap(A: np.ndarray, k: int, title: str, out_path: str,
                       norm_mode: str = "per_feature", log_norm: bool = False):
    Aabs = np.abs(A)
    idx = _topk_feature_indices(Aabs, k)
    sub = Aabs[:, idx]  # [L, k]
    A_norm, norm_obj = _normalize_heat(sub, mode=norm_mode, log_norm=log_norm)
    plt.figure(figsize=(10, max(3, 0.18 * k)))
    if norm_obj is None:
        plt.imshow(A_norm.T, aspect="auto", interpolation="nearest")
    else:
        plt.imshow(A_norm.T, aspect="auto", interpolation="nearest", norm=norm_obj)
    plt.colorbar(); plt.xlabel("time"); plt.ylabel(f"top-{k} feature dims")
    plt.title(title)
    plt.tight_layout(); plt.savefig(out_path, dpi=140); plt.close()


def _plot_diff_heatmap(A_pred: np.ndarray, A_ref: np.ndarray, title: str, out_path: str,
                       scale: str = "sym"):
    """
    Draw (|A_pred| - |A_ref|) with blue=ref>pred, red=pred>ref.
    """
    diff = np.abs(A_pred) - np.abs(A_ref)  # [L,D]
    if scale == "sym":
        m = max(abs(diff.min()), abs(diff.max()))
        vmin, vmax = -m, m
    else:
        vmin, vmax = float(diff.min()), float(diff.max())
    plt.figure(figsize=(10, 4))
    plt.imshow(diff.T, aspect="auto", interpolation="nearest", cmap="bwr", vmin=vmin, vmax=vmax)
    plt.colorbar(); plt.xlabel("time"); plt.ylabel("feature dim")
    plt.title(title)
    plt.tight_layout(); plt.savefig(out_path, dpi=140); plt.close()


# =========================
# Public API
# =========================

@torch.no_grad()
def build_forward_from_classifier_ckpt(
    ckpt_path: str,
    fallback_attn_config: Dict = CA.ATTN_CONFIG,
    pooling: str = CA.POOLING,
) -> Tuple[nn.Module, str, Dict]:
    """
    Returns (forward_model, backbone_kind, merged_config).
    forward_model(x, m) -> logits [B, C]
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Classifier checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    keys = set(ckpt.keys())
    if ("encoder" not in keys) or ("head" not in keys):
        raise ValueError("Checkpoint must be a finetune artifact with 'encoder' and 'head'.")

    cfg = ckpt.get("config", {})
    backbone = str(cfg.get("backbone", "temporal")).lower()
    merged = _merge_attn_conf(dict(fallback_attn_config), cfg)

    # infer num_classes
    num_classes = None
    if "class_to_idx" in cfg and isinstance(cfg["class_to_idx"], dict):
        num_classes = len(cfg["class_to_idx"])
    if num_classes is None:
        for k, v in ckpt["head"].items():
            if k.endswith("weight") and v.dim() == 2:
                num_classes = v.size(0); break
    if num_classes is None:
        raise ValueError("Could not infer num_classes from checkpoint.")

    if "vr" in backbone:
        fwd = _build_vr_from_ckpt(ckpt, num_classes=num_classes, attn_conf=merged, pooling=CA.POOLING)
        return fwd, "vrtransformer", merged
    else:
        fwd = _build_temporal_from_ckpt(ckpt, num_classes=num_classes, attn_conf=merged, pooling=CA.POOLING)
        return fwd, "temporal", merged


def make_labeled_loader_from_df(
    chunked_df,
    embedder,
    label_col: str,
    batch_size: int = 16,
    shuffle: bool = True,
) -> Tuple[torch.utils.data.DataLoader, Dict[str, int]]:
    """Mirror training: build the labeled dataset + dataloader."""
    ds, class_to_idx = make_labeled_windows_dataset(
        chunked_df, embedder=embedder, label_col=label_col
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        collate_fn=labeled_collate_fn, pin_memory=torch.cuda.is_available()
    )
    return loader, class_to_idx


def run_ig_on_batch(
    forward_model: nn.Module,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    target: Optional[int] = None,
    n_steps: int = 64,
    out_dir: str = "./interpretability",
    prefix: str = "sample",
    # saving switches
    save_lineplot: bool = True,
    save_heatmap: bool = True,
    save_topk_heatmap: bool = True,
    save_diff_against: Optional[np.ndarray] = None,  # pass A_ref to draw a diff heatmap
    save_csv: bool = False,
    save_attrs_npy: bool = False,
    # visualization options
    heat_norm_mode: str = "global",   # 'global' | 'per_feature' | 'none'
    heat_log_norm: bool = False,
    top_k: int = 10,
    time_window: Optional[Tuple[int, int]] = None,   # e.g., (0, 80) to zoom first 80 steps
) -> Dict:
    """
    Run IG on the first sample of 'batch' and write requested artifacts.
    Returns a dict with paths and the raw attribution array 'A' for further use.
    """
    if not _CAPTUM_OK:
        print("⚠️ Captum not installed. Skipping IG.")
        return {"skipped": True}

    _ensure_dir(out_dir)

    x = batch["x"].to(device)      # [B,L,D]
    m = batch["mask"].to(device)   # [B,L]
    x1 = x[:1].clone().detach().requires_grad_(True)
    m1 = m[:1]

    # choose target as argmax if not provided
    with torch.no_grad():
        logits = forward_model(x1, m1)
        pred_idx = int(logits.softmax(-1).argmax(-1).item())
    tgt = pred_idx if target is None else int(target)

    ig = IntegratedGradients(forward_model)
    baseline = torch.zeros_like(x1)

    attrs, delta = ig.attribute(
        inputs=x1,
        baselines=baseline,
        additional_forward_args=(m1,),
        target=tgt,
        n_steps=n_steps,
        return_convergence_delta=True
    )  # [1,L,D]

    # Convert to numpy safely
    A = attrs.squeeze(0).detach().cpu().numpy()          # [L,D]
    A = _maybe_crop_time(A, time_window)                 # optional zoom
    ts_imp = np.abs(A).sum(axis=-1)
    ts_imp = ts_imp / (ts_imp.max() + 1e-8)

    # delta on CPU scalar
    if isinstance(delta, torch.Tensor):
        delta_mean_abs = float(delta.detach().abs().mean().cpu().item())
    else:
        delta_mean_abs = float(torch.as_tensor(delta).detach().abs().mean().cpu().item())

    # ---- save light-weight artifacts ----
    out: Dict[str, Union[str, float, int, np.ndarray]] = {
        "target": tgt,
        "pred": pred_idx,
        "delta_mean_abs": delta_mean_abs,
        "A": A,                         # keep array in-memory (not in manifest)
        "timestep_scores": ts_imp,
    }

    if save_attrs_npy:
        np.save(os.path.join(out_dir, f"{prefix}_attrs.npy"), A)
        out["attrs_npy"] = os.path.join(out_dir, f"{prefix}_attrs.npy")

    if save_csv:
        np.savetxt(os.path.join(out_dir, f"{prefix}_timestep_scores.csv"), ts_imp, delimiter=",")
        out["ts_scores_csv"] = os.path.join(out_dir, f"{prefix}_timestep_scores.csv")

    title_base = "Analysis of 'emotion_classifier.pt'"

    # line plot
    if save_lineplot:
        p = os.path.join(out_dir, f"{prefix}_timestep_importance.png")
        _plot_timestep_line(ts_imp, f"{title_base} -> IG time-step importance (target={tgt}, pred={pred_idx})", p)
        out["timestep_png"] = p

    # full heatmap
    if save_heatmap:
        p = os.path.join(out_dir, f"{prefix}_heatmap.png")
        _plot_heatmap(A, f"{title_base}-> IG |attr| heatmap", p,
                      norm_mode=heat_norm_mode, log_norm=heat_log_norm)
        out["heatmap_png"] = p

    # top-k heatmap
    if save_topk_heatmap:
        p = os.path.join(out_dir, f"{prefix}_top{int(top_k)}_heatmap.png")
        _plot_topk_heatmap(A, int(top_k),
                           f"{title_base}-> IG |attr| heatmap (top-{int(top_k)} feats)",
                           p, norm_mode="per_feature", log_norm=False)
        out["topk_heatmap_png"] = p

    # diff heatmap vs provided reference
    if save_diff_against is not None:
        p = os.path.join(out_dir, f"{prefix}_diff_heatmap.png")
        _plot_diff_heatmap(A, save_diff_against,
                           f"{title_base}-> IG diff (this - reference)",
                           p, scale="sym")
        out["diff_heatmap_png"] = p

    return out


def run_captum_report(
    chunked_df,
    embedder,
    label_col: str,
    clf_ckpt_path: Optional[str] = None,
    out_dir: str = "./interpretability",
    n_steps: int = 64,
    device: Optional[torch.device] = None,
    # which targets to compute in one pass
    target_modes: Union[str, List[str], Tuple[str, ...]] = ("pred",),
    # visualization knobs (apply to all runs)
    time_window: Optional[Tuple[int, int]] = None,
    heat_norm_mode: str = "global",      # 'global' | 'per_feature' | 'none'
    heat_log_norm: bool = False,
    top_k: int = 10,
    save_lineplot: bool = True,
    save_heatmap: bool = True,
    save_topk_heatmap: bool = True,
    save_csv: bool = False,
    save_attrs_npy: bool = False,
    make_pred_vs_gt_diff: bool = True,   # if both present, add a difference heatmap
) -> Dict:
    """
    Build the classifier, take one sample, run IG for requested targets, and save:
      - line plot over time,
      - full heatmap (with log/per-feature options),
      - top-K feature heatmap,
      - (optional) pred–gt difference heatmap.
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if isinstance(target_modes, str):
        target_modes = (target_modes,)
    else:
        target_modes = tuple(target_modes)

    # 1) forward model
    forward_model, backbone, merged_conf = build_forward_from_classifier_ckpt(
        ckpt_path=clf_ckpt_path,
        fallback_attn_config=CA.ATTN_CONFIG,
        pooling=CA.POOLING
    )
    forward_model = forward_model.to(device).eval()

    # 2) loader
    loader, class_to_idx = make_labeled_loader_from_df(
        chunked_df, embedder, label_col=label_col, batch_size=16, shuffle=True
    )
    batch = next(iter(loader))

    # pred/gt indices (manifest + resolving 'gt')
    with torch.no_grad():
        logits = forward_model(batch["x"][:1].to(device), batch["mask"][:1].to(device))
        pred_idx = int(logits.softmax(-1).argmax(-1).item())

    gt_idx = None
    if "y" in batch:
        try:
            gt_idx = int(batch["y"][0].item())
        except Exception:
            gt_idx = None

    # 3) loop and compute
    runs: Dict[str, Dict] = {}
    for mode in target_modes:
        if mode == "pred":
            tgt_idx, tag = pred_idx, "target-pred"
        elif mode == "gt":
            if gt_idx is None:
                raise ValueError("Ground-truth ('gt') requested but batch lacks 'y'.")
            tgt_idx, tag = gt_idx, "target-gt"
        else:
            if mode not in class_to_idx:
                raise ValueError(f"Unknown class '{mode}'. Known: {list(class_to_idx.keys())}")
            tgt_idx, tag = int(class_to_idx[mode]), f"target-{mode}"

        res = run_ig_on_batch(
            forward_model, batch, device,
            target=tgt_idx, n_steps=n_steps, out_dir=out_dir,
            prefix=f"sample0_{tag}",
            # saving
            save_lineplot=save_lineplot,
            save_heatmap=save_heatmap,
            save_topk_heatmap=save_topk_heatmap,
            save_csv=save_csv, save_attrs_npy=save_attrs_npy,
            # viz options
            heat_norm_mode=heat_norm_mode, heat_log_norm=heat_log_norm,
            top_k=top_k, time_window=time_window
        )
        runs[mode] = res

    # 4) optional diff heatmap (pred – gt)
    if make_pred_vs_gt_diff and ("pred" in runs) and ("gt" in runs):
        A_pred = runs["pred"]["A"]
        A_gt   = runs["gt"]["A"]
        p = os.path.join(out_dir, f"sample0_target-pred_minus_gt_diff_heatmap.png")
        _plot_diff_heatmap(A_pred, A_gt,
                           "Analysis of 'emotion_classifier.pt'-> IG diff (pred - gt)",
                           p, scale="sym")
        runs["pred_gt_diff_png"] = p

    def _to_jsonable(x):
        import numpy as _np
        if isinstance(x, _np.ndarray):
            return x.tolist()
        return x

    # 5) manifest (omit raw 'A' to keep it light)
    manifest = {
        "backbone": backbone,
        "pooling": CA.POOLING,
        "attn_config": merged_conf,
        "class_to_idx": class_to_idx,
        "requested_targets": list(target_modes),
        "pred_idx": pred_idx,
        "gt_idx": gt_idx,
        "time_window": time_window,
        "heat_norm_mode": heat_norm_mode,
        "heat_log_norm": heat_log_norm,
        "top_k": top_k,
        "runs": {
            k: {kk: _to_jsonable(vv) for kk, vv in v.items() if kk not in ("A",)}
            for k, v in runs.items()
        },
    }
    _ensure_dir(out_dir)
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"✅ Captum report saved under: {out_dir} (targets={','.join(target_modes)})")
    return manifest
