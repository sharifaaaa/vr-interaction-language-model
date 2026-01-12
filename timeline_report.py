# timeline_report.py
from __future__ import annotations
from typing import Optional, Callable, Dict, Any
import torch
import matplotlib
matplotlib.use("Agg")  # save-only backend (safe for scripts/IDE)
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import os
import config_pretrain as CP

# ------- Default colors & order for timeline classes -------
DEFAULT_CLASS_PALETTE = {
    "frustration": "#D55E00",  # orange-red
    "joy":         "#009E73",  # green
    "neutral":     "#999999",  # gray
    "stress":      "#0072B2",  # blue
    "surprise":    "#F0E442",  # yellow
}

DEFAULT_CLASS_ORDER = ["frustration", "joy", "neutral", "stress", "surprise"]

# reuse  interpreter helpers (already handle Temporal/VR backbones)
from interpretability_captum import (
    build_forward_from_classifier_ckpt,
    make_labeled_loader_from_df,
)

def generate_timeline(
    *,
    chunked_df: pd.DataFrame | str,
    label_col: str,
    ckpt_path: str,
    embedder: Any | None = None,
    embedder_builder: Callable[[], Any] | None = None,
    batch_size: int = CP.BATCH_SIZE,
    device: Optional[torch.device] = None,
) -> pd.DataFrame:
    """
    Build a per-chunk timeline with ground-truth and predicted emotions.

    Returns a DataFrame with: chunk_id, user_emotion, pred_emotion, pred_conf, correct
    and optional t0/t1 if available in chunked_df.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load df if a path was passed
    if isinstance(chunked_df, str):
        chunked_df = pd.read_csv(chunked_df,low_memory=False)

    # embedder: use given or build one
    if embedder is None and embedder_builder is not None:
        embedder = embedder_builder()
    if embedder is None:
        raise RuntimeError("embedder is None and embedder_builder not provided.")

    # forward model from checkpoint (Temporal or VR)
    forward_model, backbone, _ = build_forward_from_classifier_ckpt(ckpt_path)
    forward_model = forward_model.to(device).eval()

    # dataset/loader in exactly the same way as training
    loader, class_to_idx = make_labeled_loader_from_df(
        chunked_df, embedder, label_col=label_col, batch_size=CP.BATCH_SIZE, shuffle=False
    )
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # get consistent order of ids/labels from dataset
    ds = loader.dataset
    df_lab = getattr(ds, "df_lab", None)
    if df_lab is None and hasattr(ds, "dataset"):
        df_lab = getattr(ds.dataset, "df_lab", None)
    if df_lab is None:
        raise RuntimeError("Could not locate df_lab on dataset; cannot build timeline.")

    ids_in_order = df_lab["chunk_id"].tolist()
    sessions_in_order = df_lab["session_id"].tolist()
    y_true_names = df_lab[label_col].tolist()

    # predict
    y_pred_idx, y_pred_conf = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            m = batch["mask"].to(device)
            probs = torch.softmax(forward_model(x, m), dim=-1)  # [B,C]
            y_pred_idx.extend(probs.argmax(dim=-1).cpu().tolist())
            y_pred_conf.extend(probs.max(dim=-1).values.cpu().tolist())

    y_pred_names = [idx_to_class.get(i, str(i)) for i in y_pred_idx]
    correct = [int(p == class_to_idx.get(gt, -999)) for p, gt in zip(y_pred_idx, y_true_names)]

    # assemble timeline DF
    timeline = pd.DataFrame({
        "chunk_id": ids_in_order,
        "session_id":sessions_in_order,
        "user_emotion": y_true_names,
        "pred_emotion": y_pred_names,
        "pred_conf": y_pred_conf,
        "correct": correct,
    })
    print(f"timeline dataFrame(timeline_report module): {timeline.shape} ")
    # carry over time columns if present
    def _carry(name_candidates, out_name):
        for c in name_candidates:
            if c in chunked_df.columns:
                timeline[out_name] = chunked_df.set_index("chunk_id").loc[ids_in_order, c].values
                return True
        return False

    got_t0 = _carry(["t0_s","t_start_s","start_s","t0","start_time","start"], "t0")
    got_t1 = _carry(["t1_s","t_end_s","end_s","t1","end_time","end"], "t1")
    if not (got_t0 and got_t1):
        timeline["t0"] = np.arange(len(timeline), dtype=int)
        timeline["t1"] = timeline["t0"] + 1

    return timeline

def plot_timeline_stripes(
    timeline: pd.DataFrame,
    *,
    class_palette: dict[str, str] | None = None,  # optional override; defaults to constants above
    class_order: list[str] | None = None,         # optional override; defaults to constants above
    save_path: str | None = None,
    title: str = "Ground-truth vs Prediction",
    show: bool = False,
):
    """
    Two-row stripe plot (GT on top, Pred below) with fixed colors per class.
    Uses DEFAULT_CLASS_PALETTE/DEFAULT_CLASS_ORDER unless you override via args.
    """
    # Defaults from this module
    class_palette = DEFAULT_CLASS_PALETTE if class_palette is None else class_palette
    class_order   = DEFAULT_CLASS_ORDER   if class_order   is None else class_order

    # Classes present in this timeline
    present = set(timeline["user_emotion"]).union(set(timeline["pred_emotion"]))

    # Final ordered class list: keep requested order, but drop classes not present
    classes = [c for c in class_order if c in present]
    if not classes:  # fallback: any remaining/present classes in palette order, else sorted
        classes = [c for c in class_palette.keys() if c in present] or sorted(present)

    # Color list aligned to 'classes'
    colors = [class_palette.get(c, "#999999") for c in classes]

    # Map classes -> ints for discrete colormap
    cls2id = {c: i for i, c in enumerate(classes)}
    cmap = ListedColormap(colors, name="timeline")
    norm = BoundaryNorm(np.arange(-0.5, len(classes) + 0.5), len(classes))

    # Encode series as integer stripes
    gt_ids = timeline["user_emotion"].map(cls2id).to_numpy()
    pr_ids = timeline["pred_emotion"].map(cls2id).to_numpy()

    # Plot
    fig, ax = plt.subplots(2, 1, figsize=(12, 2.8), sharex=True)
    ax[0].imshow(gt_ids[None, :], aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
    ax[0].set_yticks([0]); ax[0].set_yticklabels(["ground-truth"])
    ax[0].set_title(title)

    ax[1].imshow(pr_ids[None, :], aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
    ax[1].set_yticks([0]); ax[1].set_yticklabels(["prediction"])
    ax[1].set_xlabel("chunk index (or time bins)")

    # Legend with the same colors
    patches = [Patch(facecolor=colors[i], edgecolor="none", label=classes[i]) for i in range(len(classes))]
    ax[0].legend(patches, [p.get_label() for p in patches],
                 bbox_to_anchor=(1.02, 1.0), loc="upper left", title="classes")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return fig


def plot_timeline_stripes_time(
    timeline: pd.DataFrame,
    *,
    unit: str = "s",                      # "s" | "min" | "ms"
    save_path: str | None = None,
    title: str = "VR session — emotions over time (time axis)",
    show: bool = False,
    class_palette: dict[str,str] | None = None,
    class_order: list[str] | None = None,
):
    """
    Time-aware version: draws variable-width segments using t0/t1.
    If your windows overlap (chunk_size > stride), this uses the next
    start time as the end of each bin except the last (which uses t1).
    """
    assert "t0" in timeline.columns and "t1" in timeline.columns, \
        "timeline must contain t0/t1 columns to plot by time."

    # use the same discrete colors as the index-based version
    class_palette = DEFAULT_CLASS_PALETTE if class_palette is None else class_palette
    class_order   = DEFAULT_CLASS_ORDER   if class_order   is None else class_order

    present = set(timeline["user_emotion"]).union(set(timeline["pred_emotion"]))
    classes = [c for c in class_order if c in present] or \
              [c for c in class_palette.keys() if c in present] or \
              sorted(present)
    print(f"Test -save_path:{save_path}")
    colors = [class_palette.get(c, "#999999") for c in classes]
    cls2id = {c: i for i, c in enumerate(classes)}
    cmap = ListedColormap(colors, name="timeline")
    norm = BoundaryNorm(np.arange(-0.5, len(classes) + 0.5), len(classes))

    # integer-encoded labels
    gt_ids = timeline["user_emotion"].map(cls2id).to_numpy()
    pr_ids = timeline["pred_emotion"].map(cls2id).to_numpy()

    # time edges: [t0_0, t0_1, ..., t0_{N-1}, t1_{N-1}]
    t0 = timeline["t0"].to_numpy(dtype=float)
    t1 = timeline["t1"].to_numpy(dtype=float)
    x_edges = np.empty(len(t0) + 1, dtype=float)
    x_edges[:-1] = t0
    x_edges[-1]  = t1[-1]  # last edge = last window's end

    # unit conversion
    if unit == "min":
        x_edges = x_edges / 60.0
        x_label = "time (min)"
    elif unit == "ms":
        x_edges = x_edges * 1000.0
        x_label = "time (ms)"
    else:
        x_label = "time (s)"

    # build 2×N array for pcolormesh (two rows: GT and Pred)
    Z = np.vstack([gt_ids, pr_ids])           # shape (2, N)
    Y_edges = np.array([0.0, 1.0, 2.0])       # three horizontal edges for two rows
    XX, YY = np.meshgrid(x_edges, Y_edges)

    fig, ax = plt.subplots(1, 1, figsize=(12, 2.8))
    ax.pcolormesh(XX, YY, Z, cmap=cmap, norm=norm, shading="flat")
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["ground-truth", "prediction"])
    ax.set_ylim(0, 2)
    ax.set_xlabel(x_label)
    ax.set_title(title)

    patches = [Patch(facecolor=colors[i], edgecolor="none", label=classes[i]) for i in range(len(classes))]
    ax.legend(patches, [p.get_label() for p in patches],
              bbox_to_anchor=(1.02, 1.0), loc="upper left", title="classes")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return fig

# ===== NEW: timeline with phase + phase×emotion matrix =====

def generate_timeline_with_phase(
    *,
    chunked_df: pd.DataFrame | str,
    label_col: str,
    ckpt_path: str,
    embedder: Any | None = None,
    embedder_builder: Callable[[], Any] | None = None,
    batch_size: int = CP.BATCH_SIZE,
    device: Optional[torch.device] = None,
    phase_priority: tuple[str, ...] = ("phase", "task_phase", "state", "area"),
    out_csv: str | None = None,
) -> pd.DataFrame:
    """
    Same as `generate_timeline`, but also attaches a `phase` column if available
    in the original `chunked_df` (using the first present column among
    `phase_priority`). Returns a DataFrame with:
        chunk_id, user_emotion, pred_emotion, pred_conf, correct, t0, t1, phase, session_id
    Optionally writes it to `out_csv`.
    """
    # 1) Build the base timeline
    tl = generate_timeline(
        chunked_df=chunked_df,
        label_col=label_col,
        ckpt_path=ckpt_path,
        embedder=embedder,
        embedder_builder=embedder_builder,
        batch_size=batch_size,
        device=device,
    )

    # 2) Load the original chunked_df (if path) to carry `phase` and `session_id`
    if isinstance(chunked_df, str):
        src = pd.read_csv(chunked_df,low_memory=False)
    else:
        src = chunked_df.copy()

    # Make sure we can align on chunk_id
    if "chunk_id" not in src.columns:
        raise RuntimeError("`chunked_df` must contain 'chunk_id' to merge phase info.")

    # Pick the first available phase-like column
    found_phase_col = None
    for c in phase_priority:
        if c in src.columns:
            found_phase_col = c
            break

    # Build a small lookup table to join
    cols_to_take = ["chunk_id"]
    if found_phase_col is not None:
        cols_to_take.append(found_phase_col)

    # Try to carry session_id too (handy for per-session heatmaps)
    if "session_id" in src.columns:
        cols_to_take.append("session_id")
    else:
        # derive session_id from chunk_id if not present
        def _parse_session(chunk_id: str) -> str:
            m = re.match(r"(session_[^:]+)\.json", str(chunk_id))
            return m.group(1) if m else str(chunk_id).split(":")[0]
        import re
        tmp = src[["chunk_id"]].copy()
        tmp["session_id"] = tmp["chunk_id"].apply(_parse_session)
        src = src.merge(tmp, on="chunk_id", how="left")
        cols_to_take.append("session_id")

    lut = src[cols_to_take].drop_duplicates().set_index("chunk_id")

    # Attach `phase` (if we found one) + `session_id`
    tl = tl.copy()
    tl["session_id"] = lut.loc[tl["chunk_id"], "session_id"].values
    if found_phase_col is not None:
        tl["phase"] = lut.loc[tl["chunk_id"], found_phase_col].values
    else:
        tl["phase"] = np.nan  # no phase available

    # Optional write-out
    if out_csv:
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        tl.to_csv(out_csv, index=False)

    return tl


def build_phase_emotion_matrix_from_timeline(
    timeline_with_phase: pd.DataFrame,
    *,
    session_id: str,
    phases_order: list[str] | None = None,
    emotions_order: list[str] | None = None,
    weight_mode: str = "duration*conf",   # "duration" or "duration*conf"
    phase_col: str = "phase",
    label_col: str = "pred_emotion",
) -> tuple[list[str], list[str], np.ndarray]:
    """
    Create a (phases × emotions) matrix for a single session.
    matrix[i, j] = normalized weight of emotion j inside phase i.
    - weights = duration (t1 - t0), optionally multiplied by pred_conf.

    Returns (phases, emotions, matrix)
    """
    df = timeline_with_phase.copy()
    if session_id is not None:
        df = df[df["session_id"] == session_id]

    if df.empty:
        raise ValueError(f"No rows for session_id={session_id}")

    # Ensure times and duration
    for c in ("t0", "t1"):
        if c not in df.columns:
            raise ValueError("timeline_with_phase must contain t0 and t1 columns.")
    df["duration"] = (df["t1"] - df["t0"]).astype(float).clip(lower=0.0)

    # Choose weights
    if weight_mode == "duration":
        df["w"] = df["duration"]
    elif weight_mode.lower() in {"duration*conf", "duration_conf", "dur_conf"}:
        conf = df["pred_conf"] if "pred_conf" in df.columns else 1.0
        df["w"] = df["duration"] * conf
    else:
        raise ValueError("weight_mode must be 'duration' or 'duration*conf'")

    if phase_col not in df.columns:
        raise ValueError(f"`{phase_col}` not found. Use `generate_timeline_with_phase` first "
                         "or pass the correct `phase_col`.")

    # Phase & emotion orders
    if emotions_order is None:
        emotions = list(df[label_col].astype(str).unique())
    else:
        emotions = list(emotions_order)

    if phases_order is None:
        phases = list(df[phase_col].astype(str).unique())
    else:
        phases = list(phases_order)

    # Aggregate + pivot
    agg = (
        df.groupby([phase_col, label_col], as_index=False)["w"].sum()
          .pivot(index=phase_col, columns=label_col, values="w")
          .fillna(0.0)
    )
    agg = agg.reindex(index=phases, columns=emotions, fill_value=0.0)

    # Row-normalize
    mat = agg.to_numpy(dtype=float)
    row_sum = mat.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    mat = mat / row_sum

    return phases, emotions, mat

