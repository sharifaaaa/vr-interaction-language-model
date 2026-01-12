# emotion_visuals.py
"""
One-call wrapper to produce ALL emotion visuals (A,B,C,D) per session.

Outputs are saved under out_dir/<SAFE_SESSION_ID>/...
SAFE_SESSION_ID is built with utils_paths.safe_path_id (db_session_id preferred).
"""

from __future__ import annotations
import os, re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.patches import Patch, Wedge
from matplotlib.colors import to_rgba
from sklearn.decomposition import PCA

# ── path sanitizer (preferred) ────────────────────────────────────────────────
try:
    from utils_paths import safe_path_id  # sanitizes strings for file/dir names
except Exception:
    import re as _re
    def safe_path_id(s: str) -> str:
        s = str(s)
        s = _re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", s)
        s = _re.sub(r"\s+", "_", s).strip("_")
        return s[:160]

# ---- imports from your codebase ----
from timeline_report import (
    generate_timeline_with_phase,
    build_phase_emotion_matrix_from_timeline,
)
try:
    from config_features import PHASE_VALUES as DEFAULT_PHASES_ORDER
except Exception:
    DEFAULT_PHASES_ORDER = None


# ---------- helpers: common bits ----------

def _ensure_dir(p: str) -> None:
    if p and not os.path.isdir(p):
        os.makedirs(p, exist_ok=True)

def _extract_session_from_chunk(chunk_id: str) -> str:
    s = str(chunk_id)
    m = re.search(r"(session_[^:]+?)\.json", s)
    return m.group(1) if m else s.split(":", 1)[0]

def _time_cols(df: pd.DataFrame) -> Tuple[str, str]:
    t0 = "t0" if "t0" in df.columns else ("t_start_s" if "t_start_s" in df.columns else None)
    t1 = "t1" if "t1" in df.columns else ("t_end_s"   if "t_end_s"   in df.columns else None)
    if t0 is None or t1 is None:
        raise ValueError("CSV/DF must contain t0/t1 or t_start_s/t_end_s")
    return t0, t1

# Prefer db_session_id → session_id → parsed from chunk_id (RAW for filtering)
def _derive_raw_session_id_frame(df: pd.DataFrame) -> pd.Series:
    if "db_session_id" in df.columns:
        s = df["db_session_id"].astype(str)
    elif "session_id" in df.columns:
        s = df["session_id"].astype(str)
    elif "chunk_id" in df.columns:
        s = df["chunk_id"].map(_extract_session_from_chunk)
    else:
        raise ValueError("Need one of db_session_id, session_id, or chunk_id.")
    return s

def _session_ids_from_timeline(df: pd.DataFrame) -> List[str]:
    return _derive_raw_session_id_frame(df).dropna().astype(str).unique().tolist()

def _session_filter(tl: pd.DataFrame, sid_raw: str) -> pd.DataFrame:
    # Filter by whichever column exists; fall back to parsing chunk_id
    if "db_session_id" in tl.columns:
        return tl.loc[tl["db_session_id"].astype(str) == str(sid_raw)].copy()
    if "session_id" in tl.columns:
        return tl.loc[tl["session_id"].astype(str) == str(sid_raw)].copy()
    chunk_sids = tl["chunk_id"].astype(str).map(_extract_session_from_chunk)
    return tl.loc[chunk_sids == str(sid_raw)].copy()


# ---------- A) spectrum (timeline stripe) ----------

def plot_emotion_spectrum_per_session(
    timeline_csv: str,
    session_id_raw: str,
    out_path: str,
    label_col: str = "pred_emotion",
) -> str:
    df = pd.read_csv(timeline_csv, low_memory=False)
    sub = _session_filter(df, session_id_raw)
    t0, t1 = _time_cols(sub)
    sub = (
        sub.assign(
            t0=lambda d: pd.to_numeric(d[t0], errors="coerce"),
            t1=lambda d: pd.to_numeric(d[t1], errors="coerce"),
        )
        .dropna(subset=["t0", "t1", label_col])
        .sort_values("t0")
        .reset_index(drop=True)
    )
    if sub.empty:
        raise ValueError(f"No rows for {session_id_raw}")
    emotions = pd.Categorical(sub[label_col]).categories.tolist()
    em2idx = {e: i for i, e in enumerate(emotions)}

    T = int(np.ceil(sub["t1"].max()))
    arr = np.full((1, T), np.nan)
    for _, r in sub.iterrows():
        a = int(np.floor(r.t0)); b = int(np.ceil(r.t1))
        arr[0, a:b] = em2idx[r[label_col]]

    _ensure_dir(os.path.dirname(out_path) or ".")
    plt.figure(figsize=(12, 1.4))
    im = plt.imshow(arr, aspect="auto", interpolation="nearest")
    plt.yticks([]); plt.xlabel("time (s)"); plt.title("Predicted Emotion Spectrum — session")
    cbar = plt.colorbar(im, ticks=range(len(emotions)))
    cbar.ax.set_yticklabels(emotions)
    plt.tight_layout(); plt.savefig(out_path, dpi=200, bbox_inches="tight"); plt.close()
    return out_path


# ---------- B) wheels (avg per session) ----------

def session_avg_probs_from_timeline(
    timeline_csv: str,
    session_id_raw: str,
    emotions_order: List[str],
    label_col: str = "pred_emotion",
    conf_col: str = "pred_conf",
) -> Tuple[List[str], List[float]]:
    df = pd.read_csv(timeline_csv, low_memory=False)
    sub = _session_filter(df, session_id_raw)
    t0, t1 = _time_cols(sub)
    sub["duration"] = (pd.to_numeric(sub[t1], errors="coerce") -
                       pd.to_numeric(sub[t0], errors="coerce")).astype(float).clip(lower=0.0)
    sub = sub.dropna(subset=[label_col])

    score = sub["duration"]
    if conf_col in sub.columns:
        score = score * sub[conf_col].fillna(1.0)

    g = (sub.assign(score=score)
            .groupby(label_col)["score"].sum()
            .reindex(emotions_order).fillna(0.0))
    vals = g.to_numpy(dtype=float)
    if vals.sum() > 0:
        vals = vals / vals.sum()
    return emotions_order, vals.tolist()

def plot_emotion_wheel(
    emotions: List[str],
    values: List[float],
    out_path: str,
    cmap_name: str = "rainbow",
) -> str:
    vals = np.asarray(values, dtype=float)
    norm = colors.Normalize(vmin=0.0, vmax=1.0)
    cmap = cm.get_cmap(cmap_name)
    angles = np.linspace(0, 2*np.pi, len(emotions), endpoint=False)
    width = 2*np.pi/len(emotions) * 0.92

    _ensure_dir(os.path.dirname(out_path) or ".")
    fig = plt.figure(figsize=(5.8, 5.8))
    ax = plt.subplot(111, polar=True)
    bar_colors = cmap(norm(vals))
    ax.bar(angles, vals, width=width, color=bar_colors, edgecolor="black", linewidth=0.8)
    ax.set_ylim(0, 1)
    ax.set_xticks(angles); ax.set_xticklabels(emotions)
    ax.set_yticklabels([])
    ax.set_title("Emotion Wheel (session avg)")
    handles = [Patch(facecolor=bar_colors[i], edgecolor="black", label=emotions[i]) for i in range(len(emotions))]
    ax.legend(handles=handles, bbox_to_anchor=(1.2, 1.05), loc="upper left", frameon=False)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    plt.colorbar(sm, ax=ax, pad=0.1, fraction=0.046).set_label("Probability / Intensity", rotation=90)
    plt.tight_layout(); plt.savefig(out_path, dpi=200, bbox_inches="tight"); plt.close()
    return out_path

def plot_emotion_wheel_gradient(
    emotions,
    values,
    out_path,
    base_cmap="rainbow_r",
    steps=80,
    alpha_min=0.10,
    alpha_max=0.95,
    outline_color="black",
    outline_width=1.2,
    ring_alpha=0.15,
    show_values_on_arc=True,
    start_angle_deg=0,
):
    vals = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    assert len(emotions) == len(vals), "emotions and values must have same length"
    n = len(emotions)

    full = 2 * np.pi
    width = full / n * 0.96
    angles_edge = np.linspace(0, full, n, endpoint=False) + np.deg2rad(start_angle_deg)
    angles_center = angles_edge + width / 2.0
    r_grid = np.linspace(0.0, 1.0, steps + 1)

    cmap = plt.get_cmap(base_cmap, n)
    class_rgba = [cmap(i) for i in range(n)]

    fig = plt.figure(figsize=(7, 7))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(0.0)
    ax.set_theta_direction(1)
    ax.set_ylim(0, 1)
    ax.set_xticks(angles_center)
    ax.set_xticklabels(emotions, fontsize=11)
    ax.tick_params(pad=8)
    ax.set_yticklabels([])
    ax.set_title("Emotion Wheel (session avg)", fontsize=14, pad=18)

    theta = np.linspace(0, full, 361)
    for rr in np.linspace(0.2, 1.0, 5):
        ax.plot(theta, np.full_like(theta, rr), color="black", alpha=ring_alpha, linewidth=0.8, zorder=0)

    for i, (ang0, v) in enumerate(zip(angles_edge, vals)):
        if v <= 0:
            ax.add_patch(Wedge((0,0), r=1e-6,
                               theta1=np.degrees(ang0),
                               theta2=np.degrees(ang0 + width),
                               facecolor="none", edgecolor=outline_color,
                               linewidth=outline_width, transform=ax.transData._b))
            continue

        base = class_rgba[i]
        for j in range(steps):
            r0 = r_grid[j] * v
            r1 = r_grid[j+1] * v
            t = (j + 1) / steps
            a = alpha_min + (alpha_max - alpha_min) * t
            a *= 0.40 + 0.60 * v
            ax.add_patch(Wedge((0,0), r=r1,
                               theta1=np.degrees(ang0),
                               theta2=np.degrees(ang0 + width),
                               width=(r1 - r0) if r1 > r0 else None,
                               facecolor=to_rgba(base, a),
                               edgecolor="none", transform=ax.transData._b))

        ax.add_patch(Wedge((0,0), r=v,
                           theta1=np.degrees(ang0),
                           theta2=np.degrees(ang0 + width),
                           facecolor="none", edgecolor=outline_color,
                           linewidth=outline_width, transform=ax.transData._b))

        if show_values_on_arc:
            angc = ang0 + width/2.0
            ax.text(angc, v + 0.03, f"{v:.2f}",
                    ha="center", va="center", fontsize=10,
                    rotation=np.degrees(angc), rotation_mode="anchor")

    handles = [Patch(facecolor=class_rgba[i], edgecolor=outline_color,
                     label=f"{emotions[i]} ({vals[i]:.2f})") for i in range(n)]
    ax.legend(handles=handles, bbox_to_anchor=(1.18, 1.02),
              loc="upper left", frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------- C) heatmap (phase × emotion) ----------

def plot_emotion_phase_heatmap(
    phases: List[str],
    emotions: List[str],
    matrix: np.ndarray,
    out_path: str,
) -> str:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (len(phases), len(emotions)):
        raise ValueError(f"matrix shape {matrix.shape} must be ({len(phases)}, {len(emotions)})")
    _ensure_dir(os.path.dirname(out_path) or ".")
    plt.figure(figsize=(8, 4)); ax = plt.gca()
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest")
    ax.set_yticks(range(len(phases))); ax.set_yticklabels(phases)
    ax.set_xticks(range(len(emotions))); ax.set_xticklabels(emotions)
    plt.title("Emotion Intensity by Phase"); plt.colorbar(im)
    plt.tight_layout(); plt.savefig(out_path, dpi=200, bbox_inches="tight"); plt.close()
    return out_path


# ---------- D) PCA trajectory (approx probs) ----------

def _approx_prob_matrix_per_second(
    tl_session: pd.DataFrame,
    emotions_order: List[str],
    label_col: str = "pred_emotion",
    conf_col: str = "pred_conf",
) -> np.ndarray:
    t0, t1 = _time_cols(tl_session)
    tl = (tl_session.assign(
            t0=lambda d: pd.to_numeric(d[t0], errors="coerce"),
            t1=lambda d: pd.to_numeric(d[t1], errors="coerce"),
         )
         .dropna(subset=["t0","t1",label_col])
         .sort_values("t0"))
    if tl.empty:
        return np.zeros((0, len(emotions_order)), dtype=float)

    T = int(np.ceil(tl["t1"].max()))
    C = len(emotions_order)
    em2i = {e:i for i,e in enumerate(emotions_order)}
    M = np.zeros((T, C), dtype=float)

    for _, r in tl.iterrows():
        a, b = int(np.floor(r.t0)), int(np.ceil(r.t1))
        idx = em2i.get(str(r[label_col]))
        if idx is None:
            continue
        conf = float(r.get(conf_col, 1.0)) if conf_col in tl.columns else 1.0
        M[a:b, idx] += conf

    row_sum = M.sum(axis=1, keepdims=True)
    nz = row_sum[:,0] > 0
    M[nz] = M[nz] / row_sum[nz]
    return M

def plot_emotion_trajectory_pca(
    probs: np.ndarray,
    out_path: str,
) -> str:
    _ensure_dir(os.path.dirname(out_path) or ".")
    P = np.asarray(probs, dtype=float)

    if P.ndim != 2:
        raise ValueError("probs must be a 2D array (T×C)")

    P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    keep = P.sum(axis=1) > 0
    P = P[keep]

    if P.size == 0 or P.shape[0] < 2:
        plt.figure(figsize=(6, 5)); plt.title("Emotion Trajectory (no data)")
        plt.tight_layout(); plt.savefig(out_path, dpi=200, bbox_inches="tight"); plt.close()
        return out_path

    T, C = P.shape
    if C == 1:
        x = np.arange(T); y = P[:, 0]
        if np.allclose(y.var(), 0.0):
            plt.figure(figsize=(6, 5)); plt.title("Emotion Trajectory (single class, no variance)")
            plt.tight_layout(); plt.savefig(out_path, dpi=200, bbox_inches="tight"); plt.close()
            return out_path
        plt.figure(figsize=(6, 5)); plt.scatter(x, y, c=x, s=12); plt.plot(x, y, alpha=0.5)
        plt.xlabel("Time (s)"); plt.ylabel("Probability"); plt.title("Emotion Trajectory (single class)")
        plt.tight_layout(); plt.savefig(out_path, dpi=200, bbox_inches="tight"); plt.close()
        return out_path

    if np.allclose(P.var(axis=0).sum(), 0.0):
        plt.figure(figsize=(6, 5)); plt.title("Emotion Trajectory (no variance)")
        plt.tight_layout(); plt.savefig(out_path, dpi=200, bbox_inches="tight"); plt.close()
        return out_path

    k = 2 if C >= 2 else 1
    xy = PCA(n_components=k).fit_transform(P)

    plt.figure(figsize=(6, 5))
    plt.scatter(xy[:, 0], xy[:, 1] if k == 2 else np.zeros_like(xy[:, 0]), c=np.arange(len(xy)), s=12)
    if k == 2:
        plt.plot(xy[:, 0], xy[:, 1], alpha=0.5); plt.xlabel("PC1"); plt.ylabel("PC2")
    else:
        plt.plot(xy[:, 0], np.zeros_like(xy[:, 0]), alpha=0.5); plt.xlabel("PC1"); plt.ylabel("")
    plt.title("Emotion Trajectory (PCA of class probs)")
    plt.tight_layout(); plt.savefig(out_path, dpi=200, bbox_inches="tight"); plt.close()
    return out_path


# ---------- MASTER WRAPPER ----------

def generate_all_emotion_visuals(
    *,
    timeline_csv: str,
    chunked_df,                 # path or DataFrame
    clf_ckpt_path: str,
    embedder,
    emotions,                   # list[str] OR dict[str,int]
    out_dir: str = "./plot_emotion_spectrum",
    phases_order: List[str] | None = None,
) -> Dict[str, Dict[str, str]]:

    # normalize inputs
    if isinstance(chunked_df, str):
        df_chunk = pd.read_csv(chunked_df, low_memory=False)
    else:
        df_chunk = chunked_df.copy()

    if isinstance(emotions, dict):
        emotions = [k for k, _ in sorted(emotions.items(), key=lambda kv: kv[1])]
    else:
        emotions = list(emotions)

    _ensure_dir(out_dir)
    tl = pd.read_csv(timeline_csv, low_memory=False)

    # discover sessions (RAW ids for filtering)
    session_ids_raw = _session_ids_from_timeline(tl)
    if not session_ids_raw:
        raise ValueError("No sessions found in timeline_emotions.csv")

    # sanitized ids for filesystem
    sid_map: Dict[str, str] = {sid_raw: safe_path_id(sid_raw) for sid_raw in session_ids_raw}

    # timeline WITH phase (used by the heatmap)
    tl_phase = generate_timeline_with_phase(
        chunked_df=df_chunk,
        label_col="user_emotion",
        ckpt_path=clf_ckpt_path,
        embedder=embedder,
        out_csv=None,
    )

    if phases_order is None:
        phases_order = DEFAULT_PHASES_ORDER

    results: Dict[str, Dict[str, str]] = {}

    for sid_raw in session_ids_raw:
        sid_safe = sid_map[sid_raw]
        sess_dir = os.path.join(out_dir, sid_safe); _ensure_dir(sess_dir)

        # A) spectrum
        a_path = plot_emotion_spectrum_per_session(
            timeline_csv=timeline_csv,
            session_id_raw=sid_raw,  # filter with RAW
            out_path=os.path.join(sess_dir, f"A_spectrum_{sid_safe}.png"),
            label_col="pred_emotion",
        )

        # B) wheels (avg)
        ems, vals = session_avg_probs_from_timeline(
            timeline_csv=timeline_csv,
            session_id_raw=sid_raw,
            emotions_order=emotions,
            label_col="pred_emotion",
            conf_col="pred_conf",
        )
        b1 = plot_emotion_wheel(ems, vals, out_path=os.path.join(sess_dir, f"B_wheel_{sid_safe}.png"))
        b2 = plot_emotion_wheel_gradient(ems, vals, out_path=os.path.join(sess_dir, f"B_wheelGradient_{sid_safe}.png"))

        # C) heatmap (phase×emotion) using timeline_with_phase
        phases, emos, mat = build_phase_emotion_matrix_from_timeline(
            tl_phase, session_id=sid_raw,  # RAW for filtering inside helper
            phases_order=phases_order,
            emotions_order=emotions,
            weight_mode="duration*conf",
            label_col="pred_emotion",
        )
        c_path = plot_emotion_phase_heatmap(
            phases, emos, mat,
            out_path=os.path.join(sess_dir, f"C_heatmap_{sid_safe}.png")
        )

        # D) PCA trajectory
        tl_sess = _session_filter(tl, sid_raw)
        P = _approx_prob_matrix_per_second(tl_sess, emotions, label_col="pred_emotion", conf_col="pred_conf")
        d_path = plot_emotion_trajectory_pca(P, out_path=os.path.join(sess_dir, f"D_pca_{sid_safe}.png"))

        results[sid_raw] = {
            "A_spectrum": a_path,
            "B_wheel": b1,
            "B_wheel_grad": b2,
            "C_heatmap": c_path,
            "D_pca": d_path,
        }

    return results
