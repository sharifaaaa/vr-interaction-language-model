# av_plotter.py
"""
Generate Arousal–Valence plots per session from timeline_emotions.csv

A) plot_av_trajectory_per_session → scatter + chronological path in AV plane
B) plot_av_session_mean           → single weighted-mean point in AV plane
"""

from __future__ import annotations
import os, re, argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save-only backend
import matplotlib.pyplot as plt

# ───────── import the path sanitizer from utils_paths (with a safe fallback) ─────────
try:
    # expected to exist in your repo
    from utils_paths import safe_path_id   # type: ignore
except Exception:
    # robust fallback if utils_paths is unavailable
    def safe_path_id(s: str) -> str:
        s = str(s)
        # remove Windows/Unix-invalid characters and collapse runs
        s = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", s)
        s = re.sub(r"\s+", "_", s).strip("_")
        # keep a reasonable length
        return s[:160] if len(s) > 160 else s

# ───────────────────────────── Config: data→AV mapping ─────────────────────────
EMOTION_TO_VA: Dict[str, Tuple[float, float]] = {
    "joy":          ( 0.80,  0.70),
    "stress":       (-0.60,  0.80),
    "frustration":  (-0.70,  0.60),
    "neutral":      ( 0.00,  0.00),
    "fear":         (-0.80,  0.90),
    "sad":          (-0.80, -0.60),
    "relaxed":      ( 0.60, -0.50),
    "calm":         ( 0.70, -0.40),
    "anger":        (-0.90,  0.70),
    "surprise":     ( 0.40,  0.90),
}

EMOTION_COLORS: Dict[str, str] = {
    "joy":         "#FDB813",
    "stress":      "#D62728",
    "frustration": "#FF7F0E",
    "neutral":     "#7F7F7F",
    "fear":        "#6A5ACD",
    "sad":         "#1F77B4",
    "relaxed":     "#2CA02C",
    "calm":        "#17BECF",
    "anger":       "#8B0000",
    "surprise":    "#9467BD",
}

# Background annotation points (kept inside quadrants)
ANNOTATION_POINTS: Dict[str, Tuple[float, float]] = {
    "Infuriated":  (-0.80,  0.93), "Angry":  (-0.70,  0.84), "Frustrated": (-0.70, 0.55),
    "Annoyed":     (-0.40,  0.44), "Alarmed":(-0.30,  0.20), "Fear":       (-0.88, 0.72),
    "Excited":     ( 0.28,  0.88), "Happy":  ( 0.76,  0.70), "Delighted":  ( 0.46, 0.56),
    "Glad":        ( 0.34,  0.38), "Amused": ( 0.18,  0.16), "Curious":    ( 0.80, 0.30),
    "Miserable":   (-0.84, -0.22), "Depressed":(-0.70, -0.48), "Gloomy":   (-0.60, -0.36),
    "Sad":         (-0.50, -0.60), "Tired": (-0.60, -0.88), "Bored":      (-0.32, -0.62),
    "Content":     ( 0.66, -0.10), "Satisfied":( 0.32, -0.28), "Relaxed":  ( 0.56, -0.50),
    "Calm":        ( 0.40, -0.82),
}

# ─────────────────────────────────── Helpers ───────────────────────────────────
def _ensure_dir(path: str) -> None:
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

def _parse_session_from_chunk(chunk_id: str) -> str:
    """
    Extract clean session stem from chunk_id like:
      'session_xxx.json::something' -> 'session_xxx'
    """
    s = str(chunk_id)
    m = re.search(r"(session_[^:]+?)\.json", s)
    return m.group(1) if m else s.split(":", 1)[0]

def _derive_session_id(df: pd.DataFrame) -> pd.Series:
    """
    Prefer a clean ID if available:
      1) 'db_session_id'  (already clean & stable)
      2) 'session_id'     (but may contain ':')
      3) derived from 'chunk_id'
    Then sanitize with safe_path_id for filenames.
    """
    if "db_session_id" in df.columns:
        base = df["db_session_id"].astype(str)
    elif "session_id" in df.columns:
        base = df["session_id"].astype(str)
    elif "chunk_id" in df.columns:
        base = df["chunk_id"].map(_parse_session_from_chunk)
    else:
        raise ValueError("CSV must contain one of: 'db_session_id', 'session_id', or 'chunk_id'.")
    return base.map(safe_path_id)

def _find_time_cols(df: pd.DataFrame) -> Tuple[str, str]:
    c0 = ["t0", "t_start_s", "start_s", "start_time", "start"]
    c1 = ["t1", "t_end_s", "end_s", "end_time", "end"]
    t0 = next((c for c in c0 if c in df.columns), None)
    t1 = next((c for c in c1 if c in df.columns), None)
    if not t0 or not t1:
        raise ValueError("CSV must contain time columns (e.g., t_start_s/t_end_s).")
    return t0, t1

def _add_valence_arousal(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    to_v = lambda e: EMOTION_TO_VA.get(str(e).lower(), (np.nan, np.nan))[0]
    to_a = lambda e: EMOTION_TO_VA.get(str(e).lower(), (np.nan, np.nan))[1]
    out = df.copy()
    out["valence"] = out[label_col].map(to_v)
    out["arousal"] = out[label_col].map(to_a)
    return out

def _color_for_emotion(e: str) -> str:
    return EMOTION_COLORS.get(str(e).lower(), "#333333")

def _draw_av_template(ax: plt.Axes) -> None:
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
    ax.axvspan(-1, 0, ymin=0.5, ymax=1.0, facecolor="#ead6e3", alpha=0.45, zorder=0)
    ax.axvspan( 0, 1, ymin=0.5, ymax=1.0, facecolor="#fff6d9", alpha=0.45, zorder=0)
    ax.axvspan(-1, 0, ymin=0.0, ymax=0.5, facecolor="#d8e8ff", alpha=0.45, zorder=0)
    ax.axvspan( 0, 1, ymin=0.0, ymax=0.5, facecolor="#e3f4e6", alpha=0.45, zorder=0)
    ax.axhline(0, color="#2f2f2f", linewidth=1.2, zorder=2)
    ax.axvline(0, color="#2f2f2f", linewidth=1.2, zorder=2)
    ax.set_xlabel("Valence  (Unpleasant → Pleasant)")
    ax.set_ylabel("Arousal  (Low → High)")
    ax.grid(True, alpha=0.20, zorder=1)
    for word, (vx, ay) in ANNOTATION_POINTS.items():
        ax.text(vx, ay, word, ha="center", va="center",
                fontsize=11, color="#3a3a3a", alpha=0.46, zorder=1)
    ax.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1],
            color="#666666", linewidth=0.8, alpha=0.5, zorder=2)

# ──────────────────────────────── A) Trajectory ───────────────────────────────
def plot_av_trajectory_per_session(csv_path: str,
                                   out_dir: str = "./av_plots/trajectory",
                                   label_col: str = "pred_emotion") -> List[str]:
    _ensure_dir(out_dir)
    df = pd.read_csv(csv_path)

    # Compute a safe session_id column for grouping & filenames
    df = df.copy()
    df["__sid_safe__"] = _derive_session_id(df)

    t0, _ = _find_time_cols(df)
    df = _add_valence_arousal(df, label_col).dropna(subset=["valence", "arousal"]).copy()

    out_paths: List[str] = []
    for sid_safe, g in df.groupby("__sid_safe__", sort=False):
        g = g.sort_values(t0)
        if g.empty:
            continue

        fig, ax = plt.subplots(figsize=(6, 6))
        _draw_av_template(ax)

        ax.plot(g["valence"].values, g["arousal"].values,
                color="#666666", alpha=0.6, linewidth=1.6, zorder=3)

        sizes = np.clip(
            g.get("pred_conf", pd.Series(0.6, index=g.index)).astype(float).to_numpy() * 120,
            30, 220
        )
        colors = [_color_for_emotion(e) for e in g[label_col]]
        ax.scatter(g["valence"], g["arousal"], c=colors, s=sizes,
                   edgecolors="black", linewidths=0.5, alpha=0.96, zorder=4)

        ax.set_title(f"AV Trajectory — {sid_safe}")

        # optional legend (unique emotions in this session)
        uniq = g[label_col].astype(str).str.lower().unique().tolist()
        if uniq:
            handles = [plt.Line2D([0], [0], marker="o", color="w",
                                  markerfacecolor=_color_for_emotion(em),
                                  markeredgecolor="black", markersize=8, label=em)
                       for em in uniq]
            # ax.legend(handles=handles, title="Emotion", loc="lower right", framealpha=0.9)

        outp = os.path.join(out_dir, f"av_trajectory_{sid_safe}.png")
        fig.tight_layout(); fig.savefig(outp, dpi=150); plt.close(fig)
        out_paths.append(outp)

    return out_paths

# ───────────────────────────── B) Weighted mean ───────────────────────────────
def plot_av_session_mean(csv_path: str,
                         out_dir: str = "./av_plots/mean",
                         label_col: str = "pred_emotion") -> List[str]:
    _ensure_dir(out_dir)
    df = pd.read_csv(csv_path)

    df = df.copy()
    df["__sid_safe__"] = _derive_session_id(df)

    t0, t1 = _find_time_cols(df)
    df = _add_valence_arousal(df, label_col).dropna(subset=["valence", "arousal"]).copy()

    d_end = pd.to_numeric(df[t1], errors="coerce")
    d_start = pd.to_numeric(df[t0], errors="coerce")
    df["duration"] = (d_end - d_start).fillna(0.0).clip(lower=0.0)

    out_paths: List[str] = []
    for sid_safe, g in df.groupby("__sid_safe__", sort=False):
        if g.empty:
            continue

        w = g["duration"].to_numpy(float)
        w = np.where(w <= 0, 1.0, w)
        mean_v = np.average(g["valence"].to_numpy(float), weights=w)
        mean_a = np.average(g["arousal"].to_numpy(float), weights=w)

        fig, ax = plt.subplots(figsize=(6, 6))
        _draw_av_template(ax)

        ax.scatter([mean_v], [mean_a], marker="x", color="red",
                   s=180, linewidths=3, label="Weighted mean", zorder=5)

        cent = (
            g.groupby(label_col)[["valence", "arousal", "duration"]]
             .apply(lambda d: pd.Series({
                 "v": np.average(d["valence"].to_numpy(float),
                                 weights=np.where(d["duration"] <= 0, 1.0, d["duration"])),
                 "a": np.average(d["arousal"].to_numpy(float),
                                 weights=np.where(d["duration"] <= 0, 1.0, d["duration"])),
                 "n": len(d),
             }))
             .reset_index()
        )
        for _, row in cent.iterrows():
            em = str(row[label_col])
            ax.scatter([row["v"]], [row["a"]],
                       color=_color_for_emotion(em), s=120,
                       edgecolors="black", linewidths=0.5,
                       label=f"{em} (n={int(row['n'])})", zorder=4)

        ax.set_title(f"AV Mean — {sid_safe}")
        # ax.legend(loc="lower right", framealpha=0.9)

        outp = os.path.join(out_dir, f"av_mean_{sid_safe}.png")
        fig.tight_layout(); fig.savefig(outp, dpi=150); plt.close(fig)
        out_paths.append(outp)

    return out_paths

# ─────────────────────────────────── CLI ──────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Generate Arousal–Valence plots per session.")
    ap.add_argument("--csv", required=True, help="Path to timeline_emotions.csv")
    ap.add_argument("--out", default="./av_plots", help="Output root directory")
    ap.add_argument("--label-col", default="pred_emotion", help="Column containing labels")
    args = ap.parse_args()

    traj_dir = os.path.join(args.out, "trajectory")
    mean_dir = os.path.join(args.out, "mean")
    _ensure_dir(traj_dir); _ensure_dir(mean_dir)

    print("▶ Generating AV trajectories…")
    a = plot_av_trajectory_per_session(args.csv, out_dir=traj_dir, label_col=args.label_col)
    print(f"  saved {len(a)} files → {traj_dir}")

    print("▶ Generating AV means…")
    b = plot_av_session_mean(args.csv, out_dir=mean_dir, label_col=args.label_col)
    print(f"  saved {len(b)} files → {mean_dir}")

if __name__ == "__main__":
    main()
