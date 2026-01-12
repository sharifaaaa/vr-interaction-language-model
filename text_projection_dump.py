# text_projection_dump.py
import numpy as np
import pandas as pd
import torch
from typing import Optional, Sequence, Tuple, Union

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# -----------------------------
# Utilities
# -----------------------------

def _device_of(module) -> torch.device:
    try:
        p = next(module.parameters())
        return p.device
    except Exception:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _ensure_proj_cols(df: pd.DataFrame, k: int) -> Sequence[str]:
    cols = [f"text_proj_{i}" for i in range(k)]
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0
    return cols


def _clean_text_mask(df: pd.DataFrame) -> pd.Series:
    txt = df.get("text", "").astype(str).str.strip()
    return txt.ne("") & txt.ne("none") & txt.notna()


def _project_to_kd(embs: Union[torch.Tensor, np.ndarray], k: int) -> torch.Tensor:
    """
    Simple, deterministic projection to k dims (truncate/pad zeros).
    You can replace this with a learned linear head or PCA if desired.
    """
    if not torch.is_tensor(embs):
        embs = torch.as_tensor(embs)

    B, D = embs.shape
    if D == k:
        return embs
    if D > k:
        return embs[:, :k]
    # pad
    pad = embs.new_zeros((B, k - D))
    return torch.cat([embs, pad], dim=1)


# -----------------------------
# Path A: Plain text encoder
# -----------------------------
def attach_text_projection_and_save(
    df: pd.DataFrame,
    embedder,                           # exposes .encode(list_of_texts) OR is callable(list_of_texts)
    out_csv: str,
    proj_dim: int = 16,
    batch_size: int = 64,
    device: Optional[Union[str, torch.device]] = None,
) -> str:
    """
    Encode only rows with non-empty text using a plain text encoder (e.g., DistilBERT wrapper).
    Writes/updates text_proj_* columns and saves CSV.
    """
    # 1) indices with text
    txt = df.get("text", "").astype(str).str.strip()
    has_txt = _clean_text_mask(df)
    idx = np.flatnonzero(has_txt.to_numpy())
    n_txt = len(idx)
    print(f"📝 Text rows to embed: {n_txt} / {len(df)} ({(n_txt/len(df)):.2%})")

    # 2) prepare output columns
    proj_cols = _ensure_proj_cols(df, proj_dim)

    if n_txt == 0:
        # just ensure file exists / is updated
        df.to_csv(out_csv, index=False)
        print(f"💾 Text projection saved to {out_csv} (no text rows).")
        return out_csv

    # 3) device + eval
    device = torch.device(device) if device else (_device_of(embedder))
    if hasattr(embedder, "to"):
        embedder.to(device)
    if hasattr(embedder, "eval"):
        embedder.eval()

    out = np.zeros((n_txt, proj_dim), dtype=np.float32)

    # 4) batching
    with torch.no_grad():
        for s in range(0, n_txt, batch_size):
            e = min(s + batch_size, n_txt)
            batch_texts = txt.iloc[idx[s:e]].tolist()

            # Flexible call
            if hasattr(embedder, "encode"):
                embs = embedder.encode(batch_texts, device=device)     # [B, D]
            else:
                embs = embedder(batch_texts, device=device)            # [B, D]

            if torch.is_tensor(embs):
                embs = embs.detach()

            proj = _project_to_kd(embs, proj_dim)                      # [B, K]
            out[s:e] = proj.cpu().numpy()

    # 5) write back ONLY those indices
    df.loc[df.index[idx], proj_cols] = out

    # zero-out rows with empty text (safety)
    df.loc[~has_txt, proj_cols] = 0.0

    # numeric + clip
    df[proj_cols] = df[proj_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).clip(-10.0, 10.0)
    df.to_csv(out_csv, index=False)
    print(f"💾 Text projection saved to {out_csv}")
    return out_csv


# -----------------------------
# Path B: Grouped TokenEmbedder
# -----------------------------
def _resolve_text_selector(sel) -> Tuple[str, Union[Tuple[int, int], Sequence[int]]]:
    """
    Normalize text selector. Returns ("slice", (s0, s1)) or ("index", [i1, i2, ...]).
    """
    # slice-like
    if isinstance(sel, (tuple, list)) and len(sel) == 2 and all(isinstance(v, int) for v in sel):
        s0, s1 = int(sel[0]), int(sel[1])
        return "slice", (s0, s1)
    # index-list-like
    if isinstance(sel, (list, tuple, np.ndarray)):
        return "index", list(map(int, sel))
    # torch tensor indices
    if torch.is_tensor(sel):
        return "index", sel.to("cpu").tolist()
    raise ValueError("Unsupported text selector in feature_groups['text']")


def attach_text_projection_and_save_grouped(
    df: pd.DataFrame,
    embedder,                           # callable: embedder(df_chunk, pad_to=None, device=…)
    out_csv: str = "chunked_with_textProj.csv",
    skip_if_no_text_group: bool = True,
    show_progress: bool = True,
) -> Optional[str]:
    """
    Works with TokenEmbedder that returns *sequence* embeddings and exposes
    `feature_groups['text']` either as a (start,end) slice or a list of indices.
    We compute projections per chunk (to keep positions aligned), join them back,
    coerce numerics, and zero-out rows with empty text.
    """
    # 0) require grouped mode + text group (warm up once if needed)
    if not getattr(embedder, "use_grouped", False):
        if skip_if_no_text_group:
            print("ℹ️ Skipping dump: USE_GROUPED_EMBEDDING is False (no separable text slice).")
            return None

    if getattr(embedder, "feature_groups", None) is None or "text" not in embedder.feature_groups:
        # warm up once to populate feature_groups
        sample_gid = df["chunk_id"].iloc[0]
        _ = embedder(df[df["chunk_id"] == sample_gid].head(2), pad_to=None)

    if getattr(embedder, "feature_groups", None) is None or "text" not in embedder.feature_groups:
        if skip_if_no_text_group:
            print("ℹ️ Skipping dump: no 'text' slice in feature_groups.")
            return None

    mode, selector = _resolve_text_selector(embedder.feature_groups["text"])

    # 1) iterate groups
    pieces = []
    groups = list(df.groupby("chunk_id", sort=False))
    iterator = tqdm(groups, desc="Dump TEXT proj", leave=False) if (tqdm and show_progress) else groups

    device = _device_of(embedder)
    if hasattr(embedder, "to"):
        embedder.to(device)
    if hasattr(embedder, "eval"):
        embedder.eval()

    with torch.no_grad():
        for _, g in iterator:
            out = embedder(g, pad_to=None, device=device)
            X = out[0] if isinstance(out, (tuple, list)) else out  # [L, d_model]
            if not torch.is_tensor(X):
                X = torch.as_tensor(X, device=device)

            # select text slice/indices
            if mode == "slice":
                s0, s1 = selector
                text_block = X[:, s0:s1]
            else:
                text_block = X[:, selector]

            # projection dim
            k = text_block.shape[1]
            cols = [f"text_proj_{i}" for i in range(k)]
            pieces.append(pd.DataFrame(text_block.detach().cpu().numpy(), columns=cols, index=g.index))

    # 2) merge
    proj_df = pd.concat(pieces).sort_index()
    df_out = df.join(proj_df)

    # 3) sanitize numerics + zero empty-text rows
    proj_cols = [c for c in df_out.columns if c.startswith("text_proj_")]
    df_out[proj_cols] = df_out[proj_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).clip(-10.0, 10.0)
    has_text = _clean_text_mask(df_out)
    df_out.loc[~has_text, proj_cols] = 0.0

    # 4) save
    df_out.to_csv(out_csv, index=False)
    print(f"💾 The result of grouped text projection is written to {out_csv} with {len(proj_cols)} columns.")
    return out_csv


# -----------------------------
# Convenience router
# -----------------------------
def dump_text_projections(
    df: pd.DataFrame,
    embedder,
    out_csv: str = "chunked_with_textProj.csv",
    proj_dim: int = 16,
    **kwargs,
) -> Optional[str]:
    """
    One entry point:
      - If embedder looks grouped and has 'text' in feature_groups → grouped path.
      - Else → plain encoder path (uses .encode if available).
    """
    if getattr(embedder, "use_grouped", False):
        # ensure feature_groups populated (caller might not have warmed up)
        if getattr(embedder, "feature_groups", None) is None or "text" not in embedder.feature_groups:
            sample_gid = df["chunk_id"].iloc[0]
            _ = embedder(df[df["chunk_id"] == sample_gid].head(2), pad_to=None)
        if getattr(embedder, "feature_groups", None) is not None and "text" in embedder.feature_groups:
            return attach_text_projection_and_save_grouped(df, embedder, out_csv=out_csv, **kwargs)

    # fallback to plain path
    return attach_text_projection_and_save(df, embedder, out_csv=out_csv, proj_dim=proj_dim, **kwargs)
