# pretrain_dataset.py
from __future__ import annotations
from typing import Dict, Tuple, List, Optional
import random

import torch
from torch.utils.data import Dataset
import pandas as pd
from config_pretrain import  MASK_TOKEN_SCHEME as MS, AUGS
from config_attention import ATTN_CONFIG
from vr_transformer.attention_runner import get_chunk_batch
from utils_labels import build_label_mapping


# ----------------------------
# Basic helpers
"""
Piece 1 — Header, imports, and basic helpers
What this does:
Imports your config knobs and the existing get_chunk_batch (so embeddings/masks match the rest of your pipeline).

Adds a tiny utility to compute valid length from the padding mask.
"""
# ----------------------------

def _valid_len_from_mask(pad_mask: torch.Tensor) -> int:
    """
    pad_mask: [L] with True for PAD positions, False for VALID.
    Returns number of valid tokens.
    """
    return int((~pad_mask).sum().item())


# ----------------------------
# Masking utilities (MTM)
"""
Piece 2 — Mask index selection + feature-group dropout + MTM masking
What this does:

Span+random masking on time positions.

Optional feature-group dropout (if your embedder.feature_groups exists).

Produces masked input, original target, and a boolean mask of where to compute MTM loss
"""
# ----------------------------

def _span_mask_indices(L_valid: int, n_mask: int, min_span: int, max_span: int) -> List[int]:
    """
        Create indices (0..L_valid-1) to mask, using short spans, totaling ≈ n_mask.
    """

    if n_mask <= 0 or L_valid <= 0:
        return []
    masked: List[int] = []
    remaining = n_mask
    while remaining > 0:
        span = random.randint(min_span, max_span)
        span = min(span, remaining)
        if L_valid - span <= 0:
            start = 0
        else:
            start = random.randint(0, L_valid - span)
        masked.extend(range(start, start + span))
        remaining -= span
        if len(masked) >= n_mask:
            break
    masked = sorted(set(i for i in masked if 0 <= i < L_valid))
    return masked[:n_mask]


def _random_non_overlap_indices(L_valid: int, n_mask: int, excluded: set) -> List[int]:
    pool = [i for i in range(L_valid) if i not in excluded]
    if not pool:
        return []
    n = min(n_mask, len(pool))
    return random.sample(pool, n)


def _apply_group_dropout(
    x: torch.Tensor,
    pad_mask: torch.Tensor,
    groups: Optional[Dict[str, Tuple[int, int]]],
    var_p: float
) -> torch.Tensor:
    """
    Optional variable-wise (feature-group) dropout on the EMBEDDED token.
    groups: dict like {"behavioral": (start_idx, end_idx_exclusive), ...}
    """
    if var_p <= 0:
        return x
    x2 = x.clone()
    if not groups:
        return x2
    for _, (lo, hi) in groups.items():
        if random.random() < var_p:
            x2[:, lo:hi] = 0.0
    return x2


def _apply_mtm_mask(
    x: torch.Tensor,
    pad_mask: torch.Tensor,
    group_ranges: Optional[Dict[str, Tuple[int, int]]] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create Masked Time-series Modeling triplet over EMBEDDED tokens.
    Returns: (x_mtm, target, mask_bool)
      x_mtm   : [L, D] input after masking
      target  : [L, D] original embeddings (reconstruction target)
      mask_bool: [L]   True where MTM loss applies
    """
    L, D = x.shape
    x_mtm = x.clone()
    target = x.clone()

    # Optional feature-group dropout to encourage cross-channel imputation
    var_p = MS.get("var_p", 0.0)
    x_mtm = _apply_group_dropout(x_mtm, pad_mask, group_ranges, var_p)

    L_valid = _valid_len_from_mask(pad_mask)
    if L_valid == 0:
        return x_mtm, target, torch.zeros(L, dtype=torch.bool)

    token_p = MS.get("token_p", 0.15)
    span_frac = MS.get("span_frac", 0.5)
    min_span = MS.get("min_span", 2)
    max_span = MS.get("max_span", 12)

    n_mask_total = max(1, int(round(token_p * L_valid)))
    n_span = int(round(span_frac * n_mask_total))
    n_rand = n_mask_total - n_span

    span_idxs = _span_mask_indices(L_valid, n_span, min_span, max_span)
    span_set = set(span_idxs)
    rand_idxs = _random_non_overlap_indices(L_valid, n_rand, excluded=span_set)
    mask_positions_valid = sorted(set(span_idxs + rand_idxs))

    mask_bool = torch.zeros(L, dtype=torch.bool)
    for i in mask_positions_valid:
        mask_bool[i] = True

    # BERT-style 80/10/10 on the valid prefix
    for i in mask_positions_valid:
        r = random.random()
        if r < 0.8:
            x_mtm[i] = 0.0  # zero vector as [MASK]
        elif r < 0.9:
            j = random.randint(0, L_valid - 1)
            x_mtm[i] = x[j]  # replace with another valid token
        else:
            pass  # keep original token; still predict it

    return x_mtm, target, mask_bool


# ----------------------------
# Light augmentations (contrastive)
"""
Piece 3 — Light, emotion-safe augmentations (for contrastive views)
These are conservative augmentations to preserve emotion dynamics.
"""
# ----------------------------

def _temporal_crop(
    x: torch.Tensor,
    pad_mask: torch.Tensor,
    cmin: float,
    cmax: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Crop a valid contiguous subsequence and left-pad back to length L.
    """
    L, D = x.shape
    L_valid = _valid_len_from_mask(pad_mask)
    if L_valid <= 1:
        return x.clone(), pad_mask.clone()

    keep_frac = random.uniform(cmin, cmax)
    keep = max(1, int(round(keep_frac * L_valid)))
    if keep >= L_valid:
        return x.clone(), pad_mask.clone()

    start = random.randint(0, L_valid - keep)
    end = start + keep
    x_new = torch.zeros_like(x)
    m_new = torch.ones_like(pad_mask)
    x_new[:keep] = x[start:end]
    m_new[:keep] = False
    return x_new, m_new


def _time_jitter(
    x: torch.Tensor,
    pad_mask: torch.Tensor,
    steps: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Circularly shift the valid prefix by up to ±steps.
    """
    L, D = x.shape
    L_valid = _valid_len_from_mask(pad_mask)
    if L_valid <= 1 or steps <= 0:
        return x, pad_mask
    shift = random.randint(-steps, steps)
    if shift == 0:
        return x, pad_mask
    x_valid = x[:L_valid].clone()
    if shift > 0:
        x_j = torch.cat([x_valid[shift:], x_valid[:shift]], dim=0)
    else:
        s = -shift
        x_j = torch.cat([x_valid[-s:], x_valid[:-s]], dim=0)
    x_out = x.clone()
    x_out[:L_valid] = x_j
    return x_out, pad_mask


def _feature_dropout(x: torch.Tensor, pad_mask: torch.Tensor, p: float) -> torch.Tensor:
    if p <= 0:
        return x
    L, D = x.shape
    drop = (torch.rand(D) < p).float()
    x2 = x.clone()
    x2[:, drop.bool()] = 0.0
    return x2


def _gauss_jitter(x: torch.Tensor, pad_mask: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return x
    noise = torch.randn_like(x) * sigma
    noise[pad_mask] = 0.0  # don't change padded region
    return x + noise


def _make_contrastive_views(
    x: torch.Tensor,
    pad_mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Two views: crop -> jitter -> featdrop -> gaussian noise.
    Returns: (v1_x, v1_mask, v2_x, v2_mask)
    """
    cmin, cmax = AUGS.get("temporal_crop_min", 0.9), AUGS.get("temporal_crop_max", 1.0)
    jitter = AUGS.get("time_jitter_steps", 1)
    feat_p = AUGS.get("feat_dropout_p", 0.1)
    sigma = AUGS.get("gauss_sigma", 0.02)

    v1, m1 = _temporal_crop(x, pad_mask, cmin, cmax)
    v1, m1 = _time_jitter(v1, m1, jitter)
    v1 = _feature_dropout(v1, m1, feat_p)
    v1 = _gauss_jitter(v1, m1, sigma)

    v2, m2 = _temporal_crop(x, pad_mask, cmin, cmax)
    v2, m2 = _time_jitter(v2, m2, jitter)
    v2 = _feature_dropout(v2, m2, feat_p)
    v2 = _gauss_jitter(v2, m2, sigma)

    return v1, m1, v2, m2


# ----------------------------
# Unlabeled dataset (for pretraining)
"""
Piece 4 — UnlabeledWindowsDataset (+ factory) for pretraining
Uses the existing embedder + get_chunk_batch to ensure padding/masks stay consistent.
"""
# ----------------------------

class UnlabeledWindowsDataset(Dataset):
    """
    Each item returns a dict for MTM + contrastive:
      {
        'mtm_input'   : [L,D],
        'mtm_target'  : [L,D],
        'mtm_mask'    : [L] (bool),
        'attn_mask_mtm': [L] (bool),
        'v1_x'        : [L,D],
        'v1_mask'     : [L] (bool),
        'v2_x'        : [L,D],
        'v2_mask'     : [L] (bool),
      }
    """
    def __init__(self, df: pd.DataFrame, embedder, mask_scheme: Dict, augs: Dict, pooling: str):
        self.df = df.copy()
        self.embedder = embedder
        self.mask_scheme = mask_scheme
        self.augs = augs
        self.pooling = pooling
        # Optional feature-group ranges provided by embedder (if available)
        self.group_ranges: Optional[Dict[str, Tuple[int, int]]] = getattr(embedder, "feature_groups", None)
        self.chunk_ids: List = sorted(self.df["chunk_id"].unique().tolist())


    def __len__(self) -> int:
        return len(self.chunk_ids)

    def _embed_chunk(self, cid) -> Tuple[torch.Tensor, torch.Tensor]:
        df_c = self.df[self.df["chunk_id"] == cid]
        X, M, _ = get_chunk_batch(df_c, self.embedder, pad_to=ATTN_CONFIG["max_len"])
        # X: [1, L, D]; M: [1, L]

        return X[0], M[0]

    def __getitem__(self, idx: int):
        cid = self.chunk_ids[idx]
        x, pad_mask = self._embed_chunk(cid)

        # MTM triplet
        x_mtm, target, mask_bool = _apply_mtm_mask(x, pad_mask, self.group_ranges)

        # Contrastive views
        v1_x, v1_m, v2_x, v2_m = _make_contrastive_views(x, pad_mask)

        return {
            'mtm_input': x_mtm,
            'mtm_target': target,
            'mtm_mask': mask_bool,
            'attn_mask_mtm': pad_mask,
            'v1_x': v1_x,
            'v1_mask': v1_m,
            'v2_x': v2_x,
            'v2_mask': v2_m,
        }


def make_unlabeled_windows_dataset(df: pd.DataFrame, embedder, mask_scheme: Dict, augs: Dict, pooling: str):

    return UnlabeledWindowsDataset(df, embedder, mask_scheme, augs, pooling)

# ----------------------------
# Collate function (pretraining)
"""
Piece 5 — Collate function for pretraining
This will be used by your train_pretrain.py DataLoader.
"""
# ----------------------------

def pretrain_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Stacks tensor fields across the batch; leaves non-tensors as lists (if any).
    """
    keys = batch[0].keys()
    out: Dict[str, torch.Tensor] = {}
    for k in keys:
        if isinstance(batch[0][k], torch.Tensor):
            out[k] = torch.stack([b[k] for b in batch], dim=0)
        else:
            out[k] = [b[k] for b in batch]
    return out

# ----------------------------
# Labeled dataset (for fine-tuning)
"""
Piece 6 — Labeled dataset + collate for fine-tuning
This keeps your future train_finetune.py straightforward.
"""
# ----------------------------

class LabeledWindowsDataset(Dataset):
    """
    Each item: {'x':[L,D], 'mask':[L] (bool), 'y': int}
    Uses FULL df to re-embed windows consistently via get_chunk_batch.
    """
    def __init__(self, df_full: pd.DataFrame, embedder, label_col: str, class_to_idx: Dict[str, int]):
        df_lab = df_full[df_full[label_col].notna()].copy()
        df_lab = df_lab.groupby("chunk_id").first().reset_index()  # one label per chunk
        self.df_full = df_full
        self.df_lab = df_lab
        self.embedder = embedder
        self.label_col = label_col
        self.class_to_idx = class_to_idx
        self.chunk_ids: List = self.df_lab["chunk_id"].tolist()
        self.session_ids: List = self.df_lab["session_id"].to_list()

    def __len__(self) -> int:
        return len(self.chunk_ids)



    def __getitem__(self, idx: int):
        cid = self.chunk_ids[idx]
        rows_full = self.df_full[self.df_full["chunk_id"] == cid]
        X, M, _ = get_chunk_batch(rows_full, self.embedder, pad_to=ATTN_CONFIG["max_len"])
        x = X[0];
        mask = M[0]

        row = self.df_lab[self.df_lab["chunk_id"] == cid].iloc[0]
        y = self.class_to_idx[str(row[self.label_col])]

        # prefer 'session_id', else 'db_session_id', else empty
        if "session_id" in rows_full.columns:
            sid = rows_full["session_id"].iloc[0]
        elif "db_session_id" in rows_full.columns:
            sid = rows_full["db_session_id"].iloc[0]
        else:
            sid = ""

        return {
            'x': x,
            'mask': mask,
            'y': y,
            'chunk_id': cid,  # <-- expose as chunk_id (not cid)
            'db_session_id': sid,  # keep name used by your CSV
            'session_id': sid,  # optional alias
        }


def labeled_collate_fn(batch):

    X = torch.stack([b['x'] for b in batch], dim=0)
    M = torch.stack([b['mask'] for b in batch], dim=0)
    y = torch.tensor([b['y'] for b in batch], dtype=torch.long)
    out = {'x': X, 'mask': M, 'y': y}
    if 'chunk_id' in batch[0]:      out['chunk_id'] = [str(b['chunk_id']) for b in batch]
    if 'db_session_id' in batch[0]: out['db_session_id'] = [str(b['db_session_id']) for b in batch]
    if 'session_id' in batch[0]:    out['session_id'] = [str(b['session_id']) for b in batch]
    return out


def make_labeled_windows_dataset(df: pd.DataFrame, embedder, label_col: str):
    #labels = sorted(df[label_col].dropna().astype(str).unique().tolist())
    #class_to_idx = {c: i for i, c in enumerate(labels)}
    class_to_idx = build_label_mapping(df)
    ds = LabeledWindowsDataset(df, embedder, label_col, class_to_idx)
    return ds, class_to_idx



