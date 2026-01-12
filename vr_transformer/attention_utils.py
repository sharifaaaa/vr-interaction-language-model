# attention_utils.py
from __future__ import annotations
from typing import Iterable, Optional, Tuple
import os
import numpy as np
import torch

from single_attention import SingleHeadAttention
from .multihead_attention import MultiHeadAttention
from .transformer_blocks import TransformerEncoder
from pooling import pool_tokens, pool_batch
#from config_attention import ATTN_CONFIG, POOLING
import config_attention as CA

# ---------------- helpers ----------------

def _to_tensor(x, dtype, *, is_bool: bool = False) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        t = x.detach().clone()
        return t.bool() if is_bool else t.to(dtype)
    return torch.as_tensor(x, dtype=(torch.bool if is_bool else dtype))

def _print_scores_block(scores: torch.Tensor, *, head: int = 0, k: int = 5) -> None:
    s = scores
    if s.dim() == 4:   # (B,h,L,L)
        s = s[0, head]
    elif s.dim() == 3: # (h,L,L)
        s = s[head]
    elif s.dim() == 2: # (L,L)
        pass
    else:
        raise ValueError("Unexpected scores shape")
    arr = s[:k, :k].detach().cpu().numpy()
    print(np.array_str(arr, precision=4, suppress_small=True))

def _use_cls_from_pooling() -> bool:
    return str(POOLING).lower() == "cls"

def _mask_for_pool(mask: torch.Tensor, add_cls: bool) -> torch.Tensor:
    mask = mask.to(torch.bool)
    if add_cls:
        cls_bit = torch.ones(1, dtype=torch.bool, device=mask.device)
        return torch.cat([cls_bit, mask], dim=0)
    return mask

# --------------- builders ----------------

def build_single_head(device: Optional[torch.device] = None):
    attn = SingleHeadAttention(
        d_model=CA.ATTN_CONFIG["d_model"],
        d_k=CA.ATTN_CONFIG["d_k"],
        d_v=CA.ATTN_CONFIG["d_v"],
        pos_type=CA.ATTN_CONFIG["pos_type"],
        max_len=CA.ATTN_CONFIG["max_len"],
        init=CA.ATTN_CONFIG["init"],
        seed=CA.ATTN_CONFIG["seed"],
        use_cls=_use_cls_from_pooling(),  # CLS follows POOLING
    )
    return attn.to(device) if device else attn

def build_multi_head(device: Optional[torch.device] = None):
    attn = MultiHeadAttention(
        d_model=CA.ATTN_CONFIG["d_model"],
        n_heads=CA.ATTN_CONFIG["n_heads"],
        d_k=CA.ATTN_CONFIG["d_k"],
        d_v=CA.ATTN_CONFIG["d_v"],
        pos_type=CA.ATTN_CONFIG["pos_type"],
        max_len=CA.ATTN_CONFIG["max_len"],
        init=CA.ATTN_CONFIG["init"],
        seed=CA.ATTN_CONFIG["seed"],
        use_cls=_use_cls_from_pooling(),
    )
    return attn.to(device) if device else attn

def build_encoder(device: Optional[torch.device] = None):
    enc = TransformerEncoder()
    return enc.to(device) if device else enc

# -------- single-head runners -------------

def run_single_head_chunk(attn: SingleHeadAttention, x, mask,
                          *, print_scores: bool = True, save_scores: Optional[str] = None):
    x_t = _to_tensor(x, torch.float32)
    m_t = _to_tensor(mask, torch.bool, is_bool=True)

    if attn.WQ.device != x_t.device:
        x_t = x_t.to(attn.WQ.device); m_t = m_t.to(attn.WQ.device)

    scores, weights, out = attn(x_t, m_t)
    mask_for_pool = _mask_for_pool(m_t, _use_cls_from_pooling())
    para_vec = pool_tokens(out, mask_for_pool)

    if print_scores:
        print(f"Scores shape: {tuple(scores.shape)} (L or L+1, L or L+1). Top-left 5x5:")
        _print_scores_block(scores, k=5)
    if save_scores:
        np.save(save_scores, scores.detach().cpu().numpy())
    return scores, weights, out, para_vec

# -------- multi-head runners --------------

def run_multi_head_chunk(attn: MultiHeadAttention, x, mask,
                         *, print_scores: bool = True, save_scores: Optional[str] = None):
    x_t = _to_tensor(x, torch.float32)
    m_t = _to_tensor(mask, torch.bool, is_bool=True)

    if attn.WQ.device != x_t.device:
        x_t = x_t.to(attn.WQ.device); m_t = m_t.to(attn.WQ.device)

    scores, weights, out = attn(x_t, m_t)  # scores:(1,h,L',L'), out:(1,L',E)
    scores = scores.squeeze(0); weights = weights.squeeze(0); out = out.squeeze(0)

    mask_for_pool = _mask_for_pool(m_t, _use_cls_from_pooling())
    para_vec = pool_tokens(out, mask_for_pool)

    if print_scores:
        print(f"Scores shape: {tuple(scores.shape)} (h,L',L'). Head-0 top-left 5x5:")
        _print_scores_block(scores, head=0, k=5)
    if save_scores:
        np.save(save_scores, scores.detach().cpu().numpy())
    return scores, weights, out, para_vec

def run_multi_head_batch(attn: MultiHeadAttention, X, M, ids: Optional[Iterable[str]] = None,
                         *, print_first: bool = True, save_dir: Optional[str] = None):
    X_t = _to_tensor(X, torch.float32)
    M_t = _to_tensor(M, torch.bool, is_bool=True)

    if attn.WQ.device != X_t.device:
        X_t = X_t.to(attn.WQ.device); M_t = M_t.to(attn.WQ.device)

    scores, weights, out = attn(X_t, M_t)  # (B,h,L',L'), (B,h,L',L'), (B,L',E)

    add_cls = _use_cls_from_pooling()
    if add_cls:
        B = M_t.size(0)
        cls_bits = torch.ones(B, 1, dtype=torch.bool, device=M_t.device)
        M_pool = torch.cat([cls_bits, M_t], dim=1)         # (B,L'+)
    else:
        M_pool = M_t

    para_vecs = pool_batch(out, M_pool)

    if print_first:
        print(f"Batch scores shape: {tuple(scores.shape)} (B,h,L',L')")
        print("Sample-0, Head-0, top-left 5x5:")
        _print_scores_block(scores[0], head=0, k=5)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "scores.npy"), scores.detach().cpu().numpy())
        np.save(os.path.join(save_dir, "weights.npy"), weights.detach().cpu().numpy())
        np.save(os.path.join(save_dir, "out.npy"), out.detach().cpu().numpy())
        np.save(os.path.join(save_dir, "para_vecs.npy"), para_vecs.detach().cpu().numpy())
        if ids is not None:
            with open(os.path.join(save_dir, "ids.txt"), "w", encoding="utf-8") as f:
                for cid in ids:
                    f.write(str(cid) + "\n")

    return scores, weights, out, para_vecs

# -------- encoder runners -----------------

def run_encoder_chunk(enc: TransformerEncoder, x, mask, print_scores=True):
    x_t = _to_tensor(x, torch.float32)
    m_t = _to_tensor(mask, torch.bool, is_bool=True)

    scores_list, weights_list, y = enc(x_t, m_t)  # y: (L',E) or (1,L',E)
    if y.dim() == 3:
        y = y.squeeze(0)

    mask_for_pool = _mask_for_pool(m_t, _use_cls_from_pooling())
    para_vec = pool_tokens(y, mask_for_pool)

    if print_scores:
        s_last = scores_list[-1]
        print(f"Encoder layers: {len(scores_list)}; last scores shape: {tuple(s_last.shape)}")
        _print_scores_block(s_last[0] if s_last.dim()==4 else s_last, head=0, k=5)
    return scores_list, weights_list, y, para_vec

def run_encoder_batch(enc: TransformerEncoder, X, M, ids=None, print_first=True):
    X_t = _to_tensor(X, torch.float32)
    M_t = _to_tensor(M, torch.bool, is_bool=True)

    scores_list, weights_list, Y = enc(X_t, M_t)  # Y: (B,L',E)

    add_cls = _use_cls_from_pooling()
    if add_cls:
        B = M_t.size(0)
        cls_bits = torch.ones(B, 1, dtype=torch.bool, device=M_t.device)
        M_pool = torch.cat([cls_bits, M_t], dim=1)
    else:
        M_pool = M_t

    para_vecs = pool_batch(Y, M_pool)

    if print_first:
        s_last = scores_list[-1]
        print(f"Encoder layers: {len(scores_list)}; last scores shape: {tuple(s_last.shape)}")
        _print_scores_block(s_last[0], head=0, k=5)

    return scores_list, weights_list, Y, para_vecs
