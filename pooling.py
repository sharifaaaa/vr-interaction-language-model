# pooling.py
#Purpose: convert token-level outputs (e.g., Transformer outputs of shape (L, D) per chunk) into
# a single fixed-size vector per chunk (or per batch), using a chosen pooling strategy

from __future__ import annotations
from typing import Optional
import torch
import config_attention as CA
def use_cls_from_pooling(pooling: str) -> bool:
    return str(pooling).strip().lower() == "cls"
def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over dim=0 using mask (True=keep)."""
    if mask.dtype is not torch.bool:
        mask = mask.bool()
    return x[mask].mean(dim=0) if mask.any() else x.mean(dim=0)

def masked_max(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Max over dim=0 using mask (True=keep)."""
    if mask.dtype is not torch.bool:
        mask = mask.bool()
    vals, _ = (x[mask].max(dim=0) if mask.any() else x.max(dim=0))
    return vals

def pool_tokens(out: torch.Tensor,
                mask: torch.Tensor,
                pooling: Optional[str] = None) -> torch.Tensor:
    """
    Pool token-level outputs -> one vector.
    out: (L, D), mask: (L,)
    pooling: "mean" | "max" | "cls"  (defaults to config POOLING)
    """
    mode = (pooling or CA.POOLING).lower()
    if mode == "mean":
        return masked_mean(out, mask)
    if mode == "max":
        return masked_max(out, mask)
    if mode == "cls":
        # Assumes you prepended a CLS row (mask[0] == True). If not, fallback to first real.
        if mask[0].item():
            return out[0]
        first_real = torch.nonzero(mask, as_tuple=False)
        return out[first_real[0, 0]] if first_real.numel() > 0 else out[0]
    raise ValueError("POOLING must be one of: 'mean', 'max', 'cls'")

def pool_batch(out: torch.Tensor,
               mask: torch.Tensor,
               pooling: Optional[str] = None) -> torch.Tensor:
    """
    Batch pooling.
    out:  (B, L, D)
    mask: (B, L)
    return: (B, D)
    """
    B = out.size(0)
    return torch.stack([pool_tokens(out[b], mask[b], pooling) for b in range(B)], dim=0)

def _safe_pool(Y, m, pooling, add_cls):
    try:
        return pool_batch(Y, m, pooling=pooling, add_cls_to_mask=add_cls)
    except TypeError:
        return pool_batch(Y, m, pooling=pooling)
