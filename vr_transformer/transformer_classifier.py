# transformer_classifier.py
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
from vr_transformer.attention_utils import build_encoder
from pooling import pool_batch
from config_attention import ATTN_CONFIG, POOLING
from config_pretrain import DROPOUT_HEAD

def _use_cls_from_pooling() -> bool:
    return str(POOLING).lower() == "cls"

def _mask_for_pool(mask: torch.Tensor, add_cls: bool) -> torch.Tensor:
    mask = mask.to(torch.bool)
    if add_cls:

        if mask.dim() == 1:
            return torch.cat([torch.ones(1, dtype=torch.bool, device=mask.device), mask], dim=0)
        if mask.dim() == 2:
            B = mask.size(0)
            return torch.cat([torch.ones(B, 1, dtype=torch.bool, device=mask.device), mask], dim=1)
    return mask

class TransformerClassifier(nn.Module):
    """
    Transformer encoder + pooling (mean/max/cls) + linear head.
    Uses DROPOUT_HEAD from config_pretrain.
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self.encoder = build_encoder()                   # respects POOLING== "cls" for CLS handling
        d_model = ATTN_CONFIG["d_model"]
        self.head_drop = nn.Dropout(DROPOUT_HEAD)
        self.classifier = nn.Linear(d_model, num_classes)


    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        scores_list, weights_list, Y = self.encoder(x, mask)     # Y: (B,L',E) or (L',E)

        if Y.dim() == 2:  # normalize to batch
            Y = Y.unsqueeze(0)
            if mask is not None and mask.dim() == 1:
                mask = mask.unsqueeze(0)

        if mask is None:
            B, Lp, _ = Y.shape
            base_len = Lp - (1 if _use_cls_from_pooling() else 0)
            mask = torch.ones(B, base_len, dtype=torch.bool, device=Y.device)

        M_pool = _mask_for_pool(mask, _use_cls_from_pooling())
        pooled = pool_batch(Y, M_pool)                   # (B,E)
        logits = self.classifier(self.head_drop(pooled)) # (B,C)
        return logits, pooled
