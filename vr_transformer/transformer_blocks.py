# transformer_blocks.py
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config_attention import ATTN_CONFIG, POOLING

from .multihead_attention import MultiHeadAttention


def _use_cls_from_pooling() -> bool:
    return str(POOLING).lower() == "cls"
class TransformerEncoderLayer(nn.Module):
    """
    PreNorm encoder layer:  LN -> MHA -> residual  ;  LN -> FFN -> residual
    Uses your MultiHeadAttention (handles positions and optional CLS).
    """


    def __init__(self,
                 d_model=ATTN_CONFIG["d_model"],
                 n_heads=ATTN_CONFIG["n_heads"],
                 d_k=ATTN_CONFIG["d_k"],
                 d_v=ATTN_CONFIG["d_v"],
                 pos_type=ATTN_CONFIG["pos_type"],
                 max_len=ATTN_CONFIG["max_len"],
                 dropout=ATTN_CONFIG.get("dropout", 0.1),
                 ffn_mult=ATTN_CONFIG.get("ffn_mult", 4),
                 layer_norm_eps=ATTN_CONFIG.get("layer_norm_eps", 1e-5),
                 init=ATTN_CONFIG["init"],
                 seed=ATTN_CONFIG["seed"],
                 use_cls=_use_cls_from_pooling()):
        super().__init__()
        self.mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v,
                                      pos_type=pos_type, max_len=max_len,
                                      init=init, seed=seed, use_cls=use_cls)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Linear(ffn_mult * d_model, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
                   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # PreNorm
        x_norm = self.norm1(x)

        # Run attention (may change length if use_cls=True)
        scores, weights, mha_out = self.mha(x_norm, mask)   # mha_out: (B,L',E) or (L',E)

        # Normalize shapes to batch
        if x_norm.dim() == 2:
            x_res = x_norm.unsqueeze(0)                     # (1,L,E)
        else:
            x_res = x_norm                                  # (B,L,E)

        # ------- Align residual length to mha_out length -------
        L_res = x_res.size(1)
        L_mha = mha_out.size(1)
        if L_res != L_mha:
            # Typical case with CLS inside MHA: L_mha = L_res + 1
            if getattr(self.mha, "use_cls", False) and (L_mha == L_res + 1):
                B = mha_out.size(0)
                # prepend same learnable CLS used in MHA to the residual branch
                cls_res = self.mha.cls.expand(B, 1, -1).to(x_res.dtype)
                x_res = torch.cat([cls_res, x_res], dim=1)  # (B,L_res+1,E)
            else:
                # Fallback: pad/truncate residual to match MHA length (shouldn't happen normally)
                if L_res < L_mha:
                    pad_len = L_mha - L_res
                    x_res = F.pad(x_res, (0, 0, 0, pad_len), value=0.0)  # pad on length dim
                else:
                    x_res = x_res[:, :L_mha, :]

        # Residual add
        y = x_res + self.drop(mha_out)

        # FFN (PreNorm + residual)
        y2 = self.norm2(y)
        y2 = y2 + self.drop(self.ffn(y2))
        return scores, weights, y2
class TransformerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        num_layers = ATTN_CONFIG["num_layers"]
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(use_cls=(i == 0 and _use_cls_from_pooling()))
            for i in range(num_layers)
        ])


    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        scores_list, weights_list = [], []
        y = x
        m = mask
        for layer in self.layers:
            scores, weights, y = layer(y, m)
            scores_list.append(scores)
            weights_list.append(weights)
        return scores_list, weights_list, y  # y: (B,L',E) or (L',E)
