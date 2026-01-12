# single_attention.py
from typing import Optional, Literal
import math
import torch
import torch.nn as nn

InitName = Literal[
    "xavier_uniform", "xavier_normal",
    "kaiming_uniform", "kaiming_normal",
    "zeros", "eye"
]
# NOTE: now includes "rope"
PosType = Literal["learned", "sinusoidal", "none", "rope"]


# ----------------------------- utils -----------------------------

def init_tensor_(t: torch.Tensor, method: InitName = "xavier_uniform") -> None:
    if method == "xavier_uniform":
        nn.init.xavier_uniform_(t)
    elif method == "xavier_normal":
        nn.init.xavier_normal_(t)
    elif method == "kaiming_uniform":
        nn.init.kaiming_uniform_(t, a=math.sqrt(5))
    elif method == "kaiming_normal":
        nn.init.kaiming_normal_(t, a=math.sqrt(5))
    elif method == "zeros":
        nn.init.zeros_(t)
    elif method == "eye":
        nn.init.zeros_(t)
        m, n = t.shape
        eye = torch.eye(min(m, n), device=t.device, dtype=t.dtype)
        t[:eye.size(0), :eye.size(1)] = eye
    else:
        raise ValueError(f"Unknown init method: {method}")


def sinusoidal_pe(max_len: int, d_model: int, device=None, dtype=None) -> torch.Tensor:
    """[max_len, d_model] fixed sinusoidal PE (Vaswani et al.)."""
    pe = torch.zeros(max_len, d_model, device=device, dtype=dtype)
    position = torch.arange(0, max_len, device=device, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device, dtype=torch.float32)
                         * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.to(dtype if dtype is not None else torch.float32)


def _rope_cos_sin(seq_len: int, dim: int, device=None, dtype=None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build RoPE cos/sin tables with shape [seq_len, dim], interleaved for pairs.
    """
    assert dim % 2 == 0, "RoPE requires an even dimension (d_k)."
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)  # [L]
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))  # [dim/2]
    freqs = torch.einsum("i,j->ij", pos, inv_freq)  # [L, dim/2]
    cos = torch.cos(freqs).repeat_interleave(2, dim=1)  # [L, dim]
    sin = torch.sin(freqs).repeat_interleave(2, dim=1)  # [L, dim]
    cos = cos.to(dtype if dtype is not None else torch.float32)
    sin = sin.to(dtype if dtype is not None else torch.float32)
    return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """(x_even, x_odd) -> (-x_odd, x_even)."""
    x_even = x[..., ::2]
    x_odd  = x[..., 1::2]
    out = torch.empty_like(x)
    out[..., ::2] = -x_odd
    out[..., 1::2] =  x_even
    return out


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [L, d], cos/sin: [L, d]
    return (x * cos) + (_rotate_half(x) * sin)


# ---------------------- Single-head attention ---------------------

class SingleHeadAttention(nn.Module):
    """
    Single-head scaled dot-product attention for ONE chunk (no batch).

    Inputs
    ------
    x:    [L, E]    token embeddings (E == d_model)
    mask: [L]       bool/0-1; True/1 for real tokens, False/0 for padding

    Returns
    -------
    scores:  [L', L']   pre-softmax dot-product (masked on key side)
    weights: [L', L']   attention weights (rows sum to 1)
    out:     [L', d_v]  attention output (weighted sum of V)
    """

    def __init__(
        self,
        d_model: int,
        d_k: int,
        d_v: Optional[int] = None,
        init: InitName = "xavier_uniform",
        seed: Optional[int] = None,
        pos_type: PosType = "learned",
        max_len: int = 240,
        use_cls: bool = True,
    ):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_k if d_v is None else d_v
        self.use_cls = use_cls
        self.pos_type = pos_type

        # projections
        self.WQ = nn.Parameter(torch.empty(d_model, self.d_k))
        self.WK = nn.Parameter(torch.empty(d_model, self.d_k))
        self.WV = nn.Parameter(torch.empty(d_model, self.d_v))
        self.reset_parameters(init)

        # optional CLS token
        if self.use_cls:
            self.cls = nn.Parameter(torch.zeros(1, 1, d_model))

        # positional encodings (allocate for +1 CLS if present)
        pos_len = max_len + (1 if self.use_cls else 0)

        if self.pos_type == "learned":
            self.pos_embed = nn.Embedding(pos_len, d_model)
            nn.init.normal_(self.pos_embed.weight, std=0.02)
            self.register_buffer("sin_pe", None, persistent=False)
            self.register_buffer("rope_cos", None, persistent=False)
            self.register_buffer("rope_sin", None, persistent=False)

        elif self.pos_type == "sinusoidal":
            self.pos_embed = None
            self.register_buffer("sin_pe", sinusoidal_pe(pos_len, d_model), persistent=False)
            self.register_buffer("rope_cos", None, persistent=False)
            self.register_buffer("rope_sin", None, persistent=False)

        elif self.pos_type == "rope":
            # RoPE acts on Q/K (d_k), not on the token x directly.
            self.pos_embed = None
            self.register_buffer("sin_pe", None, persistent=False)
            cos, sin = _rope_cos_sin(pos_len, self.d_k)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

        else:  # "none" or unknown → no absolute PE, no RoPE
            self.pos_embed = None
            self.register_buffer("sin_pe", None, persistent=False)
            self.register_buffer("rope_cos", None, persistent=False)
            self.register_buffer("rope_sin", None, persistent=False)

    @torch.no_grad()
    def reset_parameters(self, init: InitName = "xavier_uniform") -> None:
        init_tensor_(self.WQ, init)
        init_tensor_(self.WK, init)
        init_tensor_(self.WV, init)

    @torch.no_grad()
    def set_weights(
        self,
        WQ: Optional[torch.Tensor] = None,
        WK: Optional[torch.Tensor] = None,
        WV: Optional[torch.Tensor] = None,
        freeze: bool = False,
    ) -> None:
        if WQ is not None:
            assert WQ.shape == self.WQ.shape
            self.WQ.copy_(WQ)
        if WK is not None:
            assert WK.shape == self.WK.shape
            self.WK.copy_(WK)
        if WV is not None:
            assert WV.shape == self.WV.shape
            self.WV.copy_(WV)
        if freeze:
            self.WQ.requires_grad_(False)
            self.WK.requires_grad_(False)
            self.WV.requires_grad_(False)

    def _add_absolute_pe(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Add absolute PE for 'learned' or 'sinusoidal' modes."""
        L, _ = x.shape
        if self.pos_type == "none" or self.pos_type == "rope":
            return x  # no absolute PE in these modes

        pos_ids = torch.arange(L, device=x.device)
        if self.pos_type == "learned":
            pos_vecs = self.pos_embed(pos_ids)  # [L,E]
        else:
            # self.sin_pe is guaranteed a Tensor in 'sinusoidal' mode
            pos_vecs = self.sin_pe[:L].to(x.device, x.dtype)  # [L,E]

        pos_vecs = pos_vecs * mask.unsqueeze(-1).to(x.dtype)
        return x + pos_vecs

    def _maybe_rope_qk(self, Q: torch.Tensor, K: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE to Q/K if enabled."""
        if self.pos_type != "rope":
            return Q, K
        L = Q.size(0)
        cos = self.rope_cos[:L].to(Q.device, Q.dtype)
        sin = self.rope_sin[:L].to(Q.device, Q.dtype)
        return _apply_rope(Q, cos, sin), _apply_rope(K, cos, sin)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        assert x.dim() == 2, "x must be [L, E]"
        L, E = x.shape
        assert E == self.d_model

        if mask is None:
            mask = torch.ones(L, dtype=torch.bool, device=x.device)
        else:
            mask = mask.to(dtype=torch.bool, device=x.device)

        # prepend CLS if requested
        if self.use_cls:
            x = x.unsqueeze(0)          # [1, L, E]
            mask = mask.unsqueeze(0)    # [1, L]
            x = torch.cat([self.cls.expand(1, 1, -1), x], dim=1)  # [1, L+1, E]
            cls_mask = torch.ones(1, 1, dtype=torch.bool, device=x.device)
            mask = torch.cat([cls_mask, mask], dim=1)             # [1, L+1]
            x = x.squeeze(0)             # [L', E]
            mask = mask.squeeze(0)       # [L']

        # absolute positional encoding (if any)
        x = self._add_absolute_pe(x, mask)  # [L', E]
        Lp = x.size(0)

        # projections
        Q = x @ self.WQ          # [L', d_k]
        K = x @ self.WK          # [L', d_k]
        V = x @ self.WV          # [L', d_v]

        # RoPE on Q/K (if enabled)
        Q, K = self._maybe_rope_qk(Q, K)

        # attention
        scores = (Q @ K.T) / math.sqrt(self.d_k)  # [L', L']
        if (~mask).any():
            scores[:, ~mask] = float("-inf")      # mask invalid keys

        weights = torch.softmax(scores, dim=-1)   # [L', L']
        out = weights @ V                         # [L', d_v]
        return scores, weights, out
