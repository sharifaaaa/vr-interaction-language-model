# multihead_attention.py
from typing import Optional, Literal, Tuple
import math
import torch
import torch.nn as nn

InitName = Literal["xavier_uniform", "xavier_normal",
                   "kaiming_uniform", "kaiming_normal",
                   "zeros"]
# add "rope"
PosType = Literal["learned", "sinusoidal", "none", "rope"]

def init_(w: torch.Tensor, how: InitName = "xavier_uniform") -> None:
    if how == "xavier_uniform":
        nn.init.xavier_uniform_(w)
    elif how == "xavier_normal":
        nn.init.xavier_normal_(w)
    elif how == "kaiming_uniform":
        nn.init.kaiming_uniform_(w, a=math.sqrt(5))
    elif how == "kaiming_normal":
        nn.init.kaiming_normal_(w, a=math.sqrt(5))
    elif how == "zeros":
        nn.init.zeros_(w)
    else:
        raise ValueError(how)

def sinusoidal_pe(max_len: int, d_model: int, device=None, dtype=None) -> torch.Tensor:
    pe = torch.zeros(max_len, d_model, device=device, dtype=dtype)
    pos = torch.arange(0, max_len, device=device, dtype=dtype).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, device=device, dtype=dtype)
                    * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe

# -------------------- Rotary Embedding (RoPE) --------------------
class RotaryEmbedding(nn.Module):
    """
    Applies Rotary Position Embeddings to Q and K.
    Expects Q,K of shape [B, H, L, Dh]; requires even Dh.
    """
    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE head_dim must be even, got {head_dim}.")
        self.head_dim = head_dim
        self.base = base
        self._seq_cached: Optional[int] = None
        self._cos: Optional[torch.Tensor] = None
        self._sin: Optional[torch.Tensor] = None

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        # even indices, odd indices
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)

    def _build_sin_cos(self, L: int, device, dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        half = self.head_dim // 2
        inv_freq = 1.0 / (self.base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
        t = torch.arange(L, device=device, dtype=torch.float32)
        freqs = torch.einsum("l,d->ld", t, inv_freq)  # [L, half]
        cos = torch.repeat_interleave(freqs.cos(), 2, dim=-1).to(dtype)  # [L, Dh]
        sin = torch.repeat_interleave(freqs.sin(), 2, dim=-1).to(dtype)
        return cos, sin

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # q,k: [B, H, L, Dh]
        L = q.size(2)
        device, dtype = q.device, q.dtype
        if self._seq_cached != L or self._cos is None or self._cos.device != device:
            self._cos, self._sin = self._build_sin_cos(L, device, dtype)
            self._seq_cached = L
        cos = self._cos[None, None, :, :]  # [1,1,L,Dh]
        sin = self._sin[None, None, :, :]
        q_rot = (q * cos) + (self._rotate_half(q) * sin)
        k_rot = (k * cos) + (self._rotate_half(k) * sin)
        return q_rot, k_rot
# -----------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """
    Multi-head scaled dot-product attention with optional positional embeddings.
    Supports inputs of shape:
      - (L, E)             -> treated as batch size 1
      - (B, L, E)

    Args
    ----
    d_model : token embedding size (E)
    n_heads : number of heads (h)
    d_k     : key/query size per head
    d_v     : value size per head (defaults to d_k)
    pos_type: "learned" | "sinusoidal" | "none" | "rope"
    max_len : maximum sequence length for positions
    init    : initialization for projection weights
    seed    : manual seed for reproducibility

    Projections are stored as nn.Parameter to allow manual setting.
    Shapes:
      WQ, WK : (d_model, n_heads * d_k)
      WV     : (d_model, n_heads * d_v)
      WO     : (n_heads * d_v, d_model)
    """

    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 d_k: int,
                 d_v: Optional[int] = None,
                 *,
                 pos_type: str = "learned",
                 max_len: int = 240,
                 init: str = "xavier_uniform",
                 seed: Optional[int] = None,
                 use_cls: bool = False,
                 rope_base: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        self.h = n_heads
        self.d_k = d_k
        self.d_v = d_v or d_k
        self.pos_type = pos_type.lower()
        self.use_cls = bool(use_cls)

        # projections
        self.WQ = nn.Parameter(torch.empty(d_model, self.h * self.d_k))
        self.WK = nn.Parameter(torch.empty(d_model, self.h * self.d_k))
        self.WV = nn.Parameter(torch.empty(d_model, self.h * self.d_v))
        self.WO = nn.Parameter(torch.empty(self.h * self.d_v, d_model))
        self.reset_parameters(init)

        # ---- CLS token (optional) ----
        if self.use_cls:
            self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.cls, mean=0.0, std=0.02)

        # ---- positional embeddings: allocate room for CLS ----
        # small headroom helps if pipeline occasionally delivers L == max_len + 1,
        # then +CLS makes it +2
        pe_len = max_len + (2 if self.use_cls else 1)
        self.max_len = max_len
        self.pe_len = pe_len

        # Absolute PE tables (used only for learned/sinusoidal)
        if self.pos_type == "learned":
            self.pos_embed = nn.Embedding(pe_len, d_model)
            nn.init.normal_(self.pos_embed.weight, mean=0.0, std=0.02)
        elif self.pos_type == "sinusoidal":
            self.register_buffer("sin_pe", sinusoidal_pe(pe_len, d_model), persistent=False)
        elif self.pos_type == "rope":
            # no absolute table; rotate Q,K instead
            if self.d_k % 2 != 0:
                raise ValueError(f"RoPE requires even d_k; got d_k={self.d_k}.")
            self.rope = RotaryEmbedding(self.d_k, base=rope_base)
        elif self.pos_type == "none":
            pass
        else:
            raise ValueError("pos_type must be 'learned' | 'sinusoidal' | 'none' | 'rope'")

    # ----- init / manual set --------------------------------------------------

    @torch.no_grad()
    def reset_parameters(self, how: InitName = "xavier_uniform") -> None:
        for p in (self.WQ, self.WK, self.WV, self.WO):
            init_(p, how)

    @torch.no_grad()
    def set_weights(self,
                    WQ: Optional[torch.Tensor] = None,
                    WK: Optional[torch.Tensor] = None,
                    WV: Optional[torch.Tensor] = None,
                    WO: Optional[torch.Tensor] = None,
                    freeze: bool = False) -> None:
        if WQ is not None: self._copy_check(self.WQ, WQ)
        if WK is not None: self._copy_check(self.WK, WK)
        if WV is not None: self._copy_check(self.WV, WV)
        if WO is not None: self._copy_check(self.WO, WO)
        if freeze:
            for p in (self.WQ, self.WK, self.WV, self.WO):
                p.requires_grad_(False)

    @staticmethod
    def _copy_check(dst: torch.Tensor, src: torch.Tensor) -> None:
        assert dst.shape == src.shape, f"shape mismatch: {dst.shape} vs {src.shape}"
        dst.copy_(src)

    # ----- forward ------------------------------------------------------------

    def _add_positional_abs(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Add absolute (learned/sinusoidal) positions to x with mask alignment."""
        B, L, E = x.shape
        pos_ids = torch.arange(L, device=x.device)
        if self.pos_type == "learned":
            pos = self.pos_embed(pos_ids)  # (L,E)
        else:
            pos = self.sin_pe[:L].to(x.device, x.dtype)  # (L,E)
        pos = pos.unsqueeze(0) * mask.unsqueeze(-1).to(x.dtype)
        return x + pos

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        x:    (L, E) or (B, L, E) with E == d_model
        mask: (L,) or (B, L)  where True=real, False=pad
        returns:
            scores:  (B, h, L', L')   # L' == L (+1 if use_cls)
            weights: (B, h, L', L')
            out:     (B, L', d_model)
        """
        device = x.device
        dtype = x.dtype

        # -------- Normalize input to (B,L,E) and mask to (B,L) ----------
        if x.dim() == 2:
            x = x.unsqueeze(0)  # (1,L,E)
            B, L, E = x.shape
            if mask is None:
                mask = torch.ones(L, dtype=torch.bool, device=device)
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)  # (1,L)
        elif x.dim() == 3:
            B, L, E = x.shape
            if mask is None:
                mask = torch.ones(B, L, dtype=torch.bool, device=device)
            elif mask.dim() == 1:
                mask = mask.unsqueeze(0).expand(B, -1)
        else:
            raise ValueError("x must be (L,E) or (B,L,E)")

        if E != self.d_model:
            raise ValueError(f"d_model mismatch: got E={E}, expected {self.d_model}")

        mask = mask.to(torch.bool, copy=False).to(device)

        # ----------------- Prepend learnable CLS (optional) --------------
        if getattr(self, "use_cls", False):
            cls = self.cls.expand(B, 1, -1)  # (B,1,E)
            x = torch.cat([cls, x], dim=1)  # (B,L+1,E)
            cls_mask = torch.ones(B, 1, dtype=torch.bool, device=device)
            mask = torch.cat([cls_mask, mask], dim=1)  # (B,L+1)
            L = L + 1

        # -------------------- Positional encodings (robust) -------------------
        # Absolute add only for learned/sinusoidal; skip if "rope" or "none"
        if self.pos_type in ("learned", "sinusoidal"):
            # 1) derive lengths from the actual tensors
            B, Lx, E = x.shape
            Lm = mask.size(1)

            # 2) ensure mask tracks x length in front-CLS convention
            if Lm != Lx:
                if Lm > Lx:
                    mask = mask[:, :Lx]
                else:  # Lm < Lx  -> pad mask at the FRONT by True (CLS is at the front)
                    pad = torch.ones(B, Lx - Lm, dtype=mask.dtype, device=mask.device)
                    mask = torch.cat([pad, mask], dim=1)

            # 3) cap by available positional table, then slice x & mask
            Lcap = min(Lx, self.pe_len)
            if Lx != Lcap:
                x = x[:, :Lcap, :]
                mask = mask[:, :Lcap]
                Lx = Lcap

            # 4) final add (all three share dim-1 == Lcap)
            x = self._add_positional_abs(x, mask)

        # keep a single source of truth for length afterwards
        L = x.size(1)

        # -------------------- Projections & reshape ----------------------
        Q = x @ self.WQ
        K = x @ self.WK
        V = x @ self.WV

        # -> (B,h,L,d)
        Q = Q.view(B, L, self.h, self.d_k).transpose(1, 2)  # (B,h,L,d_k)
        K = K.view(B, L, self.h, self.d_k).transpose(1, 2)  # (B,h,L,d_k)
        V = V.view(B, L, self.h, self.d_v).transpose(1, 2)  # (B,h,L,d_v)

        # -------------------- RoPE rotation (if enabled) -----------------
        if self.pos_type == "rope":
            Q, K = self.rope(Q, K)

        # -------------------- Scaled dot-product -------------------------
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B,h,L,L)

        # ---------- NEW — force-align mask to L (works for all PE types) ----------
        Lm = mask.size(1)
        if Lm != L:
            if Lm > L:
                # if mask is longer (e.g., has CLS when x doesn't), trim the LEFT
                mask = mask[:, -L:]
            else:
                # if mask is shorter, pad on the LEFT with True (keep CLS-at-front convention)
                pad = torch.ones(B, L - Lm, dtype=mask.dtype, device=mask.device)
                mask = torch.cat([pad, mask], dim=1)
        # ---------------------------------------------------------------------------

        # Key padding mask over the LAST dimension (keys): (B,1,1,L)
        key_pad = (~mask).unsqueeze(1).unsqueeze(1)  # (B,1,1,L)
        scores = scores.masked_fill(key_pad, float("-inf"))

        # Softmax over keys
        weights = torch.softmax(scores, dim=-1)  # (B,h,L,L)

        # Weighted sum
        head_out = weights @ V  # (B,h,L,d_v)
        head_out = head_out.transpose(1, 2).contiguous()  # (B,L,h,d_v)
        head_out = head_out.view(B, L, self.h * self.d_v)  # (B,L,h*d_v)

        # Output projection
        out = head_out @ self.WO  # (B,L,E=d_model)

        return scores, weights, out
