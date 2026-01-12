# token_embedder.py  (full file replacement is okay; only the diffs shown)

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import json
import torch
import torch.nn as nn
import pandas as pd

from config_attention import ATTN_CONFIG
from config_features import (
    USE_GROUPED_EMBEDDING,
    BEHAVIORAL_GROUPS,
    GROUP_OUT_DIMS,
)

from config_features import (TEXT_MODEL_NAME, TEXT_MAX_LEN,TEXT_POOLING,TEXT_FREEZE,
TEXT_BATCH_SIZE,TEXT_EMPTY_RATIO_CUTOFF,TEXT_SHOW_PROGRESS,TEXT_PROGRESS_DESC,SPEECH_COL)


from text_encoder_distilbert import TextEncoderConfig, DistilBERTTextEncoder


import json

def load_vocabs(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@dataclass
class TokenEmbedderConfig:
    behavioral_cols: List[str]
    categorical_cols: List[str]
    meta_cols: List[str]
    cat_embed_dim: int
    proj_dim: int
    use_text : bool
    speech_col : str


def build_categorical_vocabs(df: pd.DataFrame, categorical_cols: List[str]) -> Dict[str, Dict[str, int]]:
    """
    Build a mapping {col: {category_string: index}}.
    Index 0 is reserved for UNK if use_unk_token=True in the config.
    """
    vocabs: Dict[str, Dict[str, int]] = {}
    for col in categorical_cols:
        ser = df[col].astype("string").fillna("NA")
        uniq = pd.Index(ser.unique()).tolist()
        vocabs[col] = {cat: i + 1 for i, cat in enumerate(sorted(map(str, uniq)))}  # 0 reserved for UNK
    return vocabs


def save_vocabs(vocabs: Dict[str, Dict[str, int]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocabs, f, ensure_ascii=False, indent=2)


def load_vocabs(path: str) -> Dict[str, Dict[str, int]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class TokenEmbedder(nn.Module):
    def __init__(self, cfg: TokenEmbedderConfig, vocabs: Dict[str, Dict[str, int]]):
        super().__init__()
        self.cfg = cfg
        self.vocabs = vocabs

        # categorical embeddings (unchanged)
        self.cat_embeddings = nn.ModuleDict()
        for col in cfg.categorical_cols:
            vocab_size = (max(vocabs[col].values()) if vocabs[col] else 0) + 1  # +1 for UNK
            self.cat_embeddings[col] = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=cfg.cat_embed_dim,
                padding_idx=0
            )

        # --- TEXT ENCODER (via separate module) ---
        self.text_encoder: Optional[DistilBERTTextEncoder] = None
        self.text_hidden_size: int = 0
        if self.cfg.use_text:

            te_cfg = TextEncoderConfig(
                model_name=TEXT_MODEL_NAME,
                max_len=TEXT_MAX_LEN,
                pooling=TEXT_POOLING,
                freeze=TEXT_FREEZE,
                empty_ratio_cutoff=TEXT_EMPTY_RATIO_CUTOFF,
                show_progress=TEXT_SHOW_PROGRESS,  # ← quiet mode
                progress_desc=TEXT_PROGRESS_DESC
            )



            self.text_encoder = DistilBERTTextEncoder(te_cfg)
            self.text_hidden_size = self.text_encoder.hidden_size

        # legacy projector bookkeeping
        meta_dim = 0 if not cfg.meta_cols else len(cfg.meta_cols)
        self._static_in_dim = (
            len(cfg.behavioral_cols) + (len(cfg.categorical_cols) * cfg.cat_embed_dim) + meta_dim
        )
        self.proj: Optional[nn.Linear] = None
        self._in_dim_ready = False

        # device tracker
        self.register_buffer("_dummy", torch.tensor(0))

        # grouped-vs-legacy mode
        self.use_grouped: bool = bool(USE_GROUPED_EMBEDDING)
        self.group_projectors = nn.ModuleDict()
        self._text_proj: Optional[nn.Linear] = None
        self.feature_groups: Optional[Dict[str, Tuple[int, int]]] = None

        if self.use_grouped:
            # behavioral groups
            for gname, cols in BEHAVIORAL_GROUPS.items():
                out_dim = int(GROUP_OUT_DIMS.get(gname, 0))
                if out_dim > 0:
                    self.group_projectors[gname] = nn.Linear(len(cols), out_dim)

            # categoricals
            cat_out = int(GROUP_OUT_DIMS.get("categoricals", 0))
            if cat_out > 0:
                self.group_projectors["categoricals"] = nn.Linear(
                    len(cfg.categorical_cols) * cfg.cat_embed_dim, cat_out
                )

            # meta
            meta_out = int(GROUP_OUT_DIMS.get("meta", 0))
            if meta_out > 0 and meta_dim > 0:
                self.group_projectors["meta"] = nn.Linear(meta_dim, meta_out)

            # sum check
            total_out = 0
            for gname in BEHAVIORAL_GROUPS:
                total_out += int(GROUP_OUT_DIMS.get(gname, 0))
            total_out += int(GROUP_OUT_DIMS.get("categoricals", 0))
            total_out += int(GROUP_OUT_DIMS.get("text", 0))   # now TEXT not TF-IDF
            total_out += int(GROUP_OUT_DIMS.get("meta", 0))
            #It will only stop if the dimensions do NOT match
            assert total_out == self.cfg.proj_dim, (
                f"[TokenEmbedder] GROUP_OUT_DIMS total ({total_out}) must equal proj_dim ({self.cfg.proj_dim})."
            )

    def _map_categorical(self, ser: pd.Series, vocab: Dict[str, int]) -> torch.Tensor:
        s = ser.astype("string").fillna("NA").astype(str)
        ids = s.map(vocab).fillna(0).astype(int).to_numpy()
        return torch.as_tensor(ids, dtype=torch.long)

    def _maybe_init_legacy_proj(self, text_dim: int):
        if not self._in_dim_ready:
            in_dim = self._static_in_dim + text_dim
            self.proj = nn.Linear(in_dim, self.cfg.proj_dim)
            self._in_dim_ready = True

    def forward(
        self,
        chunk_df: pd.DataFrame,
        pad_to: Optional[int] = None,
        device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = device or self._dummy.device
        L_df = len(chunk_df)

        # behavioral numeric block
        behav_all = torch.tensor(
            chunk_df[self.cfg.behavioral_cols].to_numpy(),
            dtype=torch.float32, device=device
        )

        # categoricals block
        cat_vecs = []
        for col in self.cfg.categorical_cols:
            ids = self._map_categorical(chunk_df[col], self.vocabs[col]).to(device)
            cat_vecs.append(self.cat_embeddings[col](ids))
        cats_full = torch.cat(cat_vecs, dim=1) if cat_vecs else torch.empty((L_df, 0), device=device)

        # text (through the external encoder)

        if self.text_encoder is not None:
            texts = chunk_df[self.cfg.speech_col].fillna("").astype(str).tolist()
            # runs tokenization + DistilBERT forward + pooling → returns a tensor of shape [L, H]
            # (H = self.text_hidden_size). [L, H] — zeros for empties
            text_ctx = self.text_encoder.encode(texts, device=device)    # <-- EMBEDDING HERE
        else:
            text_ctx = torch.empty((L_df, 0), device=device)

        # meta
        # meta
        if self.cfg.meta_cols:
            meta_full = torch.tensor(
                chunk_df[self.cfg.meta_cols].to_numpy(),
                dtype=torch.float32, device=device
            )
        else:
            meta_full = torch.empty((L_df, 0), device=device)


        # ---- legacy mode ----
        if not self.use_grouped:
            x_in = torch.cat([behav_all, cats_full, text_ctx, meta_full], dim=1)
            self._maybe_init_legacy_proj(text_ctx.shape[1])
            x = self.proj(x_in)
            feature_groups = None
        # ---- grouped mode ----
        else:
            pieces = []
            feature_groups: Dict[str, Tuple[int, int]] = {}
            cursor = 0

            # behavioral groups
            for gname, cols in BEHAVIORAL_GROUPS.items():
                out_dim = int(GROUP_OUT_DIMS.get(gname, 0))
                if out_dim == 0:
                    continue
                idxs = [self.cfg.behavioral_cols.index(c) for c in cols]
                g_in = behav_all[:, idxs]
                g_proj = self.group_projectors[gname](g_in)
                pieces.append(g_proj)
                feature_groups[gname] = (cursor, cursor + out_dim)
                cursor += out_dim

            # categoricals slice
            cat_out = int(GROUP_OUT_DIMS.get("categoricals", 0))
            if cat_out > 0:
                g_proj = self.group_projectors["categoricals"](cats_full) if cats_full.shape[1] > 0 \
                    else torch.zeros((L_df, cat_out), device=device)
                pieces.append(g_proj)
                feature_groups["categoricals"] = (cursor, cursor + cat_out)
                cursor += cat_out

            # speech text slice (lazy projector since hidden size is known after model load)
            text_out = int(GROUP_OUT_DIMS.get("text", 0))
            if text_out > 0:
                if self.text_encoder is not None:
                    if self._text_proj is None:
                        self._text_proj = nn.Linear(self.text_hidden_size, text_out).to(device)
                    g_proj = self._text_proj(text_ctx) if text_ctx.shape[1] > 0 \
                        else torch.zeros((L_df, text_out), device=device)
                else:
                    g_proj = torch.zeros((L_df, text_out), device=device)
                pieces.append(g_proj)
                feature_groups["text"] = (cursor, cursor + text_out)
                cursor += text_out
            # (debug)
            #s0, s1 = feature_groups["text"]
            #mean_norm = x[:, s0:s1].norm(dim=1).mean().item()
            #if torch.isfinite(torch.tensor(mean_norm)):
            #    print(f"[debug] TEXT slice mean norm = {mean_norm:.4f}")

            # meta slice
            meta_out = int(GROUP_OUT_DIMS.get("meta", 0))
            if meta_out > 0:
                g_proj = self.group_projectors["meta"](meta_full) if meta_full.shape[1] > 0 \
                    else torch.zeros((L_df, meta_out), device=device)
                pieces.append(g_proj)
                feature_groups["meta"] = (cursor, cursor + meta_out)
                cursor += meta_out

            x = torch.cat(pieces, dim=1) if pieces else torch.empty((L_df, 0), device=device)

        # padding & mask (unchanged)
        L = x.shape[0]
        if pad_to is not None and L != pad_to:
            if L > pad_to:
                x = x[:pad_to]
                mask = torch.ones(pad_to, dtype=torch.bool, device=device)
            else:
                pad_rows = torch.zeros(pad_to - L, x.shape[1], dtype=x.dtype, device=device)
                x = torch.vstack([x, pad_rows])
                mask = torch.cat([torch.ones(L, dtype=torch.bool, device=device),
                                  torch.zeros(pad_to - L, dtype=torch.bool, device=device)])
        else:
            mask = torch.ones(L, dtype=torch.bool, device=device)

        self.feature_groups = feature_groups if self.use_grouped else None
        # x: [B, L, D] before return
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        ##End of sanitize
        return x, mask
