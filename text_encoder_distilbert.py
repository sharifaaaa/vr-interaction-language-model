from dataclasses import dataclass
from typing import List, Optional, Sequence
import time, math
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

import torch
import torch.nn as nn
"""
In this design,BERT runs inside TokenEmbedder.forward(...) and returns tensors that go straight 
into the model. Our CSVs still only carry the raw text column; no BERT vectors are persisted.
So,featureVector_original.csv / tagged_vr_data.csv / chunked_vr_data.csv: contain raw text only.

When we call the Tokenembedder:

It encodes the batch texts → pooled DistilBERT [L, 768].

Projects them (e.g., to 16 dims if GROUP_OUT_DIMS["text"]=16).

Concats that slice with other groups and returns x to the Transformer.

Nothing gets written back to disk.
"""
# optional: pretty progress bars
try:
    from tqdm import tqdm
except Exception:
    tqdm = None  # graceful fallback

# huggingface
try:
    from transformers import DistilBertTokenizerFast, DistilBertModel
except Exception as e:
    DistilBertTokenizerFast = None
    DistilBertModel = None
    _TRANSFORMERS_IMPORT_ERROR = e


@dataclass
class TextEncoderConfig:
    model_name: str = "distilbert-base-uncased"
    max_len: int = 64
    pooling: str = "cls"        # "cls" or "mean"
    freeze: bool = True         # freeze backbone params by default
    batch_size: int = 128       # encode texts in mini-batches
    empty_ratio_cutoff: float = 1.0  # if this fraction (or more) are empty -> all zeros fast path
    # progress bar settings
    show_progress: bool = True
    progress_desc: str = "DistilBERT (text)"
"""
DistilBERT’s 768-d pooled vector is projected down inside the embedder to 
fit proj_dim (and, if you use grouped embeddings, to GROUP_OUT_DIMS["text"]).
"""

class DistilBERTTextEncoder(nn.Module):
    """
    Turns a list of strings into [L, H] pooled embeddings.
    - Skips model work on fully-empty chunks.
    - Encodes only non-empty rows, in mini-batches, with a progress bar.
    - Prints a short 'completed' message with basic stats.
    """
    def __init__(self, cfg: TextEncoderConfig):
        super().__init__()
        if DistilBertTokenizerFast is None or DistilBertModel is None:
            raise ImportError(
                "transformers is required for DistilBERT but couldn't be imported. "
                f"Original error: {_TRANSFORMERS_IMPORT_ERROR}"
            )

        self.cfg = cfg
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(cfg.model_name)
        self.model = DistilBertModel.from_pretrained(cfg.model_name)
        self.model.eval()  # inference by default

        if cfg.freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.hidden_size)

    @torch.no_grad()
    def _pool_mean(self, last_hidden: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        attn = attn_mask.unsqueeze(-1).float()        # [B, T, 1]
        summed = (last_hidden * attn).sum(dim=1)      # [B, H]
        denom = attn.sum(dim=1).clamp_min(1.0)        # [B, 1]
        return summed / denom

    @torch.no_grad()
    # text_encoder_distilbert.py  (inside class DistilBERTTextEncoder)

    @torch.no_grad()
    def encode(self, texts: Sequence[str], device: Optional[torch.device] = None) -> torch.Tensor:
        start_time = time.monotonic()

        if device is None:
            device = next(self.model.parameters()).device
        self.model.to(device)

        L = len(texts)
        out = torch.zeros((L, self.hidden_size), device=device)

        # which rows actually have text?
        nonempty_idx = [i for i, t in enumerate(texts) if isinstance(t, str) and t.strip() != ""]
        n_nonempty = len(nonempty_idx)
        empty_ratio = 1.0 - (n_nonempty / max(L, 1))

        # fast path: all empty → zeros
        if L == 0 or n_nonempty == 0 or empty_ratio >= self.cfg.empty_ratio_cutoff:
            # print once to avoid spam
            if getattr(self, "_warned_empty_once", False) is not True and self.cfg.show_progress:
                print(f"🗣️ {self.cfg.progress_desc}: all texts empty (L={L}). Returning zeros.")
                print(f"✅ {self.cfg.progress_desc}: completed in {time.monotonic() - start_time:.2f}s")
                self._warned_empty_once = True
            return out

        # mini-batch loop + optional tqdm
        bs = max(1, int(self.cfg.batch_size))
        num_batches = math.ceil(n_nonempty / bs)
        use_tqdm = bool(self.cfg.show_progress and (tqdm is not None))

        iterator = tqdm(range(num_batches), total=num_batches,
                        desc=self.cfg.progress_desc, leave=False) if use_tqdm else range(num_batches)

        for bi in iterator:
            start = bi * bs
            idxs = nonempty_idx[start:start + bs]
            batch_texts = [texts[i] for i in idxs]

            enc = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.cfg.max_len,
                return_tensors="pt"
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            last_hidden = self.model(**enc).last_hidden_state  # [B, T, H]
            if self.cfg.pooling == "mean":
                attn = enc["attention_mask"].unsqueeze(-1).float()
                pooled = (last_hidden * attn).sum(dim=1) / attn.sum(dim=1).clamp_min(1.0)
            else:
                pooled = last_hidden[:, 0, :]  # first-token as CLS proxy

            out[idxs] = pooled

        if self.cfg.show_progress:
            elapsed = time.monotonic() - start_time
            print(f"✅ {self.cfg.progress_desc}: encoded {n_nonempty}/{L} rows in {elapsed:.2f}s (batch={bs}).")

        return out
"""

TF-IDF vs distilBERT

* TF-IDF creates a vector whose **length equals the vocabulary size** used at build time.
* If a new dataset has a different vocab, the TF-IDF vector length changes → your linear layer no longer matches.

How DistilBERT fixes mismatch

Fixed output size. DistilBERT always produces hidden states of size 768 per token. 
After pooling (`"cls"` or `"mean"`), each utterance becomes a **single 768-d vector**, regardless of its words or corpus.
Your pipeline keeps it fixed**: we optionally **project 768 → d\_model / GROUP\_OUT\_DIMS\["text"]
with a linear layer. That projection size is a constant in config, so downstream layers always see the same 
dimension.
*Subword tokenizer (WordPiece) avoids OOV issues and vocab explosion; any new word is decomposed into 
known subwords, but the embedding width stays 768.
*Length handling: texts are padded/truncated to `text_max_len` only for batching; after pooling, you still get one fixed-width vector.
*Edge case “silent” rows**: our encoder returns a **zero vector of the same width, 
so shapes remain consistent even when text is missing.


"""