# attention_runner.py
from __future__ import annotations
from typing import Tuple, List
import torch

import config_attention as CA

from .attention_utils import (
    build_single_head, build_multi_head, build_encoder,
    run_single_head_chunk, run_multi_head_chunk, run_multi_head_batch,
    run_encoder_chunk, run_encoder_batch,
)


def _safe_detach_to(x, dtype):
    # Detach if it's a tensor; otherwise create a new tensor
    if isinstance(x, torch.Tensor):
        return x.detach().to(dtype)
    return torch.as_tensor(x, dtype=dtype)

def get_chunk_batch(chunked_df, embedder, pad_to: int):
    """
    Convert all chunks to tensors.
    Returns:
      X: (B, L, E)  float32 (DETACHED!)
      M: (B, L)     bool    (DETACHED!)
      ids: list of chunk_ids (len=B)

      Key point: never use torch.as_tensor(tensor) to “convert” tensors you got from a model—use
       .detach() first to break their graph.
       X_list.append(torch.as_tensor(x, dtype=torch.float32))
    """
    X_list, M_list, ids = [], [], []
    for cid, g in chunked_df.groupby("chunk_id", sort=False):
        x, mask = embedder(g, pad_to=pad_to)  # x may carry grad graph!

        x_t = _safe_detach_to(x, torch.float32)      # <-- detach here
        m_t = _safe_detach_to(mask, torch.bool)      # <-- detach here

        X_list.append(x_t)
        M_list.append(m_t)
        ids.append(cid)

    X = torch.stack(X_list, dim=0)  # (B,L,E)
    M = torch.stack(M_list, dim=0)  # (B,L)
    return X, M, ids

def run_attention(choice: str, chunked_df, embedder) -> None:
    """
    Central switch to run attention in various modes:
      'single' | 'multi' | 'batch' | 'enc-single' | 'enc-batch'
    """
    choice = choice.strip().lower()
    pad_to = CA.ATTN_CONFIG["max_len"]

    print(f"\n⚙️  MODE = {choice.upper()}  |  POOLING = {CA.POOLING}  "
          f"|  d_model = {CA.ATTN_CONFIG['d_model']}  |  heads = {CA.ATTN_CONFIG['n_heads']}")

    if choice == "single":
        attn = build_single_head()
        cid, g = next(iter(chunked_df.groupby("chunk_id", sort=False)))
        x, mask = embedder(g, pad_to=pad_to)
        print(f"\n--- SINGLE-HEAD for chunk {cid} ---")
        run_single_head_chunk(attn, x, mask, print_scores=True)

    elif choice == "multi":
        attn = build_multi_head()
        cid, g = next(iter(chunked_df.groupby("chunk_id", sort=False)))
        x, mask = embedder(g, pad_to=pad_to)
        print(f"\n--- MULTI-HEAD for chunk {cid} ---")
        run_multi_head_chunk(attn, x, mask, print_scores=True)

    elif choice == "batch":
        attn = build_multi_head()
        X, M, ids = get_chunk_batch(chunked_df, embedder, pad_to=pad_to)
        print(f"\n--- MULTI-HEAD BATCH for {len(ids)} chunks ---")
        run_multi_head_batch(attn, X, M, ids=ids, print_first=True)



    elif choice == "enc-single":
        enc = build_encoder()
        cid, g = next(iter(chunked_df.groupby("chunk_id", sort=False)))
        x, mask = embedder(g, pad_to=pad_to)
        print(f"\n--- ENCODER (stacked) for chunk {cid} ---")
        run_encoder_chunk(enc, x, mask, print_scores=True)

    elif choice == "enc-batch":
        enc = build_encoder()
        X, M, ids = get_chunk_batch(chunked_df, embedder, pad_to=pad_to)
        print(f"\n--- ENCODER (stacked) BATCH for {len(ids)} chunks ---")
        run_encoder_batch(enc, X, M, ids=ids, print_first=True)

    else:
        raise ValueError("Invalid MODE. Use 'single' | 'multi' | 'batch' | 'enc-single' | 'enc-batch'.")
