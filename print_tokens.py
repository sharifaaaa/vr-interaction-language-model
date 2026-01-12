# print_tokens.py

import torch
from token_embedder import TokenEmbedderConfig, TokenEmbedder, build_categorical_vocabs, save_vocabs

def print_chunk_tokens(chunked_df, target_len=240, print_first_n_tokens=3):
    """
    Given a chunked VR dataframe, tokenize each chunk using TokenEmbedder
    and print the first few token vectors for inspection.
    """

    # --- Column configuration ---
    behavioral_cols = [
        "gaze_x","gaze_y","gaze_z",
        "head_pos_x","head_pos_y","head_pos_z",
        "r_pos_x","r_pos_y","r_pos_z",
        "l_pos_x","l_pos_y","l_pos_z",
        "movement_speed","r_movement_speed","l_movement_speed",
    ]

    categorical_cols = [
        "phase","area","speaker","gaze_actor",
        "r_interacted_actor","l_interacted_actor","session_density",
    ]

    meta_cols = []  # optional numeric meta features

    # --- Config and vocabs ---
    cfg = TokenEmbedderConfig(
        behavioral_cols=behavioral_cols,
        categorical_cols=categorical_cols,
        meta_cols=meta_cols,
        cat_embed_dim=8,
        proj_dim=128,   # d_model for Transformer
    )

    vocabs = build_categorical_vocabs(chunked_df, categorical_cols)
    save_vocabs(vocabs, "cat_vocabs.json")

    embedder = TokenEmbedder(cfg, vocabs)

    # --- Process chunks ---
    examples = []
    labels = []

    for cid, g in chunked_df.groupby("chunk_id", sort=False):
        x, mask = embedder(g, pad_to=target_len)
        y = g["user_emotion"].iloc[0]

        examples.append((x, mask))
        labels.append(y)

        print(f"\nChunk ID: {cid}")
        print(f"Label: {y}")
        print(f"Shape: tokens={x.shape[0]}, emb_dim={x.shape[1]} (real tokens: {mask.sum().item()}, padded: {(~mask).sum().item()})")

        for t in range(min(print_first_n_tokens, int(mask.sum().item()))):
            print(f"  token[{t}]: {x[t].detach().cpu().numpy()}")

    print(f"\n✅ Built {len(examples)} chunk tensors.")
    return examples, labels
