# utils_labels.py
from __future__ import annotations
import torch
import pandas as pd

def labeled_chunk_ids(df: pd.DataFrame, label_col: str = "user_emotion") -> list[str]:
    """
    Return chunk_ids that have a non-NaN label.
    We take the first label per chunk_id (your data has one per chunk).
    """
    first = df.groupby("chunk_id", as_index=False)[label_col].first()
    keep = first[first[label_col].notna()]
    return keep["chunk_id"].tolist()

def build_label_mapping(df: pd.DataFrame, label_col: str = "user_emotion") -> dict[str, int]:
    """Map unique, non-NaN labels to indices."""
    unique_labels = sorted(df[label_col].dropna().unique().tolist())
    return {label: i for i, label in enumerate(unique_labels)}

def labels_tensor(df: pd.DataFrame, chunk_ids: list[str], class_to_idx: dict[str, int],
                  label_col: str = "user_emotion") -> torch.Tensor:
    """
    Build label tensor for the given chunk_ids order.
    Assumes chunk_ids were filtered with labeled_chunk_ids (so no NaNs).
    """
    labels = []
    for cid in chunk_ids:
        lab = df.loc[df["chunk_id"] == cid, label_col].iloc[0]
        labels.append(class_to_idx[lab])
    return torch.tensor(labels, dtype=torch.long)
