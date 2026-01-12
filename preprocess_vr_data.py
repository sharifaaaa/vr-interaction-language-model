# preprocess_vr_data.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from config_features import (
    BEHAVIORAL_COLS,
    CATEGORICAL_COLS,
    GROUP_KEYS,
    assert_required_columns,
    cast_categoricals,
)

def preprocess_and_tag_groups(csv_path, output_path=None):
    df = pd.read_csv(csv_path, low_memory=False)

    # --- basic checks ---
    assert_required_columns(df)

    # --- sort & timestamp ---
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["session_id", "timestamp"]).reset_index(drop=True)

    # --- normalize categoricals (the same ones you later embed) ---
    cast_categoricals(df)

    # --- scale behavioral features (IMPORTANT: fit on TRAIN ONLY in real splits) ---
    #scaler = StandardScaler()
    #df[BEHAVIORAL_COLS] = scaler.fit_transform(df[BEHAVIORAL_COLS])  --> cause leakage

    # --- text column robustification ---
    # keep raw text for DistilBERT; empty rows become ""
    df["text"] = df["text"].fillna("").astype(str)


    # --- contiguous-run grouping per session ---
    shifted = df.groupby("session_id")[GROUP_KEYS].shift()
    breaks = (df[GROUP_KEYS] != shifted).any(axis=1)
    run_id = breaks.groupby(df["session_id"]).cumsum().astype(int)
    run_str = "run:" + run_id.astype(str)

    df["group_id"] = (
        df["session_id"].astype(str) + "|" +
        df["phase"].astype(str) + "|" +
        df["area"].astype(str) + "|" +
        df["user_emotion"].astype(str) + "|" +
        run_str
    )

    if output_path:
        df.to_csv(output_path, index=False)

    return df


def chunk_grouped_data(df, chunk_size=50, stride=25):
    """Chunk within each contiguous run. Optional overlap with `stride`."""
    print(f"Chunk within each contiguous run.The real chunk_size is: {chunk_size} The value of stride is: {stride}")
    if stride is None:
        stride = chunk_size  # no overlap
    chunks = []
    for gid, g in df.groupby('group_id', sort=False):
        g = g.sort_values(['timestamp'])  # already sorted, but safe
        n = len(g)
        starts = range(0, n, stride)
        for i, s in enumerate(starts, start=1):
            e = min(s + chunk_size, n)
            if e - s <= 0:
                continue
            chunk = g.iloc[s:e].copy()
            chunk['chunk_id'] = f"{gid}::chunk{i}"
            chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)


#This ensures chunks never stitch non-contiguous stretches together and lets you optionally use
# overlapping windows (stride < chunk_size) to augment data.