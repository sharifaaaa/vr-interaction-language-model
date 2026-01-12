import pandas as pd
import numpy as np
from config_features import HAND_COLS


#Version 0
def compute_hand_signal_quality(df: pd.DataFrame) -> int:
    """
    Decide if a session has usable hand/controller signals.

    Returns:
        1 if valid, 0 if invalid (frozen / garbage / almost constant).
    """
    # make sure columns exist
    for c in HAND_COLS:
        if c not in df.columns:
            df[c] = np.nan

    block = df[HAND_COLS].astype("float32")

    # all NaN -> invalid
    if block.isna().all().all():
        return 0

    # how much is non-zero?
    nonzero_frac = (block.abs() > 1e-6).sum().sum() / float(block.size)

    # detect "frozen" signals (almost no variation / very few unique values)
    std_per_col = block.std(skipna=True)
    low_variance = (std_per_col < 1e-4).all()

    nunique_per_col = block.nunique(dropna=True)
    ultra_low_unique = (nunique_per_col <= 3).all()

    looks_frozen = low_variance or ultra_low_unique

    has_valid = int((nonzero_frac > 0.05) and (not looks_frozen))
    return has_valid


def impute_hand_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute/neutralise hand features using medians from VALID sessions.

    Adds:
        - hands_imputed: 1 if this row's hands were overwritten/imputed
                         0 otherwise.
    """

    # ensure columns exist
    for c in HAND_COLS:
        if c not in df.columns:
            df[c] = np.nan

    if "has_valid_hands" not in df.columns:
        raise ValueError("impute_hand_features expects a 'has_valid_hands' column")

    mask_valid = df["has_valid_hands"] == 1
    mask_bad   = df["has_valid_hands"] == 0

    # medians from valid rows; if all NaN, fall back to 0
    medians = df.loc[mask_valid, HAND_COLS].median()
    medians = medians.fillna(0.0)

    # Overwrite bad rows with these medians (simple, stable baseline)
    df.loc[mask_bad, HAND_COLS] = medians.values

    # mark which rows were imputed
    df["hands_imputed"] = 0
    df.loc[mask_bad, "hands_imputed"] = 1

    return df
#Version 1
def compute_hand_signal_quality1(df_session: pd.DataFrame,
                                eps: float = 1e-6,
                                min_dynamic_ratio: float = 0.02) -> int:
    """
    Determine whether the hand/controller motion in a session is valid.

    Returns:
        1 = valid hand/controller signal
        0 = invalid or static (garbage) signal
    """

    cols = [
        "r_pos_x", "r_pos_y", "r_pos_z",
        "l_pos_x", "l_pos_y", "l_pos_z"
    ]

    # If columns are missing → invalid
    if not set(cols).issubset(df_session.columns):
        return 0

    arr = df_session[cols].values.astype(float)

    # 1. variance check (detect flat signals)
    std = arr.std(axis=0)
    if np.all(std < eps):
        return 0

    # 2. dynamic change (frame-to-frame movement)
    diffs = np.abs(arr[1:] - arr[:-1])
    dynamic_frames = (diffs > eps).any(axis=1)
    dynamic_ratio = dynamic_frames.mean()

    return int(dynamic_ratio >= min_dynamic_ratio)



def impute_hand_features1(df: pd.DataFrame,
                         flag_col: str = "has_valid_hands",
                         imputed_col: str = "hands_imputed") -> pd.DataFrame:
    """
    Impute controller / hand position features for rows where hand data is bad.

    - Only rows with has_valid_hands == 0 are modified.
    - Stats are computed from rows with has_valid_hands == 1.
    - A new column `hands_imputed` marks which rows were changed.
    """
    df = df.copy()

    # create flag column (0 = not imputed by default)
    if imputed_col not in df.columns:
        df[imputed_col] = 0

    # sanity checks
    missing_cols = [c for c in HAND_COLS if c not in df.columns]
    if missing_cols:
        print(f"[impute_hand_features] Missing hand cols, skipping: {missing_cols}")
        return df

    if flag_col not in df.columns:
        print(f"[impute_hand_features] Missing flag_col '{flag_col}', skipping.")
        return df

    mask_good = df[flag_col] == 1
    mask_bad  = df[flag_col] == 0

    if not mask_bad.any():
        # no bad rows => nothing to do
        return df

    df_good = df.loc[mask_good].copy()
    if df_good.empty:
        print("[impute_hand_features] No rows with valid hands; not imputing.")
        return df

    # ---- context-based means from VALID rows ----
    group_keys = ["phase", "area", "user_emotion"]
    have_all_keys = all(k in df_good.columns for k in group_keys)

    if have_all_keys:
        ctx_means = (
            df_good.groupby(group_keys)[HAND_COLS]
                  .mean()
                  .reset_index()
        )
    else:
        ctx_means = None
        print("[impute_hand_features] Some context keys missing; using global means only.")

    global_means = df_good[HAND_COLS].mean()

    # ---- apply imputation to BAD rows ----
    df_bad = df.loc[mask_bad].copy()

    if ctx_means is not None:
        df_bad = df_bad.merge(
            ctx_means,
            on=group_keys,
            how="left",
            suffixes=("", "_ctx_mean"),
        )
        for col in HAND_COLS:
            ctx_col = f"{col}_ctx_mean"
            df.loc[mask_bad, col] = np.where(
                df_bad[ctx_col].notna(),
                df_bad[ctx_col].values,
                float(global_means[col]),
            )
    else:
        for col in HAND_COLS:
            df.loc[mask_bad, col] = float(global_means[col])

    # mark imputed rows
    df.loc[mask_bad, imputed_col] = 1

    return df
