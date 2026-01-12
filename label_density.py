import pandas as pd

def label_session_density(
    df: pd.DataFrame,
    text_column: str = "text",
    session_column: str = "session_id",   # ← matches db_sessions.py
    sparse_max: int = 19,                 # sparse = 1..29 (default)
    silent_tokens: tuple = ("", "none"),  # treated as no speech after normalization
) -> pd.DataFrame:
    """
    Add a 'session_density' column per session based on how many rows
    contain valid speech text.

    Categories:
      - 'silent' : exactly 0 valid speech rows
      - 'sparse' : 1..sparse_max valid speech rows (default 1..19)
      - 'normal' : >= sparse_max + 1

    Returns the input DataFrame with the new 'session_density' column and
    prints a summary of unique sessions per density.
    """
    if session_column not in df.columns:
        raise KeyError(f"'{session_column}' not found in dataframe columns.")

    # Normalize text and mark valid-speech rows
    txt = (
        df[text_column]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )
    is_valid = ~txt.isin(silent_tokens)

    # Count valid speech per session
    speech_counts = is_valid.groupby(df[session_column]).sum().astype(int)

    # Categorize sessions
    def categorize(n: int) -> str:
        if n == 0:
            return "silent"
        elif 1 <= n <= sparse_max:
            return "sparse"
        else:
            return "normal"

    density_map = speech_counts.apply(categorize)
    df["session_density"] = df[session_column].map(density_map).fillna("silent")

    # Summary in terms of UNIQUE sessions (not rows)
    summary = (
        df[[session_column, "session_density"]]
        .drop_duplicates([session_column])
        .groupby("session_density")[session_column]
        .nunique()
        .sort_index()
    )
    print("✅ Session Speech Density Summary (unique sessions)-They are just informative data and does not affect train/val:")
    print(summary)

    return df
