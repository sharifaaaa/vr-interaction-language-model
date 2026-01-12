#merge_sessions.py

import os, glob, json, sqlite3
import pandas as pd
from parsing_script import parse_json_to_dataframe
from convert_timeStamp import convert_timestamps_safely,convert_timestamps_resilient,fill_missing_timestamps_inplace
from label_density import label_session_density
from handTracking_flags import compute_hand_signal_quality1,impute_hand_features1
from db_sessions import update_hand_quality

DB_PATH = "sessions.db"

def _load_splits_from_db() -> dict:
    if not os.path.exists(DB_PATH):
        return {}
    con = sqlite3.connect(DB_PATH)
    try:
        cur = con.execute("SELECT session_id, split FROM sessions")
        return {sid: sp for sid, sp in cur.fetchall()}
    finally:
        con.close()

def merge_jsons_to_featurevector(
    data_dir: str | None = None,
    file_list: list[str] | None = None,
    out_csv: str = "featureVector_original.csv",
) -> None:
    if file_list is None:
        assert data_dir is not None, "Provide either data_dir or file_list."
        paths = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    else:
        paths = list(file_list)

    if not paths:
        raise RuntimeError("No JSON files to merge.")

    splits = _load_splits_from_db()

    frames = []
    #
    for p in paths:
        base = os.path.basename(p)
        session_stem = os.path.splitext(base)[0]

        df = parse_json_to_dataframe(p)


        #-- Compute & add the quality flag of hand-tracking
        has_valid_hands = compute_hand_signal_quality1(df)

        # Add to every row of this session
        df["has_valid_hands"] = has_valid_hands
        update_hand_quality(session_stem, int(has_valid_hands))
        #---------------------------

        # add these BEFORE any timestamp conversion
        df["source_file"] = base
        df["db_session_id"] = session_stem
        df["split"] = splits.get(session_stem, "train")

        # now do the resilient timestamp steps (group by source_file)
        df = convert_timestamps_resilient(df, "timestamp", groupby="source_file")
        fill_missing_timestamps_inplace(df, "timestamp", groupby="source_file")
        print("NaT after fill:",df["timestamp"].isna().sum())

        # keep original session_id but make it unique per-file
        df["session_id"] = df["source_file"].astype(str) + "::" + df["session_id"].astype(str)
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True, sort=False)

    #New
    dedup_key = ["db_session_id", "timestamp", "phase", "area", "speaker", "text"]
    before = len(merged)
    merged = merged.sort_values(dedup_key, kind="stable").drop_duplicates(subset=dedup_key, keep="first")
    print(f"🧹 key-dedup removed {before - len(merged):,} rows using {dedup_key}.")
    #--

    # --- NOW do global imputation ---(0 = original, 1 = imputed)
    merged = impute_hand_features1(merged)

    merged = label_session_density(merged, text_column="text", session_column="session_id",sparse_max=19)
    merged.to_csv(out_csv, index=False)
    print(f"📦 Merged {len(merged)} total records into {out_csv}")
