# convert_timeStamp.py
import pandas as pd

def convert_timestamps_resilient(df, timestamp_column="timestamp", groupby="source_file"):
    # First pass: ISO/normal strings → UTC-aware
    ts = pd.to_datetime(df[timestamp_column], errors="coerce", utc=True)

    # Second pass: try clock-only formats like "HH:MM:SS[.fff]"
    mask = ts.isna() & df[timestamp_column].notna()
    if mask.any():
        # try with milliseconds
        ts2 = pd.to_datetime(df.loc[mask, timestamp_column], format="%H:%M:%S.%f", errors="coerce")
        # try without milliseconds
        still = ts2.isna()
        if still.any():
            ts3 = pd.to_datetime(df.loc[mask].loc[still, timestamp_column], format="%H:%M:%S", errors="coerce")
            ts2.loc[still] = ts3

        # If we parsed only a time (date=1900-01-01), attach an anchor date per file
        if groupby in df.columns:
            # build per-group anchor dates from any valid ISO rows
            anchor = (
                pd.Series(ts, index=df.index)
                .groupby(df[groupby])
                .transform(lambda s: s.dropna().min())
                .dt.date
            )
            # combine time-only values with anchor date (if available)
            good_time = ts2.notna()
            if good_time.any():
                # make a datetime by combining anchor (date) + time-of-day
                t_only = ts2[good_time]
                # if anchor unavailable, keep NaT
                a = anchor[good_time]
                combined = pd.to_datetime(
                    a.astype("string") + " " + t_only.dt.strftime("%H:%M:%S.%f").str.rstrip("0").str.rstrip("."),
                    errors="coerce",
                    utc=True
                )
                ts.loc[good_time.index] = ts.loc[good_time.index].where(~good_time, combined)

    # Do NOT drop rows; just store tz-naive
    df = df.copy()
    df[timestamp_column] = ts.dt.tz_convert(None)
    # Small report (without dropping)
    n_na = df[timestamp_column].isna().sum()
    if n_na:
        print(f"⚠️ {n_na} rows still have NaT timestamps (kept).")
    print(f"✅ Timestamp conversion done. Rows kept: {len(df)}")
    return df




def fill_missing_timestamps_inplace(df, ts_col="timestamp", groupby=None):
    """
    Fill NaT timestamps in-place.
    - Works with or without a groupby column.
    - No groupby.apply() to avoid FutureWarning.
    """
    if "_rowid" not in df.columns:
        df["_rowid"] = range(len(df))

    if groupby and groupby in df.columns:
        # sort within groups for stable ffill/bfill
        df.sort_values([groupby, ts_col, "_rowid"], inplace=True)

        # process each group explicitly (no FutureWarning)
        for _, gidx in df.groupby(groupby, sort=False).groups.items():
            s = df.loc[gidx, ts_col]
            # forward/back fill
            s = s.ffill().bfill()

            # if entire group is NaT, synthesize a simple monotonic sequence
            if s.isna().all():
                start = pd.Timestamp.utcnow().tz_localize(None)
                s = pd.date_range(start, periods=len(gidx), freq="S")

            # if any NaT remain (rare), fill with "now"
            if s.isna().any():
                s = s.fillna(pd.Timestamp.utcnow().tz_localize(None))

            df.loc[gidx, ts_col] = s.values
    else:
        # no grouping column—do a global pass
        df.sort_values([ts_col, "_rowid"], inplace=True)
        s = df[ts_col].ffill().bfill()
        if s.isna().all():
            start = pd.Timestamp.utcnow().tz_localize(None)
            s = pd.date_range(start, periods=len(df), freq="S")
        if s.isna().any():
            s = s.fillna(pd.Timestamp.utcnow().tz_localize(None))
        df[ts_col] = s.values

def convert_timestamps_safely(df, timestamp_column="timestamp"):
    # Step 1: Convert all timestamps to tz-aware
    parsed_ts = pd.to_datetime(df[timestamp_column], errors="coerce", utc=True)

    # Step 2: Drop unparseable entries
    num_na = parsed_ts.isna().sum()
    if num_na > 0:
        print(f"⚠️ Warning: {num_na} timestamps could not be parsed and will be dropped.")

    # Step 3: Keep only valid rows and replace the column with tz-naive datetime
    df = df.loc[parsed_ts.notna()].copy()
    df[timestamp_column] = parsed_ts.dropna().dt.tz_convert(None)

    print(f"✅ Timestamp conversion done. Final rows: {len(df)}")
    return df
