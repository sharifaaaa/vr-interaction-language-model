# utils_splits.py
import csv, re
from pathlib import Path
import pandas as pd
from typing import Dict

_UUID_RE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")

def canon_sid(x: str) -> str:
    """Return a canonical session key (lowercase, prefer UUID if present)."""
    if x is None:
        return ""
    s = str(x).strip().lower()
    m = _UUID_RE.search(s)
    if m:
        return m.group(0)  # just the UUID
    # strip common prefixes like "session_"
    if s.startswith("session_"):
        s = s[len("session_"):]
    return s

def load_stratified_mapping(csv_path: str) -> Dict[str, str]:
    """Returns {canonical_session_key: split} from saved stratified CSV."""
    p = Path(csv_path)
    if not p.is_file():
        raise FileNotFoundError(f"Stratified split CSV not found: {csv_path}")
    mp: Dict[str, str] = {}
    with p.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sid = canon_sid(row.get("session_id"))
            if sid:
                mp[sid] = row.get("split", "")
    return mp

def auto_session_col(df: pd.DataFrame) -> str:
    """Pick the best session-id column present."""
    for c in ("db_session_id", "session_id", "session", "sid"):
        if c in df.columns:
            return c
    raise KeyError("No session-id column found (tried db_session_id, session_id, session, sid).")

def apply_split_override(df: pd.DataFrame,
                         mapping: Dict[str, str],
                         session_col: str | None = None,
                         split_col: str = "split",
                         new_col: str | None = None,
                         strict: bool = False,
                         default_for_missing: str | None = None) -> pd.DataFrame:
    """Override splits using mapping; works with db_session_id/session_id; canonicalizes keys."""
    df = df.copy()
    sc = session_col or auto_session_col(df)
    if sc not in df.columns:
        raise KeyError(f"'{sc}' column not found in DataFrame.")

    # Build canonical series of session keys for the DF
    can = df[sc].astype(str).map(canon_sid)
    # Map using canonical keys
    mapped = can.map(mapping)

    tgt = new_col or split_col
    if strict:
        mask = mapped.notna()
        out = df.loc[mask].copy()
        out[tgt] = mapped[mask]
        return out

    if default_for_missing is None:
        # mapped -> mapping; missing -> keep existing split
        df[tgt] = mapped.fillna(df.get(split_col))
    else:
        # mapped -> mapping; missing -> default
        df[tgt] = mapped.fillna(default_for_missing)
    return df

def assert_split_consistency(df: pd.DataFrame,
                             mapping: Dict[str, str],
                             session_col: str | None = None,
                             split_col: str = "split",
                             sample: int = 10) -> None:
    sc = session_col or auto_session_col(df)
    can = df[sc].astype(str).map(canon_sid)
    mapped = can.map(mapping)
    mism = df[mapped.notna() & (mapped != df[split_col])]
    if not mism.empty:
        print("⚠️  Split mismatches detected. Showing a few rows:")
        print(mism[[sc, split_col]].head(sample))
        raise AssertionError(f"{len(mism)} row(s) disagree with the stratified split mapping.")

def report_split_status(name: str, df: pd.DataFrame, mapping: Dict[str, str]) -> None:
    sc = auto_session_col(df)
    sess = pd.Series(df[sc].astype(str).map(canon_sid).unique())
    covered = sess.map(lambda s: s in mapping).sum()
    vc = df["split"].value_counts(dropna=False).to_dict() if "split" in df.columns else {}
    print(f"🔎 {name}: sessions={len(sess)}, covered_by_mapping={covered} "
          f"({0 if len(sess)==0 else covered/len(sess):.1%}) | splits={vc}")
