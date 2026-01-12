# policy_manager.py (complete)
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List, Dict, Iterable, Optional, Literal
import os, glob, sqlite3, random
from collections import defaultdict
import csv
from datetime import datetime
import shutil


DB_PATH = os.path.join(os.path.dirname(__file__), "sessions.db")
"""
When the pipeline runs, the policy_manager (via SQLite) will:
1.	Look at all sessions.
2.	If stratify_by is set (e.g., "dominant_label"), group sessions by that column.
3.	Within each group, assign splits according to stratified_ratio.
4.	If stratify_by=None, skip grouping — just apply the ratio globally.
5.	Write those split assignments (train, val, test) back into sessions.db.
6.If stratify_when : 
    * = "if_db_missing": only stratify when DB `split` is NULL/empty for eligible sessions;
    * = "always": ignore DB split and compute stratified sets fresh each time.
Important tip: Because the field of 'split' is set inside register_scanned_session module, so this filed has always value. 
So, if we want to take advantage if in-memory split and dominant_label at a same time
, we have no way except to set stratify_when = "always".
"""
# -----------------------
# Data policy (config)
# -----------------------
@dataclass
class DataPolicy:
    # Labeled/unlabeled participation in modes
    use_labeled_in_pretrain: bool = True
    unlabeled_both_except_finetune_selected: bool = True
    pretrain_splits_allowed: Tuple[str, ...] = ("train", "val")
    finetune_splits_allowed: Tuple[str, ...] = ("train", "val")

    # ---- In-memory stratified split (no DB writes) ----
    stratified_ratio: Optional[Tuple[float, float, float]] = None   # e.g. (0.8, 0.1, 0.1). None => disabled
    stratify_by: str = "dominant_label"                             # column in sessions.db
    stratify_scope: Literal["all", "labeled"] = "labeled"           # which sessions are eligible
    stratify_seed: int = 42
    stratify_when: Literal["always", "if_db_missing"] = "if_db_missing"

# -----------------------
# DB helpers
# -----------------------
def _connect():
    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con

def counts_summary() -> Dict[str, int]:
    with _connect() as con:
        row = con.execute("""
          SELECT
            SUM(CASE WHEN has_labels=1 THEN 1 ELSE 0 END),
            SUM(CASE WHEN has_labels=0 THEN 1 ELSE 0 END)
          FROM sessions
        """).fetchone()
    labeled = row[0] or 0
    unlabeled = row[1] or 0
    return {"labeled": labeled, "unlabeled": unlabeled, "total": labeled + unlabeled}

def get_db_split_counts() -> Dict[str, int]:
    with _connect() as con:
        rows = con.execute("""
            SELECT COALESCE(split,'') AS split, COUNT(*) FROM sessions GROUP BY split
        """).fetchall()
    return { (r[0] or "(empty)"): r[1] for r in rows }

def _fetch_rows_for_scope(db_path: str, scope: str) -> List[Dict]:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    scope_sql = "" if scope == "all" else "WHERE COALESCE(has_labels,0)=1"
    cur.execute(f"""
        SELECT session_id,
               COALESCE(split,'') AS split,
               COALESCE(has_labels,0) AS has_labels,
               COALESCE(dominant_label,'') AS dominant_label
        FROM sessions
        {scope_sql}
    """)
    rows = [dict(r) for r in cur.fetchall()]
    con.close()
    return rows

# -----------------------
# Policy: allowed_modes
# -----------------------
def apply_allowed_modes(policy: DataPolicy) -> None:
    """
    Enforce allowed_modes according to policy (writes to DB).
    - Labeled: 'both' if use_labeled_in_pretrain else 'finetune_only'
    - Unlabeled:
        if unlabeled_both_except_finetune_selected:
            'both' by default; but 'finetune_only' where lock_to_finetune=1
          else:
            'pretrain_only'
    """
    with _connect() as con:
        if policy.use_labeled_in_pretrain:
            con.execute("UPDATE sessions SET allowed_modes='both' WHERE has_labels=1;")
        else:
            con.execute("UPDATE sessions SET allowed_modes='finetune_only' WHERE has_labels=1;")

        if policy.unlabeled_both_except_finetune_selected:
            con.execute("UPDATE sessions SET allowed_modes='both' WHERE has_labels=0;")
            con.execute("UPDATE sessions SET allowed_modes='finetune_only' WHERE has_labels=0 AND COALESCE(lock_to_finetune,0)=1;")
        else:
            con.execute("UPDATE sessions SET allowed_modes='pretrain_only' WHERE has_labels=0;")
        con.commit()

def print_policy_and_counts(policy: DataPolicy) -> None:
    cs = counts_summary()
    #print("📑 Policy:",
          #f"use_labeled_in_pretrain={policy.use_labeled_in_pretrain},",
          #f"unlabeled_both_except_finetune_selected={policy.unlabeled_both_except_finetune_selected},",
          #f"pretrain_splits={policy.pretrain_splits_allowed},",
          #f"finetune_splits={policy.finetune_splits_allowed},",
          #f"stratified_ratio={policy.stratified_ratio},",
          #f"stratify_by='{policy.stratify_by}', scope='{policy.stratify_scope}', when='{policy.stratify_when}', seed={policy.stratify_seed}")
    print("📊 DB counts(Has-Label or Not?):", cs)
    print("📊 Static DB split histogram:", get_db_split_counts())

# -----------------------
# Stratified split logic
# -----------------------
def _should_use_stratified(policy: DataPolicy, rows: List[Dict]) -> Tuple[bool, str]:
    if policy.stratified_ratio is None:
        return False, "policy.stratified_ratio is None"
    when = str(getattr(policy, "stratify_when", "if_db_missing")).lower()
    if when not in {"always", "if_db_missing"}:
        when = "if_db_missing"
    if when == "always":
        return True, "policy.stratify_when='always'"
    # if_db_missing: only if any eligible row is missing a split
    missing = any(not str(r.get("split","")).strip() for r in rows)
    return (True, "some sessions have no split in DB") if missing else (False, "all sessions already have split in DB")

def _stratified_partition(rows: List[Dict], label_key: str,
                          ratio: Tuple[float,float,float], seed: int) -> Dict[str,set]:
    tr, va, te = ratio
    if abs((tr+va+te) - 1.0) > 1e-6:
        raise ValueError(f"stratified_ratio must sum to 1.0, got {ratio}")
    by_cls: Dict[str, List[str]] = defaultdict(list)
    for r in rows:
        sid = r["session_id"]
        lab = str(r.get(label_key,"")).strip()
        if lab:  # skip unlabeled buckets
            by_cls[lab].append(sid)

    rng = random.Random(seed)
    out = {"train": set(), "val": set(), "test": set()}
    for _, ids in by_cls.items():
        rng.shuffle(ids)
        n = len(ids)
        n_tr = int(round(tr * n))
        n_va = int(round(va * n))
        n_te = n - n_tr - n_va
        out["train"].update(ids[:n_tr])
        out["val"].update(ids[n_tr:n_tr+n_va])
        out["test"].update(ids[n_tr+n_va:])
    return out

def _ids_to_paths(ids: List[str], base_dir: str) -> List[str]:
    paths = []
    for sid in ids:
        p_exact = os.path.join(base_dir, f"{sid}.json")
        if os.path.isfile(p_exact):
            paths.append(p_exact); continue
        # fallback: any file starting with session_id; prefer one without "with text"
        candidates = sorted(glob.glob(os.path.join(base_dir, f"{sid}*.json")))
        if candidates:
            prefer = [p for p in candidates if "with text" not in os.path.basename(p).lower()]
            paths.append(prefer[0] if prefer else candidates[0])
    return paths

# -----------------------
# Main selection API
# -----------------------
def select_paths_for_mode(mode: str, policy: DataPolicy, base_dir: str) -> List[str]:
    """
    Returns JSON file paths for the requested mode.
    - If stratification is enabled by policy, uses in-memory stratified split.
    - Otherwise, falls back to DB 'split'.
    Prints a one-line summary of which path was used.
    The pipeline does not always use all data splits.If Mode allows for 'train/val' splits,
    Only train + val sessions are passed to 'merge_jsons_to_featurevector'.
    Instead, it selectively loads sessions depending on the current MODE of the system:

    Pretrain mode:
    Uses only train and validation sessions.
    (Unlabeled data can also be included, depending on policy.)

    Finetune or Interpret mode:
    Also restricted to the train and validation splits.
    This prevents accidental leakage of the held-out test set during training.

    Evaluate mode:
    Uses only the test sessions to produce final performance metrics.

    This ensures that:

    The test set is always kept untouched during pretraining and finetuning, preserving it for unbiased
    evaluation.
    Train/val data are the only splits allowed to influence the model during learning.

    """
    rows = _fetch_rows_for_scope(DB_PATH, policy.stratify_scope)
    use_strat, reason = _should_use_stratified(policy, rows)

    if mode == "pretrain":
        allowed = set(policy.pretrain_splits_allowed)
    elif mode in ("finetune", "interpret"):
        allowed = set(policy.finetune_splits_allowed)
    elif mode == "evaluate":
        allowed = {"test"}
    else:
        allowed = {"train","val","test"}

    if use_strat:
        assignment = _stratified_partition(rows, policy.stratify_by, policy.stratified_ratio, policy.stratify_seed)
        selected_ids = sorted(set().union(*(assignment[s] for s in allowed if s in assignment)))
        counts = {k: len(v) for k, v in assignment.items()}
        print(f"🔀 In-memory Split was activated(Using STRATIFIED split) — {reason}. ratio={policy.stratified_ratio}, by='{policy.stratify_by}', "
              f"scope='{policy.stratify_scope}', seed={policy.stratify_seed}. Stratified split counts={counts} | selected for Mode = {mode}:{len(selected_ids)}")
        save_stratified_split_csv(assignment, rows)
        return _ids_to_paths(selected_ids, base_dir), True    # when stratified

    # Fallback: use DB 'split'
    allowed_lower = {s.lower() for s in allowed}
    from_ids = [r["session_id"] for r in rows if str(r.get("split","")).lower() in allowed_lower]
    print(f"❌ In-memory Split was inactivated(NOT using stratification) — {reason}. Returning {len(from_ids)} file(s) from DB splits={sorted(allowed)}.")
    return _ids_to_paths(from_ids, base_dir), False       # when DB split



def save_stratified_split_csv(assignment: dict, rows: list[dict], out_dir: str = "./artifacts/stratified_splits") -> str:
    """
    Save the in-memory stratified split assignments to a CSV file.

    Parameters
    ----------
    assignment : dict
        Mapping of split names to sets/lists of session IDs, e.g. {'train': [...], 'val': [...], 'test': [...]}
    rows : list of dict
        Session metadata rows used for stratification (should include 'session_id' and optionally 'dominant_label')
    out_dir : str, optional
        Directory where the CSV file will be saved (default: ./artifacts/stratified_splits)

    Returns
    -------
    str
        The full path to the saved CSV file.
    """
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(out_dir, f"stratified_split_{timestamp}.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["session_id", "split", "dominant_label"])  # header
        for split_name, ids in assignment.items():
            for sid in ids:
                label = next((r.get("dominant_label", "") for r in rows if r["session_id"] == sid), "")
                writer.writerow([sid, split_name, label])

    print(f"📁 Stratified split results saved to: {csv_path}")
    # 🔁 also update a stable 'latest.csv' pointer
    latest_path = os.path.join(out_dir, "latest.csv")
    try:
        shutil.copy(csv_path, latest_path)
        print(f"🔁 Updated latest stratified mapping: {latest_path}")
    except Exception as e:
        print(f"⚠️ Could not update {latest_path}: {e}")
    return csv_path



