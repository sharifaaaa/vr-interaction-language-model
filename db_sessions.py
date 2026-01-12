# db_sessions.py
from __future__ import annotations
import os, sqlite3, hashlib, time, json, glob
from typing import Iterable, List, Tuple, Optional, Dict

DB_PATH = os.path.join(os.path.dirname(__file__), "sessions.db")  # absolute path

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
  session_id       TEXT PRIMARY KEY,
  user_id          TEXT,
  data_path        TEXT NOT NULL,
  source           TEXT CHECK(source IN ('remote','local')),
  created_at       TEXT,
  downloaded_at    TEXT,
  n_records        INTEGER,
  has_labels       INTEGER CHECK(has_labels IN (0,1)) DEFAULT 0,
  split            TEXT CHECK(split IN ('train','val','test')),
  allowed_modes    TEXT CHECK(allowed_modes IN ('pretrain_only','finetune_only','both')) DEFAULT 'both',
  hash_bucket      INTEGER,
  lock_to_finetune INTEGER DEFAULT 0,
  -- NEW: quality of hand/controller tracking (0=bad, 1=good, NULL=unknown)
  has_valid_hands  INTEGER CHECK(has_valid_hands IN (0,1)),
  -- NEW: session-level label summary (for stratified selection etc.)
  dominant_label   TEXT,
  label_mix        TEXT,   -- JSON like {"neutral":0.83,"joy":0.12,...}
  notes            TEXT
);
CREATE INDEX IF NOT EXISTS idx_sessions_split      ON sessions(split);
CREATE INDEX IF NOT EXISTS idx_sessions_modes      ON sessions(allowed_modes);
CREATE INDEX IF NOT EXISTS idx_sessions_domlab     ON sessions(dominant_label);
CREATE INDEX IF NOT EXISTS idx_sessions_hand_qual  ON sessions(has_valid_hands);
"""

def _connect():
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con

def init_db():
    with _connect() as con:
        con.executescript(SCHEMA_SQL)
        # In case DB existed before and lacks newer columns, add them defensively:
        for col_sql in (
            "ALTER TABLE sessions ADD COLUMN lock_to_finetune INTEGER DEFAULT 0;",
            "ALTER TABLE sessions ADD COLUMN dominant_label TEXT;",
            "ALTER TABLE sessions ADD COLUMN label_mix TEXT;",
            # NEW: hand-quality column (nullable; 0/1 only when known)
            "ALTER TABLE sessions ADD COLUMN has_valid_hands INTEGER CHECK(has_valid_hands IN (0,1));",
        ):
            try:
                con.execute(col_sql)
            except sqlite3.OperationalError:
                # column already exists
                pass
        # index may already exist; ignore errors
        try:
            con.execute("CREATE INDEX IF NOT EXISTS idx_sessions_hand_qual ON sessions(has_valid_hands);")
        except sqlite3.OperationalError:
            pass

def _stable_bucket(key: str, buckets: int = 1000) -> int:
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % buckets

def _assign_split_from_bucket(bucket: int) -> str:
    if bucket < 800: return "train"
    if bucket < 900: return "val"
    return "test"

# -----------------------------
# Session metadata extraction
# -----------------------------

def _normalize_label(v) -> Optional[str]:
    """Return a clean label string or None if NA/empty."""
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.upper() == "NA":
        return None
    return s

def _load_json_any(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        pass
    # Try JSONL (one object per line)
    try:
        items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except Exception:
                    pass
        if items:
            return items
    except Exception:
        pass
    return {}

def _infer_session_meta(json_path: str) -> Dict:
    """
    Parse a session JSON/JSONL and return:
      session_id, user_id, created_at, has_labels, n_records,
      dominant_label (or None), label_mix (dict)
    NOTE: we do NOT compute has_valid_hands here (too heavy for plain JSON).
          That is computed later from the parsed DataFrame and written back
          using `update_hand_quality(...)`.
    """
    data = _load_json_any(json_path)

    # basic header-ish fields
    session_id = None
    user_id = None
    created_at = None
    if isinstance(data, dict):
        session_id = data.get("session_id")
        user_id = data.get("user_id")
        created_at = data.get("created_at")

    if not session_id:
        session_id = os.path.splitext(os.path.basename(json_path))[0]

    # collect "records"
    records = []
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        if isinstance(data.get("records"), list): records = data["records"]
        elif isinstance(data.get("events"), list): records = data["events"]
        else:
            # best-effort: first list-typed value
            for v in data.values():
                if isinstance(v, list):
                    records = v; break
    if not isinstance(records, list):
        records = []

    # label stats
    counts: Dict[str, int] = {}
    for r in records:
        if not isinstance(r, dict):
            continue
        lab = _normalize_label(r.get("user_emotion"))
        if lab is None:
            continue
        counts[lab] = counts.get(lab, 0) + 1

    n_records = len(records)
    has_labels = int(sum(counts.values()) > 0)

    if counts:
        # dominant = argmax count (ties broken lexicographically by label name for determinism)
        dominant_label = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        total_lab = float(sum(counts.values()))
        label_mix = {k: (v / total_lab) for k, v in counts.items()}
    else:
        dominant_label = None
        label_mix = {}

    return dict(
        session_id=session_id,
        user_id=user_id,
        created_at=created_at,
        n_records=n_records,
        has_labels=has_labels,
        dominant_label=dominant_label,
        label_mix=label_mix,
    )

# -----------------------------
# Upserts / registration
# -----------------------------

def upsert_session(
    *,
    session_id: str,
    data_path: str,
    source: str,
    n_records: int,
    has_labels: int,
    user_id: Optional[str] = None,
    created_at: Optional[str] = None,
    downloaded_at: Optional[str] = None,
    allowed_modes: Optional[str] = None,
    dominant_label: Optional[str] = None,
    label_mix: Optional[Dict[str, float]] = None,
    has_valid_hands: Optional[int] = None,   # NEW: optional quality flag
) -> None:
    """
    Insert/update a session row.
    - Split is determined by a stable hash bucket of session_id (80/10/10).
    - has_valid_hands is optional; when None, we keep any existing value.
      It can later be set via `update_hand_quality(...)`.
    """
    if allowed_modes is None:
        allowed_modes = "both"

    # *** POLICY 1: split is determined by SESSION_ID ***
    key_for_split = session_id
    bucket = _stable_bucket(key_for_split)
    split = _assign_split_from_bucket(bucket)

    ts = downloaded_at or time.strftime("%Y-%m-%d %H:%M:%S")
    label_mix_json = json.dumps(label_mix or {}, ensure_ascii=False)

    with _connect() as con:
        con.execute(
            """
            INSERT INTO sessions(
                session_id, user_id, data_path, source, created_at,
                downloaded_at, n_records, has_labels, split,
                allowed_modes, hash_bucket, has_valid_hands,
                dominant_label, label_mix
            )
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(session_id) DO UPDATE SET
                user_id        = COALESCE(excluded.user_id, sessions.user_id),
                data_path      = excluded.data_path,
                source         = excluded.source,
                created_at     = COALESCE(excluded.created_at, sessions.created_at),
                downloaded_at  = excluded.downloaded_at,
                n_records      = excluded.n_records,
                has_labels     = excluded.has_labels,
                -- keep existing split if already set
                split          = COALESCE(sessions.split, excluded.split),
                allowed_modes  = excluded.allowed_modes,
                hash_bucket    = excluded.hash_bucket,
                -- always refresh label summary from latest scan
                dominant_label = excluded.dominant_label,
                label_mix      = excluded.label_mix,
                -- NEW: only overwrite quality if caller provided a non-NULL flag
                has_valid_hands = COALESCE(excluded.has_valid_hands, sessions.has_valid_hands);
            """,
            (
                session_id, user_id, data_path, source, created_at,
                ts, n_records, has_labels, split,
                allowed_modes, bucket, has_valid_hands,
                dominant_label, label_mix_json,
            ),
        )

def register_scanned_sessions(folder: str, source: str) -> int:
    """
    Scan a folder for *.json, infer metadata (including dominant_label/label_mix),
    and upsert into sessions.db.
    At this stage we do NOT know has_valid_hands yet; it stays NULL until
    merge_jsons_to_featurevector computes it.
    """
    init_db()
    paths = glob.glob(os.path.join(folder, "*.json"))
    for p in paths:
        meta = _infer_session_meta(p)
        upsert_session(
            session_id=meta["session_id"],
            user_id=meta.get("user_id"),
            data_path=p,
            source=source,
            created_at=meta.get("created_at"),
            n_records=meta["n_records"],
            has_labels=meta["has_labels"],
            dominant_label=meta.get("dominant_label"),
            label_mix=meta.get("label_mix"),
            has_valid_hands=None,   # not known yet
        )
    return len(paths)

# -----------------------------
# Queries / utilities
# -----------------------------

def list_sessions_for_mode(
    *, mode: str, include_splits: Iterable[str] = ("train","val","test")
) -> List[Tuple[str, str]]:
    allowed = ("both", f"{mode}_only")
    q = f"""
      SELECT session_id, data_path
      FROM sessions
      WHERE allowed_modes IN ({",".join("?"*len(allowed))})
        AND split IN ({",".join("?"*len(tuple(include_splits)))})
      ORDER BY session_id;
    """
    with _connect() as con:
        return con.execute(q, (*allowed, *tuple(include_splits))).fetchall()

def counts_summary() -> Dict[str,int]:
    with _connect() as con:
        row = con.execute("""
          SELECT
            SUM(CASE WHEN has_labels=1 THEN 1 ELSE 0 END) as labeled,
            SUM(CASE WHEN has_labels=0 THEN 1 ELSE 0 END) as unlabeled
          FROM sessions
        """).fetchone()
    return {"labeled": row[0] or 0, "unlabeled": row[1] or 0}

# NEW: mark/unmark a list of sessions as "finetune-selected" (exclude from pretrain under policy)
def mark_finetune_selection(session_ids: List[str], lock: bool = True) -> int:
    if not session_ids: return 0
    with _connect() as con:
        con.executemany(
            "UPDATE sessions SET lock_to_finetune=? WHERE session_id=?;",
            [(1 if lock else 0, sid) for sid in session_ids]
        )
        con.commit()
    return len(session_ids)

# NEW: simple helper to write quality flag from merge step
def update_hand_quality(session_id: str, has_valid_hands: int) -> None:
    """
    Set the has_valid_hands flag for a session.
    has_valid_hands must be 0 or 1 (int/bool).
    """
    val = int(has_valid_hands)
    if val not in (0, 1):
        raise ValueError("has_valid_hands must be 0 or 1")
    with _connect() as con:
        con.execute(
            "UPDATE sessions SET has_valid_hands=? WHERE session_id=?;",
            (val, session_id),
        )
        con.commit()
