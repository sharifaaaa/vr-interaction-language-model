# sessions_fetcher.py
from __future__ import annotations
import os
import json
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import requests
from requests.exceptions import RequestException
import config_attention as CA

# ----------------------------- defaults -----------------------------
BASE = "https://1emotional-tracking.vercel.app"   # change if needed
TIMEOUT_SEC = 60

# Default directories
EXPERIMENT_DIR = "../experiment_data/"
EXPERIMENT_DIR_FINETUNE = "../experiment_data_Finetune/"

# ----------------------------- utils -----------------------------
def _ensure_dir(path: Path | str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)

def _download_json(url: str) -> dict:
    r = requests.get(url, timeout=TIMEOUT_SEC)
    r.raise_for_status()
    try:
        return r.json()
    except ValueError:
        return json.loads(r.content.decode("utf-8", errors="replace"))

# ----------------------------- API calls -----------------------------
def list_sessions_from_api(base: str = BASE) -> List[Dict]:
    data = _download_json(f"{base}/api/sessions")
    if isinstance(data, dict) and "items" in data and isinstance(data["items"], list):
        return data["items"]
    if isinstance(data, list) and (not data or isinstance(data[0], dict)):
        return data
    raise RuntimeError("Unexpected /api/sessions response shape")

def download_all_sessions_to_dir(
    out_dir: Path | str,
    base: str = BASE,
    skip_existing: bool = True
) -> int:
    out_dir = Path(out_dir)
    _ensure_dir(out_dir)
    sessions = list_sessions_from_api(base)
    session_ids: List[str] = []

    for s in sessions:
        sid = s.get("session_id") or s.get("id")
        if isinstance(sid, str):
            session_ids.append(sid)

    if not session_ids:
        raise RuntimeError("API returned 0 sessions or missing 'session_id'.")

    print(f"🌐 [Remote] Found {len(session_ids)} session(s) on the server")

    written = 0
    for i, sid in enumerate(session_ids, start=1):
        fname = f"session_{sid}.json"
        fpath = out_dir / fname
        if skip_existing and fpath.exists():
            #print(f"[{i}/{len(session_ids)}] Skipping {fname} (already exists)")
            continue

        url = f"{base}/api/sessions/{sid}/download"
        try:
            payload = _download_json(url)
        except RequestException as e:
            print(f"[{i}/{len(session_ids)}] ⚠️ Failed {sid}: {e}")
            continue

        with fpath.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        written += 1
        print(f"[{i}/{len(session_ids)}] ✅ Saved {fname}")

    print(f"🌐 [Remote] Done. New files: {written}")
    return written


def fetch_sessions_with_fallback(
    experiment_dir: Path | str = EXPERIMENT_DIR,
    base: str = BASE
) -> Tuple[str, bool, int]:
    """
    Try to download from API into experiment_dir.
    On error, fall back to local_fallback_dir.

    Returns: (used_dir, remote_ok, new_files_count)
    """
    used_dir = str(Path(experiment_dir))
    remote_ok = False
    new_files = 0

    try:
        new_files = download_all_sessions_to_dir(experiment_dir, base=base, skip_existing=True)
        remote_ok = True
        if new_files == 0:
            print("ℹ️ Remote API reachable, but no new files were downloaded (maybe all are already present).")
    except (RequestException, RuntimeError) as e:
        if CA.MODE == "finetune":
            fbk = EXPERIMENT_DIR_FINETUNE
        else:
            fbk = EXPERIMENT_DIR
        print(f"🔁 Remote API not available or malformed: {e}\n"
              f"➡️ Falling back to local folder: {fbk}")
        used_dir = str(Path(fbk))

    return used_dir, remote_ok, new_files
