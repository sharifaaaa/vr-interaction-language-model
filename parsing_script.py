# parsing_script.py
from __future__ import annotations

import json
import re
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Your project config should define these lists
# - BEHAVIORAL_COLS: numeric sensor/pose columns
# - CATEGORICAL_COLS: string categorical columns
# - META_COLS: meta columns such as session_id, phase, area, timestamp, speaker, text, etc.
from config_features import BEHAVIORAL_COLS, CATEGORICAL_COLS, META_COLS

_NUMERIC_DEFAULT = 0.0
_TEXT_DEFAULT = ""

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _g(d: Dict[str, Any], *keys, default=None):
    """Nested dict getter with default."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def _conv(d: Dict[str, Any]) -> Dict[str, Any]:
    """Locate conversation block under flexible keys."""
    return d.get("conversation_data") or d.get("conversation") or d.get("dialog") or {}

_ZW = re.compile(r"[\u200b\u200c\ufeff]")  # zero-width chars

def _clean_text(x: Any) -> str:
    if not isinstance(x, str):
        return ""
    return _ZW.sub("", x).strip()

def _first_nonempty(*vals, fallback=""):
    for v in vals:
        if isinstance(v, str) and v.strip():
            return v
    return fallback

def _to_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return np.nan

# ------------------------------------------------------------
# Parser
# ------------------------------------------------------------

def parse_json_to_dataframe(json_path: str) -> pd.DataFrame:
    """
    Parse one session JSON into a flat DataFrame.
    Emits one row per 'frame-like' node (has pose and/or carries conversation text).
    Avoids double-emit by NOT recursing into pose/conversation children when a row is already emitted at this node.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        root = json.load(f)

    rows: List[Dict[str, Any]] = []

    # DFS with context propagation
    def visit(node: Any, ctx: Dict[str, Any]):
        if isinstance(node, dict):
            # 1) Carry context from ancestors
            new_ctx = dict(ctx)
            for k in ("session_id", "phase", "area", "timestamp", "gaze_actor"):
                v = node.get(k)
                if v not in (None, ""):
                    new_ctx[k] = v

            # 2) Conversation fields (various shapes)
            conv = _conv(node)
            speaker = _first_nonempty(
                _clean_text(conv.get("speaker")),
                _clean_text(node.get("speaker")),
                _clean_text(new_ctx.get("speaker", "")),
                fallback=""
            )
            text = _first_nonempty(
                _clean_text(conv.get("text")),
                _clean_text(node.get("text")),
                fallback=""
            )

            # 3) Pose / controller / speeds (read from typical locations, fallback to flat keys)
            # HMD pose
            head_x = _g(node, "hmd_data", "position", "x", default=np.nan)
            head_y = _g(node, "hmd_data", "position", "y", default=np.nan)
            head_z = _g(node, "hmd_data", "position", "z", default=np.nan)
            # Gaze vector
            gaze_x = _g(node, "hmd_data", "gaze_vector", "x", default=np.nan)
            gaze_y = _g(node, "hmd_data", "gaze_vector", "y", default=np.nan)
            gaze_z = _g(node, "hmd_data", "gaze_vector", "z", default=np.nan)
            # Controllers
            rpx = _g(node, "controller_data", "r_position", "x", default=np.nan)
            rpy = _g(node, "controller_data", "r_position", "y", default=np.nan)
            rpz = _g(node, "controller_data", "r_position", "z", default=np.nan)
            lpx = _g(node, "controller_data", "l_position",  "x", default=np.nan)
            lpy = _g(node, "controller_data", "l_position",  "y", default=np.nan)
            lpz = _g(node, "controller_data", "l_position",  "z", default=np.nan)

            # Interacted actors (controller-level or flat)
            r_inter = _g(node, "controller_data", "r_interacted_actor",
                         default=node.get("r_interacted_actor", _TEXT_DEFAULT))
            l_inter = _g(node, "controller_data", "l_interacted_actor",
                         default=node.get("l_interacted_actor", _TEXT_DEFAULT))

            # Movement speeds (controller-level or hmd-level or flat)
            r_ms = _g(node, "controller_data", "r_movement_speed",
                      default=node.get("r_movement_speed", _NUMERIC_DEFAULT))
            l_ms = _g(node, "controller_data", "l_movement_speed",
                      default=node.get("l_movement_speed", _NUMERIC_DEFAULT))
            ms   = _g(node, "hmd_data", "movement_speed",
                      default=node.get("movement_speed", _NUMERIC_DEFAULT))

            # Gaze actor (hmd-level or node-level or carried from context)
            gaze_actor_here = _g(node, "hmd_data", "gaze_actor", default=node.get("gaze_actor", ""))
            gaze_actor = _first_nonempty(_clean_text(gaze_actor_here), _clean_text(new_ctx.get("gaze_actor", "")))

            # User emotion (top-1 label you carried in timeline)
            user_emotion = _clean_text(node.get("user_emotion", ""))

            # 4) Should we emit at this node?
            has_pose = ("hmd_data" in node) or ("controller_data" in node)
            emitted_here = False
            if text or has_pose:
                rows.append({
                    # meta/context
                    "session_id":  _clean_text(new_ctx.get("session_id", "")),
                    "timestamp":   new_ctx.get("timestamp", None),  # keep raw; convert later in pipeline
                    "phase":       _clean_text(new_ctx.get("phase", "")),
                    "area":        _clean_text(new_ctx.get("area", "")),
                    "speaker":     speaker if speaker != "none" else "",
                    "text":        text if text != "none" else "",

                    # gaze / pose / controllers
                    "gaze_x": gaze_x, "gaze_y": gaze_y, "gaze_z": gaze_z,
                    "gaze_actor": gaze_actor,
                    "head_pos_x": head_x, "head_pos_y": head_y, "head_pos_z": head_z,
                    "r_pos_x": rpx, "r_pos_y": rpy, "r_pos_z": rpz,
                    "l_pos_x": lpx, "l_pos_y": lpy, "l_pos_z": lpz,

                    # interactions & speeds
                    "r_interacted_actor": _clean_text(r_inter),
                    "l_interacted_actor": _clean_text(l_inter),
                    "r_movement_speed": r_ms,
                    "l_movement_speed": l_ms,
                    "movement_speed": ms,

                    # labels
                    "user_emotion": user_emotion,
                })
                emitted_here = True

            # 5) Recurse — but if we already emitted a “frame” here, skip
            #    direct recursion into pose blocks & conversation block, to avoid double-emits.
            for key, val in node.items():
                if emitted_here and key in ("hmd_data", "controller_data", "conversation_data", "conversation", "dialog"):
                    continue
                visit(val, new_ctx)

        elif isinstance(node, list):
            for v in node:
                visit(v, ctx)

    visit(root, {})

    # Build DataFrame
    df = pd.DataFrame(rows)

    # Ensure required columns exist and dtypes are correct.
    required_numeric = set(BEHAVIORAL_COLS)
    required_other   = set(CATEGORICAL_COLS + META_COLS) - required_numeric

    for c in required_numeric:
        if c not in df.columns:
            df[c] = _NUMERIC_DEFAULT
    for c in required_other:
        if c not in df.columns:
            df[c] = _TEXT_DEFAULT

    # Coerce numeric behavioral columns
    for c in required_numeric:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)

    # Normalize some string columns (trim)
    for c in (set(df.columns) & set(CATEGORICAL_COLS + META_COLS)):
        df[c] = df[c].astype(str).map(lambda s: s.strip() if isinstance(s, str) else s)

    return df
