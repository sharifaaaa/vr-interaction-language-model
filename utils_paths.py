# utils_paths.py
import re

# characters that are illegal in Windows file names
_ILLEGAL = r'[<>:"/\\|?*\x00-\x1F]'

def safe_filename(name: str, max_len: int = 200) -> str:
    s = str(name)
    # if your IDs have a "json::..." suffix, drop that for file names
    if "::" in s:
        s = s.split("::", 1)[0]
    # replace illegal chars with underscore
    s = re.sub(_ILLEGAL, "_", s)
    # collapse whitespace → underscore, trim leading/trailing dots/underscores
    s = re.sub(r"\s+", "_", s).strip("._")
    # optional: keep it short for OS limits
    return s[:max_len]
