from __future__ import annotations

import re
from typing import Any


_ws = re.compile(r"\s+")


def clean_headline(text: Any) -> str:
    """
    Defensive text cleaner:
    - Handles NaN/None/non-strings safely
    - Normalizes whitespace
    """
    if text is None:
        return ""
    # Pandas may pass floats for NaN; convert safely
    try:
        t = str(text)
    except Exception:
        return ""

    # Treat literal "nan" (string) as empty
    if t.lower() == "nan":
        return ""

    t = t.strip()
    if not t:
        return ""

    t = _ws.sub(" ", t)
    return t