from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Iterable, List


def to_date(x: str) -> date:
    # Accept YYYY-MM-DD
    return datetime.strptime(x, "%Y-%m-%d").date()


def daterange(start: date, end: date) -> List[date]:
    # inclusive start, inclusive end
    if end < start:
        return []
    days = (end - start).days
    return [start + timedelta(days=i) for i in range(days + 1)]


def utc_today() -> date:
    # Good enough for this pipeline; avoids timezone bugs.
    return date.today()