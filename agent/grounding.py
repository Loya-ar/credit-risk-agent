"""Check that every number in an answer comes from the data.

Used by the agent (it rewrites or withholds an answer with unsupported numbers)
and by the eval harness (which fails such answers), so both apply the same rule.
"""
from __future__ import annotations

import math
import re

_NUMBER = re.compile(r"(?<![\w.])-?\$?(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(%|k\b)?", re.IGNORECASE)


def is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and not math.isnan(v)


def numbers_in(text: str) -> list[tuple[float, int]]:
    """(value, decimals shown) for every number in the text; '$150k' -> 150000."""
    out = []
    for m in _NUMBER.finditer(text or ""):
        whole, frac, suffix = m.group(1).replace(",", ""), m.group(2) or "", (m.group(3) or "").lower()
        value = float(whole + frac)
        if m.group(0).startswith("-"):
            value = -value
        decimals = len(frac) - 1 if frac else 0
        if suffix == "k":
            value, decimals = value * 1000, 0
        out.append((value, decimals))
    return out


def shown_as(value: float, decimals: int, target) -> bool:
    """The number as written could be `target` rounded (as is or as a percent)."""
    if not is_number(target):
        return False
    tol = 0.5 * 10 ** -decimals + 1e-9
    return any(abs(value - target * s) <= tol * max(1, s / 100) for s in (1, 100))


def collect_numbers(obj, out: list | None = None) -> list[float]:
    out = [] if out is None else out
    if isinstance(obj, dict):
        for v in obj.values():
            collect_numbers(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            collect_numbers(v, out)
    elif is_number(obj):
        out.append(float(obj))
    elif isinstance(obj, str):
        out.extend(v for v, _ in numbers_in(obj))
    return out


def ungrounded_numbers(question: str, answer: str, tool_calls: list[dict]) -> list[float]:
    """Numbers in the answer that no tool output, tool argument or the question contains.

    Integers up to 10 are ignored ("top 5 drivers", "decile 10").
    """
    pool = [v for v, _ in numbers_in(question)]
    for call in tool_calls:
        collect_numbers(call.get("args"), pool)
        collect_numbers(call.get("result"), pool)
    missing = []
    for value, decimals in numbers_in(answer):
        if decimals == 0 and abs(value) <= 10:
            continue
        if not any(shown_as(value, decimals, p) for p in pool):
            missing.append(value)
    return missing
