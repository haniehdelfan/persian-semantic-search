"""Deterministic 'realistic noise' for queries.

Persian Wikipedia is already clean, so normalisation can look useless on clean queries.
Real users type on Arabic keyboards (ي/ك instead of ی/ک), use Arabic-Indic digits and
forget the half-space. ``noisy_query`` simulates exactly that.
"""
from __future__ import annotations

_NOISE = str.maketrans(
    {
        "\u06CC": "\u064A",  # ی -> ي
        "\u06A9": "\u0643",  # ک -> ك
        **dict(zip("۰۱۲۳۴۵۶۷۸۹", "٠١٢٣٤٥٦٧٨٩")),
    }
)


def noisy_query(q: str) -> str:
    return q.translate(_NOISE).replace("\u200c", " ")
