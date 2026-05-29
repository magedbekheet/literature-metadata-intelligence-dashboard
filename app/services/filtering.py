from __future__ import annotations

import re
from typing import Dict, List

import pandas as pd

from .schema import metadata_blob, safe_text

FIELDS = ["title", "abstract", "authors", "journal", "year", "doi", "keywords", "source"]


def _tokens(query: str) -> List[str]:
    query = safe_text(query)
    if not query:
        return []
    parts = re.split(r"\s+OR\s+|\s+AND\s+|,|;", query, flags=re.I)
    return [p.strip().strip('"') for p in parts if p.strip()]


def match_query(text: str, query: str, mode: str = "AND") -> bool:
    terms = _tokens(query)
    if not terms:
        return True
    text = safe_text(text).lower()
    checks = [t.lower() in text for t in terms]
    return any(checks) if mode.upper() == "OR" else all(checks)


def _parse_year(value: str) -> int | None:
    text = safe_text(value)
    if not text:
        return None
    m = re.search(r"\b(19|20)\d{2}\b", text)
    return int(m.group(0)) if m else None


def filter_dataframe(df: pd.DataFrame, *, global_query: str = "", global_fields: List[str] | None = None, field_queries: Dict[str, str] | None = None, logic: str = "AND", source_filter: List[str] | None = None, year_from: str = "", year_to: str = "") -> pd.DataFrame:
    if df.empty:
        return df
    rows = []
    field_queries = field_queries or {}
    from_year = _parse_year(year_from)
    to_year = _parse_year(year_to)
    for _, row in df.iterrows():
        paper = row.to_dict()
        ok = True
        if global_query:
            ok = ok and match_query(metadata_blob(paper, global_fields), global_query, logic)
        for field, q in field_queries.items():
            if safe_text(q):
                value = metadata_blob(paper, [field])
                ok = ok and match_query(value, q, logic)
        y = _parse_year(paper.get("year"))
        if from_year is not None:
            ok = ok and y is not None and y >= from_year
        if to_year is not None:
            ok = ok and y is not None and y <= to_year
        if source_filter:
            ok = ok and safe_text(paper.get("source")) in source_filter
        if ok:
            rows.append(paper)
    return pd.DataFrame(rows).reset_index(drop=True) if rows else df.iloc[0:0].copy()
