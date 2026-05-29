from __future__ import annotations

import json
import re
from typing import List

import pandas as pd

from .schema import safe_text


def _bibtex_escape(value: str) -> str:
    return safe_text(value).replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


def _key(row: pd.Series, used: set[str]) -> str:
    title = safe_text(row.get("title"))
    year = safe_text(row.get("year"))
    author = safe_text(row.get("authors_display") or row.get("authors"))
    author_part = re.findall(r"[A-Za-z0-9]+", author.lower())
    title_part = re.findall(r"[A-Za-z0-9]+", title.lower())
    parts = [author_part[-1] if author_part else "paper"]
    if year:
        parts.append(year[:4])
    parts.extend(title_part[:2])
    key = re.sub(r"[^A-Za-z0-9_:-]", "", "_".join(parts)) or "paper"
    base = key
    suffix = 2
    while key.lower() in used:
        key = f"{base}_{suffix}"
        suffix += 1
    used.add(key.lower())
    return key


def to_bibtex(df: pd.DataFrame) -> str:
    entries: List[str] = []
    used_keys: set[str] = set()
    for _, row in df.iterrows():
        title = safe_text(row.get("title"))
        key = _key(row, used_keys)
        authors = safe_text(row.get("authors_display") or row.get("authors")).replace(", ", " and ")
        fields = {
            "title": title,
            "author": authors,
            "journal": safe_text(row.get("journal")),
            "year": safe_text(row.get("year")),
            "doi": safe_text(row.get("doi")),
            "url": safe_text(row.get("url")),
        }
        body = "\n".join(f"  {k} = {{{_bibtex_escape(v)}}}," for k, v in fields.items() if v)
        entries.append(f"@article{{{key},\n{body}\n}}")
    return "\n\n".join(entries) + "\n"


def to_ris(df: pd.DataFrame) -> str:
    entries: List[str] = []
    for _, row in df.iterrows():
        lines = ["TY  - JOUR"]
        if safe_text(row.get("title")): lines.append(f"TI  - {safe_text(row.get('title'))}")
        for au in safe_text(row.get("authors_display") or row.get("authors")).split(", "):
            if au: lines.append(f"AU  - {au}")
        if safe_text(row.get("journal")): lines.append(f"JO  - {safe_text(row.get('journal'))}")
        if safe_text(row.get("year")): lines.append(f"PY  - {safe_text(row.get('year'))}")
        if safe_text(row.get("doi")): lines.append(f"DO  - {safe_text(row.get('doi'))}")
        if safe_text(row.get("url")): lines.append(f"UR  - {safe_text(row.get('url'))}")
        if safe_text(row.get("abstract")): lines.append(f"AB  - {safe_text(row.get('abstract'))}")
        lines.append("ER  -")
        entries.append("\n".join(lines))
    return "\n\n".join(entries) + "\n"


def to_markdown(df: pd.DataFrame) -> str:
    lines = ["# Selected literature", ""]
    for i, (_, row) in enumerate(df.iterrows(), 1):
        lines += [
            f"## {i}. {safe_text(row.get('title')) or 'Untitled'}",
            f"- Authors: {safe_text(row.get('authors_display') or row.get('authors'))}",
            f"- Journal: {safe_text(row.get('journal'))}",
            f"- Year: {safe_text(row.get('year'))}",
            f"- DOI: {safe_text(row.get('doi'))}",
            f"- URL: {safe_text(row.get('url'))}",
            "",
            safe_text(row.get("abstract")),
            "",
        ]
    return "\n".join(lines)


def to_json(df: pd.DataFrame) -> str:
    return json.dumps(df.to_dict(orient="records"), ensure_ascii=False, indent=2)


def to_jsonl(df: pd.DataFrame) -> str:
    return "\n".join(json.dumps(x, ensure_ascii=False) for x in df.to_dict(orient="records")) + "\n"
