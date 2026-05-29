from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path
from typing import Dict, List

from .schema import normalize_paper, safe_text


def parse_json_or_jsonl(text: str) -> List[Dict]:
    text = text.strip()
    if not text:
        return []
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return [normalize_paper(x) for x in obj if isinstance(x, dict)]
        if isinstance(obj, dict):
            if isinstance(obj.get("papers"), list):
                return [normalize_paper(x) for x in obj["papers"] if isinstance(x, dict)]
            return [normalize_paper(obj)]
    except Exception:
        pass
    rows = []
    for line in text.splitlines():
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(normalize_paper(obj))
        except Exception:
            continue
    return rows


def parse_csv(text: str) -> List[Dict]:
    f = io.StringIO(text.lstrip("\ufeff"))
    reader = csv.DictReader(f)
    return [normalize_paper(row) for row in reader]


def parse_bibtex(text: str) -> List[Dict]:
    entries = re.split(r"(?=@\w+\s*\{)", text.strip())
    papers = []
    for entry in entries:
        if not entry.strip().startswith("@"):
            continue
        fields = {}
        for m in re.finditer(r"(\w+)\s*=\s*[\{\"](.*?)[\}\"]\s*,?\s*(?=\s*\w+\s*=|\s*\})", entry, flags=re.S):
            fields[m.group(1).lower()] = re.sub(r"\s+", " ", m.group(2)).strip()
        papers.append(normalize_paper({
            "title": fields.get("title", ""),
            "authors": fields.get("author", "").replace(" and ", "; "),
            "journal": fields.get("journal") or fields.get("booktitle", ""),
            "year": fields.get("year", ""),
            "doi": fields.get("doi", ""),
            "url": fields.get("url", ""),
            "abstract": fields.get("abstract", ""),
            "source": "bibtex_import",
        }))
    return papers


def parse_ris(text: str) -> List[Dict]:
    chunks = re.split(r"\nER\s*-\s*", text)
    papers = []
    for ch in chunks:
        if not ch.strip():
            continue
        data = {"authors": []}
        for line in ch.splitlines():
            if len(line) < 6 or "  - " not in line:
                continue
            tag = line[:2]
            val = line.split("  - ", 1)[1].strip()
            if tag in {"TI", "T1"}:
                data["title"] = val
            elif tag in {"AB", "N2"}:
                data["abstract"] = val
            elif tag in {"AU", "A1"}:
                data["authors"].append(val)
            elif tag in {"JO", "JF", "JA", "T2"}:
                data["journal"] = val
            elif tag == "PY":
                data["year"] = val[:4]
            elif tag == "DO":
                data["doi"] = val
            elif tag in {"UR", "L1"}:
                data["url"] = val
        data["source"] = "ris_import"
        papers.append(normalize_paper(data))
    return papers


def parse_uploaded_file(name: str, content: bytes) -> List[Dict]:
    suffix = Path(name).suffix.lower()
    text = content.decode("utf-8-sig", errors="ignore")
    if suffix in {".json", ".jsonl"}:
        return parse_json_or_jsonl(text)
    if suffix == ".csv":
        return parse_csv(text)
    if suffix == ".bib":
        return parse_bibtex(text)
    if suffix == ".ris":
        return parse_ris(text)
    return []
