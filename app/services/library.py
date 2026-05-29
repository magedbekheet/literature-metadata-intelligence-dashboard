from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from .schema import dedupe_papers, normalize_paper


def load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
            except Exception:
                continue
    return dedupe_papers(rows)


def save_jsonl(path: Path, papers: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = dedupe_papers(papers)
    with path.open("w", encoding="utf-8") as f:
        for p in clean:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


def merge_into_library(path: Path, new_papers: Iterable[Dict]) -> List[Dict]:
    old = load_jsonl(path)
    merged = dedupe_papers([*old, *new_papers])
    save_jsonl(path, merged)
    return merged


def to_dataframe(papers: List[Dict]) -> pd.DataFrame:
    rows = [normalize_paper(p) for p in papers]
    if not rows:
        return pd.DataFrame(columns=["id","title","authors_display","abstract","journal","year","doi","keywords","source","url","pdf_url","local_pdf","citation_count"])
    return pd.DataFrame(rows)
