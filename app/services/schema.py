from __future__ import annotations

import hashlib
import html
import json
import re
from typing import Any, Dict, Iterable, List, Tuple

try:
    from rapidfuzz import fuzz
except Exception:  # lightweight fallback if rapidfuzz is not installed
    import difflib
    class _Fuzz:
        @staticmethod
        def token_sort_ratio(a: str, b: str) -> float:
            aa = " ".join(sorted(a.lower().split()))
            bb = " ".join(sorted(b.lower().split()))
            return difflib.SequenceMatcher(None, aa, bb).ratio() * 100
    fuzz = _Fuzz()


def clean_html_text(text: str) -> str:
    text = html.unescape(str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        try:
            import math
            if math.isnan(value):
                return ""
        except Exception:
            pass
    if isinstance(value, list):
        return ", ".join(safe_text(v) for v in value if safe_text(v))
    if isinstance(value, dict):
        # OpenAlex abstract inverted index support
        if value and all(isinstance(v, list) for v in value.values()):
            try:
                pairs = []
                for word, positions in value.items():
                    for pos in positions:
                        pairs.append((int(pos), word))
                return clean_html_text(" ".join(w for _, w in sorted(pairs)))
            except Exception:
                pass
        return json.dumps(value, ensure_ascii=False)
    return clean_html_text(str(value))


def normalize_doi(value: Any) -> str:
    text = safe_text(value).lower().strip()
    text = text.replace("https://doi.org/", "").replace("http://doi.org/", "")
    text = text.replace("doi:", "").strip()
    text = text.strip(" .;,()[]{}<>")
    m = re.search(r"10\.\d{4,9}/[-._;()/:a-z0-9]+", text, flags=re.I)
    return m.group(0).rstrip(".,;)]}>") if m else text


def normalize_title(value: Any) -> str:
    text = safe_text(value).lower()
    text = html.unescape(text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\b(the|a|an|and|or|of|for|in|on|with|to|from|by)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_authors(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        out = []
        for item in value:
            if isinstance(item, dict):
                author_obj = item.get("author") if isinstance(item.get("author"), dict) else {}
                name = (
                    safe_text(item.get("name"))
                    or safe_text(item.get("display_name"))
                    or safe_text(author_obj.get("display_name"))
                    or safe_text(author_obj.get("name"))
                    or " ".join([safe_text(item.get("given")), safe_text(item.get("family"))]).strip()
                )
            else:
                name = safe_text(item)
            if name:
                out.append(name)
        return out
    text = safe_text(value)
    if not text:
        return []
    parts = re.split(r";|\band\b|\|", text)
    return [p.strip() for p in parts if p.strip()]


def first_nonempty(*values: Any) -> str:
    for v in values:
        s = safe_text(v)
        if s:
            return s
    return ""


def pdf_url_from(value: Any) -> str:
    if isinstance(value, dict):
        return first_nonempty(value.get("url"), value.get("pdf_url"), value.get("url_for_pdf"))
    return safe_text(value)


def paper_id(paper: Dict[str, Any]) -> str:
    doi = normalize_doi(paper.get("doi"))
    if doi:
        return "doi_" + hashlib.sha1(doi.encode()).hexdigest()[:16]
    title = normalize_title(paper.get("title"))
    year = safe_text(paper.get("year"))
    return "title_" + hashlib.sha1(f"{title}|{year}".encode()).hexdigest()[:16]


def normalize_paper(raw: Dict[str, Any]) -> Dict[str, Any]:
    authors = normalize_authors(raw.get("authors") or raw.get("author") or raw.get("authorships"))
    year = safe_text(raw.get("year") or raw.get("publication_year") or raw.get("published") or raw.get("publicationDate"))
    if len(year) >= 4:
        m = re.search(r"(19|20)\d{2}", year)
        year = m.group(0) if m else year
    journal = first_nonempty(
        raw.get("journal"), raw.get("venue"), raw.get("container_title"), raw.get("container-title"),
        raw.get("source_title"), raw.get("publicationName"), raw.get("host_venue")
    )
    if journal.startswith("{"):
        try:
            obj = json.loads(journal)
            journal = first_nonempty(obj.get("display_name"), obj.get("name"), obj.get("publisher"))
        except Exception:
            pass
    kw = raw.get("keywords") if isinstance(raw.get("keywords"), list) else ([safe_text(raw.get("keyword"))] if safe_text(raw.get("keyword")) else [])
    source_raw = raw.get("source") or "imported"
    source = safe_text(source_raw)
    sources = raw.get("sources") if isinstance(raw.get("sources"), list) else ([source] if source else [])
    paper = {
        "id": "",
        "title": first_nonempty(raw.get("title"), raw.get("display_name")),
        "authors": authors,
        "authors_display": ", ".join(authors),
        "abstract": first_nonempty(raw.get("abstract"), raw.get("summary"), raw.get("abstract_inverted_index")),
        "journal": journal,
        "year": year,
        "doi": normalize_doi(raw.get("doi") or raw.get("DOI")),
        "keywords": kw,
        "source": source,
        "sources": sources,
        "url": first_nonempty(raw.get("url"), raw.get("URL"), raw.get("landing_page_url"), raw.get("id")),
        "pdf_url": first_nonempty(raw.get("pdf_url"), pdf_url_from(raw.get("openAccessPdf")), pdf_url_from(raw.get("pdf"))),
        "local_pdf": safe_text(raw.get("local_pdf")),
        "citation_count": raw.get("citation_count") or raw.get("citationCount") or raw.get("cited_by_count") or "",
        "duplicate_count": int(raw.get("duplicate_count") or 1),
        "duplicate_reason": safe_text(raw.get("duplicate_reason")),
        "raw": raw,
    }
    paper["id"] = paper_id(paper)
    return paper


def _citation_number(x: Any) -> int:
    try:
        return int(float(safe_text(x).replace(",", "")))
    except Exception:
        return 0


def _merge_lists(a: Any, b: Any) -> List[str]:
    vals = []
    for x in [a, b]:
        if isinstance(x, list):
            vals.extend([safe_text(v) for v in x if safe_text(v)])
        else:
            vals.extend([v.strip() for v in safe_text(x).split(",") if v.strip()])
    seen, out = set(), []
    for v in vals:
        key = v.lower()
        if key not in seen:
            out.append(v)
            seen.add(key)
    return out


def merge_paper_records(old: Dict[str, Any], new: Dict[str, Any], reason: str = "") -> Dict[str, Any]:
    old = normalize_paper(old)
    new = normalize_paper(new)
    merged = dict(old)
    for k, v in new.items():
        if k in {"raw", "id", "source", "sources", "duplicate_count", "duplicate_reason"}:
            continue
        if not safe_text(merged.get(k)) and safe_text(v):
            merged[k] = v
    # Prefer longest useful text fields.
    for k in ["abstract", "journal", "url", "pdf_url", "title"]:
        if len(safe_text(new.get(k))) > len(safe_text(merged.get(k))):
            merged[k] = new.get(k)
    # Merge authors, keywords, sources.
    merged["authors"] = _merge_lists(old.get("authors"), new.get("authors"))
    merged["authors_display"] = ", ".join(merged["authors"])
    merged["keywords"] = _merge_lists(old.get("keywords"), new.get("keywords"))
    merged["sources"] = _merge_lists(old.get("sources") or old.get("source"), new.get("sources") or new.get("source"))
    merged["source"] = "merged" if len(merged["sources"]) > 1 else (merged["sources"][0] if merged["sources"] else old.get("source", ""))
    # Keep max citation count.
    if _citation_number(new.get("citation_count")) > _citation_number(merged.get("citation_count")):
        merged["citation_count"] = new.get("citation_count")
    merged["duplicate_count"] = int(old.get("duplicate_count") or 1) + int(new.get("duplicate_count") or 1)
    reasons = _merge_lists(old.get("duplicate_reason"), reason or new.get("duplicate_reason"))
    merged["duplicate_reason"] = ", ".join(reasons)
    merged["id"] = paper_id(merged)
    return merged


def _same_year(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    ya, yb = safe_text(a.get("year")), safe_text(b.get("year"))
    return bool(ya and yb and ya == yb)


def _author_overlap(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    aa = {normalize_title(x) for x in normalize_authors(a.get("authors"))}
    bb = {normalize_title(x) for x in normalize_authors(b.get("authors"))}
    aa = {x for x in aa if x}
    bb = {x for x in bb if x}
    return bool(aa and bb and (aa & bb))


def are_duplicates(a: Dict[str, Any], b: Dict[str, Any]) -> Tuple[bool, str]:
    da, db = normalize_doi(a.get("doi")), normalize_doi(b.get("doi"))
    if da and db and da == db:
        return True, "doi"
    ta, tb = normalize_title(a.get("title")), normalize_title(b.get("title"))
    if not ta or not tb:
        return False, ""
    if ta == tb and (_same_year(a, b) or _author_overlap(a, b)):
        return True, "exact_title"
    score = fuzz.token_sort_ratio(ta, tb)
    if score >= 96 and (_same_year(a, b) or _author_overlap(a, b)):
        return True, f"fuzzy_title_{int(score)}"
    if score >= 92 and _same_year(a, b) and _author_overlap(a, b):
        return True, f"fuzzy_title_year_author_{int(score)}"
    return False, ""


def dedupe_papers(papers: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    masters: List[Dict[str, Any]] = []
    doi_index: Dict[str, int] = {}
    title_index: Dict[str, List[int]] = {}
    for raw in papers:
        p = normalize_paper(raw)
        doi = normalize_doi(p.get("doi"))
        if doi and doi in doi_index:
            idx = doi_index[doi]
            masters[idx] = merge_paper_records(masters[idx], p, "doi")
            continue
        title_key = normalize_title(p.get("title"))[:80]
        candidate_idxs = list(title_index.get(title_key, []))
        # Also check all masters when there are not too many; this catches close title variants.
        if len(masters) < 2000:
            candidate_idxs = list(dict.fromkeys(candidate_idxs + list(range(len(masters)))))
        matched = False
        for idx in candidate_idxs:
            ok, reason = are_duplicates(masters[idx], p)
            if ok:
                masters[idx] = merge_paper_records(masters[idx], p, reason)
                if doi:
                    doi_index[doi] = idx
                matched = True
                break
        if not matched:
            idx = len(masters)
            masters.append(p)
            if doi:
                doi_index[doi] = idx
            if title_key:
                title_index.setdefault(title_key, []).append(idx)
    return masters


def dedupe_stats(raw_count: int, deduped: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    deduped_list = list(deduped)
    merged = sum(max(int(p.get("duplicate_count") or 1) - 1, 0) for p in deduped_list)
    return {
        "raw_records": raw_count,
        "unique_papers": len(deduped_list),
        "duplicates_merged": max(raw_count - len(deduped_list), merged),
    }


def metadata_blob(paper: Dict[str, Any], fields: List[str] | None = None) -> str:
    mapping = {
        "title": paper.get("title"),
        "abstract": paper.get("abstract"),
        "authors": paper.get("authors_display") or paper.get("authors"),
        "journal": paper.get("journal"),
        "year": paper.get("year"),
        "doi": paper.get("doi"),
        "keywords": paper.get("keywords"),
        "source": paper.get("source"),
        "url": paper.get("url"),
    }
    chosen = fields or list(mapping)
    return " ".join(safe_text(mapping.get(f)) for f in chosen).lower()
