from __future__ import annotations

import os
import re
import time
from datetime import datetime, timezone
from typing import Dict, List
from urllib.parse import quote

import feedparser
import requests

from .schema import normalize_doi, safe_text

USER_AGENT = "literature-metadata-dashboard/0.5 (mailto:your_email@example.com)"
SEARCH_TIMEOUT = 40

# Lightweight in-process cache to reduce repeated API calls and rate-limit issues.
_SEMANTIC_CACHE: dict[tuple[str, str, str, int], list[dict]] = {}
_SEMANTIC_RATE_LIMIT_UNTIL = 0.0
_SEMANTIC_LAST_REQUEST_AT = 0.0


def _semantic_get(url: str, *, params: Dict, headers: Dict) -> requests.Response:
    """Call Semantic Scholar while respecting the 1 request/second key limit."""
    global _SEMANTIC_LAST_REQUEST_AT, _SEMANTIC_RATE_LIMIT_UNTIL
    elapsed = time.time() - _SEMANTIC_LAST_REQUEST_AT
    if elapsed < 1.05:
        time.sleep(1.05 - elapsed)
    response = requests.get(url, params=params, headers=headers, timeout=SEARCH_TIMEOUT)
    _SEMANTIC_LAST_REQUEST_AT = time.time()
    if response.status_code == 429:
        retry_after = response.headers.get("Retry-After")
        try:
            wait_seconds = int(retry_after) if retry_after else 300
        except ValueError:
            wait_seconds = 300
        _SEMANTIC_RATE_LIMIT_UNTIL = time.time() + max(wait_seconds, 60)
    return response



def _year_from_crossref(item: Dict) -> str:
    for key in ["published-print", "published-online", "issued", "created"]:
        try:
            return str(item[key]["date-parts"][0][0])
        except Exception:
            pass
    return ""


def _split_pages(total: int, page_size: int, max_page_size: int) -> List[tuple[int, int]]:
    total = max(0, int(total))
    page_size = min(max(1, int(page_size)), max_page_size)
    out = []
    start = 0
    while start < total:
        n = min(page_size, total - start)
        out.append((start, n))
        start += n
    return out


def search_crossref(query: str, *, title: str = "", author: str = "", journal: str = "", year_from: str = "", year_to: str = "", rows: int = 25) -> List[Dict]:
    records = []
    for offset, n in _split_pages(rows, 100, 100):
        params = {
            "rows": n,
            "offset": offset,
            "sort": "relevance",
            "order": "desc",
            "select": "DOI,title,author,abstract,container-title,issued,published-print,published-online,created,URL,subject,is-referenced-by-count",
        }
        if title:
            params["query.title"] = title
        if author:
            params["query.author"] = author
        if journal:
            params["query.container-title"] = journal
        if query:
            params["query"] = query
        filters = []
        if year_from:
            filters.append(f"from-pub-date:{year_from}-01-01")
        if year_to:
            filters.append(f"until-pub-date:{year_to}-12-31")
        if filters:
            params["filter"] = ",".join(filters)
        r = requests.get("https://api.crossref.org/works", params=params, headers={"User-Agent": USER_AGENT}, timeout=SEARCH_TIMEOUT)
        r.raise_for_status()
        items = r.json().get("message", {}).get("items", []) or []
        if not items:
            break
        for item in items:
            authors = []
            for a in item.get("author", []) or []:
                authors.append(" ".join([safe_text(a.get("given")), safe_text(a.get("family"))]).strip())
            records.append({
                "source": "crossref",
                "title": safe_text((item.get("title") or [""])[0]),
                "authors": authors,
                "abstract": safe_text(item.get("abstract")),
                "journal": safe_text((item.get("container-title") or [""])[0]),
                "year": _year_from_crossref(item),
                "doi": normalize_doi(item.get("DOI")),
                "url": safe_text(item.get("URL")),
                "keywords": item.get("subject") or [],
                "citation_count": item.get("is-referenced-by-count", ""),
            })
        time.sleep(0.15)
    return records


def search_semantic_scholar(query: str, *, year_from: str = "", year_to: str = "", rows: int = 25) -> List[Dict]:
    """Optional Semantic Scholar search.

    Semantic Scholar is useful for citations/abstract enrichment, but it can throttle
    anonymous requests. The dashboard treats it as an optional enrichment source, not
    the primary search backend. OpenAlex remains the most stable large-scale backend.
    """
    query = safe_text(query).strip()
    if not query:
        return []

    global _SEMANTIC_RATE_LIMIT_UNTIL
    now = time.time()
    if now < _SEMANTIC_RATE_LIMIT_UNTIL:
        wait = int(_SEMANTIC_RATE_LIMIT_UNTIL - now)
        raise RuntimeError(f"Semantic Scholar is temporarily cooling down after HTTP 429. Try again in about {wait} seconds or add SEMANTIC_SCHOLAR_API_KEY.")

    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    rows = min(max(int(rows), 1), 50 if api_key else 10)
    cache_key = (query.lower(), str(year_from), str(year_to), rows)
    if cache_key in _SEMANTIC_CACHE:
        return list(_SEMANTIC_CACHE[cache_key])

    records = []
    offset = 0
    while len(records) < rows:
        limit = min(25, rows - len(records))
        params = {
            "query": query,
            "limit": limit,
            "offset": offset,
            "fields": "title,abstract,authors,year,venue,url,externalIds,publicationDate,citationCount",
        }
        if year_from or year_to:
            params["year"] = f"{year_from or ''}-{year_to or ''}"
        headers = {"User-Agent": USER_AGENT}
        if api_key:
            headers["x-api-key"] = api_key

        last_exc = None
        data = []
        for attempt in range(3):
            try:
                r = _semantic_get("https://api.semanticscholar.org/graph/v1/paper/search", params=params, headers=headers)
                if r.status_code == 429:
                    auth_state = "API key detected" if api_key else "no API key detected"
                    raise RuntimeError(f"Semantic Scholar rate limit HTTP 429 ({auth_state}). Try again later and keep requests below 1 per second.")
                r.raise_for_status()
                data = r.json().get("data", []) or []
                break
            except Exception as exc:
                last_exc = exc
                if "HTTP 429" in str(exc):
                    raise
                time.sleep(1.5 * (attempt + 1))
        else:
            raise RuntimeError(f"Semantic Scholar request failed after retries: {last_exc}")

        if api_key and not data and offset == 0:
            # Fallback endpoint sometimes returns results when the paginated search endpoint does not.
            try:
                bulk_params = {
                    "query": query,
                    "limit": min(rows, 100),
                    "fields": params["fields"],
                }
                if year_from or year_to:
                    bulk_params["year"] = f"{year_from or ''}-{year_to or ''}"
                rb = _semantic_get("https://api.semanticscholar.org/graph/v1/paper/search/bulk", params=bulk_params, headers=headers)
                if rb.status_code == 200:
                    data = rb.json().get("data", []) or []
            except Exception:
                data = []
        if not data:
            break
        for item in data:
            ext = item.get("externalIds") or {}
            records.append({
                "source": "semantic_scholar",
                "title": safe_text(item.get("title")),
                "authors": [safe_text(a.get("name")) for a in item.get("authors", [])],
                "abstract": safe_text(item.get("abstract")),
                "journal": safe_text(item.get("venue")),
                "year": safe_text(item.get("year")),
                "doi": normalize_doi(ext.get("DOI")),
                "url": safe_text(item.get("url")),
                "pdf_url": "",
                "keywords": item.get("fieldsOfStudy") or [],
                "citation_count": item.get("citationCount", ""),
            })
        offset += len(data)
        time.sleep(0.8)
        if len(data) < limit:
            break

    _SEMANTIC_CACHE[cache_key] = list(records)
    return records


def search_openalex(query: str, *, title: str = "", author: str = "", journal: str = "", year_from: str = "", year_to: str = "", rows: int = 25) -> List[Dict]:
    q = " ".join(x for x in [query, title, author, journal] if x).strip()
    if not q:
        return []
    out = []
    page = 1
    while len(out) < rows:
        per_page = min(200, rows - len(out))
        filters = []
        if year_from:
            filters.append(f"from_publication_date:{year_from}-01-01")
        if year_to:
            filters.append(f"to_publication_date:{year_to}-12-31")
        params = {
            "search": q,
            "per-page": per_page,
            "page": page,
            "sort": "relevance_score:desc",
            "mailto": os.getenv("OPENALEX_MAILTO", ""),
        }
        if filters:
            params["filter"] = ",".join(filters)
        r = requests.get("https://api.openalex.org/works", params={k: v for k, v in params.items() if v}, headers={"User-Agent": USER_AGENT}, timeout=SEARCH_TIMEOUT)
        r.raise_for_status()
        results = r.json().get("results", []) or []
        if not results:
            break
        for item in results:
            authors = []
            for au in item.get("authorships", []) or []:
                author_obj = au.get("author") or {}
                if author_obj.get("display_name"):
                    authors.append(author_obj["display_name"])
            host = item.get("primary_location", {}) or {}
            src = host.get("source", {}) or {}
            oa = item.get("open_access", {}) or {}
            concepts = [c.get("display_name") for c in item.get("concepts", [])[:8] if c.get("display_name")]
            out.append({
                "source": "openalex",
                "title": safe_text(item.get("display_name")),
                "authors": authors,
                "abstract": safe_text(item.get("abstract_inverted_index")),
                "journal": safe_text(src.get("display_name")),
                "year": safe_text(item.get("publication_year")),
                "doi": normalize_doi(item.get("doi")),
                "url": safe_text(host.get("landing_page_url") or item.get("id")),
                "pdf_url": safe_text(host.get("pdf_url") or oa.get("oa_url")),
                "keywords": concepts,
                "citation_count": item.get("cited_by_count", ""),
            })
        page += 1
        time.sleep(0.15)
        if len(results) < per_page:
            break
    return out


def _scholar_year_from_text(text: str) -> str:
    m = re.search(r"\b(19|20)\d{2}\b", safe_text(text))
    return m.group(0) if m else ""


def search_google_scholar_scholarly(query: str, *, rows: int = 25) -> List[Dict]:
    """Free/local Google Scholar via scholarly. Fragile by design: Scholar can rate-limit or CAPTCHA."""
    query = safe_text(query).strip()
    if not query:
        return []
    try:
        from scholarly import scholarly  # type: ignore
    except Exception as exc:
        raise RuntimeError("Install optional dependency with: pip install scholarly") from exc

    out = []
    search = scholarly.search_pubs(query)
    for _ in range(min(rows, 300)):
        try:
            item = next(search)
        except StopIteration:
            break
        bib = item.get("bib") or {}
        pub_year = safe_text(bib.get("pub_year")) or _scholar_year_from_text(" ".join(str(v) for v in bib.values()))
        venue = safe_text(bib.get("venue")) or safe_text(bib.get("journal")) or safe_text(bib.get("publisher"))
        author_text = safe_text(bib.get("author"))
        authors = [a.strip() for a in re.split(r"\s+and\s+|;|,", author_text) if a.strip()]
        out.append({
            "source": "google_scholar_scholarly",
            "title": safe_text(bib.get("title")),
            "authors": authors,
            "abstract": safe_text(bib.get("abstract")),
            "journal": venue,
            "year": pub_year,
            "doi": "",
            "url": safe_text(item.get("pub_url")) or safe_text(item.get("eprint_url")),
            "pdf_url": safe_text(item.get("eprint_url")),
            "keywords": [],
            "citation_count": item.get("num_citations", ""),
        })
        time.sleep(0.5)
    return out


def search_serpapi_google_scholar(query: str, *, rows: int = 25) -> List[Dict]:
    key = os.getenv("SERPAPI_KEY", "")
    if not key or not query:
        return []
    out = []
    start = 0
    while len(out) < rows:
        num = min(20, rows - len(out))
        params = {"engine": "google_scholar", "q": query, "api_key": key, "num": num, "start": start}
        r = requests.get("https://serpapi.com/search.json", params=params, timeout=SEARCH_TIMEOUT)
        r.raise_for_status()
        results = r.json().get("organic_results", []) or []
        if not results:
            break
        for item in results:
            pub = item.get("publication_info", {}) or {}
            summary = safe_text(pub.get("summary"))
            out.append({
                "source": "google_scholar_serpapi",
                "title": safe_text(item.get("title")),
                "authors": [],
                "abstract": safe_text(item.get("snippet")),
                "journal": summary,
                "year": _scholar_year_from_text(summary),
                "doi": "",
                "url": safe_text(item.get("link")),
                "keywords": [],
                "citation_count": safe_text((item.get("inline_links") or {}).get("cited_by", {}).get("total")),
            })
        start += len(results)
        time.sleep(0.2)
        if len(results) < num:
            break
    return out


def _arxiv_query(parts: List[str], rows: int) -> List[Dict]:
    if not parts:
        return []
    search_query = "+AND+".join(quote(p, safe=':"()') for p in parts)
    url = f"https://export.arxiv.org/api/query?search_query={search_query}&start=0&max_results={rows}&sortBy=relevance&sortOrder=descending"
    feed = feedparser.parse(url)
    records = []
    for e in feed.entries:
        published = ""
        year = ""
        if getattr(e, "published_parsed", None):
            dt = datetime(*e.published_parsed[:6], tzinfo=timezone.utc)
            published = dt.date().isoformat()
            year = str(dt.year)
        records.append({
            "source": "arxiv",
            "title": safe_text(e.title),
            "authors": [a.name for a in getattr(e, "authors", [])],
            "abstract": safe_text(e.summary),
            "journal": "arXiv",
            "year": year,
            "doi": normalize_doi(getattr(e, "arxiv_doi", "")),
            "url": safe_text(e.link),
            "published": published,
            "keywords": [safe_text(t.get("term")) for t in getattr(e, "tags", []) if getattr(e, "tags", None)],
        })
    time.sleep(0.5)
    return records


def search_arxiv(query: str, *, title: str = "", abstract: str = "", author: str = "", rows: int = 25) -> List[Dict]:
    query = safe_text(query)
    parts = []
    if query:
        parts.append(f'all:"{query}"')
    if title:
        parts.append(f'ti:"{title}"')
    if abstract:
        parts.append(f'abs:"{abstract}"')
    if author:
        parts.append(f'au:"{author}"')
    records = _arxiv_query(parts, rows)
    if records:
        return records
    for fallback in [query, title, abstract, author]:
        fallback = safe_text(fallback)
        if fallback:
            records = _arxiv_query([f'all:"{fallback}"'], rows)
            if records:
                return records
    return []


def _unique_queries(*values: str) -> List[str]:
    seen = set()
    out = []
    for value in values:
        value = " ".join(safe_text(value).split())
        candidates = [value]
        if re.search(r"\bOR\b|\|", value, flags=re.I):
            candidates.extend(x.strip(" ()") for x in re.split(r"\bOR\b|\|", value, flags=re.I))
        for candidate in candidates:
            candidate = " ".join(safe_text(candidate).split())
            if candidate and candidate.lower() not in seen:
                out.append(candidate)
                seen.add(candidate.lower())
    return out




def _scholar_record_quality(rec: Dict, strictness: str = "medium") -> bool:
    """Local quality gate for Google Scholar records.

    Google Scholar often returns broad snippets without DOI/abstract/journal.
    This gate lets the UI choose whether to keep weak records for broad scouting
    or require stronger metadata for review-ready exports.
    """
    strictness = (strictness or "medium").lower()
    title = safe_text(rec.get("title"))
    if not title:
        return False
    if strictness == "loose":
        return True
    has_basic = bool(safe_text(rec.get("year")) or safe_text(rec.get("url")) or safe_text(rec.get("citation_count")))
    if strictness == "medium":
        return has_basic
    # strict: keep only records that can be enriched or are already metadata-rich
    return bool(safe_text(rec.get("doi")) or safe_text(rec.get("abstract")) or safe_text(rec.get("journal")))

def search_all(fields: Dict, sources: Dict[str, bool], rows_per_source: int = 25, *, semantic_rows: int | None = None, scholar_raw_limit: int | None = None, scholar_strictness: str = "medium") -> List[Dict]:
    query = fields.get("global", "")
    title = fields.get("title", "")
    abstract = fields.get("abstract", "")
    author = fields.get("author", "")
    journal = fields.get("journal", "")
    keywords = fields.get("keywords", "")
    doi = fields.get("doi", "")
    year_from = fields.get("year_from", "")
    year_to = fields.get("year_to", "")
    combined = " ".join(x for x in [query, title, abstract, author, journal, keywords, doi] if x)
    out: List[Dict] = []
    errors = []

    def add_source(name, fn):
        if not sources.get(name, False):
            return
        try:
            out.extend(fn())
        except Exception as exc:
            errors.append({"source": name, "error": str(exc)})

    add_source("crossref", lambda: search_crossref(combined or query, title=title, author=author, journal=journal, year_from=year_from, year_to=year_to, rows=rows_per_source))
    add_source("openalex", lambda: search_openalex(combined or query, title=title, author=author, journal=journal, year_from=year_from, year_to=year_to, rows=rows_per_source))

    if sources.get("semantic_scholar", False):
        try:
            records = []
            semantic_limit = min(int(semantic_rows or rows_per_source), 50)
            semantic_queries = _unique_queries(query, keywords, title, abstract, author, journal, doi, combined)
            if not os.getenv("SEMANTIC_SCHOLAR_API_KEY"):
                semantic_limit = min(semantic_limit, 10)
                semantic_queries = semantic_queries[:2]
            for q in semantic_queries:
                records = search_semantic_scholar(q, year_from=year_from, year_to=year_to, rows=semantic_limit)
                if records:
                    break
            out.extend(records)
        except Exception as exc:
            errors.append({"source": "semantic_scholar", "error": str(exc)})

    add_source("arxiv", lambda: search_arxiv(query or combined, title=title, abstract=abstract, author=author, rows=rows_per_source))

    # Google Scholar does not support the same structured field search as OpenAlex/Crossref.
    # Use the simplest/high-signal query to reduce blocking and improve hit rate.
    scholar_query = (query or title or abstract or author or journal or combined).strip()
    if sources.get("google_scholar_scholarly", False):
        try:
            gs_rows = int(scholar_raw_limit or rows_per_source)
            gs_records = search_google_scholar_scholarly(scholar_query, rows=gs_rows)
            gs_records = [r for r in gs_records if _scholar_record_quality(r, scholar_strictness)]
            out.extend(gs_records)
        except Exception as exc:
            errors.append({"source": "google_scholar_scholarly", "error": str(exc)})
    if sources.get("google_scholar_serpapi", False):
        try:
            gs_rows = int(scholar_raw_limit or rows_per_source)
            gs_records = search_serpapi_google_scholar(scholar_query, rows=gs_rows)
            gs_records = [r for r in gs_records if _scholar_record_quality(r, scholar_strictness)]
            out.extend(gs_records)
        except Exception as exc:
            errors.append({"source": "google_scholar_serpapi", "error": str(exc)})

    for rec in out:
        rec.setdefault("error", "")
    for e in errors:
        out.append({"source": e["source"], "title": "", "abstract": "", "error": e["error"]})
    return out
