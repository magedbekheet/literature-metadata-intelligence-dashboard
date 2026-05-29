from __future__ import annotations

import re
import os
import time
from urllib.parse import quote, unquote
from typing import Any, Dict, Iterable, List, Tuple

import requests

from .schema import normalize_doi, normalize_paper, safe_text, normalize_title

USER_AGENT = "literature-metadata-dashboard/0.6 (metadata-enrichment; mailto:your_email@example.com)"
DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.I)
SEMANTIC_FIELDS = "title,abstract,authors,year,venue,url,externalIds,publicationDate,citationCount,openAccessPdf,fieldsOfStudy,journal"
_SEMANTIC_LAST_REQUEST_AT = 0.0


def semantic_headers() -> Dict[str, str]:
    headers = {"User-Agent": USER_AGENT}
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key
    return headers


def semantic_get(url: str, *, params: Dict[str, Any]) -> requests.Response:
    """Respect Semantic Scholar's 1 request/second limit during enrichment."""
    global _SEMANTIC_LAST_REQUEST_AT
    elapsed = time.time() - _SEMANTIC_LAST_REQUEST_AT
    if elapsed < 1.05:
        time.sleep(1.05 - elapsed)
    response = requests.get(url, params=params, headers=semantic_headers(), timeout=20)
    _SEMANTIC_LAST_REQUEST_AT = time.time()
    return response


def is_missing(value: Any) -> bool:
    return not bool(safe_text(value).strip())


def extract_doi(text: str) -> str:
    text = safe_text(text)
    m = DOI_RE.search(text)
    if not m:
        return ""
    return normalize_doi(m.group(0).rstrip(".,;)]}>"))


def merge_metadata(base: Dict[str, Any], new: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    out = dict(base)
    changed: List[str] = []
    for key in ["doi", "abstract", "journal", "year", "url", "pdf_url", "citation_count"]:
        if is_missing(out.get(key)) and not is_missing(new.get(key)):
            out[key] = new.get(key)
            changed.append(key)
    if is_missing(out.get("authors_display")) and new.get("authors"):
        out["authors"] = new.get("authors")
        out["authors_display"] = ", ".join(new.get("authors") or [])
        changed.append("authors")
    if (not out.get("keywords")) and new.get("keywords"):
        out["keywords"] = new.get("keywords")
        changed.append("keywords")
    return out, changed


def _crossref_year(item: Dict[str, Any]) -> str:
    for key in ["published-print", "published-online", "issued", "created"]:
        try:
            return str(item[key]["date-parts"][0][0])
        except Exception:
            pass
    return ""



def enrich_crossref_by_doi(doi: str) -> Dict[str, Any]:
    """Direct Crossref lookup by DOI. Better than title search when DOI exists."""
    doi = normalize_doi(doi)
    if not doi:
        return {}
    r = requests.get(f"https://api.crossref.org/works/{quote(doi, safe='')}", headers={"User-Agent": USER_AGENT}, timeout=20)
    if r.status_code == 404:
        return {}
    r.raise_for_status()
    item = r.json().get("message", {}) or {}
    authors = []
    for a in item.get("author", []) or []:
        name = " ".join([safe_text(a.get("given")), safe_text(a.get("family"))]).strip()
        if name:
            authors.append(name)
    return normalize_paper({
        "source": "crossref_doi_enrichment",
        "title": safe_text((item.get("title") or [""])[0]),
        "authors": authors,
        "abstract": safe_text(item.get("abstract")),
        "journal": safe_text((item.get("container-title") or [""])[0]),
        "year": _crossref_year(item),
        "doi": normalize_doi(item.get("DOI")),
        "url": safe_text(item.get("URL")),
        "keywords": item.get("subject") or [],
        "citation_count": item.get("is-referenced-by-count", ""),
    })


def enrich_openalex_by_doi(doi: str) -> Dict[str, Any]:
    """Direct OpenAlex lookup by DOI. Often returns reconstructed abstracts."""
    doi = normalize_doi(doi)
    if not doi:
        return {}
    # OpenAlex accepts DOI URL as a work id.
    url = "https://api.openalex.org/works/" + quote("https://doi.org/" + doi, safe="")
    params = {}
    mailto = os.getenv("OPENALEX_MAILTO", "").strip()
    if mailto:
        params["mailto"] = mailto
    r = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=20)
    if r.status_code == 404:
        return {}
    r.raise_for_status()
    item = r.json() or {}
    authors = []
    for au in item.get("authorships", []) or []:
        author_obj = au.get("author") or {}
        if author_obj.get("display_name"):
            authors.append(author_obj["display_name"])
    host = item.get("primary_location", {}) or {}
    src = host.get("source", {}) or {}
    oa = item.get("open_access", {}) or {}
    concepts = [c.get("display_name") for c in item.get("concepts", [])[:8] if c.get("display_name")]
    return normalize_paper({
        "source": "openalex_doi_enrichment",
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


def enrich_unpaywall_by_doi(doi: str, email: str = "") -> Dict[str, Any]:
    """Unpaywall can provide OA URL, journal/year and sometimes title; abstracts are usually absent."""
    doi = normalize_doi(doi)
    email = (email or os.getenv("UNPAYWALL_EMAIL", "")).strip()
    if not doi or not email:
        return {}
    r = requests.get(f"https://api.unpaywall.org/v2/{quote(doi, safe='')}", params={"email": email}, headers={"User-Agent": USER_AGENT}, timeout=20)
    if r.status_code == 404:
        return {}
    r.raise_for_status()
    item = r.json() or {}
    best = item.get("best_oa_location") or {}
    return normalize_paper({
        "source": "unpaywall_enrichment",
        "title": item.get("title"),
        "doi": item.get("doi"),
        "journal": item.get("journal_name"),
        "year": item.get("year"),
        "url": item.get("url") or best.get("url"),
        "pdf_url": best.get("url_for_pdf") or best.get("url"),
        "abstract": "",
    })


def enrich_elsevier_by_doi(doi: str, api_key: str = "") -> Dict[str, Any]:
    """Optional Elsevier/ScienceDirect API lookup. Requires ELSEVIER_API_KEY.

    This is the most reliable way to enrich ScienceDirect abstracts when the
    public APIs do not provide them. It only works if the API key/account has
    access to the article metadata.
    """
    doi = normalize_doi(doi)
    api_key = (api_key or os.getenv("ELSEVIER_API_KEY", "")).strip()
    if not doi or not api_key:
        return {}
    headers = {
        "User-Agent": USER_AGENT,
        "X-ELS-APIKey": api_key,
        "Accept": "application/json",
    }
    url = "https://api.elsevier.com/content/article/doi/" + quote(doi, safe="")
    r = requests.get(url, headers=headers, timeout=25)
    if r.status_code in {401, 403, 404}:
        return {}
    r.raise_for_status()
    data = r.json() or {}
    full = data.get("full-text-retrieval-response") or {}
    core = full.get("coredata") or data.get("coredata") or {}
    abstract = safe_text(core.get("dc:description"))
    # Sometimes abstract appears in nested item/bibrecord structures; keep conservative fallback.
    if not abstract:
        abstract = safe_text(full.get("originalText"))[:2000]
    authors = []
    creator = core.get("dc:creator")
    if isinstance(creator, list):
        authors = [safe_text(x) for x in creator if safe_text(x)]
    elif creator:
        authors = [safe_text(creator)]
    return normalize_paper({
        "source": "elsevier_enrichment",
        "title": core.get("dc:title") or core.get("title"),
        "authors": authors,
        "abstract": abstract,
        "journal": core.get("prism:publicationName"),
        "year": core.get("prism:coverDate") or core.get("prism:coverDisplayDate"),
        "doi": core.get("prism:doi") or doi,
        "url": core.get("prism:url") or core.get("link"),
        "pdf_url": "",
    })

def enrich_crossref_by_title(title: str) -> Dict[str, Any]:
    title = safe_text(title)
    if not title:
        return {}
    params = {
        "query.title": title,
        "rows": 1,
        "sort": "relevance",
        "order": "desc",
        "select": "DOI,title,author,abstract,container-title,issued,published-print,published-online,created,URL,subject,is-referenced-by-count",
    }
    r = requests.get("https://api.crossref.org/works", params=params, headers={"User-Agent": USER_AGENT}, timeout=20)
    r.raise_for_status()
    items = r.json().get("message", {}).get("items", []) or []
    if not items:
        return {}
    item = items[0]
    authors = []
    for a in item.get("author", []) or []:
        name = " ".join([safe_text(a.get("given")), safe_text(a.get("family"))]).strip()
        if name:
            authors.append(name)
    return normalize_paper({
        "source": "crossref_enrichment",
        "title": safe_text((item.get("title") or [""])[0]),
        "authors": authors,
        "abstract": safe_text(item.get("abstract")),
        "journal": safe_text((item.get("container-title") or [""])[0]),
        "year": _crossref_year(item),
        "doi": normalize_doi(item.get("DOI")),
        "url": safe_text(item.get("URL")),
        "keywords": item.get("subject") or [],
        "citation_count": item.get("is-referenced-by-count", ""),
    })


def enrich_openalex_by_title(title: str) -> Dict[str, Any]:
    title = safe_text(title)
    if not title:
        return {}
    params = {"search": title, "per-page": 1, "sort": "relevance_score:desc"}
    r = requests.get("https://api.openalex.org/works", params=params, headers={"User-Agent": USER_AGENT}, timeout=20)
    r.raise_for_status()
    results = r.json().get("results", []) or []
    if not results:
        return {}
    item = results[0]
    authors = []
    for au in item.get("authorships", []) or []:
        author_obj = au.get("author") or {}
        if author_obj.get("display_name"):
            authors.append(author_obj["display_name"])
    host = item.get("primary_location", {}) or {}
    src = host.get("source", {}) or {}
    oa = item.get("open_access", {}) or {}
    concepts = [c.get("display_name") for c in item.get("concepts", [])[:8] if c.get("display_name")]
    return normalize_paper({
        "source": "openalex_enrichment",
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


def enrich_semantic_by_doi_or_title(doi: str = "", title: str = "") -> Dict[str, Any]:
    q = normalize_doi(doi) or safe_text(title)
    if not q:
        return {}
    if normalize_doi(doi):
        url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{normalize_doi(doi)}"
        params = {"fields": SEMANTIC_FIELDS}
        r = semantic_get(url, params=params)
        if r.status_code == 404:
            return {}
        if r.status_code == 429:
            raise RuntimeError("Semantic Scholar enrichment rate limit HTTP 429. Deep enrichment is optional; try fewer records later or keep Semantic Scholar for selected missing metadata only.")
        r.raise_for_status()
        item = r.json()
    else:
        params = {"query": q, "limit": 1, "fields": SEMANTIC_FIELDS}
        r = semantic_get("https://api.semanticscholar.org/graph/v1/paper/search", params=params)
        if r.status_code == 429:
            raise RuntimeError("Semantic Scholar enrichment rate limit HTTP 429. Deep enrichment is optional; try fewer records later or keep Semantic Scholar for selected missing metadata only.")
        r.raise_for_status()
        data = r.json().get("data", []) or []
        if not data:
            return {}
        item = data[0]
    ext = item.get("externalIds") or {}
    pdf = item.get("openAccessPdf") or {}
    journal_obj = item.get("journal") or {}
    return normalize_paper({
        "source": "semantic_scholar_enrichment",
        "title": safe_text(item.get("title")),
        "authors": [safe_text(a.get("name")) for a in item.get("authors", [])],
        "abstract": safe_text(item.get("abstract")),
        "journal": safe_text(journal_obj.get("name")) or safe_text(item.get("venue")),
        "year": safe_text(item.get("year")),
        "doi": normalize_doi(ext.get("DOI")),
        "url": safe_text(item.get("url")),
        "pdf_url": safe_text(pdf.get("url")) if isinstance(pdf, dict) else "",
        "keywords": item.get("fieldsOfStudy") or [],
        "citation_count": item.get("citationCount", ""),
    })


def metadata_from_landing_page(url: str) -> Dict[str, Any]:
    """Best-effort metadata extraction from publisher landing pages.

    This is intentionally conservative: many publishers block bots or hide
    metadata behind scripts/paywalls. We extract DOI from URL/meta tags/HTML
    and common citation/DC/OG meta tags when available.
    """
    url = safe_text(url)
    if not url.startswith(("http://", "https://")):
        return {}
    try:
        r = requests.get(
            url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            },
            timeout=20,
            allow_redirects=True,
        )
        if not r.ok:
            return {}
        html = r.text[:500000]
        blob = unquote(r.url) + "\n" + html
        meta: Dict[str, Any] = {"url": r.url, "doi": extract_doi(blob)}

        # Use BeautifulSoup when available; regex fallback below still catches DOI.
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            def metas(*names: str) -> str:
                for name in names:
                    tag = soup.find("meta", attrs={"name": name}) or soup.find("meta", attrs={"property": name})
                    if tag and tag.get("content"):
                        return safe_text(tag.get("content"))
                return ""
            meta["doi"] = normalize_doi(meta.get("doi") or metas("citation_doi", "dc.identifier", "DC.Identifier", "prism.doi"))
            meta["title"] = metas("citation_title", "dc.title", "DC.Title", "og:title")
            meta["journal"] = metas("citation_journal_title", "prism.publicationName", "dc.source", "DC.Source")
            meta["year"] = metas("citation_publication_date", "prism.publicationDate", "dc.date", "DC.Date")
            meta["abstract"] = metas("description", "dc.description", "DC.Description", "og:description")
            authors = []
            for tag in soup.find_all("meta", attrs={"name": "citation_author"}):
                if tag.get("content"):
                    authors.append(safe_text(tag.get("content")))
            if authors:
                meta["authors"] = authors
            pdf = metas("citation_pdf_url")
            if pdf:
                meta["pdf_url"] = pdf
        except Exception:
            pass

        return normalize_paper(meta) if any(safe_text(v) for v in meta.values()) else {}
    except Exception:
        return {}


def doi_from_landing_page(url: str) -> str:
    return normalize_doi((metadata_from_landing_page(url) or {}).get("doi"))


def enrichment_needed(paper: Dict[str, Any]) -> bool:
    return any(is_missing(paper.get(k)) for k in ["doi", "abstract", "journal", "year"])


def _looks_like_same_paper(base: Dict[str, Any], candidate: Dict[str, Any]) -> bool:
    bt = normalize_title(base.get("title"))
    ct = normalize_title(candidate.get("title"))
    if not bt or not ct:
        return True
    if bt == ct or bt in ct or ct in bt:
        return True
    try:
        from rapidfuzz import fuzz
        return fuzz.token_sort_ratio(bt, ct) >= 82
    except Exception:
        return True


def enrich_one_paper(
    paper: Dict[str, Any],
    *,
    mode: str = "balanced",
    fetch_url_doi: bool = False,
    use_elsevier: bool = True,
    unpaywall_email: str = "",
    elsevier_api_key: str = "",
    polite_delay: float = 0.15,
) -> Dict[str, Any]:
    p = normalize_paper(paper)
    changed_total: List[str] = []
    sources_used: List[str] = []

    if not enrichment_needed(p):
        p["enrichment_status"] = "complete"
        return p

    title = safe_text(p.get("title"))
    mode = (mode or "balanced").lower()
    use_semantic = mode == "deep"
    use_unpaywall = mode in {"balanced", "deep"}
    use_elsevier = bool(use_elsevier and mode == "deep")
    fetch_url_doi = bool(fetch_url_doi and mode == "deep")
    polite_delay = 0.05 if mode == "fast" else polite_delay

    # 0. Direct DOI-based enrichment when DOI is already known.
    # This is especially important for Elsevier/ScienceDirect DOIs, where title search
    # may find the record but still not return the abstract.
    if p.get("doi"):
        doi = safe_text(p.get("doi"))
        doi_sources = [
            ("crossref_doi", lambda: enrich_crossref_by_doi(doi)),
            ("openalex_doi", lambda: enrich_openalex_by_doi(doi)),
        ]
        if use_unpaywall:
            doi_sources.append(("unpaywall", lambda: enrich_unpaywall_by_doi(doi, unpaywall_email)))
        if use_semantic:
            doi_sources.append(("semantic_scholar_doi", lambda: enrich_semantic_by_doi_or_title(doi, title)))
        if use_elsevier:
            doi_sources.append(("elsevier", lambda: enrich_elsevier_by_doi(doi, elsevier_api_key)))
        for source_name, fn in doi_sources:
            if not enrichment_needed(p):
                break
            try:
                meta = fn()
                if meta and _looks_like_same_paper(p, meta):
                    p, changed = merge_metadata(p, meta)
                    if changed:
                        changed_total.extend(changed)
                        sources_used.append(source_name)
                time.sleep(polite_delay)
            except Exception:
                pass

    # 1. DOI/metadata from URL/page when available, especially useful for Google Scholar links.
    if any(is_missing(p.get(k)) for k in ["doi", "abstract", "journal", "year"]):
        doi = extract_doi(safe_text(p.get("url")) + " " + safe_text(p.get("pdf_url")))
        if doi and is_missing(p.get("doi")):
            p["doi"] = doi
            changed_total.append("doi")
            sources_used.append("url_doi")
        if fetch_url_doi and safe_text(p.get("url")):
            page_meta = metadata_from_landing_page(safe_text(p.get("url")))
            if page_meta and _looks_like_same_paper(p, page_meta):
                p, changed = merge_metadata(p, page_meta)
                if changed:
                    changed_total.extend(changed)
                    sources_used.append("landing_page")
            time.sleep(polite_delay)

    # 2. Crossref is usually best for DOI/journal/year.
    if enrichment_needed(p) and title:
        try:
            cr = enrich_crossref_by_title(title)
            p, changed = merge_metadata(p, cr)
            if changed:
                changed_total.extend(changed)
                sources_used.append("crossref")
            time.sleep(polite_delay)
        except Exception:
            pass

    # 3. OpenAlex is usually best for abstract/citations/concepts.
    if enrichment_needed(p) and title:
        try:
            oa = enrich_openalex_by_title(title)
            p, changed = merge_metadata(p, oa)
            if changed:
                changed_total.extend(changed)
                sources_used.append("openalex")
            time.sleep(polite_delay)
        except Exception:
            pass

    # 4. Semantic Scholar can fill abstract/pdf/citations when DOI or title is known.
    if use_semantic and enrichment_needed(p) and (p.get("doi") or title):
        try:
            ss = enrich_semantic_by_doi_or_title(safe_text(p.get("doi")), title)
            p, changed = merge_metadata(p, ss)
            if changed:
                changed_total.extend(changed)
                sources_used.append("semantic_scholar")
            time.sleep(polite_delay)
        except Exception:
            pass

    p["enrichment_status"] = "updated: " + ", ".join(sorted(set(changed_total))) if changed_total else "no_match"
    p["enrichment_sources"] = ", ".join(sorted(set(sources_used)))
    p["id"] = normalize_paper(p)["id"]
    return p


def enrich_papers(
    papers: Iterable[Dict[str, Any]],
    *,
    max_records: int = 50,
    mode: str = "balanced",
    fetch_url_doi: bool = False,
    use_elsevier: bool = True,
    unpaywall_email: str = "",
    elsevier_api_key: str = "",
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, paper in enumerate(papers):
        if i >= max_records:
            out.append(normalize_paper(paper))
            continue
        out.append(enrich_one_paper(
            paper,
            mode=mode,
            fetch_url_doi=fetch_url_doi,
            use_elsevier=use_elsevier,
            unpaywall_email=unpaywall_email,
            elsevier_api_key=elsevier_api_key,
        ))
    return out
