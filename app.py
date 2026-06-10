from __future__ import annotations

import base64
import html
import json
import os
import re
from collections import Counter
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from dotenv import load_dotenv

from app.services.enrichment import enrich_papers
from app.services.exporter import to_bibtex, to_json, to_markdown, to_ris
from app.services.filtering import FIELDS, filter_dataframe
from app.services.importer import parse_uploaded_file
from app.services.library import load_jsonl, merge_into_library, save_jsonl, to_dataframe
from app.services.search_sources import search_all
from app.services.schema import dedupe_papers, dedupe_stats, safe_text

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env", encoding="utf-8-sig")
for secret_name in [
    "OPENALEX_API_KEY",
    "OPENALEX_MAILTO",
    "SEMANTIC_SCHOLAR_API_KEY",
    "GROQ_API_KEY",
    "GEMINI_API_KEY",
    "SERPAPI_KEY",
    "UNPAYWALL_EMAIL",
    "ELSEVIER_API_KEY",
]:
    try:
        secret_value = safe_text(st.secrets.get(secret_name, ""))
    except Exception:
        secret_value = ""
    if secret_value and not os.getenv(secret_name):
        os.environ[secret_name] = secret_value
STORAGE = ROOT / "storage"
LIBRARY_FILE = STORAGE / "library" / "papers.jsonl"
for folder in [LIBRARY_FILE.parent, STORAGE / "exports", STORAGE / "uploads"]:
    folder.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Literature Metadata Intelligence", page_icon="📚", layout="wide")

STOPWORDS = set("""
a an and are as at be by for from has have in into is it its of on or that the their this to was were with without using use based via toward towards between under over after before synthesis study studies paper papers results result performance properties effect effects material materials preparation method methods data showed shows show high low new novel investigation analysis review applications application due through various prepared obtained compared demonstrate demonstrated significant improved improvement excellent enhanced observed reported approach work research samples sample different developed enabled related respectively also
""".split())

ANALYSIS_SCOPES = ["All papers", "By theme", "Customized theme/keywords"]

# Theme terms are transparent and editable in code. They are used only when the user
# explicitly selects "By theme" in the Analysis Scope controls.
THEME_TERMS = {
    "Battery": ["battery", "batteries", "lithium", "sodium", "anode", "cathode", "capacity", "electrochemical", "cycling"],
    "Synthesis": ["synthesis", "prepared", "fabricated", "pyrolysis", "precursor", "sol-gel", "annealing", "crosslinking", "polymerization", "hydrothermal", "autoclave"],
    "Characterization": ["xrd", "raman", "sem", "tem", "xps", "ftir", "nmr", "spectroscopy", "elemental analysis"],
    "Mechanical/Thermal": ["mechanical", "hardness", "modulus", "thermal", "oxidation", "strength"],
    "AI/Data/Computation": ["machine learning", "artificial intelligence", "deep learning", "dft", "computing", "computation", "python", "simulation"],
}

REVIEW_ARTIFACT_PHRASES = [
    "author response for",
    "decision letter for",
    "reviewer report for",
    "referee report for",
    "peer review report for",
    "response to reviewer",
    "response to reviewers",
    "interactive comment on",
]

OLLAMA_MODEL_OPTIONS = [
    "llama3.2:3b",
    "qwen2.5:7b",
    "qwen2.5:14b",
    "llama3.1:8b",
    "mistral:7b",
    "phi3:mini",
]

GROQ_MODEL_OPTIONS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-20b",
    "qwen/qwen3-32b",
]

GEMINI_MODEL_OPTIONS = [
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
]


def style() -> None:
    st.markdown(
        """
        <style>
        .main .block-container {padding-top: 1.7rem; max-width: 1450px;}
        .hero {
            padding: 1.4rem 1.6rem; border-radius: 24px;
            background: linear-gradient(135deg, #eef6ff 0%, #f7fbff 48%, #f3fff8 100%);
            border: 1px solid #dbeafe; margin-bottom: 1rem;
        }
        .hero h1 {margin: 0; font-size: 2.15rem; line-height: 1.15;}
        .hero p {margin: .45rem 0 0 0; color: #475569; font-size: 1.02rem;}
        .badge {display:inline-block; padding:.22rem .55rem; margin:.15rem .18rem .15rem 0; border-radius:999px; background:#e0f2fe; color:#075985; font-size:.78rem;}
        .brand {font-size:.82rem; color:#64748b; margin-top:.5rem;}
        .metric-card {background:white; border:1px solid #e2e8f0; border-radius:18px; padding:1rem; box-shadow:0 6px 20px rgba(15,23,42,.04);}
        .small-note {font-size:.84rem; color:#64748b;}
        .app-footer {border-top:1px solid #d8e3ea; margin-top:2.6rem; padding:1.2rem 0 1.8rem; display:flex; justify-content:space-between; align-items:center; gap:1rem; flex-wrap:wrap;}
        .app-footer .credit {color:#334155; font-size:.95rem;}
        .app-footer .links a {display:inline-block; margin-left:.5rem; padding:.42rem .75rem; border:1px solid #b7d9d6; border-radius:999px; color:#0f4f5c; text-decoration:none; font-weight:650; background:#f7fffd;}
        .app-footer .links a:hover {background:#e7fbf7;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero() -> None:
    st.markdown(
        """
        <div class="hero">
          <h1>📚 Literature Metadata Intelligence Dashboard</h1>
          <p>Search, import, clean, analyze, summarize, compare, export and share scholarly metadata for fast literature reviews.</p>
          <div style="margin-top:.65rem">
            <span class="badge">Search → Analyze → Export → Share</span>
            <span class="badge">External search data import</span>
            <span class="badge">OpenAlex + Crossref + Scholar + metadata enrichment</span>
          </div>
          <div class="brand">Developed by <b>Maged Bekheet</b> for AI-assisted literature review, materials research intelligence, and scientific metadata analysis.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def footer() -> None:
    st.markdown(
        """
        <div class="app-footer">
          <div class="credit">Developed by <b>Dr.-Ing. Maged Bekheet</b></div>
          <div class="links">
            <a href="https://github.com/magedbekheet" target="_blank" rel="noopener noreferrer">GitHub</a>
            <a href="https://de.linkedin.com/in/magedbekheet" target="_blank" rel="noopener noreferrer">LinkedIn</a>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def norm_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    for col in ["title", "abstract", "authors_display", "journal", "year", "doi", "source", "url", "pdf_url"]:
        if col not in out.columns:
            out[col] = ""
    return out.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def cached_library(path: str) -> pd.DataFrame:
    return norm_df(to_dataframe(load_jsonl(Path(path))))


def refresh_cache() -> None:
    st.cache_data.clear()


def get_library_df() -> pd.DataFrame:
    return cached_library(str(get_selected_library_file()))


def get_active_df() -> pd.DataFrame:
    df = st.session_state.get("active_df")
    if isinstance(df, pd.DataFrame) and not df.empty:
        return norm_df(df)
    return get_library_df()


def library_files() -> list[Path]:
    files = sorted(LIBRARY_FILE.parent.glob("*.jsonl"))
    if LIBRARY_FILE not in files:
        files.insert(0, LIBRARY_FILE)
    return files


def get_selected_library_file() -> Path:
    selected = st.session_state.get("library_file_select")
    if selected:
        path = LIBRARY_FILE.parent / safe_library_filename(selected)
        if path.suffix == ".jsonl":
            return path
    selected = st.session_state.get("selected_library_file")
    if selected:
        path = LIBRARY_FILE.parent / safe_library_filename(selected)
        if path.suffix == ".jsonl":
            return path
    return LIBRARY_FILE


def on_library_file_change() -> None:
    selected = st.session_state.get("library_file_select", LIBRARY_FILE.name)
    st.session_state["selected_library_file"] = selected
    st.session_state.pop("library_filtered_df", None)
    st.session_state.pop("active_df", None)


def safe_library_filename(name: str) -> str:
    name = safe_text(name).strip()
    if not name:
        return LIBRARY_FILE.name
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    if not name:
        name = "papers"
    if not name.endswith(".jsonl"):
        name += ".jsonl"
    return name


def save_to_library(df: pd.DataFrame, path: Path | None = None, merge: bool = True) -> None:
    if df.empty:
        st.warning("No records to save.")
        return
    path = path or get_selected_library_file()
    records = df.to_dict(orient="records")
    if merge:
        merged = merge_into_library(path, records)
    else:
        save_jsonl(path, records)
        merged = load_jsonl(path)
    refresh_cache()
    st.success(f"Saved to {path.name}. Library now contains {len(merged)} unique papers.")


def render_save_library_controls(df: pd.DataFrame, prefix: str) -> None:
    st.markdown("##### Save to library")
    mode = st.radio(
        "Save mode",
        ["Add to selected library", "Save as named library"],
        horizontal=True,
        key=f"{prefix}_save_mode",
    )
    if mode == "Add to selected library":
        path = get_selected_library_file()
        if st.button(f"Add to {path.name}", type="primary", key=f"{prefix}_save_selected"):
            save_to_library(df, path=path, merge=True)
    else:
        name = st.text_input("Library name", placeholder="sioc_battery_review", key=f"{prefix}_library_name")
        c1, c2 = st.columns(2)
        if c1.button("Create / replace named library", type="primary", key=f"{prefix}_save_named_replace"):
            filename = safe_library_filename(name)
            save_to_library(df, path=LIBRARY_FILE.parent / filename, merge=False)
            st.session_state["selected_library_file"] = filename
        if c2.button("Add to named library", key=f"{prefix}_save_named_merge"):
            filename = safe_library_filename(name)
            save_to_library(df, path=LIBRARY_FILE.parent / filename, merge=True)
            st.session_state["selected_library_file"] = filename


def compact_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["title", "authors_display", "journal", "year", "doi", "source", "sources", "citation_count", "url"]
    out = norm_df(df)[[c for c in cols if c in df.columns]].copy()
    for c in out.columns:
        out[c] = out[c].map(lambda x: safe_text(x)[:300])
    return out


def metadata_quality(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ["title", "abstract", "authors_display", "journal", "year", "doi", "url"]:
        if col not in df:
            continue
        missing = df[col].map(lambda x: not bool(safe_text(x))).sum()
        rows.append({"Field": col, "Available": len(df)-missing, "Missing": int(missing), "Coverage %": round((len(df)-missing)/max(len(df),1)*100, 1)})
    return pd.DataFrame(rows)


def field_coverage(df: pd.DataFrame, field: str) -> float:
    if df.empty or field not in df:
        return 0.0
    return float(df[field].map(lambda x: bool(safe_text(x))).mean() * 100)


def enrichment_recommendation(df: pd.DataFrame) -> str:
    doi_cov = field_coverage(df, "doi")
    abstract_cov = field_coverage(df, "abstract")
    journal_cov = field_coverage(df, "journal")
    source_text = " ".join(df.get("source", pd.Series(dtype=str)).fillna("").astype(str).unique()).lower() if not df.empty else ""
    likely_open_sources = any(src in source_text for src in ["openalex", "crossref"])
    likely_import_or_scholar = any(src in source_text for src in ["import", "scholar", "serpapi", "ris", "bibtex"])
    if likely_open_sources and doi_cov >= 80 and journal_cov >= 70:
        return (
            f"Enrichment is probably optional: DOI coverage is {doi_cov:.1f}%, journal coverage is {journal_cov:.1f}%, "
            "and these records appear to include OpenAlex/Crossref metadata."
        )
    if likely_import_or_scholar or doi_cov < 60 or abstract_cov < 50:
        return (
            f"Enrichment may help: DOI coverage is {doi_cov:.1f}% and abstract coverage is {abstract_cov:.1f}%. "
            "It is most useful for imported files and Google Scholar-derived records."
        )
    return (
        f"Enrichment is optional: DOI coverage is {doi_cov:.1f}% and abstract coverage is {abstract_cov:.1f}%. "
        "Run it only if important fields are missing."
    )


def enrichment_report(before: pd.DataFrame, after: pd.DataFrame, row_indices: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return field-count summary and row-level report for metadata added/updated."""
    fields = ["doi", "abstract", "journal", "year", "citation_count", "pdf_url", "url"]
    rows = []
    counts = Counter()
    before = norm_df(before).copy()
    after = norm_df(after).copy()
    for idx in row_indices:
        added, updated = [], []
        title = safe_text(after.at[idx, "title"] if "title" in after.columns and idx in after.index else "")
        for field in fields:
            if field not in after.columns:
                continue
            old = safe_text(before.at[idx, field]) if field in before.columns and idx in before.index else ""
            new = safe_text(after.at[idx, field]) if idx in after.index else ""
            if not old and new:
                added.append(field)
                counts[field] += 1
            elif old and new and old != new and field in ["citation_count", "pdf_url", "url"]:
                updated.append(field)
                counts[f"{field}_updated"] += 1
        if added or updated:
            rows.append({
                "title": title[:220] or "Untitled",
                "fields_added": ", ".join(added),
                "fields_updated": ", ".join(updated),
                "doi": safe_text(after.at[idx, "doi"]) if "doi" in after.columns and idx in after.index else "",
            })
    summary = pd.DataFrame([{
        "Field": k.replace("_updated", " updated"),
        "Filled/updated": int(v),
    } for k, v in counts.items()])
    details = pd.DataFrame(rows)
    return summary, details


def parse_terms(text: str) -> list[str]:
    terms = re.split(r"\s+OR\s+|\s+AND\s+|,|;|\n", safe_text(text), flags=re.I)
    return [t.strip().lower() for t in terms if len(t.strip()) >= 3]


def blob(row: pd.Series | dict) -> str:
    return " ".join(safe_text((row.get(c) if hasattr(row, "get") else "")) for c in ["title", "abstract", "keywords", "journal", "authors_display", "doi", "source"]).lower()


def filter_by_focus(df: pd.DataFrame, focus: str, mode: str = "OR") -> pd.DataFrame:
    terms = parse_terms(focus)
    if df.empty or not terms:
        return df
    rows = []
    for _, row in df.iterrows():
        b = blob(row)
        checks = [t in b for t in terms]
        ok = all(checks) if mode == "AND" else any(checks)
        if ok:
            rows.append(row.to_dict())
    return pd.DataFrame(rows).reset_index(drop=True) if rows else df.iloc[0:0].copy()


def term_matches(text: str, term: str) -> bool:
    """Case-insensitive term match with simple wildcard support, e.g. electrochem*."""
    t = safe_text(text).lower()
    term = safe_text(term).lower().strip()
    if not term:
        return False
    if term.endswith("*"):
        return term[:-1] in t
    return term in t


def row_match_terms(row: pd.Series | dict, terms: list[str], mode: str = "OR") -> tuple[bool, list[str]]:
    b = blob(row)
    matched = [term for term in terms if term_matches(b, term)]
    if not terms:
        return True, []
    ok = len(matched) == len(terms) if mode == "AND" else bool(matched)
    return ok, matched


def is_review_artifact(row: pd.Series | dict) -> bool:
    title = safe_text(row.get("title") if hasattr(row, "get") else "").lower()
    doi = safe_text(row.get("doi") if hasattr(row, "get") else "").lower()
    if any(phrase in title for phrase in REVIEW_ARTIFACT_PHRASES):
        return True
    if re.search(r"/v\d+/(review|decision)\d*$", doi):
        return True
    if re.search(r"/(review|decision)\d*$", doi) and any(word in title for word in ["review", "decision", "response"]):
        return True
    return False


def filter_review_artifacts(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if df.empty:
        return df, 0
    mask = norm_df(df).apply(is_review_artifact, axis=1)
    return norm_df(df)[~mask].reset_index(drop=True), int(mask.sum())


def normalize_custom_themes(items: list[dict] | None) -> list[dict]:
    """Clean user-defined theme/topic entries from session state."""
    themes = []
    for item in items or []:
        name = safe_text(item.get("theme") or item.get("topic")).strip()
        terms = parse_terms(item.get("keywords", ""))
        if name and terms:
            themes.append({"theme": name, "keywords": ", ".join(terms), "terms": terms})
    return themes


def flatten_theme_terms(themes: list[dict]) -> list[str]:
    seen = set()
    terms = []
    for theme in themes:
        for term in theme.get("terms", []):
            if term not in seen:
                terms.append(term)
                seen.add(term)
    return terms


def apply_analysis_scope(
    df: pd.DataFrame,
    scope: str,
    custom_terms: str = "",
    mode: str = "OR",
    theme_name: str = "Battery",
    theme_names: list[str] | None = None,
    custom_themes: list[dict] | None = None,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame, str, pd.DataFrame]:
    """Apply an explicit analysis scope and return matched rows + match report + label."""
    if df.empty:
        return df, [], pd.DataFrame(), scope, pd.DataFrame()

    if scope == "All papers":
        out = norm_df(df).copy()
        out["match_reason"] = "All papers"
        report = out[["title", "journal", "year", "match_reason"]].copy()
        stats = pd.DataFrame([{"theme": "All papers", "matched_papers": len(out), "matched_pct": 100.0 if len(out) else 0.0, "terms_used": 0}])
        return out, [], report, "All papers", stats

    if scope == "By theme":
        selected = theme_names or [theme_name]
        theme_sets = [{"theme": name, "terms": [t.lower() for t in THEME_TERMS.get(name, [])]} for name in selected]
        terms = flatten_theme_terms(theme_sets)
        label = "Themes: " + ", ".join(selected)
    else:
        theme_sets = normalize_custom_themes(custom_themes)
        if theme_sets:
            terms = flatten_theme_terms(theme_sets)
            label = "Customized themes: " + ", ".join(t["theme"] for t in theme_sets)
        else:
            terms = parse_terms(custom_terms)
            theme_sets = [{"theme": "Customized theme/keywords", "terms": terms}] if terms else []
            label = "Customized themes"

    if not terms:
        out = norm_df(df).copy()
        out["match_reason"] = "No custom themes entered"
        out["custom_theme"] = ""
        report = out[["title", "journal", "year", "match_reason"]].copy()
        stats = pd.DataFrame([
            {
                "theme": safe_text(theme.get("theme")) or "Custom theme",
                "matched_papers": 0,
                "matched_pct": 0.0,
                "terms_used": len(theme.get("terms", [])),
            }
            for theme in theme_sets
        ])
        return out, [], report, label, stats

    rows = []
    report_rows = []
    theme_counts = Counter()
    for _, row in norm_df(df).iterrows():
        theme_matches = []
        matched_terms = []
        reasons = []
        for theme in theme_sets:
            ok, matched = row_match_terms(row, theme.get("terms", []), mode)
            if ok:
                theme_label = safe_text(theme.get("theme")) or "Custom theme"
                theme_matches.append(theme_label)
                theme_counts[theme_label] += 1
                matched_terms.extend(matched)
                reasons.append(f"{theme_label}: {', '.join(matched)}")
        if theme_matches:
            rd = row.to_dict()
            rd["custom_theme"] = ", ".join(theme_matches)
            rd["match_reason"] = "; ".join(reasons)
            rows.append(rd)
            report_rows.append({
                "title": safe_text(row.get("title"))[:220] or "Untitled",
                "journal": safe_text(row.get("journal")),
                "year": safe_text(row.get("year")),
                "matched_theme": ", ".join(theme_matches),
                "match_reason": ", ".join(dict.fromkeys(matched_terms)),
            })
    matched_df = pd.DataFrame(rows).reset_index(drop=True) if rows else norm_df(df).iloc[0:0].copy()
    report = pd.DataFrame(report_rows)
    total = max(len(df), 1)
    stats = pd.DataFrame([
        {
            "theme": safe_text(theme.get("theme")) or "Custom theme",
            "matched_papers": int(theme_counts.get(safe_text(theme.get("theme")) or "Custom theme", 0)),
            "matched_pct": round(int(theme_counts.get(safe_text(theme.get("theme")) or "Custom theme", 0)) / total * 100, 1),
            "terms_used": len(theme.get("terms", [])),
        }
        for theme in theme_sets
    ])
    return matched_df, terms, report, label, stats


def render_custom_theme_builder(prefix: str) -> tuple[list[dict], str]:
    state_key = "shared_custom_themes"
    st.session_state.setdefault(state_key, [])

    st.markdown("##### Customized theme/keywords")
    st.caption("Saved custom theme/keyword sets are shared between Insights and Analytics.")
    c1, c2, c3 = st.columns([1, 2, .7])
    topic = c1.text_input(
        "Theme/topic",
        placeholder="Battery performance",
        key=f"{prefix}_new_theme_topic",
    )
    keywords = c2.text_input(
        "Corresponding keywords",
        placeholder="capacity, cycling, retention, rate capability",
        key=f"{prefix}_new_theme_keywords",
    )
    if c3.button("+ Add", key=f"{prefix}_add_theme", type="primary"):
        terms = parse_terms(keywords)
        if safe_text(topic) and terms:
            st.session_state[state_key].append({"theme": safe_text(topic), "keywords": ", ".join(terms)})
            st.rerun()
        else:
            st.warning("Add both a theme/topic and at least one keyword.")

    themes = normalize_custom_themes(st.session_state.get(state_key, []))
    raw_themes = st.session_state.get(state_key, [])
    if raw_themes:
        st.caption("Customized theme/keywords included in this analysis:")
        for i, item in enumerate(list(raw_themes)):
            r1, r2, r3, r4 = st.columns([1, 2.2, .55, .7])
            edited_theme = r1.text_input(
                "Theme/topic",
                value=safe_text(item.get("theme")),
                key=f"{prefix}_edit_theme_{i}",
                label_visibility="collapsed",
            )
            edited_keywords = r2.text_input(
                "Corresponding keywords",
                value=safe_text(item.get("keywords")),
                key=f"{prefix}_edit_keywords_{i}",
                label_visibility="collapsed",
            )
            if r3.button("Save", key=f"{prefix}_save_theme_{i}"):
                terms = parse_terms(edited_keywords)
                if safe_text(edited_theme) and terms:
                    st.session_state[state_key][i] = {"theme": safe_text(edited_theme), "keywords": ", ".join(terms)}
                    st.rerun()
                else:
                    st.warning("Theme/topic and keywords are required before saving.")
            if r4.button("Remove", key=f"{prefix}_remove_theme_{i}"):
                st.session_state[state_key].pop(i)
                st.rerun()
    else:
        st.info("Add one or more customized theme/keyword sets. Each theme has its own keyword list.")

    themes = normalize_custom_themes(st.session_state.get(state_key, []))
    flattened = ", ".join(flatten_theme_terms(themes))
    return themes, flattened


def render_scope_selector(prefix: str, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], pd.DataFrame, str, str, str]:
    st.markdown("##### Analysis scope")
    st.caption("Choose whether the review uses all active papers, a transparent theme keyword list, or your own customized theme/keyword sets.")

    scope = st.radio("Scope", ANALYSIS_SCOPES, index=0, key=f"{prefix}_scope", horizontal=True)

    mode = st.radio(
        "Keyword matching",
        ["OR", "AND"],
        horizontal=True,
        index=0,
        key=f"{prefix}_match_mode",
        help="OR keeps papers matching any keyword. AND keeps only papers matching all keywords.",
        disabled=(scope == "All papers"),
    )

    theme_name = "Battery"
    theme_names: list[str] = ["Battery"]
    custom_terms = ""
    custom_themes: list[dict] = []
    c1, c2 = st.columns([1, 1.4])
    if scope == "By theme":
        theme_names = c1.multiselect("Themes", list(THEME_TERMS.keys()), default=["Battery"], key=f"{prefix}_themes")
        if not theme_names:
            theme_names = ["Battery"]
            c1.warning("Select at least one theme. Battery is used as the fallback.")
        custom_terms = ", ".join(flatten_theme_terms([
            {"theme": name, "terms": [t.lower() for t in THEME_TERMS.get(name, [])]}
            for name in theme_names
        ]))
        c2.info("Theme terms used: " + custom_terms)
    elif scope == "Customized theme/keywords":
        custom_themes, custom_terms = render_custom_theme_builder(prefix)
    else:
        c1.info("All active papers will be used. No keyword filtering is applied.")

    scoped_df, terms, report, scope_label, theme_stats = apply_analysis_scope(df, scope, custom_terms, mode, theme_name, theme_names, custom_themes)
    exclude_artifacts = st.checkbox(
        "Exclude decision letters, author responses, and peer-review artifacts",
        value=True,
        key=f"{prefix}_exclude_review_artifacts",
        help="Keeps these records in the Library but removes them from Insights/Analytics calculations by default.",
    )
    excluded_count = 0
    if exclude_artifacts:
        scoped_df, excluded_count = filter_review_artifacts(scoped_df)
        if not report.empty:
            report, _ = filter_review_artifacts(report)
    total = max(len(df), 1)
    pct = round(len(scoped_df) / total * 100, 1)
    m1, m2, m3 = st.columns(3)
    m1.metric("Matched papers", f"{len(scoped_df)} / {len(df)}")
    m2.metric("Matched %", f"{pct}%")
    m3.metric("Terms used", len(terms))

    if terms:
        if custom_themes:
            st.caption("Customized theme/keywords: " + "; ".join(f"{t['theme']} ({t['keywords']})" for t in custom_themes))
        else:
            st.caption("Terms used: " + ", ".join(terms))
    elif scope == "All papers":
        st.caption("Scope uses all active papers. No keyword filter is applied.")
    else:
        st.warning("No custom themes entered, so all active papers are used. Add a theme/topic and keywords to filter the review.")

    if excluded_count:
        st.caption(f"Excluded {excluded_count} review artifact record(s) from this analysis scope.")
    if not theme_stats.empty and scope != "All papers":
        with st.expander("Theme/topic match counts", expanded=True):
            stat_cols = st.columns(min(3, len(theme_stats)))
            for i, (_, row) in enumerate(theme_stats.iterrows()):
                col = stat_cols[i % len(stat_cols)]
                with col:
                    st.markdown(f"**{safe_text(row.get('theme'))}**")
                    st.metric("Matched papers", f"{int(row.get('matched_papers', 0))} / {len(df)}")
                    st.metric("Matched %", f"{row.get('matched_pct', 0)}%")
                    st.metric("Terms used", int(row.get("terms_used", 0)))
    if scoped_df.empty:
        st.warning("No papers matched this scope. Try OR matching, fewer terms, another theme, customized theme/keywords, include review artifacts, or All papers.")
    elif terms:
        with st.expander("Show matched papers and match reasons", expanded=False):
            display_report = report.copy()
            if not theme_stats.empty:
                theme_values = theme_stats["theme"].dropna().astype(str).tolist()
                labels = ["All themes"] + [
                    f"{theme} ({int(theme_stats.loc[theme_stats['theme'] == theme, 'matched_papers'].iloc[0])})"
                    for theme in theme_values
                ]
                selected_theme = st.selectbox(
                    "Theme/topic",
                    labels,
                    key=f"{prefix}_matched_theme_filter",
                )
                if selected_theme != "All themes":
                    selected_theme = selected_theme.rsplit(" (", 1)[0]
                    display_report = display_report[
                        display_report["matched_theme"].astype(str).map(
                            lambda value: selected_theme in [x.strip() for x in value.split(",")]
                        )
                    ]
                    if display_report.empty:
                        st.info("No papers matched this theme/topic.")
            st.dataframe(display_report, width="stretch", hide_index=True, height=280)
            st.download_button("Download matched papers report", display_report.to_csv(index=False), f"{prefix}_matched_papers.csv", "text/csv", key=f"{prefix}_matched_download")
    return scoped_df, terms, report, scope_label, custom_terms, mode

def clean_terms(text: str, n: int = 18) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", safe_text(text).lower())
    counts = Counter(w for w in words if w not in STOPWORDS and len(w) > 2)
    return [w for w, _ in counts.most_common(n)]


def classify_theme(text: str) -> str:
    t = safe_text(text).lower()
    scores = {k: sum(term in t for term in v) for k, v in THEME_TERMS.items()}
    best, score = max(scores.items(), key=lambda x: x[1])
    return best if score else "General"


def summarize_abstract(abstract: str, focus: str = "", max_sentences: int = 2) -> str:
    text = re.sub(r"\s+", " ", safe_text(abstract)).strip()
    if not text:
        return "No abstract available."
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 25]
    if not sentences:
        return text[:650] + ("..." if len(text) > 650 else "")
    terms = parse_terms(focus)
    if terms:
        sentences = sorted(sentences, key=lambda s: sum(t in s.lower() for t in terms), reverse=True)
    result = " ".join(sentences[:max_sentences])
    return result[:900] + ("..." if len(result) > 900 else "")


RANKING_OPTIONS = [
    "Relevance to theme/keywords",
    "Hybrid relevance + citations + recency",
    "Citation count",
    "Newest first",
    "Metadata completeness",
    "Current order",
]


def numeric_value(value) -> float:
    text = safe_text(value).replace(",", "")
    try:
        return float(text)
    except Exception:
        return 0.0


def metadata_completeness_score(row: pd.Series | dict) -> float:
    fields = ["title", "abstract", "authors_display", "journal", "year", "doi", "url"]
    filled = sum(1 for field in fields if safe_text(row.get(field) if hasattr(row, "get") else ""))
    return filled / len(fields)


def relevance_score(row: pd.Series | dict, terms: list[str]) -> float:
    if not terms:
        return 0.0
    text = blob(row)
    score = 0.0
    for term in terms:
        term = safe_text(term).lower().strip()
        if not term:
            continue
        if term.endswith("*"):
            score += text.count(term[:-1])
        else:
            score += text.count(term)
    return score


def relevance_percentage(row: pd.Series | dict, terms: list[str]) -> float:
    if not terms:
        return 0.0
    matched = [term for term in terms if term_matches(blob(row), term)]
    return len(set(matched)) / len(set(terms)) * 100


def normalize_series(series: pd.Series) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce").fillna(0)
    max_value = series.max()
    if max_value <= 0:
        return pd.Series(0.0, index=series.index)
    return series / max_value


def rank_papers(df: pd.DataFrame, terms: list[str], ranking: str) -> pd.DataFrame:
    ranked = norm_df(df).copy()
    if ranked.empty:
        return ranked
    ranked["_original_order"] = range(len(ranked))
    ranked["_relevance_score"] = ranked.apply(lambda row: relevance_score(row, terms), axis=1)
    ranked["_relevance_pct"] = ranked.apply(lambda row: relevance_percentage(row, terms), axis=1)
    ranked["_citation_score"] = ranked["citation_count"].map(numeric_value) if "citation_count" in ranked else 0.0
    ranked["_year_score"] = pd.to_numeric(ranked["year"], errors="coerce").fillna(0) if "year" in ranked else 0.0
    ranked["_metadata_score"] = ranked.apply(metadata_completeness_score, axis=1)

    if ranking == "Relevance to theme/keywords":
        ranked["_rank_score"] = ranked["_relevance_score"]
        sort_cols = ["_rank_score", "_citation_score", "_year_score", "_original_order"]
        ascending = [False, False, False, True]
    elif ranking == "Hybrid relevance + citations + recency":
        ranked["_rank_score"] = (
            normalize_series(ranked["_relevance_score"]) * 0.45
            + normalize_series(ranked["_citation_score"]) * 0.30
            + normalize_series(ranked["_year_score"]) * 0.15
            + ranked["_metadata_score"] * 0.10
        )
        sort_cols = ["_rank_score", "_original_order"]
        ascending = [False, True]
    elif ranking == "Citation count":
        ranked["_rank_score"] = ranked["_citation_score"]
        sort_cols = ["_rank_score", "_year_score", "_original_order"]
        ascending = [False, False, True]
    elif ranking == "Newest first":
        ranked["_rank_score"] = ranked["_year_score"]
        sort_cols = ["_rank_score", "_citation_score", "_original_order"]
        ascending = [False, False, True]
    elif ranking == "Metadata completeness":
        ranked["_rank_score"] = ranked["_metadata_score"]
        sort_cols = ["_rank_score", "_relevance_score", "_original_order"]
        ascending = [False, False, True]
    else:
        ranked["_rank_score"] = 0.0
        sort_cols = ["_original_order"]
        ascending = [True]

    return ranked.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)


def build_summary_df(df: pd.DataFrame, focus: str, n: int) -> pd.DataFrame:
    rows = []
    for citation_number, (_, row) in enumerate(norm_df(df).head(n).iterrows(), 1):
        text = f"{row.get('title','')} {row.get('abstract','')}"
        theme = safe_text(row.get("custom_theme")) or classify_theme(text)
        rows.append({
            "citation": f"[{citation_number}]",
            "title": safe_text(row.get("title")) or "Untitled",
            "year": safe_text(row.get("year")),
            "journal": safe_text(row.get("journal")) or "Unknown journal",
            "authors": safe_text(row.get("authors_display"))[:140],
            "doi": safe_text(row.get("doi")),
            "source": safe_text(row.get("source")) or "Unknown",
            "theme": theme,
            "quick_summary": summarize_abstract(row.get("abstract"), focus),
            "key_terms": ", ".join(clean_terms(text, 8)),
            "match_reason": safe_text(row.get("match_reason")),
            "rank_score": round(numeric_value(row.get("_rank_score")), 3),
            "relevance_score": round(numeric_value(row.get("_relevance_score")), 3),
            "relevance_pct": round(numeric_value(row.get("_relevance_pct")), 1),
            "citation_count": safe_text(row.get("citation_count")),
            "record_type": "Review artifact" if is_review_artifact(row) else "Research record",
        })
    return pd.DataFrame(rows)


def safe_slider(label: str, max_value: int, default: int, key: str) -> int:
    max_value = int(max_value)
    if max_value <= 1:
        st.caption(f"{label}: using 1 paper")
        return 1
    return st.slider(label, 1, max_value, min(default, max_value), key=key)


def source_badges(df: pd.DataFrame) -> None:
    if df.empty:
        return
    label_counts = Counter(df["source"].dropna().astype(str)) if "source" in df else Counter()
    contributing = Counter()
    if "sources" in df:
        for value in df["sources"]:
            if isinstance(value, list):
                parts = value
            else:
                parts = [x.strip() for x in safe_text(value).split(",")]
            for part in parts:
                if part:
                    contributing[part] += 1
    if not contributing:
        contributing = label_counts
    if label_counts:
        label_html = " ".join([f'<span class="badge">{html.escape(k)}: {v}</span>' for k, v in label_counts.items()])
        st.caption("Displayed source labels")
        st.markdown(label_html, unsafe_allow_html=True)
    if contributing:
        contrib_html = " ".join([f'<span class="badge">{html.escape(k)}: {v}</span>' for k, v in contributing.items()])
        st.caption("Contributing sources after deduplication")
        st.markdown(contrib_html, unsafe_allow_html=True)


def source_contribution_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if df.empty:
        return pd.DataFrame(columns=["Source", "Papers"])
    for _, row in df.iterrows():
        raw_sources = row.get("sources")
        if isinstance(raw_sources, list):
            sources = raw_sources
        else:
            sources = [x.strip() for x in safe_text(raw_sources).split(",") if x.strip()]
        if not sources:
            sources = [safe_text(row.get("source")) or "Unknown"]
        for source in sources:
            rows.append({"Source": safe_text(source) or "Unknown", "paper_id": safe_text(row.get("id")) or str(row.name)})
    exploded = pd.DataFrame(rows)
    if exploded.empty:
        return pd.DataFrame(columns=["Source", "Papers"])
    out = exploded.groupby("Source")["paper_id"].nunique().reset_index(name="Papers")
    return out.sort_values("Papers", ascending=False)


def render_enrichment(df: pd.DataFrame, session_key: str | None = None, save: bool = False, context: str = "library") -> pd.DataFrame:
    if df.empty:
        return df
    title = "✨ Enrich missing metadata"
    with st.expander(title, expanded=False):
        st.caption("Fill missing DOI, abstract, journal, year, citation count and PDF links for saved library records using Crossref, OpenAlex, Unpaywall, optional Elsevier, and limited publisher-page checks.")
        if context == "library":
            st.info(enrichment_recommendation(df))
            st.warning("Enrichment can take several minutes for many records. It is optional; skip it when OpenAlex/Crossref search results already have good metadata coverage.")
        mode_help = {
            "Fast": "Crossref + OpenAlex only. Best first pass.",
            "Balanced": "Crossref + OpenAlex + Unpaywall. Good metadata/PDF balance.",
            "Deep": "All sources, including Semantic Scholar. Best for selected missing metadata; slowest because Semantic Scholar is limited to about 1 request/second.",
        }
        c0, c1, c2, c3 = st.columns([1.1, .9, 1, 1])
        enrich_mode = c0.selectbox("Enrichment mode", ["Fast", "Balanced", "Deep"], index=0, key=f"{session_key}_enrich_mode")
        c0.caption(mode_help[enrich_mode])
        if enrich_mode == "Deep":
            st.caption("Recommendation: use OpenAlex + Crossref for main search, then use Deep enrichment only for selected records that still need abstracts, citation counts, or PDF links. Keep the batch modest to avoid Semantic Scholar rate limits.")
        default_n = min(20 if enrich_mode == "Fast" else 50, len(df))
        max_records = c1.number_input("Records to enrich", 1, min(500, len(df)), default_n, key=f"{session_key}_enrich_n")
        only_missing = c2.checkbox("Only rows with missing metadata", True, key=f"{session_key}_only_missing")
        fetch_url = c3.checkbox("Try publisher URL", False, key=f"{session_key}_fetch_url", help="Deep mode only. Slower and may fail on paywalls/JavaScript pages.", disabled=(enrich_mode != "Deep"))
        c4, c5 = st.columns(2)
        unpaywall_email = c4.text_input("Unpaywall email", value=os.getenv("UNPAYWALL_EMAIL", ""), key=f"{session_key}_unpaywall")
        elsevier_key = c5.text_input("Elsevier API key", value=os.getenv("ELSEVIER_API_KEY", ""), type="password", key=f"{session_key}_elsevier")
        if st.button("Enrich now", type="primary", key=f"{session_key}_enrich_btn"):
            before = norm_df(df).copy()
            target = before.copy()
            if only_missing:
                mask = pd.Series(False, index=target.index)
                for col in ["doi", "abstract", "journal", "year", "citation_count", "pdf_url"]:
                    if col not in target.columns:
                        target[col] = ""
                    mask = mask | target[col].map(lambda x: not bool(safe_text(x)))
                work = target[mask].head(int(max_records))
            else:
                work = target.head(int(max_records))
            if work.empty:
                st.info("No missing metadata found in selected rows.")
                return df
            with st.spinner("Enriching metadata..."):
                enriched = enrich_papers(
                    work.to_dict(orient="records"),
                    max_records=int(max_records),
                    mode=enrich_mode.lower(),
                    fetch_url_doi=fetch_url,
                    use_elsevier=bool(elsevier_key),
                    unpaywall_email=unpaywall_email,
                    elsevier_api_key=elsevier_key,
                )
            enriched_df = pd.DataFrame(enriched)
            changed_indices = []
            for idx, (_, row) in zip(work.index, enriched_df.iterrows()):
                changed_indices.append(idx)
                for col, val in row.items():
                    if col not in target.columns:
                        target[col] = ""
                    target.at[idx, col] = val
            summary, details = enrichment_report(before, target, changed_indices)
            if session_key:
                st.session_state[session_key] = target
                st.session_state[f"{session_key}_enrichment_summary"] = summary
                st.session_state[f"{session_key}_enrichment_details"] = details
            if save:
                save_to_library(target, path=get_selected_library_file(), merge=True)
            st.success("Metadata enrichment completed.")
            m1, m2, m3 = st.columns(3)
            m1.metric("Records processed", len(work))
            m2.metric("Records improved", len(details))
            m3.metric("Records unchanged", max(len(work) - len(details), 0))
            if not summary.empty:
                st.markdown("##### Fields filled / updated")
                st.dataframe(summary, width="stretch", hide_index=True)
            else:
                st.info("No new fields were filled. This can happen when external sources do not expose more metadata for these papers.")
            if not details.empty:
                st.markdown("##### Enriched records")
                st.dataframe(details, width="stretch", hide_index=True, height=260)
                st.download_button(
                    "Download enrichment report CSV",
                    details.to_csv(index=False),
                    "metadata_enrichment_report.csv",
                    "text/csv",
                    key=f"{session_key}_enrichment_report_download",
                )
            st.markdown("##### Metadata coverage after enrichment")
            st.dataframe(metadata_quality(target), width="stretch", hide_index=True)
            return target
    # Show latest report after rerun, if available.
    if session_key and f"{session_key}_enrichment_details" in st.session_state:
        details = st.session_state.get(f"{session_key}_enrichment_details")
        if isinstance(details, pd.DataFrame) and not details.empty:
            with st.expander("Latest enrichment report", expanded=False):
                st.dataframe(details, width="stretch", hide_index=True, height=240)
                st.download_button("Download latest enrichment report", details.to_csv(index=False), "metadata_enrichment_report.csv", "text/csv", key=f"{session_key}_latest_enrich_download")
    return df


def render_search() -> None:
    st.subheader("🔎 Search papers")
    st.caption("Clean default mode. Advanced source controls are hidden below for power users.")
    with st.form("search_form"):
        q = st.text_input("Search topic",
                          placeholder="SiOC, polymer-derived ceramics, SiOCN",
                          help=( "Search multiple keywords or phrases at once. " "Separate them with commas, semicolons, or OR. " "Examples: keyword 1, keyword 2, keyword 3 | " "keyword 1; keyword 2; keyword 3 | " "keyword 1 OR keyword 2 OR keyword 3"
                          )
        )
        cols = st.columns([1, 1, 1, 1])
        year_from = cols[0].text_input("Year from", placeholder="1990")
        year_to = cols[1].text_input("Year to", placeholder="2026")
        rows = cols[2].number_input("Results/source", 10, 1000, 250, 10, help="Increase this number to retrieve more results from each source. Maximum: 1000 results per source due to API limits.")

        source_preset = cols[3].selectbox(
            "Search mode",
            ["Balanced", "Fast", "Deep"],
            index=0,
            help="Balanced: OpenAlex + Crossref. Fast: OpenAlex only. Deep: optional slower sources enabled in Advanced controls.",
        )
        with st.expander("Advanced field search and sources", expanded=False):
            logic = st.radio("Keyword logic", ["AND", "OR"], horizontal=True, index=1, help="OR keeps papers matching any keyword. AND keeps only papers matching all keywords.")
            c1, c2 = st.columns(2)
            title_q = c1.text_input("Title contains")
            abstract_q = c1.text_input("Abstract contains")
            author_q = c1.text_input("Author contains", placeholder="John, Smith", help="To search for a specific author, select AND above, then enter the author’s first and last name separated by a comma.")
            journal_q = c2.text_input("Journal contains")
            keyword_q = c2.text_input("Topic / keywords contain")
            doi_q = c2.text_input("DOI contains")
            s1, s2, s3, s4 = st.columns(4)
            use_openalex = s1.checkbox("OpenAlex", True)
            use_crossref = s2.checkbox("Crossref", source_preset in {"Balanced", "Deep"})
            use_semantic = s3.checkbox("Semantic Scholar", False, help="Optional and often slower/rate-limited.")
            use_arxiv = s4.checkbox("arXiv", False)
            semantic_limit = st.number_input(
                "Semantic Scholar limit",
                5,
                50,
                10,
                5,
                help="Recommended: 10 for normal searches, 20-25 for broader checks. 50 is the maximum and may be slow or trigger rate limits because Semantic Scholar allows about 1 request/second.",
            )
            if use_semantic:
                if os.getenv("SEMANTIC_SCHOLAR_API_KEY"):
                    st.caption("Tip: Semantic Scholar is using SEMANTIC_SCHOLAR_API_KEY from .env. Keep 10 as the normal setting; use 20-25 for broader checks. 50 is only a maximum and may be slow due to the 1 request/second limit.")
                else:
                    st.caption("Tip: no SEMANTIC_SCHOLAR_API_KEY was found. The app caps Semantic Scholar to 10 records and pauses after HTTP 429. Add a free API key for better reliability.")
            g1, g2, g3 = st.columns(3)
            use_gs = g1.checkbox("Google Scholar local (experimental)", False, help="Free but fragile. Google may block, rate-limit, CAPTCHA-check automated requests, or return no results.")
            use_serp = g2.checkbox("Google Scholar SerpAPI (stable, requires key)", False, help="Requires SERPAPI_KEY in .env.")
            scholar_limit = g3.number_input("Scholar raw limit", 10, 300, 50, 10)
            strictness = st.selectbox("Scholar strictness", ["loose", "medium", "strict"], index=1)
            if use_gs:
                st.warning("Google Scholar local is experimental and may return no results because Google Scholar often blocks, rate-limits, or CAPTCHA-checks automated requests. Use SerpAPI for stable Scholar results.")
        submitted = st.form_submit_button("Search", type="primary")
    if submitted:
        if not any([q, title_q, abstract_q, author_q, journal_q, keyword_q, doi_q]):
            st.warning("Enter at least one search term.")
            return
        fields = {"global": q, "title": title_q, "abstract": abstract_q, "author": author_q, "journal": journal_q, "keywords": keyword_q, "doi": doi_q, "year_from": year_from, "year_to": year_to}
        sources = {"openalex": use_openalex, "crossref": use_crossref, "semantic_scholar": use_semantic, "arxiv": use_arxiv, "google_scholar_scholarly": use_gs, "google_scholar_serpapi": use_serp}
        selected_source_names = [
            name for name, enabled in {
                "OpenAlex": use_openalex,
                "Crossref": use_crossref,
                "Semantic Scholar": use_semantic,
                "arXiv": use_arxiv,
                "Google Scholar local": use_gs,
                "Google Scholar SerpAPI": use_serp,
            }.items() if enabled
        ]
        source_limits = []
        for name, enabled, limit in [
            ("OpenAlex", use_openalex, int(rows)),
            ("Crossref", use_crossref, int(rows)),
            ("Semantic Scholar", use_semantic, int(semantic_limit)),
            ("arXiv", use_arxiv, int(rows)),
            ("Google Scholar local", use_gs, int(scholar_limit)),
            ("Google Scholar SerpAPI", use_serp, int(scholar_limit)),
        ]:
            if enabled:
                source_limits.append(f"{name}: up to {limit}")
        st.info(f"Searching {', '.join(source_limits) if source_limits else 'no sources'}.")
        with st.spinner("Searching selected sources, cleaning records, and merging duplicates..."):
            import time
            t0 = time.perf_counter()
            raw = search_all(fields, sources, rows_per_source=int(rows), semantic_rows=int(semantic_limit), scholar_raw_limit=int(scholar_limit), scholar_strictness=strictness)
            search_seconds = round(time.perf_counter() - t0, 1)
            errors = [r for r in raw if safe_text(r.get("error"))]
            records = [r for r in raw if not safe_text(r.get("error"))]
            deduped = dedupe_papers(records)
            df = to_dataframe(deduped)
            field_queries = {"title": title_q, "abstract": abstract_q, "authors": author_q, "journal": journal_q, "doi": doi_q, "keywords": keyword_q}
            df = filter_dataframe(norm_df(df), global_query="", global_fields=None, field_queries=field_queries, logic=logic, year_from=year_from, year_to=year_to)
            st.session_state["active_df"] = df
            st.session_state["search_errors"] = errors
            st.session_state["dedupe_stats"] = dedupe_stats(len(records), deduped)
            st.session_state["search_raw_source_counts"] = dict(Counter(safe_text(r.get("source")) for r in records if safe_text(r.get("source"))))
            st.session_state["search_display_source_counts"] = dict(Counter(df["source"].dropna().astype(str))) if "source" in df else {}
            st.session_state["search_runtime_seconds"] = search_seconds
    search_errors = st.session_state.get("search_errors", [])
    if search_errors:
        with st.expander("Advanced source messages", expanded=True):
            st.warning("Some optional sources were unavailable, rate-limited, or returned an API error.")
            st.dataframe(pd.DataFrame(search_errors), width="stretch", hide_index=True)

    df = st.session_state.get("active_df")
    if isinstance(df, pd.DataFrame) and not df.empty:
        stats = st.session_state.get("dedupe_stats", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Papers shown", len(df))
        c2.metric("Unique before filters", stats.get("unique_papers", len(df)))
        c3.metric("Duplicates merged", stats.get("duplicates_merged", 0))
        c4.metric("Raw records", stats.get("raw_records", len(df)))
        if "search_runtime_seconds" in st.session_state:
            st.caption(f"Latest search runtime: {st.session_state['search_runtime_seconds']} seconds")
        source_badges(df)
        with st.expander("Search count details", expanded=False):
            st.write("Raw fetched records by source, before deduplication and filters:")
            raw_counts = st.session_state.get("search_raw_source_counts", {})
            st.dataframe(pd.DataFrame([{"source": k, "raw_records": v} for k, v in raw_counts.items()]), width="stretch", hide_index=True)
            st.write("Displayed records are after deduplication and your field/year filters. Papers found by multiple sources are labeled `merged`, while the contributing-source badges show which sources were merged into those papers.")
        st.dataframe(compact_table(df), width="stretch", hide_index=True, height=440)
        render_save_library_controls(df, "search")
        c2, c3 = st.columns(2)
        c2.download_button("CSV", df.to_csv(index=False), "search_results.csv", "text/csv")
        c3.download_button("JSON", to_json(df), "search_results.json", "application/json")
        if not search_errors:
            with st.expander("Advanced source messages", expanded=False):
                st.success("No source errors in the latest search.")


def render_import() -> None:
    st.subheader("📥 Import external search data")
    st.caption("Upload JSONL/JSON/BibTeX/RIS/CSV from external databases such as Zotero, Scopus, Web of Science, ScienceDirect, Mendeley, or from my companion automation project ai-literature-feed-automation: https://github.com/magedbekheet/ai-literature-feed-automation")
    files = st.file_uploader("Upload literature files", type=["json", "jsonl", "bib", "ris", "csv"], accept_multiple_files=True)
    if files:
        papers = []
        for f in files:
            try:
                papers.extend(parse_uploaded_file(f.name, f.read()))
            except Exception as exc:
                st.error(f"Could not parse {f.name}: {exc}")
        if papers:
            deduped = dedupe_papers(papers)
            df = to_dataframe(deduped)
            st.session_state["active_df"] = df
            st.success(f"Imported {len(papers)} raw records → {len(df)} unique papers.")
            st.dataframe(compact_table(df), width="stretch", hide_index=True, height=360)
            render_save_library_controls(df, "import")


def render_library() -> None:
    st.subheader("📚 Library")
    files = library_files()
    file_names = [p.name for p in files]
    current_file = get_selected_library_file().name
    current_index = file_names.index(current_file) if current_file in file_names else 0
    selected_name = st.selectbox(
        "Library file",
        file_names,
        index=current_index,
        key="library_file_select",
        on_change=on_library_file_change,
        help="Choose the JSONL library file to open. Named libraries are saved under storage/library/.",
    )
    st.session_state["selected_library_file"] = selected_name
    selected_path = LIBRARY_FILE.parent / safe_library_filename(selected_name)
    df = cached_library(str(selected_path))
    st.caption(f"Loaded library: {selected_path.name} ({len(df)} unique papers)")
    if df.empty:
        st.info("Your local library is empty. Search online or import external search data first.")
        return
    with st.form("library_filters_form"):
        st.markdown("##### Library filters")
        q = st.text_input("Search within library", placeholder="battery, SiOC, author, journal, DOI")
        c1, c2, c3 = st.columns(3)
        year_from = c1.text_input("From year", key="lib_year_from")
        year_to = c2.text_input("To year", key="lib_year_to")
        sources = sorted([s for s in df["source"].dropna().astype(str).unique() if s])
        selected_sources = c3.multiselect("Sources", sources, default=sources)
        apply_filters = st.form_submit_button("Apply filters", type="primary")
    if apply_filters or "library_filtered_df" not in st.session_state:
        filtered = filter_dataframe(
            df,
            global_query=q,
            global_fields=FIELDS,
            field_queries={},
            logic="OR",
            source_filter=selected_sources,
            year_from=year_from,
            year_to=year_to,
        )
        st.session_state["library_filtered_df"] = filtered
    else:
        filtered = st.session_state.get("library_filtered_df", df)
    filtered = norm_df(filtered)
    st.session_state["active_df"] = filtered
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Active papers", len(filtered))
    c2.metric("Library size", len(df))
    c3.metric("Journals", filtered["journal"].replace("", pd.NA).dropna().nunique())
    c4.metric("DOI coverage", f"{round(filtered['doi'].map(lambda x: bool(safe_text(x))).mean()*100,1) if len(filtered) else 0}%")
    st.dataframe(compact_table(filtered), width="stretch", hide_index=True, height=440)
    render_enrichment(filtered, session_key="active_df", save=True, context="library")


def render_insights() -> None:
    st.subheader("✨ Literature insights")
    df = get_active_df()
    if df.empty:
        st.info("No active papers. Search or import first.")
        return

    focus_df, terms, match_report, scope, custom_focus, mode = render_scope_selector("insights", df)
    if focus_df.empty:
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Papers analyzed", len(focus_df))
    c2.metric("Journals", focus_df["journal"].replace("", pd.NA).dropna().nunique())
    years = pd.to_numeric(focus_df["year"], errors="coerce").dropna()
    c3.metric("Year span", f"{int(years.min())}–{int(years.max())}" if not years.empty else "-")
    c4.metric("Abstract coverage", f"{round(focus_df['abstract'].map(lambda x: bool(safe_text(x))).mean()*100,1) if len(focus_df) else 0}%")

    ranking = st.selectbox(
        "Rank matched papers by",
        RANKING_OPTIONS,
        index=0,
        key="insights_ranking",
        help="Controls which matched papers are selected first for the summary and paper cards.",
    )
    ranked_focus_df = rank_papers(focus_df, terms, ranking)
    top_n = safe_slider("Use top N ranked matched papers for summary", min(300, len(ranked_focus_df)), 30, "insights_top_n")
    cards_n = safe_slider("Show N expandable summary cards", min(100, len(focus_df)), 20, "insights_cards_n")
    focus_text = ", ".join(terms)
    summary_df = build_summary_df(ranked_focus_df, focus_text, top_n)

    st.markdown("#### Executive highlights")
    text = " ".join((focus_df["title"].fillna("") + " " + focus_df["abstract"].fillna("")))
    themes = summary_df["theme"].value_counts().head(5) if not summary_df.empty else pd.Series(dtype=int)
    frequent_terms = clean_terms(text, 15)
    st.markdown(f"- **Analysis scope:** {scope}")
    st.markdown(f"- **{len(focus_df)} / {len(df)} papers** are included in this summary.")
    st.markdown(f"- **Ranking:** {ranking}.")
    if terms:
        st.markdown(f"- **Scope terms:** {', '.join(terms[:18])}{'...' if len(terms) > 18 else ''}")
    st.markdown(f"- Dominant themes: **{', '.join([f'{k} ({v})' for k, v in themes.items()]) or 'not enough metadata'}**.")
    st.markdown(f"- Frequent terms: **{', '.join(frequent_terms[:12]) or 'not enough text'}**.")
    st.markdown("- Summaries are generated from ranked metadata, titles, abstracts, keywords and journal information for speed and reliability.")

    brief = make_review_brief(summary_df, focus_text or scope, focus_df)
    narrative_theme_values = sorted({
        theme.strip()
        for value in summary_df["theme"].dropna().astype(str)
        for theme in value.split(",")
        if theme.strip()
    }) if "theme" in summary_df else []
    narrative_theme = st.selectbox(
        "Narrative review scope",
        ["All matched themes", *narrative_theme_values],
        key="narrative_theme_scope",
        help="Choose whether the narrative review uses all matched papers or only one selected theme/topic.",
    )
    narrative_df = summary_df
    if narrative_theme != "All matched themes":
        narrative_df = summary_df[
            summary_df["theme"].astype(str).map(
                lambda value: narrative_theme in [x.strip() for x in value.split(",")]
            )
        ].reset_index(drop=True)
        narrative_df["citation"] = [f"[{i}]" for i in range(1, len(narrative_df) + 1)]
    narrative_focus = narrative_theme if narrative_theme != "All matched themes" else (focus_text or scope)
    narrative = make_narrative_review(narrative_df, narrative_focus)
    c1, c2, c3 = st.columns(3)
    c1.download_button("Download summary CSV", summary_df.to_csv(index=False), "literature_summary.csv", "text/csv")
    c2.download_button("Download brief review", brief, "literature_review_brief.md", "text/markdown")
    c3.download_button("Download narrative review", narrative, "narrative_literature_review.md", "text/markdown")

    with st.expander("Preview narrative review", expanded=False):
        st.markdown(narrative)

    with st.expander("AI polish narrative", expanded=False):
        st.caption("Optional polishing uses selected paper metadata and abstracts only. It does not analyze full PDFs. For cloud providers, use your own API key and avoid confidential or unpublished text.")
        provider = st.selectbox(
            "AI polishing provider",
            ["Ollama local", "Groq API (user key)", "Gemini API (user key)"],
            key="ai_polish_provider",
        )
        polished = ""
        cache_key = ""
        if provider == "Ollama local":
            o1, o2, o3 = st.columns([1, 1.4, .8])
            ollama_host = o2.text_input("Ollama host", value="http://localhost:11434", key="ollama_host")
            if ollama_host.startswith("http://localhost") or ollama_host.startswith("http://127.0.0.1"):
                st.caption("Deployment note: on Streamlit Cloud, localhost points to the cloud container, not your laptop. Local Ollama models cannot be loaded there unless Ollama is exposed through a reachable private host.")
            if st.button("Load installed Ollama models", key="ollama_load_models"):
                models = fetch_ollama_models(ollama_host)
                if models:
                    st.session_state["ollama_models"] = models
                    st.success(f"Loaded {len(models)} Ollama model(s).")
                else:
                    st.warning("Could not load Ollama models from this host. If this app is deployed, local Ollama on your laptop is not reachable from Streamlit Cloud. Use local Streamlit for Ollama, provide a reachable Ollama host, or skip AI polishing.")
            model_options = st.session_state.get("ollama_models") or OLLAMA_MODEL_OPTIONS
            if "llama3.2:3b" in model_options:
                default_model_idx = model_options.index("llama3.2:3b")
            else:
                default_model_idx = 0
            ollama_model = o1.selectbox("Model", model_options, index=default_model_idx, key="ollama_model")
            ollama_timeout = o3.number_input("Timeout sec", 30, 600, 180, 30, key="ollama_timeout")
            ollama_records = st.slider(
                "Records sent to Ollama",
                1,
                min(40, len(narrative_df)),
                min(8, len(narrative_df)),
                key="ollama_records",
                help="Lower values are faster. Use the ranked top records first.",
            )
            cache_key = f"{scope}|{ranking}|{top_n}|{ollama_model}|{ollama_records}|{hash(narrative)}"
            if st.button("Polish narrative with Ollama", type="primary", key="ollama_polish_btn"):
                with st.spinner("Asking local Ollama to polish the narrative..."):
                    try:
                        polished = polish_narrative_with_ollama(
                            narrative,
                            narrative_df,
                            narrative_focus,
                            model=ollama_model,
                            host=ollama_host,
                            timeout=int(ollama_timeout),
                            max_records=int(ollama_records),
                        )
                        st.session_state["ollama_polished_narrative"] = polished
                        st.session_state["ollama_polished_key"] = cache_key
                        st.success("Ollama polished narrative generated.")
                    except Exception as exc:
                        st.error(f"Ollama polishing failed: {exc}")
        elif provider == "Groq API (user key)":
            st.info("Groq works in deployed Streamlit. Enter your own key or set GROQ_API_KEY in Streamlit secrets. API usage may incur token costs beyond free limits.")
            g1, g2, g3 = st.columns([1.2, 1.2, .8])
            groq_model = g1.selectbox("Groq model", GROQ_MODEL_OPTIONS, index=0, key="groq_model")
            groq_key = g2.text_input("Groq API key", value=os.getenv("GROQ_API_KEY", ""), type="password", key="groq_api_key")
            groq_timeout = g3.number_input("Timeout sec", 30, 600, 180, 30, key="groq_timeout")
            groq_records = st.slider("Records sent to Groq", 1, min(30, len(narrative_df)), min(8, len(narrative_df)), key="groq_records")
            cache_key = f"{scope}|{ranking}|{top_n}|groq|{groq_model}|{groq_records}|{hash(narrative)}"
            if st.button("Polish narrative with Groq", type="primary", key="groq_polish_btn"):
                with st.spinner("Asking Groq to polish the narrative..."):
                    try:
                        polished = polish_narrative_with_groq(
                            narrative,
                            narrative_df,
                            narrative_focus,
                            api_key=groq_key,
                            model=groq_model,
                            timeout=int(groq_timeout),
                            max_records=int(groq_records),
                        )
                        st.session_state["ai_polished_narrative"] = polished
                        st.session_state["ai_polished_key"] = cache_key
                        st.success("Groq-polished narrative generated.")
                    except Exception as exc:
                        st.error(f"Groq polishing failed: {exc}")
        else:
            st.info("Gemini works in deployed Streamlit. Enter your own key or set GEMINI_API_KEY in Streamlit secrets. Check Google AI Studio terms before sending sensitive text.")
            m1, m2, m3 = st.columns([1.2, 1.2, .8])
            gemini_model = m1.selectbox("Gemini model", GEMINI_MODEL_OPTIONS, index=0, key="gemini_model")
            gemini_key = m2.text_input("Gemini API key", value=os.getenv("GEMINI_API_KEY", ""), type="password", key="gemini_api_key")
            gemini_timeout = m3.number_input("Timeout sec", 30, 600, 180, 30, key="gemini_timeout")
            gemini_records = st.slider("Records sent to Gemini", 1, min(30, len(narrative_df)), min(8, len(narrative_df)), key="gemini_records")
            cache_key = f"{scope}|{ranking}|{top_n}|gemini|{gemini_model}|{gemini_records}|{hash(narrative)}"
            if st.button("Polish narrative with Gemini", type="primary", key="gemini_polish_btn"):
                with st.spinner("Asking Gemini to polish the narrative..."):
                    try:
                        polished = polish_narrative_with_gemini(
                            narrative,
                            narrative_df,
                            narrative_focus,
                            api_key=gemini_key,
                            model=gemini_model,
                            timeout=int(gemini_timeout),
                            max_records=int(gemini_records),
                        )
                        st.session_state["ai_polished_narrative"] = polished
                        st.session_state["ai_polished_key"] = cache_key
                        st.success("Gemini-polished narrative generated.")
                    except Exception as exc:
                        st.error(f"Gemini polishing failed: {exc}")
        legacy_polished = st.session_state.get("ollama_polished_narrative", "")
        polished = st.session_state.get("ai_polished_narrative", "") or legacy_polished
        key_match = st.session_state.get("ai_polished_key") == cache_key or st.session_state.get("ollama_polished_key") == cache_key
        if polished and key_match:
            st.markdown("##### Polished narrative")
            st.markdown(polished)
            st.download_button(
                "Download AI-polished review",
                polished,
                "ai_polished_literature_review.md",
                "text/markdown",
                key="ai_polished_download",
            )

    st.markdown("#### Expandable paper summary cards")
    for i, row in summary_df.head(cards_n).iterrows():
        title = safe_text(row.get("title")) or "Untitled"
        match_reason = safe_text(row.get("match_reason"))
        with st.expander(f"{i+1}. {title[:120]}", expanded=False):
            st.caption(f"{row['year']} · {row['journal']} · {row['theme']} · DOI: {row['doi'] or 'N/A'}")
            st.caption(f"Source: {row.get('source') or 'Unknown'} | Relevance: {row.get('relevance_pct', 0)}%")
            if match_reason:
                st.caption(f"Match reason: {match_reason}")
            st.write(row["quick_summary"])
            st.caption(f"Key terms: {row['key_terms']}")

    st.markdown("#### Literature review table")
    st.dataframe(summary_df, width="stretch", hide_index=True, height=430)


def make_review_brief(summary_df: pd.DataFrame, focus: str, df: pd.DataFrame) -> str:
    lines = ["# Literature Review Brief", "", f"Focus: {focus or 'General'}", f"Papers analyzed: {len(summary_df)}", "", "## Representative papers"]
    for _, r in summary_df.head(40).iterrows():
        lines.append(
            f"- **{r['title']}** ({r['year']}, {r['journal']}; source: {r.get('source', 'Unknown')}; "
            f"relevance: {r.get('relevance_pct', 0)}%): {r['quick_summary']}"
        )
    return "\n".join(lines) + "\n"


def make_narrative_review(summary_df: pd.DataFrame, focus: str) -> str:
    if summary_df.empty:
        return "# Narrative Literature Review\n\nNo papers available for this scope.\n"

    focus_label = focus or "the selected literature scope"
    themes = summary_df["theme"].replace("", "General").value_counts().head(5)
    top_refs = " ".join(summary_df["citation"].head(min(5, len(summary_df))).tolist())
    theme_sentence = ", ".join(f"{theme} ({count} papers)" for theme, count in themes.items())

    paragraphs = [
        "# Narrative Literature Review",
        "",
        (
            f"This review summarizes the ranked literature for {focus_label}. "
            f"The selected records are organized around {theme_sentence or 'general metadata patterns'}, "
            f"and the highest-ranked papers provide the main evidence base {top_refs}."
        ),
        "",
        "## Abstract-Based Summary",
        "",
        "The abstracts of the ranked papers can be summarized as follows:",
    ]

    for i, (_, row) in enumerate(summary_df.iterrows(), 1):
        summary = safe_text(row.get("quick_summary"))
        if not summary or summary == "No abstract available.":
            summary = "No abstract was available, so this record contributes mainly bibliographic or metadata context"
        title = safe_text(row.get("title"))
        relevance = safe_text(row.get("relevance_pct"))
        paragraphs.append(
            f"{i}. {title}: {summary.rstrip('.')}. "
            f"This record is associated with {safe_text(row.get('theme'))} and has {relevance}% keyword relevance {row['citation']}."
        )

    paragraphs.extend(["", "## Cross-Paper Summary", ""])

    for theme, group in summary_df.groupby("theme", sort=False):
        cited = " ".join(group["citation"].head(5).tolist())
        top_terms = ", ".join(clean_terms(" ".join(group["key_terms"].fillna("")), 8))
        paragraphs.append(
            f"Taken together, the {theme} records point toward recurring concepts such as {top_terms or 'the selected scope terms'}. "
            f"The evidence for this theme is distributed across {len(group)} ranked record(s), with representative support from {cited}."
        )

    high_relevance = summary_df.sort_values("relevance_pct", ascending=False).head(5)
    if not high_relevance.empty and numeric_value(high_relevance.iloc[0].get("relevance_pct")) > 0:
        title_refs = "; ".join(
            f"{safe_text(row.get('title'))} {row['citation']}"
            for _, row in high_relevance.head(3).iterrows()
        )
        paragraphs.append(
            f"The strongest keyword alignment is found in {title_refs}. "
            f"These records should be reviewed first when the goal is to understand the selected theme."
        )

    paragraphs.extend(["", "## References"])
    for _, row in summary_df.iterrows():
        authors = safe_text(row.get("authors"))
        year = safe_text(row.get("year"))
        title = safe_text(row.get("title"))
        journal = safe_text(row.get("journal"))
        doi = safe_text(row.get("doi"))
        source = safe_text(row.get("source"))
        ref = f"{row['citation']} "
        if authors:
            ref += f"{authors}. "
        ref += f"{title}. "
        if journal or year:
            ref += f"{journal} ({year}). "
        if doi:
            ref += f"DOI: {doi}. "
        if source:
            ref += f"Source: {source}."
        paragraphs.append(ref.strip())

    return "\n\n".join(paragraphs) + "\n"


def ollama_payload_from_summary(summary_df: pd.DataFrame, focus: str, max_records: int = 20) -> str:
    lines = [f"Focus: {focus or 'selected literature scope'}", ""]
    for _, row in summary_df.head(max_records).iterrows():
        lines.append(
            "\n".join(
                [
                    f"{row['citation']} {safe_text(row.get('title'))}",
                    f"Year: {safe_text(row.get('year'))}",
                    f"Journal: {safe_text(row.get('journal'))}",
                    f"Theme: {safe_text(row.get('theme'))}",
                    f"Source: {safe_text(row.get('source'))}",
                    f"Relevance: {safe_text(row.get('relevance_pct'))}%",
                    f"Summary: {safe_text(row.get('quick_summary'))}",
                    f"Key terms: {safe_text(row.get('key_terms'))}",
                ]
            )
        )
    return "\n\n".join(lines)


def fetch_ollama_models(host: str = "http://localhost:11434", timeout: int = 5) -> list[str]:
    try:
        response = requests.get(f"{host.rstrip('/')}/api/tags", timeout=timeout)
        response.raise_for_status()
        data = response.json()
        models = []
        for item in data.get("models", []):
            name = safe_text(item.get("name"))
            if name and "embed" not in name.lower():
                models.append(name)
        return sorted(dict.fromkeys(models))
    except Exception:
        return []


def build_polish_prompt(narrative: str, summary_df: pd.DataFrame, focus: str, max_records: int = 20) -> str:
    evidence = ollama_payload_from_summary(summary_df, focus, max_records=max_records)
    return f"""Rewrite the draft literature review into a polished, continuous scientific narrative.

Rules:
- Preserve every citation marker exactly, such as [1], [2], [3].
- Put citations immediately after the sentence they support.
- Do not invent facts, methods, results, authors, journals, years, or citations.
- Use only the evidence provided below.
- Keep a References section using the same numbered references from the draft.
- Write clear paragraphs, not bullet points.
- Keep the tone suitable for a scientific literature review.
- This is an abstract-level synthesis from metadata and abstracts only, not a full-PDF review.

Evidence:
{evidence}

Draft review:
{narrative}
"""


def polish_narrative_with_ollama(
    narrative: str,
    summary_df: pd.DataFrame,
    focus: str,
    *,
    model: str = "llama3.2:3b",
    host: str = "http://localhost:11434",
    timeout: int = 180,
    max_records: int = 20,
) -> str:
    prompt = build_polish_prompt(narrative, summary_df, focus, max_records=max_records)
    response = requests.post(
        f"{host.rstrip('/')}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    return safe_text(data.get("response")) or narrative


def polish_narrative_with_groq(
    narrative: str,
    summary_df: pd.DataFrame,
    focus: str,
    *,
    api_key: str,
    model: str = "llama-3.1-8b-instant",
    timeout: int = 180,
    max_records: int = 20,
) -> str:
    if not api_key:
        raise ValueError("Groq API key is required.")
    prompt = build_polish_prompt(narrative, summary_df, focus, max_records=max_records)
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": "You polish scientific literature-review text using only supplied evidence."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 2200,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    return safe_text(data.get("choices", [{}])[0].get("message", {}).get("content")) or narrative


def polish_narrative_with_gemini(
    narrative: str,
    summary_df: pd.DataFrame,
    focus: str,
    *,
    api_key: str,
    model: str = "gemini-2.5-flash-lite",
    timeout: int = 180,
    max_records: int = 20,
) -> str:
    if not api_key:
        raise ValueError("Gemini API key is required.")
    prompt = build_polish_prompt(narrative, summary_df, focus, max_records=max_records)
    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        params={"key": api_key},
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 2200},
        },
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    candidates = data.get("candidates") or []
    parts = (((candidates[0] if candidates else {}).get("content") or {}).get("parts") or [])
    return safe_text(" ".join(safe_text(part.get("text")) for part in parts)) or narrative


def render_analysis() -> None:
    st.subheader("📊 Review-ready analytics")
    df = get_active_df()
    if df.empty:
        st.info("No active papers to analyze.")
        return

    scoped, terms, match_report, scope, custom_terms, mode = render_scope_selector("analytics", df)
    if scoped.empty:
        return

    dfa = norm_df(scoped)
    dfa["year_num"] = pd.to_numeric(dfa["year"], errors="coerce")
    dfa["journal_clean"] = dfa["journal"].map(lambda x: safe_text(x) or "Unknown")
    dfa["source_clean"] = dfa["source"].map(lambda x: safe_text(x) or "Unknown")
    dfa["theme"] = dfa.apply(
        lambda row: safe_text(row.get("custom_theme")) or classify_theme(f"{row.get('title', '')} {row.get('abstract', '')}"),
        axis=1,
    )

    st.caption(f"Charts and comparison below are based on the selected scope: {scope} ({len(dfa)} papers).")

    c1, c2 = st.columns(2)
    years = dfa.dropna(subset=["year_num"]).copy()
    if not years.empty:
        years["year_num"] = years["year_num"].astype(int)
        yc = years.groupby("year_num").size().reset_index(name="papers")
        yc["year"] = yc["year_num"].astype(str)
        fig = px.bar(yc, x="year", y="papers", text="papers", title="Publications by year")
        fig.update_traces(textposition="outside")
        fig.update_layout(xaxis_title="Year", yaxis_title="Papers", bargap=.12)
        fig.update_xaxes(type="category")
        c1.plotly_chart(fig, width="stretch")
    src = source_contribution_frame(dfa)
    c2.plotly_chart(px.pie(src, values="Papers", names="Source", title="Contributing source distribution", hole=.38), width="stretch")
    c2.caption("Source counts are non-exclusive: a merged paper can contribute to multiple sources, so source totals may exceed the number of displayed papers.")

    c3, c4 = st.columns(2)
    journals = dfa["journal_clean"].value_counts().head(20).reset_index()
    journals.columns = ["Journal", "Papers"]
    c3.plotly_chart(px.bar(journals, x="Papers", y="Journal", orientation="h", title="Top journals / sources"), width="stretch")
    theme_counts_df = dfa.assign(theme_item=dfa["theme"].str.split(r",\s*")).explode("theme_item")
    theme_counts_df["theme_item"] = theme_counts_df["theme_item"].map(lambda x: safe_text(x) or "General")
    themes = theme_counts_df["theme_item"].value_counts().reset_index()
    themes.columns = ["Theme", "Papers"]
    c4.plotly_chart(px.pie(themes, values="Papers", names="Theme", title="Theme distribution"), width="stretch")

    c5, c6 = st.columns(2)
    qdf = metadata_quality(dfa)
    c5.plotly_chart(px.bar(qdf, x="Field", y="Coverage %", text="Coverage %", title="Metadata completeness"), width="stretch")
    text = " ".join((dfa["title"].fillna("") + " " + dfa["abstract"].fillna("")))
    term_df = pd.DataFrame({"Term": clean_terms(text, 25)})
    c6.dataframe(term_df, width="stretch", hide_index=True)

    scored_dfa = rank_papers(dfa, terms, "Relevance to theme/keywords")
    if not scored_dfa.empty:
        scored_dfa["citation_num"] = scored_dfa["citation_count"].map(numeric_value) if "citation_count" in scored_dfa else 0
        c7, c8 = st.columns(2)
        top_relevance = scored_dfa.sort_values("_relevance_pct", ascending=False).head(12).copy()
        top_relevance = top_relevance.reset_index(drop=True)
        top_relevance["paper_label"] = top_relevance.index.map(lambda i: f"Paper {i + 1}")
        top_relevance["short_title"] = top_relevance["title"].map(lambda x: safe_text(x)[:95] + ("..." if len(safe_text(x)) > 95 else ""))
        relevance_fig = px.bar(
            top_relevance.sort_values("_relevance_pct", ascending=True),
            x="_relevance_pct",
            y="paper_label",
            color="theme",
            orientation="h",
            title="Top papers by relevance to selected scope",
            labels={"_relevance_pct": "Matched terms (%)", "paper_label": "Paper"},
            hover_data={"short_title": True, "_relevance_pct": ":.1f", "paper_label": False},
        )
        relevance_fig.update_traces(text=top_relevance.sort_values("_relevance_pct", ascending=True)["_relevance_pct"].map(lambda v: f"{v:.1f}%"), textposition="outside", cliponaxis=False)
        relevance_fig.update_layout(height=430, margin=dict(l=70, r=40, t=70, b=60), xaxis_range=[0, max(5, min(100, float(top_relevance["_relevance_pct"].max()) * 1.15))])
        c7.plotly_chart(relevance_fig, width="stretch")
        with c7.expander("Paper labels", expanded=False):
            st.dataframe(top_relevance[["paper_label", "short_title", "_relevance_pct"]].rename(columns={"paper_label": "Paper", "short_title": "Title", "_relevance_pct": "Matched terms %"}), width="stretch", hide_index=True)
        c8.plotly_chart(
            px.scatter(
                scored_dfa,
                x="_relevance_pct",
                y="citation_num",
                color="theme",
                hover_name="title",
                size="_metadata_score",
                title="Relevance vs citation count",
                labels={"_relevance_pct": "Relevance %", "citation_num": "Citations"},
            ),
            width="stretch",
        )

    if not years.empty:
        heat = dfa.dropna(subset=["year_num"]).copy()
        heat["year"] = heat["year_num"].astype(int).astype(str)
        heat = heat.assign(theme_item=heat["theme"].str.split(r",\s*")).explode("theme_item")
        heat["theme_item"] = heat["theme_item"].map(lambda x: safe_text(x) or "General")
        heat = heat.groupby(["year", "theme_item"]).size().reset_index(name="Papers")
        heat = heat.rename(columns={"theme_item": "theme"})
        hfig = px.density_heatmap(heat, x="year", y="theme", z="Papers", histfunc="sum", title="Theme evolution over time")
        hfig.update_xaxes(type="category")
        st.plotly_chart(hfig, width="stretch")

    st.markdown("#### Compare papers")
    st.caption("The comparison table uses the same analysis scope selected above.")
    ranking = st.selectbox(
        "Rank matched papers by",
        RANKING_OPTIONS,
        index=0,
        key="analytics_ranking",
        help="Controls which matched papers are selected first for the comparison table.",
    )
    ranked_dfa = rank_papers(dfa, terms, ranking)
    n = safe_slider("Compare top N ranked matched papers", min(100, len(ranked_dfa)), 20, "compare_top_n")
    comp_df = build_summary_df(ranked_dfa, ", ".join(terms), n)
    st.dataframe(comp_df, width="stretch", hide_index=True, height=430)
    st.download_button("Download comparison CSV", comp_df.to_csv(index=False), "paper_comparison.csv", "text/csv")


def export_payload(df: pd.DataFrame, fmt: str) -> tuple[str, str, str]:
    if fmt == "CSV":
        return "literature_records.csv", df.to_csv(index=False), "text/csv"
    if fmt == "BibTeX":
        return "literature_records.bib", to_bibtex(df), "text/plain"
    if fmt == "RIS":
        return "literature_records.ris", to_ris(df), "text/plain"
    if fmt == "Markdown":
        return "literature_records.md", to_markdown(df), "text/markdown"
    return "literature_records.json", to_json(df), "application/json"


def email_body(sender: str, n: int, topic: str) -> str:
    return f"""Dear colleague,

I am sharing a curated literature dataset prepared with the Literature Metadata Intelligence Dashboard.

Dataset overview:
- Topic/focus: {topic or 'selected scholarly records'}
- Number of records: {n}
- Suggested use: literature review screening, bibliography management, trend analysis, and metadata-based comparison.

The dataset can be opened in Excel, imported into Zotero/Mendeley/EndNote, or used as an input file for downstream literature intelligence workflows.

Best regards,
{sender or 'Literature Dashboard User'}
""".strip()


def parse_recipients(text: str) -> list[str]:
    recipients = []
    seen = set()
    for part in re.split(r"[,;\n]+", safe_text(text)):
        email = part.strip()
        if email and email.lower() not in seen:
            recipients.append(email)
            seen.add(email.lower())
    return recipients


def eml_with_attachment(sender: str, recipient: str, subject: str, body: str, filename: str, content: str, mime: str) -> bytes:
    from email.message import EmailMessage
    msg = EmailMessage()
    msg["From"] = sender or ""
    msg["To"] = recipient or ""
    msg["Subject"] = subject
    msg.set_content(body)
    maintype, subtype = (mime.split("/", 1) + ["plain"])[:2]
    msg.add_attachment(content.encode("utf-8"), maintype=maintype, subtype=subtype, filename=filename)
    return bytes(msg)


def render_export() -> None:
    st.subheader("📤 Export & share")
    df = get_active_df()
    if df.empty:
        st.info("No active papers. Search, import, or open the library first.")
        return
    st.caption("Export the active result set. Use filters in Library/Insights first if you want a smaller dataset.")
    c1, c2, c3 = st.columns(3)
    fmt = c1.selectbox("Export format", ["CSV", "BibTeX", "RIS", "Markdown", "JSON"])
    n = c2.number_input("Records to export", 1, len(df), len(df))
    topic = c3.text_input("Topic label", placeholder="SiOC battery anodes")
    export_df = df.head(int(n))
    filename, content, mime = export_payload(export_df, fmt)
    st.download_button(f"Download {fmt}", content, filename, mime, type="primary")
    st.markdown("#### Email handoff")
    st.caption("Browser email drafts cannot attach files automatically. Download the attachment above, or download the .eml file below with the attachment already embedded.")
    e1, e2 = st.columns(2)
    sender = e1.text_input("Sender name/email", placeholder="Maged Bekheet <you@example.com>")
    recipient_text = e2.text_area("Recipient emails", placeholder="colleague@example.com, collaborator@example.com", height=70)
    recipients = parse_recipients(recipient_text)
    recipient = ", ".join(recipients)
    subject = st.text_input("Email subject", value=f"Literature dataset: {topic or 'selected papers'}")
    body = st.text_area("Email message", value=email_body(sender, len(export_df), topic), height=230)
    mailto = f"mailto:{quote(recipient)}?subject={quote(subject)}&body={quote(body)}"
    gmail = f"https://mail.google.com/mail/?view=cm&to={quote(recipient)}&su={quote(subject)}&body={quote(body)}"
    outlook = f"https://outlook.office.com/mail/deeplink/compose?to={quote(recipient)}&subject={quote(subject)}&body={quote(body)}"
    if recipients:
        st.caption(f"Recipients: {', '.join(recipients)}")
    st.markdown(f"[Open default email app]({mailto}) · [Open Gmail draft]({gmail}) · [Open Outlook draft]({outlook})")
    st.download_button("Download .eml with attachment", eml_with_attachment(sender, recipient, subject, body, filename, content, mime), "literature_dataset_email.eml", "message/rfc822")


def render_about() -> None:
    st.subheader("ℹ️ About this dashboard")
    st.markdown(
        """
This is a public, clean metadata intelligence dashboard for scholarly search results. It can also import outputs generated by my companion automation project **ai-literature-feed-automation**.

**Recommended workflow:**
1. Search online sources or import external search data.
2. Save useful records to the library and enrich missing metadata there when needed.
3. Review insights, charts, summaries and comparisons.
4. Export CSV/BibTeX/RIS/Markdown/JSON or share by email.

**Stable source strategy:**
- **OpenAlex**: primary large-scale backend.
- **Crossref**: DOI/journal metadata.
- **Google Scholar**: optional broad discovery/enrichment; may be blocked or rate-limited.
- **Semantic Scholar**: optional enrichment, not required for core search.

This version focuses on fast, transparent metadata review: search/import, deduplication, library enrichment, theme-based analysis, comparison, and export.
        """
    )


def main() -> None:
    style()
    hero()
    page = st.sidebar.radio(
        "Navigation",
        ["Search", "Import External Data", "Library", "Insights", "Analytics", "Export & Share", "About"],
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("Developed by Maged Bekheet")
    st.sidebar.caption("For AI-assisted literature review and research intelligence")
    if page == "Search":
        render_search()
    elif page == "Import External Data":
        render_import()
    elif page == "Library":
        render_library()
    elif page == "Insights":
        render_insights()
    elif page == "Analytics":
        render_analysis()
    elif page == "Export & Share":
        render_export()
    else:
        render_about()
    footer()


if __name__ == "__main__":
    main()
