# Companion Automation Workflow

This dashboard is designed to work alongside:

[magedbekheet/ai-literature-feed-automation](https://github.com/magedbekheet/ai-literature-feed-automation)

The companion project can automate literature discovery and produce structured search outputs. This dashboard then acts as the interactive review, enrichment, analysis, and export layer.

## Recommended Two-Project Flow

```text
ai-literature-feed-automation
  -> scheduled search feeds
  -> JSONL / BibTeX / RIS / digest files
  -> import into this dashboard
  -> deduplicate and enrich metadata
  -> analyze by theme or custom theme/keywords
  -> export curated datasets and review briefs
```

## Supported Companion Outputs

Import these files through **Import External Data**:

```text
data/feeds/papers.jsonl
data/bibtex/selected_papers.bib
data/ris/selected_papers.ris
data/digests/latest_digest.md
```

## Handoff Schema

The most useful fields are:

```json
{
  "title": "",
  "authors": [],
  "abstract": "",
  "journal": "",
  "year": "",
  "doi": "",
  "url": "",
  "source": "",
  "keywords": [],
  "pdf_url": "",
  "citation_count": 0
}
```

## Division Of Labor

Use the automation project for:

- Scheduled discovery.
- Feed generation.
- Repeated query monitoring.
- Producing raw or semi-curated literature files.

Use this dashboard for:

- Interactive screening.
- Deduplication and metadata cleanup.
- Library enrichment.
- Theme-based analytics.
- Narrative summaries and comparison tables.
- Export/share handoff.
