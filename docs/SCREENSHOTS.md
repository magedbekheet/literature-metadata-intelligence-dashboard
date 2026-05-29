# Screenshots

Add GitHub screenshots to `docs/screenshots/` using these filenames so the README renders them automatically:

```text
docs/screenshots/search.png
docs/screenshots/library-enrichment.png
docs/screenshots/insights-custom-themes.png
docs/screenshots/analytics.png
docs/screenshots/export-share.png
```

Recommended capture flow:

1. Start the app with `streamlit run app.py`.
2. Use a browser width of roughly `1440px`.
3. Capture clean states with representative data loaded.
4. Avoid showing private API keys, emails, or unpublished library data.

Suggested screenshots:

- **Search**: query form plus merged result table.
- **Library enrichment**: enrichment mode selector and coverage/report output.
- **Insights**: customized theme/keywords with theme match counts and narrative review.
- **Analytics**: year/source/theme charts plus relevance-vs-citation plot.
- **Export & Share**: export controls and multiple-recipient email handoff.
