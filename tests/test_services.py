import unittest

import pandas as pd

from app.services.exporter import to_bibtex
from app.services.filtering import filter_dataframe
from app.services.importer import parse_bibtex, parse_csv
from app.services.schema import normalize_paper


class ServiceTests(unittest.TestCase):
    def test_normalizes_api_shaped_authors_and_pdf_urls(self):
        paper = normalize_paper(
            {
                "title": "Example Paper",
                "authorships": [{"author": {"display_name": "Ada Lovelace"}}],
                "openAccessPdf": {"url": "https://example.org/paper.pdf"},
            }
        )

        self.assertEqual(paper["authors"], ["Ada Lovelace"])
        self.assertEqual(paper["pdf_url"], "https://example.org/paper.pdf")

    def test_filter_year_bounds_are_safe_and_strict(self):
        df = pd.DataFrame(
            [
                {"title": "Old", "year": "2019"},
                {"title": "Current", "year": "2024-03-01"},
                {"title": "Unknown", "year": ""},
            ]
        )

        filtered = filter_dataframe(df, year_from="2020", year_to="2025")

        self.assertEqual(filtered["title"].tolist(), ["Current"])
        self.assertEqual(filter_dataframe(df, year_from="not a year")["title"].tolist(), ["Old", "Current", "Unknown"])

    def test_imports_bom_csv_and_single_line_bibtex(self):
        csv_records = parse_csv("\ufefftitle,year,author\nCSV Paper,2024,A. Author\n")
        bib_records = parse_bibtex('@article{key, title={Bib Paper}, author={B. Author and C. Author}, year={2023}, doi={10.1234/example}}')

        self.assertEqual(csv_records[0]["title"], "CSV Paper")
        self.assertEqual(bib_records[0]["title"], "Bib Paper")
        self.assertEqual(bib_records[0]["authors"], ["B. Author", "C. Author"])

    def test_bibtex_export_keys_are_informative_unique_and_escaped(self):
        df = pd.DataFrame(
            [
                {"title": "Battery {Study}", "authors_display": "A. Smith", "year": "2024", "journal": "J", "doi": "10.1/a"},
                {"title": "Battery {Study}", "authors_display": "A. Smith", "year": "2024", "journal": "J", "doi": "10.1/b"},
            ]
        )

        bib = to_bibtex(df)

        self.assertIn("@article{smith_2024_battery_study,", bib)
        self.assertIn("@article{smith_2024_battery_study_2,", bib)
        self.assertIn("Battery \\{Study\\}", bib)


if __name__ == "__main__":
    unittest.main()
