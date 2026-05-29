from app.services.importer import parse_uploaded_file
from app.services.schema import normalize_paper, safe_text

p = normalize_paper({
    "title": "<b>SiOC Battery Anode</b>",
    "abstract": "<jats:p>Silicon oxycarbide is useful.</jats:p>",
    "authors": ["A. Test"],
    "journal": "Journal X",
    "year": "2024",
})
assert p["title"] == "SiOC Battery Anode"
assert "Silicon oxycarbide" in p["abstract"]
assert p["journal"] == "Journal X"
print("Smoke test passed")
