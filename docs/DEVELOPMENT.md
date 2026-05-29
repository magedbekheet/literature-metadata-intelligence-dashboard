# Development

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

On Windows:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
streamlit run app.py
```

## Validation

```bash
python smoke_test.py
python -m unittest discover -s tests
python -m compileall app.py app smoke_test.py tests
```

On Windows systems where only the Python launcher is available, replace `python` with `py`.

## Git Hygiene

Ignored by default:

- `.env`
- virtual environments
- Python cache files
- local library JSONL files
- exports/uploads/PDF/vector stores

Keep sample data small and anonymized if you add it later.
