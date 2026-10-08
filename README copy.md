<<<<<<< HEAD
# nbharatiya_usecase1
=======
# IDAMP

Intent-Driven Agentic Medallion Pipeline turns uploaded CSV files into reviewed Bronze, Silver, and Gold Parquet layers, then produces a self-contained HTML executive report.

## Requirements

- Python 3.10 or newer
- A Groq API key for STTM generation and report authoring

## Setup

From this `idamp` folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `GROQ_API_KEY` in `.env`. `GROQ_MODEL` defaults to `openai/gpt-oss-120b`.

## Run

```powershell
streamlit run app/streamlit_app.py
```

Upload CSV files, state the business intent, and review each generated mapping before approving its layer. The app stores Parquet outputs under `data/`, reports under `reports/`, STTM snapshots under `sttm_history/`, and the SQLite audit trail under `audit/audit.db`.

## Notes

The reporter accepts one read-only DuckDB `SELECT`/`WITH` statement against `gold_data`. Transformation expressions in Gold STTMs support `GROUP BY`, `SUM`, `COUNT`, `AVG`, `MIN`, `MAX`, and ISO-date comparisons. The Gold mapping is capped at eight entries; the UI setting limits Gold output rows. API credentials are read from the local `.env` file and are never included in exported session data.
>>>>>>> 6653639 (Initial commit)
