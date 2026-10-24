# Deployment

## Local (live or offline)

```bash
uv sync
```

```bash
uv run python scripts/bootstrap.py
```

```bash
uv run streamlit run app/main.py
```

Mode comes from `APP_MODE` in `.env` (`local_offline` by default). The database defaults to
`data/harm_detection.sqlite`. Migrations run automatically on startup in local modes.

## Public demo

1. In local mode, create the runs you want to show (for example, live Jev and Moderation runs on Civil
   Comments, an optimization experiment, a holdout evaluation).
2. Export the curated artifact:

```bash
uv run python scripts/export_public_demo.py --runs 3 4 5
```

3. Commit `data/public_demo/public_demo.sqlite` (it contains only CC0/synthetic content, normalized signals,
   metrics and configs).
4. Deploy with `APP_MODE=public_demo` (for example on Streamlit Community Cloud with main file `app/main.py`,
   and `APP_MODE` set as a secret/env var). **Do not** configure provider keys for the public deployment. Public
   mode ignores them anyway.

Public mode behavior:

* Opens the SQLite artifact with `mode=ro`, so no writes are possible even if code tried.
* Disables live inference, GPT analysis, dataset ingestion, configuration edits, and run deletion.
* Threshold exploration (what-if sliders, candidate search) works in memory per session and is never saved.
* Works on hosts with ephemeral or read-only storage.

## Data retention

Local runs, datasets and configurations can be deleted from the Experiment History and Configuration pages.
Deleting the SQLite file removes all local state. Raw dataset downloads live in `data/raw/` (git-ignored).
Playground inputs are stored only in the local database, under the `playground` dataset, and are never exported.
