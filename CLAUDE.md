# Betting the Regression — Claude Instructions

## Project overview

NBA player props mean-reversion model. Identifies hot streaks (z-score ≥ 1.0 vs. season baseline) and fades them with UNDER bets. Two main scripts:
- `daily_picks.py` — grades yesterday, generates today's picks, injury check
- `odds_compare.py` — pulls live DraftKings lines, generates `dashboard.html`

API keys live in `~/.zshrc` as environment variables only. Read with `os.environ.get('KEY_NAME')`. Never hardcode any key.

---

## Data pipelines

Work incrementally:
1. Read CSV headers and sample rows (first 5–10) before processing full files
2. Confirm structure looks right before committing to full processing
3. Save intermediate results to disk after each stage — never lose progress to a crash

For any pipeline with external data dependencies, **validate the source before building dependent steps**. Confirm the required fields exist and are populated at adequate coverage (>80%) before writing processing code that depends on them.

---

## Data validation

Never apply blanket numeric rescaling (e.g. ×100) across all columns. For each transform:
1. Declare the unit/scale of each column individually
2. Explain the transform and why it applies to that specific field
3. Add before/after assertions confirming known reference values

Percentage fields (like TOV%, FG%, win rates) must never be multiplied blindly — they are already in 0–100 or 0–1 scale and must be handled per-column.

---

## External data sourcing

For this project, prefer stable local datasets over live scraping when both can provide the same field. Scraping is appropriate for:
- Live game data (NBA API, ESPN API) — no local alternative
- DraftKings prop lines — no local alternative

For historical player stats, use `nba_api` (reliable, authenticated). For player attributes, use local CSV files when available.

---

## Pipeline checkpointing

For multi-stage pipelines, make each stage idempotent:
- Check if the output file already exists and is valid before reprocessing
- Write a checkpoint after each stage completes
- On rerun, skip completed stages and resume from the last incomplete one

This keeps long backtest runs (like `hothandfade_v3.py`) resumable after interruption.

---

## Key constants

```python
STAT_ENCODE = {"FG3M": 0, "PTS": 1, "PR": 2, "PA": 3, "PRA": 4}
CURRENT_NBA_SEASON = "2025-26"
PREDICTIONS_FILE = "daily_predictions.csv"
```

These must stay consistent between `train_model.py` and `daily_picks.py`.

---

## Model

- GradientBoosting GB-100, AUC 0.579
- FG3M accuracy: 72.4% (3 seasons, 2,118 events)
- Multi-stat accuracy: 64.0% (4,553 events across all stat types)
- ML probability score only applies to FG3M picks — other stats use rule-based threshold only
- Model files: `model_regression.pkl`, `model_features.json`, `player_regression_rates.json`

---

## Available slash commands

- `/ingest` — header-first CSV inspection before full processing
- `/validate-transforms` — per-column unit verification before applying any numeric transforms
