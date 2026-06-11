# Betting the Regression

A mean-reversion model for NBA player props that identifies hot streaks and predicts when players will come back down to earth.

---

## The thesis

NBA sportsbooks set player prop lines based heavily on recent form — a player's last 3 games. When a role player goes on a hot streak, that line gets inflated above their true skill level. This model identifies those moments (z-score > 1.0 vs. season baseline) and fades them, predicting the player will revert toward their season average in the next game.

**The strategy: bet UNDER on inflated hot-streak lines.** The effect is strongest for three-point shooting (FG3M), where variance is high and sportsbooks consistently overprice recency. The model has validated at 73.9% accuracy over 2,118 events across three NBA seasons.

---

## Results

### Backtest (2022-25, 2,118 events, 558 player-seasons)

| Season | Players | Hot Events | Win Rate | Avg Error (Model) | Avg Error (Naive) |
|--------|---------|------------|----------|-------------------|-------------------|
| 2022-23 | 186 | 678 | 74.2% | 1.03 threes | 1.84 threes |
| 2023-24 | 173 | 730 | 72.6% | 1.04 threes | 1.84 threes |
| 2024-25 | 199 | 710 | 73.9% | 1.02 threes | 1.82 threes |

The model beat the naive rolling-average baseline nearly 2-to-1 in every season tested.

### Model evolution

| Model | Accuracy | ROC-AUC | Key finding |
|-------|----------|---------|-------------|
| Rule-based (v1) | 73.9% | — | Baseline; consistent across 3 seasons |
| Logistic regression (v2.0) | 73.9% | 0.538 | Rule-based is already near-optimal for FG3M |
| GradientBoosting GB-100 (v2.1) | 72.0% | 0.580 | 80.5% actual regression at ≥80% model confidence |
| GB-100 + multi-stat validation (v2.2) | 64.0%\* | 0.579 | Regression signal is stat-specific |
| XGBoost-100 (v2.2) | 64.6%\* | 0.565 | GradientBoosting preferred |

\* v2.2 accuracy measured across all 5 stat types (FG3M + PTS + PR + PA + PRA, 4,553 events). FG3M accuracy unchanged at 72.4%.

### Key findings

- **92–98% of role players** show the regression pattern consistently across all 3 seasons
- **Stable minutes is the critical filter** — players whose minutes are trending upward may be experiencing genuine role expansion, not a hot streak
- **FG3M has the strongest regression signal** (73.9%) — three-point shooting is high variance, and sportsbooks consistently overweight recent form
- **At ≥80% model confidence**, actual regression rate is **80.5%** — the ML model adds real calibration value for high-conviction picks
- **The model does not generalise across stat types** — trained on FG3M, it achieves only ~64% on PTS/PRA/PR/PA (close to the 65% predict-all baseline)
- **Most reliable fades:** Aaron Nesmith (100%, 7 events), Andrew Nembhard (100%, 5 events), Cameron Johnson (100%, 4 events)
- **Least reliable fades:** Cory Joseph (20%, 10 events), Cedi Osman (28.6%, 7 events), Quentin Grimes (30%, 10 events)

### Confounder analysis

| Confounder | Effect on regression |
|-----------|----------------------|
| Rest days (back-to-back vs rested) | None — 1.66 vs 1.72 avg regression |
| Opponent defense (elite vs bad) | None — 1.57 vs 1.56 |
| Home vs away | Slight — 1.53 home vs 1.70 away |
| Minutes trend (stable vs increasing) | **Significant** — stable minutes cleanest signal |
| Recent blowout context | None |

### Cross-stat performance (2024-25 season)

| Stat | Events | Backtest Win Rate | Regression Signal |
|------|--------|-------------------|-------------------|
| FG3M (3-pointers) | 710 | **73.9%** | Strongest |
| PTS (points) | 933 | 66.2% | Moderate |
| PR (pts+reb) | 943 | 62.9% | Moderate |
| PA (pts+ast) | 967 | 63.5% | Moderate |
| PRA (pts+reb+ast) | 1,000 | 60.9% | Weakest |

---

## How the model works

1. **Pull NBA role players** (15–32 MPG, 30+ games played, 2+ 3PA/game)
2. **Compute season baseline** — true skill level; uses playoffs if 5+ games, blends current/previous seasons if early in year
3. **Compute rolling 3-game average** — recent form that triggered the flag
4. **Calculate z-score** — how unusual is recent form relative to baseline?
5. **Flag hot streaks** (z ≥ 1.0) and cold streaks (z ≤ −1.0)
6. **Score regression probability** — GradientBoosting model (FG3M picks only)
7. **Cross-reference injury report** — OUT players removed, GTD/DOUBTFUL flagged
8. **Compare fair line (baseline) vs DraftKings line** — compute edge gap and bet threshold

### The proxy backtest methodology

Without access to historical bookmaker lines, the model is validated by comparing:
- **The model's "fair line"** — the player's season baseline (true skill)
- **The naive line** — the rolling 3-game average during the hot streak

A pick **wins** if the actual game outcome lands closer to the fair line than the naive line. This is a standard proxy methodology used in sports analytics research.

---

## Project structure

```
betting-the-regression/
├── daily_picks.py               # Daily prediction tool — grades yesterday, flags today, injury check
├── odds_compare.py              # Live DraftKings line comparison + auto-generates dashboard.html
├── train_model.py               # ML training pipeline (v2.2): GradientBoosting + XGBoost
├── hothandfade_v3.py            # Full multi-season backtest pipeline (cached; run once)
├── dashboard.html               # Auto-generated daily dashboard (open in browser)
├── model_regression.pkl         # Trained GradientBoosting model (FG3M, 1,408 training events)
├── model_features.json          # Feature list + training means for imputation
├── player_regression_rates.json # Per-player historical regression rates (238 players)
├── model_validation_summary.csv # Model comparison table across all versions
└── *.csv                        # Backtest results, daily predictions, graded history
```

---

## Dashboard

`odds_compare.py` auto-generates `dashboard.html` and opens it in the browser after each run. The dashboard includes:

- **Game strip** — tonight's matchup, tip-off time, venue, series record
- **Injury report** — all injured players from both teams (OUT / DOUBTFUL / GTD / Returning), fresh from ESPN
- **Metric cards** — total picks, positive gaps, actionable count, model record
- **Per-stat performance table** — live win rates for FG3M / PTS / PRA vs backtest baselines
- **Pick cards** — tiered STRONG → MODERATE → WEAK; sorted by gap; injury flags inline; OUT picks dimmed
- **Probability pills** — GradientBoosting confidence score on FG3M picks: green ≥70%, amber 60–69%, grey <60%
- **Paper trading tracker** — place bets, grade outcomes, track bankroll with a live chart; undo/redo; bet history; all persisted in `localStorage`
- **Save snapshot button** — downloads the current dashboard (including paper bets) as `dashboard_snapshot_YYYY-MM-DD.html`
- **Print-friendly layout** — `Cmd+P` hides the paper trading section and prints only the picks in black and white

---

## Setup

```bash
# Requires Anaconda Python 3.12 for the ML model
conda install scikit-learn xgboost joblib pandas numpy
brew install libomp  # required for XGBoost on macOS
pip install nba_api
```

### Run the daily workflow

```bash
# Step 1 — generate today's picks (grades yesterday, checks injuries, saves predictions)
python3.12 daily_picks.py

# Step 2 — pull live DraftKings lines, generate dashboard, open in browser
python3.12 odds_compare.py
```

### Retrain the model

```bash
python3.12 train_model.py
```

### Reproduce the full backtest

```bash
python3.12 hothandfade_v3.py
```

---

## Limitations & roadmap

**Current limitations:**
- ML model trained on FG3M only — PTS/PRA picks use rule-based threshold only (no v2 probability score)
- Player regression rates computed from 2 seasons of FG3M data only (some players have small samples)
- No streak length feature yet — requires raw game-log reprocessing to count consecutive hot games

**Roadmap:**
- Stat-specific models: train separate GradientBoosting models per stat type (PTS, PRA, etc.)
- Streak length + recency slope features: how long has the streak lasted? is it accelerating?
- Bayesian shrinkage on player regression rates: pull sparse samples toward the global mean
- Automated scheduling: launchd on Mac to run the pipeline automatically at 5 PM on game days (see [AUTOMATION.md](AUTOMATION.md))

---

## Tech stack

- **Python 3.12** (Anaconda environment)
- **pandas** — data manipulation, CSV tracking
- **scikit-learn** — GradientBoosting classifier, StandardScaler, metrics
- **XGBoost** — gradient boosting comparison model
- **joblib** — model serialisation (pkl files)
- **nba_api** — live NBA stats (player game logs, rosters, scoreboards)
- **ESPN Core API** (`sports.core.api.espn.com`) — DraftKings prop lines, no auth required
- **ESPN Site API** (`site.api.espn.com`) — injury report, no auth required
- **curl** — all ESPN API calls (SSL compatibility on macOS)
- **Chart.js** (CDN) — bankroll history chart in the dashboard

---

## Author

**Abhi Murmu** — sports betting analytics portfolio project, May–June 2026.

Public repo: [github.com/abhisaradev/betting-the-regression](https://github.com/abhisaradev/betting-the-regression)
