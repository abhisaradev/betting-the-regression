# Betting the Regression

**A mean reversion model for NBA player props that exploits short-term hot and cold streak overreactions.**

Tested across 3 NBA seasons (2022-23 through 2024-25) with **73.9% accuracy** on a 2,118-event proxy backtest.

---

## The thesis

Public bettors heavily react to recent player performance — a 3-game hot stretch makes a role player's prop line spike higher than their true skill warrants. This model identifies when a player's recent rolling average significantly exceeds their season baseline (z-score > 1.0) and predicts they'll regress in their next game.

**The strategy: fade hot streaks by betting UNDER the inflated line.**

---

## v1.2 feature set

| Feature | Detail |
|---------|--------|
| **Tiered picks** | STRONG / MODERATE / WEAK by z-score confidence (|z| ≥ 1.2 / 1.0 / 0.85) |
| **Live DraftKings lines** | Via ESPN Core API — no auth required, no third-party keys |
| **Bet threshold logic** | BET UNDER / BET OVER / Watch / No edge — each pick has a minimum-gap trigger |
| **Auto-generated dashboard** | Self-contained `dashboard.html` opens in browser after each run |
| **Injury check** | ESPN injury API: OUT players removed, GTD/DOUBTFUL/Returning flagged |
| **Season transition logic** | Auto-selects playoff / current / blended / previous baseline by sample size |
| **Auto-grading** | Previous day's predictions graded against actual stats on each run |
| **Per-stat tracking** | FG3M, PTS, PRA tracked separately — win rates logged by stat |
| **NBA + WNBA** | Covers both leagues every day |
| **Paper trading** | In-browser tracker with undo/redo, bankroll chart, and bet history |

---

## Key findings

### Multi-season backtest (2022-25, 558 player-seasons, 2,118 hot streak events)

| Season | Players | Hot Events | Win Rate | Avg Distance (Model) | Avg Distance (Naive) |
|--------|---------|------------|----------|----------------------|----------------------|
| 2022-23 | 186 | 678 | 74.2% | 1.03 threes | 1.84 threes |
| 2023-24 | 173 | 730 | 72.6% | 1.04 threes | 1.84 threes |
| 2024-25 | 199 | 710 | 73.9% | 1.02 threes | 1.82 threes |

The model beat a naive rolling-average baseline by nearly 2-to-1 across every season tested, with average prediction error of ~1.0 threes per game vs ~1.8 for the naive baseline.

### Methodology validation

- **92-98% of role players** show the regression pattern across all 3 seasons
- **Normal-game control group** showed essentially zero regression (-0.09 avg) — confirming the effect is real and not statistical artifact
- **Filtered subset** (high-volume + stable minutes + away games) hits 78.9% accuracy

### Confounder analysis

The model was tested against potential confounders:

| Confounder | Effect on regression |
|-----------|----------------------|
| Rest days (back-to-back vs rested) | None — 1.66 vs 1.72 |
| Opponent defense (elite vs bad) | None — 1.57 vs 1.56 |
| Home vs away | Slight (1.53 home vs 1.70 away) |
| Minutes trend (stable vs increasing) | **Significant** — stable minutes show the cleanest signal |
| Recent blowout context | None |

**Key takeaway: stable minutes is the critical filter.** Players whose minutes are trending up may be experiencing real role expansion rather than statistical noise.

### Cross-stat performance (2024-25 season)

| Stat | Events | Backtest Win Rate | Signal Strength |
|------|--------|-------------------|-----------------|
| FG3M (3-pointers) | 710 | 73.9% | Strongest |
| PTS (points) | 933 | 66.2% | Moderate |
| PRA (pts+reb+ast) | 1,000 | 60.9% | Weakest |

Higher-variance stats (threes) show the strongest mean reversion effect. The dashboard tracks live win rates per stat alongside these backtest baselines.

---

## Project structure

```
betting-the-regression/
├── daily_picks.py               # Production tool: grades yesterday, flags today, injury check
├── odds_compare.py              # Pulls DraftKings lines via ESPN, generates dashboard.html
├── hothandfade_v3.py            # Full multi-season backtest pipeline (cached)
├── dashboard.html               # Auto-generated daily dashboard (open in browser)
├── 2026wcfgame7.py              # Real-world test: 2026 WCF Game 7 (SAS vs OKC)
├── nbagame7analysis.py          # Refined Game 7 analysis with playoff baselines
└── *.csv                        # Backtest results, daily predictions, performance log
```

---

## How the model works

1. **Pull NBA role players** (15-32 MPG, 30+ games played)
2. **Compute season baseline** — true skill level from all season data (playoff / current / blended / previous, whichever has sufficient sample)
3. **Compute rolling 3-game average** — recent form
4. **Calculate z-score** — how unusual is recent form relative to baseline?
5. **Flag hot streaks** (z ≥ 1.0) and cold streaks (z ≤ -1.0)
6. **Predict regression** — next game will revert toward baseline
7. **Cross-reference injury report** — OUT players removed, GTD/DOUBTFUL flagged
8. **Compare fair line (baseline) vs DraftKings line** — compute edge gap and bet threshold

### The proxy backtest methodology

Without access to historical bookmaker prop lines, the model is validated by comparing two candidate lines:

- **The model's "fair line"** — the player's season baseline (true skill)
- **The naive line** — the recent rolling 3-game average (what casual bettors would use)

For each historical hot streak event, the model "wins" if the actual game outcome lands closer to the fair line than the naive line. This is a defensible academic methodology used in sports analytics research.

---

## Dashboard

`odds_compare.py` auto-generates `dashboard.html` and opens it in the browser after each run. The dashboard includes:

- **Game strip** — tonight's matchup, tip-off time, venue, series record
- **Injury report** — all injured players from both teams (OUT / DOUBTFUL / GTD / Returning), pulled fresh from ESPN every run
- **Metric cards** — total picks, positive gaps, actionable count, model record
- **Per-stat performance table** — live win rates for FG3M / PTS / PRA vs backtest baselines
- **Pick cards** — tiered STRONG → MODERATE → WEAK, sorted by gap size; injury flags shown inline; OUT picks dimmed with "Skip" button
- **Paper trading tracker** — place bets, grade outcomes, track bankroll with a live chart; undo/redo support; bet history with search; all persisted in `localStorage`

---

## Real-world test: 2026 Western Conference Finals Game 7

The model was applied to a live, high-stakes game (Spurs vs Thunder, Game 7) before tipoff. Of the recommendations on players who actually played, **4 of 5 hit correctly**, including the highest-confidence call (Kenrich Williams UNDER hit dramatically — model fair line of 3.2 points vs actual 2 points scored on minimal minutes).

The model correctly flagged its own limitations — players with "limited playoff sample" warnings (Carlson, Barnhizer) all DNP'd, confirming the model's caveats were appropriate.

---

## Limitations & future improvements

- **Tested on regular season data primarily** — playoff dynamics differ (tighter rotations, defensive game-planning)
- **No real bookmaker line validation** — proxy methodology vs actual sportsbook line data; live edge tracking is early
- **Single model (rule-based)** — future versions will add logistic regression and XGBoost for confidence calibration
- **Cold signals historically weaker** — OVER picks (cold streak fades) run at ~44% vs ~74% for UNDER picks (hot streak fades)

---

## Setup

```bash
pip install -r requirements.txt
```

### Run the daily workflow

```bash
# Step 1 — generate today's picks (grades yesterday, checks injuries, saves predictions)
python3.11 daily_picks.py

# Step 2 — pull live DraftKings lines, generate dashboard, open in browser
python3.11 odds_compare.py
```

`daily_picks.py` will:
1. Auto-grade yesterday's predictions against actual outcomes
2. Fetch the live ESPN injury report — OUT players are removed, GTD/DOUBTFUL flagged
3. Identify today's hot/cold streak candidates across all NBA and WNBA games
4. Save predictions to `daily_predictions.csv` for future grading

`odds_compare.py` will:
1. Pull live prop lines from DraftKings via ESPN's public API (no auth required)
2. Fetch the injury report for tonight's teams — annotates pick cards and populates the injury section
3. Fuzzy-match flagged players, compute edge gaps, rank by tier
4. Write `dashboard.html` and open it in the browser automatically

### Reproduce the multi-season backtest

```bash
python hothandfade_v3.py
```

This runs the full 3-season backtest, caches results to CSVs, and outputs the multi-season summary.

---

## Tech stack

- **Python 3.11**
- **pandas** for data manipulation and CSV tracking
- **nba_api** for live NBA stats data
- **ESPN Core API** (`sports.core.api.espn.com`) — DraftKings prop lines, no auth
- **ESPN Site API** (`site.api.espn.com`) — injury report, no auth
- **curl** for all ESPN calls (SSL compatibility)
- **Chart.js** (CDN) for the bankroll history chart in the dashboard
- Statistical methodology: z-score thresholds, proxy backtest, paired comparison

---

## Author

**Abhi Murmu** — built as a sports betting analytics portfolio project, May–June 2026.

> **Note:** The GitHub repository is still named `hot-hand-fader`. To rename it to `betting-the-regression`, go to **github.com → repo → Settings → Repository name** → type `betting-the-regression` → click Rename.
