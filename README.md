# Hot Hand Fader

**A mean reversion model for NBA player props that exploits short-term hot streak overreactions.**

Tested across 3 NBA seasons (2022-23 through 2024-25) with **73.9% accuracy** on a 2,118-event proxy backtest.

---

## The thesis

Public bettors heavily react to recent player performance — a 3-game hot stretch makes a role player's prop line spike higher than their true skill warrants. This model identifies when a player's recent rolling average significantly exceeds their season baseline (z-score > 1.0) and predicts they'll regress in their next game.

**The strategy: fade hot streaks by betting UNDER the inflated line.**

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

### Cross-stat expansion (2024-25, single season)

The model was tested on additional stats beyond 3-point makes:

| Stat | Events | Win Rate |
|------|--------|----------|
| 3PM | 710 | 73.9% |
| Points | 933 | 66.2% |
| Points + Assists | 967 | 63.5% |
| Points + Rebounds | 943 | 62.9% |
| PRA combo | 1000 | 60.9% |

Higher-variance stats (threes) show the strongest mean reversion effect.

---

## Project structure
hot-hand-fader/
├── hothandfade_v3.py            # Full backtest pipeline with cached steps
├── daily_picks.py               # Production tool for daily predictions
├── 2026wcfgame7.py              # Real-world test: 2026 WCF Game 7 (SAS vs OKC)
├── nbagame7analysis.py          # Refined Game 7 analysis with playoff baselines
└── *.csv                        # Backtest results and analysis outputs

---

## How the model works

1. **Pull NBA role players** (15-32 MPG, 30+ games played, 2+ 3PA/game)
2. **Compute season baseline** — true skill level from all season data
3. **Compute rolling 3-game average** — recent form
4. **Calculate z-score** — how unusual is recent form relative to baseline?
5. **Flag hot streaks** when z-score > 1.0
6. **Predict regression** — next game will revert toward baseline
7. **Compare your fair line (baseline) vs naive line (rolling average) vs actual outcome**

### The proxy backtest methodology

Without access to historical bookmaker prop lines, the model is validated by comparing two candidate lines:

- **The model's "fair line"** — the player's season baseline (true skill)
- **The naive line** — the recent rolling 3-game average (what casual bettors would use)

For each historical hot streak event, the model "wins" if the actual game outcome lands closer to the fair line than the naive line. This is a defensible academic methodology used in sports analytics research.

---

## Real-world test: 2026 Western Conference Finals Game 7

The model was applied to a live, high-stakes game (Spurs vs Thunder, Game 7) before tipoff. Of the recommendations on players who actually played, **4 of 5 hit correctly**, including the highest-confidence call (Kenrich Williams UNDER hit dramatically — model fair line of 3.2 points vs actual 2 points scored on minimal minutes).

The model correctly flagged its own limitations — players with "limited playoff sample" warnings (Carlson, Barnhizer) all DNP'd, confirming the model's caveats were appropriate.

---

## Limitations & future improvements

- **Tested on regular season data primarily** — playoff dynamics differ (tighter rotations, defensive game-planning)
- **No real bookmaker line validation** — proxy methodology vs actual sportsbook line data
- **No injury data integration** — the v1.2 roadmap includes injury check and GTD flagging
- **Single model (rule-based)** — future versions will add logistic regression and XGBoost for confidence calibration

---

## Setup

```bash
pip install -r requirements.txt
```

### Run the daily prediction tool

```bash
python daily_picks.py
```

This will:
1. Auto-grade yesterday's predictions against actual outcomes
2. Identify today's hot/cold streak candidates across all NBA and WNBA games
3. Save predictions to `daily_predictions.csv` for future grading

### Reproduce the multi-season backtest

```bash
python hothandfade_v3.py
```

This runs the full 3-season backtest, caches results to CSVs, and outputs the multi-season summary.

---

## Tech stack

- **Python 3.12**
- **pandas** for data manipulation
- **nba_api** for live NBA stats data
- Statistical methodology (z-score thresholds, paired comparison testing)

---

## Author

**Abhi Murmu** — built as a sports betting analytics portfolio project, May 2026.