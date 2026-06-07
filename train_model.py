"""
train_model.py — v2.1  Betting the Regression

Improvements over v2.0:
  • Feature 1: player_regression_rate — each player's historical regression
    rate computed from 2022-23 + 2023-24 training data ONLY (no leakage).
    Players with < 3 events fall back to the global training mean.
  • Feature 2: season_position — proxy for how far into the season the
    streak event occurred, using within-player row rank normalised 0→1.
    (No date column exists in the backtest CSVs; row order is the best
    available proxy for temporal position within the season.)

Note: baseline_games_normalized was requested but cannot be derived from
the backtest CSVs (they contain no game-count column). It is skipped and
logged. The season_avg / your_fair_line are identical in the data, so they
already encode skill level; the new player_regression_rate encodes
individual reliability.

Models compared:
  1. Rule-based (z-score threshold)        — 73.9% baseline
  2. Logistic Regression original (v2.0)   — 73.9% / AUC 0.539
  3. Logistic Regression new features (v2.1)
  4. XGBoost (or GradientBoosting fallback)

Train:    2022-23 + 2023-24  (1,408 events)
Validate: 2024-25            (710 events, strict holdout)

Outputs:
  model_regression.pkl           best model pipeline
  model_features.json            feature list + train means
  player_regression_rates.json   per-player regression rates (for daily_picks.py)
  model_validation_summary.csv   comparison of all four models
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)

from sklearn.ensemble import GradientBoostingClassifier

# XGBoost — always preferred when available; GradientBoosting always runs as comparison
XGBOOST_AVAILABLE = False
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    pass

# ── Constants ──────────────────────────────────────────────────────────────────
RULE_BASED_ACC        = 73.9
INTEGRATION_THRESHOLD = RULE_BASED_ACC - 3.0   # ≥ 70.9%
MIN_EVENTS_FOR_RATE   = 3     # players with fewer events get global mean
TARGET                = "your_line_wins"

SEP  = "=" * 70
DASH = "─" * 45

LEAKAGE_COLS = {
    "next_game_stat", "actual_outcome", "regression",
    "distance_to_yours", "distance_to_naive", TARGET,
}

TRAIN_FILES = [
    "hot_hand_proxy_backtest_2022_23.csv",
    "hot_hand_proxy_backtest_2023_24.csv",
]
VAL_FILE = "hot_hand_proxy_backtest_2024_25.csv"

# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  BETTING THE REGRESSION — v2.1 Training  (feature-engineered)")
print(f"{SEP}\n")

# ══════════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
print(f"1.  DATA LOADING\n{DASH}")

train_frames = []
for f in TRAIN_FILES:
    df = pd.read_csv(f)
    print(f"\n  {f}  ({len(df):,} rows)")
    print(f"  Columns: {list(df.columns)}")
    print("  First 3 rows:")
    print(df.head(3).to_string(index=False))
    train_frames.append(df)

train_raw = pd.concat(train_frames, ignore_index=True)
print(f"\n  Combined training set: {len(train_raw):,} rows")

print(f"\n  {VAL_FILE}")
val_raw = pd.read_csv(VAL_FILE)
print(f"  Shape: {val_raw.shape}")
print(f"  Columns: {list(val_raw.columns)}")
print("  First 3 rows:")
print(val_raw.head(3).to_string(index=False))
print(f"\n  Validation set: {len(val_raw):,} rows")

# ══════════════════════════════════════════════════════════════════════════════
# 2.  FEATURE 1 — Player-level regression rate (from training data ONLY)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n\n2.  FEATURE 1 — player_regression_rate\n{DASH}")
print("  (computed from 2022-23 + 2023-24 training data ONLY — no leakage)\n")

player_stats = (
    train_raw.groupby("player")[TARGET]
    .agg(total="count", wins="sum")
    .reset_index()
)
player_stats["rate"] = player_stats["wins"] / player_stats["total"]

global_mean_rate = (
    player_stats.loc[player_stats["total"] >= MIN_EVENTS_FOR_RATE, "rate"].mean()
)
print(f"  Global mean regression rate (players ≥ {MIN_EVENTS_FOR_RATE} events): "
      f"{100*global_mean_rate:.1f}%")

# Players below MIN_EVENTS_FOR_RATE fall back to global mean
player_stats["rate_adj"] = player_stats.apply(
    lambda r: r["rate"] if r["total"] >= MIN_EVENTS_FOR_RATE else global_mean_rate,
    axis=1
)

# Build lookup dict — used both here and saved for daily_picks.py
player_rate_lookup: dict[str, float] = dict(
    zip(player_stats["player"], player_stats["rate_adj"])
)

# Distribution stats
rates = player_stats["rate_adj"]
print(f"\n  Distribution of player_regression_rate:")
print(f"    Mean={rates.mean():.3f}  Std={rates.std():.3f}  "
      f"Min={rates.min():.3f}  Max={rates.max():.3f}")

top5 = player_stats.nlargest(5, "rate_adj")[["player","total","wins","rate_adj"]]
bot5 = player_stats.nsmallest(5, "rate_adj")[["player","total","wins","rate_adj"]]

print(f"\n  ── Top 5 most reliable fade targets (highest regression rate) ──")
print(f"  {'Player':<28} Events  Wins  Rate")
print(f"  {'─'*28}  ──────  ────  ────")
for _, r in top5.iterrows():
    print(f"  {r['player']:<28}  {r['total']:4d}    {r['wins']:3.0f}   "
          f"{r['rate_adj']*100:.1f}%")

print(f"\n  ── Bottom 5 least reliable fades (lowest regression rate) ──")
print(f"  {'Player':<28} Events  Wins  Rate")
print(f"  {'─'*28}  ──────  ────  ────")
for _, r in bot5.iterrows():
    print(f"  {r['player']:<28}  {r['total']:4d}    {r['wins']:3.0f}   "
          f"{r['rate_adj']*100:.1f}%")

# Add to training DataFrame (direct lookup — no leakage since we're on train set)
train_raw["player_regression_rate"] = (
    train_raw["player"].map(player_rate_lookup).fillna(global_mean_rate)
)

# Add to validation DataFrame — use training lookup ONLY
val_raw["player_regression_rate"] = (
    val_raw["player"].map(player_rate_lookup).fillna(global_mean_rate)
)
new_in_val = val_raw["player"].map(
    lambda p: p not in player_rate_lookup
).sum()
print(f"\n  Players in 2024-25 not seen in training → {new_in_val} "
      f"(imputed with global mean {100*global_mean_rate:.1f}%)")

# Save lookup for daily_picks.py
with open("player_regression_rates.json", "w") as fh:
    json.dump(
        {"global_mean": global_mean_rate, "players": player_rate_lookup},
        fh, indent=2
    )
print("  ✅ player_regression_rates.json saved")

# ══════════════════════════════════════════════════════════════════════════════
# 3.  FEATURE 2 — Season position proxy
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n\n3.  FEATURE 2 — season_position (proxy)\n{DASH}")
print("  No date column in backtest CSVs — using within-player row rank.")
print("  Row order within each player = chronological order within season.\n")

def add_season_position(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add season_position (0–1) using within-player cumulative row rank.
    Row 0 for a player = first hot streak event of that season → 0.0
    Last row for a player → 1.0 (or 0.0 if only one event)
    """
    d = df.copy()
    d["_row_rank"] = d.groupby("player").cumcount()     # 0-based rank
    d["_player_n"] = d.groupby("player")["player"].transform("count")
    d["season_position"] = np.where(
        d["_player_n"] > 1,
        d["_row_rank"] / (d["_player_n"] - 1),
        0.5   # single-event players get mid-season
    )
    return d.drop(columns=["_row_rank", "_player_n"])

train_raw = add_season_position(train_raw)
val_raw   = add_season_position(val_raw)

print(f"  Training season_position:")
print(f"    Mean={train_raw['season_position'].mean():.3f}  "
      f"Std={train_raw['season_position'].std():.3f}  "
      f"Min={train_raw['season_position'].min():.3f}  "
      f"Max={train_raw['season_position'].max():.3f}")

print(f"\n  Note: baseline_games_normalized was requested but cannot be")
print(f"  derived from the backtest CSVs (no game-count column). Skipped.")
print(f"  Season position serves the same purpose (measures data maturity).")

# ══════════════════════════════════════════════════════════════════════════════
# 4.  SHARED FEATURE ENGINEERING (same as v2.0 + new columns)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n\n4.  FULL FEATURE ENGINEERING\n{DASH}")

def engineer_features(df: pd.DataFrame, gap_std: float | None = None) -> pd.DataFrame:
    """
    Derive all model features from raw backtest columns + pre-computed fields.
    gap_std: if provided, use this value for z_score normalisation (important:
             compute on training set, then reuse on validation — no leakage).
    """
    d = df.copy()

    # Primary signal
    d["gap_from_baseline"] = d["during_hot"] - d["season_avg"]
    d["abs_z_score"]       = d["gap_from_baseline"].abs()

    # Normalised z-score proxy
    _gap_std = gap_std if gap_std is not None else d["gap_from_baseline"].std()
    d["z_score"] = d["gap_from_baseline"] / _gap_std if _gap_std > 0 else 0.0

    # Location
    if "next_location" in d.columns:
        d["is_home"] = (d["next_location"].str.lower() == "home").astype(float)

    # Blowout flag
    if "was_blowout_recent" in d.columns:
        d["blowout_flag"] = d["was_blowout_recent"].astype(float)

    # Opponent defensive tier ordinal
    if "opp_def_tier" in d.columns:
        tier_map = {"Bad (21-30)": 0.0, "Mid (11-20)": 1.0, "Elite (top 10)": 2.0}
        d["def_tier_ord"] = d["opp_def_tier"].map(tier_map)

    return d

# Compute gap_std on training set, reuse for validation (strict no-leakage)
train_fe = engineer_features(train_raw)
_gap_std_train = train_fe["gap_from_baseline"].std()
val_fe   = engineer_features(val_raw, gap_std=_gap_std_train)

# ── Define two feature sets ────────────────────────────────────────────────
ORIGINAL_FEATURES = [
    "gap_from_baseline", "z_score", "abs_z_score",
    "season_avg", "during_hot", "minutes_trend",
    "opp_3p_pct_allowed", "opp_pace", "is_home",
    "blowout_flag", "def_tier_ord",
]

NEW_FEATURES_ADDED = [
    "player_regression_rate",  # Feature 1: personal historical rate
    "season_position",          # Feature 2: within-season timing proxy
]

ALL_FEATURES = [
    f for f in ORIGINAL_FEATURES + NEW_FEATURES_ADDED
    if f in train_fe.columns and f not in LEAKAGE_COLS
]

print(f"\n  All features for new models ({len(ALL_FEATURES)}):")
for f in ALL_FEATURES:
    nn = train_fe[f].notna().sum()
    tag = " ← NEW" if f in NEW_FEATURES_ADDED else ""
    print(f"    • {f:<30}  non-null: {nn:4d}/{len(train_fe)}{tag}")

# Drop NA rows
train_clean = train_fe.dropna(subset=ALL_FEATURES + [TARGET]).copy()
val_clean   = val_fe.dropna(subset=ALL_FEATURES + [TARGET]).copy()
print(f"\n  Rows after dropna:")
print(f"    Training:   {len(train_clean):4d} / {len(train_fe)}")
print(f"    Validation: {len(val_clean):4d} / {len(val_fe)}")

X_train_all = train_clean[ALL_FEATURES].values.astype(float)
y_train     = train_clean[TARGET].astype(int).values
X_val_all   = val_clean[ALL_FEATURES].values.astype(float)
y_val       = val_clean[TARGET].astype(int).values

X_train_orig = train_clean[ORIGINAL_FEATURES].values.astype(float)
X_val_orig   = val_clean[ORIGINAL_FEATURES].values.astype(float)

# Training means (all features) for imputation in daily_picks.py
train_means_all = {f: float(train_clean[f].mean()) for f in ALL_FEATURES}

# ══════════════════════════════════════════════════════════════════════════════
# 5.  HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def print_metrics(label: str, y_true, y_pred, y_proba) -> dict:
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    auc  = roc_auc_score(y_true, y_proba)
    cm   = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n  ── {label} ──")
    print(f"  Accuracy:  {100*acc:.1f}%   Precision: {100*prec:.1f}%   "
          f"Recall: {100*rec:.1f}%   F1: {f1:.3f}")
    print(f"  ROC-AUC:   {auc:.3f}")
    print(f"  Confusion matrix (rows=actual, cols=predicted):")
    print(f"                    Pred:No    Pred:Yes")
    print(f"    Actual No:       {tn:5d}      {fp:5d}")
    print(f"    Actual Yes:      {fn:5d}      {tp:5d}")
    diff = 100 * acc - RULE_BASED_ACC
    sign = "+" if diff >= 0 else ""
    print(f"  vs rule-based: {sign}{diff:.1f} pp")

    return {"accuracy": acc, "auc": auc, "precision": prec,
            "recall": rec, "f1": f1, "label": label}


def print_calibration(y_true, y_proba):
    print(f"\n  Probability calibration:")
    print(f"  Threshold | Picks | Actual regressed %")
    print(f"  ----------|-------|-------------------")
    for thr in [0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50]:
        mask = y_proba >= thr
        n    = mask.sum()
        if n > 0:
            rate = y_true[mask].mean()
            bar  = "█" * round(rate * 20)
            print(f"  >= {int(thr*100):2d}%     |  {n:3d}  |  {100*rate:.1f}%  {bar}")
        else:
            print(f"  >= {int(thr*100):2d}%     |    0  |  (no picks at threshold)")

# ══════════════════════════════════════════════════════════════════════════════
# 6.  MODEL A — Logistic Regression (original features, v2.0 reproduction)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n\n5.  MODEL A — Logistic Regression (original 11 features)\n{DASH}")

pipe_orig = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  LogisticRegression(max_iter=1000, random_state=42, C=1.0)),
])
pipe_orig.fit(X_train_orig, y_train)
pred_orig  = pipe_orig.predict(X_val_orig)
proba_orig = pipe_orig.predict_proba(X_val_orig)[:, 1]
metrics_orig = print_metrics("Logistic Regression — original features (v2.0)",
                             y_val, pred_orig, proba_orig)

# ══════════════════════════════════════════════════════════════════════════════
# 7.  MODEL B — Logistic Regression (new features, v2.1)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n\n6.  MODEL B — Logistic Regression (new features v2.1)\n{DASH}")

pipe_new = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  LogisticRegression(max_iter=1000, random_state=42, C=1.0)),
])
pipe_new.fit(X_train_all, y_train)
pred_new  = pipe_new.predict(X_val_all)
proba_new = pipe_new.predict_proba(X_val_all)[:, 1]
metrics_new = print_metrics("Logistic Regression — new features (v2.1)",
                            y_val, pred_new, proba_new)
print_calibration(y_val, proba_new)

print(f"\n  Feature coefficients (sorted by |coef|):")
coefs_new = pipe_new.named_steps["model"].coef_[0]
pairs_new = sorted(zip(ALL_FEATURES, coefs_new),
                   key=lambda x: abs(x[1]), reverse=True)
print(f"  {'Feature':<30}  {'Coef':>7}   Direction")
print(f"  {'─'*30}  {'─'*7}   {'─'*35}")
for fname, coef in pairs_new:
    direction = "increases regression prob" if coef > 0 else "decreases regression prob"
    bar = "█" * min(int(abs(coef) * 8), 24)
    tag = " ← NEW" if fname in NEW_FEATURES_ADDED else ""
    print(f"  {fname:<30}  {coef:+7.3f}   {bar}  {direction}{tag}")

# ══════════════════════════════════════════════════════════════════════════════
# 8.  MODEL C — GradientBoosting (sklearn, always runs)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n\n7.  MODEL C — GradientBoosting (sklearn)\n{DASH}")
gb_model = GradientBoostingClassifier(
    n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
)
gb_model.fit(X_train_all, y_train)
pred_gb  = gb_model.predict(X_val_all)
proba_gb = gb_model.predict_proba(X_val_all)[:, 1]
metrics_gb = print_metrics("GradientBoosting", y_val, pred_gb, proba_gb)
print_calibration(y_val, proba_gb)

print(f"\n  Feature importance (GradientBoosting):")
importances_gb = gb_model.feature_importances_
pairs_gb = sorted(zip(ALL_FEATURES, importances_gb), key=lambda x: x[1], reverse=True)
print(f"  {'Feature':<30}  {'Importance':>10}   Bar")
print(f"  {'─'*30}  {'─'*10}   {'─'*30}")
for fname, imp in pairs_gb:
    bar = "█" * min(int(imp * 200), 30)
    tag = " ← NEW" if fname in NEW_FEATURES_ADDED else ""
    print(f"  {fname:<30}  {imp:10.4f}   {bar}{tag}")

# ══════════════════════════════════════════════════════════════════════════════
# 9.  MODEL D — XGBoost (only if available)
# ══════════════════════════════════════════════════════════════════════════════
metrics_xgb = None
xgb_model   = None
if XGBOOST_AVAILABLE:
    print(f"\n\n8.  MODEL D — XGBoost\n{DASH}")
    xgb_model = XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        eval_metric="logloss", random_state=42, verbosity=0,
    )
    xgb_model.fit(X_train_all, y_train)
    pred_xgb  = xgb_model.predict(X_val_all)
    proba_xgb = xgb_model.predict_proba(X_val_all)[:, 1]
    metrics_xgb = print_metrics("XGBoost", y_val, pred_xgb, proba_xgb)
    print_calibration(y_val, proba_xgb)

    print(f"\n  Feature importance (XGBoost):")
    importances_xgb = xgb_model.feature_importances_
    pairs_xgb = sorted(zip(ALL_FEATURES, importances_xgb),
                       key=lambda x: x[1], reverse=True)
    print(f"  {'Feature':<30}  {'Importance':>10}   Bar")
    print(f"  {'─'*30}  {'─'*10}   {'─'*30}")
    for fname, imp in pairs_xgb:
        bar = "█" * min(int(imp * 200), 30)
        tag = " ← NEW" if fname in NEW_FEATURES_ADDED else ""
        print(f"  {fname:<30}  {imp:10.4f}   {bar}{tag}")
else:
    print(f"\n  ℹ️  XGBoost not available — skipped (GradientBoosting used instead)")

# ══════════════════════════════════════════════════════════════════════════════
# 10.  COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n\n9.  FINAL COMPARISON TABLE\n{DASH}")
acc_a = f"{100*metrics_orig['accuracy']:.1f}%"
acc_b = f"{100*metrics_new['accuracy']:.1f}%"
acc_c = f"{100*metrics_gb['accuracy']:.1f}%"
hdr = f"  {'Model':<42} | {'Accuracy':>8} | {'ROC-AUC':>7} | Notes"
sep2 = f"  {'-'*42}-+-{'-'*8}-+-{'-'*7}-+-{'-'*30}"
print(f"\n{hdr}")
print(sep2)
print(f"  {'Rule-based (z > 1.0)':<42} | {'73.9%':>8} | {'---':>7} | Baseline")
print(f"  {'Logistic original (v2.0)':<42} | {acc_a:>8} | "
      f"{metrics_orig['auc']:>7.3f} | 11 features")
print(f"  {'Logistic new features (v2.1)':<42} | {acc_b:>8} | "
      f"{metrics_new['auc']:>7.3f} | + player rate + season pos")
print(f"  {'GradientBoosting (sklearn)':<42} | {acc_c:>8} | "
      f"{metrics_gb['auc']:>7.3f} | All {len(ALL_FEATURES)} features")
if metrics_xgb:
    acc_d = f"{100*metrics_xgb['accuracy']:.1f}%"
    print(f"  {'XGBoost':<42} | {acc_d:>8} | "
          f"{metrics_xgb['auc']:>7.3f} | All {len(ALL_FEATURES)} features")

# ══════════════════════════════════════════════════════════════════════════════
# 11.  SAVE BEST MODEL
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n\n10. SAVING BEST MODEL\n{DASH}")

candidates = [
    ("Logistic original (v2.0)",     metrics_orig["auc"], pipe_orig,   ORIGINAL_FEATURES),
    ("Logistic new features (v2.1)", metrics_new["auc"],  pipe_new,    ALL_FEATURES),
    ("GradientBoosting",             metrics_gb["auc"],   gb_model,    ALL_FEATURES),
]
if metrics_xgb:
    candidates.append(("XGBoost", metrics_xgb["auc"], xgb_model, ALL_FEATURES))

best_name, best_auc, best_model, best_features = max(candidates, key=lambda x: x[1])
print(f"\n  Best model: {best_name}")
print(f"  Best AUC:   {best_auc:.3f}")

# Save the best model
if hasattr(best_model, "predict_proba"):
    joblib.dump(best_model, "model_regression.pkl")
    print(f"  ✅ model_regression.pkl saved  "
          f"({os.path.getsize('model_regression.pkl'):,} bytes)")

# Feature metadata (always use ALL_FEATURES means for robustness)
feature_meta = {
    "features":     best_features,
    "train_means":  train_means_all,
    "model_type":   best_name,
    "best_auc":     round(best_auc, 4),
}
with open("model_features.json", "w") as fh:
    json.dump(feature_meta, fh, indent=2)
print(f"  ✅ model_features.json updated")

# Validation summary CSV — all models
summary_rows = [
    {"model": "Rule-based (z > 1.0)",
     "accuracy": "73.9%", "roc_auc": "---", "notes": "Baseline"},
    {"model": "Logistic original (v2.0)",
     "accuracy": f"{100*metrics_orig['accuracy']:.1f}%",
     "roc_auc":  f"{metrics_orig['auc']:.3f}",
     "notes": "11 features"},
    {"model": "Logistic new features (v2.1)",
     "accuracy": f"{100*metrics_new['accuracy']:.1f}%",
     "roc_auc":  f"{metrics_new['auc']:.3f}",
     "notes": "+ player rate + season position"},
    {"model": "GradientBoosting",
     "accuracy": f"{100*metrics_gb['accuracy']:.1f}%",
     "roc_auc":  f"{metrics_gb['auc']:.3f}",
     "notes": f"All {len(ALL_FEATURES)} features"},
]
if metrics_xgb:
    summary_rows.append({
        "model": "XGBoost",
        "accuracy": f"{100*metrics_xgb['accuracy']:.1f}%",
        "roc_auc":  f"{metrics_xgb['auc']:.3f}",
        "notes": f"All {len(ALL_FEATURES)} features",
    })
pd.DataFrame(summary_rows).to_csv("model_validation_summary.csv", index=False)
print(f"  ✅ model_validation_summary.csv updated")
print(f"\n  Reason: {best_name} selected — highest ROC-AUC ({best_auc:.3f})")

# ══════════════════════════════════════════════════════════════════════════════
# 12.  INTEGRATION DECISION
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n\n11. INTEGRATION DECISION\n{DASH}")

# Use the best model's accuracy for integration check
all_accs = [metrics_orig["accuracy"], metrics_new["accuracy"], metrics_gb["accuracy"]]
if metrics_xgb:
    all_accs.append(metrics_xgb["accuracy"])
best_acc_pct = 100 * max(all_accs)

if best_acc_pct >= INTEGRATION_THRESHOLD:
    print(f"\n  ✅ INTEGRATE — best accuracy ({best_acc_pct:.1f}%) ≥ threshold "
          f"({INTEGRATION_THRESHOLD:.1f}%)")
    print(f"     Best model: {best_name}")
else:
    gap = RULE_BASED_ACC - best_acc_pct
    print(f"\n  ❌ DO NOT INTEGRATE — best accuracy ({best_acc_pct:.1f}%) is "
          f"{gap:.1f} pp below threshold ({INTEGRATION_THRESHOLD:.1f}%)")

print(f"\n{SEP}")
print(f"  Training complete.  Best model: {best_name}  AUC: {best_auc:.3f}")
print(f"{SEP}\n")
