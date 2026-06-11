"""
train_model.py — Betting the Regression model training pipeline.
Trains GradientBoosting/XGBoost on NBA hot streak backtest data.
Train:    FG3M 2022-23 + 2023-24  (1,408 events, richest feature set)
Validate: ALL stats 2024-25 combined
          FG3M (710) + PTS (933) + PR (943) + PA (967) + PRA (1,000) = 4,553 events
Run:      python3.12 train_model.py
Outputs:  model_regression.pkl, model_features.json,
          player_regression_rates.json, model_validation_summary.csv

v2.2 changes vs v2.1:
  • Validation expanded to all 5 stat types (FG3M, PTS, PR, PA, PRA)
  • New feature: stat_encoded (ordinal — tells model which stat it's predicting)
  • Three models compared: GradientBoosting-100, XGBoost, GradientBoosting-200
  • Per-stat accuracy breakdown in validation
  • Honest reporting of FG3M→multistat generalisation gap

Column mismatch handling:
  FG3M proxy CSVs (18 cols) have contextual features: next_location,
  opp_def_tier, opp_3p_pct_allowed, opp_pace, minutes_trend, was_blowout_recent.
  Multistat CSVs (11 cols) are missing all of these.
  Strategy: contextual features are imputed with training means for multistat rows,
  so they contribute zero variance to the prediction (honest imputation).
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

XGBOOST_AVAILABLE = False
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    pass

# ── Constants ──────────────────────────────────────────────────────────────────
# FG3M-only rule-based baseline (used for training comparison)
FG3M_RULE_BASED_ACC    = 73.9
# Multi-stat rule-based baseline = "predict all as regression" on combined val set
# FG3M 525/710 + PTS 618/933 + PR 593/943 + PA 614/967 + PRA 609/1000
MULTISTAT_BASELINE_ACC = 100 * (525+618+593+614+609) / (710+933+943+967+1000)
INTEGRATION_THRESHOLD  = FG3M_RULE_BASED_ACC - 3.0   # ≥ 70.9% (on FG3M val)
MIN_EVENTS_FOR_RATE    = 3
TARGET                 = "your_line_wins"

# Stat ordinal encoding (ascending regression rate: FG3M highest → PRA lowest)
STAT_ENCODE = {"FG3M": 0, "PTS": 1, "PR": 2, "PA": 3, "PRA": 4}

SEP  = "=" * 72
DASH = "─" * 47

LEAKAGE_COLS = {
    "next_game_stat", "actual_outcome", "regression",
    "distance_to_yours", "distance_to_naive", TARGET,
    "stat",   # raw string version — stat_encoded is safe
}

# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  BETTING THE REGRESSION — v2.2 Training  (multi-stat validation)")
print(f"{SEP}\n")

# ══════════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING & COLUMN AUDIT
# ══════════════════════════════════════════════════════════════════════════════
print(f"1.  DATA LOADING & COLUMN AUDIT\n{DASH}")

FG3M_TRAIN_FILES = [
    "hot_hand_proxy_backtest_2022_23.csv",
    "hot_hand_proxy_backtest_2023_24.csv",
]
FG3M_VAL_FILE  = "hot_hand_proxy_backtest_2024_25.csv"
MULTISTAT_FILES = {
    "PTS": "hot_hand_multistat_PTS_2024_25.csv",
    "PR":  "hot_hand_multistat_PR_2024_25.csv",
    "PA":  "hot_hand_multistat_PA_2024_25.csv",
    "PRA": "hot_hand_multistat_PRA_2024_25.csv",
}

# ── Training data (FG3M 2022-23 + 2023-24) ───────────────────────────────────
print(f"\n  TRAINING FILES (FG3M only — richest feature set)")
train_frames = []
for f in FG3M_TRAIN_FILES:
    df = pd.read_csv(f)
    df["stat"] = "FG3M"
    print(f"\n  {f}  ({len(df):,} rows)")
    print(f"  Columns ({len(df.columns)}): {list(df.columns)}")
    print("  First 2 rows:")
    print(df.head(2).to_string(index=False))
    train_frames.append(df)

train_raw = pd.concat(train_frames, ignore_index=True)
print(f"\n  Combined training set: {len(train_raw):,} rows")

# ── Validation data ───────────────────────────────────────────────────────────
print(f"\n\n  VALIDATION FILES (all stats, 2024-25 only)")
print(f"  (Multistat CSVs have 11 cols vs FG3M's 18 — contextual features absent)")

# FG3M 2024-25
fg3m_val = pd.read_csv(FG3M_VAL_FILE)
fg3m_val["stat"] = "FG3M"
print(f"\n  {FG3M_VAL_FILE}  ({len(fg3m_val):,} rows)")
print(f"  Columns: {list(fg3m_val.columns)}")

val_frames = [fg3m_val]

# Multistat 2024-25
for stat_name, fpath in MULTISTAT_FILES.items():
    df = pd.read_csv(fpath)
    # stat column already in CSV — verify it matches expected name
    if "stat" in df.columns:
        unique_stats = df["stat"].unique().tolist()
        if len(unique_stats) == 1 and unique_stats[0] == stat_name:
            pass  # correct
        else:
            print(f"  ⚠️  {fpath}: stat column has unexpected values {unique_stats}")
    else:
        df["stat"] = stat_name
    print(f"\n  {fpath}  ({len(df):,} rows)")
    print(f"  Columns ({len(df.columns)}): {list(df.columns)}")
    print(f"  First 2 rows:")
    print(df.head(2).to_string(index=False))
    pos = int(df[TARGET].sum())
    print(f"  Class balance: {pos}/{len(df)} regressed ({100*pos/len(df):.1f}%)")
    val_frames.append(df)

val_raw = pd.concat(val_frames, ignore_index=True)
print(f"\n  Combined validation set: {len(val_raw):,} rows")

# Class balance per stat in validation
print(f"\n  Validation class balance by stat:")
print(f"  {'Stat':<8}  {'N':>5}  {'Pos':>5}  {'Rate':>6}  Bar")
print(f"  {'─'*8}  {'─'*5}  {'─'*5}  {'─'*6}  {'─'*20}")
for stat in ["FG3M", "PTS", "PR", "PA", "PRA"]:
    mask = val_raw["stat"] == stat
    n    = mask.sum()
    pos  = int(val_raw.loc[mask, TARGET].sum())
    rate = pos / n if n > 0 else 0
    bar  = "█" * round(rate * 20)
    print(f"  {stat:<8}  {n:5d}  {pos:5d}  {100*rate:5.1f}%  {bar}")

print(f"\n  Multi-stat rule-based baseline (predict-all-regression): "
      f"{MULTISTAT_BASELINE_ACC:.1f}%")

# Column mismatch report
fg3m_cols      = set(pd.read_csv(FG3M_TRAIN_FILES[0]).columns)
multistat_cols = set(pd.read_csv(list(MULTISTAT_FILES.values())[0]).columns)
only_in_fg3m   = fg3m_cols - multistat_cols - {"stat"}
only_in_multi  = multistat_cols - fg3m_cols - {"stat"}
print(f"\n  Column mismatch summary:")
print(f"  Cols in FG3M only (will be imputed w/ training means for multistat):")
for c in sorted(only_in_fg3m):
    print(f"    • {c}")
if only_in_multi:
    print(f"  Cols in multistat only:")
    for c in sorted(only_in_multi):
        print(f"    • {c}")

# ══════════════════════════════════════════════════════════════════════════════
# 2.  PLAYER REGRESSION RATE (from FG3M training data ONLY — no leakage)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n\n2.  PLAYER REGRESSION RATE\n{DASH}")
print("  Source: FG3M 2022-23 + 2023-24 training data ONLY")
print("  Applied to all stat types in validation via name lookup.\n")

player_stats = (
    train_raw.groupby("player")[TARGET]
    .agg(total="count", wins="sum")
    .reset_index()
)
player_stats["rate"] = player_stats["wins"] / player_stats["total"]

global_mean_rate = (
    player_stats.loc[player_stats["total"] >= MIN_EVENTS_FOR_RATE, "rate"].mean()
)
print(f"  Global mean rate (players ≥ {MIN_EVENTS_FOR_RATE} events): "
      f"{100*global_mean_rate:.1f}%")

player_stats["rate_adj"] = player_stats.apply(
    lambda r: r["rate"] if r["total"] >= MIN_EVENTS_FOR_RATE else global_mean_rate,
    axis=1
)
player_rate_lookup = dict(zip(player_stats["player"], player_stats["rate_adj"]))

# Event-count distribution
n5plus = (player_stats["total"] >= 5).sum()
n3to4  = ((player_stats["total"] >= 3) & (player_stats["total"] < 5)).sum()
n1to2  = (player_stats["total"] < 3).sum()
print(f"\n  Event-count distribution (training players):")
print(f"    ≥ 5 events (reliable estimate):  {n5plus} players")
print(f"    3-4 events (moderate):           {n3to4} players")
print(f"    1-2 events (use global mean):    {n1to2} players")

top5 = player_stats.nlargest(5, "rate_adj")[["player","total","wins","rate_adj"]]
bot5 = player_stats.nsmallest(5, "rate_adj")[["player","total","wins","rate_adj"]]

print(f"\n  Top 5 most reliable fades:")
print(f"  {'Player':<28} Events  Wins  Rate")
for _, r in top5.iterrows():
    print(f"  {r['player']:<28}  {r['total']:4d}    {r['wins']:3.0f}   "
          f"{r['rate_adj']*100:.1f}%")

print(f"\n  Bottom 5 least reliable fades:")
print(f"  {'Player':<28} Events  Wins  Rate")
for _, r in bot5.iterrows():
    print(f"  {r['player']:<28}  {r['total']:4d}    {r['wins']:3.0f}   "
          f"{r['rate_adj']*100:.1f}%")

# Apply to all DataFrames (validation lookup uses training rates only)
train_raw["player_regression_rate"] = (
    train_raw["player"].map(player_rate_lookup).fillna(global_mean_rate)
)
val_raw["player_regression_rate"] = (
    val_raw["player"].map(player_rate_lookup).fillna(global_mean_rate)
)

new_in_val = val_raw["player"].map(lambda p: p not in player_rate_lookup).sum()
print(f"\n  Players in validation not seen in training: {new_in_val} "
      f"(imputed with {100*global_mean_rate:.1f}%)")

with open("player_regression_rates.json", "w") as fh:
    json.dump({"global_mean": global_mean_rate, "players": player_rate_lookup}, fh, indent=2)
print("  ✅ player_regression_rates.json saved")

# ══════════════════════════════════════════════════════════════════════════════
# 3.  FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n\n3.  FEATURE ENGINEERING\n{DASH}")

def add_season_position(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'season_position' column: a number from 0.0 to 1.0 indicating where
    each hot-streak event falls within that player's season for that stat.

    0.0 = first event in the season, 1.0 = last event.  Mid-season events
    get values in between.  If a player only had one event, position defaults
    to 0.5 (middle of season).

    Why it matters: regression might be more reliable late in the season
    (larger sample, established baseline) vs. early (small sample).

    Groups by player AND stat so position resets when the stat type changes —
    e.g. a player's FG3M position doesn't bleed into their PTS position.
    """
    d = df.copy()
    # Group by player AND stat so position resets per stat type
    d["_rank"] = d.groupby(["player", "stat"]).cumcount()
    d["_n"]    = d.groupby(["player", "stat"])["player"].transform("count")
    d["season_position"] = np.where(
        d["_n"] > 1,
        d["_rank"] / (d["_n"] - 1),
        0.5
    )
    return d.drop(columns=["_rank", "_n"])

train_raw = add_season_position(train_raw)
val_raw   = add_season_position(val_raw)

def engineer_features(df: pd.DataFrame,
                      gap_std: float | None = None,
                      impute_means: dict | None = None) -> pd.DataFrame:
    """
    Build feature matrix from raw columns.
    gap_std:      normalise z-score using training gap std (no leakage).
    impute_means: dict of {col → mean} for cols absent in this DataFrame.
    """
    d = df.copy()

    # Core signal
    d["gap_from_baseline"] = d["during_hot"] - d["season_avg"]
    d["abs_z_score"]       = d["gap_from_baseline"].abs()

    _std = gap_std if gap_std is not None else d["gap_from_baseline"].std()
    d["z_score"] = d["gap_from_baseline"] / _std if _std > 0 else 0.0

    # Stat identity (new in v2.2)
    d["stat_encoded"] = d["stat"].map(STAT_ENCODE).fillna(0).astype(float)

    # Context — present in FG3M only; imputed for multistat
    if "next_location" in d.columns:
        d["is_home"] = (d["next_location"].str.lower() == "home").astype(float)
    elif impute_means and "is_home" in impute_means:
        d["is_home"] = impute_means["is_home"]

    if "was_blowout_recent" in d.columns:
        d["blowout_flag"] = d["was_blowout_recent"].astype(float)
    elif impute_means and "blowout_flag" in impute_means:
        d["blowout_flag"] = impute_means["blowout_flag"]

    if "opp_def_tier" in d.columns:
        tier_map = {"Bad (21-30)": 0.0, "Mid (11-20)": 1.0, "Elite (top 10)": 2.0}
        d["def_tier_ord"] = d["opp_def_tier"].map(tier_map)
    elif impute_means and "def_tier_ord" in impute_means:
        d["def_tier_ord"] = impute_means["def_tier_ord"]

    for col in ["minutes_trend", "opp_3p_pct_allowed", "opp_pace"]:
        if col not in d.columns and impute_means and col in impute_means:
            d[col] = impute_means[col]

    # FIX: after all if/elif derivations, fillna any remaining NaN in contextual
    # cols.  When FG3M + multistat CSVs are concatenated, columns like
    # was_blowout_recent / opp_def_tier / next_location EXIST in the combined
    # DataFrame (from FG3M rows) but are NaN for multistat rows.  The
    # if-branch above fires (column present), derives NaN for multistat rows,
    # and the elif-impute branch is skipped.  This pass catches those NaNs.
    if impute_means:
        for col in ["minutes_trend", "opp_3p_pct_allowed", "opp_pace",
                    "is_home", "blowout_flag", "def_tier_ord"]:
            if col in d.columns and d[col].isna().any():
                d[col] = d[col].fillna(impute_means[col])
            elif col not in d.columns and col in impute_means:
                d[col] = impute_means[col]

    return d

# First pass: compute training features to get gap_std and training means
train_fe_pass1 = engineer_features(train_raw)
_gap_std_train = train_fe_pass1["gap_from_baseline"].std()

# All context features are present in FG3M training — compute actual means
CONTEXT_COLS = ["minutes_trend", "opp_3p_pct_allowed", "opp_pace",
                "is_home", "blowout_flag", "def_tier_ord"]
_train_context_means = {}
for col in CONTEXT_COLS:
    if col in train_fe_pass1.columns:
        _train_context_means[col] = float(train_fe_pass1[col].mean())

# Second pass on validation: apply training gap_std + impute missing context
train_fe = engineer_features(train_raw, gap_std=_gap_std_train)
val_fe   = engineer_features(val_raw, gap_std=_gap_std_train,
                              impute_means=_train_context_means)

# ── Feature list ──────────────────────────────────────────────────────────────
ALL_FEATURES = [
    "gap_from_baseline",
    "z_score",
    "abs_z_score",
    "season_avg",
    "during_hot",
    "stat_encoded",           # NEW in v2.2
    "player_regression_rate",
    "season_position",
    "minutes_trend",
    "opp_3p_pct_allowed",
    "opp_pace",
    "is_home",
    "blowout_flag",
    "def_tier_ord",
]
NEW_FEATURES_V22 = ["stat_encoded"]
IMPUTED_IN_MULTISTAT = ["minutes_trend", "opp_3p_pct_allowed", "opp_pace",
                         "is_home", "blowout_flag", "def_tier_ord"]

# Verify all features present after engineering
ALL_FEATURES = [f for f in ALL_FEATURES
                if f in train_fe.columns and f not in LEAKAGE_COLS]

print(f"\n  Features ({len(ALL_FEATURES)}):")
for f in ALL_FEATURES:
    nn_tr = train_fe[f].notna().sum()
    nn_vl = val_fe[f].notna().sum() if f in val_fe.columns else 0
    tag = ""
    if f in NEW_FEATURES_V22:         tag = " ← NEW v2.2"
    elif f in IMPUTED_IN_MULTISTAT:   tag = " ← imputed in multistat rows"
    print(f"    • {f:<28}  train: {nn_tr:4d}/{len(train_fe)}  "
          f"val: {nn_vl:4d}/{len(val_fe)}{tag}")

# Build matrices
train_clean = train_fe.dropna(subset=ALL_FEATURES + [TARGET]).copy()
val_clean   = val_fe.dropna(subset=ALL_FEATURES + [TARGET]).copy()
print(f"\n  After dropna:")
print(f"    Training:   {len(train_clean):4,d} / {len(train_fe)}")
print(f"    Validation: {len(val_clean):4,d} / {len(val_fe)}")

X_train = train_clean[ALL_FEATURES].values.astype(float)
y_train = train_clean[TARGET].astype(int).values
X_val   = val_clean[ALL_FEATURES].values.astype(float)
y_val   = val_clean[TARGET].astype(int).values

# Training means for all features (used for imputation in daily_picks.py)
train_means = {f: float(train_clean[f].mean()) for f in ALL_FEATURES}

# Keep stat column in val for per-stat breakdown
val_stat_col = val_clean["stat"].values

# ══════════════════════════════════════════════════════════════════════════════
# 4.  HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def print_metrics(label, y_true, y_pred, y_proba, stat_col=None,
                  baseline_acc=None) -> dict:
    """
    Print a full model evaluation report and return the key metrics as a dict.

    label        — name to show in the report header (e.g. "GradientBoosting-100")
    y_true       — array of ground-truth labels (0=no regression, 1=regression)
    y_pred       — array of model predictions
    y_proba      — array of predicted probabilities for class 1 (used for AUC)
    stat_col     — optional array of stat names (FG3M / PTS / …) — when provided,
                   also prints a per-stat accuracy breakdown table
    baseline_acc — what "predict-all-regression" gets you (shown for comparison)

    Returns dict with keys: accuracy, auc, precision, recall, f1, label.
    """
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    auc  = roc_auc_score(y_true, y_proba)
    cm   = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    ref = baseline_acc if baseline_acc is not None else MULTISTAT_BASELINE_ACC

    print(f"\n  ── {label} ──")
    print(f"  Accuracy:  {100*acc:.1f}%   Precision: {100*prec:.1f}%   "
          f"Recall: {100*rec:.1f}%   F1: {f1:.3f}")
    print(f"  ROC-AUC:   {auc:.3f}")
    print(f"  Confusion matrix (rows=actual, cols=predicted):")
    print(f"                    Pred:No    Pred:Yes")
    print(f"    Actual No:       {tn:5d}      {fp:5d}")
    print(f"    Actual Yes:      {fn:5d}      {tp:5d}")
    diff = 100 * acc - ref
    sign = "+" if diff >= 0 else ""
    print(f"  vs multi-stat baseline ({ref:.1f}%): {sign}{diff:.1f} pp")

    # Per-stat accuracy breakdown
    if stat_col is not None:
        print(f"\n  Per-stat accuracy:")
        print(f"  {'Stat':<8}  {'N':>5}  {'Pos':>5}  {'Base%':>6}  "
              f"{'Pred%':>6}  {'AUC':>6}  Gap")
        print(f"  {'─'*8}  {'─'*5}  {'─'*5}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*10}")
        for st in ["FG3M", "PTS", "PR", "PA", "PRA"]:
            mask = stat_col == st
            if mask.sum() == 0:
                continue
            yt = y_true[mask]; yp = y_pred[mask]
            ypr = y_proba[mask]
            n   = len(yt)
            pos = yt.sum()
            base = 100 * pos / n
            pred_acc = 100 * accuracy_score(yt, yp)
            try:
                st_auc = roc_auc_score(yt, ypr)
            except Exception:
                st_auc = float("nan")
            gap_sign = "+" if pred_acc >= base else ""
            gap = pred_acc - base
            print(f"  {st:<8}  {n:5d}  {pos:5d}  {base:5.1f}%  "
                  f"{pred_acc:5.1f}%  {st_auc:6.3f}  "
                  f"{gap_sign}{gap:.1f} pp")

    return {"accuracy": acc, "auc": auc, "precision": prec,
            "recall": rec, "f1": f1, "label": label}


def print_calibration(y_true, y_proba):
    """
    Show how well the model's confidence scores actually match reality.

    Splits predictions into buckets by confidence threshold (≥80%, ≥75%, etc.)
    and shows what fraction of those high-confidence picks actually regressed.

    A well-calibrated model at ≥80% confidence should see ~80% actual
    regression.  If it's much lower, the model is overconfident.  If it's
    higher, the model is conservative.

    y_true  — ground-truth labels (0/1 array)
    y_proba — predicted probabilities from predict_proba()[:, 1]
    """
    print(f"\n  Probability calibration:")
    print(f"  Threshold | Picks  | Actual regressed %")
    print(f"  ----------|--------|-------------------")
    for thr in [0.80, 0.75, 0.70, 0.65]:
        mask = y_proba >= thr
        n    = mask.sum()
        if n > 0:
            rate = y_true[mask].mean()
            bar  = "█" * round(rate * 20)
            print(f"  >= {int(thr*100):2d}%     |  {n:4d}  |  {100*rate:.1f}%  {bar}")
        else:
            print(f"  >= {int(thr*100):2d}%     |     0  |  (no picks)")


def print_importances(model, features, new_set):
    """
    Print a ranked list of how much each feature contributed to the model's decisions.

    Feature importance (for GradientBoosting/XGBoost) measures how often a
    feature was used to split the data and how much that improved predictions.
    Higher = more influential.

    model    — the trained sklearn/xgboost model
    features — list of feature names in the same order they were passed to fit()
    new_set  — set of feature names added in v2.2 (flagged with '← NEW' in output)

    Does nothing if the model doesn't have feature_importances_ (e.g. LogReg).
    """
    if not hasattr(model, "feature_importances_"):
        return
    imp = model.feature_importances_
    pairs = sorted(zip(features, imp), key=lambda x: x[1], reverse=True)
    print(f"\n  Feature importances:")
    print(f"  {'Feature':<30}  {'Imp':>8}   Bar")
    print(f"  {'─'*30}  {'─'*8}   {'─'*28}")
    for fname, vi in pairs:
        bar = "█" * min(int(vi * 200), 28)
        tag = " ← NEW" if fname in new_set else ""
        tag += " [imputed]" if fname in IMPUTED_IN_MULTISTAT else ""
        print(f"  {fname:<30}  {vi:8.4f}   {bar}{tag}")

# ══════════════════════════════════════════════════════════════════════════════
# 5.  PREVIOUS BEST (v2.1 GB-100 reproduction — for apples-to-apples)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n\n4.  MODEL 0 — v2.1 GB-100 on FG3M-only validation (reference)\n{DASH}")
print("  Using v2.1 features (no stat_encoded) on FG3M-2024-25 only.")
print("  This establishes the apples-to-apples baseline.\n")

# Build FG3M-only val subset with v2.1 features (no stat_encoded)
V21_FEATURES = [f for f in ALL_FEATURES if f != "stat_encoded"]
fg3m_val_mask = val_clean["stat"] == "FG3M"
X_val_fg3m_only = val_clean.loc[fg3m_val_mask, V21_FEATURES].values.astype(float)
y_val_fg3m_only = val_clean.loc[fg3m_val_mask, TARGET].astype(int).values

# Quick re-train on same training set but without stat_encoded
X_train_v21 = train_clean[V21_FEATURES].values.astype(float)
gb_ref = GradientBoostingClassifier(
    n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
)
gb_ref.fit(X_train_v21, y_train)
pred_ref  = gb_ref.predict(X_val_fg3m_only)
proba_ref = gb_ref.predict_proba(X_val_fg3m_only)[:, 1]
metrics_ref = print_metrics(
    "v2.1 GB-100 — FG3M-2024-25 only (reference)",
    y_val_fg3m_only, pred_ref, proba_ref,
    baseline_acc=FG3M_RULE_BASED_ACC
)

# ══════════════════════════════════════════════════════════════════════════════
# 6.  MODEL A — GradientBoosting 100 trees (all features incl. stat_encoded)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n\n5.  MODEL A — GradientBoosting-100  (v2.2, multi-stat val)\n{DASH}")
gb100 = GradientBoostingClassifier(
    n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
)
gb100.fit(X_train, y_train)
pred_a  = gb100.predict(X_val)
proba_a = gb100.predict_proba(X_val)[:, 1]
metrics_a = print_metrics("GradientBoosting-100", y_val, pred_a, proba_a,
                           stat_col=val_stat_col)
print_calibration(y_val, proba_a)
print_importances(gb100, ALL_FEATURES, NEW_FEATURES_V22)

# ══════════════════════════════════════════════════════════════════════════════
# 7.  MODEL B — XGBoost (if available)
# ══════════════════════════════════════════════════════════════════════════════
metrics_b = None
xgb_model = None
if XGBOOST_AVAILABLE:
    print(f"\n\n6.  MODEL B — XGBoost  (v2.2, multi-stat val)\n{DASH}")
    xgb_model = XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        eval_metric="logloss", random_state=42, verbosity=0,
    )
    xgb_model.fit(X_train, y_train)
    pred_b  = xgb_model.predict(X_val)
    proba_b = xgb_model.predict_proba(X_val)[:, 1]
    metrics_b = print_metrics("XGBoost-100", y_val, pred_b, proba_b,
                               stat_col=val_stat_col)
    print_calibration(y_val, proba_b)
    print_importances(xgb_model, ALL_FEATURES, NEW_FEATURES_V22)
else:
    print(f"\n  ℹ️  XGBoost not available")

# ══════════════════════════════════════════════════════════════════════════════
# 8.  MODEL C — GradientBoosting 200 trees
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n\n7.  MODEL C — GradientBoosting-200  (v2.2, multi-stat val)\n{DASH}")
gb200 = GradientBoostingClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42
)
gb200.fit(X_train, y_train)
pred_c  = gb200.predict(X_val)
proba_c = gb200.predict_proba(X_val)[:, 1]
metrics_c = print_metrics("GradientBoosting-200", y_val, pred_c, proba_c,
                           stat_col=val_stat_col)
print_calibration(y_val, proba_c)
print_importances(gb200, ALL_FEATURES, NEW_FEATURES_V22)

# ══════════════════════════════════════════════════════════════════════════════
# 9.  COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n\n8.  FULL COMPARISON TABLE\n{DASH}")
n_train = len(train_clean)
n_val   = len(val_clean)
n_fg3m_train = len(train_clean)

def row(name, acc, auc, n_tr, notes):
    return (f"  {name:<38} | {acc:>8} | {auc:>7} | "
            f"{n_tr:>6} | {notes}")

hdr = row("Model", "Accuracy", "ROC-AUC", "N-train", "Notes")
sep = "  " + "-"*38 + "-+-" + "-"*8 + "-+-" + "-"*7 + "-+-" + "-"*6 + "-+-" + "-"*30
print(f"\n{hdr}")
print(sep)
print(row("Rule-based (FG3M z>1.0)",
          "73.9%", "---", 0, "FG3M baseline only"))
print(row("Multi-stat rule-based",
          f"{MULTISTAT_BASELINE_ACC:.1f}%", "---", 0, "predict-all on combined val"))
print(row("v2.1 GB-100 (FG3M val only)",
          f"{100*metrics_ref['accuracy']:.1f}%",
          f"{metrics_ref['auc']:.3f}", n_fg3m_train, "reference"))
print(row("v2.2 GB-100 (multi-stat val)",
          f"{100*metrics_a['accuracy']:.1f}%",
          f"{metrics_a['auc']:.3f}", n_train,
          f"stat_encoded, {n_val:,} val events"))
if metrics_b:
    print(row("v2.2 XGBoost-100 (multi-stat val)",
              f"{100*metrics_b['accuracy']:.1f}%",
              f"{metrics_b['auc']:.3f}", n_train,
              f"stat_encoded, {n_val:,} val events"))
print(row("v2.2 GB-200 (multi-stat val)",
          f"{100*metrics_c['accuracy']:.1f}%",
          f"{metrics_c['auc']:.3f}", n_train,
          f"200 trees, {n_val:,} val events"))

# ══════════════════════════════════════════════════════════════════════════════
# 10. SAVE BEST MODEL
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n\n9.  SAVING BEST MODEL\n{DASH}")

candidates = [
    ("GB-100 v2.2",  metrics_a["auc"], gb100,    ALL_FEATURES),
    ("GB-200 v2.2",  metrics_c["auc"], gb200,    ALL_FEATURES),
]
if metrics_b:
    candidates.append(("XGBoost v2.2", metrics_b["auc"], xgb_model, ALL_FEATURES))

best_name, best_auc, best_model, best_features = max(candidates, key=lambda x: x[1])
print(f"\n  Best model: {best_name}  (AUC {best_auc:.3f})")

joblib.dump(best_model, "model_regression.pkl")
print(f"  ✅ model_regression.pkl saved  "
      f"({os.path.getsize('model_regression.pkl'):,} bytes)")

feature_meta = {
    "features":       best_features,
    "train_means":    train_means,
    "model_type":     best_name,
    "best_auc":       round(best_auc, 4),
    "training_stats": ["FG3M"],
    "val_stats":      ["FG3M", "PTS", "PR", "PA", "PRA"],
    "stat_encode":    STAT_ENCODE,
}
with open("model_features.json", "w") as fh:
    json.dump(feature_meta, fh, indent=2)
print(f"  ✅ model_features.json updated")

summary_rows = [
    {"model": "Rule-based (FG3M)", "accuracy": "73.9%",
     "roc_auc": "---", "n_val": 710, "notes": "FG3M baseline"},
    {"model": "Multi-stat rule-based",
     "accuracy": f"{MULTISTAT_BASELINE_ACC:.1f}%",
     "roc_auc": "---", "n_val": n_val, "notes": "predict-all-regression"},
    {"model": "v2.1 GB-100 (FG3M val)",
     "accuracy": f"{100*metrics_ref['accuracy']:.1f}%",
     "roc_auc": f"{metrics_ref['auc']:.3f}",
     "n_val": int(fg3m_val_mask.sum()), "notes": "reference"},
    {"model": "v2.2 GB-100",
     "accuracy": f"{100*metrics_a['accuracy']:.1f}%",
     "roc_auc": f"{metrics_a['auc']:.3f}",
     "n_val": n_val, "notes": "stat_encoded"},
]
if metrics_b:
    summary_rows.append({
        "model": "v2.2 XGBoost-100",
        "accuracy": f"{100*metrics_b['accuracy']:.1f}%",
        "roc_auc": f"{metrics_b['auc']:.3f}",
        "n_val": n_val, "notes": "stat_encoded",
    })
summary_rows.append({
    "model": "v2.2 GB-200",
    "accuracy": f"{100*metrics_c['accuracy']:.1f}%",
    "roc_auc": f"{metrics_c['auc']:.3f}",
    "n_val": n_val, "notes": "200 trees",
})
pd.DataFrame(summary_rows).to_csv("model_validation_summary.csv", index=False)
print(f"  ✅ model_validation_summary.csv updated")

# ══════════════════════════════════════════════════════════════════════════════
# 11. HONEST GENERALISATION ASSESSMENT
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n\n10. GENERALISATION ASSESSMENT\n{DASH}")
print(f"\n  Training data:      FG3M only (2022-23 + 2023-24, {n_train} events)")
print(f"  Validation data:    All stats (2024-25, {n_val} events)")
print(f"  Multi-stat baseline:{MULTISTAT_BASELINE_ACC:.1f}%  (predict-all-regression)")
best_acc = max(metrics_a["accuracy"], metrics_c["accuracy"],
               *([] if not metrics_b else [metrics_b["accuracy"]]))
print(f"  Best model accuracy:{100*best_acc:.1f}%  ({best_name})")
diff = 100 * best_acc - MULTISTAT_BASELINE_ACC
if diff >= 0:
    print(f"  Beat baseline by:  +{diff:.1f} pp  ✅")
else:
    print(f"  Below baseline by:  {diff:.1f} pp  ⚠️  model is not beating naive predictor")

fg3m_acc_new = accuracy_score(
    y_val[val_stat_col == "FG3M"],
    gb100.predict(X_val[val_stat_col == "FG3M"])
)
print(f"\n  FG3M-specific accuracy (new model): {100*fg3m_acc_new:.1f}%  "
      f"(was {100*metrics_ref['accuracy']:.1f}% in v2.1)")

if 100 * best_acc < 65.0:
    print(f"\n  ⚠️  WARNING: Best accuracy ({100*best_acc:.1f}%) < 65% threshold.")
    print(f"     Model is not generalising well to non-FG3M stats.")
    print(f"     Do NOT use for PTS/PR/PA/PRA picks without further calibration.")
else:
    print(f"\n  Model generalises reasonably. Use with caution on non-FG3M stats.")
    print(f"  FG3M remains the most reliable stat type for fade picks.")

print(f"\n{SEP}")
print(f"  Training complete.  Best: {best_name}  AUC: {best_auc:.3f}")
print(f"{SEP}\n")
