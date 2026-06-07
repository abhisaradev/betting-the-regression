"""
train_model.py — v2 logistic regression for Betting the Regression.

Train/validate split:
  Training:   2022-23 + 2023-24 backtest data  (1,408 events)
  Validation: 2024-25 backtest data             (710 events, strict holdout)

Target: your_line_wins == True
  → 1 if actual outcome lands closer to fair_line than to naive_line
  → 0 otherwise
  (same proxy backtest logic as the rule-based model)

Output:
  model_regression.pkl          sklearn Pipeline (StandardScaler + LogReg)
  model_features.json           feature names + training-set means for imputation
  model_validation_summary.csv  accuracy comparison table
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)

RULE_BASED_ACC       = 73.9   # 2024-25 proxy backtest benchmark
INTEGRATION_THRESHOLD = RULE_BASED_ACC - 3.0   # ≥ 70.9% → integrate

SEP  = "=" * 65
DASH = "─" * 40

# ==============================================================================
# 1.  DATA LOADING
# ==============================================================================

print(f"\n{SEP}")
print("  BETTING THE REGRESSION — v2 Logistic Regression Training")
print(f"{SEP}\n")

TRAIN_FILES = [
    "hot_hand_proxy_backtest_2022_23.csv",
    "hot_hand_proxy_backtest_2023_24.csv",
]
VAL_FILE = "hot_hand_proxy_backtest_2024_25.csv"

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


# ==============================================================================
# 2.  TARGET VARIABLE
# ==============================================================================

print(f"\n\n2.  TARGET VARIABLE  (your_line_wins)\n{DASH}")

TARGET = "your_line_wins"
for label, df in [("Training  (2022-24)", train_raw),
                  ("Validation (2024-25)", val_raw)]:
    pos   = int(df[TARGET].sum())
    total = len(df)
    neg   = total - pos
    print(f"  {label}: {pos:4d}/{total} regressed  ({100*pos/total:.1f}%)  |"
          f"  {neg:4d}/{total} did not  ({100*neg/total:.1f}%)")


# ==============================================================================
# 3.  FEATURE ENGINEERING
# ==============================================================================

print(f"\n\n3.  FEATURE ENGINEERING\n{DASH}")

# CRITICAL: columns that are outcomes/leakage — never use as features
LEAKAGE_COLS = {
    "next_game_stat", "actual_outcome", "regression",
    "distance_to_yours", "distance_to_naive", TARGET,
}

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive model features from raw backtest columns. No lookahead."""
    d = df.copy()

    # Primary signal — how large is the hot streak vs true baseline?
    d["gap_from_baseline"] = d["during_hot"] - d["season_avg"]
    d["abs_z_score"]       = d["gap_from_baseline"].abs()  # always ≥ 0 for hot events

    # Normalised z-score proxy (season_std not stored; global std approximation)
    gap_std    = d["gap_from_baseline"].std()
    d["z_score"] = d["gap_from_baseline"] / gap_std if gap_std > 0 else 0.0

    # Location: home=1, away=0
    if "next_location" in d.columns:
        d["is_home"] = (d["next_location"].str.lower() == "home").astype(float)

    # Blowout flag: bool → float
    if "was_blowout_recent" in d.columns:
        d["blowout_flag"] = d["was_blowout_recent"].astype(float)

    # Opponent defensive tier: ordinal encoding
    if "opp_def_tier" in d.columns:
        tier_map = {"Bad (21-30)": 0.0, "Mid (11-20)": 1.0, "Elite (top 10)": 2.0}
        d["def_tier_ord"] = d["opp_def_tier"].map(tier_map)

    return d

train_fe = engineer_features(train_raw)
val_fe   = engineer_features(val_raw)

# Candidates — ordered by expected importance
CANDIDATE_FEATURES = [
    "gap_from_baseline",    # magnitude of hot streak above baseline (key signal)
    "z_score",              # normalised version of above
    "abs_z_score",          # redundant with gap_from_baseline (all positive here)
    "season_avg",           # player's true skill level
    "during_hot",           # raw recent rolling average
    "minutes_trend",        # recent MPG minus season MPG (role change signal)
    "opp_3p_pct_allowed",   # opponent 3P% allowed (defensive quality)
    "opp_pace",             # game-pace context
    "is_home",              # next game home/away
    "blowout_flag",         # recent blowout flag
    "def_tier_ord",         # opponent tier ordinal
]

FEATURES = [f for f in CANDIDATE_FEATURES
            if f in train_fe.columns and f not in LEAKAGE_COLS]

print(f"\n  Available features ({len(FEATURES)}):")
for f in FEATURES:
    nn = train_fe[f].notna().sum()
    print(f"    • {f:<25}  non-null in train: {nn:4d}/{len(train_fe)}")

# Drop rows with NaN in any feature or target
train_clean = train_fe.dropna(subset=FEATURES + [TARGET]).copy()
val_clean   = val_fe.dropna(subset=FEATURES + [TARGET]).copy()

print(f"\n  Rows after dropna:")
print(f"    Training:   {len(train_clean):4d} / {len(train_fe)} kept  "
      f"(dropped {len(train_fe)-len(train_clean)})")
print(f"    Validation: {len(val_clean):4d} / {len(val_fe)} kept  "
      f"(dropped {len(val_fe)-len(val_clean)})")

X_train = train_clean[FEATURES].values.astype(float)
y_train = train_clean[TARGET].astype(int).values
X_val   = val_clean[FEATURES].values.astype(float)
y_val   = val_clean[TARGET].astype(int).values

# Compute training-set feature means for imputation in daily_picks.py
train_means = {f: float(train_clean[f].mean()) for f in FEATURES}


# ==============================================================================
# 4.  TRAIN
# ==============================================================================

print(f"\n\n4.  TRAINING\n{DASH}")

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  LogisticRegression(max_iter=1000, random_state=42, C=1.0)),
])
pipeline.fit(X_train, y_train)

train_acc = accuracy_score(y_train, pipeline.predict(X_train))
print(f"  Pipeline:  StandardScaler → LogisticRegression(C=1.0, max_iter=1000)")
print(f"  Samples:   {len(X_train):,}  |  Features: {len(FEATURES)}")
print(f"  Training accuracy (in-sample): {100*train_acc:.1f}%")


# ==============================================================================
# 5.  VALIDATION METRICS
# ==============================================================================

print(f"\n\n5.  VALIDATION METRICS  (2024-25 holdout)\n{DASH}")

y_pred  = pipeline.predict(X_val)
y_proba = pipeline.predict_proba(X_val)[:, 1]  # P(regression)

acc  = accuracy_score(y_val, y_pred)
prec = precision_score(y_val, y_pred, zero_division=0)
rec  = recall_score(y_val, y_pred, zero_division=0)
f1   = f1_score(y_val, y_pred, zero_division=0)
auc  = roc_auc_score(y_val, y_proba)
cm   = confusion_matrix(y_val, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\n  Accuracy:   {100*acc:.1f}%")
print(f"  Precision:  {100*prec:.1f}%")
print(f"  Recall:     {100*rec:.1f}%")
print(f"  F1 score:   {f1:.3f}")
print(f"  ROC-AUC:    {auc:.3f}")

print(f"\n  Confusion matrix  (rows = actual, cols = predicted):")
print(f"                      Pred: No    Pred: Yes")
print(f"  Actual No:           {tn:5d}       {fp:5d}")
print(f"  Actual Yes:          {fn:5d}       {tp:5d}")
print(f"\n  True  positives (caught regressions):    {tp:4d}")
print(f"  False positives (predicted, didn't):     {fp:4d}")
print(f"  True  negatives (correctly skipped):     {tn:4d}")
print(f"  False negatives (missed regressions):    {fn:4d}")

diff = 100 * acc - RULE_BASED_ACC
print(f"\n  ── Accuracy comparison ──────────────────────────")
print(f"  Rule-based model (proxy backtest):  {RULE_BASED_ACC:.1f}%")
print(f"  Logistic regression (2024-25):      {100*acc:.1f}%")
if diff >= 0:
    print(f"  Difference:  +{diff:.1f} pp  ✅ LR matches or beats rule-based")
elif diff >= -3:
    print(f"  Difference:  {diff:.1f} pp  ⚠️  within 3pp threshold — will integrate")
else:
    print(f"  Difference:  {diff:.1f} pp  ❌ more than 3pp below baseline")


# ==============================================================================
# 6.  PROBABILITY CALIBRATION
# ==============================================================================

print(f"\n\n6.  PROBABILITY CALIBRATION\n{DASH}")
print(f"  (Does model confidence track actual regression rate?)")
print(f"  Threshold | Picks | Actual regressed %")
print(f"  ----------|-------|-------------------")
for thr in [0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50]:
    mask = y_proba >= thr
    n    = mask.sum()
    if n > 0:
        actual_rate = y_val[mask].mean()
        bar = "█" * round(actual_rate * 20)
        print(f"  >= {int(thr*100):2d}%     |  {n:3d}  |  {100*actual_rate:.1f}%  {bar}")
    else:
        print(f"  >= {int(thr*100):2d}%     |    0  |  (no picks at this threshold)")


# ==============================================================================
# 7.  FEATURE IMPORTANCE
# ==============================================================================

print(f"\n\n7.  FEATURE COEFFICIENTS  (sorted by |coef|)\n{DASH}")
coefs = pipeline.named_steps["model"].coef_[0]
pairs = sorted(zip(FEATURES, coefs), key=lambda x: abs(x[1]), reverse=True)
print(f"  {'Feature':<25}  {'Coef':>7}   Effect on regression probability")
print(f"  {'─'*25}  {'─'*7}   {'─'*35}")
for fname, coef in pairs:
    direction = "increases" if coef > 0 else "decreases"
    bar = "█" * min(int(abs(coef) * 8), 24)
    print(f"  {fname:<25}  {coef:+7.3f}   {bar} {direction}")


# ==============================================================================
# 8.  SAVE
# ==============================================================================

print(f"\n\n8.  SAVING ARTIFACTS\n{DASH}")

joblib.dump(pipeline, "model_regression.pkl")
print(f"  ✅ model_regression.pkl saved  ({os.path.getsize('model_regression.pkl'):,} bytes)")

feature_meta = {
    "features":    FEATURES,
    "train_means": train_means,   # used for imputation in daily_picks.py
}
with open("model_features.json", "w") as fh:
    json.dump(feature_meta, fh, indent=2)
print(f"  ✅ model_features.json saved")

summary_rows = [
    {"metric": "Accuracy",          "rule_based": f"{RULE_BASED_ACC:.1f}%",
     "logistic_regression": f"{100*acc:.1f}%"},
    {"metric": "Precision",         "rule_based": "—",
     "logistic_regression": f"{100*prec:.1f}%"},
    {"metric": "Recall",            "rule_based": "—",
     "logistic_regression": f"{100*rec:.1f}%"},
    {"metric": "F1",                "rule_based": "—",
     "logistic_regression": f"{f1:.3f}"},
    {"metric": "ROC-AUC",           "rule_based": "—",
     "logistic_regression": f"{auc:.3f}"},
    {"metric": "Training seasons",  "rule_based": "2022-25",
     "logistic_regression": "2022-24 only"},
    {"metric": "Validation",        "rule_based": "proxy backtest (all yrs)",
     "logistic_regression": "2024-25 holdout"},
]
pd.DataFrame(summary_rows).to_csv("model_validation_summary.csv", index=False)
print(f"  ✅ model_validation_summary.csv saved")


# ==============================================================================
# 9.  INTEGRATION DECISION
# ==============================================================================

print(f"\n\n9.  INTEGRATION DECISION\n{DASH}")
lr_acc_pct = 100 * acc

if lr_acc_pct >= INTEGRATION_THRESHOLD:
    print(f"\n  ✅ INTEGRATE — LR accuracy ({lr_acc_pct:.1f}%) is within 3pp of "
          f"rule-based ({RULE_BASED_ACC:.1f}%)")
    print(f"     → regression_probability will be added to daily_picks.py output")
    print(f"     → probability pill will appear on dashboard pick cards")
    print(f"     INTEGRATE = True")
else:
    gap = RULE_BASED_ACC - lr_acc_pct
    print(f"\n  ❌ DO NOT INTEGRATE — LR accuracy ({lr_acc_pct:.1f}%) is {gap:.1f}pp "
          f"below rule-based ({RULE_BASED_ACC:.1f}%)")
    print(f"""
  Analysis of underperformance:
  ─────────────────────────────
  1. The proxy backtest data covers FG3M (3-pointers) only — a single
     high-variance stat. The available features (opponent defense, home/away,
     pace) all showed near-zero effect in the confounder analysis.

  2. The "rule-based" baseline is simply "hot streaks regress 73.9% of the
     time." A logistic regression predicting which SPECIFIC events regress
     faces an almost-flat signal landscape — the confounder analysis already
     confirmed no individual feature adds meaningful discriminative power.

  3. Class imbalance (73.9% positive): the model may default to predicting
     the majority class (regression = True) for most events, achieving near-
     baseline accuracy without learning anything meaningful.

  4. Feature collinearity: gap_from_baseline, z_score, abs_z_score, and
     during_hot all measure the same thing. L2 regularisation handles this
     but the coefficients are spread thinly.

  What to try next:
  ─────────────────
  1. XGBoost (v2.1) — handles non-linear feature interactions; will better
     capture the interaction between large z-scores and minutes_trend.

  2. Multi-stat training — include PTS and PRA backtest events to triple
     the training set and add diversity. FG3M alone is limited.

  3. Player-level base rates — add "historical regression rate per player"
     as a feature. Some players consistently regress; others maintain streaks.

  4. Calibrated probability output — even if overall accuracy is similar,
     use Platt scaling or isotonic regression to produce better-calibrated
     probabilities for high-confidence picks (>70%).

  5. Threshold-focused metric — optimise for precision@k rather than
     accuracy. If the model identifies 30 events at 80%+ confidence and
     they regress 78% of the time, that's valuable even at lower overall acc.
""")
    print(f"     INTEGRATE = False")

print(f"\n{SEP}")
print(f"  Training complete.")
print(f"{SEP}\n")
