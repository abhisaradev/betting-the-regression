# /validate-transforms — Per-column unit verification before numeric transforms

Use this before applying any rescaling, normalization, standardization, or unit conversion to a dataset. Prevents silent mislabeling bugs (like applying ×100 to a column already in 0–100 scale).

## Steps

1. **List every column** that will be touched by the transform
2. **For each column**, declare:
   - Current unit/scale (e.g. "decimal ratio 0–1", "percentage 0–100", "raw count", "z-score")
   - Intended transform (e.g. "multiply by 100", "StandardScaler", "log1p", "no change")
   - Post-transform unit/scale
   - Justification — why this transform is correct for this column
3. **Flag dangerous cases**:
   - Applying ×100 to a column that is already in 0–100 scale
   - Applying StandardScaler to a binary (0/1) column
   - Applying log to a column with zero values without handling zeros
   - Any column where the unit is ambiguous — stop and confirm before proceeding
4. **Write assertions** — for each transformed column, add a before/after check using a known reference value:
   ```python
   # Before
   assert df['fg_pct'].between(0, 1).all(), "fg_pct expected as decimal ratio"
   # After transform (if multiplying by 100)
   assert df['fg_pct'].between(0, 100).all(), "fg_pct expected as percentage after ×100"
   ```
5. **Show the plan and wait for approval** before applying any transform

## Example output format

```
Transform plan for: StandardScaler normalization + percentage conversion

Column          Current scale        Transform          Post-scale           Safe?
--------------  -------------------  -----------------  -------------------  -----
z_score         std deviations       StandardScaler     normalized z         ✅
fg3m_avg        count (threes/game)  StandardScaler     normalized count     ✅
regression_rate decimal ratio 0–1    ×100               percentage 0–100     ✅
fg_pct          decimal ratio 0–1    StandardScaler     normalized ratio     ✅
tov_pct         percentage 0–100     ×100               ❌ ALREADY 0–100     🚨 STOP

🚨 tov_pct is already a percentage — applying ×100 would produce values 0–10000.
   Please confirm the intended transform before proceeding.
```

## Arguments

Pass a description of the planned transform: `/validate-transforms "StandardScaler all numeric columns in training_df"`

If no description is given, ask what transform is being planned.
