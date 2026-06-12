# /ingest — Header-first CSV inspection

Use this before processing any CSV file. Prevents wasted work on misunderstood schemas and catches unit/naming issues early.

## Steps

1. **Read header only** — load just the column names (e.g. `pd.read_csv(path, nrows=0).columns.tolist()`)
2. **Read sample rows** — load the first 10 rows and display them
3. **Report** — for each column, state:
   - Data type (int, float, string, date)
   - Inferred unit or scale (e.g. "percentage 0–100", "count", "dollars", "decimal ratio 0–1")
   - Null count in the sample
   - Min / max values in the sample
4. **Flag anything suspicious** — columns with unexpected nulls, mixed types, or ambiguous units
5. **Pause and wait for confirmation** before processing the full file or building any dependent code

## Example output format

```
File: hot_hand_proxy_backtest_2024_25.csv
Rows in sample: 10 of ~710 total

Column               Type     Unit/Scale        Nulls   Min      Max
-------------------  -------  ----------------  ------  -------  -------
PLAYER_NAME          str      —                 0       —        —
SEASON               str      —                 0       —        —
fair_line            float    threes (count)    0       0.8      3.2
naive_line           float    threes (count)    0       1.1      4.5
actual_fg3m          float    threes (count)    0       0.0      6.0
your_line_wins       bool     binary 0/1        0       0        1
z_score              float    std deviations    0       1.01     3.84

⚠️  No suspicious columns found.

Ready to proceed with full processing? (confirm before continuing)
```

## Arguments

Pass the file path as the argument: `/ingest path/to/file.csv`

If no path is given, ask which file to inspect.
