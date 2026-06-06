"""
odds_compare.py — Compare Hot Hand Fader fair lines vs live bookmaker props.
Pulls tonight's NBA player prop lines from ESPN's public API (DraftKings data),
matches them against flagged players in daily_predictions.csv, and ranks
edges by gap size (STRONG first, then by gap descending).

Data source: ESPN Core API (sports.core.api.espn.com) — completely public,
no authentication required. Prop lines are DraftKings Over/Under values.

Tested and confirmed working: June 2026, NBA Finals Game 2 (Spurs vs Knicks).

Run after daily_picks.py:  python3.11 odds_compare.py
"""

import os
import re
import sys
import json
import time
import difflib
import subprocess
import statistics
import pandas as pd
from datetime import datetime, timedelta

# ==============================================================================
# CONFIG
# ==============================================================================

# ESPN prop type IDs → our internal stat names
ESPN_PROP_TYPES = {
    "1":  "PTS",   # Total Points
    "4":  "FG3M",  # Total 3-Point Field Goals
    "90": "PRA",   # Total Points, Rebounds, and Assists
}

# Fuzzy name match minimum similarity (0–1)
NAME_MATCH_THRESHOLD = 0.75

PREDICTIONS_FILE = "daily_predictions.csv"
TODAY            = datetime.now().strftime("%Y-%m-%d")
OUTPUT_FILE      = f"odds_comparison_{TODAY}.csv"

ESPN_BASE = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba"


# ==============================================================================
# ESPN API — helpers
# ==============================================================================

def espn_get(path, params=None):
    """
    GET an ESPN core API path. Uses curl -sk to bypass local SSL chain issues.
    Returns parsed JSON or raises RuntimeError on failure.
    """
    base_params = "lang=en&region=us"
    extra = "&" + "&".join(f"{k}={v}" for k, v in (params or {}).items())
    url = f"{ESPN_BASE}/{path}?{base_params}{extra}"
    result = subprocess.run(
        ["curl", "-sk", "--max-time", "10", url],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"curl failed for {url}: {result.stderr[:100]}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"JSON parse error for {url}: {e}\nBody: {result.stdout[:200]}")


def find_nba_event(date_str):
    """
    Find tonight's NBA event on ESPN.
    NBA games tip in the evening ET = early UTC the following day,
    so we search date_str and date_str+1 day.
    Returns (event_id, home_team, away_team) or raises.
    """
    target = datetime.strptime(date_str, "%Y-%m-%d")
    for delta in [0, 1]:
        check = (target + timedelta(days=delta)).strftime("%Y%m%d")
        data = espn_get("events", params={"dates": check, "limit": "20"})
        for item in data.get("items", []):
            ref = item.get("$ref", "")
            m = re.search(r"/events/(\d+)", ref)
            if not m:
                continue
            event_id = m.group(1)
            # Fetch event detail to check sport / get team names
            ev = espn_get(f"events/{event_id}")
            # "name" is e.g. "New York Knicks at San Antonio Spurs"
            # "shortName" is e.g. "NY @ SA"
            ev_name = ev.get("name", "")
            short   = ev.get("shortName", "")
            if ev_name:
                # Parse "Away at Home" → home, away
                if " at " in ev_name:
                    parts = ev_name.split(" at ", 1)
                    away_team, home_team = parts[0].strip(), parts[1].strip()
                else:
                    away_team = home_team = ev_name
                return event_id, home_team, away_team
    raise RuntimeError(f"No NBA event found for {date_str}")


def fetch_prop_bets(event_id):
    """
    Fetch all DraftKings prop bets for an event across all pages.
    Returns list of raw prop bet dicts.
    """
    all_items = []
    page = 1
    while True:
        data = espn_get(
            f"events/{event_id}/competitions/{event_id}/odds/100/propBets",
            params={"limit": "200", "page": str(page)}
        )
        all_items.extend(data.get("items", []))
        if page >= data.get("pageCount", 1):
            break
        page += 1
        time.sleep(0.1)
    return all_items


def resolve_athlete_names(athlete_ids):
    """
    Resolve a set of ESPN athlete IDs to display names.
    Returns dict: athlete_id (str) -> display_name (str)
    """
    names = {}
    for aid in sorted(athlete_ids):
        d = espn_get(f"seasons/2026/athletes/{aid}")
        names[aid] = d.get("displayName", d.get("fullName", f"ID:{aid}"))
        time.sleep(0.1)
    return names


def extract_prop_lines(prop_bets, athlete_names):
    """
    Walk ESPN prop bets and build:
      { (normalised_player_name, stat): median_line }

    Line value is at item["current"]["target"]["value"].
    Each player has two entries (Over and Under) with the same line — we
    deduplicate by taking the median.
    """
    raw = {}  # (norm_name, stat) -> [line, ...]
    for item in prop_bets:
        type_id = item.get("type", {}).get("id", "")
        if type_id not in ESPN_PROP_TYPES:
            continue
        stat = ESPN_PROP_TYPES[type_id]

        ref = item.get("athlete", {}).get("$ref", "")
        m = re.search(r"/athletes/(\d+)", ref)
        if not m:
            continue
        aid = m.group(1)
        player = athlete_names.get(aid, "")
        if not player or player.startswith("ID:"):
            continue

        line_val = (
            item.get("current", {}).get("target", {}).get("value")
            or item.get("odds", {}).get("total", {}).get("value")
        )
        if line_val is None:
            continue

        norm = normalise_name(player)
        raw.setdefault((norm, stat), []).append(float(line_val))

    return {k: round(statistics.median(v), 1) for k, v in raw.items()}


# ==============================================================================
# NAME NORMALISATION & FUZZY MATCHING
# ==============================================================================

def normalise_name(name):
    """Lowercase, strip punctuation for fuzzy comparison."""
    return re.sub(r"[^a-z0-9 ]", "", name.lower()).strip()


def best_match(target, candidates, threshold=NAME_MATCH_THRESHOLD):
    """
    Fuzzy-match target against candidates using difflib.
    Returns (best_candidate, score) or (None, 0) if below threshold.
    """
    t = normalise_name(target)
    best, score = None, 0.0
    for c in candidates:
        s = difflib.SequenceMatcher(None, t, normalise_name(c)).ratio()
        if s > score:
            score, best = s, c
    return (best, score) if score >= threshold else (None, 0.0)


# ==============================================================================
# GAP & THRESHOLD LOGIC
# ==============================================================================

def compute_gap(status, fair_line, book_line):
    """
    HOT  → gap = book_line - fair_line  (+ve = book overprices streak, UNDER edge)
    COLD → gap = fair_line - book_line  (+ve = book underprices fade, OVER edge)
    """
    if status == "HOT":
        return round(book_line - fair_line, 2)
    elif status == "COLD":
        return round(fair_line - book_line, 2)
    return 0.0


def parse_threshold(bet_rec):
    """'bet UNDER if line > 11.0' → 11.0"""
    m = re.search(r"[\d.]+$", str(bet_rec))
    return float(m.group()) if m else None


def threshold_crossed(status, book_line, threshold):
    """True if the book line is on the actionable side of our threshold."""
    if threshold is None:
        return False
    return book_line > threshold if status == "HOT" else book_line < threshold


# ==============================================================================
# OUTPUT
# ==============================================================================

TIER_ORDER = {"STRONG": 0, "MODERATE": 1, "WEAK": 2}


def print_edge_row(row):
    icon = "🔥" if row["status"] == "HOT" else "❄️"
    star = "⭐ " if row["threshold_met"] else "   "
    print(f"\n  {star}{icon} [{row['tier']}] {row['player']} — {row['stat']}")
    print(f"       Our fair line:   {row['fair_line']}")
    print(f"       DK book line:    {row['bookmaker_line']}")
    print(f"       Edge gap:        +{row['gap']}")
    print(f"       Bet action:      {row['bet_recommendation']}")
    if row["threshold_met"]:
        print(f"       ✅ THRESHOLD MET — book line crossed our trigger")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print(f"\n{'#'*70}")
    print(f"#  HOT HAND FADER — ODDS COMPARISON  [ESPN/DRAFTKINGS]")
    print(f"#  {TODAY}")
    print(f"{'#'*70}")

    # ------------------------------------------------------------------
    # 1. Load today's NBA predictions
    # ------------------------------------------------------------------
    if not os.path.exists(PREDICTIONS_FILE):
        print(f"\n❌  {PREDICTIONS_FILE} not found. Run daily_picks.py first.")
        sys.exit(1)

    all_preds = pd.read_csv(PREDICTIONS_FILE)
    preds = all_preds[
        (all_preds["date"].astype(str) == TODAY) &
        (all_preds["league"] == "NBA")
    ].copy()

    if preds.empty:
        print(f"\n⚠️  No NBA predictions for {TODAY} in {PREDICTIONS_FILE}.")
        sys.exit(0)

    print(f"\n  Loaded {len(preds)} NBA prediction(s) for {TODAY}:")
    for _, r in preds.iterrows():
        print(f"    {r['player']:22} {r['stat']:5} {r['status']:5} [{r['tier']}]  "
              f"fair={r['fair_line']}  z={r['z_score']}")

    # ------------------------------------------------------------------
    # 2. Find tonight's NBA event on ESPN
    # ------------------------------------------------------------------
    print(f"\n  Finding tonight's NBA event on ESPN...")
    try:
        event_id, home, away = find_nba_event(TODAY)
    except RuntimeError as e:
        print(f"❌  {e}")
        sys.exit(1)
    fixture_label = f"{away} @ {home}"
    print(f"  ✅ Event {event_id}: {fixture_label}")

    # ------------------------------------------------------------------
    # 3. Fetch prop bets (DraftKings via ESPN)
    # ------------------------------------------------------------------
    print(f"  Fetching prop bets from ESPN/DraftKings...")
    prop_bets = fetch_prop_bets(event_id)
    print(f"  {len(prop_bets)} total prop bets returned")

    # ------------------------------------------------------------------
    # 4. Resolve athlete IDs → names
    # ------------------------------------------------------------------
    athlete_ids = set()
    for item in prop_bets:
        if item.get("type", {}).get("id") in ESPN_PROP_TYPES:
            ref = item.get("athlete", {}).get("$ref", "")
            m = re.search(r"/athletes/(\d+)", ref)
            if m:
                athlete_ids.add(m.group(1))

    print(f"  Resolving {len(athlete_ids)} athlete IDs...")
    athlete_names = resolve_athlete_names(athlete_ids)
    for aid, name in sorted(athlete_names.items(), key=lambda x: x[1]):
        print(f"    {aid}: {name}")

    # ------------------------------------------------------------------
    # 5. Extract prop lines
    # ------------------------------------------------------------------
    player_lines = extract_prop_lines(prop_bets, athlete_names)
    print(f"\n  Prop lines extracted ({len(player_lines)} player/stat pairs):")
    by_player = {}
    for (norm_name, stat), line in sorted(player_lines.items()):
        by_player.setdefault(norm_name, {})[stat] = line
    for player in sorted(by_player):
        stats_str = "  ".join(f"{s}={v}" for s, v in sorted(by_player[player].items()))
        print(f"    {player}: {stats_str}")

    # ------------------------------------------------------------------
    # 6. Match predictions → prop lines (fuzzy name match)
    # ------------------------------------------------------------------
    print(f"\n  Matching predictions to prop lines...")
    results = []

    for _, pred in preds.iterrows():
        player  = pred["player"]
        stat    = pred["stat"]
        status  = pred["status"]
        tier    = pred["tier"]
        fair    = float(pred["fair_line"])
        bet_rec = pred["bet_recommendation"]

        candidates = [norm for norm, s in player_lines if s == stat]
        matched, score = best_match(player, candidates)

        if matched is None:
            print(f"    ⚠️  No match: {player} ({stat})")
            continue

        book_line = player_lines[(matched, stat)]
        gap       = compute_gap(status, fair, book_line)
        thresh    = parse_threshold(bet_rec)
        met       = threshold_crossed(status, book_line, thresh)

        print(f"    ✓  {player:22} ({stat})  book={book_line}  gap={gap:+.2f}  "
              f"match='{matched}' ({score:.2f})")

        results.append({
            "player":           player,
            "stat":             stat,
            "status":           status,
            "tier":             tier,
            "fair_line":        fair,
            "bookmaker_line":   book_line,
            "gap":              gap,
            "threshold_met":    met,
            "bet_recommendation": bet_rec,
        })

    if not results:
        print("\n  No players matched to available prop lines.")
        sys.exit(0)

    # ------------------------------------------------------------------
    # 7. Rank and display: tier (STRONG→WEAK) then gap descending
    # ------------------------------------------------------------------
    df    = pd.DataFrame(results)
    edges = df[df["gap"] > 0].copy()
    edges["tier_rank"] = edges["tier"].map(TIER_ORDER)
    edges = edges.sort_values(
        ["tier_rank", "gap"], ascending=[True, False]
    ).reset_index(drop=True)
    no_edge = df[df["gap"] <= 0]

    print(f"\n{'='*70}")
    print(f"  EDGE REPORT — {TODAY}  |  {fixture_label}")
    print(f"  Source: ESPN/DraftKings (sports.core.api.espn.com — public)")
    print(f"{'='*70}")

    if edges.empty:
        print("\n  No positive edges tonight.")
        print("  (All book lines already on the correct side of our fair lines.)")
    else:
        n_met = int(edges["threshold_met"].sum())
        print(f"\n  {len(edges)} edge(s) found  |  ⭐ {n_met} threshold(s) met\n")
        for _, row in edges.iterrows():
            print_edge_row(row)

    if not no_edge.empty:
        print(f"\n{'─'*70}")
        print("  No edge (book line already on correct side of fair line):")
        for _, row in no_edge.iterrows():
            icon = "🔥" if row["status"] == "HOT" else "❄️"
            print(f"    {icon} {row['player']:22} {row['stat']:5}  "
                  f"fair={row['fair_line']}  book={row['bookmaker_line']}  "
                  f"gap={row['gap']:+.2f}")

    # ------------------------------------------------------------------
    # 8. Save CSV
    # ------------------------------------------------------------------
    df.assign(date=TODAY, fixture=fixture_label, source="ESPN/DraftKings") \
      .to_csv(OUTPUT_FILE, index=False)

    print(f"\n{'='*70}")
    print(f"  Saved {len(df)} row(s) to {OUTPUT_FILE}")
    if not edges.empty:
        n_mon = len(edges) - n_met
        print(f"  ⭐ Actionable (threshold met): {n_met}  |  Monitor: {n_mon}")
    print()


if __name__ == "__main__":
    main()
