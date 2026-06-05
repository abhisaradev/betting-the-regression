"""
odds_compare.py — Compare Hot Hand Fader fair lines vs live bookmaker props.
Pulls tonight's NBA player prop lines from OddsPapi, matches them against
flagged players in daily_predictions.csv, and ranks edges by gap size.

Setup:  export ODDSPAPI_KEY=<your_key>   (already in ~/.zshrc)
Run after daily_picks.py:  python3.11 odds_compare.py
"""

import os
import re
import sys
import time
import difflib
import requests
import pandas as pd
from datetime import datetime

# ==============================================================================
# CONFIG
# ==============================================================================

API_KEY  = os.environ.get("ODDSPAPI_KEY", "")
BASE_URL = "https://api.oddspapi.io/v4"

# Bookmakers with best NBA player prop coverage on OddsPapi free tier
BOOKMAKERS = "fanduel,hardrockbet,thescore,betrivers"

# Exact OddsPapi market name strings (confirmed from /markets catalog)
STAT_TO_MARKET_NAME = {
    "PTS":  "Over Under Player Points (incl. overtime)",
    "FG3M": "Over Under Player 3 Point FG (incl. overtime)",
    "PRA":  "Over Under Player Points + Assists + Rebounds (incl. overtime)",
}

# Fuzzy name match minimum similarity (0-1); 0.75 handles apostrophes & spacing
NAME_MATCH_THRESHOLD = 0.75

PREDICTIONS_FILE = "daily_predictions.csv"
TODAY = datetime.now().strftime("%Y-%m-%d")
OUTPUT_FILE = f"odds_comparison_{TODAY}.csv"


# ==============================================================================
# API HELPERS
# ==============================================================================

def api_get(endpoint, **params):
    """
    GET to OddsPapi, injecting API key.
    Retries up to 3 times with exponential backoff on 429 (rate limit).
    """
    if not API_KEY:
        print("\n❌  ODDSPAPI_KEY environment variable not set.")
        print("    Add to ~/.zshrc:  export ODDSPAPI_KEY=<your_key>")
        print("    Then run:         source ~/.zshrc")
        sys.exit(1)
    params["apiKey"] = API_KEY
    url = f"{BASE_URL}/{endpoint}"
    for attempt in range(3):
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code == 429:
            wait = 5 * (2 ** attempt)  # 5s, 10s, 20s
            print(f"  ⏳ Rate limited — waiting {wait}s before retry...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()
    resp.raise_for_status()  # re-raise after all retries exhausted


def build_market_catalog():
    """
    Fetch the global /markets catalog and return:
      market_id (str) -> {"name": str, "handicap": float}

    OddsPapi uses one market ID per (stat_type, handicap_line) combination.
    E.g., market 111698 = "Over Under Player Points (incl. overtime)" at 15.5.
    """
    print("  Building market catalog (stat names → IDs)...", end="", flush=True)
    data = api_get("markets")
    catalog = {}
    relevant_names = set(STAT_TO_MARKET_NAME.values())
    for m in data:
        name = m.get("marketName", "")
        if name in relevant_names:
            mid = str(m.get("marketId", ""))
            handicap = m.get("handicap")
            if mid and handicap is not None:
                catalog[mid] = {"name": name, "handicap": float(handicap)}
    print(f" {len(catalog)} relevant markets loaded.")
    return catalog


def find_nba_fixture(date_str):
    """
    Search OddsPapi /fixtures for the NBA game on date_str.
    Returns the fixture dict with hasOdds=True, or None.
    NBA tip-offs are evening ET → early UTC next day, so we search today+1.
    """
    import datetime as dt
    base = dt.datetime.strptime(date_str, "%Y-%m-%d")
    from_date = (base - dt.timedelta(days=1)).strftime("%Y-%m-%d")
    # +2 days: NBA tip-offs are evening ET (00:30+ UTC next day), so we need
    # to include the following calendar day in UTC.
    to_date   = (base + dt.timedelta(days=2)).strftime("%Y-%m-%d")

    data = api_get("fixtures", sportId=11, **{"from": from_date, "to": to_date})
    nba = [f for f in data
           if f.get("tournamentName") == "NBA" and f.get("hasOdds")]
    return nba[0] if nba else None


def fetch_odds(fixture_id):
    """Return the raw bookmakerOdds dict for a fixture."""
    data = api_get("odds", fixtureId=fixture_id, bookmakers=BOOKMAKERS)
    return data.get("bookmakerOdds", {})


# ==============================================================================
# PLAYER NAME NORMALISATION & FUZZY MATCHING
# ==============================================================================

def normalise_oddspapi_name(raw):
    """
    OddsPapi player names are "Last, First" (e.g., "Fox, De'Aaron").
    Convert to "First Last" for comparison against our predictions.
    """
    raw = raw.strip()
    if "," in raw:
        last, first = raw.split(",", 1)
        return f"{first.strip()} {last.strip()}"
    return raw


def normalise_pred_name(name):
    """Strip punctuation and lowercase for fuzzy comparison."""
    return re.sub(r"[^a-z0-9 ]", "", name.lower()).strip()


def best_name_match(target, candidates, threshold=NAME_MATCH_THRESHOLD):
    """
    Return the best matching name from candidates using difflib SequenceMatcher.
    Uses normalised (lowercase, no punctuation) comparison.
    Returns (matched_name, score) or (None, 0) if below threshold.
    """
    target_norm = normalise_pred_name(target)
    best_name, best_score = None, 0.0
    for cand in candidates:
        cand_norm = normalise_pred_name(cand)
        score = difflib.SequenceMatcher(None, target_norm, cand_norm).ratio()
        if score > best_score:
            best_score, best_name = score, cand
    if best_score >= threshold:
        return best_name, best_score
    return None, 0.0


# ==============================================================================
# PROP LINE EXTRACTION
# ==============================================================================

def extract_player_lines(bookmaker_odds, market_catalog):
    """
    Walk the OddsPapi odds structure and return:
      { (normalised_player_name, stat): median_line_across_bookmakers }

    OddsPapi structure:
      bookmaker_odds[bookmaker][markets][market_id][outcomes][outcome_id]
                   [players][player_id] = {playerName, price, active, ...}

    The handicap (prop line) comes from market_catalog[market_id]["handicap"].
    Player names are in "Last, First" format — we normalise to "First Last".
    """
    raw_lines = {}  # (player_name_normalised, stat) -> list of line floats

    # Build reverse lookup: market_id -> stat
    mid_to_stat = {mid: None for mid in market_catalog}
    name_to_stat = {v: k for k, v in STAT_TO_MARKET_NAME.items()}
    for mid, minfo in market_catalog.items():
        mid_to_stat[mid] = name_to_stat.get(minfo["name"])

    for bookmaker, bk_data in bookmaker_odds.items():
        markets = bk_data.get("markets", {}) if isinstance(bk_data, dict) else {}

        for market_id, market_data in markets.items():
            minfo = market_catalog.get(str(market_id))
            if not minfo:
                continue  # not a stat we track
            stat     = name_to_stat.get(minfo["name"])
            handicap = minfo["handicap"]

            outcomes = market_data.get("outcomes", {}) if isinstance(market_data, dict) else {}
            for outcome_id, outcome_data in outcomes.items():
                players = outcome_data.get("players", {}) if isinstance(outcome_data, dict) else {}

                for pid, pdata in players.items():
                    if pid == "0":
                        continue  # "0" = team/game market, not a player
                    if not isinstance(pdata, dict):
                        continue
                    raw_name   = pdata.get("playerName", "")
                    is_active  = pdata.get("active", True)
                    if not raw_name or not is_active:
                        continue

                    norm_name = normalise_pred_name(normalise_oddspapi_name(raw_name))
                    key = (norm_name, stat)
                    raw_lines.setdefault(key, []).append(handicap)

    # Median across bookmakers per (player, stat)
    return {key: round(pd.Series(vals).median(), 2)
            for key, vals in raw_lines.items()}


# ==============================================================================
# GAP COMPUTATION
# ==============================================================================

def compute_gap(status, fair_line, book_line):
    """
    HOT  pick: gap = book_line - fair_line
               Positive = book has line ABOVE our fair line → UNDER edge.
    COLD pick: gap = fair_line - book_line
               Positive = book has line BELOW our fair line → OVER edge.
    """
    if status == "HOT":
        return round(book_line - fair_line, 2)
    elif status == "COLD":
        return round(fair_line - book_line, 2)
    return 0.0


def parse_threshold(bet_recommendation):
    """
    Extract numeric threshold from bet_recommendation string.
    'bet UNDER if line > 11.0' -> 11.0
    'bet OVER if line < 22.3'  -> 22.3
    """
    m = re.search(r"[\d.]+$", str(bet_recommendation))
    return float(m.group()) if m else None


def threshold_crossed(status, book_line, threshold):
    """Return True if the book line is on the actionable side of our threshold."""
    if threshold is None:
        return False
    if status == "HOT":
        return book_line > threshold   # line is high enough to bet UNDER
    elif status == "COLD":
        return book_line < threshold   # line is low enough to bet OVER
    return False


# ==============================================================================
# OUTPUT FORMATTING
# ==============================================================================

TIER_ORDER = {"STRONG": 0, "MODERATE": 1, "WEAK": 2}

def print_edge_row(row):
    icon = "🔥" if row["status"] == "HOT" else "❄️"
    star = "⭐ " if row["threshold_met"] else "   "
    print(f"\n  {star}{icon} [{row['tier']}] {row['player']} — {row['stat']}")
    print(f"       Our fair line:   {row['fair_line']}")
    print(f"       Bookmaker line:  {row['bookmaker_line']}")
    print(f"       Edge gap:        +{row['gap']}")
    print(f"       Bet action:      {row['bet_recommendation']}")
    if row["threshold_met"]:
        print(f"       ✅ THRESHOLD MET — bet this now")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print(f"\n{'#'*70}")
    print(f"#  HOT HAND FADER — ODDS COMPARISON")
    print(f"#  {TODAY}  |  NBA Finals Game 2: Spurs vs Knicks")
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
        print(f"\n⚠️  No NBA predictions for {TODAY}.")
        sys.exit(0)

    print(f"\n  Loaded {len(preds)} NBA prediction(s) for {TODAY}:")
    for _, r in preds.iterrows():
        print(f"    {r['player']:22} {r['stat']:5} {r['status']:5} [{r['tier']}]  "
              f"fair={r['fair_line']}  z={r['z_score']}")

    # ------------------------------------------------------------------
    # 2. Find tonight's NBA fixture
    # ------------------------------------------------------------------
    print(f"\n  Looking up tonight's NBA fixture...")
    fixture = find_nba_fixture(TODAY)
    time.sleep(0.3)

    if not fixture:
        print("❌  No NBA fixture with odds found for tonight.")
        print("    Check OddsPapi coverage or try again closer to tip-off.")
        sys.exit(1)

    fid   = fixture["fixtureId"]
    p1    = fixture.get("participant1Name", "?")
    p2    = fixture.get("participant2Name", "?")
    start = fixture.get("startTime", "")
    print(f"  ✅ Found: [{fid}] {p1} vs {p2}  (tip: {start})")

    # ------------------------------------------------------------------
    # 3. Build market catalog (stat → market IDs + lines)
    # ------------------------------------------------------------------
    market_catalog = build_market_catalog()
    time.sleep(0.3)

    # ------------------------------------------------------------------
    # 4. Fetch live odds for the fixture
    # ------------------------------------------------------------------
    print(f"  Fetching live odds from {BOOKMAKERS}...")
    bookmaker_odds = fetch_odds(fid)
    time.sleep(0.3)

    bk_count = len(bookmaker_odds)
    print(f"  Bookmakers returned: {list(bookmaker_odds.keys())}")

    # ------------------------------------------------------------------
    # 5. Extract player prop lines
    # ------------------------------------------------------------------
    print(f"  Extracting player prop lines...")
    player_lines = extract_player_lines(bookmaker_odds, market_catalog)

    if not player_lines:
        print("\n⚠️  No player prop lines found in tonight's odds.")
        print("    This is normal — NBA props typically go live 2-4 hours before tip-off.")
        print(f"    Tip-off: {start}")
        print(f"    Re-run this script closer to game time.")
        print("\n    If props never appear, OddsPapi free tier may not cover NBA player props.")
        print("    Alternative: The Odds API (the-odds-api.com) has a free tier with full")
        print("    NBA player prop coverage. Update BASE_URL and API key accordingly.")
        sys.exit(0)

    print(f"  Found {len(player_lines)} (player, stat) prop lines:")
    prop_names = sorted(set(name for name, _ in player_lines.keys()))
    for n in prop_names:
        print(f"    {n}")

    # ------------------------------------------------------------------
    # 6. Match predictions to bookmaker lines via fuzzy name matching
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

        # Filter candidates to correct stat
        stat_candidates = {name for name, s in player_lines.keys() if s == stat}

        matched_name, score = best_name_match(player, stat_candidates)

        if matched_name is None:
            print(f"    ⚠️  No match: {player} ({stat})  — not in tonight's props")
            continue

        book_line = player_lines[(matched_name, stat)]
        gap       = compute_gap(status, fair, book_line)
        thresh    = parse_threshold(bet_rec)
        met       = threshold_crossed(status, book_line, thresh)

        # Reconstruct original name for display
        display_match = next(
            (normalise_oddspapi_name(k[0]) for k in player_lines
             if normalise_pred_name(normalise_oddspapi_name(k[0])) == matched_name
             and k[1] == stat),
            matched_name
        )
        print(f"    ✓  {player:22} ({stat}) → '{display_match}'  "
              f"book={book_line}  gap={gap:+.2f}  match={score:.2f}")

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
    # 7. Rank: tier first (STRONG→WEAK), then gap descending
    # ------------------------------------------------------------------
    df = pd.DataFrame(results)
    edges = df[df["gap"] > 0].copy()
    edges["tier_rank"] = edges["tier"].map(TIER_ORDER)
    edges = edges.sort_values(["tier_rank", "gap"], ascending=[True, False]).reset_index(drop=True)

    no_edge = df[df["gap"] <= 0]

    # ------------------------------------------------------------------
    # 8. Print ranked output
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  EDGE REPORT — {TODAY}  |  {p1} vs {p2}")
    print(f"{'='*70}")

    if edges.empty:
        print("\n  No positive edges tonight.")
        print("  (All bookmaker lines sit on the correct side of our fair lines already.)")
    else:
        print(f"\n  {len(edges)} edge(s) found  |  "
              f"⭐ {int(edges['threshold_met'].sum())} threshold(s) met\n")
        for _, row in edges.iterrows():
            print_edge_row(row)

    if not no_edge.empty:
        print(f"\n{'─'*70}")
        print("  No edge (book line already on correct side):")
        for _, row in no_edge.iterrows():
            icon = "🔥" if row["status"] == "HOT" else "❄️"
            print(f"    {icon} {row['player']:22} {row['stat']:5} "
                  f"fair={row['fair_line']}  book={row['bookmaker_line']}  gap={row['gap']:+.2f}")

    # ------------------------------------------------------------------
    # 9. Save to CSV
    # ------------------------------------------------------------------
    save_df = df.assign(date=TODAY, fixture=f"{p1} vs {p2}")
    save_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n{'='*70}")
    print(f"  Saved {len(df)} row(s) to {OUTPUT_FILE}")
    if not edges.empty:
        print(f"  Actionable edges: {int(edges['threshold_met'].sum())} "
              f"(threshold crossed) + {int((~edges['threshold_met']).sum())} "
              f"(monitor)")
    print()


if __name__ == "__main__":
    main()
