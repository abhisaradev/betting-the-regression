"""
odds_compare.py — Compare Hot Hand Fader fair lines vs live bookmaker props.
Pulls tonight's NBA player prop lines and matches them against flagged players
in daily_predictions.csv, ranking edges by gap size.

Data source: OddsPapi (oddspapi.io) — free tier covers pre-game game markets.
NOTE: OddsPapi free tier does NOT provide NBA player prop data (confirmed via
live API testing — all player_id values are "0", indicating team/game markets
only). Player props require a paid OddsPapi plan or a different provider.

Recommended free alternative for player props: The Odds API (the-odds-api.com)
Set ODDS_PROVIDER = "theoddsapi" below and add THE_ODDS_API_KEY to ~/.zshrc
to switch providers without changing any other code.

Setup:
  export ODDSPAPI_KEY=<your_key>       # already in ~/.zshrc
  export THE_ODDS_API_KEY=<your_key>   # get free key at the-odds-api.com
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
# CONFIG — change ODDS_PROVIDER to switch data sources
# ==============================================================================

# "oddspapi"   — uses OddsPapi (free tier: game markets only, no player props)
# "theoddsapi" — uses The Odds API (free tier: full NBA player props, 500 req/mo)
ODDS_PROVIDER = "theoddsapi"

ODDSPAPI_KEY     = os.environ.get("ODDSPAPI_KEY", "")
THE_ODDS_API_KEY = os.environ.get("THE_ODDS_API_KEY", "")

ODDSPAPI_BASE    = "https://api.oddspapi.io/v4"
THEODDSAPI_BASE  = "https://api.the-odds-api.com/v4"

# OddsPapi: bookmakers to query
ODDSPAPI_BOOKMAKERS = "fanduel,hardrockbet,thescore,betrivers"

# The Odds API: regions and bookmakers
THEODDSAPI_REGIONS    = "us"
THEODDSAPI_BOOKMAKERS = "fanduel,betrivers,draftkings,betmgm"

# The Odds API market keys for our three stats
THEODDSAPI_MARKETS = {
    "PTS":  "player_points",
    "FG3M": "player_threes",
    "PRA":  "player_points_rebounds_assists",
}

# OddsPapi exact market name strings (confirmed from /markets catalog)
ODDSPAPI_STAT_TO_MARKET = {
    "PTS":  "Over Under Player Points (incl. overtime)",
    "FG3M": "Over Under Player 3 Point FG (incl. overtime)",
    "PRA":  "Over Under Player Points + Assists + Rebounds (incl. overtime)",
}

# Fuzzy name match minimum similarity (0-1); 0.75 handles apostrophes & spacing
NAME_MATCH_THRESHOLD = 0.75

PREDICTIONS_FILE = "daily_predictions.csv"
TODAY            = datetime.now().strftime("%Y-%m-%d")
OUTPUT_FILE      = f"odds_comparison_{TODAY}.csv"


# ==============================================================================
# SHARED API HELPER
# ==============================================================================

def http_get(url, params, provider_label):
    """
    GET request with retry on 429 (rate limit).
    On 403, prints the full error body and raises — caller decides how to handle.
    """
    for attempt in range(3):
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code == 429:
            wait = 5 * (2 ** attempt)
            print(f"  ⏳ [{provider_label}] Rate limited — waiting {wait}s...")
            time.sleep(wait)
            continue
        if resp.status_code == 403:
            print(f"\n  ❌ 403 Forbidden from {provider_label}:")
            print(f"     {resp.text[:400]}")
            resp.raise_for_status()
        resp.raise_for_status()
        return resp.json()
    resp.raise_for_status()


# ==============================================================================
# ODDSPAPI PROVIDER
# ==============================================================================

def oddspapi_get(endpoint, **params):
    if not ODDSPAPI_KEY:
        print("❌  ODDSPAPI_KEY not set in environment.")
        sys.exit(1)
    params["apiKey"] = ODDSPAPI_KEY
    return http_get(f"{ODDSPAPI_BASE}/{endpoint}", params, "OddsPapi")


def oddspapi_build_market_catalog():
    """Return market_id -> {name, handicap} for our three stat types."""
    print("  Building OddsPapi market catalog...", end="", flush=True)
    data = oddspapi_get("markets")
    relevant = set(ODDSPAPI_STAT_TO_MARKET.values())
    catalog = {}
    for m in data:
        name = m.get("marketName", "")
        if name in relevant:
            mid = str(m.get("marketId", ""))
            h   = m.get("handicap")
            if mid and h is not None:
                catalog[mid] = {"name": name, "handicap": float(h)}
    print(f" {len(catalog)} markets.")
    return catalog


def oddspapi_find_fixture(date_str):
    """Find the NBA fixture for date_str (searches ±2 days for UTC boundary)."""
    import datetime as dt
    base      = dt.datetime.strptime(date_str, "%Y-%m-%d")
    from_date = (base - dt.timedelta(days=1)).strftime("%Y-%m-%d")
    to_date   = (base + dt.timedelta(days=2)).strftime("%Y-%m-%d")
    data = oddspapi_get("fixtures", sportId=11, **{"from": from_date, "to": to_date})
    nba = [f for f in data if f.get("tournamentName") == "NBA" and f.get("hasOdds")]
    return nba[0] if nba else None


def oddspapi_fetch_odds(fixture_id, fixture_status):
    """
    Fetch odds for a fixture.
    If game is live (RESTRICTED_ACCESS), fall back to /historical-odds.
    Returns (bookmaker_odds_dict, source_label).
    """
    try:
        data = oddspapi_get("odds", fixtureId=fixture_id,
                            bookmakers=ODDSPAPI_BOOKMAKERS)
        return data.get("bookmakerOdds", {}), "live"
    except requests.HTTPError as e:
        if e.response.status_code == 403:
            body = e.response.json()
            code = body.get("error", {}).get("code", "")
            if code == "RESTRICTED_ACCESS":
                print(f"  ℹ️  Game is live — free tier blocks live odds.")
                print(f"     Falling back to /historical-odds (pre-game snapshot)...")
                data = oddspapi_get("historical-odds", fixtureId=fixture_id,
                                    bookmakers=ODDSPAPI_BOOKMAKERS)
                # historical-odds nests under "bookmakers" not "bookmakerOdds"
                raw = data.get("bookmakers", data)
                # normalise historical snapshot lists → single latest value per player
                return _normalise_historical(raw), "historical"
        raise


def _normalise_historical(bookmakers):
    """
    /historical-odds players values are lists of snapshots.
    Convert to the same shape as /odds (single dict per player).
    """
    normalised = {}
    for bk, bk_data in bookmakers.items():
        markets = bk_data.get("markets", {})
        norm_markets = {}
        for mid, mdata in markets.items():
            outcomes = mdata.get("outcomes", {})
            norm_outcomes = {}
            for oid, odata in outcomes.items():
                players = odata.get("players", {})
                norm_players = {}
                for pid, snapshots in players.items():
                    if isinstance(snapshots, list) and snapshots:
                        # Take the most recent active snapshot
                        active = [s for s in snapshots if s.get("active", True)]
                        snap = active[-1] if active else snapshots[-1]
                        norm_players[pid] = snap
                    elif isinstance(snapshots, dict):
                        norm_players[pid] = snapshots
                norm_outcomes[oid] = {"players": norm_players}
            norm_markets[mid] = {"outcomes": norm_outcomes}
        normalised[bk] = {"markets": norm_markets}
    return normalised


def oddspapi_extract_lines(bookmaker_odds, market_catalog):
    """
    Walk OddsPapi odds and return {(norm_player_name, stat): median_line}.
    Skips player_id == "0" (team/game markets).
    Player names in OddsPapi are "Last, First" — we convert to "First Last".
    """
    name_to_stat = {v: k for k, v in ODDSPAPI_STAT_TO_MARKET.items()}
    raw = {}

    for bk, bk_data in bookmaker_odds.items():
        markets = bk_data.get("markets", {}) if isinstance(bk_data, dict) else {}
        for mid, mdata in markets.items():
            minfo = market_catalog.get(str(mid))
            if not minfo:
                continue
            stat     = name_to_stat.get(minfo["name"])
            handicap = minfo["handicap"]
            outcomes = mdata.get("outcomes", {}) if isinstance(mdata, dict) else {}
            for oid, odata in outcomes.items():
                players = odata.get("players", {}) if isinstance(odata, dict) else {}
                for pid, pdata in players.items():
                    if pid == "0":
                        continue
                    if not isinstance(pdata, dict):
                        continue
                    raw_name = pdata.get("playerName", "")
                    if not raw_name or not pdata.get("active", True):
                        continue
                    norm = normalise_name(oddspapi_flip_name(raw_name))
                    raw.setdefault((norm, stat), []).append(handicap)

    return {k: round(pd.Series(v).median(), 2) for k, v in raw.items()}


def oddspapi_flip_name(raw):
    """'Fox, De\\'Aaron' -> 'De\\'Aaron Fox'"""
    raw = raw.strip()
    if "," in raw:
        last, first = raw.split(",", 1)
        return f"{first.strip()} {last.strip()}"
    return raw


# ==============================================================================
# THE ODDS API PROVIDER
# ==============================================================================

def theoddsapi_get(path, **params):
    if not THE_ODDS_API_KEY:
        print("\n❌  THE_ODDS_API_KEY not set.")
        print("    1. Get a free key at https://the-odds-api.com  (500 req/mo free)")
        print("    2. Add to ~/.zshrc:  export THE_ODDS_API_KEY=<your_key>")
        print("    3. Run:              source ~/.zshrc")
        sys.exit(1)
    params["apiKey"] = THE_ODDS_API_KEY
    return http_get(f"{THEODDSAPI_BASE}/{path}", params, "TheOddsAPI")


def theoddsapi_find_event(date_str):
    """
    Find tonight's NBA event ID on The Odds API.
    Returns (event_id, home_team, away_team) or None.
    """
    import datetime as dt
    base = dt.datetime.strptime(date_str, "%Y-%m-%d")
    # commence_time_from/to filter to today's games (ET evening = UTC next day)
    from_utc = base.strftime("%Y-%m-%dT00:00:00Z")
    to_utc   = (base + dt.timedelta(days=2)).strftime("%Y-%m-%dT12:00:00Z")

    events = theoddsapi_get(
        "sports/basketball_nba/events",
        commenceTimeFrom=from_utc,
        commenceTimeTo=to_utc,
    )
    if not events:
        return None
    # Prefer the Finals game (Spurs/Knicks); fall back to first event
    for ev in events:
        teams = f"{ev.get('home_team','')} {ev.get('away_team','')}".lower()
        if "spurs" in teams or "knicks" in teams:
            return ev["id"], ev["home_team"], ev["away_team"]
    ev = events[0]
    return ev["id"], ev["home_team"], ev["away_team"]


def theoddsapi_fetch_lines(event_id):
    """
    Fetch player prop lines for all three stat types.
    Returns {(norm_player_name, stat): median_line_across_bookmakers}.

    The Odds API response per market outcome:
      {"name": "De'Aaron Fox", "description": "Over", "price": -110, "point": 18.5}
    Player name is already in "First Last" format. "point" is the prop line.
    We average Over/Under point values (they're always equal) and take median
    across bookmakers.
    """
    markets_param = ",".join(THEODDSAPI_MARKETS.values())
    data = theoddsapi_get(
        f"sports/basketball_nba/events/{event_id}/odds",
        regions=THEODDSAPI_REGIONS,
        markets=markets_param,
        bookmakers=THEODDSAPI_BOOKMAKERS,
        oddsFormat="american",
    )

    # Invert: market_key -> stat
    market_to_stat = {v: k for k, v in THEODDSAPI_MARKETS.items()}

    raw = {}  # (norm_name, stat) -> [line, ...]
    for bk in data.get("bookmakers", []):
        for market in bk.get("markets", []):
            stat = market_to_stat.get(market["key"])
            if not stat:
                continue
            for outcome in market.get("outcomes", []):
                name  = outcome.get("name", "")
                point = outcome.get("point")
                if not name or point is None:
                    continue
                norm = normalise_name(name)
                raw.setdefault((norm, stat), []).append(float(point))

    return {k: round(pd.Series(v).median(), 2) for k, v in raw.items()}


# ==============================================================================
# SHARED NAME NORMALISATION & FUZZY MATCHING
# ==============================================================================

def normalise_name(name):
    """Lowercase, strip punctuation/accents for fuzzy comparison."""
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
    HOT  → gap = book_line - fair_line  (positive = book is above fair, UNDER edge)
    COLD → gap = fair_line - book_line  (positive = book is below fair, OVER edge)
    """
    if status == "HOT":
        return round(book_line - fair_line, 2)
    elif status == "COLD":
        return round(fair_line - book_line, 2)
    return 0.0


def parse_threshold(bet_rec):
    """'bet UNDER if line > 11.0' -> 11.0"""
    m = re.search(r"[\d.]+$", str(bet_rec))
    return float(m.group()) if m else None


def threshold_crossed(status, book_line, threshold):
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
    print(f"#  HOT HAND FADER — ODDS COMPARISON  [{ODDS_PROVIDER.upper()}]")
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
    # 2. Fetch player prop lines from chosen provider
    # ------------------------------------------------------------------
    player_lines = {}
    fixture_label = "?"

    if ODDS_PROVIDER == "theoddsapi":
        print(f"\n  [The Odds API] Finding tonight's NBA event...")
        result = theoddsapi_find_event(TODAY)
        time.sleep(0.3)
        if not result:
            print("❌  No NBA event found for tonight on The Odds API.")
            sys.exit(1)
        event_id, home, away = result
        fixture_label = f"{away} @ {home}"
        print(f"  ✅ Event: {fixture_label}  (id: {event_id})")

        print(f"  Fetching player props ({', '.join(THEODDSAPI_MARKETS.values())})...")
        player_lines = theoddsapi_fetch_lines(event_id)
        time.sleep(0.3)

    elif ODDS_PROVIDER == "oddspapi":
        print(f"\n  [OddsPapi] Finding tonight's NBA fixture...")
        fixture = oddspapi_find_fixture(TODAY)
        time.sleep(0.3)
        if not fixture:
            print("❌  No NBA fixture found on OddsPapi for tonight.")
            sys.exit(1)
        fid    = fixture["fixtureId"]
        p1     = fixture.get("participant1Name", "?")
        p2     = fixture.get("participant2Name", "?")
        status = fixture.get("statusName", "")
        fixture_label = f"{p2} @ {p1}"
        print(f"  ✅ Fixture [{fid}]: {fixture_label}  status={status}")

        print(f"  Building market catalog...")
        catalog = oddspapi_build_market_catalog()
        time.sleep(0.3)

        print(f"  Fetching odds...")
        bk_odds, source = oddspapi_fetch_odds(fid, status)
        print(f"  Bookmakers: {list(bk_odds.keys())}  (source: {source})")
        time.sleep(0.3)

        player_lines = oddspapi_extract_lines(bk_odds, catalog)

        if not player_lines:
            print("\n⚠️  OddsPapi returned no player prop lines.")
            print("   Root cause (confirmed by live API testing):")
            print("   → Free tier only provides game-level markets (moneyline/spread/totals)")
            print("   → All player_id values are '0' — no player-specific props exist")
            print("   → This applies to both /odds and /historical-odds responses")
            print("")
            print("   To get NBA player props, either:")
            print("   a) Upgrade to an OddsPapi paid plan")
            print("   b) Switch to The Odds API (free, 500 req/mo):")
            print("      1. Get key at https://the-odds-api.com")
            print("      2. export THE_ODDS_API_KEY=<key>  >> ~/.zshrc")
            print("      3. Set ODDS_PROVIDER = 'theoddsapi' in this file")
            sys.exit(0)

    else:
        print(f"❌  Unknown ODDS_PROVIDER: {ODDS_PROVIDER!r}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 3. Report what was found
    # ------------------------------------------------------------------
    if not player_lines:
        print(f"\n⚠️  No player prop lines found for tonight.")
        print(f"    Re-run closer to tip-off if props aren't posted yet.")
        sys.exit(0)

    print(f"  Found lines for {len(player_lines)} (player, stat) pairs:")
    for norm_name, stat in sorted(player_lines):
        print(f"    {norm_name:30} {stat}  line={player_lines[(norm_name, stat)]}")

    # ------------------------------------------------------------------
    # 4. Match predictions → bookmaker lines (fuzzy name match)
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

        # Candidates: normalised names that have this stat
        candidates = [name for name, s in player_lines if s == stat]
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
    # 5. Rank and display
    # ------------------------------------------------------------------
    df    = pd.DataFrame(results)
    edges = df[df["gap"] > 0].copy()
    edges["tier_rank"] = edges["tier"].map(TIER_ORDER)
    edges = edges.sort_values(["tier_rank", "gap"], ascending=[True, False]).reset_index(drop=True)
    no_edge = df[df["gap"] <= 0]

    print(f"\n{'='*70}")
    print(f"  EDGE REPORT — {TODAY}  |  {fixture_label}")
    print(f"{'='*70}")

    if edges.empty:
        print("\n  No positive edges tonight.")
        print("  (All bookmaker lines are already on the correct side of our fair line.)")
    else:
        print(f"\n  {len(edges)} edge(s)  |  ⭐ {int(edges['threshold_met'].sum())} threshold(s) met\n")
        for _, row in edges.iterrows():
            print_edge_row(row)

    if not no_edge.empty:
        print(f"\n{'─'*70}")
        print("  No edge (book line already on correct side):")
        for _, row in no_edge.iterrows():
            icon = "🔥" if row["status"] == "HOT" else "❄️"
            print(f"    {icon} {row['player']:22} {row['stat']:5}  "
                  f"fair={row['fair_line']}  book={row['bookmaker_line']}  gap={row['gap']:+.2f}")

    # ------------------------------------------------------------------
    # 6. Save CSV
    # ------------------------------------------------------------------
    df.assign(date=TODAY, fixture=fixture_label).to_csv(OUTPUT_FILE, index=False)
    print(f"\n{'='*70}")
    print(f"  Saved {len(df)} row(s) to {OUTPUT_FILE}")
    if not edges.empty:
        n_met  = int(edges["threshold_met"].sum())
        n_mon  = len(edges) - n_met
        print(f"  Actionable (threshold met): {n_met}  |  Monitor: {n_mon}")
    print()


if __name__ == "__main__":
    main()
