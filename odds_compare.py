"""
odds_compare.py — Betting the Regression: compare fair lines vs live DraftKings props.
Pulls tonight's NBA player prop lines from ESPN's public API (DraftKings data),
matches them against flagged players in daily_predictions.csv, ranks edges by
gap size, generates dashboard.html, and opens it in the browser automatically.

Data source: ESPN Core API (sports.core.api.espn.com) — completely public,
no authentication required. Prop lines are DraftKings Over/Under values.

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
import webbrowser
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

NAME_MATCH_THRESHOLD = 0.75

PREDICTIONS_FILE = "daily_predictions.csv"
PERFORMANCE_FILE = "model_performance.csv"
TODAY            = datetime.now().strftime("%Y-%m-%d")
OUTPUT_FILE      = f"odds_comparison_{TODAY}.csv"
DASHBOARD_FILE   = "dashboard.html"

ESPN_BASE = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba"

NBA_TEAMS = {
    1610612737: "Atlanta Hawks",        1610612738: "Boston Celtics",
    1610612751: "Brooklyn Nets",        1610612766: "Charlotte Hornets",
    1610612741: "Chicago Bulls",        1610612739: "Cleveland Cavaliers",
    1610612742: "Dallas Mavericks",     1610612743: "Denver Nuggets",
    1610612765: "Detroit Pistons",      1610612744: "Golden State Warriors",
    1610612745: "Houston Rockets",      1610612754: "Indiana Pacers",
    1610612746: "LA Clippers",          1610612747: "Los Angeles Lakers",
    1610612763: "Memphis Grizzlies",    1610612748: "Miami Heat",
    1610612749: "Milwaukee Bucks",      1610612750: "Minnesota Timberwolves",
    1610612740: "New Orleans Pelicans", 1610612752: "New York Knicks",
    1610612760: "Oklahoma City Thunder",1610612753: "Orlando Magic",
    1610612755: "Philadelphia 76ers",   1610612756: "Phoenix Suns",
    1610612757: "Portland Trail Blazers",1610612758:"Sacramento Kings",
    1610612759: "San Antonio Spurs",    1610612761: "Toronto Raptors",
    1610612762: "Utah Jazz",            1610612764: "Washington Wizards",
}


def parse_baseline_games(baseline_used):
    """Extract game count from 'current_playoffs (15g)' → 15."""
    m = re.search(r'\((\d+)g\)', str(baseline_used))
    return int(m.group(1)) if m else 0


# ==============================================================================
# ESPN API — helpers
# ==============================================================================

def espn_get(path, params=None):
    """GET an ESPN core API path. Uses curl -sk for SSL compatibility."""
    base = "lang=en&region=us"
    extra = "&" + "&".join(f"{k}={v}" for k, v in (params or {}).items())
    url = f"{ESPN_BASE}/{path}?{base}{extra}"
    return _curl_json(url)


def espn_get_url(url):
    """GET an absolute ESPN URL. Used for following $ref links."""
    return _curl_json(url)


def _curl_json(url):
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


def fetch_injury_report(home_team_name, away_team_name):
    """
    Fetch the ESPN NBA injury report and filter to tonight's two teams.

    Returns:
        pick_lookup  — dict  normalised_player_name → {status, description, team}
                       for fuzzy-matching against our picks.
        all_injuries — list[dict]  every injured player from both teams,
                       for the standalone dashboard section.
    Status values: 'OUT' | 'DOUBTFUL' | 'GTD' | raw uppercase.
    Always fetched fresh — never cached.
    """
    url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
    try:
        data = _curl_json(url)
    except Exception as e:
        print(f"  ⚠️  Injury API error: {e}")
        return {}, []

    STATUS_MAP = {
        "out":          "OUT",
        "doubtful":     "DOUBTFUL",
        "questionable": "GTD",
        "day-to-day":   "GTD",
        "probable":     "GTD",
    }

    # Normalise tonight's team names for fuzzy comparison
    targets = [home_team_name.lower(), away_team_name.lower()]

    pick_lookup  = {}   # normalised_player → {status, desc, team}
    all_injuries = []   # [{player, team, status, description}]

    for team_entry in data.get("injuries", []):
        team_display = team_entry.get("displayName", "")
        is_playing = any(
            difflib.SequenceMatcher(None, t, team_display.lower()).ratio() > 0.75
            for t in targets
        )
        if not is_playing:
            continue

        for inj in team_entry.get("injuries", []):
            player = inj.get("athlete", {}).get("displayName", "")
            if not player:
                continue

            raw_status = inj.get("status", "")
            status = STATUS_MAP.get(raw_status.lower(), raw_status.upper() or "UNKNOWN")

            details = inj.get("details", {})
            parts = [
                p for p in [
                    details.get("side", ""),
                    details.get("type", ""),
                    details.get("detail", ""),
                ]
                if p and p.lower() not in ("", "not specified")
            ]
            description = " ".join(parts) if parts else inj.get("shortComment", "")[:80]

            norm = normalise_name(player)
            pick_lookup[norm] = {
                "status":      status,
                "description": description,
                "team":        team_display,
            }
            all_injuries.append({
                "player":      player,
                "team":        team_display,
                "status":      status,
                "description": description,
            })

    return pick_lookup, all_injuries


def find_nba_event(date_str):
    """
    Find tonight's NBA event on ESPN.
    Searches date_str and date_str+1 day (games tip ET evening = UTC next day).
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
            ev = espn_get(f"events/{event_id}")
            ev_name = ev.get("name", "")
            if ev_name:
                if " at " in ev_name:
                    parts = ev_name.split(" at ", 1)
                    away_team, home_team = parts[0].strip(), parts[1].strip()
                else:
                    away_team = home_team = ev_name
                return event_id, home_team, away_team
    raise RuntimeError(f"No NBA event found for {date_str}")


def fetch_game_info(event_id, home_team, away_team):
    """
    Fetch supplementary game metadata for the dashboard.
    Returns dict with start_time_et, venue, series_record, round_label.
    Gracefully degrades — any ESPN error just leaves fields empty.
    """
    info = {
        "home_team":    home_team,
        "away_team":    away_team,
        "start_time_et": "",
        "venue":        "",
        "series_record": "",
        "round_label":  "NBA Playoffs",
    }
    try:
        ev = espn_get(f"events/{event_id}")

        # Start time: parse UTC, convert to EDT (UTC-4)
        date_utc = ev.get("date", "")
        if date_utc:
            try:
                dt = datetime.strptime(date_utc, "%Y-%m-%dT%H:%MZ")
                dt_et = dt - timedelta(hours=4)
                h = dt_et.hour % 12 or 12
                ampm = "AM" if dt_et.hour < 12 else "PM"
                info["start_time_et"] = f"{h}:{dt_et.minute:02d} {ampm} ET"
            except Exception:
                pass

        # Competition detail
        comp = espn_get(f"events/{event_id}/competitions/{event_id}")

        # Round label from competition type
        ctype = comp.get("type", {}).get("text", "")
        if ctype.lower() in ("final", "finals"):
            info["round_label"] = "NBA Finals"
        elif ctype:
            info["round_label"] = f"NBA Playoffs · {ctype}"

        # Series record from notes (e.g. "NYK leads series 3-2")
        for note in comp.get("notes", []):
            hl = note.get("headline", "")
            if hl:
                info["series_record"] = hl
                break

        # Venue — follow $ref
        venue_ref = comp.get("venue", {}).get("$ref", "")
        if venue_ref:
            vm = re.search(r"/venues/(\d+)", venue_ref)
            if vm:
                try:
                    vd = espn_get(f"venues/{vm.group(1)}")
                    name = vd.get("fullName", "")
                    city = vd.get("address", {}).get("city", "")
                    info["venue"] = f"{name}, {city}" if name and city else name
                except Exception:
                    pass
    except Exception:
        pass

    return info


def fetch_prop_bets(event_id):
    """Fetch all DraftKings prop bets for an event (all pages)."""
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
    """Resolve ESPN athlete IDs → display names."""
    names = {}
    for aid in sorted(athlete_ids):
        d = espn_get(f"seasons/2026/athletes/{aid}")
        names[aid] = d.get("displayName", d.get("fullName", f"ID:{aid}"))
        time.sleep(0.1)
    return names


def extract_prop_lines(prop_bets, athlete_names):
    """
    Walk ESPN prop bets → { (norm_player_name, stat): median_line }.
    Each player has Over + Under entries with the same line; deduplicate via median.
    """
    raw = {}
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
    return re.sub(r"[^a-z0-9 ]", "", name.lower()).strip()


def best_match(target, candidates, threshold=NAME_MATCH_THRESHOLD):
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
    if status == "HOT":
        return round(book_line - fair_line, 2)
    elif status == "COLD":
        return round(fair_line - book_line, 2)
    return 0.0


def parse_threshold(bet_rec):
    m = re.search(r"[\d.]+$", str(bet_rec))
    return float(m.group()) if m else None


def threshold_crossed(status, book_line, threshold):
    if threshold is None:
        return False
    return book_line > threshold if status == "HOT" else book_line < threshold


# ==============================================================================
# TERMINAL OUTPUT
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
# DASHBOARD GENERATION
# ==============================================================================

def generate_dashboard(picks, game_info, date_str, model_record=None, all_injuries=None):
    """
    Write dashboard.html and open it in the default browser.

    picks        — list of dicts with player/team/stat/status/tier/fair_line/
                   dk_line/gap/z_score/bet_recommendation/recent_avg/recent_mpg/
                   baseline_games/threshold/threshold_met/direction/
                   injury_flag/injury_desc
    game_info    — dict with home_team/away_team/start_time_et/venue/
                   series_record/round_label
    date_str     — "2026-06-06"
    model_record — {"wins": N, "losses": M} or None
    all_injuries — list[{player, team, status, description}] for injury section
    """
    picks_json    = json.dumps(picks,               ensure_ascii=False)
    game_json     = json.dumps(game_info,           ensure_ascii=False)
    record_json   = json.dumps(model_record,        ensure_ascii=False)
    injuries_json = json.dumps(all_injuries or [],  ensure_ascii=False)
    fixture       = f"{game_info.get('away_team','')} @ {game_info.get('home_team','')}"

    # Read per-stat performance CSV and aggregate totals per stat
    STAT_PERF_FILE = "model_performance_by_stat.csv"
    stat_perf = {}
    try:
        if os.path.exists(STAT_PERF_FILE):
            sp = pd.read_csv(STAT_PERF_FILE)
            for stat in ["FG3M", "PTS", "PRA"]:
                rows = sp[sp["stat"] == stat]
                if not rows.empty:
                    w      = int(rows["wins"].sum())
                    l      = int(rows["losses"].sum())
                    played = w + l
                    wr     = round(100.0 * w / played, 1) if played > 0 else None
                    stat_perf[stat] = {"wins": w, "losses": l, "win_rate": wr}
    except Exception:
        pass   # degrade gracefully — JS shows backtest baselines
    stat_perf_json = json.dumps(stat_perf, ensure_ascii=False)

    # ------------------------------------------------------------------
    # HTML template — plain string (not f-string), use __TOKEN__ markers
    # ------------------------------------------------------------------
    html = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Betting the Regression &mdash; __DATE__</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js" crossorigin="anonymous"></script>
<style>
:root {
  --bg:            #f0f2f5;
  --bg-card:       #ffffff;
  --bg-raised:     #f5f7fa;
  --border:        #e1e4ea;
  --text-1:        #111827;
  --text-2:        #4b5563;
  --text-3:        #9ca3af;
  --accent-hot:    #1D9E75;
  --accent-cold:   #378ADD;
  --accent-none:   #d1d5db;
  --bet-under:     #E24B4A;
  --bet-over:      #1D9E75;
  --r:             8px;
  --r-lg:          12px;
  --badge-strong-bg:   #E1F5EE; --badge-strong-tx: #0F6E56;
  --badge-mod-bg:      #FAEEDA; --badge-mod-tx:    #854F0B;
  --badge-weak-bg:     #F1EFE8; --badge-weak-tx:   #5F5E5A;
  --round-bg:      #1e3a5f;     --round-tx:        #7eb8f7;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg:        #0d0f14;
    --bg-card:   #161b24;
    --bg-raised: #1c2130;
    --border:    #2a3040;
    --text-1:    #edf0f5;
    --text-2:    #9ba8bb;
    --text-3:    #5c6878;
    --badge-strong-bg: #0b2219; --badge-strong-tx: #3ecf96;
    --badge-mod-bg:    #261603; --badge-mod-tx:    #e09754;
    --badge-weak-bg:   #1c1b18; --badge-weak-tx:   #9e9b93;
    --round-bg:  #132340;       --round-tx:        #6bb0f5;
  }
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,-apple-system,BlinkMacSystemFont,sans-serif;
     background:var(--bg);color:var(--text-1);font-size:14px;line-height:1.5;
     min-height:100vh}
.wrap{max-width:880px;margin:0 auto;padding:20px 16px 64px}

/* ── Header ── */
.hdr{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:3px}
.hdr h1{font-size:21px;font-weight:800;letter-spacing:-.4px}
.hdr .hdate{font-size:13px;color:var(--text-2)}
.sub{font-size:12px;color:var(--text-3);margin-bottom:20px}

/* ── How it works ── */
.how{background:var(--bg-card);border:1px solid var(--border);
     border-radius:var(--r-lg);padding:16px 18px;margin-bottom:20px}
.how h2{font-size:11px;font-weight:700;text-transform:uppercase;
        letter-spacing:.6px;color:var(--text-2);margin-bottom:10px}
.how p{font-size:13px;color:var(--text-2);line-height:1.65;margin-bottom:14px}
.how-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.hi{font-size:12px;color:var(--text-2)}
.hi strong{display:block;font-size:12px;color:var(--text-1);margin-bottom:1px}
.cold-note{grid-column:1/-1;font-size:12px;color:var(--text-3);
           border-top:1px solid var(--border);padding-top:9px;margin-top:4px}

/* ── Game strip ── */
.game{background:var(--bg-card);border:1px solid var(--border);
      border-radius:var(--r-lg);padding:14px 18px;margin-bottom:20px;
      display:flex;align-items:center;justify-content:space-between;
      flex-wrap:wrap;gap:8px}
.gteams{font-size:16px;font-weight:800}
.gmeta{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.gmeta span{font-size:12px;color:var(--text-2)}
.rbadge{background:var(--round-bg);color:var(--round-tx);font-size:11px;
        font-weight:700;padding:3px 9px;border-radius:5px;
        text-transform:uppercase;letter-spacing:.3px}

/* ── Metric cards ── */
.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:24px}
.mc{background:var(--bg-card);border:1px solid var(--border);
    border-radius:var(--r);padding:12px 14px}
.mc-label{font-size:10px;color:var(--text-2);text-transform:uppercase;
          letter-spacing:.5px;margin-bottom:4px}
.mc-val{font-size:24px;font-weight:800}

/* ── Section header ── */
.sec-hdr{font-size:11px;font-weight:700;text-transform:uppercase;
         letter-spacing:.5px;color:var(--text-2);margin:24px 0 10px}

/* ── Pick card ── */
.pick{background:var(--bg-card);border:1px solid var(--border);
      border-left:4px solid var(--accent-none);border-radius:var(--r-lg);
      margin-bottom:10px;padding:14px 16px;transition:opacity .15s}
.pick.hot{border-left-color:var(--accent-hot)}
.pick.cold{border-left-color:var(--accent-cold)}
.pick.dim{opacity:.55}

.ptop{display:flex;align-items:flex-start;justify-content:space-between;
      margin-bottom:9px;gap:10px}
.pleft{display:flex;align-items:center;gap:7px;min-width:0;flex-wrap:wrap}
.pname{font-size:15px;font-weight:700}
.paction{flex-shrink:0;text-align:right}

/* Tier badges */
.badge{font-size:10px;font-weight:800;padding:2px 7px;border-radius:4px;
       text-transform:uppercase;letter-spacing:.4px;white-space:nowrap}
.b-STRONG  {background:var(--badge-strong-bg);color:var(--badge-strong-tx)}
.b-MODERATE{background:var(--badge-mod-bg);   color:var(--badge-mod-tx)}
.b-WEAK    {background:var(--badge-weak-bg);   color:var(--badge-weak-tx)}

/* Stat pill */
.spill{font-size:11px;font-weight:600;padding:2px 7px;border-radius:4px;
       background:var(--bg-raised);color:var(--text-2);
       border:1px solid var(--border);white-space:nowrap}

/* Action buttons */
.abtn{font-size:13px;font-weight:700;padding:7px 15px;border-radius:var(--r);
      border:none;cursor:pointer;white-space:nowrap;line-height:1.2}
.abtn.under{background:var(--bet-under);color:#fff}
.abtn.over {background:var(--bet-over); color:#fff}
.abtn.watch{background:transparent;color:var(--text-2);font-weight:500;
            border:1.5px solid var(--border)}
.abtn.watch:hover{border-color:var(--text-2)}
.abtn.no-edge{background:transparent;color:var(--text-3);font-weight:400;
              font-size:12px;border:none;padding:0;cursor:default}
.watch-sub{font-size:11px;color:var(--text-3);margin-top:4px}

/* Pick meta */
.pmeta{font-size:12px;color:var(--text-2);margin-bottom:10px;
       display:flex;gap:10px;flex-wrap:wrap}

/* Numbers */
.pnums{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:10px}
.nc{text-align:center}
.nc-lbl{font-size:10px;color:var(--text-3);text-transform:uppercase;
        letter-spacing:.3px;margin-bottom:2px}
.nc-val{font-size:18px;font-weight:800}
.gap-hot {color:var(--accent-hot)}
.gap-cold{color:var(--accent-cold)}
.gap-neg {color:var(--text-3)}
.thresh  {color:var(--bet-under)}

/* Add to paper link */
.add-paper{font-size:12px;color:var(--text-2);cursor:pointer;
           text-decoration:underline;background:none;border:none;padding:0;
           font-family:inherit}
.add-paper:hover{color:var(--text-1)}

/* ── Paper trading ── */
.paper{background:var(--bg-card);border:1px solid var(--border);
       border-radius:var(--r-lg);padding:18px 20px;margin-top:28px}
.phdr{display:flex;justify-content:space-between;align-items:center;margin-bottom:16px}
.ptitle{font-size:15px;font-weight:800}
.bank-row{display:flex;align-items:center;gap:8px}
.bank-amt{font-size:20px;font-weight:800}
.bank-amt.up  {color:var(--accent-hot)}
.bank-amt.down{color:var(--bet-under)}
.edit-btn{font-size:12px;color:var(--text-2);cursor:pointer;
          text-decoration:underline;background:none;border:none;font-family:inherit}

/* Paper stats */
.pstats{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:16px}
.psc{background:var(--bg-raised);border:1px solid var(--border);
     border-radius:var(--r);padding:10px 12px}
.psc-lbl{font-size:10px;text-transform:uppercase;letter-spacing:.5px;
         color:var(--text-2);margin-bottom:3px}
.psc-val{font-size:20px;font-weight:800}
.psc-val.pos{color:var(--accent-hot)}
.psc-val.neg{color:var(--bet-under)}

/* Bet form */
.bform{display:flex;gap:8px;align-items:flex-end;flex-wrap:wrap;margin-bottom:12px}
.fg{display:flex;flex-direction:column;gap:4px}
.fg label{font-size:11px;color:var(--text-2);text-transform:uppercase;
          letter-spacing:.3px}
select,input[type=number]{background:var(--bg-raised);border:1px solid var(--border);
  border-radius:var(--r);color:var(--text-1);padding:7px 10px;font-size:13px;
  font-family:inherit}
select{min-width:220px}
input[type=number]{width:82px}
.btn-place{background:var(--accent-hot);color:#fff;border:none;border-radius:var(--r);
           padding:8px 18px;font-size:13px;font-weight:700;cursor:pointer;
           font-family:inherit}
.btn-place:hover{opacity:.9}

/* Action row */
.arow{display:flex;gap:8px;margin-bottom:16px;flex-wrap:wrap}
.act{background:var(--bg-raised);border:1px solid var(--border);
     border-radius:var(--r);padding:6px 12px;font-size:12px;cursor:pointer;
     color:var(--text-1);font-family:inherit}
.act:disabled{opacity:.35;cursor:not-allowed}
.act.dng{border-color:#E24B4A55;color:var(--bet-under)}
.act.dng:hover:not(:disabled){background:var(--bet-under);color:#fff;border-color:var(--bet-under)}

/* Pending */
.pend-hdr{font-size:11px;font-weight:700;text-transform:uppercase;
          letter-spacing:.4px;color:var(--text-2);margin-bottom:8px}
.pi{display:flex;align-items:center;justify-content:space-between;
    padding:10px 12px;border:1px solid var(--border);border-radius:var(--r);
    margin-bottom:6px;gap:8px;flex-wrap:wrap}
.pi-desc{font-size:13px;font-weight:600}
.pi-meta{font-size:12px;color:var(--text-2)}
.pi-acts{display:flex;gap:6px}
.gbtn{font-size:12px;font-weight:700;padding:5px 11px;border:none;
      border-radius:5px;cursor:pointer}
.gbtn.win {background:#d1fae5;color:#065f46}
.gbtn.win:hover {background:var(--accent-hot);color:#fff}
.gbtn.loss{background:#fee2e2;color:#991b1b}
.gbtn.loss:hover{background:var(--bet-under);color:#fff}

/* Chart */
.chart-wrap{margin:16px 0;height:170px}

/* History */
.hist-sec{margin-top:8px}
.srch{width:100%;padding:7px 10px;border:1px solid var(--border);border-radius:var(--r);
      background:var(--bg-raised);color:var(--text-1);font-size:13px;
      font-family:inherit;margin-bottom:10px}
.htable{width:100%;border-collapse:collapse}
.htable th{font-size:10px;text-transform:uppercase;letter-spacing:.4px;
           color:var(--text-2);text-align:left;padding:6px 8px;
           border-bottom:1px solid var(--border)}
.htable td{font-size:13px;padding:9px 8px;border-bottom:1px solid var(--border)}
.bw{background:#d1fae5;color:#065f46;font-size:10px;font-weight:800;
    padding:2px 6px;border-radius:4px}
.bl{background:#fee2e2;color:#991b1b;font-size:10px;font-weight:800;
    padding:2px 6px;border-radius:4px}
.pp{color:var(--accent-hot);font-weight:700}
.pn{color:var(--bet-under);font-weight:700}
.empty{text-align:center;color:var(--text-3);font-size:13px;padding:24px}

/* ── Per-stat performance table ── */
.sp-card{background:var(--bg-card);border:1px solid var(--border);
         border-radius:var(--r-lg);padding:14px 18px;margin-bottom:20px}
.sp-tbl{width:100%;border-collapse:collapse;margin-top:6px}
.sp-tbl th{font-size:10px;text-transform:uppercase;letter-spacing:.4px;
           color:var(--text-2);text-align:left;padding:5px 8px;
           border-bottom:2px solid var(--border)}
.sp-tbl td{font-size:13px;padding:8px 8px;border-bottom:1px solid var(--border)}
.sp-stat{font-weight:800;letter-spacing:.2px}
.sp-sig{font-size:12px;color:var(--text-2)}
.sp-bt{font-size:11px;color:var(--text-3)}
.sp-dim{color:var(--text-3)}
.sp-good{color:var(--accent-hot);font-weight:700}
.sp-ok{color:#f59e0b;font-weight:700}
.sp-bad{color:var(--bet-under);font-weight:700}
.sp-note{font-size:11px;color:var(--text-3);font-style:italic;
         margin-bottom:8px;padding:4px 0}

/* ── Injury section ── */
.inj-sec{background:var(--bg-card);border:1px solid var(--border);
         border-radius:var(--r-lg);padding:14px 18px;margin-bottom:20px}
.inj-tbl{width:100%;border-collapse:collapse;margin-top:6px}
.inj-tbl th{font-size:10px;text-transform:uppercase;letter-spacing:.4px;
            color:var(--text-2);text-align:left;padding:5px 8px;
            border-bottom:1px solid var(--border)}
.inj-tbl td{font-size:12px;padding:7px 8px;border-bottom:1px solid var(--border)}
.inj-none{font-size:12px;color:var(--text-3);padding:4px 0}
/* Injury status badges (shared by section table and pick cards) */
.ib{font-size:10px;font-weight:800;padding:2px 6px;border-radius:4px;white-space:nowrap}
.ib-OUT      {background:#fee2e2;color:#991b1b}
.ib-DOUBTFUL {background:#fef3c7;color:#92400e}
.ib-GTD      {background:#fef9c3;color:#713f12}
.ib-RETURNING{background:#dbeafe;color:#1e40af}
/* Pick-card injury flag */
.pick-inj{font-size:11px;margin-top:5px;padding:3px 8px;
          border-radius:4px;display:inline-block;font-weight:600}
.pick.inj-out{border-left-color:#991b1b!important;opacity:.55}

/* ── v2 Regression probability pill ── */
.reg-pill{display:inline-flex;align-items:center;gap:4px;font-size:10px;
          font-weight:700;padding:2px 7px;border-radius:10px;
          letter-spacing:.3px;margin-top:5px}
.reg-pill-high{background:#dcfce7;color:#166534}   /* ≥70% green   */
.reg-pill-med {background:#fef9c3;color:#713f12}   /* 60-69% amber  */
.reg-pill-low {background:#f3f4f6;color:#6b7280}   /* <60%  grey    */

/* Responsive */
@media(max-width:600px){
  .metrics,.pstats{grid-template-columns:repeat(2,1fr)}
  .pnums{grid-template-columns:repeat(2,1fr)}
  .how-grid{grid-template-columns:1fr}
  .bform{flex-direction:column}
  select{min-width:0;width:100%}
  .game{flex-direction:column;align-items:flex-start}
}
</style>
</head>
<body>
<div class="wrap">

  <!-- Header -->
  <div class="hdr">
    <h1>Betting the Regression</h1>
    <span class="hdate">__DATE__</span>
  </div>
  <p class="sub">Lines from DraftKings via ESPN &middot; no auth required</p>

  <!-- How it works -->
  <div class="how">
    <h2>How this works</h2>
    <p>The model compares each player's true season average (their real skill level) against their last 3 games. When a hot streak pushes recent form far above the true avg, the market overreacts &mdash; we fade it (<strong>BET UNDER</strong>). When a cold streak drops below the true avg, we look for a bounce (<strong>BET OVER</strong>).</p>
    <div class="how-grid">
      <div class="hi"><strong>True avg</strong>Season baseline &mdash; all games in this playoff context</div>
      <div class="hi"><strong>Last 3 games</strong>Recent form that triggered the flag</div>
      <div class="hi"><strong>Gap</strong>How far DK's line sits from the true avg</div>
      <div class="hi"><strong>Bet threshold</strong>Minimum gap needed to act on this pick</div>
      <div class="hi"><strong>Sample size</strong>Shown on each card &mdash; more games = more reliable baseline</div>
      <div class="hi"><strong>Cold signals</strong>Historically weaker (44% win rate vs 92% for hot fades)</div>
      <div class="cold-note">&#9888; Cold picks (BET OVER) have a significantly weaker track record. Hot fades are the model's core strength.</div>
    </div>
  </div>

  <!-- Game strip -->
  <div class="game">
    <div class="gteams">__FIXTURE__</div>
    <div class="gmeta">
      <span id="g-time"></span>
      <span id="g-venue"></span>
      <span id="g-series"></span>
      <span class="rbadge" id="g-round"></span>
    </div>
  </div>

  <!-- Injury report (rendered by JS from INJURIES data) -->
  <div id="inj-root"></div>

  <!-- Metric cards -->
  <div class="metrics">
    <div class="mc"><div class="mc-label">Total picks</div><div class="mc-val" id="m-total">&#8212;</div></div>
    <div class="mc"><div class="mc-label">Positive gaps</div><div class="mc-val" id="m-gaps">&#8212;</div></div>
    <div class="mc"><div class="mc-label">Actionable</div><div class="mc-val" id="m-action">&#8212;</div></div>
    <div class="mc"><div class="mc-label">Model record</div><div class="mc-val" id="m-record">&#8212;</div></div>
  </div>

  <!-- Per-stat performance (rendered by JS from STAT_PERF data) -->
  <div id="stat-perf-root"></div>

  <!-- Picks -->
  <div id="picks-root"></div>

  <!-- Paper trading -->
  <div class="paper">
    <div class="phdr">
      <span class="ptitle">Paper trading</span>
      <div class="bank-row">
        <span class="bank-amt" id="bank-disp">$100.00</span>
        <button class="edit-btn" onclick="editBankroll()">edit</button>
      </div>
    </div>

    <div class="pstats">
      <div class="psc"><div class="psc-lbl">Total P&amp;L</div><div class="psc-val" id="ps-pnl">$0.00</div></div>
      <div class="psc"><div class="psc-lbl">Wins</div><div class="psc-val" id="ps-wins">0</div></div>
      <div class="psc"><div class="psc-lbl">Losses</div><div class="psc-val" id="ps-losses">0</div></div>
      <div class="psc"><div class="psc-lbl">ROI %</div><div class="psc-val" id="ps-roi">&#8212;</div></div>
    </div>

    <div class="bform">
      <div class="fg"><label>Pick</label><select id="bet-pick"></select></div>
      <div class="fg"><label>Stake ($)</label><input type="number" id="bet-stake" value="5" min="0.01" step="0.01"></div>
      <button class="btn-place" onclick="placeBet()">Place bet</button>
    </div>

    <div class="arow">
      <button class="act" id="btn-undo" onclick="undo()" disabled>&#8629; Undo</button>
      <button class="act" id="btn-redo" onclick="redo()" disabled>&#8631; Redo</button>
      <button class="act dng" onclick="clearAllBets()">Clear all bets</button>
      <button class="act dng" onclick="resetBankroll()">Reset bankroll</button>
    </div>

    <div id="pend-sec" style="display:none">
      <div class="pend-hdr">Pending bets</div>
      <div id="pend-list"></div>
    </div>

    <div class="chart-wrap"><canvas id="bk-chart"></canvas></div>

    <div class="hist-sec" id="hist-sec" style="display:none">
      <div class="sec-hdr">Bet history</div>
      <input class="srch" id="hist-search" placeholder="Search player, stat or direction&hellip;" oninput="renderHistory()">
      <div id="hist-body"></div>
    </div>
  </div>

</div><!-- /wrap -->

<script>
// ── Data injected by Python ─────────────────────────────────────────────────
const PICKS       = __PICKS_JSON__;
const GAME_INFO   = __GAME_JSON__;
const MODEL_RECORD= __RECORD_JSON__;
const INJURIES    = __INJURIES_JSON__;
const STAT_PERF   = __STAT_PERF_JSON__;

// ── State (persisted in localStorage) ──────────────────────────────────────
let state = {
  start: 100,
  bankroll: 100,
  bets: [],           // pending bets
  history: [],        // graded bets
  bkHistory: [100],   // bankroll history for chart
};

let undoStack = [];   // session-only
let redoStack = [];

function loadState() {
  try {
    const raw = localStorage.getItem('btr_bets');
    if (raw) state = JSON.parse(raw);
    if (!Array.isArray(state.bkHistory) || !state.bkHistory.length)
      state.bkHistory = [state.start];
  } catch(e) {}
}
function saveState() {
  localStorage.setItem('btr_bets', JSON.stringify(state));
}
function pushUndo() {
  undoStack.push(JSON.stringify(state));
  redoStack = [];
  syncUndoRedo();
}
function undo() {
  if (!undoStack.length) return;
  redoStack.push(JSON.stringify(state));
  state = JSON.parse(undoStack.pop());
  saveState(); renderAll();
}
function redo() {
  if (!redoStack.length) return;
  undoStack.push(JSON.stringify(state));
  state = JSON.parse(redoStack.pop());
  saveState(); renderAll();
}
function syncUndoRedo() {
  document.getElementById('btn-undo').disabled = !undoStack.length;
  document.getElementById('btn-redo').disabled = !redoStack.length;
}

// ── HTML escaping ───────────────────────────────────────────────────────────
function esc(s) {
  return String(s)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;')
    .replace(/'/g,'&#39;');
}

// ── Injury section ──────────────────────────────────────────────────────────
function renderInjuries() {
  const root = document.getElementById('inj-root');
  if (!INJURIES || !INJURIES.length) { root.innerHTML = ''; return; }

  const statusOrder = {OUT:0, DOUBTFUL:1, GTD:2, RETURNING:3};
  const sorted = [...INJURIES].sort((a,b)=>
    (statusOrder[a.status]??9) - (statusOrder[b.status]??9)
  );

  const rows = sorted.map(inj => `
    <tr>
      <td><strong>${esc(inj.player)}</strong></td>
      <td style="color:var(--text-2)">${esc(inj.team)}</td>
      <td><span class="ib ib-${esc(inj.status)}">${esc(inj.status)}</span></td>
      <td style="color:var(--text-2)">${esc(inj.description)}</td>
    </tr>`).join('');

  root.innerHTML = `
    <div class="sec-hdr">Injury report &mdash; tonight&#39;s teams</div>
    <div class="inj-sec">
      <table class="inj-tbl">
        <thead><tr>
          <th>Player</th><th>Team</th><th>Status</th><th>Injury</th>
        </tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>`;
}

// ── Per-stat performance table ───────────────────────────────────────────────
function renderStatPerf() {
  const root = document.getElementById('stat-perf-root');
  if (!root) return;

  // Backtest baselines (from multi-season research — hardcoded)
  const BT = {
    FG3M: {rate: 73.9, label: 'Strongest', full: '73.9% backtest'},
    PTS:  {rate: 66.2, label: 'Moderate',  full: '66.2% backtest'},
    PRA:  {rate: 60.9, label: 'Weakest',   full: '60.9% backtest'},
  };

  const hasLive = STAT_PERF && Object.keys(STAT_PERF).length > 0;
  const note = hasLive
    ? ''
    : '<div class="sp-note">Live data accumulating &mdash; showing backtest baselines only</div>';

  const rows = ['FG3M','PTS','PRA'].map(stat => {
    const bt   = BT[stat];
    const live = hasLive ? STAT_PERF[stat] : null;

    let wCell, lCell, rCell;
    if (live) {
      const wr = live.win_rate !== null && live.win_rate !== undefined
        ? live.win_rate.toFixed(1)+'%' : '&mdash;';
      const wrCls = live.win_rate >= 65 ? 'sp-good'
                  : live.win_rate >= 55 ? 'sp-ok' : 'sp-bad';
      wCell = `<td>${live.wins}</td>`;
      lCell = `<td>${live.losses}</td>`;
      rCell = `<td class="${wrCls}">${wr}</td>`;
    } else {
      wCell = `<td class="sp-dim">&mdash;</td>`;
      lCell = `<td class="sp-dim">&mdash;</td>`;
      rCell = `<td class="sp-dim">${bt.rate}%</td>`;
    }

    return `<tr>
      <td class="sp-stat">${stat}</td>
      ${wCell}${lCell}${rCell}
      <td class="sp-sig">${bt.label} <span class="sp-bt">(${bt.full})</span></td>
    </tr>`;
  }).join('');

  root.innerHTML = `
    <div class="sec-hdr">Model performance by stat</div>
    <div class="sp-card">
      ${note}
      <table class="sp-tbl">
        <thead><tr>
          <th>Stat</th><th>W</th><th>L</th><th>Win Rate</th><th>Signal Strength</th>
        </tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>`;
}

// ── Picks rendering ─────────────────────────────────────────────────────────
const TIER_ORDER = {STRONG:0, MODERATE:1, WEAK:2};

function renderPicks() {
  // Metrics
  document.getElementById('m-total').textContent  = PICKS.length;
  document.getElementById('m-gaps').textContent   = PICKS.filter(p=>p.gap>0).length;
  document.getElementById('m-action').textContent = PICKS.filter(p=>p.threshold_met).length;
  if (MODEL_RECORD)
    document.getElementById('m-record').textContent =
      MODEL_RECORD.wins+'W–'+MODEL_RECORD.losses+'L';

  // Game strip
  if (GAME_INFO.start_time_et) document.getElementById('g-time').textContent   = GAME_INFO.start_time_et;
  if (GAME_INFO.venue)         document.getElementById('g-venue').textContent  = GAME_INFO.venue;
  if (GAME_INFO.series_record) document.getElementById('g-series').textContent = GAME_INFO.series_record;
  document.getElementById('g-round').textContent = GAME_INFO.round_label || 'NBA Playoffs';

  const root = document.getElementById('picks-root');
  if (!PICKS.length) { root.innerHTML='<p class="empty">No picks today.</p>'; return; }

  // Sort: tier → (gap>0 first) → gap desc
  const sorted = [...PICKS].sort((a,b)=>{
    const td = TIER_ORDER[a.tier] - TIER_ORDER[b.tier];
    if (td!==0) return td;
    const ap = a.gap>0?1:0, bp = b.gap>0?1:0;
    if (ap!==bp) return bp-ap;
    return b.gap - a.gap;
  });

  let html = '';
  for (const tier of ['STRONG','MODERATE','WEAK']) {
    const tp = sorted.filter(p=>p.tier===tier);
    if (!tp.length) continue;
    html += `<div class="sec-hdr">${tier}</div>`;
    tp.forEach((p,i) => { html += pickCard(p, sorted.indexOf(p)); });
  }
  root.innerHTML = html;
}

function pickCard(p, idx) {
  const isHot      = p.status === 'HOT';
  const hasGap     = p.gap > 0;
  const injFlag    = p.injury_flag || '';
  const isOut      = injFlag === 'OUT';
  const streakIcon = isHot ? '&#128293;' : '&#10052;&#65039;';  // 🔥 ❄️

  // Card class — OUT gets its own red border + dim style
  const cls = isOut
    ? 'dim inj-out'
    : (hasGap ? (isHot?'hot':'cold') : 'dim');

  // Injury flag badge shown below player info (non-OUT)
  let injHtml = '';
  if (injFlag && !isOut) {
    const icons = {DOUBTFUL:'&#9888;&#65039;', GTD:'&#9888;&#65039;', RETURNING:'&#8629;'};
    const icon  = icons[injFlag] || '&#9888;&#65039;';
    const desc  = p.injury_desc || injFlag;
    injHtml = `<div class="pick-inj ib-${injFlag}">${icon} ${esc(desc)}</div>`;
  }

  // v2 regression probability pill (only shown when available)
  let regPillHtml = '';
  if (p.regression_probability !== null && p.regression_probability !== undefined) {
    const pct = Math.round(p.regression_probability * 1000) / 10;  // 1 dp
    const pillCls = pct >= 70 ? 'reg-pill-high' : (pct >= 60 ? 'reg-pill-med' : 'reg-pill-low');
    regPillHtml = `<div class="reg-pill ${pillCls}">&#129302; v2 model: ${pct}% regression</div>`;
  }

  // Action button
  let actionHtml = '', watchSub = '';
  if (isOut) {
    // OUT: skip button, no paper trading
    actionHtml = `<span class="abtn no-edge" style="color:#991b1b;font-weight:700">&#9940; Skip &mdash; OUT</span>`;
  } else if (p.threshold_met) {
    const bc    = isHot ? 'under' : 'over';
    const label = isHot ? 'BET UNDER' : 'BET OVER';
    actionHtml = `<button class="abtn ${bc}" onclick="addToPaper(${idx})">${label}</button>`;
  } else if (hasGap) {
    const sub = isHot
      ? `BET UNDER if DK &gt; ${p.threshold}`
      : `BET OVER if DK &lt; ${p.threshold}`;
    actionHtml = `<button class="abtn watch" onclick="addToPaper(${idx})">Watch</button>`;
    watchSub   = `<div class="watch-sub">${sub}</div>`;
  } else {
    actionHtml = `<span class="abtn no-edge">No edge &middot; skip</span>`;
  }

  // Gap colour
  let gapCls = 'gap-neg';
  if (hasGap) gapCls = isHot ? 'gap-hot' : 'gap-cold';
  const gapTxt = hasGap ? `+${p.gap.toFixed(2)}` : p.gap.toFixed(2);

  // Add-to-paper link
  const addLink = hasGap
    ? `<button class="add-paper" onclick="addToPaper(${idx})">+ Add to paper bets</button>`
    : '';

  const threshTxt = p.threshold !== null && p.threshold !== undefined
    ? p.threshold : '&mdash;';

  return `
<div class="pick ${cls}">
  <div class="ptop">
    <div class="pleft">
      <span class="badge b-${p.tier}">${p.tier}</span>
      <span class="spill">${esc(p.stat)}</span>
      <span class="pname">${esc(p.player)}</span>
    </div>
    <div class="paction">
      ${actionHtml}
      ${watchSub}
    </div>
  </div>
  ${injHtml}
  ${regPillHtml}
  <div class="pmeta">
    <span>${esc(p.team)}</span>
    <span>${streakIcon} ${isHot?'Hot':'Cold'} streak &middot; last 3 avg: ${p.recent_avg}</span>
    <span>z = ${p.z_score>=0?'+':''}${p.z_score.toFixed(2)}</span>
    <span>built from ${p.baseline_games}g &middot; ${p.recent_mpg} MPG</span>
  </div>
  <div class="pnums">
    <div class="nc"><div class="nc-lbl">True avg</div><div class="nc-val">${p.fair_line}</div></div>
    <div class="nc"><div class="nc-lbl">DK line</div><div class="nc-val">${p.dk_line}</div></div>
    <div class="nc"><div class="nc-lbl">Gap</div><div class="nc-val ${gapCls}">${gapTxt}</div></div>
    <div class="nc"><div class="nc-lbl">Threshold</div><div class="nc-val thresh">${threshTxt}</div></div>
  </div>
  ${addLink}
</div>`;
}

// ── Paper trading ───────────────────────────────────────────────────────────
function buildDropdown() {
  const sel = document.getElementById('bet-pick');
  sel.innerHTML = '';
  const picks = PICKS.filter(p=>p.gap>0);
  if (!picks.length) {
    sel.innerHTML = '<option value="">No picks with positive gap</option>';
    return;
  }
  picks.forEach((p, i) => {
    const globalIdx = PICKS.indexOf(p);
    const dir   = p.status==='HOT' ? 'UNDER' : 'OVER';
    const last  = p.player.split(' ').pop();
    const label = `${last} — ${p.stat} ${dir} ${p.dk_line}`;
    const opt   = document.createElement('option');
    opt.value   = globalIdx;
    opt.textContent = label;
    sel.appendChild(opt);
  });
}

function addToPaper(idx) {
  const sel = document.getElementById('bet-pick');
  for (const opt of sel.options) {
    if (parseInt(opt.value) === idx) { sel.value = opt.value; break; }
  }
  document.getElementById('bet-pick').scrollIntoView({behavior:'smooth',block:'center'});
}

function placeBet() {
  const sel   = document.getElementById('bet-pick');
  const idx   = parseInt(sel.value);
  if (isNaN(idx) || idx < 0 || idx >= PICKS.length) return;
  const p     = PICKS[idx];
  const stake = parseFloat(document.getElementById('bet-stake').value);
  if (!stake || stake <= 0) { alert('Enter a valid stake.'); return; }
  if (stake > state.bankroll) { alert('Stake exceeds current bankroll.'); return; }

  pushUndo();
  state.bankroll = Math.round((state.bankroll - stake)*100)/100;
  const dir = p.status==='HOT' ? 'UNDER' : 'OVER';
  state.bets.push({
    id:     Date.now() + Math.random(),
    label:  `${p.player.split(' ').pop()} — ${p.stat} ${dir} ${p.dk_line}`,
    player: p.player, stat: p.stat, dir, line: p.dk_line, stake,
    status: 'pending',
  });
  saveState(); renderAll();
}

function gradeBet(id, won) {
  pushUndo();
  const idx = state.bets.findIndex(b=>b.id===id);
  if (idx<0) return;
  const bet    = state.bets[idx];
  const profit = won ? Math.round(bet.stake*(100/110)*100)/100 : -bet.stake;
  if (won) state.bankroll = Math.round((state.bankroll + bet.stake + profit)*100)/100;
  state.bkHistory.push(state.bankroll);
  state.history.push({...bet, profit, result: won?'WIN':'LOSS'});
  state.bets.splice(idx, 1);
  saveState(); renderAll();
}

function clearAllBets() {
  if (!confirm('Clear all pending bets and history? This cannot be undone.')) return;
  pushUndo();
  state.bets = []; state.history = [];
  state.bankroll = state.start;
  state.bkHistory = [state.start];
  saveState(); renderAll();
}

function resetBankroll() {
  const val = prompt('New starting bankroll:', state.start);
  if (val===null) return;
  const amt = parseFloat(val);
  if (isNaN(amt)||amt<=0) { alert('Invalid amount.'); return; }
  pushUndo();
  state.start = amt; state.bankroll = amt;
  state.bets = []; state.history = [];
  state.bkHistory = [amt];
  saveState(); renderAll();
}

function editBankroll() {
  const val = prompt('Edit starting bankroll:', state.start);
  if (val===null) return;
  const amt = parseFloat(val);
  if (isNaN(amt)||amt<=0) { alert('Invalid amount.'); return; }
  pushUndo();
  const diff    = amt - state.start;
  state.start   = amt;
  state.bankroll = Math.round((state.bankroll+diff)*100)/100;
  if (state.bkHistory.length) state.bkHistory[0] = amt;
  saveState(); renderAll();
}

// ── Render helpers ──────────────────────────────────────────────────────────
function renderPaperStats() {
  const pnl    = Math.round((state.bankroll - state.start)*100)/100;
  const wins   = state.history.filter(h=>h.result==='WIN').length;
  const losses = state.history.filter(h=>h.result==='LOSS').length;
  const staked = state.history.reduce((s,h)=>s+h.stake, 0);
  const roi    = staked>0 ? (pnl/staked*100) : null;

  const bd = document.getElementById('bank-disp');
  bd.textContent = '$'+state.bankroll.toFixed(2);
  bd.className   = 'bank-amt '+(state.bankroll>=state.start?'up':'down');

  const pe = document.getElementById('ps-pnl');
  pe.textContent = (pnl>=0?'+$':'-$')+Math.abs(pnl).toFixed(2);
  pe.className   = 'psc-val '+(pnl>=0?'pos':'neg');

  document.getElementById('ps-wins').textContent   = wins;
  document.getElementById('ps-losses').textContent = losses;

  const re = document.getElementById('ps-roi');
  if (roi!==null) {
    re.textContent = (roi>=0?'+':'')+roi.toFixed(1)+'%';
    re.className   = 'psc-val '+(roi>=0?'pos':'neg');
  } else {
    re.textContent='&mdash;'; re.className='psc-val';
  }
}

function renderPending() {
  const sec  = document.getElementById('pend-sec');
  const list = document.getElementById('pend-list');
  if (!state.bets.length) { sec.style.display='none'; return; }
  sec.style.display='';
  list.innerHTML = state.bets.map(b=>`
    <div class="pi">
      <div>
        <div class="pi-desc">${esc(b.label)}</div>
        <div class="pi-meta">Stake: $${b.stake.toFixed(2)} &middot; Win: +$${Math.round(b.stake*100/110*100)/100}</div>
      </div>
      <div class="pi-acts">
        <button class="gbtn win"  onclick="gradeBet(${b.id},true)">Won</button>
        <button class="gbtn loss" onclick="gradeBet(${b.id},false)">Lost</button>
      </div>
    </div>`).join('');
}

function renderHistory() {
  const sec = document.getElementById('hist-sec');
  if (!state.history.length) { sec.style.display='none'; return; }
  sec.style.display='';
  const q = (document.getElementById('hist-search')?.value||'').toLowerCase();
  const rows = state.history.filter(h=>{
    if (!q) return true;
    return h.player.toLowerCase().includes(q)
        || h.stat.toLowerCase().includes(q)
        || h.dir.toLowerCase().includes(q);
  });
  const body = document.getElementById('hist-body');
  if (!rows.length) { body.innerHTML='<p class="empty">No bets match that search.</p>'; return; }
  body.innerHTML=`<table class="htable">
    <thead><tr><th>Pick</th><th>Stake</th><th>Result</th><th>P&amp;L</th></tr></thead>
    <tbody>${rows.map(h=>`
      <tr>
        <td>${esc(h.label)}</td>
        <td>$${h.stake.toFixed(2)}</td>
        <td><span class="${h.result==='WIN'?'bw':'bl'}">${h.result}</span></td>
        <td class="${h.profit>=0?'pp':'pn'}">${h.profit>=0?'+':''}\$${Math.abs(h.profit).toFixed(2)}</td>
      </tr>`).join('')}
    </tbody></table>`;
}

// ── Chart.js ────────────────────────────────────────────────────────────────
let chart = null;
function renderChart() {
  const ctx = document.getElementById('bk-chart');
  if (!ctx) return;
  const labels = state.bkHistory.map((_,i)=>i===0?'Start':'Bet '+i);
  const color  = state.bankroll>=state.start ? '#1D9E75' : '#E24B4A';
  if (chart) { chart.destroy(); chart=null; }
  chart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets:[{
        data: state.bkHistory,
        borderColor: color,
        backgroundColor: color+'22',
        borderWidth: 2,
        pointRadius: 4,
        pointBackgroundColor: color,
        tension: 0.3,
        fill: true,
      }]
    },
    options:{
      responsive:true, maintainAspectRatio:false,
      plugins:{legend:{display:false},tooltip:{callbacks:{
        label: c=>'$'+c.parsed.y.toFixed(2)
      }}},
      scales:{
        x:{grid:{display:false},ticks:{font:{size:11}}},
        y:{ticks:{callback:v=>'$'+v,font:{size:11}},
           grid:{color:'rgba(128,128,128,0.1)'}}
      }
    }
  });
}

function renderAll() {
  renderPaperStats();
  renderPending();
  renderHistory();
  renderChart();
  syncUndoRedo();
}

// ── Init ─────────────────────────────────────────────────────────────────────
loadState();
renderInjuries();
renderStatPerf();
renderPicks();
buildDropdown();
renderAll();
</script>
</body>
</html>
"""

    # Inject Python-computed values into the template
    html = html.replace("__PICKS_JSON__",     picks_json)
    html = html.replace("__GAME_JSON__",      game_json)
    html = html.replace("__RECORD_JSON__",    record_json)
    html = html.replace("__INJURIES_JSON__",  injuries_json)
    html = html.replace("__STAT_PERF_JSON__", stat_perf_json)
    html = html.replace("__DATE__",           date_str)
    html = html.replace("__FIXTURE__",        fixture)

    with open(DASHBOARD_FILE, "w", encoding="utf-8") as fh:
        fh.write(html)

    path = os.path.abspath(DASHBOARD_FILE)
    print(f"  📊 Dashboard written → {path}")
    webbrowser.open(f"file://{path}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print(f"\n{'#'*70}")
    print(f"#  BETTING THE REGRESSION — ODDS COMPARISON  [ESPN/DRAFTKINGS]")
    print(f"#  {TODAY}")
    print(f"{'#'*70}")

    # ------------------------------------------------------------------
    # 1. Load today's NBA predictions (scan back up to 7 days if needed)
    # ------------------------------------------------------------------
    if not os.path.exists(PREDICTIONS_FILE):
        print(f"\n❌  {PREDICTIONS_FILE} not found. Run daily_picks.py first.")
        sys.exit(1)

    all_preds = pd.read_csv(PREDICTIONS_FILE)
    nba_preds = all_preds[all_preds["league"] == "NBA"].copy()

    pred_date = TODAY
    preds = nba_preds[nba_preds["date"].astype(str) == pred_date].copy()

    if preds.empty:
        # Scan back up to 7 days to find the most recent predictions
        for days_back in range(1, 8):
            candidate = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
            preds = nba_preds[nba_preds["date"].astype(str) == candidate].copy()
            if not preds.empty:
                pred_date = candidate
                print(f"\n  ℹ️  No picks for {TODAY} — using most recent picks from {candidate}")
                break

    if preds.empty:
        print(f"\n⚠️  No NBA predictions found in the last 7 days. Run daily_picks.py first.")
        sys.exit(0)

    print(f"\n  Loaded {len(preds)} NBA prediction(s) for {pred_date}:")
    for _, r in preds.iterrows():
        print(f"    {r['player']:22} {r['stat']:5} {r['status']:5} [{r['tier']}]  "
              f"fair={r['fair_line']}  z={r['z_score']}")

    # ------------------------------------------------------------------
    # 2. Find the relevant NBA event on ESPN
    #    Use pred_date when falling back to yesterday's picks so we
    #    find the event that those picks were made for.
    # ------------------------------------------------------------------
    event_search_date = pred_date  # matches the game the picks target
    print(f"\n  Finding NBA event on ESPN for {event_search_date}...")
    try:
        event_id, home, away = find_nba_event(event_search_date)
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
            "player":             player,
            "stat":               stat,
            "status":             status,
            "tier":               tier,
            "fair_line":          fair,
            "bookmaker_line":     book_line,
            "gap":                gap,
            "threshold_met":      met,
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
    print(f"  EDGE REPORT — {pred_date}  |  {fixture_label}")
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
    df.assign(date=pred_date, fixture=fixture_label, source="ESPN/DraftKings") \
      .to_csv(OUTPUT_FILE, index=False)
    print(f"\n{'='*70}")
    print(f"  Saved {len(df)} row(s) to {OUTPUT_FILE}")
    if not edges.empty:
        n_mon = len(edges) - n_met
        print(f"  ⭐ Actionable (threshold met): {n_met}  |  Monitor: {n_mon}")
    print()

    # ------------------------------------------------------------------
    # 9. Fetch injury report for tonight's teams (always fresh)
    # ------------------------------------------------------------------
    print(f"\n  Fetching injury report for {fixture_label}...")
    pick_lookup, all_injuries = fetch_injury_report(home, away)
    if all_injuries:
        print(f"  {len(all_injuries)} player(s) on injury report:")
        for inj in all_injuries:
            print(f"    {inj['status']:9}  {inj['player']} ({inj['team']}) — {inj['description']}")
    else:
        print(f"  No injuries reported for tonight's teams.")

    # ------------------------------------------------------------------
    # 10. Build enriched picks for dashboard (with injury annotations)
    # ------------------------------------------------------------------
    picks_for_dash = []
    for r in results:
        mask = (preds["player"] == r["player"]) & (preds["stat"] == r["stat"])
        matching = preds[mask]
        if matching.empty:
            continue
        row = matching.iloc[0]

        team_id        = int(row["team_id"])
        team_name      = NBA_TEAMS.get(team_id, "NBA")
        baseline_games = parse_baseline_games(str(row["baseline_used"]))
        direction      = "UNDER" if r["status"] == "HOT" else "OVER"
        thresh         = parse_threshold(r["bet_recommendation"])

        # Fuzzy-match player against injury lookup
        norm_player = normalise_name(r["player"])
        inj_info    = None
        best_score  = 0.0
        for key, val in pick_lookup.items():
            s = difflib.SequenceMatcher(None, norm_player, key).ratio()
            if s > best_score:
                best_score, inj_info = s, val
        if best_score < 0.75:
            inj_info = None

        injury_flag = ""
        injury_desc = ""
        if inj_info:
            injury_flag = inj_info["status"]
            desc_raw    = inj_info["description"]
            if injury_flag == "OUT":
                injury_desc = f"OUT — {desc_raw}" if desc_raw else "OUT"
            elif injury_flag == "DOUBTFUL":
                injury_desc = f"DOUBTFUL — high DNP risk ({desc_raw})" if desc_raw else "DOUBTFUL — high DNP risk"
            elif injury_flag == "GTD":
                injury_desc = f"GTD — confirm active before betting ({desc_raw})" if desc_raw else "GTD — confirm active"

        # Include regression_probability from CSV if available (v2 model)
        reg_prob = None
        if "regression_probability" in row and pd.notna(row["regression_probability"]):
            try:
                reg_prob = round(float(row["regression_probability"]), 3)
            except (ValueError, TypeError):
                reg_prob = None

        picks_for_dash.append({
            "player":                 r["player"],
            "team":                   team_name,
            "stat":                   r["stat"],
            "status":                 r["status"],
            "tier":                   r["tier"],
            "fair_line":              r["fair_line"],
            "dk_line":                r["bookmaker_line"],
            "gap":                    r["gap"],
            "z_score":                round(float(row["z_score"]), 2),
            "bet_recommendation":     r["bet_recommendation"],
            "recent_avg":             round(float(row["recent_avg"]), 2),
            "recent_mpg":             round(float(row["recent_mpg"]), 1),
            "baseline_games":         baseline_games,
            "threshold":              thresh,
            "threshold_met":          bool(r["threshold_met"]),
            "direction":              direction,
            "injury_flag":            injury_flag,
            "injury_desc":            injury_desc,
            "regression_probability": reg_prob,
        })

    # ------------------------------------------------------------------
    # 11. Game metadata for dashboard
    # ------------------------------------------------------------------
    print("  Fetching game metadata for dashboard...")
    game_info = fetch_game_info(event_id, home, away)
    if game_info.get("start_time_et"):
        print(f"    Tip-off: {game_info['start_time_et']}")
    if game_info.get("venue"):
        print(f"    Venue:   {game_info['venue']}")
    if game_info.get("series_record"):
        print(f"    Series:  {game_info['series_record']}")

    # ------------------------------------------------------------------
    # 12. Model record (if available)
    # ------------------------------------------------------------------
    model_record = None
    if os.path.exists(PERFORMANCE_FILE):
        try:
            perf = pd.read_csv(PERFORMANCE_FILE)
            if not perf.empty:
                last = perf.iloc[-1]
                wins   = int(last.get("cumulative_wins",   last.get("wins",   0)))
                losses = int(last.get("cumulative_losses", last.get("losses", 0)))
                model_record = {"wins": wins, "losses": losses}
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 13. Generate dashboard & open browser
    # ------------------------------------------------------------------
    generate_dashboard(picks_for_dash, game_info, pred_date, model_record,
                       all_injuries=all_injuries)


if __name__ == "__main__":
    main()
