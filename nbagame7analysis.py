from nba_api.stats.endpoints import playergamelog, commonteamroster
from nba_api.stats.static import players
import pandas as pd
import time

SAS_ID = 1610612759
OKC_ID = 1610612760
SEASON = "2025-26"

STATS_TO_CHECK = ["FG3M", "PTS", "PRA"]


def get_team_roster(team_id, season):
    roster = commonteamroster.CommonTeamRoster(team_id=team_id, season=season)
    df = roster.get_data_frames()[0]
    return df[["PLAYER", "PLAYER_ID", "POSITION"]].rename(columns={"PLAYER": "PLAYER_NAME"})


def get_playoff_only_gamelog(player_id, season):
    """Pull ONLY playoff games for this season."""
    try:
        po_log = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=season,
            season_type_all_star="Playoffs"
        )
        df = po_log.get_data_frames()[0]
        if len(df) == 0:
            return None
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], format="mixed")
        df = df.sort_values("GAME_DATE").reset_index(drop=True)
        df["PRA"] = df["PTS"] + df["REB"] + df["AST"]
        return df
    except Exception:
        return None


def get_regular_season_gamelog(player_id, season):
    """Pull ONLY regular season games."""
    try:
        rs_log = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=season,
            season_type_all_star="Regular Season"
        )
        df = rs_log.get_data_frames()[0]
        if len(df) == 0:
            return None
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], format="mixed")
        df = df.sort_values("GAME_DATE").reset_index(drop=True)
        df["PRA"] = df["PTS"] + df["REB"] + df["AST"]
        return df
    except Exception:
        return None


def analyze_player_dual(player_name, player_id, season=SEASON):
    """
    Build separate baselines for regular season and playoffs.
    Compute hot streak status using recent 3 games (playoff-only if 5+ playoff games).
    """
    rs_df = get_regular_season_gamelog(player_id, season)
    po_df = get_playoff_only_gamelog(player_id, season)
    
    if rs_df is None or len(rs_df) < 15:
        return None
    
    rs_games = len(rs_df)
    po_games = len(po_df) if po_df is not None else 0
    
    # Decide which baseline to use
    # If 5+ playoff games, use playoff-only baseline (it's their current role)
    # Otherwise, use regular season baseline
    use_playoff_baseline = po_games >= 5
    
    results = {
        "player": player_name,
        "rs_games": rs_games,
        "po_games": po_games,
        "baseline_used": "playoffs" if use_playoff_baseline else "regular_season"
    }
    
    # Recent form is always the last 3 playoff games (or all playoff games if fewer than 3)
    if po_games >= 1:
        recent_df = po_df.tail(min(3, po_games))
        recent_minutes = po_df["MIN"].tail(min(3, po_games)).mean()
    else:
        # Hasn't played playoffs — use last 3 regular season games
        recent_df = rs_df.tail(3)
        recent_minutes = rs_df["MIN"].tail(3).mean()
    
    # Baseline calculation
    baseline_df = po_df if use_playoff_baseline else rs_df
    baseline_minutes = baseline_df["MIN"].mean()
    
    for stat in STATS_TO_CHECK:
        if stat not in baseline_df.columns:
            continue
        
        season_avg = baseline_df[stat].mean()
        season_std = baseline_df[stat].std()
        recent_avg = recent_df[stat].mean()
        
        z_score = (recent_avg - season_avg) / season_std if season_std > 0 else 0
        
        if z_score > 1.0:
            status = "🔥 HOT — FADE CANDIDATE"
        elif z_score < -1.0:
            status = "❄️ COLD — possible OVER candidate"
        else:
            status = "Normal"
        
        # Flags
        flags = []
        minutes_trend = recent_minutes - baseline_minutes
        if abs(minutes_trend) > 4:
            flags.append(f"Minutes shift: {minutes_trend:+.1f} MPG vs baseline")
        if po_games < 5:
            flags.append("Limited playoff sample — using regular season baseline")
        if recent_minutes < 15:
            flags.append("Recent MPG <15 — limited prop value")
        
        results[stat] = {
            "baseline_avg": round(season_avg, 2),
            "baseline_std": round(season_std, 2),
            "recent_avg": round(recent_avg, 2),
            "z_score": round(z_score, 2),
            "fair_line": round(season_avg, 2),
            "status": status,
            "flags": flags,
            "recent_mpg": round(recent_minutes, 1),
            "baseline_mpg": round(baseline_minutes, 1)
        }
    
    return results


# ==============================================================================
# RUN ANALYSIS
# ==============================================================================

print("=" * 80)
print("GAME 7 FULL ROSTER ANALYSIS — Spurs vs Thunder")
print(f"Using 2025-26 data only. Playoff baseline preferred when player has 5+ PO games.")
print("=" * 80)

for team_id, team_name in [(SAS_ID, "SAN ANTONIO SPURS"), (OKC_ID, "OKLAHOMA CITY THUNDER")]:
    print(f"\n\n{'=' * 80}")
    print(f"  {team_name}")
    print(f"{'=' * 80}")
    
    roster = get_team_roster(team_id, SEASON)
    print(f"Analyzing {len(roster)} players (this takes ~1-2 min)...\n")
    
    team_results = []
    for _, row in roster.iterrows():
        try:
            result = analyze_player_dual(row["PLAYER_NAME"], row["PLAYER_ID"])
            if result and result.get("rs_games", 0) >= 15:
                team_results.append(result)
        except Exception:
            pass
        time.sleep(0.6)
    
    # Sort by recent MPG so we see players who actually play first
    team_results = sorted(team_results, key=lambda x: x.get("FG3M", {}).get("recent_mpg", 0), reverse=True)
    
    # === FULL SUMMARY ACROSS ALL THREE STATS ===
    for stat in STATS_TO_CHECK:
        print(f"\n--- {stat} | All players (sorted by recent MPG) ---")
        rows = []
        for r in team_results:
            if stat in r:
                rows.append({
                    "player": r["player"],
                    "po_games": r["po_games"],
                    "baseline": r["baseline_used"][:3],
                    "baseline_avg": r[stat]["baseline_avg"],
                    "recent_avg": r[stat]["recent_avg"],
                    "z_score": r[stat]["z_score"],
                    "recent_mpg": r[stat]["recent_mpg"],
                    "status": r[stat]["status"]
                })
        if rows:
            print(pd.DataFrame(rows).to_string(index=False))
    
    # === HOT STREAK FADE CANDIDATES (refined) ===
    print(f"\n\n--- 🔥 HOT STREAK FADE CANDIDATES (RECENT MPG ≥ 15) ---\n")
    found_any = False
    for r in team_results:
        for stat in STATS_TO_CHECK:
            if stat not in r:
                continue
            d = r[stat]
            # Refined criteria: hot AND playing real minutes
            if "HOT" in d["status"] and d["recent_mpg"] >= 15:
                found_any = True
                print(f"  ⚠️  {r['player']} — {stat}")
                print(f"      Baseline ({r['baseline_used']}): {d['baseline_avg']}")
                print(f"      Recent: {d['recent_avg']} (z-score {d['z_score']})")
                print(f"      Recent MPG: {d['recent_mpg']} vs baseline {d['baseline_mpg']}")
                print(f"      YOUR FAIR LINE: {d['fair_line']}")
                if d['flags']:
                    print(f"      Flags: {', '.join(d['flags'])}")
                print()
    
    if not found_any:
        print("  No high-confidence fade candidates with recent MPG ≥ 15.\n")
    
    # === COLD STREAK CANDIDATES ===
    print(f"\n--- ❄️ COLD STREAK BOUNCE CANDIDATES (RECENT MPG ≥ 15) ---\n")
    found_any = False
    for r in team_results:
        for stat in STATS_TO_CHECK:
            if stat not in r:
                continue
            d = r[stat]
            if "COLD" in d["status"] and d["recent_mpg"] >= 15:
                found_any = True
                print(f"  ⚠️  {r['player']} — {stat}")
                print(f"      Baseline ({r['baseline_used']}): {d['baseline_avg']}")
                print(f"      Recent: {d['recent_avg']} (z-score {d['z_score']})")
                print(f"      Recent MPG: {d['recent_mpg']} vs baseline {d['baseline_mpg']}")
                print(f"      YOUR FAIR LINE: {d['fair_line']}")
                if d['flags']:
                    print(f"      Flags: {', '.join(d['flags'])}")
                print()
    
    if not found_any:
        print("  No high-confidence bounce candidates with recent MPG ≥ 15.\n")


print("\n" + "=" * 80)
print("IMPORTANT REMINDERS:")
print("- Cold streaks have weaker historical signal than hot streaks (44% vs 92%)")
print("- Game 7 reduces rotation depth; expect heavy minutes from starters")
print("- Compare 'YOUR FAIR LINE' to today's actual sportsbook lines")
print("- A fade candidate is one where the sportsbook line is well above your fair line")
print("=" * 80)