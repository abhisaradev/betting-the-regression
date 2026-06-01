from nba_api.stats.endpoints import playergamelog, commonteamroster
from nba_api.stats.static import players, teams
import pandas as pd
import time

# Team IDs
SAS_ID = 1610612759  # San Antonio Spurs
OKC_ID = 1610612760  # Oklahoma City Thunder
SEASON = "2025-26"  # current season

# We'll analyze these stats
STATS_TO_CHECK = ["FG3M", "PTS", "PRA"]


def get_team_roster(team_id, season):
    """Pull the current active roster for a team."""
    roster = commonteamroster.CommonTeamRoster(team_id=team_id, season=season)
    df = roster.get_data_frames()[0]
    return df[["PLAYER", "PLAYER_ID", "POSITION"]].rename(columns={"PLAYER": "PLAYER_NAME"})


def get_combined_gamelog(player_id, season):
    """Pull both regular season and playoff game logs combined."""
    # Regular season
    rs_log = playergamelog.PlayerGameLog(
        player_id=player_id, 
        season=season,
        season_type_all_star="Regular Season"
    )
    rs_df = rs_log.get_data_frames()[0]
    
    # Playoffs
    try:
        po_log = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=season,
            season_type_all_star="Playoffs"
        )
        po_df = po_log.get_data_frames()[0]
    except Exception:
        po_df = pd.DataFrame()
    
    # Combine — playoffs come AFTER regular season chronologically
    combined = pd.concat([rs_df, po_df], ignore_index=True)
    
    if len(combined) == 0:
        return None
    
    # Sort chronologically
    combined["GAME_DATE"] = pd.to_datetime(combined["GAME_DATE"], format="mixed")
    combined = combined.sort_values("GAME_DATE").reset_index(drop=True)
    
    # Build combo stats
    combined["PRA"] = combined["PTS"] + combined["REB"] + combined["AST"]
    
    return combined


def check_hot_streak(player_name, player_id, season=SEASON):
    """
    Check if a player is currently in a hot streak for any of our tracked stats.
    Returns a dict per stat showing baseline, recent average, z-score, and fair line.
    """
    df = get_combined_gamelog(player_id, season)
    if df is None or len(df) < 15:
        return None
    
    results = {"player": player_name, "games_played": len(df)}
    
    for stat in STATS_TO_CHECK:
        if stat not in df.columns:
            continue
            
        season_avg = df[stat].mean()
        season_std = df[stat].std()
        
        # Most recent 3 games (the rolling window for "hot streak")
        recent_3 = df[stat].tail(3).mean()
        recent_5 = df[stat].tail(5).mean()
        
        # Z-score for current state
        z_score = (recent_3 - season_avg) / season_std if season_std > 0 else 0
        
        # Recent minutes trend (confounder check)
        recent_mpg = df["MIN"].tail(3).mean()
        season_mpg = df["MIN"].mean()
        minutes_trend = recent_mpg - season_mpg
        
        # Classify the streak status
        if z_score > 1.0:
            status = "🔥 HOT — FADE CANDIDATE"
        elif z_score < -1.0:
            status = "❄️ COLD"
        else:
            status = "Normal"
        
        # Filter check
        flags = []
        if minutes_trend > 2:
            flags.append("MINUTES UP — caution, role may be expanding")
        elif minutes_trend < -2:
            flags.append("MINUTES DOWN — caution, role may be shrinking")
        
        results[stat] = {
            "season_avg": round(season_avg, 2),
            "season_std": round(season_std, 2),
            "recent_3_avg": round(recent_3, 2),
            "recent_5_avg": round(recent_5, 2),
            "z_score": round(z_score, 2),
            "minutes_trend": round(minutes_trend, 2),
            "status": status,
            "fair_line": round(season_avg, 2),
            "flags": flags
        }
    
    return results


# ==============================================================================
# RUN ANALYSIS
# ==============================================================================

print("=" * 70)
print("GAME 7 HOT HAND FADER ANALYSIS — Spurs vs Thunder")
print(f"Date: May 30, 2026 — OKC home")
print("=" * 70)

for team_id, team_name in [(SAS_ID, "SAN ANTONIO SPURS"), (OKC_ID, "OKLAHOMA CITY THUNDER")]:
    print(f"\n\n{'='*70}")
    print(f"  {team_name}")
    print(f"{'='*70}")
    
    roster = get_team_roster(team_id, SEASON)
    print(f"Pulling data for {len(roster)} players...\n")
    
    team_results = []
    for _, row in roster.iterrows():
        try:
            result = check_hot_streak(row["PLAYER_NAME"], row["PLAYER_ID"])
            if result and result.get("games_played", 0) >= 15:
                team_results.append(result)
        except Exception:
            pass
        time.sleep(0.6)
    
    # Show fade candidates first
    print(f"\n--- HOT STREAK FADE CANDIDATES ---\n")
    fade_candidates_found = False
    for result in team_results:
        for stat in STATS_TO_CHECK:
            if stat not in result:
                continue
            if "HOT" in result[stat]["status"]:
                fade_candidates_found = True
                print(f"⚠️  {result['player']} — {stat}")
                print(f"    Recent 3 games avg: {result[stat]['recent_3_avg']} (z-score {result[stat]['z_score']})")
                print(f"    Season baseline: {result[stat]['season_avg']} (std {result[stat]['season_std']})")
                print(f"    YOUR FAIR LINE: {result[stat]['fair_line']}")
                if result[stat]['flags']:
                    print(f"    ⚠️  Flags: {', '.join(result[stat]['flags'])}")
                print()
    
    if not fade_candidates_found:
        print("  No players currently in hot streaks meeting our criteria.\n")
    
    # Also show overall summary so you can spot mid-tier hot streaks
    print(f"\n--- FULL PLAYER SUMMARY (3PM) ---")
    summary_rows = []
    for result in team_results:
        if "FG3M" in result:
            summary_rows.append({
                "player": result["player"],
                "games": result["games_played"],
                "season_avg": result["FG3M"]["season_avg"],
                "recent_3": result["FG3M"]["recent_3_avg"],
                "z_score": result["FG3M"]["z_score"],
                "status": result["FG3M"]["status"]
            })
    
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows).sort_values("z_score", ascending=False)
        print(summary_df.to_string(index=False))

print("\n\n" + "="*70)
print("DONE — compare 'YOUR FAIR LINE' to today's PrizePicks/DraftKings lines.")
print("If a sportsbook is offering an OVER line above your fair line by >0.5,")
print("the model suggests betting UNDER.")
print("="*70)