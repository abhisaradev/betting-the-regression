"""
Daily NBA + WNBA Hot Hand Fader Tool

Runs each day to:
1. Auto-check yesterday's predictions against actual results
2. Identify today's hot/cold streak candidates across all NBA + WNBA games
3. Save predictions to CSV for future grading

Usage: python daily_picks.py
"""

from nba_api.stats.endpoints import (
    playergamelog, commonteamroster, scoreboardv3
)
from nba_api.stats.static import players
import pandas as pd
from datetime import datetime, timedelta
import os
import time

# ==============================================================================
# CONSTANTS & CONFIG
# ==============================================================================

NBA_LEAGUE_ID = "00"
WNBA_LEAGUE_ID = "10"

STATS_TO_CHECK = ["FG3M", "PTS", "PRA"]
HOT_THRESHOLD = 1.0
COLD_THRESHOLD = -1.0
MIN_RECENT_MPG = 15

PREDICTIONS_FILE = "daily_predictions.csv"
GRADED_FILE = "graded_predictions.csv"
PERFORMANCE_FILE = "model_performance.csv"


# ==============================================================================
# DATA PULLING
# ==============================================================================

def get_games_for_date(date_str, league_id):
    """Get list of games scheduled for a specific date in a league (using V3)."""
    try:
        scoreboard = scoreboardv3.ScoreboardV3(
            game_date=date_str,
            league_id=league_id
        )
        games_data = scoreboard.get_dict()
        games_list = games_data.get("scoreboard", {}).get("games", [])
        
        games = []
        for game in games_list:
            status_text = game.get("gameStatusText", "Unknown")
            # Skip games that have already started or finished
            if any(s in status_text.lower() for s in ["final", "in progress", "qtr", "half"]):
                continue
            games.append({
                "game_id": game["gameId"],
                "home_team_id": game["homeTeam"]["teamId"],
                "away_team_id": game["awayTeam"]["teamId"],
                "game_status": status_text
            })
        return games
    except Exception as e:
        print(f"  Error pulling games for {date_str}: {str(e)[:80]}")
        return []


def get_team_roster_safe(team_id, season):
    """Get current roster."""
    try:
        roster = commonteamroster.CommonTeamRoster(team_id=team_id, season=season)
        df = roster.get_data_frames()[0]
        return df[["PLAYER", "PLAYER_ID"]].rename(columns={"PLAYER": "PLAYER_NAME"})
    except Exception as e:
        print(f"    Roster fetch failed for team {team_id}: {str(e)[:80]}")
        return pd.DataFrame()


def get_player_gamelog(player_id, season, season_type):
    """Pull a player's game log for a specific season type."""
    try:
        log = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=season,
            season_type_all_star=season_type
        )
        df = log.get_data_frames()[0]
        if len(df) == 0:
            return None
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], format="mixed")
        df = df.sort_values("GAME_DATE").reset_index(drop=True)
        df["PRA"] = df["PTS"] + df["REB"] + df["AST"]
        return df
    except Exception:
        return None


# ==============================================================================
# MODEL ANALYSIS
# ==============================================================================

def analyze_player(player_name, player_id, season):
    """
    Hot-hand fader analysis using playoff-only baseline if 5+ PO games.
    """
    rs_df = get_player_gamelog(player_id, season, "Regular Season")
    po_df = get_player_gamelog(player_id, season, "Playoffs")
    
    if rs_df is None or len(rs_df) < 15:
        return None
    
    po_games = len(po_df) if po_df is not None else 0
    use_playoff_baseline = po_games >= 5
    
    baseline_df = po_df if use_playoff_baseline else rs_df
    baseline_label = "playoffs" if use_playoff_baseline else "regular_season"
    
    if po_games >= 1:
        recent_df = po_df.tail(min(3, po_games))
        recent_minutes = po_df["MIN"].tail(min(3, po_games)).mean()
    else:
        recent_df = rs_df.tail(3)
        recent_minutes = rs_df["MIN"].tail(3).mean()
    
    baseline_minutes = baseline_df["MIN"].mean()
    
    results = {
        "player": player_name,
        "player_id": int(player_id),
        "rs_games": len(rs_df),
        "po_games": po_games,
        "baseline_used": baseline_label,
        "recent_mpg": round(recent_minutes, 1),
        "baseline_mpg": round(baseline_minutes, 1)
    }
    
    for stat in STATS_TO_CHECK:
        if stat not in baseline_df.columns:
            continue
        
        season_avg = baseline_df[stat].mean()
        season_std = baseline_df[stat].std()
        recent_avg = recent_df[stat].mean()
        z_score = (recent_avg - season_avg) / season_std if season_std > 0 else 0
        
        if z_score > HOT_THRESHOLD:
            status = "HOT"
        elif z_score < COLD_THRESHOLD:
            status = "COLD"
        else:
            status = "NORMAL"
        
        results[f"{stat}_baseline"] = round(season_avg, 2)
        results[f"{stat}_recent"] = round(recent_avg, 2)
        results[f"{stat}_zscore"] = round(z_score, 2)
        results[f"{stat}_status"] = status
        results[f"{stat}_fair_line"] = round(season_avg, 2)
    
    return results


# ==============================================================================
# GRADING
# ==============================================================================

def grade_predictions_from_date(date_str):
    """Grade yesterday's predictions against actual results."""
    if not os.path.exists(PREDICTIONS_FILE):
        return
    
    all_preds = pd.read_csv(PREDICTIONS_FILE)
    target_preds = all_preds[all_preds["date"] == date_str]
    
    if len(target_preds) == 0:
        return
    
    print(f"\nGrading predictions from {date_str}...")
    
    graded_rows = []
    
    for _, pred in target_preds.iterrows():
        player_name = pred["player"]
        season = pred["season"]
        
        player_dict = players.find_players_by_full_name(player_name)
        if not player_dict:
            continue
        player_id = player_dict[0]["id"]
        
        actual = None
        for season_type in ["Playoffs", "Regular Season"]:
            try:
                df = get_player_gamelog(player_id, season, season_type)
                if df is not None:
                    match = df[df["GAME_DATE"].dt.strftime("%Y-%m-%d") == date_str]
                    if len(match) > 0:
                        actual = match.iloc[0]
                        break
            except Exception:
                pass
            time.sleep(0.3)
        
        if actual is None:
            graded_rows.append({**pred.to_dict(), "actual": None, "result": "DNP"})
            continue
        
        stat = pred["stat"]
        actual_value = float(actual[stat]) if stat in actual else None
        
        result = "PUSH"
        if actual_value is not None:
            recent = float(pred["recent_avg"])
            
            if pred["status"] == "HOT":
                if actual_value < recent - 0.5:
                    result = "WIN"
                elif actual_value > recent + 0.5:
                    result = "LOSS"
            elif pred["status"] == "COLD":
                if actual_value > recent + 0.5:
                    result = "WIN"
                elif actual_value < recent - 0.5:
                    result = "LOSS"
        
        graded_rows.append({
            **pred.to_dict(),
            "actual": actual_value,
            "result": result
        })
    
    graded_df = pd.DataFrame(graded_rows)
    
    if os.path.exists(GRADED_FILE):
        existing = pd.read_csv(GRADED_FILE)
        graded_df = pd.concat([existing, graded_df], ignore_index=True)
    graded_df.to_csv(GRADED_FILE, index=False)
    
    results = graded_df[graded_df["date"] == date_str]
    if len(results) > 0:
        wins = (results["result"] == "WIN").sum()
        losses = (results["result"] == "LOSS").sum()
        pushes = (results["result"].isin(["PUSH", "DNP"])).sum()
        played = wins + losses
        win_rate = (100 * wins / played) if played > 0 else 0
        
        print(f"  {date_str} results: {wins}W - {losses}L - {pushes} pushes/DNP")
        if played > 0:
            print(f"  Win rate (excluding pushes/DNP): {win_rate:.1f}%")
        
        perf_row = {
            "date": date_str,
            "predictions": len(results),
            "wins": int(wins),
            "losses": int(losses),
            "pushes_dnp": int(pushes),
            "win_rate": round(win_rate, 1)
        }
        perf_df = pd.DataFrame([perf_row])
        if os.path.exists(PERFORMANCE_FILE):
            existing_perf = pd.read_csv(PERFORMANCE_FILE)
            perf_df = pd.concat([existing_perf, perf_df], ignore_index=True)
        perf_df.to_csv(PERFORMANCE_FILE, index=False)


# ==============================================================================
# MAIN
# ==============================================================================

def run_for_league(league_name, league_id, season, today_str):
    """Run the daily analysis for one league."""
    print(f"\n{'='*70}")
    print(f"  {league_name} — {today_str}")
    print(f"{'='*70}")
    
    games = get_games_for_date(today_str, league_id)
    if not games:
        print(f"  No upcoming games today.")
        return []
    
    print(f"  {len(games)} game(s) tonight")
    
    all_picks = []
    
    for game in games:
        team_ids = [game["home_team_id"], game["away_team_id"]]
        for team_id in team_ids:
            roster = get_team_roster_safe(team_id, season)
            
            for _, player_row in roster.iterrows():
                try:
                    result = analyze_player(
                        player_row["PLAYER_NAME"],
                        player_row["PLAYER_ID"],
                        season
                    )
                except Exception:
                    result = None
                time.sleep(0.4)
                
                if not result:
                    continue
                if result["recent_mpg"] < MIN_RECENT_MPG:
                    continue
                
                for stat in STATS_TO_CHECK:
                    status = result.get(f"{stat}_status", "NORMAL")
                    if status in ("HOT", "COLD"):
                        all_picks.append({
                            "date": today_str,
                            "league": league_name,
                            "season": season,
                            "player": result["player"],
                            "team_id": int(team_id),
                            "po_games": result["po_games"],
                            "baseline_used": result["baseline_used"],
                            "stat": stat,
                            "status": status,
                            "baseline_avg": result[f"{stat}_baseline"],
                            "recent_avg": result[f"{stat}_recent"],
                            "z_score": result[f"{stat}_zscore"],
                            "fair_line": result[f"{stat}_fair_line"],
                            "recent_mpg": result["recent_mpg"]
                        })
    
    return all_picks


def main():
    today = datetime.now().strftime("%Y-%m-%d")
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    
    print(f"\n{'#'*70}")
    print(f"#  HOT HAND FADER DAILY TOOL")
    print(f"#  Running on {today}")
    print(f"#  Auto-grading yesterday ({yesterday})")
    print(f"{'#'*70}")
    
    # Check if we already have predictions for today (prevent duplicates)
    if os.path.exists(PREDICTIONS_FILE):
        existing = pd.read_csv(PREDICTIONS_FILE)
        if today in existing["date"].astype(str).values:
            print(f"\n⚠️  Predictions already exist for {today}.")
            print(f"   Re-running would create duplicates.")
            response = input("   Continue anyway and replace today's picks? (y/n): ").strip().lower()
            if response != "y":
                print("   Skipping today's analysis.")
                # Still grade yesterday if not already done
                grade_predictions_from_date(yesterday)
                return
            # Remove today's existing rows before re-running
            existing = existing[existing["date"].astype(str) != today]
            existing.to_csv(PREDICTIONS_FILE, index=False)
    
    # STEP 1: Grade yesterday's predictions
    grade_predictions_from_date(yesterday)
    
    # STEP 2: Generate today's picks
    nba_picks = run_for_league("NBA", NBA_LEAGUE_ID, "2025-26", today)
    wnba_picks = run_for_league("WNBA", WNBA_LEAGUE_ID, "2026", today)
    all_picks = nba_picks + wnba_picks
    
    if not all_picks:
        print(f"\n{'='*70}")
        print(f"  NO PICKS FOR TODAY ({today})")
        print(f"{'='*70}")
        print(f"\n  Either no games scheduled, or no players met our criteria.")
        print(f"  NBA Finals starts Wednesday June 3 (SAS vs NYK).")
        return
    
    picks_df = pd.DataFrame(all_picks)
    picks_df["abs_z"] = picks_df["z_score"].abs()
    picks_df = picks_df.sort_values(["status", "abs_z"], ascending=[True, False])
    
    # Save
    if os.path.exists(PREDICTIONS_FILE):
        existing = pd.read_csv(PREDICTIONS_FILE)
        save_df = pd.concat([existing, picks_df.drop(columns=["abs_z"])], ignore_index=True)
    else:
        save_df = picks_df.drop(columns=["abs_z"])
    save_df.to_csv(PREDICTIONS_FILE, index=False)
    
    print(f"\n{'='*70}")
    print(f"  TODAY'S PICKS — {today}")
    print(f"{'='*70}\n")
    
    hot_picks = picks_df[picks_df["status"] == "HOT"]
    cold_picks = picks_df[picks_df["status"] == "COLD"]
    
    if len(hot_picks) > 0:
        print(f"\n🔥 HOT STREAK FADES (bet UNDER):\n")
        for _, pick in hot_picks.iterrows():
            print(f"  {pick['player']} ({pick['league']}) — {pick['stat']}")
            print(f"    Baseline: {pick['baseline_avg']} | Recent: {pick['recent_avg']} | z={pick['z_score']}")
            print(f"    YOUR FAIR LINE: {pick['fair_line']} (MPG: {pick['recent_mpg']})")
            print()
    
    if len(cold_picks) > 0:
        print(f"\n❄️ COLD STREAK BOUNCES (bet OVER, weaker signal):\n")
        for _, pick in cold_picks.iterrows():
            print(f"  {pick['player']} ({pick['league']}) — {pick['stat']}")
            print(f"    Baseline: {pick['baseline_avg']} | Recent: {pick['recent_avg']} | z={pick['z_score']}")
            print(f"    YOUR FAIR LINE: {pick['fair_line']} (MPG: {pick['recent_mpg']})")
            print()
    
    print(f"\nSaved {len(picks_df)} picks to {PREDICTIONS_FILE}")
    
    if os.path.exists(PERFORMANCE_FILE):
        perf = pd.read_csv(PERFORMANCE_FILE)
        total_wins = perf["wins"].sum()
        total_losses = perf["losses"].sum()
        if total_wins + total_losses > 0:
            overall = 100 * total_wins / (total_wins + total_losses)
            print(f"\n📊 MODEL TRACK RECORD: {total_wins}W - {total_losses}L ({overall:.1f}%)")


if __name__ == "__main__":
    main()