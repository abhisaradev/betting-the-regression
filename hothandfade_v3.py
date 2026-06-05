"""
hothandfade_v3.py — Full multi-season backtest pipeline for Hot Hand Fader.
Tests mean reversion hypothesis across 2022-25 NBA seasons with cached steps.
Validates 73.9% win rate across 2,118 hot streak events over 3 seasons.
Run once to reproduce backtest results: python hothandfade_v3.py
"""

from nba_api.stats.endpoints import (
    playergamelog, leaguedashplayerstats,
    commonplayerinfo, leaguedashteamstats
)
from nba_api.stats.static import players
import pandas as pd
import os
import time

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_role_players(season="2024-25", min_mpg=15, max_mpg=32, min_3pa=2.0, min_gp=30):
    """Pull all players who fit the 'role player' profile."""
    print(f"Pulling role players for {season}...")
    league_stats = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        per_mode_detailed="PerGame"
    )
    df = league_stats.get_data_frames()[0]
    role_players_df = df[
        (df["MIN"] >= min_mpg) &
        (df["MIN"] <= max_mpg) &
        (df["FG3A"] >= min_3pa) &
        (df["GP"] >= min_gp)
    ]
    print(f"  Found {len(role_players_df)} role players")
    return role_players_df[["PLAYER_ID", "PLAYER_NAME", "MIN", "FG3A", "FG3M", "GP"]].reset_index(drop=True)


def analyze_player(player_name, season="2024-25", stat="FG3M",
                   hot_threshold=1.0, cold_threshold=-1.0):
    """Analyze mean reversion pattern for one player."""
    player_dict = players.find_players_by_full_name(player_name)
    if not player_dict:
        return None
    player_id = player_dict[0]["id"]

    gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season)
    df = gamelog.get_data_frames()[0]
    if len(df) < 20:
        return None

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], format="mixed")
    df = df.sort_values("GAME_DATE").reset_index(drop=True)

    season_avg = df[stat].mean()
    season_std = df[stat].std()

    df["rolling_3"] = df[stat].rolling(window=3).mean()
    df["z_score"] = (df["rolling_3"] - season_avg) / season_std
    df["next_game"] = df[stat].shift(-1)
    df = df.dropna(subset=["next_game", "z_score"])

    hot = df[df["z_score"] > hot_threshold]
    cold = df[df["z_score"] < cold_threshold]
    normal = df[(df["z_score"] >= cold_threshold) & (df["z_score"] <= hot_threshold)]

    return {
        "player": player_name,
        "season_avg": round(season_avg, 2),
        "hot_n": len(hot),
        "hot_regression": round(hot["rolling_3"].mean() - hot["next_game"].mean(), 2) if len(hot) > 0 else None,
        "cold_n": len(cold),
        "cold_regression": round(cold["rolling_3"].mean() - cold["next_game"].mean(), 2) if len(cold) > 0 else None,
        "normal_n": len(normal),
        "normal_regression": round(normal["rolling_3"].mean() - normal["next_game"].mean(), 2) if len(normal) > 0 else None,
    }


def run_season_analysis(season):
    """Run role-player analysis for one season."""
    pool = get_role_players(season=season)
    results = []
    for i, row in pool.iterrows():
        try:
            result = analyze_player(row["PLAYER_NAME"], season=season)
            if result:
                results.append(result)
        except Exception:
            pass
        time.sleep(0.6)
        if (i + 1) % 25 == 0:
            print(f"  Progress: {i+1}/{len(pool)}")
    return pd.DataFrame(results), pool


def get_team_defensive_data(season="2024-25"):
    """Pull team-level defensive stats: def rating, opponent 3P%, pace."""
    print(f"Pulling team defensive data for {season}...")
    team_stats = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        measure_type_detailed_defense="Advanced",
        per_mode_detailed="PerGame"
    )
    df = team_stats.get_data_frames()[0]

    opp_stats = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        measure_type_detailed_defense="Opponent",
        per_mode_detailed="PerGame"
    )
    opp_df = opp_stats.get_data_frames()[0]

    team_data = df[["TEAM_ID", "TEAM_NAME", "DEF_RATING", "PACE"]].merge(
        opp_df[["TEAM_ID", "OPP_FG3_PCT", "OPP_FG3M"]],
        on="TEAM_ID"
    )

    team_name_to_abbr = {
        "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
        "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
        "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
        "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
        "LA Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
        "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
        "New Orleans Pelicans": "NOP", "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC",
        "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX",
        "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS",
        "Toronto Raptors": "TOR", "Utah Jazz": "UTA", "Washington Wizards": "WAS"
    }
    team_data["TEAM_ABBR"] = team_data["TEAM_NAME"].map(team_name_to_abbr)

    team_data["def_3p_rank"] = team_data["OPP_FG3_PCT"].rank(method="min").astype(int)
    team_data["def_tier"] = pd.cut(
        team_data["def_3p_rank"],
        bins=[0, 10, 20, 30],
        labels=["Elite (top 10)", "Mid (11-20)", "Bad (21-30)"]
    )

    return team_data.set_index("TEAM_ABBR")[["DEF_RATING", "PACE", "OPP_FG3_PCT", "def_tier"]]


def analyze_player_full_confounders(player_name, team_data, season="2024-25",
                                     stat="FG3M", hot_threshold=1.0):
    """Capture all confounders per hot streak event."""
    player_dict = players.find_players_by_full_name(player_name)
    if not player_dict:
        return []
    player_id = player_dict[0]["id"]

    gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season)
    df = gamelog.get_data_frames()[0]
    if len(df) < 20:
        return []

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], format="mixed")
    df = df.sort_values("GAME_DATE").reset_index(drop=True)

    season_avg = df[stat].mean()
    season_std = df[stat].std()
    season_mpg = df["MIN"].mean()

    df["rolling_3"] = df[stat].rolling(window=3).mean()
    df["rolling_3_min"] = df["MIN"].rolling(window=3).mean()
    df["z_score"] = (df["rolling_3"] - season_avg) / season_std
    df["next_game"] = df[stat].shift(-1)
    df["next_matchup"] = df["MATCHUP"].shift(-1)
    df["minutes_trend"] = df["rolling_3_min"] - season_mpg
    df["was_blowout"] = df["PLUS_MINUS"].abs() > 15

    def parse_matchup(matchup):
        if pd.isna(matchup):
            return None, None
        if " vs. " in matchup:
            return matchup.split(" vs. ")[1], "home"
        elif " @ " in matchup:
            return matchup.split(" @ ")[1], "away"
        return None, None

    df[["next_opp", "next_location"]] = df["next_matchup"].apply(
        lambda x: pd.Series(parse_matchup(x))
    )

    hot = df[
        (df["z_score"] > hot_threshold) &
        df["next_game"].notna() &
        df["next_opp"].notna()
    ].copy()

    events = []
    for _, row in hot.iterrows():
        opp = row["next_opp"]
        if opp in team_data.index:
            def_tier = team_data.loc[opp, "def_tier"]
            opp_3p_pct = team_data.loc[opp, "OPP_FG3_PCT"]
            pace = team_data.loc[opp, "PACE"]
        else:
            def_tier, opp_3p_pct, pace = None, None, None

        events.append({
            "player": player_name,
            "during_hot": row["rolling_3"],
            "next_game_stat": row["next_game"],
            "regression": row["rolling_3"] - row["next_game"],
            "next_opp": opp,
            "next_location": row["next_location"],
            "opp_def_tier": str(def_tier) if def_tier else None,
            "opp_3p_pct_allowed": opp_3p_pct,
            "opp_pace": pace,
            "minutes_trend": row["minutes_trend"],
            "was_blowout_recent": row["was_blowout"]
        })

    return events


# ==============================================================================
# STEP 1 — MULTI-SEASON BACKTEST (cached)
# ==============================================================================

seasons = ["2022-23", "2023-24", "2024-25"]
season_summaries = []
latest_results = None
latest_pool = None

for season in seasons:
    output_file = f"hot_hand_results_{season.replace('-', '_')}.csv"

    if os.path.exists(output_file):
        print(f"\n=== {season} === [cached]")
        season_df = pd.read_csv(output_file)
        if season == "2024-25":
            latest_pool = get_role_players(season=season)
    else:
        print(f"\n=== {season} ===")
        season_df, pool = run_season_analysis(season)
        season_df.to_csv(output_file, index=False)
        if season == "2024-25":
            latest_pool = pool

    summary = {
        "season": season,
        "players": len(season_df),
        "hot_regression": round(season_df["hot_regression"].mean(), 2),
        "cold_regression": round(season_df["cold_regression"].mean(), 2),
        "normal_regression": round(season_df["normal_regression"].mean(), 2),
        "hot_events": int(season_df["hot_n"].sum()),
        "pct_hot_regressed": round(100 * (season_df["hot_regression"] > 0).sum() / len(season_df), 1)
    }
    season_summaries.append(summary)

    if season == "2024-25":
        latest_results = season_df

print("\n=== MULTI-SEASON SUMMARY ===")
print(pd.DataFrame(season_summaries).to_string(index=False))


# ==============================================================================
# STEP 2 — SUBGROUP ANALYSIS (cached)
# ==============================================================================

subgroup_file = "hot_hand_subgroup_analysis.csv"

if os.path.exists(subgroup_file):
    print(f"\n=== SUBGROUP ANALYSIS === [cached]")
    merged = pd.read_csv(subgroup_file)
else:
    print("\n=== SUBGROUP ANALYSIS (2024-25) ===")
    merged = latest_results.merge(
        latest_pool[["PLAYER_NAME", "FG3A", "MIN", "GP"]],
        left_on="player", right_on="PLAYER_NAME", how="left"
    )

    def volume_tier(fg3a):
        if fg3a < 3: return "Low (2-3 3PA/g)"
        elif fg3a < 5: return "Medium (3-5 3PA/g)"
        else: return "High (5+ 3PA/g)"
    merged["volume_tier"] = merged["FG3A"].apply(volume_tier)

    print("\nPulling career experience (~2 min)...")
    experience_data = []
    for player_name in merged["player"]:
        try:
            player_dict = players.find_players_by_full_name(player_name)
            if player_dict:
                info = commonplayerinfo.CommonPlayerInfo(player_id=player_dict[0]["id"])
                years_pro = info.get_data_frames()[0]["SEASON_EXP"].iloc[0]
                experience_data.append({"player": player_name, "years_pro": years_pro})
        except Exception:
            pass
        time.sleep(0.6)

    merged = merged.merge(pd.DataFrame(experience_data), on="player", how="left")

    def experience_tier(years):
        if pd.isna(years): return "Unknown"
        elif years <= 2: return "Young (0-2 yrs)"
        elif years <= 7: return "Mid-career (3-7 yrs)"
        else: return "Veteran (8+ yrs)"
    merged["experience_tier"] = merged["years_pro"].apply(experience_tier)

    merged.to_csv(subgroup_file, index=False)

print("\n=== HOT REGRESSION BY VOLUME TIER ===")
print(merged.groupby("volume_tier").agg(
    players=("player", "count"),
    avg_hot_regression=("hot_regression", "mean"),
    total_hot_events=("hot_n", "sum"),
    pct_regressing=("hot_regression", lambda x: round(100 * (x > 0).sum() / len(x), 1))
).round(2))

print("\n=== HOT REGRESSION BY EXPERIENCE TIER ===")
print(merged.groupby("experience_tier").agg(
    players=("player", "count"),
    avg_hot_regression=("hot_regression", "mean"),
    total_hot_events=("hot_n", "sum"),
    pct_regressing=("hot_regression", lambda x: round(100 * (x > 0).sum() / len(x), 1))
).round(2))


# ==============================================================================
# STEP 3 — REST DAYS (cached)
# ==============================================================================

rest_file = "hot_hand_rest_days_analysis.csv"

if os.path.exists(rest_file):
    print(f"\n=== REST DAYS === [cached]")
    rest_df = pd.read_csv(rest_file)
    b2b_only = rest_df.dropna(subset=["hot_regression_b2b"])
    rested_only = rest_df.dropna(subset=["hot_regression_rested"])
    print(f"B2B regression: {b2b_only['hot_regression_b2b'].mean():.2f}")
    print(f"Rested regression: {rested_only['hot_regression_rested'].mean():.2f}")
    print(f"Difference (B2B - Rested): {b2b_only['hot_regression_b2b'].mean() - rested_only['hot_regression_rested'].mean():.2f}")


# ==============================================================================
# STEP 4 — FULL CONFOUNDER ANALYSIS (NEW — runs today)
# ==============================================================================

confounder_file = "hot_hand_full_events.csv"

if os.path.exists(confounder_file):
    print(f"\n=== FULL CONFOUNDER ANALYSIS === [cached]")
    events_df = pd.read_csv(confounder_file)
else:
    print("\n=== FULL CONFOUNDER ANALYSIS (2024-25) ===\n")

    team_data = get_team_defensive_data(season="2024-25")
    print(f"Loaded data for {len(team_data)} teams")

    all_events = []
    pool = get_role_players(season="2024-25")

    for i, row in pool.iterrows():
        player_name = row["PLAYER_NAME"]
        try:
            events = analyze_player_full_confounders(player_name, team_data, season="2024-25")
            all_events.extend(events)
        except Exception:
            pass
        time.sleep(0.6)
        if (i + 1) % 25 == 0:
            print(f"  Progress: {i+1}/{len(pool)} ({len(all_events)} events so far)")

    events_df = pd.DataFrame(all_events)
    events_df.to_csv(confounder_file, index=False)
    print(f"\nSaved {len(events_df)} events to {confounder_file}")

print(f"\nTotal hot streak events: {len(events_df)}")

# Confounder breakdowns
print("\n=== HOT REGRESSION BY OPPONENT DEFENSE TIER ===")
print(events_df.groupby("opp_def_tier", observed=True).agg(
    events=("regression", "count"),
    avg_regression=("regression", "mean"),
    avg_during_hot=("during_hot", "mean"),
    avg_next_game=("next_game_stat", "mean")
).round(2))

print("\n=== HOT REGRESSION BY HOME/AWAY (next game) ===")
print(events_df.groupby("next_location").agg(
    events=("regression", "count"),
    avg_regression=("regression", "mean"),
    avg_during_hot=("during_hot", "mean"),
    avg_next_game=("next_game_stat", "mean")
).round(2))

print("\n=== HOT REGRESSION BY MINUTES TREND ===")
events_df["minutes_trend_bucket"] = pd.cut(
    events_df["minutes_trend"],
    bins=[-100, -2, 2, 100],
    labels=["Minutes DOWN (<-2)", "Stable (±2)", "Minutes UP (>+2)"]
)
print(events_df.groupby("minutes_trend_bucket", observed=True).agg(
    events=("regression", "count"),
    avg_regression=("regression", "mean")
).round(2))

print("\n=== HOT REGRESSION BY BLOWOUT CONTEXT ===")
print(events_df.groupby("was_blowout_recent").agg(
    events=("regression", "count"),
    avg_regression=("regression", "mean")
).round(2))

# ==============================================================================
# STEP 5 — FAIR LINE vs NAIVE LINE BACKTEST
# ==============================================================================

print("\n\n=== STEP 5: FAIR LINE vs NAIVE LINE BACKTEST ===\n")

# Load the full events dataset from step 4
events_df = pd.read_csv("hot_hand_full_events.csv")

# We need the player's season average too, which we have in the subgroup file
subgroup_df = pd.read_csv("hot_hand_subgroup_analysis.csv")
player_baselines = subgroup_df[["player", "season_avg"]].drop_duplicates()

# Merge season averages into the events
backtest_df = events_df.merge(player_baselines, on="player", how="left")

# Define the two lines
# - your_fair_line: the player's season average (what the model says is "true skill")
# - naive_line: the rolling 3-game average during the hot streak (what casual bettors would use)
backtest_df["your_fair_line"] = backtest_df["season_avg"]
backtest_df["naive_line"] = backtest_df["during_hot"]
backtest_df["actual_outcome"] = backtest_df["next_game_stat"]

# Calculate how far the actual outcome was from each line (absolute distance)
backtest_df["distance_to_yours"] = (backtest_df["actual_outcome"] - backtest_df["your_fair_line"]).abs()
backtest_df["distance_to_naive"] = (backtest_df["actual_outcome"] - backtest_df["naive_line"]).abs()

# Who won? Your line wins if it's closer to the actual outcome
backtest_df["your_line_wins"] = backtest_df["distance_to_yours"] < backtest_df["distance_to_naive"]

# Summary
total_events = len(backtest_df)
your_wins = backtest_df["your_line_wins"].sum()
naive_wins = total_events - your_wins
win_pct = 100 * your_wins / total_events

print(f"Total hot streak events tested: {total_events}")
print(f"Your model's line was closer: {your_wins} ({win_pct:.1f}%)")
print(f"Naive line was closer: {naive_wins} ({100 - win_pct:.1f}%)")
print(f"\nAverage distance — your line: {backtest_df['distance_to_yours'].mean():.2f} threes off")
print(f"Average distance — naive line: {backtest_df['distance_to_naive'].mean():.2f} threes off")

# Now segment by the strategy filters we found
print("\n=== WIN RATE BY VOLUME TIER ===")
volume_tier_map = subgroup_df.set_index("player")["volume_tier"].to_dict()
backtest_df["volume_tier"] = backtest_df["player"].map(volume_tier_map)
print(backtest_df.groupby("volume_tier", dropna=True).agg(
    events=("your_line_wins", "count"),
    win_rate=("your_line_wins", lambda x: round(100 * x.sum() / len(x), 1)),
    avg_actual=("actual_outcome", "mean"),
    avg_your_line=("your_fair_line", "mean"),
    avg_naive_line=("naive_line", "mean")
).round(2))

print("\n=== WIN RATE BY HOME/AWAY ===")
print(backtest_df.groupby("next_location").agg(
    events=("your_line_wins", "count"),
    win_rate=("your_line_wins", lambda x: round(100 * x.sum() / len(x), 1))
).round(2))

# Minutes trend filter applied to events_df
backtest_df["minutes_trend_bucket"] = pd.cut(
    backtest_df["minutes_trend"],
    bins=[-100, -2, 2, 100],
    labels=["Minutes DOWN (<-2)", "Stable (±2)", "Minutes UP (>+2)"]
)
print("\n=== WIN RATE BY MINUTES TREND ===")
print(backtest_df.groupby("minutes_trend_bucket", observed=True).agg(
    events=("your_line_wins", "count"),
    win_rate=("your_line_wins", lambda x: round(100 * x.sum() / len(x), 1))
).round(2))

# The BIG one — combine all filters
print("\n=== STRATEGY: FILTERED BEST BETS (high volume + stable minutes + away) ===")
filtered = backtest_df[
    (backtest_df["volume_tier"] == "High (5+ 3PA/g)") &
    (backtest_df["minutes_trend_bucket"] == "Stable (±2)") &
    (backtest_df["next_location"] == "away")
]
if len(filtered) > 0:
    win_pct_filtered = 100 * filtered["your_line_wins"].sum() / len(filtered)
    print(f"Events matching all filters: {len(filtered)}")
    print(f"Win rate: {win_pct_filtered:.1f}%")
    print(f"Average actual outcome: {filtered['actual_outcome'].mean():.2f}")
    print(f"Average your fair line: {filtered['your_fair_line'].mean():.2f}")
    print(f"Average naive line: {filtered['naive_line'].mean():.2f}")

backtest_df.to_csv("hot_hand_backtest_results.csv", index=False)
print("\nSaved hot_hand_backtest_results.csv")

# ==============================================================================
# STEP 6 — MULTI-SEASON PROXY BACKTEST
# ==============================================================================

print("\n\n=== STEP 6: MULTI-SEASON PROXY BACKTEST ===\n")

def build_proxy_backtest_for_season(season):
    """
    Build the fair-line-vs-naive-line backtest dataset for a single season.
    Requires re-running the per-event analysis to capture during_hot, season_avg, next_game.
    """
    output_file = f"hot_hand_proxy_backtest_{season.replace('-', '_')}.csv"
    
    if os.path.exists(output_file):
        print(f"  {season} backtest [cached]")
        return pd.read_csv(output_file)
    
    print(f"  Building {season} backtest from scratch...")
    
    # Get role player pool and team data for this season
    pool = get_role_players(season=season)
    team_data = get_team_defensive_data(season=season)
    
    # Pull per-event data with confounders
    all_events = []
    for i, row in pool.iterrows():
        try:
            events = analyze_player_full_confounders(
                row["PLAYER_NAME"], team_data, season=season
            )
            all_events.extend(events)
        except Exception:
            pass
        time.sleep(0.6)
        if (i + 1) % 25 == 0:
            print(f"    Progress: {i+1}/{len(pool)} ({len(all_events)} events)")
    
    events_df = pd.DataFrame(all_events)
    
    # We also need each player's season average — pull that from the basic season results
    basic_results = pd.read_csv(f"hot_hand_results_{season.replace('-', '_')}.csv")
    player_baselines = basic_results[["player", "season_avg"]].drop_duplicates()
    
    # Merge baselines
    backtest_df = events_df.merge(player_baselines, on="player", how="left")
    
    # Build the comparison
    backtest_df["your_fair_line"] = backtest_df["season_avg"]
    backtest_df["naive_line"] = backtest_df["during_hot"]
    backtest_df["actual_outcome"] = backtest_df["next_game_stat"]
    backtest_df["distance_to_yours"] = (backtest_df["actual_outcome"] - backtest_df["your_fair_line"]).abs()
    backtest_df["distance_to_naive"] = (backtest_df["actual_outcome"] - backtest_df["naive_line"]).abs()
    backtest_df["your_line_wins"] = backtest_df["distance_to_yours"] < backtest_df["distance_to_naive"]
    
    backtest_df.to_csv(output_file, index=False)
    return backtest_df


seasons_to_backtest = ["2022-23", "2023-24", "2024-25"]
multi_season_results = []

for season in seasons_to_backtest:
    print(f"\n--- {season} ---")
    bt_df = build_proxy_backtest_for_season(season)
    
    total = len(bt_df)
    wins = bt_df["your_line_wins"].sum()
    win_pct = 100 * wins / total
    
    # Filtered subset (high volume + stable minutes + away)
    bt_df["minutes_trend_bucket"] = pd.cut(
        bt_df["minutes_trend"],
        bins=[-100, -2, 2, 100],
        labels=["Minutes DOWN", "Stable", "Minutes UP"]
    )
    
    # Need volume tier — recompute from during_hot if needed
    # We don't have FG3A directly in events_df, so we approximate via the pool
    pool = get_role_players(season=season)
    pool["volume_tier"] = pool["FG3A"].apply(
        lambda x: "High (5+ 3PA/g)" if x >= 5 else ("Low (2-3 3PA/g)" if x < 3 else "Medium")
    )
    volume_map = pool.set_index("PLAYER_NAME")["volume_tier"].to_dict()
    bt_df["volume_tier"] = bt_df["player"].map(volume_map)
    
    filtered = bt_df[
        (bt_df["volume_tier"] == "High (5+ 3PA/g)") &
        (bt_df["minutes_trend_bucket"] == "Stable") &
        (bt_df["next_location"] == "away")
    ]
    
    filtered_win_pct = 100 * filtered["your_line_wins"].sum() / len(filtered) if len(filtered) > 0 else None
    
    multi_season_results.append({
        "season": season,
        "total_events": total,
        "overall_win_rate": round(win_pct, 1),
        "avg_distance_yours": round(bt_df["distance_to_yours"].mean(), 2),
        "avg_distance_naive": round(bt_df["distance_to_naive"].mean(), 2),
        "filtered_events": len(filtered),
        "filtered_win_rate": round(filtered_win_pct, 1) if filtered_win_pct else None
    })
    
    print(f"    Total events: {total}, Overall win rate: {win_pct:.1f}%")
    print(f"    Filtered events: {len(filtered)}, Filtered win rate: {filtered_win_pct:.1f}%" if filtered_win_pct else f"    No filtered events")


print("\n=== MULTI-SEASON PROXY BACKTEST SUMMARY ===")
print(pd.DataFrame(multi_season_results).to_string(index=False))

# Calculate combined across all seasons
print("\n=== KEY METRICS ===")
overall_win_rates = [r["overall_win_rate"] for r in multi_season_results]
filtered_win_rates = [r["filtered_win_rate"] for r in multi_season_results if r["filtered_win_rate"]]
total_events = sum(r["total_events"] for r in multi_season_results)
total_filtered_events = sum(r["filtered_events"] for r in multi_season_results)

print(f"Overall win rate range: {min(overall_win_rates):.1f}% to {max(overall_win_rates):.1f}%")
print(f"Filtered subset win rate range: {min(filtered_win_rates):.1f}% to {max(filtered_win_rates):.1f}%")
print(f"Total events tested across 3 seasons: {total_events}")
print(f"Total filtered events: {total_filtered_events}")

# ==============================================================================
# STEP 7 — MULTI-STAT ANALYSIS (Points, PR, PA, PRA)
# ==============================================================================

def analyze_player_multi_stat(player_name, season="2024-25", stat="PTS",
                               hot_threshold=1.0, cold_threshold=-1.0):
    """
    Generalized version of analyze_player that handles any stat including combos.
    For combo stats, computes them on the fly.
    """
    player_dict = players.find_players_by_full_name(player_name)
    if not player_dict:
        return None
    player_id = player_dict[0]["id"]
    
    gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season)
    df = gamelog.get_data_frames()[0]
    if len(df) < 20:
        return None
    
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], format="mixed")
    df = df.sort_values("GAME_DATE").reset_index(drop=True)
    
    # Build combo stats if needed
    df["PR"] = df["PTS"] + df["REB"]
    df["PA"] = df["PTS"] + df["AST"]
    df["PRA"] = df["PTS"] + df["REB"] + df["AST"]
    
    if stat not in df.columns:
        return None
    
    season_avg = df[stat].mean()
    season_std = df[stat].std()
    
    df["rolling_3"] = df[stat].rolling(window=3).mean()
    df["z_score"] = (df["rolling_3"] - season_avg) / season_std
    df["next_game"] = df[stat].shift(-1)
    df = df.dropna(subset=["next_game", "z_score"])
    
    hot = df[df["z_score"] > hot_threshold]
    cold = df[df["z_score"] < cold_threshold]
    normal = df[(df["z_score"] >= cold_threshold) & (df["z_score"] <= hot_threshold)]
    
    return {
        "player": player_name,
        "stat": stat,
        "season_avg": round(season_avg, 2),
        "season_std": round(season_std, 2),
        "hot_n": len(hot),
        "hot_during": round(hot["rolling_3"].mean(), 2) if len(hot) > 0 else None,
        "hot_next": round(hot["next_game"].mean(), 2) if len(hot) > 0 else None,
        "hot_regression": round(hot["rolling_3"].mean() - hot["next_game"].mean(), 2) if len(hot) > 0 else None,
        "cold_n": len(cold),
        "cold_regression": round(cold["rolling_3"].mean() - cold["next_game"].mean(), 2) if len(cold) > 0 else None,
        "normal_n": len(normal),
        "normal_regression": round(normal["rolling_3"].mean() - normal["next_game"].mean(), 2) if len(normal) > 0 else None,
    }


def analyze_player_proxy_for_stat(player_name, season="2024-25", stat="PTS", hot_threshold=1.0):
    """
    Per-event analysis for the proxy backtest, for any stat including combos.
    Returns a list of per-event dicts.
    """
    player_dict = players.find_players_by_full_name(player_name)
    if not player_dict:
        return []
    player_id = player_dict[0]["id"]
    
    gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season)
    df = gamelog.get_data_frames()[0]
    if len(df) < 20:
        return []
    
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], format="mixed")
    df = df.sort_values("GAME_DATE").reset_index(drop=True)
    
    # Build combo stats
    df["PR"] = df["PTS"] + df["REB"]
    df["PA"] = df["PTS"] + df["AST"]
    df["PRA"] = df["PTS"] + df["REB"] + df["AST"]
    
    if stat not in df.columns:
        return []
    
    season_avg = df[stat].mean()
    season_std = df[stat].std()
    
    df["rolling_3"] = df[stat].rolling(window=3).mean()
    df["z_score"] = (df["rolling_3"] - season_avg) / season_std
    df["next_game"] = df[stat].shift(-1)
    
    hot = df[
        (df["z_score"] > hot_threshold) &
        df["next_game"].notna()
    ].copy()
    
    events = []
    for _, row in hot.iterrows():
        events.append({
            "player": player_name,
            "stat": stat,
            "season_avg": season_avg,
            "during_hot": row["rolling_3"],
            "next_game_stat": row["next_game"],
            "your_fair_line": season_avg,
            "naive_line": row["rolling_3"],
            "actual_outcome": row["next_game"],
            "distance_to_yours": abs(row["next_game"] - season_avg),
            "distance_to_naive": abs(row["next_game"] - row["rolling_3"]),
            "your_line_wins": abs(row["next_game"] - season_avg) < abs(row["next_game"] - row["rolling_3"])
        })
    
    return events


print("\n\n=== STEP 7: MULTI-STAT ANALYSIS (2024-25) ===\n")

# We'll test each of these stats with a relaxed role-player filter
# (since the original filter was based on 3PA which doesn't apply to points/combos)
stats_to_test = ["PTS", "PR", "PA", "PRA"]

# For non-3P stats, we want a broader pool — players with 15+ MPG and decent volume
def get_general_role_players(season="2024-25", min_mpg=15, max_mpg=32, min_gp=30):
    """Role players for general stats (no 3PA filter)."""
    print(f"Pulling general role players for {season}...")
    league_stats = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        per_mode_detailed="PerGame"
    )
    df = league_stats.get_data_frames()[0]
    role_players_df = df[
        (df["MIN"] >= min_mpg) &
        (df["MIN"] <= max_mpg) &
        (df["GP"] >= min_gp)
    ]
    print(f"  Found {len(role_players_df)} role players")
    return role_players_df[["PLAYER_ID", "PLAYER_NAME", "MIN", "PTS", "REB", "AST", "GP"]].reset_index(drop=True)


multi_stat_summaries = []

for stat in stats_to_test:
    output_file = f"hot_hand_multistat_{stat}_2024_25.csv"
    
    if os.path.exists(output_file):
        print(f"\n--- {stat} === [cached]")
        all_events = pd.read_csv(output_file)
    else:
        print(f"\n--- {stat} === [computing]")
        pool = get_general_role_players(season="2024-25")
        
        all_events_list = []
        for i, row in pool.iterrows():
            try:
                events = analyze_player_proxy_for_stat(
                    row["PLAYER_NAME"], season="2024-25", stat=stat
                )
                all_events_list.extend(events)
            except Exception:
                pass
            time.sleep(0.6)
            if (i + 1) % 25 == 0:
                print(f"  Progress: {i+1}/{len(pool)} ({len(all_events_list)} events)")
        
        all_events = pd.DataFrame(all_events_list)
        all_events.to_csv(output_file, index=False)
    
    total = len(all_events)
    wins = all_events["your_line_wins"].sum()
    win_pct = 100 * wins / total if total > 0 else 0
    avg_dist_yours = all_events["distance_to_yours"].mean()
    avg_dist_naive = all_events["distance_to_naive"].mean()
    avg_during_hot = all_events["during_hot"].mean()
    avg_next = all_events["next_game_stat"].mean()
    avg_regression = avg_during_hot - avg_next
    
    multi_stat_summaries.append({
        "stat": stat,
        "events": total,
        "win_rate": round(win_pct, 1),
        "avg_during_hot": round(avg_during_hot, 2),
        "avg_next_game": round(avg_next, 2),
        "avg_regression": round(avg_regression, 2),
        "avg_dist_yours": round(avg_dist_yours, 2),
        "avg_dist_naive": round(avg_dist_naive, 2)
    })
    
    print(f"  {stat}: {total} events, {win_pct:.1f}% win rate, regression {avg_regression:.2f}")


print("\n=== MULTI-STAT SUMMARY (2024-25) ===")
print(pd.DataFrame(multi_stat_summaries).to_string(index=False))

print("\n=== KEY TAKEAWAY ===")
print("Stat with strongest edge: ", end="")
best_stat = max(multi_stat_summaries, key=lambda x: x["win_rate"])
print(f"{best_stat['stat']} ({best_stat['win_rate']}% win rate)")