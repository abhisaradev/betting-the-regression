# Automation Plan — Betting the Regression

**Status: Research complete. Implementation scheduled for next session.**

This document outlines the plan to automatically run the daily pipeline at 5 PM ET on every day
the tool is installed. Do NOT implement this yet — read through the plan, then implement next session.

---

## The goal

Every day at 5 PM ET, the Mac should automatically:

1. Run `python3.12 daily_picks.py` (grades yesterday, generates today's picks)
2. Run `python3.12 odds_compare.py` (fetches DK lines, generates dashboard)
3. Send a Mac notification so you know to check the dashboard
4. Save all output to a dated log file

---

## Option 1 vs Option 2: how does it know if there's a game today?

**Option 1 — Run every day, handle no-games gracefully** ✅ **Recommended**

The scripts already handle this correctly. When there are no NBA games,
`daily_picks.py` prints:
```
NO PICKS FOR TODAY (2026-06-10)
Next scheduled games: 2026-06-11 — NBA (1 game), WNBA (3 games)
```

And `odds_compare.py` exits cleanly with no dashboard generated.

**Why Option 1 is better:**
- The scripts already do the right thing — no extra complexity needed
- launchd plists can't easily make conditional HTTP API calls
- Running every day costs essentially nothing (a few seconds of API calls)
- You get the "next game" notice passively, even on off-days
- Zero maintenance overhead vs. a game-day detection layer

**Option 2 — Check for a game before running**
Would require a wrapper shell script that calls the ESPN API, parses JSON,
and exits early if no games. More complex, more failure modes, not worth it.

---

## launchd setup

macOS uses `launchd` instead of cron. User-level jobs go in `~/Library/LaunchAgents/`.

### 1. Create the wrapper script

Create `/Users/abhi/Coding Projects/hot-hand-fader/run_daily.sh`:

```bash
#!/bin/bash
# Betting the Regression — daily automation script
# Called by launchd at 5 PM ET every day.

set -e  # exit on any error

PROJ="/Users/abhi/Coding Projects/hot-hand-fader"
PYTHON="/opt/anaconda3/bin/python3.12"
LOG="$PROJ/logs/run_$(date +%Y-%m-%d).log"

# Create logs directory if it doesn't exist
mkdir -p "$PROJ/logs"

# Run the pipeline, capturing all output to a dated log file
{
    echo "=== Run started at $(date) ==="
    cd "$PROJ"
    $PYTHON daily_picks.py
    $PYTHON odds_compare.py
    echo "=== Run completed at $(date) ==="
} >> "$LOG" 2>&1

# Mac notification when done
osascript -e 'display notification "Dashboard ready — open dashboard.html" with title "Betting the Regression" sound name "Glass"'
```

Make it executable:
```bash
chmod +x "/Users/abhi/Coding Projects/hot-hand-fader/run_daily.sh"
```

---

### 2. Create the launchd plist

Create `~/Library/LaunchAgents/com.abhi.bettingtheregression.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
    "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <!-- Unique identifier — reverse-DNS style -->
    <key>Label</key>
    <string>com.abhi.bettingtheregression</string>

    <!-- The script to run -->
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>/Users/abhi/Coding Projects/hot-hand-fader/run_daily.sh</string>
    </array>

    <!-- Run at 5:00 PM ET = 9:00 PM UTC (adjust for DST: EDT=UTC-4, EST=UTC-5)
         During EDT (Apr-Nov): Hour 21 UTC = 5 PM ET
         During EST (Nov-Mar): Hour 22 UTC = 5 PM ET
         Set to 21 for the NBA regular season / playoffs window -->
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>21</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>

    <!-- Run job even if the Mac was asleep at the scheduled time -->
    <!-- Note: launchd does NOT catch up on missed runs if the Mac was off -->
    <key>RunAtLoad</key>
    <false/>

    <!-- Log stdout and stderr (in addition to our own log in run_daily.sh) -->
    <key>StandardOutPath</key>
    <string>/Users/abhi/Coding Projects/hot-hand-fader/logs/launchd_stdout.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/abhi/Coding Projects/hot-hand-fader/logs/launchd_stderr.log</string>

    <!-- Working directory -->
    <key>WorkingDirectory</key>
    <string>/Users/abhi/Coding Projects/hot-hand-fader</string>
</dict>
</plist>
```

---

### 3. Load the job with launchctl

```bash
# Load (activates immediately, will run at next scheduled time)
launchctl load ~/Library/LaunchAgents/com.abhi.bettingtheregression.plist

# Verify it's loaded
launchctl list | grep bettingtheregression

# To run it manually right now (for testing)
launchctl start com.abhi.bettingtheregression

# To unload (stop scheduling)
launchctl unload ~/Library/LaunchAgents/com.abhi.bettingtheregression.plist
```

**Important:** The Mac must be awake and logged in at 5 PM ET for the job to fire.
launchd does not run jobs for times the Mac was sleeping — it won't "catch up".

---

## Mac notification command

The notification is sent at the end of `run_daily.sh` using osascript:

```bash
osascript -e 'display notification "Dashboard ready — open dashboard.html" with title "Betting the Regression" sound name "Glass"'
```

To test this right now:
```bash
osascript -e 'display notification "Test notification" with title "Betting the Regression" sound name "Glass"'
```

Available sounds: `Basso`, `Blow`, `Bottle`, `Frog`, `Funk`, `Glass`, `Hero`,
`Morse`, `Ping`, `Pop`, `Purr`, `Sosumi`, `Submarine`, `Tink`

**Note:** On macOS 10.14+, you may need to grant Terminal notification permissions in
System Settings → Notifications → Terminal → Allow Notifications.

---

## Log files

Each daily run writes to a dated log file:
```
logs/
├── run_2026-06-10.log   ← all stdout + stderr from daily_picks.py + odds_compare.py
├── run_2026-06-11.log
├── launchd_stdout.log   ← launchd's own output (usually empty)
└── launchd_stderr.log   ← launchd errors (e.g. if script not found)
```

To review yesterday's run:
```bash
cat "/Users/abhi/Coding Projects/hot-hand-fader/logs/run_$(date -v-1d +%Y-%m-%d).log"
```

To watch a run live as it happens (useful for testing):
```bash
tail -f "/Users/abhi/Coding Projects/hot-hand-fader/logs/run_$(date +%Y-%m-%d).log"
```

---

## DST note

The plist uses UTC hour 21 = 5 PM EDT (Eastern Daylight Time, UTC-4).
During Eastern Standard Time (November–March), 5 PM ET = UTC 22.

To handle this properly, you can either:
- Update the plist each time DST changes (2 minutes of work twice a year)
- Set the hour to 21 in the plist and accept that during EST the job runs at 4 PM ET

The NBA regular season runs October–April (mostly EST) and playoffs run April–June (EDT).
For the playoffs window this tool is primarily used for, UTC 21 = 5 PM ET is correct.

---

## Implementation checklist (next session)

- [ ] Create `logs/` directory
- [ ] Create and chmod `run_daily.sh`
- [ ] Create the `.plist` file in `~/Library/LaunchAgents/`
- [ ] Load with `launchctl load`
- [ ] Test with `launchctl start com.abhi.bettingtheregression`
- [ ] Verify log file is written
- [ ] Verify Mac notification fires
- [ ] Verify `dashboard.html` is updated

---

*Research completed 2026-06-10. Implement next session.*
