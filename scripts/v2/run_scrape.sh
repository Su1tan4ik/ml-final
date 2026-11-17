#!/bin/bash
# One scrape slot; called by scrape_loop.sh (collector container / Mac) or a systemd timer / cron.
cd "$(dirname "$0")" || exit 1
ROOT="$(cd ../.. && pwd)"
LOG="$ROOT/data/v2/logs/scrape.log"
mkdir -p "$(dirname "$LOG")"
export PATH="/usr/local/bin:/opt/homebrew/bin:$PATH"
PY="${PYTHON:-$ROOT/venv/bin/python}"
M="$ROOT/models_v2"
LOCK="$M/.train.lock"   # directory lock (mkdir is atomic; flock is missing on macOS)
{
  echo "=== $(date '+%F %T')"
  # make sure the DB container is up (on the Mac Docker Desktop may still be starting)
  if [ -z "$IN_DOCKER" ] && ! docker exec traffic_db pg_isready -q -U traffic 2>/dev/null; then
    [ "$(uname)" = Darwin ] && open -ga Docker
    for i in $(seq 1 30); do docker info >/dev/null 2>&1 && break; sleep 5; done
    (cd "$ROOT" && docker compose up -d db)
    for i in $(seq 1 30); do docker exec traffic_db pg_isready -q -U traffic 2>/dev/null && break; sleep 2; done
  fi
  export PYTHONWARNINGS="ignore::RuntimeWarning"   # numpy "mean of empty slice" on units without history — expected
  OUT=$("$PY" scraper_v2.py 2>&1); RC=$?
  echo "$OUT"
  echo "scrape exit $RC"
  "$PY" notify.py "$RC" "$(echo "$OUT" | tail -n 3)"   # Telegram on failure / recovery (no-op without TG_*)

  # retrain: no model yet -> now; < 7 days of data -> every 2 h; otherwise daily ~03:00 Almaty (UTC+5).
  # Runs in the BACKGROUND so a long training never delays the next scrape slot; the lock
  # keeps one training at a time, train_v2 swaps model files atomically for predict_v2.
  H=$(date -u -d '+5 hours' +%-H 2>/dev/null || date -u -v+5H +%-H)
  MIN=$(date -u +%-M)
  if [ ! -f "$M/meta.json" ] \
     || { grep -q '"small_data": true' "$M/report.json" 2>/dev/null && [ $((H % 2)) -eq 0 ] && [ "$MIN" -lt 30 ]; } \
     || { [ "$H" -eq 3 ] && [ "$MIN" -lt 30 ]; }; then
    mkdir -p "$M"
    find "$LOCK" -maxdepth 0 -mmin +240 -exec rmdir {} \; 2>/dev/null   # stale lock of a killed run
    if mkdir "$LOCK" 2>/dev/null; then
      echo "train: started in background"
      (
        "$PY" train_v2.py --days 28; echo "train exit $?"
        [ -f "$M/meta.json" ] && { "$PY" recommend_v2.py; echo "recommend exit $?"; }
        rmdir "$LOCK"
      ) >> "$LOG" 2>&1 < /dev/null &
    else
      echo "train: previous training still running — skip"
    fi
  fi
  # forecast for the new slot with the current model
  [ -f "$M/meta.json" ] && { "$PY" predict_v2.py; echo "predict exit $?"; }
} >> "$LOG" 2>&1
