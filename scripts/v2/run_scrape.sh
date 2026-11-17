#!/bin/bash
# One scrape slot; called by launchd (Mac, temporary) or a systemd timer / cron (server).
cd "$(dirname "$0")" || exit 1
ROOT="$(cd ../.. && pwd)"
LOG="$ROOT/data/v2/logs/scrape.log"
mkdir -p "$(dirname "$LOG")"
export PATH="/usr/local/bin:/opt/homebrew/bin:$PATH"
{
  echo "=== $(date '+%F %T')"
  # make sure the DB container is up (on the Mac Docker Desktop may still be starting)
  if [ -z "$IN_DOCKER" ] && ! docker exec traffic_db pg_isready -q -U traffic 2>/dev/null; then
    [ "$(uname)" = Darwin ] && open -ga Docker
    for i in $(seq 1 30); do docker info >/dev/null 2>&1 && break; sleep 5; done
    (cd "$ROOT" && docker compose up -d db)
    for i in $(seq 1 30); do docker exec traffic_db pg_isready -q -U traffic 2>/dev/null && break; sleep 2; done
  fi
  PY="${PYTHON:-$ROOT/venv/bin/python}"
  export PYTHONWARNINGS="ignore::RuntimeWarning"   # numpy "mean of empty slice" on units without history — expected
  "$PY" scraper_v2.py
  echo "scrape exit $?"
  # retrain: no model yet -> now; < 7 days of data -> every 2 h; otherwise daily ~03:00 Almaty (UTC+5)
  M="$ROOT/models_v2"
  H=$(date -u -d '+5 hours' +%-H 2>/dev/null || date -u -v+5H +%-H)
  MIN=$(date -u +%-M)
  if [ ! -f "$M/meta.json" ] \
     || { grep -q '"small_data": true' "$M/report.json" 2>/dev/null && [ $((H % 2)) -eq 0 ] && [ "$MIN" -lt 30 ]; } \
     || { [ "$H" -eq 3 ] && [ "$MIN" -lt 30 ]; }; then
    "$PY" train_v2.py --days 28; echo "train exit $?"
    [ -f "$M/meta.json" ] && { "$PY" recommend_v2.py; echo "recommend exit $?"; }
  fi
  # forecast for the new slot with the current model
  [ -f "$M/meta.json" ] && { "$PY" predict_v2.py; echo "predict exit $?"; }
} >> "$LOG" 2>&1
