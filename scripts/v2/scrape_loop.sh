#!/bin/bash
# Temporary Mac collector: runs run_scrape.sh at :02 and :32 every hour.
# (launchd/cron can't read ~/Desktop without Full Disk Access, so this runs as a
#  background process started from a terminal; caffeinate keeps the Mac from idle-sleeping.)
#   start: nohup caffeinate -i scripts/v2/scrape_loop.sh >/dev/null 2>&1 &
#   stop:  pkill -f scrape_loop.sh
DIR="$(cd "$(dirname "$0")" && pwd)"
while true; do
  m=$(date +%-M); s=$(date +%-S)
  if [ "$m" -lt 2 ]; then wait=$(( (2 - m) * 60 - s ))
  elif [ "$m" -lt 32 ]; then wait=$(( (32 - m) * 60 - s ))
  else wait=$(( (62 - m) * 60 - s )); fi
  sleep "$wait"
  "$DIR/run_scrape.sh"
done
