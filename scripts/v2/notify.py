"""
Telegram alert for the collector: one message when scraping starts failing (blocked by
Yandex, network, DB down, crash) and one when it recovers — not a message every 30 min.
No-op unless TG_BOT_TOKEN and TG_CHAT_ID are set (environment or .env in the repo root).

    python notify.py <scraper exit code> [details]      # called by run_scrape.sh
"""
from __future__ import annotations

import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MARK = ROOT / "data" / "v2" / "logs" / ".scrape_failing"


def _env(key: str) -> str:
    if os.getenv(key):
        return os.environ[key]
    f = ROOT / ".env"
    if f.exists():
        for line in f.read_text().splitlines():
            k, _, v = line.partition("=")
            if k.strip() == key and not line.lstrip().startswith("#"):
                return v.split("#")[0].strip()
    return ""


def send(text: str) -> bool:
    token, chat = _env("TG_BOT_TOKEN"), _env("TG_CHAT_ID")
    if not (token and chat):
        return False
    data = urllib.parse.urlencode({"chat_id": chat, "text": text[:4000]}).encode()
    try:
        with urllib.request.urlopen(f"https://api.telegram.org/bot{token}/sendMessage", data, timeout=10) as r:
            return bool(json.load(r).get("ok"))
    except Exception as e:  # noqa: BLE001 — alerting must never break the collector
        print("telegram failed:", e)
        return False


def main():
    rc = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    details = sys.argv[2].strip() if len(sys.argv) > 2 else ""
    if not (_env("TG_BOT_TOKEN") and _env("TG_CHAT_ID")):
        return
    if rc == 0:
        if MARK.exists() and send("✅ Сбор трафика снова работает.\n" + details):
            MARK.unlink()
    elif not MARK.exists():
        if send(f"⚠️ Сбор трафика: ошибка (код {rc}). Лог: data/v2/logs/scrape.log\n{details}"):
            MARK.parent.mkdir(parents=True, exist_ok=True)
            MARK.touch()


if __name__ == "__main__":
    main()
