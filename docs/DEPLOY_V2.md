# V2 — развёртывание на сервере (и перенос с Mac)

Состав: `docker-compose.yml`
- `db` — PostgreSQL 17 + PostGIS + pgRouting (всегда)
- `api` — FastAPI + карта (профиль `server`)
- `collector` — каждые 30 мин: скрап → прогноз → в 03:xx Алматы переобучение (профиль `server`)

Образ `api`/`collector`: `deploy/v2.Dockerfile` (python 3.12-slim, только код; `data/` и `models_v2/` монтируются с хоста).

> ⚠️ **Не пересобирать граф на сервере.** `edge_id` в `traffic_obs` ссылаются на граф, собранный из конкретной выгрузки OSM.
> Новая выгрузка → другие id. Поэтому переносим **файлы `data/osm`, `data/v2` и дамп БД** с Mac, а не качаем OSM заново.

## 0. Что узнать / получить у админа
- SSH-доступ (алиас `almaty-srv` в `~/.ssh/config` на Mac; адрес/порт в репо не пишем)
- `sudo` или членство в группе `docker`; установлен ли Docker + compose plugin
- ОС, CPU, RAM (≥ 4 ГБ, лучше 8), **диск: ~550 МБ/сутки** данных (≈ 16 ГБ/месяц) — нужно ≥ 50–100 ГБ
- Можно ли открыть порт наружу для дашборда (или только через SSH-туннель) и есть ли домен (для HTTPS)

Команды для характеристик:
```bash
uname -a; cat /etc/os-release | head -3; nproc; free -h; df -h /; docker --version; docker compose version
```

## 1. Подготовка сервера (один раз)
```bash
# ключ вместо пароля (с Mac):  ssh-copy-id -p <PORT> <user>@<host>
# на сервере, если Docker не установлен (нужен sudo):
curl -fsSL https://get.docker.com | sh && sudo usermod -aG docker $USER   # перелогиниться
git clone https://github.com/Su1tan4ik/ml-final.git && cd ml-final
```
`.env` в корне репо (в git не попадает):
```
PG_PASSWORD=<длинный случайный пароль>     # openssl rand -hex 24
PG_PORT=5442
API_PORT=8002
API_BIND=127.0.0.1        # 0.0.0.0 — если дашборд открыт наружу (лучше через reverse proxy с HTTPS)
OWM_API_KEY=<ключ OpenWeatherMap>   # без ключа погода не пишется и погодные признаки модели пустые
TG_BOT_TOKEN=<токен бота от @BotFather>   # необязательно: оповещение в Telegram, если сбор упал
TG_CHAT_ID=<id чата>                     # (одно сообщение при сбое и одно при восстановлении)
```
> Пароль БД задаётся **при первом старте пустого тома**. Если поменять потом — менять и в БД (`ALTER USER traffic PASSWORD ...`).

## 2. Перенос данных с Mac
На Mac (из корня репо):
```bash
pkill -f scrape_loop.sh                                 # остановить сбор на Mac (чтобы не было двух сборщиков)
docker exec traffic_db pg_dump -U traffic -Fc traffic > traffic.dump
tar czf v2files.tgz data/osm data/v2/pixmap_z16.npz data/v2/tiles_z16.csv models_v2 2>/dev/null
scp -P <PORT> traffic.dump v2files.tgz <user>@<host>:ml-final/
```
На сервере:
```bash
cd ml-final && tar xzf v2files.tgz
docker compose up -d db                                  # пустая БД со схемой
docker exec -i traffic_db pg_restore -U traffic -d traffic --clean --if-exists < traffic.dump
mkdir -p data/v2/logs models_v2 && sudo chown -R 1000:1000 data models_v2   # контейнер работает от uid 1000
```

## 3. Запуск
```bash
docker compose --profile server up -d --build
docker compose ps
tail -f data/v2/logs/scrape.log          # первый срез в ближайшие :02 / :32
curl -s localhost:8002/api/horizons
```
Дашборд без открытого порта (с Mac): `ssh -p <PORT> -L 8002:localhost:8002 <user>@<host>` → http://localhost:8002

## 4. Эксплуатация
- Обновить код: `git pull && docker compose --profile server up -d --build`
  (лучше не в первые ~10 мин после :00 и :30 — в это время идёт сбор; перезапуск коллектора теряет максимум один срез)
- Стоп/старт сбора: `docker compose stop collector` / `docker compose start collector`
- Переобучить вручную: `docker compose exec collector python train_v2.py --days 28`
- Бэкап (cron на хосте, раз в сутки):
  `docker exec traffic_db pg_dump -U traffic -Fc traffic > ~/backups/traffic_$(date +\%F).dump`
- Миграции существующей БД: `scripts/v2/migrate_*.py` (запуск: `docker compose exec collector python migrate_001_lanes.py`)
  - `migrate_002_forecast_index.py` — индекс `forecast(made_at)` (ускоряет карточки улиц/участков и очистку прогнозов)
- Переобучение идёт в фоне (блокировка `models_v2/.train.lock`), сбор следующего среза не ждёт его.
  Модели подменяются атомарно, `predict_v2` никогда не читает недописанный файл.
- Срезы, где скачано < 80% тайлов, не используются в обучении, рекомендациях и статистике дашборда.
- Проверка сбора: `SELECT ts, tiles_ok, rows_written, note FROM scrape_run ORDER BY ts DESC LIMIT 10;`

## 5. HTTPS (когда будет домен)
Caddy перед `api` (автоматический Let's Encrypt), пример `Caddyfile`:
```
traffic.example.kz {
    reverse_proxy 127.0.0.1:8002
}
```
Без домена — только SSH-туннель или доступ из внутренней сети.

## Проверено локально (2026-11-17)
- `docker compose --profile server build api` — образ 1.36 ГБ собирается.
- `api` на порту 8003 → `/api/horizons`, `/api/route` отвечают из контейнера (БД через сеть compose, `PG_HOST=db`).
- `collector` (`run_scrape.sh` в контейнере, `IN_DOCKER=1`) → корректно пропустил уже собранный слот.
