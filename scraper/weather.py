"""
Optional weather data collection via OpenWeatherMap free tier.
One API call per scraper run (city-level, not per tile).

Returns a dict with weather fields, or a dict of None values if
OWM_API_KEY is not configured or the call fails.

Free key: https://openweathermap.org/api  (Current Weather Data — free tier)
"""

import logging

import aiohttp

from config import OWM_API_KEY, OWM_CITY

log = logging.getLogger(__name__)

OWM_URL = "https://api.openweathermap.org/data/2.5/weather"

EMPTY = {
    "weather_temp_c":    None,
    "weather_feels_c":   None,
    "weather_humidity":  None,
    "weather_wind_ms":   None,
    "weather_precip_1h": None,   # mm in last hour
    "weather_visibility_m": None,
    "weather_condition": None,   # e.g. "Clear", "Snow", "Rain"
    "weather_description": None, # e.g. "light snow"
}


async def fetch_weather(session: aiohttp.ClientSession) -> dict:
    """
    Fetch current weather for Almaty.
    Returns a flat dict with weather_* keys.
    If OWM_API_KEY is empty, returns EMPTY dict silently.
    """
    if not OWM_API_KEY:
        return EMPTY.copy()

    params = {
        "q":     OWM_CITY,
        "appid": OWM_API_KEY,
        "units": "metric",
        "lang":  "en",
    }

    try:
        async with session.get(
            OWM_URL,
            params=params,
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            if resp.status != 200:
                log.warning("OpenWeatherMap returned HTTP %s — skipping weather.", resp.status)
                return EMPTY.copy()

            data = await resp.json()

            rain = data.get("rain", {})
            snow = data.get("snow", {})
            precip = rain.get("1h", 0.0) + snow.get("1h", 0.0)

            weather_list = data.get("weather", [{}])
            condition   = weather_list[0].get("main", "")
            description = weather_list[0].get("description", "")

            main = data.get("main", {})
            wind = data.get("wind", {})

            return {
                "weather_temp_c":      round(main.get("temp", 0), 1),
                "weather_feels_c":     round(main.get("feels_like", 0), 1),
                "weather_humidity":    main.get("humidity"),
                "weather_wind_ms":     round(wind.get("speed", 0), 1),
                "weather_precip_1h":   round(precip, 2),
                "weather_visibility_m": data.get("visibility"),
                "weather_condition":   condition,
                "weather_description": description,
            }

    except Exception as exc:
        log.warning("Weather fetch failed: %s", exc)
        return EMPTY.copy()
