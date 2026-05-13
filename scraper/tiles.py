"""
Tile math utilities.

Converts between lat/lon coordinates and XYZ slippy-map tile indices
(the standard used by 2GIS, OpenStreetMap, Google Maps, etc.).
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Tile:
    x: int
    y: int
    z: int

    @property
    def segment_id(self) -> str:
        return f"z{self.z}_x{self.x}_y{self.y}"

    def center_latlon(self) -> tuple[float, float]:
        """Returns the (lat, lon) of the tile's centre pixel."""
        n = 2 ** self.z
        lon = (self.x + 0.5) / n * 360.0 - 180.0
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * (self.y + 0.5) / n)))
        lat = math.degrees(lat_rad)
        return round(lat, 6), round(lon, 6)


def latlon_to_tile(lat: float, lon: float, zoom: int) -> Tile:
    """Convert a lat/lon pair to the tile that contains it at the given zoom."""
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return Tile(x=x, y=y, z=zoom)


def tiles_for_bbox(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    zoom: int,
) -> list[Tile]:
    """
    Returns all tiles that overlap the given bounding box at the given zoom.
    Note: in tile coordinates Y increases southward, so lat_max → smaller y.
    """
    top_left = latlon_to_tile(lat_max, lon_min, zoom)
    bottom_right = latlon_to_tile(lat_min, lon_max, zoom)

    tiles = []
    for x in range(top_left.x, bottom_right.x + 1):
        for y in range(top_left.y, bottom_right.y + 1):
            tiles.append(Tile(x=x, y=y, z=zoom))
    return tiles


if __name__ == "__main__":
    from config import ALMATY_BOUNDS, ZOOM

    tiles = tiles_for_bbox(
        ALMATY_BOUNDS["lat_min"],
        ALMATY_BOUNDS["lat_max"],
        ALMATY_BOUNDS["lon_min"],
        ALMATY_BOUNDS["lon_max"],
        ZOOM,
    )
    print(f"Tiles at zoom {ZOOM}: {len(tiles)}")
    print(f"Sample tile: {tiles[0]}  centre: {tiles[0].center_latlon()}")
