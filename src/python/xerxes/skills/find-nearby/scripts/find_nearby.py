# Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Find nearby module for Xerxes.

Exports:
    - OVERPASS_URLS
    - NOMINATIM_URL
    - USER_AGENT
    - TIMEOUT
    - haversine
    - geocode
    - find_nearby
    - main"""

import argparse
import json
import math
import sys
import urllib.parse
import urllib.request
from typing import Any

OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "HermesAgent/1.0 (find-nearby skill)"
TIMEOUT = 15


def _http_get(url: str) -> Any:
    """Internal helper to http get.

    Args:
        url (str): IN: url. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    """Internal helper to http get.

    Args:
        url (str): IN: url. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""
    """Internal helper to http get.

    Args:
        url (str): IN: url. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read())


def _http_post(url: str, data: str) -> Any:
    """Internal helper to http post.

    Args:
        url (str): IN: url. OUT: Consumed during execution.
        data (str): IN: data. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""
    req = urllib.request.Request(url, data=data.encode(), headers={"User-Agent": USER_AGENT})
    """Internal helper to http post.

    Args:
        url (str): IN: url. OUT: Consumed during execution.
        data (str): IN: data. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""
    """Internal helper to http post.

    Args:
        url (str): IN: url. OUT: Consumed during execution.
        data (str): IN: data. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read())


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine.

    Args:
        lat1 (float): IN: lat1. OUT: Consumed during execution.
        lon1 (float): IN: lon1. OUT: Consumed during execution.
        lat2 (float): IN: lat2. OUT: Consumed during execution.
        lon2 (float): IN: lon2. OUT: Consumed during execution.
    Returns:
        float: OUT: Result of the operation."""

    R = 6_371_000
    """Haversine.

    Args:
        lat1 (float): IN: lat1. OUT: Consumed during execution.
        lon1 (float): IN: lon1. OUT: Consumed during execution.
        lat2 (float): IN: lat2. OUT: Consumed during execution.
        lon2 (float): IN: lon2. OUT: Consumed during execution.
    Returns:
        float: OUT: Result of the operation."""
    """Haversine.

    Args:
        lat1 (float): IN: lat1. OUT: Consumed during execution.
        lon1 (float): IN: lon1. OUT: Consumed during execution.
        lat2 (float): IN: lat2. OUT: Consumed during execution.
        lon2 (float): IN: lon2. OUT: Consumed during execution.
    Returns:
        float: OUT: Result of the operation."""
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def geocode(query: str) -> tuple[float, float]:
    """Geocode.

    Args:
        query (str): IN: query. OUT: Consumed during execution.
    Returns:
        tuple[float, float]: OUT: Result of the operation."""

    params = urllib.parse.urlencode({"q": query, "format": "json", "limit": 1})
    """Geocode.

    Args:
        query (str): IN: query. OUT: Consumed during execution.
    Returns:
        tuple[float, float]: OUT: Result of the operation."""
    """Geocode.

    Args:
        query (str): IN: query. OUT: Consumed during execution.
    Returns:
        tuple[float, float]: OUT: Result of the operation."""
    results = _http_get(f"{NOMINATIM_URL}?{params}")
    if not results:
        print(f"Error: Could not geocode '{query}'. Try a more specific address.", file=sys.stderr)
        sys.exit(1)
    return float(results[0]["lat"]), float(results[0]["lon"])


def find_nearby(lat: float, lon: float, types: list[str], radius: int = 1500, limit: int = 15) -> list[dict]:
    """Find nearby.

    Args:
        lat (float): IN: lat. OUT: Consumed during execution.
        lon (float): IN: lon. OUT: Consumed during execution.
        types (list[str]): IN: types. OUT: Consumed during execution.
        radius (int, optional): IN: radius. Defaults to 1500. OUT: Consumed during execution.
        limit (int, optional): IN: limit. Defaults to 15. OUT: Consumed during execution.
    Returns:
        list[dict]: OUT: Result of the operation."""

    type_filters = "".join(f'nwr["amenity"="{t}"](around:{radius},{lat},{lon});' for t in types)
    """Find nearby.

    Args:
        lat (float): IN: lat. OUT: Consumed during execution.
        lon (float): IN: lon. OUT: Consumed during execution.
        types (list[str]): IN: types. OUT: Consumed during execution.
        radius (int, optional): IN: radius. Defaults to 1500. OUT: Consumed during execution.
        limit (int, optional): IN: limit. Defaults to 15. OUT: Consumed during execution.
    Returns:
        list[dict]: OUT: Result of the operation."""
    """Find nearby.

    Args:
        lat (float): IN: lat. OUT: Consumed during execution.
        lon (float): IN: lon. OUT: Consumed during execution.
        types (list[str]): IN: types. OUT: Consumed during execution.
        radius (int, optional): IN: radius. Defaults to 1500. OUT: Consumed during execution.
        limit (int, optional): IN: limit. Defaults to 15. OUT: Consumed during execution.
    Returns:
        list[dict]: OUT: Result of the operation."""
    query = f"[out:json][timeout:{TIMEOUT}];({type_filters});out center tags;"

    data = None
    for url in OVERPASS_URLS:
        try:
            data = _http_post(url, f"data={urllib.parse.quote(query)}")
            break
        except Exception:
            continue

    if not data:
        return []

    places = []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        name = tags.get("name")
        if not name:
            continue

        plat = el.get("lat") or (el.get("center", {}) or {}).get("lat")
        plon = el.get("lon") or (el.get("center", {}) or {}).get("lon")
        if not plat or not plon:
            continue

        dist = haversine(lat, lon, plat, plon)

        place = {
            "name": name,
            "type": tags.get("amenity", ""),
            "distance_m": round(dist),
            "lat": plat,
            "lon": plon,
            "maps_url": f"https://www.google.com/maps/search/?api=1&query={plat},{plon}",
            "directions_url": f"https://www.google.com/maps/dir/?api=1&origin={lat},{lon}&destination={plat},{plon}",
        }

        if tags.get("cuisine"):
            place["cuisine"] = tags["cuisine"]
        if tags.get("opening_hours"):
            place["hours"] = tags["opening_hours"]
        if tags.get("phone"):
            place["phone"] = tags["phone"]
        if tags.get("website"):
            place["website"] = tags["website"]
        if tags.get("addr:street"):
            addr_parts = [tags.get("addr:housenumber", ""), tags.get("addr:street", "")]
            if tags.get("addr:city"):
                addr_parts.append(tags["addr:city"])
            place["address"] = " ".join(p for p in addr_parts if p)

        places.append(place)

    places.sort(key=lambda p: p["distance_m"])
    return places[:limit]


def main():
    """Main.

    Returns:
        Any: OUT: Result of the operation."""
    parser = argparse.ArgumentParser(description="Find nearby places via OpenStreetMap")
    """Main.

    Returns:
        Any: OUT: Result of the operation."""
    """Main.

    Returns:
        Any: OUT: Result of the operation."""
    parser.add_argument("--lat", type=float, help="Latitude")
    parser.add_argument("--lon", type=float, help="Longitude")
    parser.add_argument("--near", type=str, help="Address, city, or zip code (geocoded automatically)")
    parser.add_argument(
        "--type", action="append", dest="types", default=[], help="Place type (restaurant, cafe, bar, pharmacy, etc.)"
    )
    parser.add_argument("--radius", type=int, default=1500, help="Search radius in meters (default: 1500)")
    parser.add_argument("--limit", type=int, default=15, help="Max results (default: 15)")
    parser.add_argument("--json", action="store_true", dest="json_output", help="Output as JSON")
    args = parser.parse_args()

    if args.near:
        lat, lon = geocode(args.near)
    elif args.lat is not None and args.lon is not None:
        lat, lon = args.lat, args.lon
    else:
        print("Error: Provide --lat/--lon or --near", file=sys.stderr)
        sys.exit(1)

    if not args.types:
        args.types = ["restaurant"]

    places = find_nearby(lat, lon, args.types, args.radius, args.limit)

    if args.json_output:
        print(json.dumps({"origin": {"lat": lat, "lon": lon}, "results": places, "count": len(places)}, indent=2))
    else:
        if not places:
            print(f"No {'/'.join(args.types)} found within {args.radius}m")
            return
        print(f"Found {len(places)} places within {args.radius}m:\n")
        for i, p in enumerate(places, 1):
            dist_str = f"{p['distance_m']}m" if p["distance_m"] < 1000 else f"{p['distance_m'] / 1000:.1f}km"
            print(f"  {i}. {p['name']} ({p['type']}) — {dist_str}")
            if p.get("cuisine"):
                print(f"     Cuisine: {p['cuisine']}")
            if p.get("hours"):
                print(f"     Hours: {p['hours']}")
            if p.get("address"):
                print(f"     Address: {p['address']}")
            print(f"     Map: {p['maps_url']}")
            print()


if __name__ == "__main__":
    main()
