---
name: maps
description: "Geocode, POIs, routes, timezones via OpenStreetMap/OSRM."
version: 1.0.0
author: Nous Research (adapted for Xerxes)
platforms: [linux, macos, windows]
tags: [maps, geocoding, places, routing, directions, location, openstreetmap, productivity]
source: https://raw.githubusercontent.com/NousResearch/hermes-agent/main/skills/productivity/maps/SKILL.md
---

# Maps Skill

Location intelligence using free, open data sources — no API key required,
Python stdlib only.

Data sources: OpenStreetMap/Nominatim (geocoding), Overpass API (POIs), OSRM
(routing), TimeAPI.io (timezones).

**Relationship to `find-nearby`:** this skill covers broad location work
(geocoding, routing, timezones, bounding-box searches) and its `nearby`
operation overlaps with the dedicated `find-nearby` skill. For a simple
"what's near me" POI lookup, prefer `find-nearby`; reach for this skill when
the request also involves addresses, distances, turn-by-turn directions, or
timezones.

## When to Use

- User wants coordinates for a place name → `search`
- User has coordinates and wants the address → `reverse`
- User asks for nearby restaurants, hospitals, pharmacies, hotels, etc. → `nearby` (or `find-nearby` for the simple case)
- User wants driving/walking/cycling distance or travel time → `distance`
- User wants turn-by-turn directions between two places → `directions`
- User wants timezone information for a location → `timezone`
- User wants to search for POIs within a geographic area → `area` + `bbox`

## Prerequisites

Python 3.8+ (stdlib only — no pip installs needed). A live internet connection
to the public APIs above.

## Inline Python client

Xerxes skills ship no script assets. Save this client to a temp file with the
file write tool, run it with `python3`, and delete it when the turn ends
(justified: one bounded helper file inside the workspace temp dir):

```python
#!/usr/bin/env python3
"""maps.py — geocode / POI / routing helpers on free open APIs."""
import json, sys, time, urllib.parse, urllib.request

UA = {"User-Agent": "xerxes-maps-skill/1.0 (personal productivity)"}
OVERPASS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"]

def get(url):
    req = urllib.request.Request(url, headers=UA)
    return json.load(urllib.request.urlopen(req, timeout=30))

def geocode(q):
    u = "https://nominatim.openstreetmap.org/search?format=json&limit=1&q=" + urllib.parse.quote(q)
    r = get(u)
    time.sleep(1.1)  # Nominatim ToS: max 1 req/s
    return r[0] if r else None

CATEGORY_MAP = {  # common categories -> OSM tag filters
    "restaurant": '["amenity"="restaurant"]', "cafe": '["amenity"="cafe"]',
    "bar": '["amenity"="bar"]', "hospital": '["amenity"="hospital"]',
    "pharmacy": '["amenity"="pharmacy"]', "hotel": '["tourism"="hotel"]',
    "supermarket": '["shop"="supermarket"]', "atm": '["amenity"="atm"]',
    "gas_station": '["amenity"="fuel"]', "parking": '["amenity"="parking"]',
    "museum": '["tourism"="museum"]', "park": '["leisure"="park"]',
    "school": '["amenity"="school"]', "university": '["amenity"="university"]',
    "bank": '["amenity"="bank"]', "police": '["amenity"="police"]',
    "library": '["amenity"="library"]', "dentist": '["amenity"="dentist"]',
    "doctor": '["amenity"="doctors"]', "cinema": '["amenity"="cinema"]',
    "gym": '["leisure"="fitness_centre"]', "post_office": '["amenity"="post_office"]',
    "bakery": '["shop"="bakery"]', "nightclub": '["amenity"="nightclub"]',
}

def nearby(lat, lon, category, radius=1000, limit=10):
    f = CATEGORY_MAP.get(category)
    if not f:
        print(f"unknown category: {category}", file=sys.stderr); return
    q = f'[out:json][timeout:25];node(around:{radius},{lat},{lon}){f};out {limit};'
    for mirror in OVERPASS:
        try:
            data = urllib.parse.urlencode({"data": q}).encode()
            req = urllib.request.Request(mirror, data=data, headers=UA)
            els = json.load(urllib.request.urlopen(req, timeout=60))["elements"]
            for e in els:
                tags = e.get("tags", {})
                ll = f"{e['lat']},{e['lon']}"
                print(f"- {tags.get('name','(unnamed)')} | {ll} | {tags.get('opening_hours','')} | {tags.get('phone','')}")
            return
        except Exception:
            continue
    print("all Overpass mirrors failed", file=sys.stderr)

def osrm(mode, a, b):
    prof = {"driving": "driving", "walking": "foot", "cycling": "bike"}[mode]
    def coord(p):
        g = geocode(p) or sys.exit(f"cannot geocode: {p}")
        return f"{g['lon']},{g['lat']}"
    u = f"https://router.project-osrm.org/route/v1/{prof}/{coord(a)};{coord(b)}?overview=false&steps=true"
    r = get(u)["routes"][0]
    print(f"distance: {r['distance']/1000:.1f} km, duration: {r['duration']/60:.0f} min")
    for i, s in enumerate(r["legs"][0]["steps"], 1):
        print(f"{i}. {s.get('maneuver',{}).get('type','')} {s.get('name','')} ({s['distance']:.0f} m)")

def tz(lat, lon):
    r = get(f"https://timeapi.io/api/Time/zone?latitude={lat}&longitude={lon}")
    print(f"timezone: {r.get('timeZone')} (UTC{r.get('currentUtcOffset','')})")

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "search": print(json.dumps(geocode(" ".join(sys.argv[2:])), indent=2))
    elif cmd == "reverse":
        lat, lon = sys.argv[2], sys.argv[3]
        u = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}"
        print(json.dumps(get(u), indent=2))
    elif cmd == "nearby":
        args = sys.argv[2:]
        if "--near" in args:
            i = args.index("--near"); place = args[i+1]
            g = geocode(place); lat, lon = g["lat"], g["lon"]
            args = args[:i] + args[i+2:]
        else: lat, lon = args[0], args[1]; args = args[2:]
        cat = args[0]; rad = int(args[1]) if len(args) > 1 else 1000
        nearby(lat, lon, cat, rad)
    elif cmd == "distance": osrm(sys.argv[2] if sys.argv[2] in ("driving","walking","cycling") else "driving", sys.argv[-2], sys.argv[-1])
    elif cmd == "directions": osrm(sys.argv[2] if sys.argv[2] in ("driving","walking","cycling") else "driving", sys.argv[-2], sys.argv[-1])
    elif cmd == "timezone": tz(sys.argv[2], sys.argv[3])
    else: print("commands: search, reverse, nearby, distance, directions, timezone", file=sys.stderr)
```

Invoke as `python3 /tmp/maps.py <command> …` (examples below use this shape).

## Commands

### search — Geocode a place name
```bash
python3 /tmp/maps.py search "Eiffel Tower"
python3 /tmp/maps.py search "1600 Pennsylvania Ave, Washington DC"
```
Returns: lat, lon, display name, type, bounding box, importance score.

### reverse — Coordinates to address
```bash
python3 /tmp/maps.py reverse 48.8584 2.2945
```
Returns: full address breakdown (street, city, state, country, postcode).

### nearby — Find places by category
```bash
python3 /tmp/maps.py nearby 48.8584 2.2945 restaurant 500
python3 /tmp/maps.py nearby --near "Times Square, New York" cafe
```
Covered categories (extensible via `CATEGORY_MAP`): restaurant, cafe, bar,
hospital, pharmacy, hotel, supermarket, atm, gas_station, parking, museum,
park, school, university, bank, police, library, dentist, doctor, cinema, gym,
post_office, bakery, nightclub — plus any OSM amenity/shop/leisure tag you add.
Each result includes name, coordinates, hours, and phone when available. Build
a clickable map link as `https://www.google.com/maps?q=<lat>,<lon>`.

### distance — Travel distance and time
```bash
python3 /tmp/maps.py distance driving "Paris" "Lyon"
python3 /tmp/maps.py distance walking "Big Ben" "Tower Bridge"
```
Modes: driving (default), walking, cycling. Returns road distance and duration
plus numbered turn-by-turn steps.

### directions — Turn-by-turn navigation
```bash
python3 /tmp/maps.py directions walking "Eiffel Tower" "Louvre Museum"
```
Returns numbered steps with distance, road name, and maneuver type.

### timezone — Timezone for coordinates
```bash
python3 /tmp/maps.py timezone 48.8584 2.2945
```
Returns timezone name and UTC offset.

## Workflow Examples

**"Find Italian restaurants near the Colosseum":**
1. `python3 /tmp/maps.py nearby --near "Colosseum Rome" restaurant` — auto-geocoded.

**"How do I walk from the hotel to the conference center?":**
1. `python3 /tmp/maps.py directions walking "Hotel Name" "Conference Center"`.

**"What's the address at these coordinates?":**
1. `python3 /tmp/maps.py reverse 48.8584 2.2945`.

## Pitfalls

- Nominatim ToS: max 1 req/s (the client sleeps automatically between geocode calls).
- `nearby` needs either lat/lon or `--near "<address>"` — one of the two is required.
- OSRM public-server routing coverage is best for Europe and North America; for production routing, self-host OSRM.
- Overpass can be slow at peak hours; the client falls back between mirrors automatically.
- If a place name alone gives ambiguous results globally, include country/state.
- Clean up the temp `maps.py` file at the end of the turn.

## Verification

```bash
python3 /tmp/maps.py search "Statue of Liberty"
# Should return lat ~40.689, lon ~-74.044

python3 /tmp/maps.py nearby --near "Times Square" restaurant
# Should return a list of restaurants within ~1km of Times Square
```

---

Adapted from the `maps` skill in [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) (MIT License), copyright Mibayy.
