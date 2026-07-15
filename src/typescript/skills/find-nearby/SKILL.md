---
name: find-nearby
description: Find nearby public places through OpenStreetMap using the native Bun client.
version: 0.3.0
tags: [location, maps, nearby, places]
source: bundled
subcommands: [find-nearby]
---

# Find Nearby

Search by coordinates or a location name. The command uses OpenStreetMap's public geocoder and Overpass data; no API key or Python package is required.

```bash
# Coordinates
xerxes skill find-nearby --lat 40.7580 --lon -73.9855 --type cafe --radius 1500

# Address, city, zip code, or landmark
xerxes skill find-nearby --near "Times Square, New York" --type restaurant --type bar --limit 10

# Machine-readable results
xerxes skill find-nearby --near "90210" --type pharmacy --json
```

`--type` is repeatable. Supported flags are `--near`, `--lat`, `--lon`, `--type`, `--radius`, `--limit`, and `--json`.
