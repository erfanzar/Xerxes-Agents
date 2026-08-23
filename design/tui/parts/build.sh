#!/usr/bin/env bash
# Assembles screens/*.body.html + parts/base.css into self-contained HTML files.
set -euo pipefail
cd "$(dirname "$0")/.."
for f in screens/*.body.html; do
  name="$(basename "$f" .body.html)"
  title="$(sed -n '1s/^<!--TITLE: \(.*\)-->$/\1/p' "$f")"
  {
    printf '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
    printf '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
    printf '<title>Xerxes TUI — %s</title>\n<style>\n' "$title"
    cat parts/base.css
    printf '</style>\n</head>\n<body>\n'
    tail -n +2 "$f"
    printf '</body>\n</html>\n'
  } > "$name.html"
  echo "built $name.html"
done
