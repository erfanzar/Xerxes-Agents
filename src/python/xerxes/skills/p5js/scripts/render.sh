#!/bin/bash
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
set -euo pipefail

# Defaults
WIDTH=1920
HEIGHT=1080
FPS=30
DURATION=10
CRF=18
FRAMES_ONLY=false

# Parse arguments
INPUT="${1:?Usage: render.sh <input.html> <output.mp4> [options]}"
OUTPUT="${2:?Usage: render.sh <input.html> <output.mp4> [options]}"
shift 2

while [[ $# -gt 0 ]]; do
  case $1 in
    --width) WIDTH="$2"; shift 2 ;;
    --height) HEIGHT="$2"; shift 2 ;;
    --fps) FPS="$2"; shift 2 ;;
    --duration) DURATION="$2"; shift 2 ;;
    --quality) CRF="$2"; shift 2 ;;
    --frames-only) FRAMES_ONLY=true; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

TOTAL_FRAMES=$((FPS * DURATION))
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FRAME_DIR=$(mktemp -d)

echo "=== p5.js Render Pipeline ==="
echo "Input:      $INPUT"
echo "Output:     $OUTPUT"
echo "Resolution: ${WIDTH}x${HEIGHT}"
echo "FPS:        $FPS"
echo "Duration:   ${DURATION}s (${TOTAL_FRAMES} frames)"
echo "Quality:    CRF $CRF"
echo "Frame dir:  $FRAME_DIR"
echo ""

# Check dependencies
command -v node >/dev/null 2>&1 || { echo "Error: Node.js required"; exit 1; }
if [ "$FRAMES_ONLY" = false ]; then
  command -v ffmpeg >/dev/null 2>&1 || { echo "Error: ffmpeg required for MP4"; exit 1; }
fi

# Step 1: Capture frames via Puppeteer
echo "Step 1/2: Capturing ${TOTAL_FRAMES} frames..."
node "$SCRIPT_DIR/export-frames.js" \
  "$INPUT" \
  --output "$FRAME_DIR" \
  --width "$WIDTH" \
  --height "$HEIGHT" \
  --frames "$TOTAL_FRAMES" \
  --fps "$FPS"

echo "Frames captured to $FRAME_DIR"

if [ "$FRAMES_ONLY" = true ]; then
  echo "Frames saved to: $FRAME_DIR"
  echo "To encode manually:"
  echo "  ffmpeg -framerate $FPS -i $FRAME_DIR/frame-%04d.png -c:v libx264 -crf $CRF -pix_fmt yuv420p $OUTPUT"
  exit 0
fi

# Step 2: Encode to MP4
echo "Step 2/2: Encoding MP4..."
ffmpeg -y \
  -framerate "$FPS" \
  -i "$FRAME_DIR/frame-%04d.png" \
  -c:v libx264 \
  -preset slow \
  -crf "$CRF" \
  -pix_fmt yuv420p \
  -movflags +faststart \
  "$OUTPUT" \
  2>"$FRAME_DIR/ffmpeg.log"

# Cleanup
rm -rf "$FRAME_DIR"

# Report
FILE_SIZE=$(ls -lh "$OUTPUT" | awk '{print $5}')
echo ""
echo "=== Done ==="
echo "Output: $OUTPUT ($FILE_SIZE)"
echo "Duration: ${DURATION}s at ${FPS}fps, ${WIDTH}x${HEIGHT}"
