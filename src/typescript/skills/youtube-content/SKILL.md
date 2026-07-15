---
name: youtube-content
description: Fetch a YouTube transcript through the native Bun client and return structured transcript data.
version: 0.3.0
tags: [youtube, transcript, video, research]
source: bundled
subcommands: [youtube-transcript]
---

# YouTube Content

Fetch captions from a standard YouTube URL, short link, Shorts URL, embed URL, live URL, or raw video ID:

```bash
# Structured JSON with metadata
xerxes skill youtube-transcript "https://youtube.com/watch?v=VIDEO_ID"

# Plain transcript text
xerxes skill youtube-transcript "VIDEO_ID" --text-only

# Timestamped text and language preference
xerxes skill youtube-transcript "VIDEO_ID" --language tr,en --timestamps --text-only
```

The native client discovers available caption tracks and parses JSON3 or XML captions. It does not invoke Python or require `youtube-transcript-api`.
