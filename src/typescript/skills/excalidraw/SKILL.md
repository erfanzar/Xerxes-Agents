---
name: excalidraw
description: Upload an Excalidraw JSON document and return a shareable link without a Python dependency.
version: 0.3.0
tags: [diagrams, excalidraw, visualization]
source: bundled
subcommands: [upload]
---

# Excalidraw

Create a standard `.excalidraw` file, then upload it with the native Bun command:

```bash
xerxes skill excalidraw upload ./architecture.excalidraw
```

The command reads the JSON document, uses the native Excalidraw upload protocol, and prints the shareable URL. It does not invoke Python or require a Python package.
