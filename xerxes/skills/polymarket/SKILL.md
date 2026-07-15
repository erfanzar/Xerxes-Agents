---
name: polymarket
description: Read public Polymarket market, price, order-book, history, and trade data through native Bun clients.
version: 0.3.0
tags: [polymarket, prediction-markets, market-data]
source: bundled
subcommands: [search, trending, market, event, price, book, history, trades]
---

# Polymarket

All commands use Polymarket's public read-only APIs and require no authentication or Python dependency.

```bash
xerxes skill polymarket search "election"
xerxes skill polymarket trending --limit 10
xerxes skill polymarket market <market-slug>
xerxes skill polymarket event <event-slug>
xerxes skill polymarket price <token-id>
xerxes skill polymarket book <token-id>
xerxes skill polymarket history <condition-id> --interval all --fidelity 50
xerxes skill polymarket trades --market <condition-id> --limit 10
```

The command formats probabilities as percentages and exposes identifiers needed for deeper public-data queries.
