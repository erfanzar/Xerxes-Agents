---
name: arxiv
description: Search arXiv papers by query, author, category, or identifier using the native Bun client.
version: 0.3.0
tags: [research, arxiv, papers, academic]
source: bundled
subcommands: [arxiv]
---

# arXiv Research

Use the native client to query arXiv's public Atom API and render readable paper metadata:

```bash
xerxes skill arxiv "GRPO reinforcement learning" --max 10 --sort date
xerxes skill arxiv --author "Yann LeCun" --max 5
xerxes skill arxiv --category cs.AI --sort date
xerxes skill arxiv --id 2402.03300,2401.12345
```

`--sort` accepts `relevance`, `date`, or `updated`. The command uses no SDK and no Python process.
