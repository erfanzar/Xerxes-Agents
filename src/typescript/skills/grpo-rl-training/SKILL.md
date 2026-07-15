---
name: grpo-rl-training
description: Plan and orchestrate GRPO training through a native Bun workflow with explicit accelerator ownership.
version: 0.3.0
tags: [post-training, reinforcement-learning, GRPO, reward-modeling, reasoning, structured-output]
dependencies: [bun]
source: bundled
subcommands: [dry-run]
---

# GRPO/RL Training

Use the native Bun workflow to validate GSM8K input, construct a reference GRPO request,
evaluate structured-output rewards, and coordinate a caller-owned training host.

```bash
xerxes skill grpo-rl-training --dry-run --dataset ./gsm8k.jsonl
```

The JSONL file must have string `question` and `answer` fields. The final answer is read
from the segment after `####` in each GSM8K answer.

The dry run emits a JSON request with the reference Qwen 2.5 1.5B, LoRA, and reward
configuration. It does not allocate a GPU, load a model, run an optimizer, or write model
bytes.

## Host-Owned Execution

Actual execution is available only to a Bun/TypeScript embedding application that injects:

- an `accelerator` port that starts and waits for a training job;
- a `storage` port for run records, metrics, checkpoints, and final-model references;
- optionally, a `reporter` for lifecycle events.

Use `BundledSkills.GrpoTraining.runBasicGrpoTraining` with those explicit ports. The host
chooses its accelerator SDK, remote training service, or JavaScript/WASM backend. The
workflow never uses a Python fallback and never fabricates unavailable accelerator services.

## Default Reward Plan

- `incremental-format`: `0.125` for each XML tag and a trailing-text penalty;
- `format`: `0.5` for complete `<reasoning>` then `<answer>` structure;
- `correctness`: `2.0` for an exact extracted answer match.

See the source API under `src/skills/grpoTraining/` for the request types, data adapters,
reward functions, and host lifecycle contracts.
