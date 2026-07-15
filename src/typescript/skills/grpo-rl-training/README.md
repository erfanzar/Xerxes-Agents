# GRPO/RL Training Skill

The GRPO template is now native Bun/TypeScript code. It validates GSM8K JSONL data,
builds the reference GRPO request, evaluates structured-output rewards, and coordinates
host-owned execution without requiring a Python runtime.

## Start With a Dry Run

```bash
xerxes skill grpo-rl-training --dry-run --dataset ./gsm8k.jsonl
```

Each JSONL record must contain string `question` and `answer` fields. The final GSM8K
answer is read from the text following `####`.

## Execution Boundary

The native workflow does not invent an accelerator. A Bun embedding application must
explicitly supply an accelerator provider plus durable storage before it can execute a
job. That provider owns model loading, device placement, optimization, checkpoint
materialization, and final artifact storage.

See [SKILL.md](SKILL.md) for the reference configuration, reward semantics, and host
integration contract.
