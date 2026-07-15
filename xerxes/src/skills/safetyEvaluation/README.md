# Defensive safety evaluation

This Bun/TypeScript package intentionally does not port the legacy offensive skill behavior. It excludes attack-prompt construction, refusal suppression, automatic canary execution, and any model-control bypass logic.

Instead, it offers a defensive evaluation boundary:

- Models are explicit caller-injected ports; the package does not load credentials, select providers, or start processes.
- Built-in probes are benign policy and compliance checks, with caller-supplied probes expected to follow the same boundary.
- Untrusted text is normalized and classified for defensive review without decoding or executing it.
- Scores are deterministic and expose every criterion, weight, and rationale.
- Reports stay in memory unless a caller explicitly supplies `SafetyEvaluationReportStore`; `JsonlSafetyReportStore` is the native opt-in durable implementation.
