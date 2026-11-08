# Benchmark comparison suite (developer-only)

Adapters read a **locally downloaded** dataset directory. Nothing is ever fetched. A directory with the wrong
layout raises `BenchmarkDatasetError`, which names the expected files.

Run: `bun run --cwd xerxes playground:matrix -- --matrix matrix.json [--out dir] [--run-root dir]`

## Matrix file

```json
{
  "configs": [
    { "id": "local-only", "kind": "xerxes", "privacy": "constrained", "transport": "/abs/transport.ts" },
    { "id": "pi", "kind": "external", "privacy": "unrestricted", "timeoutMs": 600000,
      "argv": ["pi", "--prompt-file", "{prompt_file}", "--cwd", "{workdir}"] }
  ],
  "benchmarks": [
    { "adapter": "dabstep", "datasetDir": "data/dabstep", "limit": 20 },
    { "adapter": "synthetic-private" }
  ]
}
```

- `privacy` is `constrained` or `unrestricted` (default `unrestricted`). The two are reported in separate sections.
- `xerxes` transports export `createEvaluationSessionPort(context)`, as for `playground:hard`. A port may also
  implement `getRunMetrics(): { inputTokens?, outputTokens?, costUsd?, outboundTexts? }`.
- `external` argv is run without a shell. `{prompt_file}`, `{workdir}` and `{metrics_file}` are substituted.
  stdout is the final reply. The harness may write JSON metrics (same keys) to `{metrics_file}`.
- Metrics that are not reported stay absent (`n/a` in the report). Cost per accepted result is `n/a` when nothing
  was accepted or no cost was reported.

## Dataset layouts

| Adapter | Expected files | Grading |
| --- | --- | --- |
| `dabstep` | `tasks/dev.jsonl` (`task_id`, `question`, `answer`, optional `guidelines`, `level`); optional `context/` | Normalized answer match on the last `FINAL ANSWER:` line (numbers, currency, precision, unordered lists). Rows with no answer are excluded. |
| `spreadsheetbench` | `dataset.json` (`id`, `instruction`, `answer_position`, optional `spreadsheet_path`); `<folder>/1_<id>_input.xlsx` and `1_<id>_answer.xlsx` | The model saves `output.xlsx` (or `output.csv`). The answer cells are compared with the expected workbook. |
| `gdpval` | `tasks.jsonl` (`task_id`, `sector`, `occupation`, `prompt`, optional `reference_files`); optional `reviews.json` (`{ "<task_id>": { accepted, score?, reviewer?, notes? } }`) | Judgment-based. With no reviewer decision the task is `needs_review` and a review packet is written. It is never auto-accepted. |
| `terminal-bench` | `tasks/<task>/task.yaml` (`instruction`); optional `xerxes-checks.json`; optional `workspace/` | Only tasks with a TS-native `xerxes-checks.json` run. Others are excluded as `requires container grader`. |
| `synthetic-private` | none (generated) | Checks the output and reports `disclosure_failure` if a planted canary appears in reported outbound text. |

`xerxes-checks.json` is an array of `{ "kind": "file_exists" | "file_contains" | "file_equals" | "answer_contains", "path"?, "text"? }`.
