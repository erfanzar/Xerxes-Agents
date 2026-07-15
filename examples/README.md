# Bun / TypeScript examples

Each example is a native Bun/TypeScript program. The TypeScript versions use the native runtime directly and default to a local,
deterministic LLM or host port, so running an example never reads provider
credentials from the environment or makes a network call.

Run an example from the repository root:

```sh
bun examples/scenario_1_conversational_assistant.ts
bun examples/cortex_deepsearch_agent.ts --topic "Bun-native agent runtimes"
bun examples/cortex_parallel_benchmark.ts --agents 4 --min-context-tokens 512
```

Examples that use an LLM accept a deliberate live mode. Supply all connection
details on the command line; nothing is inferred from environment variables:

```sh
bun examples/cortex_parallel_benchmark.ts \
  --live --model gpt-4o-mini --base-url https://api.example.test/v1 --api-key "$KEY"
```

Generated reports are opt-in too. Add `--write`, optionally with
`--output path/to/report.json`. Validate the complete TypeScript example set
without running external services with:

```sh
bunx tsc --noEmit -p examples/tsconfig.json
bun test src/typescript/test/rootExamples.test.ts
```
