# Testing, Packaging, Skills, Docs & CI Rules

Applies across all repository changes, release/packaging edits, `.github/workflows/`, `xerxes/skills/`, `docs/`, `examples/`, and `xerxes/playground/`.

## 1. Tiered Verification & Test Runner Routing

- Iterate with the narrowest test, then run `bun run verify` (`check && test && build`) and `git diff --check` before handing over cross-cutting changes. Add `bun run smoke` for daemon/TUI edits and `bun run docs:build` for docs/API edits. Report only completed checks.
- Match the owning test runner:
  - **Runtime, Daemon, ACP, Swarm, Hub, Slack, Desktop & VS Code**: `bun test xerxes/test/<file>.test.{ts,tsx}` (`test:runtime`; `bunfig.toml` preloads `isolatedHome.ts`).
  - **OpenTUI**: `bun run test:ui` (`vitest run --config xerxes/vitest.ui.config.ts`).
  - **Mobile PWA & Shared Client**: `bun run --cwd xerxes test:mobile` (`vitest run --config xerxes/vitest.mobile.config.ts`).
- Keep unit tests and `examples/` deterministic and offline by default via injected ports (live tests opt-in with external credentials).
- Never manually edit generated `dist/` output; rebuild via the owning Bun script. Root `bun run build` covers `runtime + ui + mobile` only (run `build:desktop` / `build:vscode` explicitly when needed).

## 2. Bun-Only Hygiene, Generated Catalogs & CI Contracts

- **Pinned runtime**: Keep Bun pinned to `1.3.12` across `Dockerfile`, `.github/workflows/{bun-ci,bun-release,vscode-marketplace}.yml`, and `xerxes/scripts/releasePackage.ts` (`engines.bun: ">=1.3.12"`).
- **Bun-only gate (`bun run repo:check`)**: `assertBunOnly.ts` (`bunOnlyRepository.test.ts`) fails on ANY tracked or unignored `.py`, `.pyc`, or root Python metadata (`.pre-commit-config.yaml`, `.python-version`, `pyproject.toml`, `requirements.txt`, `setup.cfg`, `setup.py`, `uv.lock`).
- **Image model catalog**: `check:image-catalog` (`generateImageModels.ts --check`) fails `typecheck` if `xerxes/src/images/imageModels.generated.json` drifts from `@earendil-works/pi-ai`; refresh via `bun run --cwd xerxes generate:image-catalog`.
- **CI contracts (`nativeWorkflowContract.test.ts`)**: Pin every GitHub Action in `.github/workflows/` to an exact 40-char SHA + `# vN`, keep `bun-release.yml` limited to one isolated `contents: write` job (`attach-github-release`), and ban Python/`npm publish` in workflows.

## 3. Version-Bump & Packaging Synchronization

- Every version bump (`0.8.56`) MUST update **all 9 version-bearing files + 4 hardcoded tests** (exact lines in `.agents/references/projects-map.md`):
  - Manifests (3): `package.json`, `xerxes/package.json`, `packaging/runtime/package.json`
  - Source (5): `src/core/version.ts`, `src/core/config.ts`, `src/desktop/renderer/buildInfo.ts`, `src/llms/{anthropic,mistral}.ts`
  - Docs (1): `docs/changelog.md`
  - Tests (4): `xerxes/test/{installerParity,daemonServer,desktopStore,bridgePythonParity}.test.ts`
- **Runtime packaging lockstep**: Whenever `@opentui/core`, `@opentui/react`, or `react` changes in `xerxes/package.json`, sync `packaging/runtime/package.json` and `packaging/runtime/bun.lock` (`containerDistribution.test.ts`, `nativeWorkflowContract.test.ts`).
- **Redistributable asset safety**: `copyBundledSkills.ts` and `releasePackage.ts` reject symlinks, non-regular files, executable `0o111` files under `xerxes/skills/` or `xerxes/src/agents/default/`, and non-redistributable directories (`powerpoint/*` except `SKILL.md`, `creative-ideation`, `research/research-paper-writing/templates`, `training/{axolotl,pytorch-fsdp,unsloth}`).
- **Local release flow**: `bun run release:prepare -- --output <dir>` -> `bun pm pack` -> `bun run release:check -- --package <dir> --archive <tgz>` -> `bun run release:smoke -- <tgz>`. Never push, tag, or publish unless explicitly asked.

## 4. Bundled Skills, Harness Lint, Docs, Examples & Playground

- **Strict `SKILL.md` & harness rules (`skillLint.ts`, `harnessLint.ts`, `promptScanner.ts`)**:
  - Start YAML frontmatter at byte 0 (`---` ... `---`) using only the 14 `SKILL_FRONTMATTER_KEYS`; keep `description` single-line (no `|`/`>-` block scalars, multiline lists, or nested mappings); match `name` to parent dir; avoid collisions with `COMMAND_REGISTRY` or the 119 bundled skills; never cite unshipped relative `scripts/...` paths.
  - Keep `AGENTS.md`, rules, skills, commands, and agents free of prompt-scanner threat phrases, credential shapes, hostile/oversized HTML comments, and invisible Unicode characters.
  - Respect runtime budgets: root `AGENTS.md` `< 16 KiB` (repo target `< 14,500 B`; universal rules inside first 1,500 chars for `leanPrompt.ts`), rules/workspace `.md` files `< 6,000 B` (repo target `< 5,800 B`), skill index cap 128 entries / 16 KiB.
- **Docs**: `docs/bun-docs.json` rejects retired `legacy` keys; build with `bun run docs:build` into manifest-owned `docs/_bun/`.
- **Examples**: Keep `examples/` offline by default (`--live`/`--write` opt-in); validate via `bunx tsc --noEmit -p examples/tsconfig.json` and `bun test xerxes/test/rootExamples.test.ts`.
- **Playground**: All 6 suites (`warmup`, `hard`, `sweCli.ts`, `improve`, `recall`, `matrix`) require explicit `--transport <module>` and isolated `--run-root`; never read ambient credentials or mutate `process.env.XERXES_HOME`.
