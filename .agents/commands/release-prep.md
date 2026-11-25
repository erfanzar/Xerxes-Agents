---
name: release-prep
description: Audit version sync across all 9 files and 4 tests, verify packaging/runtime parity, run CI-parity verification, and stage/check/smoke the Bun release tarball without publishing.
tags: [release, packaging, version, ci]
---

# Release Readiness & Version-Sync Audit

Run a thorough release readiness audit for target version or mode: `$ARGUMENTS` (pass a semver like `0.8.57`, or `audit` / empty to validate the current version in `package.json`).

**Hard safety rule**: NEVER stage, commit, tag, push, create a GitHub release, or publish npm/Marketplace/VSIX artifacts unless the user explicitly asks you to execute that release action.

## 1. Version-Sync Audit (All 9 Sites + 4 Hardcoded Tests)

Confirm every file below matches the exact target semver (note: `releasePackage.ts` only checks `package.json` and `xerxes/package.json` — the other sites must be verified explicitly):

- **Manifests (3)**:
  1. `package.json` (`version`)
  2. `xerxes/package.json` (`version`)
  3. `packaging/runtime/package.json` (`version`)
- **Source Constants & Headers (5)**:
  4. `xerxes/src/core/version.ts` (`XERXES_VERSION`)
  5. `xerxes/src/core/config.ts` (`ObservabilityConfig` `serviceVersion` default)
  6. `xerxes/src/desktop/renderer/buildInfo.ts` (fallback version string)
  7. `xerxes/src/llms/anthropic.ts` (`User-Agent`)
  8. `xerxes/src/llms/mistral.ts` (`User-Agent`)
- **Docs (1)**:
  9. `docs/changelog.md` (release heading entry)
- **Hardcoded Tests (4 — fail if unbumped)**:
  10. `xerxes/test/installerParity.test.ts`
  11. `xerxes/test/daemonServer.test.ts`
  12. `xerxes/test/desktopStore.test.ts`
  13. `xerxes/test/bridgePythonParity.test.ts`

## 2. Runtime Packaging, Skills & CI Contract Audit

1. **Runtime dependency lockstep**: Verify `packaging/runtime/package.json` (`@opentui/core`, `@opentui/react`, `react`) and `packaging/runtime/bun.lock` exactly match `xerxes/package.json`.
2. **Bun-only & image catalog hygiene**: Run `bun run repo:check` (zero `.py`/`.pyc`/legacy Python metadata) and `bun run --cwd xerxes check:image-catalog` (if stale, regenerate via `bun run --cwd xerxes generate:image-catalog`).
3. **CI, Docker, Installer & Bundled Skill Contracts**:
   ```bash
   bun test xerxes/test/nativeWorkflowContract.test.ts xerxes/test/containerDistribution.test.ts xerxes/test/installerParity.test.ts xerxes/test/bundledSkillsLint.test.ts xerxes/test/bundledSkillAssets.test.ts
   ```
   Confirm all GitHub Actions in `.github/workflows/` remain pinned to 40-char SHAs, Bun remains `1.3.12`, and `xerxes/skills/` + `xerxes/src/agents/default/` contain zero symlinks, executable bits (`0o111`), `.py` files, or non-redistributable directories.

## 3. Full CI-Parity Verification Gate

```bash
bun run verify
bun run smoke
bun run docs:build
git diff --check
```

## 4. Stage, Pack, Check & Smoke the Release Tarball

Replace `<ver>` with the target semver (e.g., `0.8.56`) and use temp directories under `/tmp`:

```bash
bun run release:prepare -- --output /tmp/xerxes-release-pkg --expected-version <ver>
(cd /tmp/xerxes-release-pkg && bun pm pack --filename /tmp/xerxes-agents-<ver>.tgz --ignore-scripts && bun pm pack --dry-run --ignore-scripts)
bun run release:check -- --package /tmp/xerxes-release-pkg --archive /tmp/xerxes-agents-<ver>.tgz --expected-version <ver>
bun run release:smoke -- /tmp/xerxes-agents-<ver>.tgz
bun run release:publish-dry-run -- --package /tmp/xerxes-release-pkg --archive /tmp/xerxes-agents-<ver>.tgz --expected-version <ver>
```

Verify `release:prepare` writes `release-manifest.json` with SHA-256 digests and excludes blocked non-redistributable skills, `release:check` passes tar checksum + mode validation, and `release:smoke` installs cleanly into a temp project and outputs `--version` plus `xerxes-tui: no TTY`.

## 5. Optional Surface Packaging Checks (Run When Requested or Applicable)

- **Docker**: `XERXES_DAEMON_TOKEN=ci-compose-config-only docker compose config` and `docker build -t xerxes-release-check .` (verify `--version`, non-TTY TUI smoke, and `/workspace` write probe as `USER xerxes`).
- **VS Code VSIX**: `bun run build:vscode` (or `bun xerxes/scripts/buildVscode.ts --all` for all 5 platforms); confirm git tag `v<ver>` will match `xerxes/package.json` for `.github/workflows/vscode-marketplace.yml`.
- **macOS Desktop**: `bun run --cwd xerxes build:desktop` (set `XERXES_DESKTOP_PACKAGE_DIR` to avoid overwriting a running `.app`; inside-out signing and `{ verbatimSymlinks: true }` preserved).

## 6. Report Format

Present a concise readiness report with:
- Version-sync checklist (13/13 items: match vs drift).
- Verification gate results (`verify`, `smoke`, `docs:build`, `git diff --check`).
- Tarball pipeline results (`prepare`, `pack`, `check`, `smoke`, `publish-dry-run`).
- Blocking gaps before tagging/publishing.
