---
name: xerxes-release-and-packaging
description: Use when preparing a Xerxes release, bumping version strings, syncing runtime packaging, validating npm tarballs, Docker images, macOS Desktop .app/.dmg builds, or 5-target VS Code VSIX packages.
version: 1.0
author: Xerxes-Agents
tags: [xerxes, release, packaging, docker, desktop, vscode]
---

# Xerxes Release & Packaging Runbook

Checklist for version bumps, npm tarball staging, Docker checks, macOS `.app`/`.dmg` packaging, and 5-target VS Code VSIX builds. See `.agents/rules/testing-packaging-and-ci.md`.

> **Safety Gate**: Never stage, commit, tag, push, create a GitHub release, or publish to npm / VS Code Marketplace unless explicitly requested. Local builds and dry-runs are safe.

## When to use

- Bumping the release version across manifests, source, docs, and tests.
- Syncing `packaging/runtime/` or verifying tarballs, Docker, Desktop `.app`/`.dmg`, or VSIX packages.

## Step 1: Bump All 9 Version Sites + 4 Hardcoded Tests

`releasePackage.ts` validates only `package.json` and `xerxes/package.json`; update all 13 sites together:

**9 manifests, source & docs:**
1. `package.json` (`version`)
2. `xerxes/package.json` (`version`) — also checked by `vscode-marketplace.yml` and dynamic builders
3. `packaging/runtime/package.json` (`version` + `@opentui/core`, `@opentui/react`, `react`) — checked by `containerDistribution.test.ts`
4. `xerxes/src/core/version.ts` (`XERXES_VERSION`)
5. `xerxes/src/core/config.ts` (`ObservabilityConfig` `serviceVersion` default)
6. `xerxes/src/desktop/renderer/buildInfo.ts` (fallback version)
7. `xerxes/src/llms/anthropic.ts` (`User-Agent`)
8. `xerxes/src/llms/mistral.ts` (`User-Agent`)
9. `docs/changelog.md` (release heading)

**4 tests hardcoding the version (fail if unbumped):**
`xerxes/test/installerParity.test.ts`, `xerxes/test/daemonServer.test.ts`, `xerxes/test/desktopStore.test.ts`, `xerxes/test/bridgePythonParity.test.ts`.

## Step 2: Sync `packaging/runtime/` & CI Pins

- Whenever `@opentui/core`, `@opentui/react`, or `react` changes in `xerxes/package.json`, sync `packaging/runtime/package.json` and `packaging/runtime/bun.lock` (`containerDistribution.test.ts`, `nativeWorkflowContract.test.ts`).
- Keep pinned Bun (`1.3.12`) aligned across `Dockerfile`, `.github/workflows/{bun-ci,bun-release,vscode-marketplace}.yml`, and `xerxes/scripts/releasePackage.ts` (`engines.bun`).
- Keep all GitHub Actions pinned to 40-char SHAs + `# vN` comments (`nativeWorkflowContract.test.ts`).

## Step 3: Verify, Stage, Pack, Check, Smoke & Dry-Run Publish

```bash
bun run verify && bun run smoke && bun run docs:build && git diff --check

PKG_DIR=$(mktemp -d)/xerxes-bun-package
ART_DIR=$(mktemp -d)/xerxes-bun-artifacts
mkdir -p "$ART_DIR"
bun run release:prepare -- --output "$PKG_DIR" --expected-version <version>
(cd "$PKG_DIR" && bun pm pack --filename "$ART_DIR/xerxes-bun.tgz" --ignore-scripts && bun pm pack --dry-run --ignore-scripts)
bun run release:check -- --package "$PKG_DIR" --archive "$ART_DIR/xerxes-bun.tgz" --expected-version <version>
bun run release:smoke -- "$ART_DIR/xerxes-bun.tgz"
bun run release:publish-dry-run -- --package "$PKG_DIR" --archive "$ART_DIR/xerxes-bun.tgz" --expected-version <version>
```

`releasePackage.ts` verifies format-2 `release-manifest.json` SHA-256 hashes, `0o755` launchers (`bin/{xerxes,xerxes-agents,xerxes-acp}`), tar headers/checksums, and absence of non-redistributable skills (`powerpoint/*` except `SKILL.md`, `creative-ideation`, `research/research-paper-writing/templates`, `training/{axolotl,pytorch-fsdp,unsloth}`). `smokeReleasePackage.ts` installs via `bun add --exact`, asserts `--version`, imports `@opentui/*`, and expects `xerxes-tui: no TTY`.

## Step 4: Docker, macOS Desktop & 5-Target VS Code VSIX

```bash
# Container, workflow & installer contract tests
bun test xerxes/test/containerDistribution.test.ts xerxes/test/nativeWorkflowContract.test.ts xerxes/test/installerParity.test.ts

# Docker parity (when Docker is running)
XERXES_DAEMON_TOKEN=ci-compose-config-only docker compose config
docker build --build-arg XERXES_UID="$(id -u)" -t xerxes:ci .
docker run --rm xerxes:ci --version
docker run --rm --entrypoint bun xerxes:ci /app/xerxes/dist/ui/entry.js

# macOS Desktop .app / .dmg (signing:local preserves TCC grants across rebuilds)
bun run --cwd xerxes signing:local
bun run --cwd xerxes build:desktop      # set XERXES_DESKTOP_PACKAGE_DIR if app is running
bun run --cwd xerxes build:installer

# VS Code VSIX: current host or all 5 targets (darwin-{arm64,x64}, linux-{x64,arm64}, win32-x64)
bun run build:vscode
bun run build:runtime && bun xerxes/scripts/buildDesktop.ts && bun xerxes/scripts/buildVscode.ts --all
# Publish ONLY when explicitly asked (preloads xerxes/scripts/vsceRequestTimeout.ts for 30m timeout):
# bun run publish:vscode
```

## Pitfalls

- **macOS signing**: `packageDesktopMac.ts` and `buildDesktopInstaller.ts` MUST copy `.app` bundles with `{ verbatimSymlinks: true }`; sign `.dylib`s, helpers, frameworks, and bundled `bun` inside-out (never `--deep` on hardened runtime).
- **Omitted runtime extras**: `releasePackage.ts` and `Dockerfile` omit `dist/sandboxShim.ts`, `dist/build-id`, and `dist/mobile/`.
- **Installers & assets**: Root installers (`scripts/install.sh`, `scripts/install.ps1`) are checked by `installerParity.test.ts` and CI PowerShell AST checks; `copyBundledSkills.ts` rejects symlinks and executable (`0o111`) files.
