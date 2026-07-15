// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { existsSync } from "node:fs";
import { release as osRelease } from "node:os";
import { resolve } from "node:path";

const DEFAULT_HOMEPAGE = "https://github.com/erfanzar/Xerxes-Agents";
const DEFAULT_BUN_FORMULA = "oven-sh/bun/bun";
const DEFAULT_ENTRYPOINT = "xerxes/src/cli.ts";

/** Resolve the OpenTUI artifact in either a source checkout or staged release package. */
export function resolveTuiEntry(
  moduleDirectory: string,
  fileExists: (path: string) => boolean = existsSync,
): string | undefined {
  const candidates = [
    resolve(moduleDirectory, "../ui/entry.js"),
    resolve(moduleDirectory, "ui/entry.js"),
    resolve(moduleDirectory, "../dist/ui/entry.js"),
  ];
  return candidates.find(fileExists);
}

/** Snapshot of the Bun host used to select distribution guidance. */
export interface PlatformInfo {
  readonly bunVersion: string;
  readonly isDarwin: boolean;
  readonly isLinux: boolean;
  readonly isTermux: boolean;
  readonly isWindows: boolean;
  readonly isWsl: boolean;
  readonly platform: string;
  readonly release: string;
}

/** Injectable process values make platform detection deterministic in callers and tests. */
export interface DetectPlatformOptions {
  readonly bunVersion?: string;
  readonly environment?: Readonly<Record<string, string | undefined>>;
  readonly platform?: string;
  readonly release?: string;
}

/** Detect Bun's host platform, Termux, and WSL without mutating process state. */
export function detectPlatform(
  options: DetectPlatformOptions = {},
): PlatformInfo {
  const platform = options.platform ?? process.platform;
  const release = options.release ?? osRelease();
  const environment = options.environment ?? process.env;
  const normalizedRelease = release.toLowerCase();
  const prefix = environment.PREFIX ?? "";
  const isTermux =
    prefix.startsWith("/data/data/com.termux/") ||
    normalizedRelease.includes("termux") ||
    environment.TERMUX_VERSION !== undefined;
  const isWsl =
    normalizedRelease.includes("microsoft") ||
    environment.WSL_DISTRO_NAME !== undefined;

  return Object.freeze({
    bunVersion: options.bunVersion ?? Bun.version,
    isDarwin: platform === "darwin",
    isLinux: platform === "linux",
    isTermux,
    isWindows: platform === "win32",
    isWsl,
    platform,
    release,
  });
}

/** Packages with native/runtime assumptions that are not supported by the Termux Bun target. */
export const TERMUX_UNSUPPORTED_PACKAGES: ReadonlySet<string> = new Set([
  "@playwright/test",
  "onnxruntime-node",
  "playwright",
  "playwright-core",
]);

/** Return whether an npm package should be omitted from a Termux Bun install. */
export function isTermuxUnsupportedPackage(packageName: string): boolean {
  return TERMUX_UNSUPPORTED_PACKAGES.has(normalizePackageName(packageName));
}

/**
 * Copy a package.json-style dependency mapping while omitting known Termux-incompatible packages.
 *
 * This only filters dependency declarations; it does not make an arbitrary npm package Android-safe.
 */
export function filterTermuxDependencies(
  dependencies: Readonly<Record<string, string>>,
): Readonly<Record<string, string>> {
  const filtered: Record<string, string> = {};
  for (const [packageName, version] of Object.entries(dependencies)) {
    if (!isTermuxUnsupportedPackage(packageName)) {
      filtered[packageName] = version;
    }
  }
  return Object.freeze(filtered);
}

/** Inputs needed to render a Bun-oriented Homebrew formula template. */
export interface HomebrewFormulaOptions {
  readonly bunFormula?: string;
  readonly entrypoint?: string;
  readonly homepage?: string;
  readonly sha256: string;
  readonly tarballUrl: string;
  readonly version: string;
}

/**
 * A source formula template for a Bun release artifact.
 *
 * It intentionally does not assert that a package has been published or that a tap has accepted it.
 */
export const HOMEBREW_BUN_FORMULA_TEMPLATE = `class XerxesAgent < Formula
  desc "Multi-agent orchestration framework + terminal coding agent"
  homepage {homepage}
  url {tarball_url}
  version {version}
  sha256 {sha256}
  license "Apache-2.0"

  depends_on {bun_formula}

  def install
    libexec.install Dir["*"]
    cd libexec do
      system Formula[{bun_formula}].opt_bin/"bun", "install", "--production", "--frozen-lockfile"
    end
    (bin/"xerxes").write <<~EOS
      #!/bin/bash
      exec "#{Formula[{bun_formula}].opt_bin}/bun" "#{libexec}/{entrypoint}" "$@"
    EOS
    chmod 0755, bin/"xerxes"
  end

  test do
    system bin/"xerxes", "--help"
  end
end
`;

/** Render a reviewable Homebrew formula for a Bun source tarball. */
export function renderHomebrewFormula(options: HomebrewFormulaOptions): string {
  const homepage = options.homepage ?? DEFAULT_HOMEPAGE;
  const bunFormula = options.bunFormula ?? DEFAULT_BUN_FORMULA;
  const entrypoint = options.entrypoint ?? DEFAULT_ENTRYPOINT;
  return HOMEBREW_BUN_FORMULA_TEMPLATE.replaceAll(
    "{homepage}",
    rubyString(homepage, "homepage"),
  )
    .replaceAll("{tarball_url}", rubyString(options.tarballUrl, "tarballUrl"))
    .replaceAll("{version}", rubyString(options.version, "version"))
    .replaceAll("{sha256}", rubyString(options.sha256, "sha256"))
    .replaceAll("{bun_formula}", rubyString(bunFormula, "bunFormula"))
    .replaceAll("{entrypoint}", formulaPath(entrypoint, "entrypoint"));
}

/**
 * A Bun install template that deliberately requires a release-provided package or source spec.
 * It does not imply that Xerxes is already published to an npm registry.
 */
export const BUN_SHELL_INSTALL_SNIPPET = `#!/usr/bin/env bash
set -euo pipefail
if ! command -v bun >/dev/null 2>&1; then
  echo "Installing Bun..."
  curl -fsSL https://bun.sh/install | bash
  export BUN_INSTALL="\${BUN_INSTALL:-$HOME/.bun}"
  export PATH="$BUN_INSTALL/bin:$PATH"
fi
: "\${XERXES_PACKAGE:?Set XERXES_PACKAGE to a release-approved npm package or source spec.}"
bun add --global "$XERXES_PACKAGE"
echo "Xerxes installed through Bun. Restart your shell, then run 'xerxes'."
`;

/** PowerShell counterpart to {@link BUN_SHELL_INSTALL_SNIPPET}. */
export const BUN_POWERSHELL_INSTALL_SNIPPET = `$ErrorActionPreference = 'Stop'
if (-not (Get-Command bun -ErrorAction SilentlyContinue)) {
  Write-Host "Installing Bun..."
  powershell -c "irm bun.sh/install.ps1|iex"
}
if (-not $env:XERXES_PACKAGE) {
  throw "Set XERXES_PACKAGE to a release-approved npm package or source spec."
}
bun add --global $env:XERXES_PACKAGE
Write-Host "Xerxes installed through Bun."
`;

function normalizePackageName(packageName: string): string {
  return packageName.trim().toLowerCase().replaceAll("_", "-");
}

function rubyString(value: string, field: string): string {
  if (!value.trim()) {
    throw new TypeError(`${field} must not be empty`);
  }
  if (value.includes("\r") || value.includes("\n")) {
    throw new TypeError(`${field} must not contain a newline`);
  }
  return JSON.stringify(value);
}

function formulaPath(value: string, field: string): string {
  if (!value.trim()) {
    throw new TypeError(`${field} must not be empty`);
  }
  if (
    value.startsWith("/") ||
    value.includes("\\") ||
    value.includes("\r") ||
    value.includes("\n") ||
    value.split("/").some((segment) => segment === "..")
  ) {
    throw new TypeError(`${field} must be a relative, newline-free path`);
  }
  return value;
}
