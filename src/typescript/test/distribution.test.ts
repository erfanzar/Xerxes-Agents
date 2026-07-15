// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from "bun:test";
import { resolve } from "node:path";

import {
  BUN_POWERSHELL_INSTALL_SNIPPET,
  BUN_SHELL_INSTALL_SNIPPET,
  TERMUX_UNSUPPORTED_PACKAGES,
  detectPlatform,
  filterTermuxDependencies,
  isTermuxUnsupportedPackage,
  renderHomebrewFormula,
  resolveTuiEntry,
} from "../src/runtime/distribution.js";

test("TUI entry resolution supports staged packages and source workspaces", () => {
  const packageBin = "/tmp/xerxes-release/bin";
  const packagedEntry = resolve(packageBin, "../ui/entry.js");
  const workspaceModule = "/workspace/src/typescript/dist";
  const workspaceEntry = resolve(workspaceModule, "ui/entry.js");
  const sourceModule = "/workspace/src/typescript/src";
  const sourceEntry = resolve(sourceModule, "../dist/ui/entry.js");

  expect(resolveTuiEntry(packageBin, (path) => path === packagedEntry)).toBe(
    packagedEntry,
  );
  expect(
    resolveTuiEntry(workspaceModule, (path) => path === workspaceEntry),
  ).toBe(workspaceEntry);
  expect(resolveTuiEntry(sourceModule, (path) => path === sourceEntry)).toBe(
    sourceEntry,
  );
  expect(resolveTuiEntry(packageBin, () => false)).toBeUndefined();
});

test("Bun platform detection identifies Termux and WSL from injected host values", () => {
  const termux = detectPlatform({
    bunVersion: "1.3.12",
    environment: { PREFIX: "/data/data/com.termux/files/usr" },
    platform: "linux",
    release: "6.1.0-android",
  });
  expect(termux).toEqual({
    bunVersion: "1.3.12",
    isDarwin: false,
    isLinux: true,
    isTermux: true,
    isWindows: false,
    isWsl: false,
    platform: "linux",
    release: "6.1.0-android",
  });
  expect(Object.isFrozen(termux)).toBe(true);

  const wsl = detectPlatform({
    bunVersion: "1.3.12",
    environment: { WSL_DISTRO_NAME: "Ubuntu-24.04" },
    platform: "linux",
    release: "6.6.87.2-microsoft-standard-WSL2",
  });
  expect(wsl).toMatchObject({ isLinux: true, isTermux: false, isWsl: true });

  const windows = detectPlatform({
    bunVersion: "1.3.12",
    platform: "win32",
    release: "10.0.26100",
  });
  expect(windows).toMatchObject({
    isDarwin: false,
    isLinux: false,
    isWindows: true,
  });
});

test("Termux filtering copies npm dependency maps and only removes declared unsupported packages", () => {
  const dependencies = {
    "@playwright/test": "^1.52.0",
    fastify: "^5.3.0",
    "onnxruntime-node": "^1.22.0",
    playwright: "^1.52.0",
  };
  const filtered = filterTermuxDependencies(dependencies);

  expect(filtered).toEqual({ fastify: "^5.3.0" });
  expect(dependencies).toHaveProperty("playwright", "^1.52.0");
  expect(isTermuxUnsupportedPackage("PLAYWRIGHT_CORE")).toBe(true);
  expect(isTermuxUnsupportedPackage("fastify")).toBe(false);
  expect(TERMUX_UNSUPPORTED_PACKAGES).toContain("@playwright/test");
  expect(Object.isFrozen(filtered)).toBe(true);
});

test("the Homebrew renderer emits a Bun source formula without Python packaging claims", () => {
  const formula = renderHomebrewFormula({
    sha256: "deadbeef",
    tarballUrl: "https://example.test/xerxes-agent-0.3.0.tar.gz",
    version: "0.3.0",
  });

  expect(formula).toContain("class XerxesAgent < Formula");
  expect(formula).toContain('version "0.3.0"');
  expect(formula).toContain('sha256 "deadbeef"');
  expect(formula).toContain('depends_on "oven-sh/bun/bun"');
  expect(formula).toContain('"install", "--production", "--frozen-lockfile"');
  expect(formula).toContain("src/typescript/src/cli.ts");
  expect(formula).toContain('system bin/"xerxes", "--help"');
  expect(formula).not.toContain("Language::Python");
  expect(formula).not.toContain("python@3.11");

  expect(() =>
    renderHomebrewFormula({
      sha256: "a",
      tarballUrl: "https://example.test/archive\nmalicious",
      version: "0.3.0",
    }),
  ).toThrow("tarballUrl must not contain a newline");
  expect(() =>
    renderHomebrewFormula({
      entrypoint: "../cli.ts",
      sha256: "a",
      tarballUrl: "https://example.test/archive",
      version: "0.3.0",
    }),
  ).toThrow("entrypoint must be a relative");
});

test("Bun install snippets require a release-approved spec instead of asserting a registry publication", () => {
  expect(BUN_SHELL_INSTALL_SNIPPET).toContain("https://bun.sh/install");
  expect(BUN_SHELL_INSTALL_SNIPPET).toContain("XERXES_PACKAGE");
  expect(BUN_SHELL_INSTALL_SNIPPET).toContain(
    'bun add --global "$XERXES_PACKAGE"',
  );
  expect(BUN_SHELL_INSTALL_SNIPPET).not.toContain("uv pip install");

  expect(BUN_POWERSHELL_INSTALL_SNIPPET).toContain("bun.sh/install.ps1");
  expect(BUN_POWERSHELL_INSTALL_SNIPPET).toContain("XERXES_PACKAGE");
  expect(BUN_POWERSHELL_INSTALL_SNIPPET).toContain(
    "bun add --global $env:XERXES_PACKAGE",
  );
  expect(BUN_POWERSHELL_INSTALL_SNIPPET).not.toContain("uv tool install");
});
