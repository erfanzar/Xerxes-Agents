// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from "bun:test";
import { resolve } from "node:path";

const PROJECT_ROOT = resolve(import.meta.dir, "../..");
const INSTALLER_PATH = resolve(PROJECT_ROOT, "scripts", "install.ps1");

test("PowerShell installer mirrors the POSIX installer contract", async () => {
  // Windows PowerShell 5.1 reads BOM-less scripts as ANSI; the UTF-8 BOM is
  // what keeps the ✓ markers (and the whole file) parsing correctly there.
  const raw = await Bun.file(INSTALLER_PATH).arrayBuffer();
  expect([...new Uint8Array(raw, 0, 3)]).toEqual([0xef, 0xbb, 0xbf]);

  const source = await Bun.file(INSTALLER_PATH).text();

  // Same managed PATH block markers as the POSIX installer.
  expect(source).toContain("# >>> xerxes PATH >>>");
  expect(source).toContain("# <<< xerxes PATH <<<");
  // Same build and launcher contract.
  expect(source).toContain("bun install --frozen-lockfile");
  expect(source).toContain("bun run build");
  expect(source).toContain("xerxes\\dist\\cli.js");
  expect(source).toContain("xerxes-acp");
  expect(source).toContain("XERXES_BIN_DIRECTORY");
  expect(source).toContain("XERXES_INSTALL_DIRECTORY");
  expect(source).toContain("XERXES_SOURCE_DIRECTORY");
  // Windows-native mechanics instead of POSIX-isms.
  expect(source).toContain("SetEnvironmentVariable");
  expect(source).not.toContain("/usr/bin/env");
  expect(source).not.toContain("chmod");
  expect(source).not.toContain("/tmp/");
});

test("PowerShell installer parses without syntax errors when PowerShell is available", async () => {
  const powershell = Bun.which("powershell") ?? Bun.which("pwsh");
  if (!powershell) {
    // No PowerShell on this host (e.g. minimal Linux CI): the static parity
    // assertions above still run; the parser check is best-effort.
    return;
  }
  const script = [
    "$errs = $null",
    `[System.Management.Automation.Language.Parser]::ParseFile('${INSTALLER_PATH.replaceAll("'", "''")}', [ref]$null, [ref]$errs) | Out-Null`,
    "if ($errs) { $errs | ForEach-Object { Write-Error $_.Message }; exit 1 }",
  ].join("; ");
  const process = Bun.spawn([powershell, "-NoProfile", "-NonInteractive", "-Command", script], {
    stderr: "pipe",
    stdout: "pipe",
  });
  const [exitCode, stderr] = await Promise.all([
    process.exited,
    new Response(process.stderr).text(),
  ]);
  expect(stderr.trim()).toBe("");
  expect(exitCode).toBe(0);
});
