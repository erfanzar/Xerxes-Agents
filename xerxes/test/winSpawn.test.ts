// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from "bun:test";

import { resolveWindowsSpawn } from "../src/security/winSpawn.js";

const WHICH_NPX_CMD = (name: string) => (name === "npx" ? "C:\\Program Files\\nodejs\\npx.cmd" : null);

test("posix hosts spawn the exact argv unchanged", () => {
  const resolved = resolveWindowsSpawn("npx", ["-y", "server"], { platform: "linux", which: WHICH_NPX_CMD });
  expect(resolved).toEqual({ command: "npx", args: ["-y", "server"], wrapped: false });
});

test("win32 wraps an explicit .cmd shim in cmd.exe /d /s /c", () => {
  const resolved = resolveWindowsSpawn("npx.cmd", ["-y", "@modelcontextprotocol/server-filesystem"], {
    platform: "win32",
  });
  expect(resolved.wrapped).toBe(true);
  expect(resolved.command).toBe("cmd.exe");
  expect(resolved.args.slice(0, 3)).toEqual(["/d", "/s", "/c"]);
  expect(resolved.args[3]).toBe('"npx.cmd -y @modelcontextprotocol/server-filesystem"');
});

test("win32 resolves a bare name to its .cmd shim through which", () => {
  const resolved = resolveWindowsSpawn("npx", ["-y", "server"], {
    platform: "win32",
    which: WHICH_NPX_CMD,
  });
  expect(resolved.wrapped).toBe(true);
  expect(resolved.args[3]).toBe('""C:\\Program Files\\nodejs\\npx.cmd" -y server"');
});

test("win32 quoting protects arguments with spaces and cmd metacharacters", () => {
  const resolved = resolveWindowsSpawn("tool.cmd", ["two words", "a&b", 'say "hi"'], {
    platform: "win32",
  });
  expect(resolved.args[3]).toBe('"tool.cmd "two words" "a&b" "say \\"hi\\"""');
});

test("win32 passes real executables and paths through unchanged", () => {
  for (const command of ["git.exe", "C:\\tools\\git", "node", "./runner"]) {
    const resolved = resolveWindowsSpawn(command, ["--version"], {
      platform: "win32",
      which: () => null,
    });
    expect(resolved).toEqual({ command, args: ["--version"], wrapped: false });
  }
  // A bare name that resolves to a real .exe is left untouched as well.
  const resolved = resolveWindowsSpawn("node", ["--version"], {
    platform: "win32",
    which: () => "C:\\Program Files\\nodejs\\node.exe",
  });
  expect(resolved.wrapped).toBe(false);
});

test("win32 unknown bare names fall back to the raw argv", () => {
  const resolved = resolveWindowsSpawn("missing", [], { platform: "win32", which: () => null });
  expect(resolved).toEqual({ command: "missing", args: [], wrapped: false });
});
