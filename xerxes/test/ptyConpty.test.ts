// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from "bun:test";

import { PtySessionManager } from "../src/operators/pty.js";

// ConPTY sessions only exist on native Windows; Bun.Terminal covers POSIX.
const testWindows = test.skipIf(process.platform !== "win32");

testWindows("ConPTY session runs cmd.exe and captures output plus exit code", async () => {
  const manager = new PtySessionManager();
  try {
    const output = await manager.createSession("echo xerxes-conpty-ok", { yieldTimeMs: 5_000 });
    expect(output.stdout).toContain("xerxes-conpty-ok");
    expect(output.exitCode).toBe(0);
    expect(output.running).toBe(false);
  } finally {
    await manager.closeAll();
  }
});

testWindows("ConPTY interactive session accepts input and closes cleanly", async () => {
  const manager = new PtySessionManager();
  try {
    const opened = await manager.createSession("", { yieldTimeMs: 5_000, shell: "cmd.exe" });
    expect(opened.running).toBe(true);
    const sessionId = opened.sessionId;

    const echoed = await manager.write(sessionId, { chars: "echo interactive-ok\r", yieldTimeMs: 5_000 });
    expect(echoed.stdout).toContain("interactive-ok");

    const listed = manager.listSessions().map((session) => session.sessionId);
    expect(listed).toContain(sessionId);

    const closed = await manager.close(sessionId);
    expect(closed.closed).toBe(true);
    expect(manager.listSessions()).toHaveLength(0);
  } finally {
    await manager.closeAll();
  }
});

testWindows("ConPTY session terminates a running command via sendEof", async () => {
  const manager = new PtySessionManager();
  try {
    const opened = await manager.createSession("", { yieldTimeMs: 5_000, shell: "cmd.exe" });
    expect(opened.running).toBe(true);
    // `exit\r` ends cmd.exe the way Ctrl-D ends a POSIX shell.
    const output = await manager.write(opened.sessionId, {
      closeStdin: true,
      yieldTimeMs: 5_000,
    });
    expect(output.exitCode).not.toBeNull();
  } finally {
    await manager.closeAll();
  }
});
