// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from "bun:test";

import {
  exitShellInput,
  interruptTerminalInput,
  resolveDefaultShell,
  shellInvocation,
} from "../src/core/shell.js";

test("posix shell resolution keeps the $SHELL then /bin/sh chain", () => {
  expect(resolveDefaultShell({ SHELL: "/bin/zsh" }, "linux")).toBe("/bin/zsh");
  expect(resolveDefaultShell({}, "darwin")).toBe("/bin/sh");
  expect(resolveDefaultShell({ SHELL: "  " }, "linux")).toBe("/bin/sh");
});

test("win32 shell resolution honors SHELL and COMSPEC before probing", () => {
  expect(resolveDefaultShell({ SHELL: "C:\\tools\\bash.exe" }, "win32")).toBe("C:\\tools\\bash.exe");
  expect(resolveDefaultShell({ COMSPEC: "C:\\WINDOWS\\system32\\cmd.exe" }, "win32")).toBe(
    "C:\\WINDOWS\\system32\\cmd.exe",
  );
});

test("win32 shell resolution prefers pwsh, then powershell, then cmd.exe", () => {
  const pwsh = resolveDefaultShell({}, "win32", (name) => (name === "pwsh" ? "C:\\pwsh\\pwsh.exe" : null));
  expect(pwsh).toBe("C:\\pwsh\\pwsh.exe");
  const powershell = resolveDefaultShell({}, "win32", (name) =>
    name === "powershell" ? "C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" : null,
  );
  expect(powershell).toContain("powershell.exe");
  expect(resolveDefaultShell({}, "win32", () => null)).toBe("cmd.exe");
});

test("posix shell invocation keeps -l and -c semantics", () => {
  expect(shellInvocation("/bin/bash", "ls", true, "linux")).toEqual(["/bin/bash", "-l", "-c", "ls"]);
  expect(shellInvocation("/bin/bash", "ls", false, "linux")).toEqual(["/bin/bash", "-c", "ls"]);
  expect(shellInvocation("/bin/sh", "ls", true, "linux")).toEqual(["/bin/sh", "-c", "ls"]);
  expect(shellInvocation("/bin/bash", "", true, "linux")).toEqual(["/bin/bash", "-l"]);
});

test("win32 cmd.exe invocation uses /d /s /c and never POSIX flags", () => {
  expect(shellInvocation("cmd.exe", "dir", true, "win32")).toEqual(["cmd.exe", "/d", "/s", "/c", "dir"]);
  expect(shellInvocation("cmd.exe", "", true, "win32")).toEqual(["cmd.exe"]);
});

test("win32 powershell invocation uses -NoLogo -NoProfile -Command", () => {
  for (const shell of ["powershell.exe", "C:\\Pwsh\\pwsh.exe", "pwsh"]) {
    expect(shellInvocation(shell, "Get-ChildItem", true, "win32")).toEqual([
      shell,
      "-NoLogo",
      "-NoProfile",
      "-Command",
      "Get-ChildItem",
    ]);
  }
});

test("terminal control input is platform specific", () => {
  expect(interruptTerminalInput("win32")).toBe("\u0003");
  expect(interruptTerminalInput("linux")).toBe("");
  expect(exitShellInput("win32")).toBe("exit\r");
  expect(exitShellInput("linux")).toBe("\u0004");
});
