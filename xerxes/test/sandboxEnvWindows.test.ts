// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from "bun:test";

import { safeParentEnvironmentNames } from "../src/security/subprocessSandbox.js";

test("win32 sandbox children receive the Windows-critical environment names", () => {
  const names = safeParentEnvironmentNames("win32");
  for (const required of [
    "SystemRoot",
    "SystemDrive",
    "COMSPEC",
    "PATHEXT",
    "TEMP",
    "TMP",
    "USERPROFILE",
    "APPDATA",
    "LOCALAPPDATA",
    "ProgramFiles",
    "ProgramFiles(x86)",
    "OS",
  ]) {
    expect(names, `missing ${required}`).toContain(required);
  }
  // The POSIX baseline names are retained on Windows too.
  for (const base of ["PATH", "HOME", "LANG", "LC_ALL", "TERM"]) {
    expect(names).toContain(base);
  }
});

test("posix sandbox environment stays minimal and unchanged", () => {
  expect(safeParentEnvironmentNames("linux")).toEqual(["PATH", "HOME", "LANG", "LC_ALL", "TERM"]);
  expect(safeParentEnvironmentNames("darwin")).toEqual(["PATH", "HOME", "LANG", "LC_ALL", "TERM"]);
});
