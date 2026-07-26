// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from "bun:test";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { applyDiff, createUnifiedDiff, findAndReplace } from "../src/tools/codingTools.js";
import { WorkspacePathResolver } from "../src/tools/pathSafety.js";

async function inWorkspace(run: (workspace: string, paths: WorkspacePathResolver) => Promise<void>): Promise<void> {
  const workspace = await mkdtemp(join(tmpdir(), "xerxes-crlf-tools-"));
  try {
    await run(workspace, new WorkspacePathResolver(workspace));
  } finally {
    await rm(workspace, { force: true, recursive: true });
  }
}

test("applyDiff matches LF diff context against a CRLF original and preserves CRLF", () => {
  const original = "alpha\r\nbeta\r\ngamma\r\n";
  const diff = createUnifiedDiff("alpha\nbeta\ngamma\n", "alpha\nBETA\ngamma\n");
  const updated = applyDiff({ original, diff });
  expect(updated).toBe("alpha\r\nBETA\r\ngamma\r\n");
});

test("applyDiff tolerates a CRLF diff against an LF original", () => {
  const original = "one\ntwo\n";
  const diff = createUnifiedDiff("one\ntwo\n", "one\nTWO\n").replaceAll("\n", "\r\n");
  const updated = applyDiff({ original, diff });
  expect(updated).toBe("one\nTWO\n");
});

test("find_and_replace retries an LF search against a CRLF file and keeps CRLF", async () => {
  await inWorkspace(async (workspace, paths) => {
    await Bun.write(join(workspace, "target.txt"), "alpha\r\nbeta\r\ngamma\r\n");
    const result = await findAndReplace({
      backup: false,
      file_path: "target.txt",
      replace: "delta\nepsilon",
      search: "beta\ngamma",
    }, paths);
    expect(result).toContain("Replaced 1");
    expect(await Bun.file(join(workspace, "target.txt")).text()).toBe("alpha\r\ndelta\r\nepsilon\r\n");
  });
});

test("find_and_replace case-insensitive mode is CRLF tolerant as well", async () => {
  await inWorkspace(async (workspace, paths) => {
    await Bun.write(join(workspace, "target.txt"), "one\r\nTWO\r\nthree\r\n");
    const result = await findAndReplace({
      backup: false,
      case_sensitive: false,
      file_path: "target.txt",
      replace: "two\nthree",
      search: "TWO\nthree",
    }, paths);
    expect(result).toContain("Replaced 1");
    expect(await Bun.file(join(workspace, "target.txt")).text()).toBe("one\r\ntwo\r\nthree\r\n");
  });
});

test("find_and_replace reports zero replacements when neither EOL variant matches", async () => {
  await inWorkspace(async (workspace, paths) => {
    await Bun.write(join(workspace, "target.txt"), "alpha\r\nbeta\r\n");
    const result = await findAndReplace({
      backup: false,
      file_path: "target.txt",
      replace: "x",
      search: "missing\ntext",
    }, paths);
    expect(result).toContain("Replaced 0");
    expect(await Bun.file(join(workspace, "target.txt")).text()).toBe("alpha\r\nbeta\r\n");
  });
});
