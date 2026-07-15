// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from "bun:test";
import { mkdir, mkdtemp, realpath, rm, symlink } from "node:fs/promises";
import { dirname, join } from "node:path";
import { tmpdir } from "node:os";

import {
  extractAtMentionedFiles,
  MAX_ATTACHMENT_BYTES,
  processAtMentions,
} from "../src/daemon/atMentions.js";

test("file mentions parse quoted paths and line ranges without matching email addresses", () => {
  expect(
    extractAtMentionedFiles(
      'mail a@b.test then @src/main.ts#L2-L4 and @"docs/my file.md"',
    ),
  ).toEqual([
    {
      filePath: "docs/my file.md",
      raw: "docs/my file.md",
    },
    {
      endLine: 4,
      filePath: "src/main.ts",
      raw: "src/main.ts#L2-L4",
      startLine: 2,
    },
  ]);
});

test("file mentions attach numbered workspace text and reject unsafe or binary paths", async () => {
  const root = await mkdtemp(join(tmpdir(), "xerxes-at-mentions-"));
  const outside = await mkdtemp(join(tmpdir(), "xerxes-at-outside-"));
  try {
    await writeFixture(root, "src/main.ts", "one\ntwo\nthree\nfour");
    await writeFixture(root, "docs/my file.md", "space path");
    await writeFixture(outside, "secret.txt", "must not attach");
    await symlink(join(outside, "secret.txt"), join(root, "escape.txt"));
    await Bun.write(join(root, "binary.dat"), new Uint8Array([0, 1, 2, 3]));
    const canonicalRoot = await realpath(root);

    const authored =
      'review @src/main.ts#L2-L3 and @"docs/my file.md"; ignore @escape.txt @binary.dat';
    const processed = await processAtMentions(authored, root);

    expect(processed.enhancedMessage).toContain("<attached_files>");
    expect(processed.enhancedMessage).toContain("2 | two\n3 | three");
    expect(processed.enhancedMessage).toContain("1 | space path");
    expect(processed.enhancedMessage).not.toContain("must not attach");
    expect(processed.enhancedMessage).not.toContain("binary.dat: lines");
    expect(processed.enhancedMessage.endsWith(authored)).toBeTrue();
    expect(processed.mentionedFiles).toEqual([
      join(canonicalRoot, "docs/my file.md"),
      join(canonicalRoot, "src/main.ts"),
    ]);
  } finally {
    await rm(root, { recursive: true, force: true });
    await rm(outside, { recursive: true, force: true });
  }
});

test("file mention payloads use one bounded 100 KiB attachment budget", async () => {
  const root = await mkdtemp(join(tmpdir(), "xerxes-at-budget-"));
  try {
    await writeFixture(root, "first.txt", "a".repeat(MAX_ATTACHMENT_BYTES * 2));
    await writeFixture(root, "second.txt", "b".repeat(MAX_ATTACHMENT_BYTES * 2));

    const authored = "review @first.txt and @second.txt";
    const processed = await processAtMentions(authored, root);
    const attachment = processed.enhancedMessage.slice(
      0,
      processed.enhancedMessage.lastIndexOf(`\n\n${authored}`),
    );

    expect(new TextEncoder().encode(attachment).byteLength).toBeLessThanOrEqual(
      MAX_ATTACHMENT_BYTES,
    );
    expect(attachment).toContain("truncated");
    expect(processed.mentionedFiles.length).toBeLessThanOrEqual(2);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

async function writeFixture(
  root: string,
  relativePath: string,
  content: string,
): Promise<void> {
  const path = join(root, ...relativePath.split("/"));
  await mkdir(dirname(path), { recursive: true });
  await Bun.write(path, content);
}
