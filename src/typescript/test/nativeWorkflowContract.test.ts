// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, test } from "bun:test";
import { readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const repositoryRoot = resolve(
  dirname(fileURLToPath(import.meta.url)),
  "../../..",
);
const actionPins: Readonly<
  Record<string, Readonly<{ sha: string; version: string }>>
> = Object.freeze({
  "actions/cache": Object.freeze({
    sha: "0057852bfaa89a56745cba8c7296529d2fc39830",
    version: "v4",
  }),
  "actions/checkout": Object.freeze({
    sha: "34e114876b0b11c390a56381ad16ebd13914f8d5",
    version: "v4",
  }),
  "actions/download-artifact": Object.freeze({
    sha: "d3f86a106a0bac45b974a628896c90dbdf5c8093",
    version: "v4",
  }),
  "actions/upload-artifact": Object.freeze({
    sha: "ea165f8d65b6e75b540449e92b4886f43607fa02",
    version: "v4",
  }),
  "oven-sh/setup-bun": Object.freeze({
    sha: "0c5077e51419868618aeaa5fe8019c62421857d6",
    version: "v2",
  }),
  "softprops/action-gh-release": Object.freeze({
    sha: "3bb12739c298aeb8a4eeaf626c5b8d85266b0e65",
    version: "v2",
  }),
});

describe("native Bun workflow contracts", () => {
  test("Bun CI validates the root native lifecycle without Python commands", async () => {
    const workflow = await readWorkflow("bun-ci.yml");

    expect(() => Bun.YAML.parse(workflow)).not.toThrow();
    expectPinnedActions(workflow);
    expect(workflow).toContain(pinnedUse("oven-sh/setup-bun"));
    expect(workflow).toContain("bun install --frozen-lockfile");
    expect(workflow).toContain("bun run verify");
    expect(workflow).toContain("bun pm pack");
    expect(workflow).toContain(pinnedUse("actions/upload-artifact"));
    expect(workflow).not.toMatch(/\b(?:python|pip|pypi|uv)\b/iu);
  });

  test("Bun release stages a private artifact with write permission isolated to attachment", async () => {
    const workflow = await readWorkflow("bun-release.yml");
    const parsed = record(Bun.YAML.parse(workflow), "release workflow");
    const jobs = record(parsed.jobs, "release jobs");
    const packageJob = record(jobs.package, "release package job");
    const attachJob = record(
      jobs["attach-github-release"],
      "release attachment job",
    );

    expect(() => Bun.YAML.parse(workflow)).not.toThrow();
    expect(record(parsed.permissions, "workflow permissions").contents).toBe(
      "read",
    );
    expect(record(packageJob.permissions, "package permissions").contents).toBe(
      "read",
    );
    expect(
      record(attachJob.permissions, "attachment permissions").contents,
    ).toBe("write");
    expect(workflow.match(/contents:\s+write/gu)).toHaveLength(1);
    expectPinnedActions(workflow);
    expect(workflow).toContain("bun run release:prepare");
    expect(workflow).toContain("bun run release:check");
    expect(workflow).toContain("bun pm pack");
    expect(workflow).toContain(pinnedUse("actions/upload-artifact"));
    expect(workflow).toContain(pinnedUse("softprops/action-gh-release"));
    expect(workflow).toContain("inputs.publish == true");
    expect(workflow).not.toMatch(/\b(?:python|pip|pypi|npm publish)\b/iu);
  });
});

function expectPinnedActions(workflow: string): void {
  const uses = [
    ...workflow.matchAll(
      /^\s*(?:-\s*)?uses:\s+([^\s#]+)(?:\s+#\s+(\S+))?\s*$/gmu,
    ),
  ];

  expect(uses.length).toBeGreaterThan(0);
  for (const match of uses) {
    const reference = match[1];
    const version = match[2];
    expect(reference).toMatch(/^[^@\s]+@[0-9a-f]{40}$/u);

    const separator = reference?.lastIndexOf("@") ?? -1;
    const action = reference?.slice(0, separator);
    const expected = action === undefined ? undefined : actionPins[action];
    expect(expected).toBeDefined();
    if (expected === undefined || action === undefined) continue;

    expect(reference).toBe(`${action}@${expected.sha}`);
    expect(version).toBe(expected.version);
  }
}

function pinnedUse(action: string): string {
  const pin = actionPins[action];
  if (pin === undefined)
    throw new Error(`Missing workflow action pin for ${action}`);
  return `${action}@${pin.sha} # ${pin.version}`;
}

function record(value: unknown, label: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error(`${label} must be an object`);
  }
  return value as Record<string, unknown>;
}

async function readWorkflow(name: string): Promise<string> {
  return readFile(joinWorkflowPath(name), "utf8");
}

function joinWorkflowPath(name: string): string {
  return resolve(repositoryRoot, ".github/workflows", name);
}
