// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from "bun:test";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";

import {
  BUNDLED_SKILLS_DIRECTORY,
  resolveBundledSkillsDirectory,
  SkillRegistry,
  defaultSkillDiscoveryDirectories,
  parseSkillMarkdown,
  skillMatchesPlatform,
  skillPromptSection,
} from "../src/extensions/skills.js";
import { DEFAULT_OFFICIAL_SKILLS_DIRECTORY } from "../src/extensions/skillsHub.js";

test("skill parser handles frontmatter, inferred subcommands, and prompt rendering", async () => {
  const root = await mkdtemp(join(tmpdir(), "xerxes-skill-"));
  const skillDirectory = join(root, "review");
  try {
    await mkdir(join(skillDirectory, "references"), { recursive: true });
    await writeFile(
      join(skillDirectory, "references", "security-workflow.md"),
      "# workflow",
      "utf8",
    );
    const skill = parseSkillMarkdown(
      `---
name: review
description: Review a pull request
tags: [code, quality]
resources: [references]
dependencies:
  - git
---
Inspect the diff.`,
      join(skillDirectory, "SKILL.md"),
    );
    expect(skill.metadata).toMatchObject({
      name: "review",
      tags: ["code", "quality"],
      dependencies: ["git"],
      subcommands: ["security"],
    });
    expect(skillPromptSection(skill)).toContain("## Skill: review");
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("discovery retains higher-priority duplicate skills and validates dependencies", async () => {
  const root = await mkdtemp(join(tmpdir(), "xerxes-skills-"));
  const primary = join(root, "primary");
  const fallback = join(root, "fallback");
  try {
    await mkdir(join(primary, "same"), { recursive: true });
    await mkdir(join(fallback, "same"), { recursive: true });
    await writeFile(
      join(primary, "same", "SKILL.md"),
      "---\nname: same\ndescription: primary\n---\nPrimary",
      "utf8",
    );
    await writeFile(
      join(fallback, "same", "SKILL.md"),
      "---\nname: same\ndescription: fallback\n---\nFallback",
      "utf8",
    );
    const registry = new SkillRegistry();
    expect(await registry.discover(primary, fallback)).toEqual(["same"]);
    expect(registry.get("same")?.metadata.description).toBe("primary");
    expect(registry.validateDependencies({ hasTool: () => true })).toEqual([]);
    expect(skillMatchesPlatform(registry.get("same")!, process.platform)).toBe(
      true,
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("discovery rejects generated transcript wrappers and falls through to a healthy bundled skill", async () => {
  const root = await mkdtemp(join(tmpdir(), "xerxes-skill-transcript-"));
  const userRoot = join(root, "user");
  const bundledRoot = join(root, "bundled");
  try {
    await mkdir(join(userRoot, "deepscan"), { recursive: true });
    await mkdir(join(bundledRoot, "deepscan"), { recursive: true });
    await writeFile(
      join(userRoot, "deepscan", "SKILL.md"),
      "<think>Draft a deepscan skill.</think>\n\n```yaml\n---\nname: deepscan\n---\n\nCreate SKILL.md.\n```\n",
      "utf8",
    );
    await writeFile(
      join(bundledRoot, "deepscan", "SKILL.md"),
      "---\nname: deepscan\ndescription: Run the installed scan\n---\nUse SpawnAgents now.\n",
      "utf8",
    );

    const registry = new SkillRegistry();
    expect(await registry.discover(userRoot, bundledRoot)).toEqual(["deepscan"]);
    expect(registry.get("deepscan")?.metadata.description).toBe("Run the installed scan");
    expect(registry.get("deepscan")?.instructions).toBe("Use SpawnAgents now.");

    await writeFile(
      join(userRoot, "deepscan", "SKILL.md"),
      "```yaml\n---\nname: deepscan\n---\n\nCreate SKILL.md.\n```\n",
      "utf8",
    );
    expect(await registry.refresh(userRoot, bundledRoot)).toEqual(["deepscan"]);
    expect(registry.get("deepscan")?.metadata.description).toBe("Run the installed scan");

    await writeFile(
      join(userRoot, "deepscan", "SKILL.md"),
      "---\nname: deepscan\ndescription: Valid user override\n---\nUse the customized scan.\n",
      "utf8",
    );
    expect(await registry.refresh(userRoot, bundledRoot)).toEqual(["deepscan"]);
    expect(registry.get("deepscan")?.metadata.description).toBe("Valid user override");

    await rm(join(userRoot, "deepscan"), { recursive: true, force: true });
    expect(await registry.refresh(userRoot, bundledRoot)).toEqual(["deepscan"]);
    expect(registry.get("deepscan")?.metadata.description).toBe("Run the installed scan");
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("skill prompt labels installed instructions as execution rather than authoring content", () => {
  const skill = parseSkillMarkdown(
    "---\nname: review\n---\nInspect the current diff.",
    "/skills/review/SKILL.md",
  );

  expect(skillPromptSection(skill)).toContain("already-installed operational skill");
  expect(skillPromptSection(skill)).toContain("Do not create, install, or rewrite a SKILL.md");
});

test("skill refresh retains host-registered skills", async () => {
  const registry = new SkillRegistry();
  const skill = parseSkillMarkdown(
    "---\nname: injected\n---\nUse the host port.",
    "/virtual/injected/SKILL.md",
  );

  registry.register(skill);
  expect(await registry.refresh()).toEqual([]);
  expect(registry.get("injected")).toBe(skill);
});

test("skill refresh swaps one complete snapshot after discovery finishes", async () => {
  const root = await mkdtemp(join(tmpdir(), "xerxes-skill-atomic-refresh-"));
  const firstRoot = join(root, "first");
  const secondRoot = join(root, "second");
  try {
    await mkdir(join(firstRoot, "first"), { recursive: true });
    await mkdir(join(secondRoot, "second"), { recursive: true });
    await writeFile(
      join(firstRoot, "first", "SKILL.md"),
      "---\nname: first\n---\nFirst instructions.\n",
      "utf8",
    );
    await writeFile(
      join(secondRoot, "second", "SKILL.md"),
      "---\nname: second\n---\nSecond instructions.\n",
      "utf8",
    );

    const registry = new SkillRegistry();
    await registry.refresh(firstRoot);
    const refreshing = registry.refresh(secondRoot);
    expect(registry.get("first")?.instructions).toBe("First instructions.");
    expect(registry.get("second")).toBeUndefined();

    await refreshing;
    expect(registry.get("first")).toBeUndefined();
    expect(registry.get("second")?.instructions).toBe("Second instructions.");
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("native bundled skill documentation is discoverable independently from the workspace directory", async () => {
  const registry = new SkillRegistry();
  const discovered = await registry.discover(BUNDLED_SKILLS_DIRECTORY);

  expect(discovered).toEqual(
    expect.arrayContaining([
      "arxiv",
      "excalidraw",
      "find-nearby",
      "google-workspace",
      "grpo-rl-training",
      "ocr-and-documents",
      "polymarket",
      "youtube-content",
    ]),
  );
  expect(
    defaultSkillDiscoveryDirectories({ cwd: "/unrelated-workspace" }),
  ).toContain(BUNDLED_SKILLS_DIRECTORY);
  expect(DEFAULT_OFFICIAL_SKILLS_DIRECTORY).toBe(BUNDLED_SKILLS_DIRECTORY);
  expect(registry.get("youtube-content")?.instructions).toContain(
    "xerxes skill youtube-transcript",
  );
  expect(registry.get("google-workspace")?.instructions).toContain(
    "adapter selected by the caller",
  );
  expect(registry.get("grpo-rl-training")?.instructions).toContain(
    "xerxes skill grpo-rl-training --dry-run",
  );
  expect(registry.get("ocr-and-documents")?.instructions).toContain(
    "Bun-native CLI routes",
  );
});

test("bundled skill resolution skips a source directory without skill assets", async () => {
  const root = await mkdtemp(join(tmpdir(), "xerxes-skill-layout-"));
  const moduleDirectory = join(root, "src", "extensions");
  const sourceSkillsDirectory = join(root, "src", "skills");
  const bundledSkillsDirectory = join(root, "skills", "fixture");
  try {
    await mkdir(sourceSkillsDirectory, { recursive: true });
    await mkdir(bundledSkillsDirectory, { recursive: true });
    await writeFile(join(sourceSkillsDirectory, "runtime.ts"), "export {};\n", "utf8");
    await writeFile(join(bundledSkillsDirectory, "SKILL.md"), "# Fixture\n", "utf8");

    expect(resolveBundledSkillsDirectory({ moduleDirectory })).toBe(join(root, "skills"));
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
