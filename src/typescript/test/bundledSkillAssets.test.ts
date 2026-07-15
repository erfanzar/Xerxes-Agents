// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from "bun:test";
import { existsSync } from "node:fs";
import {
  chmod,
  mkdir,
  mkdtemp,
  readFile,
  readdir,
  rm,
  stat,
  writeFile,
} from "node:fs/promises";
import { join, resolve } from "node:path";
import { tmpdir } from "node:os";

import {
  BUNDLED_AGENTS_SOURCE_DIRECTORY,
  BUNDLED_SKILLS_SOURCE_DIRECTORY,
  copyBundledAgents,
  copyBundledSkills,
} from "../scripts/copyBundledSkills.js";
import { loadBuiltinAgentDefinitions } from "../src/agents/definitions.js";
import { SkillRegistry } from "../src/extensions/skills.js";

const EXPECTED_BUNDLED_SKILLS = [
  "arxiv",
  "ascii-video",
  "excalidraw",
  "find-nearby",
  "google-workspace",
  "grpo-rl-training",
  "ocr-and-documents",
  "polymarket",
  "research-paper-writing",
  "youtube-content",
] as const;

const NON_REDISTRIBUTABLE_SKILL_DIRECTORIES = [
  "creative-ideation",
  "research/research-paper-writing/templates",
  "training/axolotl",
  "training/pytorch-fsdp",
  "training/unsloth",
] as const;

test("native bundled skills have unique names and preserve migrated safe assets", async () => {
  const registry = new SkillRegistry();
  const discovered = await registry.discover(BUNDLED_SKILLS_SOURCE_DIRECTORY);
  const skillFileCount = await countFilesNamed(
    BUNDLED_SKILLS_SOURCE_DIRECTORY,
    "SKILL.md",
  );

  expect(discovered).toEqual(expect.arrayContaining(EXPECTED_BUNDLED_SKILLS));
  expect(discovered).toHaveLength(skillFileCount);
  expect(new Set(discovered).size).toBe(skillFileCount);
  expect(registry.get("godmode")).toBeUndefined();
  const deepScan = registry.get("deepscan");
  expect(deepScan?.metadata.requiredTools).toEqual(
    expect.arrayContaining([
      "SpawnAgents",
      "agent_memory_status",
      "agent_memory_read",
      "agent_memory_write",
      "agent_memory_append",
      "agent_memory_journal",
    ]),
  );
  expect(deepScan?.instructions).toContain("SpawnAgents");
  expect(deepScan?.instructions).toContain("timeout=1800");
  expect(deepScan?.instructions).toContain("Do not compile the report while any returned agent has a non-terminal status");
  expect(deepScan?.instructions).toContain("Do not fall back to `tmp-files`");
  expect(deepScan?.instructions).not.toMatch(/mkdir\s+-p\s+tmp-files/u);
  expect(
    existsSync(
      join(
        BUNDLED_SKILLS_SOURCE_DIRECTORY,
        "excalidraw",
        "references",
        "colors.md",
      ),
    ),
  ).toBeTrue();
  for (const directory of NON_REDISTRIBUTABLE_SKILL_DIRECTORIES) {
    expect(
      existsSync(join(BUNDLED_SKILLS_SOURCE_DIRECTORY, directory)),
    ).toBeFalse();
  }
  const powerpointDirectory = join(
    BUNDLED_SKILLS_SOURCE_DIRECTORY,
    "powerpoint",
  );
  if (existsSync(powerpointDirectory)) {
    expect(await relativeFilePaths(powerpointDirectory)).toEqual(["SKILL.md"]);
  }
  expect(
    existsSync(
      join(
        BUNDLED_SKILLS_SOURCE_DIRECTORY,
        "research",
        "research-paper-writing",
        "references",
        "checklists.md",
      ),
    ),
  ).toBeTrue();
  expect(
    await hasFileWithExtension(BUNDLED_SKILLS_SOURCE_DIRECTORY, ".py"),
  ).toBe(false);
  expect(await executableFilePaths(BUNDLED_SKILLS_SOURCE_DIRECTORY)).toEqual(
    [],
  );
  expect(
    existsSync(
      join(
        BUNDLED_SKILLS_SOURCE_DIRECTORY,
        "github",
        "github-auth",
        "scripts",
        "gh-env.sh",
      ),
    ),
  ).toBeFalse();
  const githubSkillText = await textFileContents(
    join(BUNDLED_SKILLS_SOURCE_DIRECTORY, "github"),
  );
  expect(githubSkillText).toContain("host-injected");
  expect(githubSkillText).not.toMatch(
    /source[^\n]*gh-env\.sh|grep[^\n]*(?:\.git-credentials|\.xerxes\/\.env)/u,
  );
});

test("systematic debugging uses the Bun-native production harness", async () => {
  const registry = new SkillRegistry();
  await registry.discover(BUNDLED_SKILLS_SOURCE_DIRECTORY);

  const skill = registry.get("systematic-debugging");
  expect(skill).toBeDefined();
  expect(skill?.metadata.requiredTools).toEqual([
    "ReadFile",
    "GrepTool",
    "GlobTool",
    "ListDir",
    "exec_command",
    "FileEditTool",
    "WriteFile",
    "AgentTool",
  ]);

  const instructions = skill?.instructions ?? "";
  expect(instructions).toContain(
    'exec_command(cmd="bun", args=["test", "src/typescript/test/failing.test.ts"], workdir=".")',
  );
  expect(instructions).toContain('title="Trace cache invalidation"');
  expect(instructions).toContain("reproduce -> trace -> hypothesize -> fix -> verify");
  expect(instructions).toContain("do not create Python runtime dependencies");
  expect(instructions).not.toMatch(/\b(?:read_file|search_files|terminal)\b/u);
  expect(instructions).not.toMatch(/\b(?:python3|pytest|pip|uv|ruff|mypy)\b/iu);
  expect(new TextEncoder().encode(instructions).byteLength).toBeLessThan(7_000);
});

test("build asset copier creates a discoverable native dist skill tree", async () => {
  const root = await mkdtemp(join(tmpdir(), "xerxes-bundled-skills-"));
  const destinationDirectory = join(root, "dist", "skills");
  try {
    await copyBundledSkills({ destinationDirectory });

    const sourceRegistry = new SkillRegistry();
    const packagedRegistry = new SkillRegistry();
    const sourceNames = await sourceRegistry.discover(
      BUNDLED_SKILLS_SOURCE_DIRECTORY,
    );
    const packagedNames = await packagedRegistry.discover(destinationDirectory);

    expect([...packagedNames].sort()).toEqual([...sourceNames].sort());
    expect(
      existsSync(
        join(
          destinationDirectory,
          "google-workspace",
          "references",
          "gmail-search-syntax.md",
        ),
      ),
    ).toBeTrue();
    expect(
      existsSync(
        join(
          destinationDirectory,
          "polymarket",
          "references",
          "api-endpoints.md",
        ),
      ),
    ).toBeTrue();
    expect(
      existsSync(join(destinationDirectory, "grpo-rl-training", "README.md")),
    ).toBeTrue();
    for (const directory of NON_REDISTRIBUTABLE_SKILL_DIRECTORIES) {
      expect(existsSync(join(destinationDirectory, directory))).toBeFalse();
    }
    const powerpointDirectory = join(destinationDirectory, "powerpoint");
    if (existsSync(powerpointDirectory)) {
      expect(await relativeFilePaths(powerpointDirectory)).toEqual([
        "SKILL.md",
      ]);
    }
    expect(await executableFilePaths(destinationDirectory)).toEqual([]);
  } finally {
    await rm(root, { force: true, recursive: true });
  }
});

test("build asset copier preserves the built-in agent family for installed runtimes", async () => {
  const root = await mkdtemp(join(tmpdir(), "xerxes-bundled-agents-"));
  const destinationDirectory = join(root, "dist", "default");
  try {
    await copyBundledAgents(
      BUNDLED_AGENTS_SOURCE_DIRECTORY,
      destinationDirectory,
    );

    const source = loadBuiltinAgentDefinitions(
      BUNDLED_AGENTS_SOURCE_DIRECTORY,
    );
    const packaged = loadBuiltinAgentDefinitions(destinationDirectory);

    expect([...packaged.keys()].sort()).toEqual([...source.keys()].sort());
    expect(packaged.get("default")?.tools).toContain("SpawnAgents");
    expect(packaged.get("default")?.tools).toContain("AwaitAgents");
    expect(packaged.get("coder")?.systemPrompt).toContain(
      "software engineering implementation",
    );
    expect(await executableFilePaths(destinationDirectory)).toEqual([]);
  } finally {
    await rm(root, { force: true, recursive: true });
  }
});

test("build asset copier rejects executable bundled skill files", async () => {
  const root = await mkdtemp(join(tmpdir(), "xerxes-bundled-skill-mode-"));
  const sourceDirectory = join(root, "source");
  try {
    await mkdir(sourceDirectory, { recursive: true });
    const skillPath = join(sourceDirectory, "SKILL.md");
    await writeFile(
      skillPath,
      "---\nname: executable-fixture\ndescription: fixture\n---\n",
      "utf8",
    );
    await chmod(skillPath, 0o755);

    await expect(
      copyBundledSkills({
        destinationDirectory: join(root, "destination"),
        sourceDirectory,
      }),
    ).rejects.toThrow("Bundled skill asset must not be executable: SKILL.md");
  } finally {
    await rm(root, { force: true, recursive: true });
  }
});

async function hasFileWithExtension(
  directory: string,
  extension: string,
): Promise<boolean> {
  const entries = await readdir(directory, { withFileTypes: true });
  for (const entry of entries) {
    const path = resolve(directory, entry.name);
    if (entry.isDirectory() && (await hasFileWithExtension(path, extension))) {
      return true;
    }
    if (entry.isFile() && entry.name.endsWith(extension)) {
      return true;
    }
  }
  return false;
}

async function relativeFilePaths(
  directory: string,
  prefix = "",
): Promise<string[]> {
  const files: string[] = [];
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const path = prefix ? `${prefix}/${entry.name}` : entry.name;
    if (entry.isDirectory()) {
      files.push(
        ...(await relativeFilePaths(join(directory, entry.name), path)),
      );
    } else if (entry.isFile()) {
      files.push(path);
    }
  }
  return files.sort();
}

async function executableFilePaths(
  directory: string,
  prefix = "",
): Promise<string[]> {
  const files: string[] = [];
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const path = join(directory, entry.name);
    const relativePath = prefix ? `${prefix}/${entry.name}` : entry.name;
    if (entry.isDirectory()) {
      files.push(...(await executableFilePaths(path, relativePath)));
    } else if (entry.isFile() && ((await stat(path)).mode & 0o111) !== 0) {
      files.push(relativePath);
    }
  }
  return files.sort();
}

async function textFileContents(directory: string): Promise<string> {
  const chunks: string[] = [];
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) {
      chunks.push(await textFileContents(path));
    } else if (entry.isFile() && /\.(?:md|sh|ts)$/u.test(entry.name)) {
      chunks.push(await readFile(path, "utf8"));
    }
  }
  return chunks.join("\n");
}

async function countFilesNamed(
  directory: string,
  name: string,
): Promise<number> {
  const entries = await readdir(directory, { withFileTypes: true });
  let count = 0;
  for (const entry of entries) {
    const path = resolve(directory, entry.name);
    if (entry.isDirectory()) {
      count += await countFilesNamed(path, name);
    }
    if (entry.isFile() && entry.name === name) {
      count += 1;
    }
  }
  return count;
}
