// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { existsSync, readdirSync } from "node:fs";
import { readdir, readFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { xerxesHome } from "../daemon/paths.js";

export interface SkillMetadata {
  readonly author: string;
  readonly configVars: readonly string[];
  readonly dependencies: readonly string[];
  readonly description: string;
  readonly name: string;
  readonly platforms: readonly string[];
  readonly requiredTools: readonly string[];
  readonly resources: readonly string[];
  readonly setupCommand: string;
  readonly source: string;
  readonly subcommands: readonly string[];
  readonly tags: readonly string[];
  readonly trustLevel: string;
  readonly version: string;
}

export interface Skill {
  readonly instructions: string;
  readonly metadata: SkillMetadata;
  readonly resourcesDirectory?: string;
  readonly sourcePath: string;
}

export interface SkillDependencyLookup {
  hasTool(name: string): boolean;
}

const activeSkills = new Set<string>();
const PLATFORM_MAP: Readonly<Record<string, NodeJS.Platform>> = {
  macos: "darwin",
  linux: "linux",
  windows: "win32",
};
const SKILL_MODULE_DIRECTORY = dirname(fileURLToPath(import.meta.url));

export interface BundledSkillDirectoryOptions {
  /** Directory containing the compiled runtime module. */
  readonly moduleDirectory?: string;
}

/** Candidate roots for bundled assets across source, dist, and release-package layouts. */
export function bundledSkillDirectoryCandidates(
  options: BundledSkillDirectoryOptions = {},
): string[] {
  const moduleDirectory = resolve(
    options.moduleDirectory ?? SKILL_MODULE_DIRECTORY,
  );
  return [
    resolve(moduleDirectory, "skills"),
    resolve(moduleDirectory, "..", "skills"),
    resolve(moduleDirectory, "..", "..", "skills"),
  ].filter((directory, index, candidates) => candidates.indexOf(directory) === index);
}

/** Resolve native bundled skill assets without depending on the caller's workspace. */
export function resolveBundledSkillsDirectory(
  options: BundledSkillDirectoryOptions = {},
): string {
  const candidates = bundledSkillDirectoryCandidates(options);
  return (
    candidates.find(containsSkillMarkdown) ??
    candidates.find(existsSync) ??
    candidates[0]!
  );
}

/** Native bundled skill documentation, resolved independently from a caller's workspace. */
export const BUNDLED_SKILLS_DIRECTORY = resolveBundledSkillsDirectory();

export function activateSkill(name: string): void {
  activeSkills.add(name);
}

export function activeSkillNames(): string[] {
  return [...activeSkills];
}

export function clearActiveSkills(): void {
  activeSkills.clear();
}

/** Parse a SKILL.md frontmatter block without imposing a YAML runtime dependency. */
export function parseSkillMarkdown(content: string, sourcePath: string): Skill {
  assertOperationalSkillDocument(content, sourcePath);
  const frontmatter = content.match(
    /^---\s*\r?\n([\s\S]*?)\r?\n---\s*\r?\n?([\s\S]*)$/,
  );
  const fields = frontmatter ? parseFrontmatter(frontmatter[1] ?? "") : {};
  const sourceDirectory = dirname(sourcePath);
  const name =
    scalar(fields.name) ||
    sourceDirectory.split(/[\\/]/).filter(Boolean).at(-1) ||
    "skill";
  const explicitSubcommands = stringList(fields.subcommands);
  const metadata: SkillMetadata = {
    name,
    description: scalar(fields.description),
    version: scalar(fields.version) || "1.0",
    tags: stringList(fields.tags),
    resources: stringList(fields.resources),
    author: scalar(fields.author),
    dependencies: stringList(fields.dependencies),
    requiredTools: stringList(fields.required_tools),
    platforms: stringList(fields.platforms),
    configVars: stringList(fields.config_vars),
    trustLevel: scalar(fields.trust_level) || "community",
    source: scalar(fields.source) || "local",
    setupCommand: scalar(fields.setup_command),
    subcommands: explicitSubcommands.length
      ? explicitSubcommands
      : detectedSubcommands(sourceDirectory),
  };
  return {
    metadata,
    instructions: (frontmatter?.[2] ?? content).trim(),
    sourcePath,
    ...(metadata.resources.length
      ? { resourcesDirectory: sourceDirectory }
      : {}),
  };
}

export function skillPromptSection(skill: Skill): string {
  const header = `## Skill: ${skill.metadata.name}${skill.metadata.description ? `\n${skill.metadata.description}` : ""}`;
  return [
    header,
    "This is an already-installed operational skill. Execute its instructions for the current request. Do not create, install, or rewrite a SKILL.md unless the user explicitly asked to author or modify a skill.",
    skill.instructions,
  ].join("\n\n");
}

/**
 * Reject accidentally persisted model transcripts before they can shadow a
 * healthy lower-priority skill with the same name. A valid frontmatter skill
 * may discuss reasoning tags in its body; only wrapper text at the beginning
 * of the file is considered a generated transcript.
 */
function assertOperationalSkillDocument(content: string, sourcePath: string): void {
  const beginning = content.trimStart();
  if (/^<think(?:\s[^>]*)?>/iu.test(beginning)) {
    throw new Error(`Skill document is a model reasoning transcript: ${sourcePath}`);
  }
  if (/^```(?:ya?ml|markdown|md)?\s*\r?\n---(?:\s|$)/iu.test(beginning)) {
    throw new Error(`Skill document wraps its frontmatter in a code fence: ${sourcePath}`);
  }
}

/** In-memory skill index with first-root-wins discovery precedence. */
export class SkillRegistry {
  private discoveryQueue: Promise<void> = Promise.resolve();
  private readonly registeredSkills = new Map<string, Skill>();
  private skills = new Map<string, Skill>();

  get names(): string[] {
    return [...this.skills.keys()];
  }

  async discover(...directories: readonly string[]): Promise<string[]> {
    return this.enqueueDiscovery(async () => {
      const next = new Map(this.skills);
      const discovered = await discoverInto(next, directories);
      this.commitSnapshot(next);
      return discovered;
    });
  }

  /** Re-read discovery roots while retaining explicitly registered host skills. */
  async refresh(...directories: readonly string[]): Promise<string[]> {
    return this.enqueueDiscovery(async () => {
      const next = new Map(this.registeredSkills);
      const discovered = await discoverInto(next, directories);
      this.commitSnapshot(next);
      return discovered;
    });
  }

  register(skill: Skill): void {
    this.registeredSkills.set(skill.metadata.name, skill);
    this.skills.set(skill.metadata.name, skill);
  }

  get(name: string): Skill | undefined {
    return this.skills.get(name);
  }

  all(): Skill[] {
    return [...this.skills.values()];
  }

  private commitSnapshot(next: Map<string, Skill>): void {
    for (const [name, skill] of this.registeredSkills) {
      next.set(name, skill);
    }
    this.skills = next;
  }

  private enqueueDiscovery<T>(operation: () => Promise<T>): Promise<T> {
    const result = this.discoveryQueue.then(operation, operation);
    this.discoveryQueue = result.then(
      () => undefined,
      () => undefined,
    );
    return result;
  }

  search(query = "", tags: readonly string[] = []): Skill[] {
    const normalizedQuery = query.toLowerCase();
    return this.all().filter((skill) => {
      const textMatches =
        !normalizedQuery ||
        skill.metadata.name.toLowerCase().includes(normalizedQuery) ||
        skill.metadata.description.toLowerCase().includes(normalizedQuery);
      const tagMatches =
        !tags.length || tags.some((tag) => skill.metadata.tags.includes(tag));
      return textMatches && tagMatches;
    });
  }

  validateDependencies(tools?: SkillDependencyLookup): string[] {
    const failures: string[] = [];
    for (const skill of this.skills.values()) {
      for (const dependency of skill.metadata.dependencies) {
        if (!this.skills.has(dependency)) {
          failures.push(
            `Skill '${skill.metadata.name}' requires missing dependency '${dependency}'`,
          );
        }
      }
      if (tools) {
        for (const tool of skill.metadata.requiredTools) {
          if (!tools.hasTool(tool)) {
            failures.push(
              `Skill '${skill.metadata.name}' requires missing tool '${tool}'`,
            );
          }
        }
      }
    }
    return failures;
  }

  markdownIndex(): string {
    if (!this.skills.size) {
      return "";
    }
    return [
      "Available skills:",
      ...this.all().map((skill) => {
        const tags = skill.metadata.tags.length
          ? ` [${skill.metadata.tags.join(", ")}]`
          : "";
        return `  - ${skill.metadata.name}: ${skill.metadata.description || "No description"}${tags}`;
      }),
    ].join("\n");
  }
}

async function discoverInto(
  skills: Map<string, Skill>,
  directories: readonly string[],
): Promise<string[]> {
  const discovered: string[] = [];
  for (const directory of directories) {
    for await (const skillPath of skillFiles(directory)) {
      try {
        const skill = parseSkillMarkdown(
          await readFile(skillPath, "utf8"),
          skillPath,
        );
        if (!skills.has(skill.metadata.name)) {
          skills.set(skill.metadata.name, skill);
          discovered.push(skill.metadata.name);
        }
      } catch {
        // A corrupt third-party skill is isolated; remaining skills stay discoverable.
      }
    }
  }
  return discovered;
}

export function defaultSkillDiscoveryDirectories(
  options: {
    readonly cwd?: string;
    readonly userSkillsDirectory?: string;
  } = {},
): string[] {
  const cwd = resolve(options.cwd ?? process.cwd());
  const roots = [
    join(cwd, ".agents", "skills"),
    join(cwd, "skills"),
    options.userSkillsDirectory ?? join(xerxesHome(), "skills"),
    join(xerxesHome(), "agents", "skills"),
    BUNDLED_SKILLS_DIRECTORY,
  ];
  const seen = new Set<string>();
  return roots.filter((root) => {
    const canonical = existsSync(root) ? resolve(root) : resolve(root);
    if (seen.has(canonical)) {
      return false;
    }
    seen.add(canonical);
    return true;
  });
}

export function skillMatchesPlatform(
  skill: Skill,
  currentPlatform: NodeJS.Platform = process.platform,
): boolean {
  if (!skill.metadata.platforms.length) {
    return true;
  }
  return skill.metadata.platforms.some(
    (platform) => PLATFORM_MAP[platform.toLowerCase()] === currentPlatform,
  );
}

async function* skillFiles(directory: string): AsyncGenerator<string> {
  try {
    const entries = await readdir(directory, {
      encoding: "utf8",
      withFileTypes: true,
    });
    for (const entry of entries) {
      const path = join(directory, entry.name);
      if (entry.isDirectory()) {
        yield* skillFiles(path);
      } else if (entry.isFile() && entry.name === "SKILL.md") {
        yield path;
      }
    }
  } catch {
    return;
  }
}

/** Whether a directory is an asset tree rather than a TypeScript source directory. */
function containsSkillMarkdown(directory: string): boolean {
  if (!existsSync(directory)) return false;
  const pending = [directory];
  while (pending.length) {
    const current = pending.pop()!;
    try {
      for (const entry of readdirSync(current, { withFileTypes: true })) {
        if (entry.isFile() && entry.name === "SKILL.md") return true;
        if (entry.isDirectory()) pending.push(join(current, entry.name));
      }
    } catch {
      return false;
    }
  }
  return false;
}

function detectedSubcommands(sourceDirectory: string): string[] {
  const referencesDirectory = join(sourceDirectory, "references");
  try {
    return readdirSync(referencesDirectory, { withFileTypes: true })
      .filter((entry) => entry.isFile() && entry.name.endsWith("-workflow.md"))
      .map((entry) => entry.name.slice(0, -"-workflow.md".length))
      .filter(Boolean)
      .sort();
  } catch {
    return [];
  }
}

type FrontmatterValue = string | string[];

function parseFrontmatter(content: string): Record<string, FrontmatterValue> {
  const fields: Record<string, FrontmatterValue> = {};
  let listKey: string | undefined;
  for (const rawLine of content.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) {
      continue;
    }
    if (line.startsWith("- ") && listKey) {
      const current = fields[listKey];
      fields[listKey] = [
        ...(Array.isArray(current) ? current : []),
        stripQuotes(line.slice(2).trim()),
      ];
      continue;
    }
    const separator = line.indexOf(":");
    if (separator < 0) {
      continue;
    }
    const key = line.slice(0, separator).trim();
    const value = line.slice(separator + 1).trim();
    listKey = value ? undefined : key;
    fields[key] =
      value.startsWith("[") && value.endsWith("]")
        ? value
            .slice(1, -1)
            .split(",")
            .map((item) => stripQuotes(item.trim()))
            .filter(Boolean)
        : stripQuotes(value);
  }
  return fields;
}

function scalar(value: FrontmatterValue | undefined): string {
  return typeof value === "string" ? value : "";
}

function stringList(value: FrontmatterValue | undefined): string[] {
  if (typeof value === "string") {
    return value ? [value] : [];
  }
  return value?.filter(Boolean) ?? [];
}

function stripQuotes(value: string): string {
  return value.replace(/^(?:"([\s\S]*)"|'([\s\S]*)')$/, "$1$2").trim();
}
