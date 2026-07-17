// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import {
  lstat,
  mkdir,
  readFile,
  readdir,
  realpath,
  rename,
  rmdir,
  rm,
  unlink,
} from "node:fs/promises";
import { homedir } from "node:os";
import {
  basename,
  dirname,
  isAbsolute,
  join,
  relative,
  resolve,
  sep,
} from "node:path";

import { PathEscape, resolveWithin } from "../security/pathSecurity.js";
import { isJsonObject, type JsonObject } from "../types/toolCalls.js";
import { BUNDLED_SKILLS_DIRECTORY, parseSkillMarkdown } from "./skills.js";
import {
  HUB_DIR as DEFAULT_HUB_DIR,
  SKILLS_DIR as DEFAULT_SKILLS_DIR,
  SkillGuardPathError,
  approveSkill as approveGuardedSkill,
  loadTrustedHashes,
  quarantineSkill,
  scanSkill,
  type SkillGuardPaths,
} from "./skillsGuard.js";

export const SKILL_HUB_LOCK_FILE = join(DEFAULT_HUB_DIR, "lock.json");
export const SKILL_HUB_AUDIT_LOG = join(DEFAULT_HUB_DIR, "audit.log");
export const DEFAULT_OFFICIAL_SKILLS_DIRECTORY = BUNDLED_SKILLS_DIRECTORY;

/** Raised when an install, uninstall, or lock operation would use an unsafe local path. */
export class SkillHubPathError extends SkillGuardPathError {
  constructor(path: string, reason: string) {
    super(path, reason);
    this.name = "SkillHubPathError";
  }
}

/** A transport-independent skill payload. Bundle files are intentionally not installed by the Python-compatible hub. */
export interface SkillBundle {
  readonly content: string;
  readonly files?: Readonly<Record<string, string>>;
  readonly metadata?: Readonly<JsonObject>;
  readonly name: string;
}

export interface SkillSearchResult {
  readonly identifier: string;
  readonly name: string;
  readonly source: string;
}

/** A source is local code or an explicitly supplied transport; this module performs no network requests itself. */
export interface SkillSource {
  readonly name: string;
  fetch(identifier: string): Promise<SkillBundle>;
  search(query: string, limit?: number): Promise<readonly SkillSearchResult[]>;
}

/** Explicit host boundary for remote skill transport. Implementations decide authentication and network policy. */
export interface RemoteSkillTransport {
  fetch(identifier: string): Promise<SkillBundle>;
  search(query: string, limit: number): Promise<readonly SkillSearchResult[]>;
}

export interface LocalSkillSourceOptions {
  readonly skillsDirectory?: string;
}

export interface OfficialSkillSourceOptions {
  readonly directory?: string;
}

export interface SkillsHubOptions extends SkillGuardPaths {
  readonly now?: () => Date;
  readonly officialSkillsDirectory?: string;
  /** Replaces the default local/official source pair when supplied. */
  readonly sources?: readonly SkillSource[];
}

export interface SkillHubEntry {
  readonly identifier: string;
  readonly installedAt: number;
  readonly metadata: Readonly<JsonObject>;
  readonly name: string;
  readonly path: string;
  readonly source: string;
}

interface StoredSkillEntry {
  readonly identifier: string;
  readonly installedAt: number;
  readonly metadata: Readonly<JsonObject>;
  readonly source: string;
}

interface HubDirectories {
  readonly hub: string;
  readonly skills: string;
}

type ChildEntry =
  | { readonly kind: "directory"; readonly path: string }
  | { readonly kind: "file"; readonly path: string }
  | { readonly kind: "missing" }
  | { readonly kind: "other"; readonly path: string };

/** Read a skill bundle from a local directory or direct SKILL.md path without following symlinks. */
export class LocalSkillSource implements SkillSource {
  readonly name = "local";
  readonly skillsDirectory: string;

  constructor(options: LocalSkillSourceOptions = {}) {
    this.skillsDirectory = normalizePath(
      options.skillsDirectory ?? DEFAULT_SKILLS_DIR,
    );
  }

  async fetch(identifier: string): Promise<SkillBundle> {
    const skillMarkdown = await resolveExternalSkillMarkdown(
      identifier,
      "local skill",
    );
    const content = await readFile(skillMarkdown, "utf8");
    const parsed = parseSkillMarkdown(content, skillMarkdown);
    return {
      name: parsed.metadata.name,
      content,
      files: {},
      metadata: { path: dirname(skillMarkdown) },
    };
  }

  async search(
    query: string,
    limit = 10,
  ): Promise<readonly SkillSearchResult[]> {
    return searchDirectory(this.skillsDirectory, query, limit, this.name, {
      skipHubDirectory: true,
    });
  }
}

/** Read bundled official skills from an explicit local directory. No Python package or remote fallback is used. */
export class OfficialSkillSource implements SkillSource {
  readonly directory: string;
  readonly name = "official";

  constructor(options: OfficialSkillSourceOptions = {}) {
    this.directory = normalizePath(
      options.directory ?? DEFAULT_OFFICIAL_SKILLS_DIRECTORY,
    );
  }

  async fetch(identifier: string): Promise<SkillBundle> {
    const name = validateSkillName(identifier);
    const root = await requireDirectory(
      this.directory,
      "official skills directory",
    );
    const skillDirectory = await requireChildDirectory(
      root,
      name,
      "official skill directory",
    );
    const skillMarkdown = await requireChildFile(
      skillDirectory,
      "SKILL.md",
      "official SKILL.md",
    );
    return {
      name,
      content: await readFile(skillMarkdown, "utf8"),
      files: {},
      metadata: { official: true },
    };
  }

  async search(
    query: string,
    limit = 10,
  ): Promise<readonly SkillSearchResult[]> {
    return searchDirectory(this.directory, query, limit, this.name);
  }
}

/** Adapt a host-provided remote transport without embedding network, authentication, or scraping behavior. */
export class InjectedRemoteSkillSource implements SkillSource {
  readonly name: string;
  private readonly transport: RemoteSkillTransport;

  constructor(name: string, transport: RemoteSkillTransport) {
    this.name = validateSourceName(name);
    this.transport = transport;
  }

  async fetch(identifier: string): Promise<SkillBundle> {
    return normalizeBundle(await this.transport.fetch(identifier));
  }

  async search(
    query: string,
    limit = 10,
  ): Promise<readonly SkillSearchResult[]> {
    return normalizeSearchResults(
      await this.transport.search(query, normalizeLimit(limit)),
      this.name,
    );
  }
}

/**
 * Local skill installer and index. Remote installation is possible only when callers explicitly inject a SkillSource.
 */
export class SkillsHub {
  private readonly auditLocks = new Map<string, Promise<void>>();
  private readonly guardPaths: SkillGuardPaths;
  private readonly now: () => Date;
  private readonly sources = new Map<string, SkillSource>();

  constructor(options: SkillsHubOptions = {}) {
    this.guardPaths = {
      ...(options.skillsDirectory === undefined
        ? {}
        : { skillsDirectory: options.skillsDirectory }),
      ...(options.quarantineDirectory === undefined
        ? {}
        : { quarantineDirectory: options.quarantineDirectory }),
    };
    this.now = options.now ?? (() => new Date());
    const defaults: readonly SkillSource[] = [
      new LocalSkillSource(
        options.skillsDirectory === undefined
          ? {}
          : { skillsDirectory: options.skillsDirectory },
      ),
      new OfficialSkillSource(
        options.officialSkillsDirectory === undefined
          ? {}
          : { directory: options.officialSkillsDirectory },
      ),
    ];
    for (const source of options.sources ?? defaults)
      this.registerSource(source);
  }

  /** Install SKILL.md from one configured local or explicitly injected source. */
  async install(
    uri: string,
    options: { readonly force?: boolean } = {},
  ): Promise<string> {
    const { identifier, sourceName } = parseInstallUri(uri, this.sources);
    const source = this.sources.get(sourceName);
    if (source === undefined) return `[Error] Unknown source: ${sourceName}`;

    let bundle: SkillBundle;
    try {
      bundle = normalizeBundle(await source.fetch(identifier));
    } catch (error) {
      return `[Error] Failed to fetch ${uri}: ${errorMessage(error)}`;
    }

    let skillName: string;
    try {
      skillName = validateSkillName(bundle.name);
    } catch (error) {
      return `[Error] Invalid skill name from ${sourceName}: ${errorMessage(error)}`;
    }

    let directories: HubDirectories;
    try {
      directories = await this.ensureDirectories();
      const existing = await inspectChild(
        directories.skills,
        skillName,
        "installed skill directory",
      );
      if (existing.kind === "file" || existing.kind === "other") {
        return `[Error] Skill '${skillName}' has an unsafe existing destination.`;
      }
      if (existing.kind === "directory") {
        if (!options.force)
          return `[Error] Skill '${skillName}' already installed. Use force=true to overwrite.`;
        await removeDirectoryTree(directories.skills, existing.path);
      }

      const target = await ensureChildDirectory(
        directories.skills,
        skillName,
        "installed skill directory",
      );
      await writeDirectFile(
        target,
        "SKILL.md",
        bundle.content,
        "installed SKILL.md",
      );
      const scan = await scanSkill(target, {
        trustedHashes: await loadTrustedHashes(this.guardPaths),
      });
      if (!scan.isSafe) {
        // Failing content is quarantined for operator review instead of activated.
        await quarantineSkill(target, this.guardPaths);
        await this.appendAudit(
          directories,
          "quarantine",
          `${skillName} from ${sourceName}:${identifier}: ${scan.summary}`,
        );
        return `[Error] Skill '${skillName}' failed the security scan and was quarantined: ${scan.summary}`;
      }
      const lock = await this.loadLock(directories);
      lock[skillName] = {
        source: sourceName,
        identifier,
        installedAt: validNow(this.now()).valueOf() / 1000,
        metadata: normalizedMetadata(bundle.metadata),
      };
      await this.saveLock(directories, lock);
      await this.appendAudit(
        directories,
        "install",
        `${skillName} from ${sourceName}:${identifier}`,
      );
    } catch (error) {
      return `[Error] Failed to install ${uri}: ${errorMessage(error)}`;
    }

    return `Installed skill '${skillName}' from ${sourceName}:${identifier}`;
  }

  /** Remove one installed direct-child skill and its lock record, without following unsafe paths. */
  async uninstall(skillName: string): Promise<string> {
    let name: string;
    try {
      name = validateSkillName(skillName);
    } catch (error) {
      return `[Error] Invalid skill name: ${errorMessage(error)}`;
    }

    let directories: HubDirectories;
    try {
      directories = await this.ensureDirectories();
      const installed = await inspectChild(
        directories.skills,
        name,
        "installed skill directory",
      );
      if (installed.kind === "missing")
        return `[Error] Skill '${name}' is not installed.`;
      if (installed.kind !== "directory")
        return `[Error] Skill '${name}' has an unsafe installed path.`;
      await removeDirectoryTree(directories.skills, installed.path);
      const lock = await this.loadLock(directories);
      if (lock[name] !== undefined) {
        delete lock[name];
        await this.saveLock(directories, lock);
      }
      await this.appendAudit(directories, "uninstall", name);
    } catch (error) {
      return `[Error] Failed to uninstall '${name}': ${errorMessage(error)}`;
    }
    return `Uninstalled skill '${name}'`;
  }

  /** Delegate contained quarantine approval to SkillsGuard, then record the hub audit event. */
  async approveSkill(skillName: string): Promise<string> {
    const result = await approveGuardedSkill(skillName, this.guardPaths);
    if (result.startsWith("[Error]")) return result;
    const directories = await this.ensureDirectories();
    await this.appendAudit(directories, "approve", skillName);
    return result;
  }

  /** Return lock-file records for installed skills in deterministic name order. */
  async listInstalled(): Promise<SkillHubEntry[]> {
    const directories = await this.existingDirectories();
    if (directories === undefined) return [];
    const lock = await this.loadLock(directories);
    return Object.entries(lock)
      .sort(([left], [right]) => compareLexical(left, right))
      .map(([name, entry]) => ({
        name,
        source: entry.source,
        identifier: entry.identifier,
        installedAt: entry.installedAt,
        metadata: entry.metadata,
        path: join(directories.skills, name),
      }));
  }

  /** Search all configured sources. Each source receives the requested per-source limit. */
  async search(query = "", limit = 10): Promise<SkillSearchResult[]> {
    const normalizedLimit = normalizeLimit(limit);
    const results: SkillSearchResult[] = [];
    for (const [sourceName, source] of this.sources) {
      try {
        const matches = await source.search(query, normalizedLimit);
        for (const match of matches) {
          results.push({
            name: match.name,
            identifier: match.identifier,
            source: sourceName,
          });
        }
      } catch {
        // One optional source must not prevent search over other configured sources.
      }
    }
    return results;
  }

  getSource(name: string): SkillSource | undefined {
    return this.sources.get(name);
  }

  registerSource(source: SkillSource): void {
    const name = validateSourceName(source.name);
    if (this.sources.has(name))
      throw new Error(`Skill source '${name}' is already registered`);
    this.sources.set(name, source);
  }

  private async ensureDirectories(): Promise<HubDirectories> {
    const skillsDirectory = normalizePath(
      this.guardPaths.skillsDirectory ?? DEFAULT_SKILLS_DIR,
    );
    const skills = await ensureDirectory(skillsDirectory, "skills directory");
    const hub = await ensureChildDirectory(
      skills,
      ".hub",
      "skills hub directory",
    );
    return { skills, hub };
  }

  private async existingDirectories(): Promise<HubDirectories | undefined> {
    const skillsDirectory = normalizePath(
      this.guardPaths.skillsDirectory ?? DEFAULT_SKILLS_DIR,
    );
    const skills = await existingDirectory(skillsDirectory, "skills directory");
    if (skills === undefined) return undefined;
    const hub = await existingChildDirectory(
      skills,
      ".hub",
      "skills hub directory",
    );
    if (hub === undefined) return { skills, hub: undefinedHubPath(skills) };
    return { skills, hub };
  }

  private async loadLock(
    directories: HubDirectories,
  ): Promise<Record<string, StoredSkillEntry>> {
    const lock = await existingFile(
      directories.hub,
      "lock.json",
      "skills hub lock file",
    );
    if (lock === undefined) return {};
    try {
      return parseLock(await readFile(lock, "utf8"));
    } catch {
      return {};
    }
  }

  private async saveLock(
    directories: HubDirectories,
    lock: Readonly<Record<string, StoredSkillEntry>>,
  ): Promise<void> {
    await writeDirectFile(
      directories.hub,
      "lock.json",
      JSON.stringify(lock, null, 2),
      "skills hub lock file",
    );
  }

  private async appendAudit(
    directories: HubDirectories,
    event: string,
    detail: string,
  ): Promise<void> {
    return this.withAuditLock(directories.hub, async () => {
      const audit = await existingFile(
        directories.hub,
        "audit.log",
        "skills hub audit log",
      );
      const previous = audit === undefined ? "" : await readFile(audit, "utf8");
      const now = validNow(this.now());
      const line = `${formatTimestamp(now)}  ${event.padEnd(12)}  ${detail.replace(/[\r\n]+/g, " ")}\n`;
      await writeDirectFile(
        directories.hub,
        "audit.log",
        previous + line,
        "skills hub audit log",
      );
    });
  }

  private async withAuditLock<T>(
    key: string,
    operation: () => Promise<T>,
  ): Promise<T> {
    const previous = this.auditLocks.get(key) ?? Promise.resolve();
    let release: (() => void) | undefined;
    const current = new Promise<void>((resolveLock) => {
      release = resolveLock;
    });
    this.auditLocks.set(key, current);
    await previous;
    try {
      return await operation();
    } finally {
      release?.();
      if (this.auditLocks.get(key) === current) this.auditLocks.delete(key);
    }
  }
}

async function searchDirectory(
  path: string,
  query: string,
  limit: number,
  source: string,
  options: { readonly skipHubDirectory?: boolean } = {},
): Promise<SkillSearchResult[]> {
  const maximum = normalizeLimit(limit);
  let root: string;
  try {
    root = await requireDirectory(path, "skill search directory");
  } catch {
    return [];
  }
  const matches: SkillSearchResult[] = [];
  await collectSearchResults(
    root,
    root,
    query.toLowerCase(),
    maximum,
    source,
    options.skipHubDirectory ?? false,
    matches,
  );
  return matches;
}

async function collectSearchResults(
  root: string,
  directory: string,
  query: string,
  limit: number,
  source: string,
  skipHubDirectory: boolean,
  matches: SkillSearchResult[],
): Promise<void> {
  if (matches.length >= limit) return;
  let entries;
  try {
    entries = await readdir(directory, { withFileTypes: true });
  } catch {
    return;
  }
  const names = entries.map((entry) => entry.name).sort(compareLexical);
  for (const name of names) {
    if (matches.length >= limit) return;
    if (skipHubDirectory && directory === root && name === ".hub") continue;
    const candidate = join(directory, name);
    let metadata;
    try {
      metadata = await lstat(candidate);
    } catch {
      continue;
    }
    if (metadata.isSymbolicLink()) continue;
    if (metadata.isDirectory()) {
      await collectSearchResults(
        root,
        candidate,
        query,
        limit,
        source,
        skipHubDirectory,
        matches,
      );
      continue;
    }
    if (!metadata.isFile() || name !== "SKILL.md") continue;
    try {
      const content = await readFile(candidate, "utf8");
      if (content.toLowerCase().includes(query)) {
        matches.push({
          name: basename(dirname(candidate)),
          source,
          identifier: dirname(candidate),
        });
      }
    } catch {
      // Corrupt third-party skill content is skipped while discovery continues.
    }
  }
}

async function resolveExternalSkillMarkdown(
  identifier: string,
  label: string,
): Promise<string> {
  const path = normalizePath(identifier);
  let metadata;
  try {
    metadata = await lstat(path);
  } catch (error) {
    if (isMissing(error)) throw new Error(`${label} not found: ${identifier}`);
    throw hubPathError(path, `cannot inspect ${label}`, error);
  }
  if (metadata.isSymbolicLink())
    throw new SkillHubPathError(path, `${label} must not be a symbolic link`);
  if (metadata.isFile()) {
    if (basename(path) !== "SKILL.md")
      throw new Error(`No SKILL.md at ${path}`);
    return requireRegularFile(path, "SKILL.md");
  }
  if (!metadata.isDirectory())
    throw new Error(`${label} must be a directory or SKILL.md file`);
  const directory = await requireDirectory(path, label);
  return requireChildFile(directory, "SKILL.md", "SKILL.md");
}

async function inspectChild(
  root: string,
  name: string,
  label: string,
): Promise<ChildEntry> {
  const candidate = directChildPath(root, name, label);
  let metadata;
  try {
    metadata = await lstat(candidate);
  } catch (error) {
    if (isMissing(error)) return { kind: "missing" };
    throw hubPathError(candidate, `cannot inspect ${label}`, error);
  }
  if (metadata.isSymbolicLink())
    throw new SkillHubPathError(
      candidate,
      `${label} must not be a symbolic link`,
    );
  const path = await containedChildPath(root, name, label);
  if (metadata.isDirectory()) return { kind: "directory", path };
  if (metadata.isFile()) return { kind: "file", path };
  return { kind: "other", path };
}

async function requireDirectory(path: string, label: string): Promise<string> {
  let metadata;
  try {
    metadata = await lstat(path);
  } catch (error) {
    if (isMissing(error))
      throw new SkillHubPathError(path, `${label} does not exist`);
    throw hubPathError(path, `cannot inspect ${label}`, error);
  }
  if (metadata.isSymbolicLink())
    throw new SkillHubPathError(path, `${label} must not be a symbolic link`);
  if (!metadata.isDirectory())
    throw new SkillHubPathError(path, `${label} must be a directory`);
  try {
    return await realpath(path);
  } catch (error) {
    throw hubPathError(path, `cannot resolve ${label}`, error);
  }
}

async function existingDirectory(
  path: string,
  label: string,
): Promise<string | undefined> {
  try {
    await lstat(path);
  } catch (error) {
    if (isMissing(error)) return undefined;
    throw hubPathError(path, `cannot inspect ${label}`, error);
  }
  return requireDirectory(path, label);
}

async function ensureDirectory(path: string, label: string): Promise<string> {
  const existing = await existingDirectory(path, label);
  if (existing !== undefined) return existing;
  try {
    await mkdir(path, { recursive: true });
  } catch (error) {
    throw hubPathError(path, `cannot create ${label}`, error);
  }
  return requireDirectory(path, label);
}

async function existingChildDirectory(
  root: string,
  name: string,
  label: string,
): Promise<string | undefined> {
  const entry = await inspectChild(root, name, label);
  if (entry.kind === "missing") return undefined;
  if (entry.kind !== "directory")
    throw new SkillHubPathError(entry.path, `${label} must be a directory`);
  return entry.path;
}

async function requireChildDirectory(
  root: string,
  name: string,
  label: string,
): Promise<string> {
  const directory = await existingChildDirectory(root, name, label);
  if (directory === undefined)
    throw new SkillHubPathError(join(root, name), `${label} does not exist`);
  return directory;
}

async function ensureChildDirectory(
  root: string,
  name: string,
  label: string,
): Promise<string> {
  const existing = await existingChildDirectory(root, name, label);
  if (existing !== undefined) return existing;
  const candidate = directChildPath(root, name, label);
  try {
    await mkdir(candidate);
  } catch (error) {
    if (!isAlreadyExists(error))
      throw hubPathError(candidate, `cannot create ${label}`, error);
  }
  return requireChildDirectory(root, name, label);
}

async function existingFile(
  root: string,
  name: string,
  label: string,
): Promise<string | undefined> {
  const entry = await inspectChild(root, name, label);
  if (entry.kind === "missing") return undefined;
  if (entry.kind !== "file")
    throw new SkillHubPathError(entry.path, `${label} must be a regular file`);
  return entry.path;
}

async function requireChildFile(
  root: string,
  name: string,
  label: string,
): Promise<string> {
  const file = await existingFile(root, name, label);
  if (file === undefined) throw new Error(`${label} not found`);
  return file;
}

async function requireRegularFile(
  path: string,
  label: string,
): Promise<string> {
  let metadata;
  try {
    metadata = await lstat(path);
  } catch (error) {
    if (isMissing(error)) throw new Error(`${label} not found`);
    throw hubPathError(path, `cannot inspect ${label}`, error);
  }
  if (metadata.isSymbolicLink())
    throw new SkillHubPathError(path, `${label} must not be a symbolic link`);
  if (!metadata.isFile())
    throw new SkillHubPathError(path, `${label} must be a regular file`);
  try {
    return await realpath(path);
  } catch (error) {
    throw hubPathError(path, `cannot resolve ${label}`, error);
  }
}

async function writeDirectFile(
  root: string,
  name: string,
  content: string,
  label: string,
): Promise<void> {
  const existing = await existingFile(root, name, label);
  const target = existing ?? (await containedChildPath(root, name, label));
  const temporary = join(
    dirname(target),
    `.${basename(target)}.${crypto.randomUUID()}.tmp`,
  );
  try {
    await Bun.write(temporary, content);
    await rename(temporary, target);
  } catch (error) {
    throw hubPathError(target, `cannot write ${label}`, error);
  } finally {
    try {
      await rm(temporary, { force: true });
    } catch {
      // A failed temporary cleanup cannot justify masking the primary storage error.
    }
  }
}

async function removeDirectoryTree(
  root: string,
  directory: string,
): Promise<void> {
  const safeDirectory = await containedPath(root, directory, "skill directory");
  let entries;
  try {
    entries = await readdir(safeDirectory, { withFileTypes: true });
  } catch (error) {
    throw hubPathError(
      safeDirectory,
      "cannot list skill directory for removal",
      error,
    );
  }
  for (const entry of entries) {
    const child = join(safeDirectory, entry.name);
    let metadata;
    try {
      metadata = await lstat(child);
    } catch (error) {
      throw hubPathError(
        child,
        "cannot inspect skill entry for removal",
        error,
      );
    }
    if (metadata.isSymbolicLink())
      throw new SkillHubPathError(
        child,
        "skill removal refuses symbolic links",
      );
    if (metadata.isDirectory()) {
      await removeDirectoryTree(root, child);
      continue;
    }
    if (!metadata.isFile())
      throw new SkillHubPathError(
        child,
        "skill removal refuses non-file entries",
      );
    try {
      await unlink(child);
    } catch (error) {
      throw hubPathError(child, "cannot remove skill file", error);
    }
  }
  try {
    await rmdir(safeDirectory);
  } catch (error) {
    throw hubPathError(safeDirectory, "cannot remove skill directory", error);
  }
}

async function containedChildPath(
  root: string,
  name: string,
  label: string,
): Promise<string> {
  directChildPath(root, name, label);
  try {
    return await resolveWithin(root, name);
  } catch (error) {
    if (error instanceof PathEscape)
      throw new SkillHubPathError(name, `${label} escapes root ${root}`);
    throw hubPathError(name, `cannot resolve ${label}`, error);
  }
}

async function containedPath(
  root: string,
  candidate: string,
  label: string,
): Promise<string> {
  if (!isWithin(root, candidate))
    throw new SkillHubPathError(candidate, `${label} escapes root ${root}`);
  const name = relative(root, candidate);
  if (!name || name === ".." || name.includes(`..${sep}`)) {
    throw new SkillHubPathError(
      candidate,
      `${label} must be below root ${root}`,
    );
  }
  try {
    return await resolveWithin(root, name);
  } catch (error) {
    if (error instanceof PathEscape)
      throw new SkillHubPathError(candidate, `${label} escapes root ${root}`);
    throw hubPathError(candidate, `cannot resolve ${label}`, error);
  }
}

function directChildPath(root: string, name: string, label: string): string {
  if (
    !name ||
    name.includes("\0") ||
    basename(name) !== name ||
    name === "." ||
    name === ".."
  ) {
    throw new SkillHubPathError(name, `${label} must use one path segment`);
  }
  const candidate = resolve(root, name);
  if (!isWithin(root, candidate))
    throw new SkillHubPathError(candidate, `${label} escapes root ${root}`);
  return candidate;
}

function validateSkillName(name: string): string {
  const isPlainName =
    typeof name === "string" &&
    Boolean(name) &&
    name !== "." &&
    name !== ".." &&
    name !== ".hub" &&
    !name.includes("\0") &&
    basename(name) === name &&
    !name.includes("..");
  if (!isPlainName)
    throw new SkillHubPathError(
      String(name),
      "skill name must be a plain directory name",
    );
  return name;
}

function validateSourceName(name: string): string {
  if (typeof name !== "string" || !/^[a-z][a-z0-9_-]*$/i.test(name)) {
    throw new TypeError(`invalid skill source name: ${JSON.stringify(name)}`);
  }
  return name;
}

function normalizeBundle(value: SkillBundle): SkillBundle {
  const isBundle =
    typeof value === "object" &&
    value !== null &&
    typeof value.name === "string" &&
    Boolean(value.name) &&
    typeof value.content === "string";
  if (!isBundle) {
    throw new TypeError("skill source returned an invalid bundle");
  }
  if (value.files !== undefined && !isStringRecord(value.files)) {
    throw new TypeError("skill bundle files must map strings to strings");
  }
  return {
    name: value.name,
    content: value.content,
    ...(value.files === undefined ? {} : { files: { ...value.files } }),
    ...(value.metadata === undefined
      ? {}
      : { metadata: normalizedMetadata(value.metadata) }),
  };
}

function normalizeSearchResults(
  values: readonly SkillSearchResult[],
  source: string,
): SkillSearchResult[] {
  const results: SkillSearchResult[] = [];
  for (const value of values) {
    if (typeof value?.name !== "string" || typeof value.identifier !== "string")
      continue;
    results.push({ name: value.name, identifier: value.identifier, source });
  }
  return results;
}

function normalizedMetadata(
  value: Readonly<JsonObject> | undefined,
): Readonly<JsonObject> {
  if (value === undefined) return {};
  try {
    const parsed: unknown = JSON.parse(JSON.stringify(value));
    return isJsonObject(parsed) ? parsed : {};
  } catch {
    return {};
  }
}

function parseLock(raw: string): Record<string, StoredSkillEntry> {
  try {
    const decoded: unknown = JSON.parse(raw);
    if (!isJsonObject(decoded)) return {};
    const entries: Record<string, StoredSkillEntry> = {};
    for (const [name, value] of Object.entries(decoded)) {
      if (!isSafeSkillName(name) || !isJsonObject(value)) continue;
      const source = value.source;
      const identifier = value.identifier;
      const installedAt = value.installedAt;
      const hasValidEntry =
        typeof source === "string" &&
        typeof identifier === "string" &&
        typeof installedAt === "number" &&
        Number.isFinite(installedAt);
      if (!hasValidEntry) {
        continue;
      }
      const metadata = normalizedMetadata(
        isJsonObject(value.metadata) ? value.metadata : {},
      );
      entries[name] = { source, identifier, installedAt, metadata };
    }
    return entries;
  } catch {
    return {};
  }
}

function parseInstallUri(
  uri: string,
  sources: ReadonlyMap<string, SkillSource>,
): {
  readonly identifier: string;
  readonly sourceName: string;
} {
  if (typeof uri !== "string" || !uri.trim())
    throw new TypeError("skill URI must be a non-empty string");
  const separator = uri.indexOf(":");
  if (separator > 0) {
    const candidate = uri.slice(0, separator);
    // Only a registered source name is a scheme; anything else (for example a
    // Windows drive prefix like 'C:') is part of a local path identifier.
    if (sources.has(candidate)) {
      return { sourceName: candidate, identifier: uri.slice(separator + 1) };
    }
  }
  return { sourceName: "local", identifier: uri };
}

function normalizePath(path: string): string {
  if (typeof path !== "string")
    throw new TypeError("skill path must be a string");
  const trimmed = path.trim();
  if (!trimmed || trimmed.includes("\0")) {
    throw new SkillHubPathError(
      path,
      "path must be non-empty and contain no null bytes",
    );
  }
  return resolve(expandHome(trimmed));
}

function expandHome(path: string): string {
  if (path === "~") return homedir();
  if (path.startsWith("~/") || path.startsWith("~\\"))
    return join(homedir(), path.slice(2));
  return path;
}

function formatTimestamp(date: Date): string {
  const offset = -date.getTimezoneOffset();
  const sign = offset >= 0 ? "+" : "-";
  const absoluteOffset = Math.abs(offset);
  const datePart = `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`;
  const timePart = `${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
  const offsetPart = `${sign}${pad(Math.floor(absoluteOffset / 60))}${pad(absoluteOffset % 60)}`;
  return `${datePart}T${timePart}${offsetPart}`;
}

function validNow(now: Date): Date {
  if (!(now instanceof Date) || Number.isNaN(now.valueOf()))
    throw new TypeError("now must return a valid Date");
  return now;
}

function normalizeLimit(limit: number): number {
  if (!Number.isSafeInteger(limit) || limit < 1)
    throw new RangeError("limit must be a positive safe integer");
  return limit;
}

function isStringRecord(
  value: unknown,
): value is Readonly<Record<string, string>> {
  return (
    typeof value === "object" &&
    value !== null &&
    !Array.isArray(value) &&
    Object.values(value).every((entry) => typeof entry === "string")
  );
}

function isSafeSkillName(value: string): boolean {
  try {
    validateSkillName(value);
    return true;
  } catch {
    return false;
  }
}

function isWithin(root: string, candidate: string): boolean {
  const pathFromRoot = relative(root, candidate);
  return (
    pathFromRoot === "" ||
    (!pathFromRoot.startsWith(`..${sep}`) &&
      pathFromRoot !== ".." &&
      !isAbsolute(pathFromRoot))
  );
}

function undefinedHubPath(skills: string): string {
  return join(skills, ".hub");
}

function pad(value: number): string {
  return String(value).padStart(2, "0");
}

function compareLexical(left: string, right: string): number {
  if (left < right) return -1;
  if (left > right) return 1;
  return 0;
}

function hubPathError(
  path: string,
  action: string,
  error: unknown,
): SkillHubPathError {
  return new SkillHubPathError(path, `${action}: ${errorMessage(error)}`);
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function isMissing(error: unknown): boolean {
  return (
    typeof error === "object" &&
    error !== null &&
    "code" in error &&
    error.code === "ENOENT"
  );
}

function isAlreadyExists(error: unknown): boolean {
  return (
    typeof error === "object" &&
    error !== null &&
    "code" in error &&
    error.code === "EEXIST"
  );
}
