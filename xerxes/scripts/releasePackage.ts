// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHash } from "node:crypto";
import {
  access,
  chmod,
  copyFile,
  cp,
  lstat,
  mkdir,
  readFile,
  readdir,
  stat,
  writeFile,
} from "node:fs/promises";
import { dirname, isAbsolute, join, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

export const RELEASE_PACKAGE_FORMAT = 2;
export const RELEASE_PACKAGE_NAME = "xerxes-bun";
export const RELEASE_PACKAGE_MINIMUM_BUN_VERSION = "1.3.12";
export const RELEASE_REQUIRED_TUI_DEPENDENCIES = Object.freeze([
  "@opentui/core",
  "@opentui/react",
  "react",
] as const);

const RELEASE_PACKAGE_REPOSITORY = Object.freeze({
  type: "git",
  url: "git+https://github.com/erfanzar/Xerxes-Agents.git",
});
const RELEASE_PACKAGE_HOMEPAGE =
  "https://github.com/erfanzar/Xerxes-Agents#readme";
const RELEASE_PACKAGE_BUGS = Object.freeze({
  url: "https://github.com/erfanzar/Xerxes-Agents/issues",
});
const RELEASE_PACKAGE_KEYWORDS = Object.freeze([
  "ai",
  "agent",
  "coding-agent",
  "multi-agent",
  "bun",
  "typescript",
  "terminal",
  "tui",
  "opentui",
  "mcp",
] as const);

const REPOSITORY_ROOT = resolve(
  dirname(fileURLToPath(import.meta.url)),
  "../..",
);
const VERSION_PATTERN =
  /^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$/u;
const BUNDLED_SKILLS_DIRECTORY = "skills";
const BUNDLED_AGENTS_DIRECTORY = "bin/default";
const NON_REDISTRIBUTABLE_BUNDLED_SKILL_PREFIXES = Object.freeze([
  "creative-ideation",
  "research/research-paper-writing/templates",
  "training/axolotl",
  "training/pytorch-fsdp",
  "training/unsloth",
] as const);
const SOURCE_ARTIFACTS = Object.freeze([
  {
    destination: "bin/xerxes.js",
    kind: "file",
    source: "xerxes/dist/cli.js",
  },
  {
    destination: "ui/entry.js",
    kind: "file",
    source: "xerxes/dist/ui/entry.js",
  },
  { destination: "LICENSE", kind: "file", source: "LICENSE" },
  { destination: "README.md", kind: "file", source: "README.md" },
  {
    destination: "THIRD_PARTY_NOTICES.md",
    kind: "file",
    source: "THIRD_PARTY_NOTICES.md",
  },
  {
    destination: BUNDLED_SKILLS_DIRECTORY,
    kind: "directory",
    source: "xerxes/dist/skills",
  },
  {
    destination: BUNDLED_AGENTS_DIRECTORY,
    kind: "directory",
    source: "xerxes/dist/default",
  },
] as const);
const REQUIRED_RELEASE_FILES = Object.freeze([
  "LICENSE",
  "README.md",
  "THIRD_PARTY_NOTICES.md",
  "bin/xerxes",
  "bin/xerxes-acp",
  "bin/xerxes-bun",
  "bin/xerxes.js",
  "package.json",
  "ui/entry.js",
] as const);

interface SourceArtifact {
  readonly destination: string;
  readonly kind: "directory" | "file";
  readonly source: string;
}

export interface PrepareReleasePackageOptions {
  readonly expectedVersion?: string;
  readonly outputDirectory: string;
  readonly repositoryRoot?: string;
}

export interface ValidateReleasePackageOptions {
  readonly archivePath?: string;
  readonly expectedVersion?: string;
  readonly packageDirectory: string;
}

export interface DryRunReleasePublishOptions {
  readonly archivePath: string;
  readonly expectedVersion?: string;
  readonly packageDirectory: string;
}

export interface ReleasePackageFile {
  readonly path: string;
  readonly sha256: string;
  readonly size: number;
}

export interface ReleasePackageReport {
  readonly files: readonly ReleasePackageFile[];
  readonly packageDirectory: string;
  readonly version: string;
}

export interface ReleasePublishDryRunReport extends ReleasePackageReport {
  readonly output: string;
}

interface ReleasePackageManifest {
  readonly files: readonly ReleasePackageFile[];
  readonly format: number;
  readonly name: string;
  readonly version: string;
}

interface ParsedCommand {
  readonly archivePath: string | undefined;
  readonly expectedVersion: string | undefined;
  readonly kind: "check" | "prepare" | "publish-dry-run";
  readonly packageDirectory: string | undefined;
  readonly outputDirectory: string | undefined;
}

interface PackedArchiveFile {
  readonly mode: number;
  readonly path: string;
  readonly sha256: string;
  readonly size: number;
}

interface PendingTarPath {
  readonly path: string;
}

/** Prepare an installable Bun release package from already-built artifacts. */
export async function prepareReleasePackage(
  options: PrepareReleasePackageOptions,
): Promise<ReleasePackageReport> {
  const repositoryRoot = resolve(options.repositoryRoot ?? REPOSITORY_ROOT);
  const outputDirectory = resolve(options.outputDirectory);
  const version = await releaseVersion(repositoryRoot, options.expectedVersion);

  assertSafeOutputDirectory(repositoryRoot, outputDirectory);
  await ensureEmptyDirectory(outputDirectory);

  const bundledAssetFiles = await copyReleaseArtifacts(
    repositoryRoot,
    outputDirectory,
  );

  const launcherPath = join(outputDirectory, "bin/xerxes");
  await writeFile(
    launcherPath,
    "#!/usr/bin/env bun\nimport './xerxes.js'\n",
    "utf8",
  );
  await chmod(launcherPath, 0o755);
  const packageLauncherPath = join(outputDirectory, "bin/xerxes-bun");
  await writeFile(
    packageLauncherPath,
    "#!/usr/bin/env bun\nimport './xerxes.js'\n",
    "utf8",
  );
  await chmod(packageLauncherPath, 0o755);
  const acpLauncherPath = join(outputDirectory, "bin/xerxes-acp");
  await writeFile(
    acpLauncherPath,
    "#!/usr/bin/env bun\nprocess.argv.splice(2, 0, 'acp')\nawait import('./xerxes.js')\n",
    "utf8",
  );
  await chmod(acpLauncherPath, 0o755);

  const packagePath = join(outputDirectory, "package.json");
  await writeJson(packagePath, await packageMetadata(repositoryRoot, version));

  const files: ReleasePackageFile[] = [];
  for (const filePath of [...REQUIRED_RELEASE_FILES, ...bundledAssetFiles]) {
    files.push(await describeReleaseFile(outputDirectory, filePath));
  }
  files.sort(compareReleaseFiles);

  const manifest: ReleasePackageManifest = {
    files: Object.freeze(files),
    format: RELEASE_PACKAGE_FORMAT,
    name: RELEASE_PACKAGE_NAME,
    version,
  };
  await writeJson(join(outputDirectory, "release-manifest.json"), manifest);

  return validateReleasePackage({
    ...(options.expectedVersion === undefined
      ? {}
      : { expectedVersion: options.expectedVersion }),
    packageDirectory: outputDirectory,
  });
}

/** Validate staged native release files and an optional packed Bun tarball. */
export async function validateReleasePackage(
  options: ValidateReleasePackageOptions,
): Promise<ReleasePackageReport> {
  const packageDirectory = resolve(options.packageDirectory);
  await assertDirectory(
    packageDirectory,
    `Release package directory does not exist: ${packageDirectory}`,
  );

  const manifest = await readReleaseManifest(
    join(packageDirectory, "release-manifest.json"),
  );
  if (manifest.format !== RELEASE_PACKAGE_FORMAT) {
    throw new Error(
      `Unsupported release manifest format: ${String(manifest.format)}`,
    );
  }
  if (manifest.name !== RELEASE_PACKAGE_NAME) {
    throw new Error(`Unexpected release package name: ${manifest.name}`);
  }
  assertVersion(manifest.version, "Release manifest version");
  if (
    options.expectedVersion !== undefined &&
    normalizeVersion(options.expectedVersion) !== manifest.version
  ) {
    throw new Error(
      `Release manifest version ${manifest.version} does not match expected ${normalizeVersion(options.expectedVersion)}.`,
    );
  }

  const packageMetadataValue = await readJson(
    join(packageDirectory, "package.json"),
  );
  assertReleasePackageMetadata(packageMetadataValue, manifest.version);

  const expectedPaths = new Set<string>([
    ...REQUIRED_RELEASE_FILES,
    ...(await bundledSkillReleaseFiles(packageDirectory)),
    ...(await bundledAgentReleaseFiles(packageDirectory)),
  ]);
  const manifestPaths = new Set<string>();
  for (const file of manifest.files) {
    assertSafePackagePath(file.path);
    if (!expectedPaths.has(file.path))
      throw new Error(
        `Release manifest contains an unexpected file: ${file.path}`,
      );
    if (manifestPaths.has(file.path))
      throw new Error(`Release manifest contains duplicate file: ${file.path}`);
    manifestPaths.add(file.path);

    const absolutePath = join(packageDirectory, file.path);
    await assertRegularFile(
      absolutePath,
      `Release package file is missing: ${file.path}`,
    );
    const actual = await describeReleaseFile(packageDirectory, file.path);
    if (actual.sha256 !== file.sha256 || actual.size !== file.size) {
      throw new Error(`Release package file integrity mismatch: ${file.path}`);
    }
  }
  for (const expectedPath of expectedPaths) {
    if (!manifestPaths.has(expectedPath))
      throw new Error(
        `Release manifest is missing required file: ${expectedPath}`,
      );
  }

  for (const launcher of ["xerxes", "xerxes-acp", "xerxes-bun"]) {
    const launcherMode = (await stat(join(packageDirectory, "bin", launcher)))
      .mode;
    if ((launcherMode & 0o111) === 0)
      throw new Error(`Release launcher must be executable: bin/${launcher}`);
  }

  if (options.archivePath !== undefined) {
    await assertPackedArchive(options.archivePath, [
      ...manifest.files,
      await describeReleaseFile(packageDirectory, "release-manifest.json"),
    ]);
  }
  return Object.freeze({
    files: Object.freeze([...manifest.files]),
    packageDirectory,
    version: manifest.version,
  });
}

/** Validate an exact release archive, then ask Bun to simulate publishing it. */
export async function dryRunReleasePublish(
  options: DryRunReleasePublishOptions,
): Promise<ReleasePublishDryRunReport> {
  const archivePath = resolve(options.archivePath);
  const report = await validateReleasePackage({
    archivePath,
    ...(options.expectedVersion === undefined
      ? {}
      : { expectedVersion: options.expectedVersion }),
    packageDirectory: options.packageDirectory,
  });
  const child = Bun.spawn(
    [
      process.execPath,
      "publish",
      "--dry-run",
      "--ignore-scripts",
      "--no-progress",
      archivePath,
    ],
    {
      cwd: report.packageDirectory,
      env: publishDryRunEnvironment(),
      stderr: "pipe",
      stdout: "pipe",
    },
  );
  const [stdout, stderr, exitCode] = await Promise.all([
    new Response(child.stdout).text(),
    new Response(child.stderr).text(),
    child.exited,
  ]);
  const output = [stdout.trim(), stderr.trim()].filter(Boolean).join("\n");
  if (exitCode !== 0) {
    throw new Error(
      `Bun publish dry-run failed with exit code ${exitCode}: ${output || "no output"}`,
    );
  }
  return Object.freeze({ ...report, output });
}

/** Parse only the explicit release staging/check options accepted by this script. */
export function parseReleasePackageCommand(
  args: readonly string[],
): ParsedCommand | undefined {
  const [kind, ...rest] = args;
  if (kind === "--help" || kind === "-h" || kind === undefined) {
    return undefined;
  }
  if (kind !== "prepare" && kind !== "check" && kind !== "publish-dry-run")
    throw new Error(`Unknown release package command: ${kind}`);

  let archivePath: string | undefined;
  let expectedVersion: string | undefined;
  let outputDirectory: string | undefined;
  let packageDirectory: string | undefined;
  for (let index = 0; index < rest.length; index += 1) {
    const argument = rest[index];
    if (argument === undefined) continue;
    if (argument === "--help" || argument === "-h") return undefined;
    const value = rest[index + 1];
    if (value === undefined || value.startsWith("--"))
      throw new Error(`Option ${argument} requires a value.`);
    if (argument === "--archive") archivePath = value;
    else if (argument === "--expected-version") expectedVersion = value;
    else if (argument === "--output") outputDirectory = value;
    else if (argument === "--package") packageDirectory = value;
    else throw new Error(`Unknown release package option: ${argument}`);
    index += 1;
  }

  if (kind === "prepare" && outputDirectory === undefined) {
    throw new Error("prepare requires --output <directory>.");
  }
  if (kind !== "prepare" && packageDirectory === undefined) {
    throw new Error(`${kind} requires --package <directory>.`);
  }
  if (kind === "publish-dry-run" && archivePath === undefined) {
    throw new Error("publish-dry-run requires --archive <artifact.tgz>.");
  }
  if (
    kind === "prepare" &&
    (archivePath !== undefined || packageDirectory !== undefined)
  ) {
    throw new Error("prepare only accepts --output and --expected-version.");
  }
  if (kind !== "prepare" && outputDirectory !== undefined) {
    throw new Error(
      `${kind} only accepts --package, --archive, and --expected-version.`,
    );
  }
  return {
    archivePath,
    expectedVersion,
    kind,
    outputDirectory,
    packageDirectory,
  };
}

/** Run the release staging or validation command as a Bun executable. */
export async function main(
  args: readonly string[] = process.argv.slice(2),
): Promise<number> {
  try {
    const command = parseReleasePackageCommand(args);
    if (command === undefined) {
      console.log(usage());
      return 0;
    }

    const expectedVersion =
      command.expectedVersion === undefined
        ? {}
        : { expectedVersion: command.expectedVersion };
    if (command.kind === "prepare") {
      const report = await prepareReleasePackage({
        ...expectedVersion,
        outputDirectory: command.outputDirectory!,
      });
      console.log(
        `Prepared ${RELEASE_PACKAGE_NAME}@${report.version} in ${report.packageDirectory}`,
      );
      return 0;
    }

    const archivePath =
      command.archivePath === undefined
        ? {}
        : { archivePath: command.archivePath };
    if (command.kind === "publish-dry-run") {
      const report = await dryRunReleasePublish({
        archivePath: command.archivePath!,
        ...expectedVersion,
        packageDirectory: command.packageDirectory!,
      });
      if (report.output) console.log(report.output);
      console.log(
        `Dry-run verified ${RELEASE_PACKAGE_NAME}@${report.version} from ${command.archivePath}`,
      );
      return 0;
    }
    const report = await validateReleasePackage({
      ...archivePath,
      ...expectedVersion,
      packageDirectory: command.packageDirectory!,
    });
    console.log(
      `Validated ${RELEASE_PACKAGE_NAME}@${report.version} in ${report.packageDirectory}`,
    );
    return 0;
  } catch (error) {
    console.error(`release-package: ${errorMessage(error)}`);
    return 1;
  }
}

async function releaseVersion(
  repositoryRoot: string,
  expectedVersion: string | undefined,
): Promise<string> {
  const packagePaths = ["package.json", "xerxes/package.json"];
  const versions = await Promise.all(
    packagePaths.map(async (packagePath) => {
      const value = await readJson(join(repositoryRoot, packagePath));
      const version = recordString(
        assertRecord(value, `Package manifest ${packagePath}`),
        "version",
        `Package manifest ${packagePath}`,
      );
      assertVersion(version, `Package manifest ${packagePath} version`);
      return version;
    }),
  );
  const version = versions[0];
  if (
    version === undefined ||
    versions.some((candidate) => candidate !== version)
  ) {
    throw new Error(
      `Native package versions must match before release: ${versions.join(", ")}`,
    );
  }
  if (
    expectedVersion !== undefined &&
    normalizeVersion(expectedVersion) !== version
  ) {
    throw new Error(
      `Package version ${version} does not match expected ${normalizeVersion(expectedVersion)}.`,
    );
  }
  return version;
}

async function packageMetadata(
  repositoryRoot: string,
  version: string,
): Promise<Readonly<Record<string, unknown>>> {
  const runtimeManifest = assertRecord(
    await readJson(join(repositoryRoot, "xerxes/package.json")),
    "Runtime package manifest",
  );
  const runtimeDependencies = assertRecord(
    runtimeManifest.dependencies,
    "Runtime package manifest dependencies",
  );
  const dependencies: Record<string, string> = {};
  // The TUI bundle deliberately keeps only React and OpenTUI external. Other
  // source dependencies (for example nanostores) are bundled into ui/entry.js
  // and must not inflate an installed release package.
  for (const packageName of RELEASE_REQUIRED_TUI_DEPENDENCIES) {
    const specifier = recordString(
      runtimeDependencies,
      packageName,
      "Runtime package manifest dependencies",
    ).trim();
    if (!specifier)
      throw new Error(`TUI dependency ${packageName} must not be empty.`);
    dependencies[packageName] = specifier;
  }

  return Object.freeze({
    bin: Object.freeze({
      xerxes: "./bin/xerxes",
      "xerxes-bun": "./bin/xerxes-bun",
      "xerxes-acp": "./bin/xerxes-acp",
    }),
    bugs: RELEASE_PACKAGE_BUGS,
    description:
      "Bun-native TypeScript distribution of the Xerxes agent runtime.",
    dependencies: Object.freeze(dependencies),
    engines: Object.freeze({ bun: `>=${RELEASE_PACKAGE_MINIMUM_BUN_VERSION}` }),
    files: Object.freeze([
      "bin",
      BUNDLED_SKILLS_DIRECTORY,
      "ui",
      "LICENSE",
      "README.md",
      "THIRD_PARTY_NOTICES.md",
      "release-manifest.json",
    ]),
    homepage: RELEASE_PACKAGE_HOMEPAGE,
    keywords: RELEASE_PACKAGE_KEYWORDS,
    license: "Apache-2.0",
    name: RELEASE_PACKAGE_NAME,
    private: false,
    publishConfig: Object.freeze({ access: "public" }),
    repository: RELEASE_PACKAGE_REPOSITORY,
    type: "module",
    version,
  });
}

async function copyReleaseArtifacts(
  repositoryRoot: string,
  outputDirectory: string,
): Promise<readonly string[]> {
  const bundledAssetFiles: string[] = [];
  for (const artifact of SOURCE_ARTIFACTS) {
    const copiedSkillFiles = await copyReleaseArtifact(
      repositoryRoot,
      outputDirectory,
      artifact,
    );
    bundledAssetFiles.push(...copiedSkillFiles);
  }
  return bundledAssetFiles;
}

async function copyReleaseArtifact(
  repositoryRoot: string,
  outputDirectory: string,
  artifact: SourceArtifact,
): Promise<readonly string[]> {
  assertSafePackagePath(artifact.destination);
  const source = join(repositoryRoot, artifact.source);
  const destination = join(outputDirectory, artifact.destination);
  const missingMessage = `Built release input is missing: ${artifact.source}. Run "bun run build" first.`;

  if (artifact.kind === "file") {
    await assertRegularFile(source, missingMessage);
    await mkdir(dirname(destination), { recursive: true });
    await copyFile(source, destination);
    return [];
  }

  if (
    artifact.destination !== BUNDLED_SKILLS_DIRECTORY &&
    artifact.destination !== BUNDLED_AGENTS_DIRECTORY
  ) {
    throw new Error(
      `Unsupported release directory artifact: ${artifact.destination}`,
    );
  }
  const sourceFiles = await regularDirectoryFiles(source, missingMessage);
  if (artifact.destination === BUNDLED_SKILLS_DIRECTORY) {
    assertBundledSkillFiles(
      sourceFiles,
      `Built release input ${artifact.source}`,
    );
  } else {
    assertBundledAgentFiles(
      sourceFiles,
      `Built release input ${artifact.source}`,
    );
  }
  await cp(source, destination, { recursive: true });
  const copiedFiles = await regularDirectoryFiles(
    destination,
    `Release package directory is missing: ${artifact.destination}`,
  );
  if (!samePaths(sourceFiles, copiedFiles)) {
    throw new Error(`Bundled asset copy is incomplete: ${artifact.source}`);
  }
  return sourceFiles.map((path) => `${artifact.destination}/${path}`);
}

async function bundledSkillReleaseFiles(
  packageDirectory: string,
): Promise<readonly string[]> {
  const skillsDirectory = join(packageDirectory, BUNDLED_SKILLS_DIRECTORY);
  const files = await regularDirectoryFiles(
    skillsDirectory,
    `Release package bundled skill directory is missing: ${BUNDLED_SKILLS_DIRECTORY}`,
  );
  assertBundledSkillFiles(files, "Release package bundled skills");
  return files.map((path) => `${BUNDLED_SKILLS_DIRECTORY}/${path}`);
}

async function bundledAgentReleaseFiles(
  packageDirectory: string,
): Promise<readonly string[]> {
  const agentsDirectory = join(packageDirectory, BUNDLED_AGENTS_DIRECTORY);
  const files = await regularDirectoryFiles(
    agentsDirectory,
    `Release package bundled agent directory is missing: ${BUNDLED_AGENTS_DIRECTORY}`,
  );
  assertBundledAgentFiles(files, "Release package bundled agents");
  return files.map((path) => `${BUNDLED_AGENTS_DIRECTORY}/${path}`);
}

async function regularDirectoryFiles(
  directory: string,
  missingMessage: string,
): Promise<string[]> {
  await assertDirectory(directory, missingMessage);
  const files: string[] = [];
  await collectRegularDirectoryFiles(directory, "", files, missingMessage);
  return files.sort();
}

async function collectRegularDirectoryFiles(
  directory: string,
  prefix: string,
  files: string[],
  invalidEntryMessage: string,
): Promise<void> {
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const relativePath =
      prefix.length === 0 ? entry.name : `${prefix}/${entry.name}`;
    assertSafePackagePath(relativePath);
    const path = join(directory, entry.name);
    const value = await lstat(path);
    if (value.isSymbolicLink() || (!value.isDirectory() && !value.isFile())) {
      throw new Error(`${invalidEntryMessage}: ${relativePath}`);
    }
    if (value.isDirectory()) {
      await collectRegularDirectoryFiles(
        path,
        relativePath,
        files,
        invalidEntryMessage,
      );
      continue;
    }
    files.push(relativePath);
  }
}

function assertBundledSkillFiles(
  files: readonly string[],
  label: string,
): void {
  const nonRedistributablePath = files.find(
    (path) =>
      (path.startsWith("powerpoint/") && path !== "powerpoint/SKILL.md") ||
      NON_REDISTRIBUTABLE_BUNDLED_SKILL_PREFIXES.some(
        (prefix) => path === prefix || path.startsWith(`${prefix}/`),
      ),
  );
  if (nonRedistributablePath !== undefined) {
    throw new Error(
      `${label} contains a non-redistributable bundled skill asset: ${nonRedistributablePath}`,
    );
  }
  if (
    !files.some((path) => path === "SKILL.md" || path.endsWith("/SKILL.md"))
  ) {
    throw new Error(`${label} must contain at least one SKILL.md file.`);
  }
}

function assertBundledAgentFiles(
  files: readonly string[],
  label: string,
): void {
  const required = ["agent.yaml", "system.md"];
  for (const path of required) {
    if (!files.includes(path)) {
      throw new Error(`${label} must contain ${path}.`);
    }
  }
  const unsupported = files.find(
    (path) => !path.endsWith(".yaml") && !path.endsWith(".md"),
  );
  if (unsupported !== undefined) {
    throw new Error(`${label} contains an unsupported asset: ${unsupported}`);
  }
}

function samePaths(left: readonly string[], right: readonly string[]): boolean {
  return (
    left.length === right.length &&
    left.every((path, index) => path === right[index])
  );
}

async function describeReleaseFile(
  packageDirectory: string,
  packagePath: string,
): Promise<ReleasePackageFile> {
  assertSafePackagePath(packagePath);
  const absolutePath = join(packageDirectory, packagePath);
  await assertRegularFile(
    absolutePath,
    `Release package file is missing: ${packagePath}`,
  );
  const file = await stat(absolutePath);
  return Object.freeze({
    path: packagePath,
    sha256: createHash("sha256")
      .update(await readFile(absolutePath))
      .digest("hex"),
    size: file.size,
  });
}

async function readReleaseManifest(
  path: string,
): Promise<ReleasePackageManifest> {
  const value = await readJson(path);
  const record = assertRecord(value, "Release manifest");
  const format = recordNumber(record, "format", "Release manifest");
  const name = recordString(record, "name", "Release manifest");
  const version = recordString(record, "version", "Release manifest");
  const rawFiles = record.files;
  if (!Array.isArray(rawFiles))
    throw new Error("Release manifest files must be an array.");
  const files = rawFiles.map((value, index) => {
    const file = assertRecord(value, `Release manifest file ${index}`);
    return Object.freeze({
      path: recordString(file, "path", `Release manifest file ${index}`),
      sha256: recordString(file, "sha256", `Release manifest file ${index}`),
      size: recordNumber(file, "size", `Release manifest file ${index}`),
    });
  });
  return Object.freeze({ files: Object.freeze(files), format, name, version });
}

function assertReleasePackageMetadata(value: unknown, version: string): void {
  const metadata = assertRecord(value, "Release package metadata");
  if (
    recordString(metadata, "name", "Release package metadata") !==
    RELEASE_PACKAGE_NAME
  ) {
    throw new Error("Release package metadata has an unexpected name.");
  }
  if (
    recordString(metadata, "version", "Release package metadata") !== version
  ) {
    throw new Error(
      "Release package metadata version does not match release manifest.",
    );
  }
  if (
    recordString(metadata, "license", "Release package metadata") !==
    "Apache-2.0"
  ) {
    throw new Error("Release package metadata must declare Apache-2.0.");
  }
  if (metadata.private !== false)
    throw new Error("Release package metadata must be publishable.");
  if (recordString(metadata, "type", "Release package metadata") !== "module")
    throw new Error("Release package metadata type must be module.");
  if (!recordString(metadata, "description", "Release package metadata").trim())
    throw new Error("Release package metadata description must not be empty.");

  const repository = assertRecord(
    metadata.repository,
    "Release package metadata repository",
  );
  if (
    recordString(repository, "type", "Release package metadata repository") !==
      RELEASE_PACKAGE_REPOSITORY.type ||
    recordString(repository, "url", "Release package metadata repository") !==
      RELEASE_PACKAGE_REPOSITORY.url
  ) {
    throw new Error("Release package metadata repository is incorrect.");
  }
  if (
    recordString(metadata, "homepage", "Release package metadata") !==
    RELEASE_PACKAGE_HOMEPAGE
  ) {
    throw new Error("Release package metadata homepage is incorrect.");
  }
  const bugs = assertRecord(metadata.bugs, "Release package metadata bugs");
  if (
    recordString(bugs, "url", "Release package metadata bugs") !==
    RELEASE_PACKAGE_BUGS.url
  ) {
    throw new Error("Release package metadata bugs URL is incorrect.");
  }

  const publishConfig = assertRecord(
    metadata.publishConfig,
    "Release package metadata publishConfig",
  );
  if (
    recordString(
      publishConfig,
      "access",
      "Release package metadata publishConfig",
    ) !== "public"
  ) {
    throw new Error(
      "Release package metadata must publish with public access.",
    );
  }

  const engines = assertRecord(
    metadata.engines,
    "Release package metadata engines",
  );
  if (
    recordString(engines, "bun", "Release package metadata engines") !==
    `>=${RELEASE_PACKAGE_MINIMUM_BUN_VERSION}`
  ) {
    throw new Error(
      `Release package metadata must require Bun >=${RELEASE_PACKAGE_MINIMUM_BUN_VERSION}.`,
    );
  }

  assertExactStringArray(
    metadata.keywords,
    RELEASE_PACKAGE_KEYWORDS,
    "Release package metadata keywords",
  );

  const dependencies = assertRecord(
    metadata.dependencies,
    "Release package metadata dependencies",
  );
  for (const packageName of RELEASE_REQUIRED_TUI_DEPENDENCIES) {
    if (
      !recordString(
        dependencies,
        packageName,
        "Release package metadata dependencies",
      ).trim()
    ) {
      throw new Error(
        `Release package runtime dependency ${packageName} must not be empty.`,
      );
    }
  }

  assertExactStringArray(
    metadata.files,
    [
      "bin",
      BUNDLED_SKILLS_DIRECTORY,
      "ui",
      "LICENSE",
      "README.md",
      "THIRD_PARTY_NOTICES.md",
      "release-manifest.json",
    ],
    "Release package metadata files",
  );

  const bin = assertRecord(metadata.bin, "Release package metadata bin");
  if (
    recordString(bin, "xerxes", "Release package metadata bin") !==
    "./bin/xerxes"
  ) {
    throw new Error("Release package launcher must be ./bin/xerxes.");
  }
  if (
    recordString(bin, "xerxes-bun", "Release package metadata bin") !==
    "./bin/xerxes-bun"
  ) {
    throw new Error("Release package name launcher must be ./bin/xerxes-bun.");
  }
  if (
    recordString(bin, "xerxes-acp", "Release package metadata bin") !==
    "./bin/xerxes-acp"
  ) {
    throw new Error("Release package ACP launcher must be ./bin/xerxes-acp.");
  }
}

function assertExactStringArray(
  value: unknown,
  expected: readonly string[],
  label: string,
): void {
  if (
    !Array.isArray(value) ||
    !value.every((entry) => typeof entry === "string") ||
    value.length !== expected.length ||
    value.some((entry, index) => entry !== expected[index])
  ) {
    throw new Error(`${label} is incorrect.`);
  }
}

function publishDryRunEnvironment(): Record<string, string | undefined> {
  const environment = { ...process.env };
  delete environment.BUN_AUTH_TOKEN;
  delete environment.NODE_AUTH_TOKEN;
  delete environment.NPM_TOKEN;
  environment.NPM_CONFIG_DRY_RUN = "true";
  // Bun requires an auth-shaped value even for --dry-run. A fixed invalid token
  // keeps this command deterministic and makes an accidental real publish fail.
  environment.NPM_CONFIG_TOKEN = "xerxes-bun-dry-run-not-a-credential";
  return environment;
}

async function assertPackedArchive(
  path: string,
  expectedFiles: readonly ReleasePackageFile[],
): Promise<void> {
  if (!path.endsWith(".tgz"))
    throw new Error(`Release archive must use a .tgz extension: ${path}`);
  await assertRegularFile(path, `Packed release archive is missing: ${path}`);
  if ((await stat(path)).size === 0)
    throw new Error(`Packed release archive is empty: ${path}`);

  let tar: Uint8Array;
  try {
    tar = Bun.gunzipSync(await readFile(path));
  } catch (error) {
    throw new Error(
      `Packed release archive is not a valid gzip stream: ${errorMessage(error)}`,
    );
  }

  const packedFiles = readPackedArchiveFiles(tar);
  const expectedByPath = new Map(
    expectedFiles.map((file) => [file.path, file] as const),
  );
  const packedByPath = new Map(
    packedFiles.map((file) => [file.path, file] as const),
  );

  for (const file of expectedFiles) {
    const packed = packedByPath.get(file.path);
    if (packed === undefined) {
      throw new Error(
        `Packed release archive is missing required file: ${file.path}`,
      );
    }
    if (packed.size !== file.size || packed.sha256 !== file.sha256) {
      throw new Error(
        `Packed release archive file integrity mismatch: ${file.path}`,
      );
    }
  }
  for (const file of packedFiles) {
    if (!expectedByPath.has(file.path)) {
      throw new Error(
        `Packed release archive contains unexpected file: ${file.path}`,
      );
    }
  }
  for (const launcher of ["bin/xerxes", "bin/xerxes-acp", "bin/xerxes-bun"]) {
    if (((packedByPath.get(launcher)?.mode ?? 0) & 0o111) === 0) {
      throw new Error(
        `Packed release archive launcher must be executable: ${launcher}`,
      );
    }
  }
}

const TAR_BLOCK_SIZE = 512;
const TAR_TEXT = new TextDecoder();

function readPackedArchiveFiles(tar: Uint8Array): PackedArchiveFile[] {
  const files: PackedArchiveFile[] = [];
  const seenPaths = new Set<string>();
  let offset = 0;
  let terminated = false;
  let pendingPath: PendingTarPath | undefined;

  while (offset + TAR_BLOCK_SIZE <= tar.byteLength) {
    const header = tar.subarray(offset, offset + TAR_BLOCK_SIZE);
    if (header.every((byte) => byte === 0)) {
      terminated = true;
      for (const byte of tar.subarray(offset)) {
        if (byte !== 0) {
          throw new Error(
            "Packed release archive contains data after its tar terminator.",
          );
        }
      }
      break;
    }

    assertTarHeaderChecksum(header);
    const headerPath = tarHeaderPath(header);
    const size = tarHeaderNumber(header.subarray(124, 136), "size");
    const mode = tarHeaderNumber(header.subarray(100, 108), "mode");
    const type = header[156] ?? 0;
    const bodyStart = offset + TAR_BLOCK_SIZE;
    const bodyEnd = bodyStart + size;
    if (bodyEnd > tar.byteLength) {
      throw new Error(
        `Packed release archive entry is truncated: ${headerPath}`,
      );
    }
    const body = tar.subarray(bodyStart, bodyEnd);
    offset = bodyStart + Math.ceil(size / TAR_BLOCK_SIZE) * TAR_BLOCK_SIZE;

    if (type === 120) {
      if (pendingPath !== undefined) {
        throw new Error(
          "Packed release archive contains stacked path metadata.",
        );
      }
      const paxPath = readPaxPath(body);
      if (paxPath === undefined) {
        throw new Error(
          "Packed release archive PAX metadata does not define a path.",
        );
      }
      pendingPath = { path: paxPath };
      continue;
    }
    if (type === 76) {
      if (pendingPath !== undefined) {
        throw new Error(
          "Packed release archive contains stacked path metadata.",
        );
      }
      const longPath = tarText(body).replace(/[\0\n]+$/gu, "");
      if (!longPath) {
        throw new Error(
          "Packed release archive contains an empty GNU long path.",
        );
      }
      pendingPath = { path: longPath };
      continue;
    }

    const archivePath = pendingPath?.path ?? headerPath;
    pendingPath = undefined;
    if (type === 53) {
      assertPackedArchivePath(archivePath, true);
      continue;
    }
    if (type !== 0 && type !== 48) {
      throw new Error(
        `Packed release archive contains unsupported entry type ${String.fromCharCode(type)}: ${archivePath}`,
      );
    }

    const packagePath = assertPackedArchivePath(archivePath, false);
    if (seenPaths.has(packagePath)) {
      throw new Error(
        `Packed release archive contains duplicate file: ${packagePath}`,
      );
    }
    seenPaths.add(packagePath);
    files.push(
      Object.freeze({
        mode,
        path: packagePath,
        sha256: createHash("sha256").update(body).digest("hex"),
        size,
      }),
    );
  }

  if (!terminated) {
    throw new Error("Packed release archive is missing its tar terminator.");
  }
  if (pendingPath !== undefined) {
    throw new Error(
      "Packed release archive ends with unresolved path metadata.",
    );
  }
  return files.sort((left, right) => left.path.localeCompare(right.path));
}

function assertPackedArchivePath(path: string, directory: boolean): string {
  if (path.includes("\\")) {
    throw new Error(
      `Packed release archive path must use forward slashes: ${path}`,
    );
  }
  if (!directory && path.endsWith("/")) {
    throw new Error(
      `Packed release archive file path must not end with a slash: ${path}`,
    );
  }
  const normalized = path.replace(/\/+$/u, "");
  if (directory && normalized === "package") return "";
  if (!normalized.startsWith("package/")) {
    throw new Error(
      `Packed release archive entry must be under package/: ${path}`,
    );
  }
  const packagePath = normalized.slice("package/".length);
  assertSafePackagePath(packagePath);
  return packagePath;
}

function assertTarHeaderChecksum(header: Uint8Array): void {
  const expected = tarHeaderNumber(header.subarray(148, 156), "checksum");
  let actual = 0;
  for (let index = 0; index < header.byteLength; index += 1) {
    actual += index >= 148 && index < 156 ? 32 : (header[index] ?? 0);
  }
  if (actual !== expected) {
    throw new Error(
      "Packed release archive contains an invalid tar header checksum.",
    );
  }
}

function tarHeaderPath(header: Uint8Array): string {
  const name = tarText(header.subarray(0, 100));
  const prefix = tarText(header.subarray(345, 500));
  return prefix ? `${prefix}/${name}` : name;
}

function tarHeaderNumber(field: Uint8Array, label: string): number {
  if ((field[0] ?? 0) >= 128) {
    throw new Error(
      `Packed release archive uses an unsupported binary tar ${label}.`,
    );
  }
  const value = tarText(field).trim();
  if (!value) return 0;
  if (!/^[0-7]+$/u.test(value)) {
    throw new Error(`Packed release archive has an invalid tar ${label}.`);
  }
  const parsed = Number.parseInt(value, 8);
  if (!Number.isSafeInteger(parsed) || parsed < 0) {
    throw new Error(`Packed release archive tar ${label} is out of range.`);
  }
  return parsed;
}

function tarText(value: Uint8Array): string {
  const terminator = value.indexOf(0);
  return TAR_TEXT.decode(
    terminator === -1 ? value : value.subarray(0, terminator),
  );
}

function readPaxPath(body: Uint8Array): string | undefined {
  let offset = 0;
  let path: string | undefined;
  while (offset < body.byteLength) {
    const space = body.indexOf(32, offset);
    if (space === -1) {
      throw new Error("Packed release archive has malformed PAX metadata.");
    }
    const lengthText = TAR_TEXT.decode(body.subarray(offset, space));
    if (!/^[1-9]\d*$/u.test(lengthText)) {
      throw new Error(
        "Packed release archive has malformed PAX record length.",
      );
    }
    const length = Number.parseInt(lengthText, 10);
    const end = offset + length;
    if (
      !Number.isSafeInteger(length) ||
      end > body.byteLength ||
      end <= space + 1
    ) {
      throw new Error("Packed release archive has truncated PAX metadata.");
    }
    const record = TAR_TEXT.decode(body.subarray(space + 1, end));
    if (!record.endsWith("\n")) {
      throw new Error("Packed release archive has malformed PAX metadata.");
    }
    const separator = record.indexOf("=");
    if (separator <= 0) {
      throw new Error("Packed release archive has malformed PAX metadata.");
    }
    const key = record.slice(0, separator);
    if (key !== "path") {
      throw new Error(
        `Packed release archive uses unsupported PAX metadata: ${key}`,
      );
    }
    if (path !== undefined) {
      throw new Error(
        "Packed release archive contains duplicate PAX path metadata.",
      );
    }
    path = record.slice(separator + 1, -1);
    offset = end;
  }
  return path;
}

function assertSafeOutputDirectory(
  repositoryRoot: string,
  outputDirectory: string,
): void {
  if (isSameOrDescendant(outputDirectory, repositoryRoot)) {
    throw new Error(
      `Release output directory cannot contain the repository root: ${outputDirectory}`,
    );
  }
  const protectedDirectories = [
    join(repositoryRoot, "xerxes"),
    join(repositoryRoot, ".git"),
    join(repositoryRoot, "node_modules"),
  ];
  for (const protectedDirectory of protectedDirectories) {
    if (isSameOrDescendant(outputDirectory, protectedDirectory)) {
      throw new Error(
        `Release output directory cannot contain source or metadata: ${outputDirectory}`,
      );
    }
  }
}

async function ensureEmptyDirectory(path: string): Promise<void> {
  try {
    const existing = await lstat(path);
    if (existing.isSymbolicLink() || !existing.isDirectory()) {
      throw new Error(`Release output path must be a real directory: ${path}`);
    }
    if ((await readdir(path)).length > 0)
      throw new Error(`Release output directory must be empty: ${path}`);
  } catch (error) {
    if (errorCode(error) !== "ENOENT") throw error;
    await mkdir(path, { recursive: true });
  }
}

async function assertRegularFile(path: string, message: string): Promise<void> {
  try {
    const value = await lstat(path);
    if (!value.isFile() || value.isSymbolicLink()) throw new Error(message);
    await access(path);
  } catch (error) {
    if (error instanceof Error && error.message === message) throw error;
    throw new Error(message);
  }
}

async function assertDirectory(path: string, message: string): Promise<void> {
  try {
    const value = await lstat(path);
    if (!value.isDirectory() || value.isSymbolicLink())
      throw new Error(message);
  } catch (error) {
    if (error instanceof Error && error.message === message) throw error;
    throw new Error(message);
  }
}

function assertSafePackagePath(path: string): void {
  if (
    path.length === 0 ||
    isAbsolute(path) ||
    path.split(/[\\/]/u).some((part) => part === ".." || part.length === 0)
  ) {
    throw new Error(`Unsafe release package path: ${path}`);
  }
}

function normalizeVersion(value: string): string {
  const normalized = value.startsWith("v") ? value.slice(1) : value;
  assertVersion(normalized, "Expected release version");
  return normalized;
}

function assertVersion(value: string, label: string): void {
  if (!VERSION_PATTERN.test(value))
    throw new Error(`${label} must be a semantic version, received: ${value}`);
}

function isSameOrDescendant(parent: string, candidate: string): boolean {
  const path = relative(parent, candidate);
  return (
    path === "" ||
    (!path.startsWith(`..${sep}`) && path !== ".." && !isAbsolute(path))
  );
}

async function readJson(path: string): Promise<unknown> {
  try {
    return JSON.parse(await readFile(path, "utf8")) as unknown;
  } catch (error) {
    throw new Error(`Cannot read JSON file ${path}: ${errorMessage(error)}`);
  }
}

async function writeJson(path: string, value: unknown): Promise<void> {
  await writeFile(path, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

function assertRecord(
  value: unknown,
  label: string,
): Readonly<Record<string, unknown>> {
  if (value === null || typeof value !== "object" || Array.isArray(value))
    throw new Error(`${label} must be an object.`);
  return value as Readonly<Record<string, unknown>>;
}

function recordString(
  value: Readonly<Record<string, unknown>>,
  key: string,
  label: string,
): string {
  const field = value[key];
  if (typeof field !== "string")
    throw new Error(`${label} ${key} must be a string.`);
  return field;
}

function recordNumber(
  value: Readonly<Record<string, unknown>>,
  key: string,
  label: string,
): number {
  const field = value[key];
  if (typeof field !== "number" || !Number.isSafeInteger(field) || field < 0) {
    throw new Error(`${label} ${key} must be a non-negative integer.`);
  }
  return field;
}

function errorCode(error: unknown): string | undefined {
  if (error === null || typeof error !== "object" || !("code" in error))
    return undefined;
  const code = error.code;
  return typeof code === "string" ? code : undefined;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function compareReleaseFiles(
  left: ReleasePackageFile,
  right: ReleasePackageFile,
): number {
  return left.path.localeCompare(right.path);
}

function usage(): string {
  return [
    "Usage:",
    "  bun releasePackage.ts prepare --output <directory> [--expected-version <version-or-vtag>]",
    "  bun releasePackage.ts check --package <directory> [--archive <artifact.tgz>] [--expected-version <version-or-vtag>]",
    "  bun releasePackage.ts publish-dry-run --package <directory> --archive <artifact.tgz> [--expected-version <version-or-vtag>]",
  ].join("\n");
}

if (import.meta.main) process.exitCode = await main();
