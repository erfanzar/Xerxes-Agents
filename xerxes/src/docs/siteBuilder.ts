// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import {
  mkdir,
  mkdtemp,
  readFile,
  readdir,
  rm,
  stat,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import {
  basename,
  dirname,
  extname,
  isAbsolute,
  join,
  posix,
  relative,
  resolve,
  sep,
} from "node:path";
import { fileURLToPath } from "node:url";

import { generateTypeScriptApiDocs } from "../maintenance/apiDocsGenerator.js";

export const BUN_DOCS_CONFIG_FILENAME = "bun-docs.json";
export const BUN_DOCS_MANIFEST_FILENAME = ".xerxes-bun-docs.json";
export const BUN_DOCS_BUILD_REPORT_FILENAME = "build-report.json";
export const BUN_DOCS_GENERATOR_NAME = "xerxes-bun-docs";
export const BUN_DOCS_MANIFEST_VERSION = 1;

const PACKAGE_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const WORKSPACE_ROOT = resolve(PACKAGE_ROOT, "..");
const DEFAULT_CONFIG_PATH = join(
  WORKSPACE_ROOT,
  "docs",
  BUN_DOCS_CONFIG_FILENAME,
);
const DOCUMENT_EXTENSIONS = new Set([".md"]);
const IGNORED_SOURCE_DIRECTORIES = new Set([
  ".git",
  "_build",
  "_bun",
  "_static",
  "node_modules",
  "typescript-api",
]);

export type BunDocsChangeKind = "created" | "deleted" | "unchanged" | "updated";
export type BunDocsDiagnosticLevel = "warning";
export type BunDocsSourceKind = "markdown" | "typescript-api";

export interface BunDocsDiagnostic {
  readonly code: string;
  readonly level: BunDocsDiagnosticLevel;
  readonly line?: number;
  readonly message: string;
  readonly source: string;
}

export interface BunDocsChange {
  readonly action: BunDocsChangeKind;
  readonly path: string;
}

export interface BunDocsDocument {
  readonly kind: BunDocsSourceKind;
  readonly outputPath: string;
  readonly sourcePath: string;
  readonly title: string;
}

export interface BunDocsBuildOptions {
  readonly apiPackageName?: string;
  readonly apiSourceDirectory?: string;
  readonly configPath?: string;
  readonly dryRun?: boolean;
  readonly outputDirectory?: string;
  readonly sourceDirectory?: string;
  readonly title?: string;
}

export interface BunDocsBuildResult {
  readonly changed: boolean;
  readonly changes: readonly BunDocsChange[];
  readonly diagnostics: readonly BunDocsDiagnostic[];
  readonly documents: readonly BunDocsDocument[];
  readonly dryRun: boolean;
  readonly outputDirectory: string;
  readonly typeScriptApiModules: number;
}

export interface ResolvedBunDocsConfiguration {
  readonly apiOutputDirectory: string;
  readonly apiPackageName: string;
  readonly apiSourceDirectory: string;
  readonly configPath: string;
  readonly outputDirectory: string;
  readonly sourceDirectory: string;
  readonly staticDirectory?: string;
  readonly title: string;
}

interface RawBunDocsConfiguration {
  readonly outputDirectory: string;
  readonly sourceDirectory: string;
  readonly staticDirectory?: string;
  readonly title: string;
  readonly typescriptApi: {
    readonly outputDirectory: string;
    readonly packageName: string;
    readonly sourceDirectory: string;
  };
  readonly version: number;
}

interface SourceDocument extends BunDocsDocument {
  readonly content: string;
}

interface OwnedManifest {
  readonly files: readonly string[];
  readonly generator: typeof BUN_DOCS_GENERATOR_NAME;
  readonly version: typeof BUN_DOCS_MANIFEST_VERSION;
}

interface RenderedOutput {
  readonly bytes?: Uint8Array;
  readonly text?: string;
}

/** Error raised when a native Bun documentation build cannot safely proceed. */
export class BunDocsBuildError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "BunDocsBuildError";
  }
}

/** Load and resolve a declarative native Bun documentation configuration. */
export async function resolveBunDocsConfiguration(
  options: BunDocsBuildOptions = {},
): Promise<ResolvedBunDocsConfiguration> {
  const configPath = resolve(options.configPath ?? DEFAULT_CONFIG_PATH);
  const configDirectory = dirname(configPath);
  const raw = await readConfiguration(configPath);

  const sourceDirectory = resolveOptionPath(
    configDirectory,
    options.sourceDirectory ?? raw.sourceDirectory,
  );
  const outputDirectory =
    options.outputDirectory === undefined
      ? resolveOptionPath(configDirectory, raw.outputDirectory)
      : resolve(options.outputDirectory);
  const staticDirectory =
    raw.staticDirectory === undefined
      ? undefined
      : resolveOptionPath(sourceDirectory, raw.staticDirectory);
  const apiSourceDirectory =
    options.apiSourceDirectory === undefined
      ? resolveOptionPath(configDirectory, raw.typescriptApi.sourceDirectory)
      : resolve(options.apiSourceDirectory);
  const apiOutputDirectory = validateOutputDirectory(
    raw.typescriptApi.outputDirectory,
  );

  if (sourceDirectory === outputDirectory) {
    throw new BunDocsBuildError(
      "Documentation source and output directories must be different.",
    );
  }
  if (isPathInside(sourceDirectory, outputDirectory)) {
    throw new BunDocsBuildError(
      `Documentation output cannot contain its source directory: ${outputDirectory}`,
    );
  }

  return {
    apiOutputDirectory,
    apiPackageName: options.apiPackageName ?? raw.typescriptApi.packageName,
    apiSourceDirectory,
    configPath,
    outputDirectory,
    sourceDirectory,
    ...(staticDirectory === undefined ? {} : { staticDirectory }),
    title: options.title ?? raw.title,
  };
}

/**
 * Build a deterministic static documentation site with Bun and TypeScript only.
 *
 * The builder reads Markdown and generated TypeScript API pages only. It never launches a
 * second documentation runtime or executes source content during a build.
 */
export async function buildBunDocsSite(
  options: BunDocsBuildOptions = {},
): Promise<BunDocsBuildResult> {
  const configuration = await resolveBunDocsConfiguration(options);
  const dryRun = options.dryRun ?? false;
  const diagnostics: BunDocsDiagnostic[] = [];

  const sourceDocuments = await discoverSourceDocuments(
    configuration,
    diagnostics,
  );
  const temporaryApiDirectory = await mkdtemp(
    join(tmpdir(), "xerxes-bun-docs-api-"),
  );
  let apiDocuments: SourceDocument[] = [];
  let typeScriptApiModules = 0;
  try {
    const apiResult = await generateTypeScriptApiDocs({
      clean: true,
      outputDirectory: temporaryApiDirectory,
      packageName: configuration.apiPackageName,
      sourceDirectory: configuration.apiSourceDirectory,
    });
    typeScriptApiModules = apiResult.modules.length;
    apiDocuments = await discoverGeneratedApiDocuments(
      temporaryApiDirectory,
      configuration.apiOutputDirectory,
    );
  } finally {
    await rm(temporaryApiDirectory, { force: true, recursive: true });
  }

  const documents = [...sourceDocuments, ...apiDocuments].sort(
    compareDocuments,
  );
  assertUniqueOutputPaths(documents);
  const staticOutputs = await collectStaticOutputs(configuration);
  const pageOutputs = new Map<string, RenderedOutput>();
  const generatedStylePath = "assets/xerxes-bun-docs.css";
  pageOutputs.set(generatedStylePath, { text: generatedStyleSheet() });
  for (const [path, output] of staticOutputs) pageOutputs.set(path, output);

  const stylePaths = [
    generatedStylePath,
    ...[...staticOutputs.keys()]
      .filter((path) => path.endsWith(".css"))
      .sort(compareText),
  ];
  const pageNavigation = createPageNavigation(
    documents,
    configuration.apiOutputDirectory,
  );
  for (const document of documents) {
    const body = renderDocumentBody(document);
    pageOutputs.set(document.outputPath, {
      text: renderHtmlPage(
        document.title,
        document.outputPath,
        body,
        pageNavigation,
        stylePaths,
        configuration.title,
      ),
    });
  }

  const documentationReportOutputPath = "documentation-report.html";
  pageOutputs.set(documentationReportOutputPath, {
    text: renderHtmlPage(
      "Native Bun documentation report",
      documentationReportOutputPath,
      renderDocumentationReport(diagnostics),
      pageNavigation,
      stylePaths,
      configuration.title,
    ),
  });

  const sitemapOutputPath = "sitemap.html";
  pageOutputs.set(sitemapOutputPath, {
    text: renderHtmlPage(
      "Documentation site map",
      sitemapOutputPath,
      renderSitemap(documents, sitemapOutputPath),
      pageNavigation,
      stylePaths,
      configuration.title,
    ),
  });

  const report = renderBuildReport({
    configuration,
    diagnostics,
    documents,
    typeScriptApiModules,
  });
  pageOutputs.set(BUN_DOCS_BUILD_REPORT_FILENAME, { text: report });

  const manifest = await readOwnedManifest(configuration.outputDirectory);
  await assertOutputOwnership(configuration.outputDirectory, manifest);
  const currentPaths = [...pageOutputs.keys()].sort(compareText);
  const previousPaths = manifest?.files ?? [];
  const currentPathSet = new Set(currentPaths);
  const changes: BunDocsChange[] = [];

  for (const stalePath of previousPaths
    .filter((path) => !currentPathSet.has(path))
    .sort(compareText)) {
    const deleted = await deleteOwnedOutput(
      configuration.outputDirectory,
      stalePath,
      dryRun,
    );
    if (deleted)
      changes.push({
        action: "deleted",
        path: join(configuration.outputDirectory, stalePath),
      });
  }

  for (const path of currentPaths) {
    const output = pageOutputs.get(path);
    if (output === undefined)
      throw new BunDocsBuildError(`Missing rendered output: ${path}`);
    changes.push(
      await writeOwnedOutput(
        configuration.outputDirectory,
        path,
        output,
        dryRun,
      ),
    );
  }

  const manifestText = renderOwnedManifest(currentPaths);
  changes.push(
    await writeOwnedOutput(
      configuration.outputDirectory,
      BUN_DOCS_MANIFEST_FILENAME,
      { text: manifestText },
      dryRun,
    ),
  );

  return {
    changed: changes.some((change) => change.action !== "unchanged"),
    changes: changes.sort((left, right) => compareText(left.path, right.path)),
    diagnostics: diagnostics.sort(compareDiagnostics),
    documents: documents.map((document) => ({
      kind: document.kind,
      outputPath: document.outputPath,
      sourcePath: document.sourcePath,
      title: document.title,
    })),
    dryRun,
    outputDirectory: configuration.outputDirectory,
    typeScriptApiModules,
  };
}

function assertUniqueOutputPaths(documents: readonly SourceDocument[]): void {
  const paths = new Set<string>();
  for (const document of documents) {
    if (paths.has(document.outputPath)) {
      throw new BunDocsBuildError(
        `Multiple documentation sources produce ${document.outputPath}`,
      );
    }
    paths.add(document.outputPath);
  }
}

async function assertOutputOwnership(
  outputDirectory: string,
  manifest: OwnedManifest | undefined,
): Promise<void> {
  const outputStat = await safeStat(outputDirectory);
  if (outputStat === undefined) return;
  if (!outputStat.isDirectory()) {
    throw new BunDocsBuildError(
      `Documentation output is not a directory: ${outputDirectory}`,
    );
  }

  const entries = await readdir(outputDirectory);
  if (entries.length === 0 || manifest !== undefined) return;
  throw new BunDocsBuildError(
    `Refusing to write a non-empty documentation output not owned by ${BUN_DOCS_GENERATOR_NAME}: ${outputDirectory}`,
  );
}

async function collectFiles(
  directory: string,
  predicate: (path: string) => boolean,
  paths: string[] = [],
): Promise<string[]> {
  const directoryStat = await safeStat(directory);
  if (directoryStat === undefined || !directoryStat.isDirectory()) return paths;

  const entries = await readdir(directory, { withFileTypes: true });
  for (const entry of entries.sort((left, right) =>
    compareText(left.name, right.name),
  )) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) {
      await collectFiles(path, predicate, paths);
    } else if (entry.isFile() && predicate(path)) {
      paths.push(path);
    }
  }
  return paths;
}

async function collectStaticOutputs(
  configuration: ResolvedBunDocsConfiguration,
): Promise<Map<string, RenderedOutput>> {
  const outputs = new Map<string, RenderedOutput>();
  if (configuration.staticDirectory === undefined) return outputs;
  const staticFiles = await collectFiles(
    configuration.staticDirectory,
    () => true,
  );
  for (const sourcePath of staticFiles.sort(compareText)) {
    const sourceRelativePath = toPosixPath(
      relative(configuration.staticDirectory, sourcePath),
    );
    const outputPath = `assets/${sourceRelativePath}`;
    outputs.set(outputPath, { bytes: await readFile(sourcePath) });
  }
  return outputs;
}

function compareDiagnostics(
  left: BunDocsDiagnostic,
  right: BunDocsDiagnostic,
): number {
  return compareText(
    `${left.source}:${left.line ?? 0}:${left.code}`,
    `${right.source}:${right.line ?? 0}:${right.code}`,
  );
}

function compareDocuments(left: SourceDocument, right: SourceDocument): number {
  return compareText(left.outputPath, right.outputPath);
}

function compareText(left: string, right: string): number {
  return left.localeCompare(right, "en");
}

function createDiagnostic(
  code: string,
  message: string,
  source: string,
  line?: number,
): BunDocsDiagnostic {
  return line === undefined
    ? { code, level: "warning", message, source }
    : { code, level: "warning", line, message, source };
}

function createDocument(
  sourcePath: string,
  sourceRelativePath: string,
  kind: BunDocsSourceKind,
  content: string,
): SourceDocument {
  const extension = extname(sourceRelativePath).toLowerCase();
  const outputPath = `${sourceRelativePath.slice(0, -extension.length)}.html`;
  return {
    content,
    kind,
    outputPath,
    sourcePath: sourceRelativePath,
    title: documentTitle(content, basename(sourceRelativePath, extension)),
  };
}

function createPageNavigation(
  documents: readonly SourceDocument[],
  apiOutputDirectory: string,
): readonly SourceDocument[] {
  const topLevel = documents.filter((document) => {
    if (document.kind === "typescript-api") return false;
    return !document.sourcePath.includes("/");
  });
  const apiIndex = documents.find(
    (document) => document.outputPath === `${apiOutputDirectory}/index.html`,
  );
  return [
    ...topLevel.sort(compareDocuments),
    ...(apiIndex === undefined ? [] : [apiIndex]),
  ];
}

async function deleteOwnedOutput(
  outputDirectory: string,
  outputPath: string,
  dryRun: boolean,
): Promise<boolean> {
  const path = ownedOutputPath(outputDirectory, outputPath);
  if (!(await pathExists(path))) return false;
  if (!dryRun) await rm(path, { force: true });
  return true;
}

function documentTitle(content: string, fallback: string): string {
  const markdownTitle = content.match(/^#\s+(.+)$/mu)?.[1];
  return markdownTitle === undefined ? fallback : plainText(markdownTitle);
}

async function discoverGeneratedApiDocuments(
  directory: string,
  apiOutputDirectory: string,
): Promise<SourceDocument[]> {
  const paths = await collectFiles(
    directory,
    (path) => extname(path).toLowerCase() === ".md",
  );
  const documents: SourceDocument[] = [];
  for (const sourcePath of paths.sort(compareText)) {
    const relativePath = toPosixPath(relative(directory, sourcePath));
    documents.push(
      createDocument(
        sourcePath,
        `${apiOutputDirectory}/${relativePath}`,
        "typescript-api",
        await readFile(sourcePath, "utf8"),
      ),
    );
  }
  return documents;
}

async function discoverSourceDocuments(
  configuration: ResolvedBunDocsConfiguration,
  diagnostics: BunDocsDiagnostic[],
): Promise<SourceDocument[]> {
  const sourcePaths: string[] = [];
  await collectDocumentationSources(
    configuration.sourceDirectory,
    configuration,
    sourcePaths,
  );
  const documents: SourceDocument[] = [];
  for (const sourcePath of sourcePaths.sort(compareText)) {
    const sourceRelativePath = toPosixPath(
      relative(configuration.sourceDirectory, sourcePath),
    );
    const content = await readFile(sourcePath, "utf8");
    documents.push(
      createDocument(sourcePath, sourceRelativePath, "markdown", content),
    );
  }

  const notebooks = await collectFiles(
    configuration.sourceDirectory,
    (path) => extname(path).toLowerCase() === ".ipynb",
  );
  for (const notebook of notebooks.sort(compareText)) {
    diagnostics.push(
      createDiagnostic(
        "unsupported-jupyter-notebook",
        "Jupyter notebooks are preserved as source but are not executed or rendered by the native Bun builder.",
        toPosixPath(relative(configuration.sourceDirectory, notebook)),
      ),
    );
  }
  return documents;
}

async function collectDocumentationSources(
  directory: string,
  configuration: ResolvedBunDocsConfiguration,
  paths: string[],
): Promise<void> {
  const entries = await readdir(directory, { withFileTypes: true });
  for (const entry of entries.sort((left, right) =>
    compareText(left.name, right.name),
  )) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) {
      if (shouldIgnoreSourceDirectory(path, entry.name, configuration))
        continue;
      await collectDocumentationSources(path, configuration, paths);
    } else if (
      entry.isFile() &&
      DOCUMENT_EXTENSIONS.has(extname(entry.name).toLowerCase())
    ) {
      paths.push(path);
    }
  }
}

function shouldIgnoreSourceDirectory(
  path: string,
  name: string,
  configuration: ResolvedBunDocsConfiguration,
): boolean {
  return (
    IGNORED_SOURCE_DIRECTORIES.has(name) ||
    path === configuration.outputDirectory ||
    path === configuration.staticDirectory
  );
}

function escapeHtml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function generatedStyleSheet(): string {
  return [
    ":root { color-scheme: light dark; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }",
    "body { line-height: 1.55; margin: 0; }",
    "header, main, footer { margin: 0 auto; max-width: 72rem; padding: 1rem 1.5rem; }",
    "header { border-bottom: 1px solid color-mix(in srgb, currentColor 20%, transparent); }",
    "header nav ul { display: flex; flex-wrap: wrap; gap: .8rem; list-style: none; padding: 0; }",
    "header nav li { display: inline; }",
    "pre { overflow-x: auto; padding: 1rem; }",
    "code, pre { border-radius: .35rem; }",
    "code { padding: .1rem .25rem; }",
    ".notice { border-left: .3rem solid #d38a00; padding: .75rem 1rem; }",
    ".unresolved { opacity: .75; }",
    "footer { border-top: 1px solid color-mix(in srgb, currentColor 20%, transparent); font-size: .9rem; }",
    "",
  ].join("\n");
}

function isPathInside(path: string, parent: string): boolean {
  const pathRelative = relative(parent, path);
  return (
    pathRelative === "" ||
    (!pathRelative.startsWith(`..${sep}`) &&
      pathRelative !== ".." &&
      !isAbsolute(pathRelative))
  );
}

function ownedOutputPath(outputDirectory: string, outputPath: string): string {
  if (
    outputPath === "" ||
    isAbsolute(outputPath) ||
    outputPath.split(/[\\/]/u).includes("..")
  ) {
    throw new BunDocsBuildError(`Unsafe generated output path: ${outputPath}`);
  }
  const path = resolve(outputDirectory, outputPath);
  if (!isPathInside(path, outputDirectory)) {
    throw new BunDocsBuildError(
      `Generated output escapes documentation directory: ${outputPath}`,
    );
  }
  return path;
}

function plainText(value: string): string {
  return value
    .replaceAll("`", "")
    .replaceAll("*", "")
    .replaceAll("_", " ")
    .trim();
}

async function pathExists(path: string): Promise<boolean> {
  return (await safeStat(path)) !== undefined;
}

async function pathIsFile(path: string): Promise<boolean> {
  return (await safeStat(path))?.isFile() ?? false;
}

async function readConfiguration(
  path: string,
): Promise<RawBunDocsConfiguration> {
  let parsed: unknown;
  try {
    parsed = JSON.parse(await readFile(path, "utf8"));
  } catch (error) {
    throw new BunDocsBuildError(
      `Cannot read Bun docs configuration at ${path}: ${errorMessage(error)}`,
    );
  }
  if (!isRecord(parsed))
    throw new BunDocsBuildError(
      `Bun docs configuration must be a JSON object: ${path}`,
    );
  const allowedKeys = new Set([
    "outputDirectory",
    "sourceDirectory",
    "staticDirectory",
    "title",
    "typescriptApi",
    "version",
  ]);
  for (const key of Object.keys(parsed)) {
    if (!allowedKeys.has(key))
      throw new BunDocsBuildError(
        `Unsupported Bun docs configuration key "${key}": ${path}`,
      );
  }
  if (
    parsed.version !== 1 ||
    typeof parsed.title !== "string" ||
    typeof parsed.sourceDirectory !== "string" ||
    typeof parsed.outputDirectory !== "string" ||
    !isRecord(parsed.typescriptApi)
  ) {
    throw new BunDocsBuildError(`Invalid Bun docs configuration: ${path}`);
  }
  const api = parsed.typescriptApi;
  if (
    typeof api.sourceDirectory !== "string" ||
    typeof api.packageName !== "string" ||
    typeof api.outputDirectory !== "string"
  ) {
    throw new BunDocsBuildError(
      `Invalid TypeScript API configuration: ${path}`,
    );
  }
  return {
    outputDirectory: parsed.outputDirectory,
    sourceDirectory: parsed.sourceDirectory,
    ...(typeof parsed.staticDirectory === "string"
      ? { staticDirectory: parsed.staticDirectory }
      : {}),
    title: parsed.title,
    typescriptApi: {
      outputDirectory: api.outputDirectory,
      packageName: api.packageName,
      sourceDirectory: api.sourceDirectory,
    },
    version: parsed.version,
  };
}

async function readOwnedManifest(
  outputDirectory: string,
): Promise<OwnedManifest | undefined> {
  const path = join(outputDirectory, BUN_DOCS_MANIFEST_FILENAME);
  if (!(await pathIsFile(path))) return undefined;
  let parsed: unknown;
  try {
    parsed = JSON.parse(await readFile(path, "utf8"));
  } catch (error) {
    throw new BunDocsBuildError(
      `Cannot read documentation output manifest: ${errorMessage(error)}`,
    );
  }
  if (
    !isRecord(parsed) ||
    parsed.generator !== BUN_DOCS_GENERATOR_NAME ||
    parsed.version !== BUN_DOCS_MANIFEST_VERSION ||
    !Array.isArray(parsed.files) ||
    !parsed.files.every((path) => typeof path === "string")
  ) {
    throw new BunDocsBuildError(
      `Invalid documentation output manifest: ${path}`,
    );
  }
  return {
    files: [...parsed.files].sort(compareText),
    generator: BUN_DOCS_GENERATOR_NAME,
    version: BUN_DOCS_MANIFEST_VERSION,
  };
}

function renderBuildReport(input: {
  readonly configuration: ResolvedBunDocsConfiguration;
  readonly diagnostics: readonly BunDocsDiagnostic[];
  readonly documents: readonly SourceDocument[];
  readonly typeScriptApiModules: number;
}): string {
  return `${JSON.stringify(
    {
      documents: input.documents.map((document) => ({
        kind: document.kind,
        output: document.outputPath,
        source: document.sourcePath,
        title: document.title,
      })),
      generator: BUN_DOCS_GENERATOR_NAME,
      notices: input.diagnostics.slice().sort(compareDiagnostics),
      typescriptApi: {
        modules: input.typeScriptApiModules,
        outputDirectory: input.configuration.apiOutputDirectory,
        packageName: input.configuration.apiPackageName,
      },
      version: BUN_DOCS_MANIFEST_VERSION,
    },
    null,
    2,
  )}\n`;
}

function renderDocumentBody(document: SourceDocument): string {
  return renderMarkdownBody(document.content);
}

function renderHtmlPage(
  title: string,
  outputPath: string,
  body: string,
  navigation: readonly SourceDocument[],
  stylePaths: readonly string[],
  siteTitle: string,
): string {
  const styles = stylePaths
    .map(
      (path) =>
        `<link rel="stylesheet" href="${escapeHtml(relativeHref(outputPath, path))}">`,
    )
    .join("\n    ");
  const links = [
    ...navigation.map((document) => ({
      href: relativeHref(outputPath, document.outputPath),
      title: document.title,
    })),
    { href: relativeHref(outputPath, "sitemap.html"), title: "Site map" },
    {
      href: relativeHref(outputPath, "documentation-report.html"),
      title: "Build report",
    },
  ];
  const navigationHtml = links
    .map(
      (link) =>
        `<li><a href="${escapeHtml(link.href)}">${escapeHtml(link.title)}</a></li>`,
    )
    .join("");
  return [
    "<!doctype html>",
    '<html lang="en">',
    "<head>",
    '  <meta charset="utf-8">',
    '  <meta name="viewport" content="width=device-width, initial-scale=1">',
    `  <title>${escapeHtml(`${title} · ${siteTitle}`)}</title>`,
    `    ${styles}`,
    "</head>",
    "<body>",
    "  <header>",
    `    <p><a href="${escapeHtml(relativeHref(outputPath, "index.html"))}">${escapeHtml(siteTitle)}</a></p>`,
    `    <nav aria-label="Site"><ul>${navigationHtml}</ul></nav>`,
    "  </header>",
    "  <main>",
    `    <article data-source="${escapeHtml(outputPath)}">`,
    body,
    "    </article>",
    "  </main>",
    "  <footer>Generated deterministically by the native Bun documentation builder.</footer>",
    "</body>",
    "</html>",
    "",
  ].join("\n");
}

function renderInline(value: string): string {
  const escaped = escapeHtml(value);
  const linked = escaped.replace(
    /\[([^\]]+)\]\(([^)\s]+)\)/gu,
    (_match, label: string, href: string) => {
      const destination = documentationHref(href);
      return `<a href="${destination}">${label}</a>`;
    },
  );
  return linked
    .replace(/`([^`]+)`/gu, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/gu, "<strong>$1</strong>");
}

function renderMarkdownBody(source: string): string {
  const lines = source.split(/\r?\n/u);
  const output: string[] = [];
  const paragraph: string[] = [];
  let list: { readonly items: string[]; readonly ordered: boolean } | undefined;
  let codeFence:
    { readonly language: string; readonly lines: string[] } | undefined;

  const flushParagraph = (): void => {
    if (paragraph.length === 0) return;
    output.push(`<p>${renderInline(paragraph.join(" "))}</p>`);
    paragraph.length = 0;
  };
  const flushList = (): void => {
    if (list === undefined) return;
    const tag = list.ordered ? "ol" : "ul";
    output.push(
      `<${tag}>${list.items.map((item) => `<li>${renderInline(item)}</li>`).join("")}</${tag}>`,
    );
    list = undefined;
  };

  for (const line of lines) {
    const fence = line.match(/^```([^`]*)$/u);
    if (fence !== null) {
      flushParagraph();
      flushList();
      if (codeFence === undefined) {
        codeFence = { language: fence[1]?.trim() ?? "", lines: [] };
      } else {
        const language =
          codeFence.language === ""
            ? ""
            : ` class="language-${escapeHtml(codeFence.language)}"`;
        output.push(
          `<pre><code${language}>${escapeHtml(codeFence.lines.join("\n"))}</code></pre>`,
        );
        codeFence = undefined;
      }
      continue;
    }
    if (codeFence !== undefined) {
      codeFence.lines.push(line);
      continue;
    }
    const heading = line.match(/^(#{1,6})\s+(.+)$/u);
    if (heading !== null) {
      flushParagraph();
      flushList();
      const level = heading[1]?.length ?? 1;
      output.push(`<h${level}>${renderInline(heading[2] ?? "")}</h${level}>`);
      continue;
    }
    if (/^\s*([-*_])\1\1+\s*$/u.test(line)) {
      flushParagraph();
      flushList();
      output.push("<hr>");
      continue;
    }
    const unordered = line.match(/^\s*[-*+]\s+(.+)$/u);
    const ordered = line.match(/^\s*\d+\.\s+(.+)$/u);
    if (unordered !== null || ordered !== null) {
      flushParagraph();
      const orderedList = ordered !== null;
      if (list === undefined || list.ordered !== orderedList) {
        flushList();
        list = { items: [], ordered: orderedList };
      }
      list.items.push((ordered ?? unordered)?.[1] ?? "");
      continue;
    }
    const listContinuation = line.match(/^\s{2,}(\S.*)$/u);
    if (
      list !== undefined &&
      listContinuation !== null &&
      list.items.length > 0
    ) {
      const lastItem = list.items.length - 1;
      list.items[lastItem] = `${list.items[lastItem]} ${listContinuation[1]}`;
      continue;
    }
    if (line.trim() === "") {
      flushParagraph();
      flushList();
      continue;
    }
    const quote = line.match(/^>\s?(.+)$/u);
    if (quote !== null) {
      flushParagraph();
      flushList();
      output.push(
        `<blockquote><p>${renderInline(quote[1] ?? "")}</p></blockquote>`,
      );
      continue;
    }
    flushList();
    paragraph.push(line.trim());
  }
  flushParagraph();
  flushList();
  if (codeFence !== undefined) {
    output.push(
      `<pre><code>${escapeHtml(codeFence.lines.join("\n"))}</code></pre>`,
    );
  }
  return output.join("\n");
}

function renderDocumentationReport(
  diagnostics: readonly BunDocsDiagnostic[],
): string {
  const noticeItems =
    diagnostics.length === 0
      ? "<p>No build notices were reported.</p>"
      : `<ul>${diagnostics
          .slice()
          .sort(compareDiagnostics)
          .map((notice) => {
            const line = notice.line === undefined ? "" : `:${notice.line}`;
            return `<li><code>${escapeHtml(notice.source)}${line}</code> — ${escapeHtml(notice.message)}</li>`;
          })
          .join("")}</ul>`;
  return [
    "<h1>Native Bun documentation report</h1>",
    "<p>The builder renders Markdown and generated TypeScript API pages deterministically.</p>",
    "<h2>Build notices</h2>",
    noticeItems,
  ].join("\n");
}

function renderOwnedManifest(paths: readonly string[]): string {
  return `${JSON.stringify(
    {
      files: paths,
      generator: BUN_DOCS_GENERATOR_NAME,
      version: BUN_DOCS_MANIFEST_VERSION,
    },
    null,
    2,
  )}\n`;
}

function renderSitemap(
  documents: readonly SourceDocument[],
  outputPath: string,
): string {
  const groups = new Map<string, SourceDocument[]>();
  for (const document of documents) {
    const group =
      document.kind === "typescript-api"
        ? "Native TypeScript API"
        : "Documentation";
    const current = groups.get(group) ?? [];
    current.push(document);
    groups.set(group, current);
  }
  const sections = [...groups.entries()]
    .sort(([left], [right]) => compareText(left, right))
    .map(([name, entries]) =>
      [
        `<h2>${escapeHtml(name)}</h2>`,
        "<ul>",
        ...entries.sort(compareDocuments).map((document) => {
          const href = escapeHtml(
            relativeHref(outputPath, document.outputPath),
          );
          const title = escapeHtml(document.title);
          const source = escapeHtml(document.sourcePath);
          return `<li><a href="${href}">${title}</a> <code>${source}</code></li>`;
        }),
        "</ul>",
      ].join("\n"),
    );
  return ["<h1>Documentation site map</h1>", ...sections].join("\n");
}

function resolveConfiguredPath(basePath: string, value: string): string {
  return isAbsolute(value) ? resolve(value) : resolve(basePath, value);
}

function resolveOptionPath(basePath: string, value: string): string {
  return resolveConfiguredPath(basePath, value);
}

function relativeHref(fromOutputPath: string, toOutputPath: string): string {
  const href = posix.relative(posix.dirname(fromOutputPath), toOutputPath);
  return href === "" ? posix.basename(toOutputPath) : href;
}

async function safeStat(
  path: string,
): Promise<Awaited<ReturnType<typeof stat>> | undefined> {
  try {
    return await stat(path);
  } catch (error) {
    if (isNotFoundError(error)) return undefined;
    throw error;
  }
}

function toPosixPath(path: string): string {
  return path.split(sep).join("/");
}

function validateOutputDirectory(path: string): string {
  const normalized = path.replaceAll("\\", "/");
  if (
    normalized === "" ||
    normalized.startsWith("/") ||
    normalized.split("/").includes("..")
  ) {
    throw new BunDocsBuildError(
      `TypeScript API output directory must be a safe relative path: ${path}`,
    );
  }
  return normalized.replace(/\/$/u, "");
}

async function writeOwnedOutput(
  outputDirectory: string,
  outputPath: string,
  output: RenderedOutput,
  dryRun: boolean,
): Promise<BunDocsChange> {
  const path = ownedOutputPath(outputDirectory, outputPath);
  const previous = await readOptionalBytes(path);
  const next = output.bytes ?? new TextEncoder().encode(output.text ?? "");
  if (previous !== undefined && bytesEqual(previous, next))
    return { action: "unchanged", path };
  if (!dryRun) {
    await mkdir(dirname(path), { recursive: true });
    await writeFile(path, next);
  }
  return { action: previous === undefined ? "created" : "updated", path };
}

async function readOptionalBytes(
  path: string,
): Promise<Uint8Array | undefined> {
  try {
    return await readFile(path);
  } catch (error) {
    if (isNotFoundError(error)) return undefined;
    throw error;
  }
}

function bytesEqual(left: Uint8Array, right: Uint8Array): boolean {
  if (left.byteLength !== right.byteLength) return false;
  for (let index = 0; index < left.byteLength; index += 1) {
    if (left[index] !== right[index]) return false;
  }
  return true;
}

function documentationHref(href: string): string {
  const normalized = href.replace(/\.md(?=#[^#]*$|$)/u, ".html");
  const lower = normalized.toLowerCase();
  if (lower.startsWith("javascript:") || lower.startsWith("data:"))
    return "#unsafe-link";
  return normalized;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function isNotFoundError(error: unknown): boolean {
  return isRecord(error) && error.code === "ENOENT";
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}
