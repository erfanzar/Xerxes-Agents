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
  const generatedScriptPath = "assets/docs.js";
  const generatedFaviconPath = "assets/favicon.svg";
  const searchIndexPath = "assets/search-index.json";
  pageOutputs.set(generatedStylePath, { text: generatedStyleSheet() });
  pageOutputs.set(generatedScriptPath, { text: generatedClientScript() });
  pageOutputs.set(generatedFaviconPath, { text: generatedFaviconSvg() });
  for (const [path, output] of staticOutputs) pageOutputs.set(path, output);

  const stylePaths = [
    generatedStylePath,
    ...[...staticOutputs.entries()]
      .filter(
        ([path, output]) =>
          path.endsWith(".css") &&
          ((output.bytes?.byteLength ?? 0) > 0 || (output.text?.length ?? 0) > 0),
      )
      .map(([path]) => path)
      .sort(compareText),
  ];
  const pageNavigation = createPageNavigation(
    documents,
    configuration.apiOutputDirectory,
  );
  const searchEntries = documents.map((document) => ({
    kind: document.kind,
    path: document.outputPath,
    summary: documentDescription(document.content),
    title: document.title,
  }));
  pageOutputs.set(searchIndexPath, {
    text: `${JSON.stringify(searchEntries, null, 2)}\n`,
  });
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
        documentDescription(document.content),
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
      "Deterministic build diagnostics reported by the native Bun documentation builder.",
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
      "Complete index of guides and generated TypeScript API pages.",
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

function generatedFaviconSvg(): string {
  return [
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">',
    '  <rect width="32" height="32" rx="7" fill="#0c0d0f"/>',
    '  <path d="M9 9l14 14M23 9L9 23" stroke="#60a5fa" stroke-width="3.2" stroke-linecap="round"/>',
    "</svg>",
    "",
  ].join("\n");
}

function generatedClientScript(): string {
  return [
    "(() => {",
    "  const toggle = document.querySelector('[data-nav-toggle]');",
    "  const sidebar = document.getElementById('docs-sidebar');",
    "  if (toggle && sidebar) {",
    "    const setOpen = (open) => {",
    "      sidebar.classList.toggle('is-open', open);",
    "      toggle.setAttribute('aria-expanded', open ? 'true' : 'false');",
    "    };",
    "    toggle.addEventListener('click', () => setOpen(!sidebar.classList.contains('is-open')));",
    "    document.addEventListener('keydown', (event) => {",
    "      if (event.key === 'Escape' && sidebar.classList.contains('is-open')) {",
    "        setOpen(false);",
    "        toggle.focus();",
    "      }",
    "    });",
    "  }",
    "  const form = document.querySelector('[data-docs-search]');",
    "  const input = form?.querySelector('input[type=\"search\"]');",
    "  const results = document.getElementById('docs-search-results');",
    "  if (form && input && results) {",
    "    let indexPromise = null;",
    "    const baseUrl = form.getAttribute('data-index-url') || 'assets/search-index.json';",
    "    const rootPrefix = form.getAttribute('data-root-prefix') || '';",
    "    const loadIndex = () => {",
    "      if (!indexPromise) {",
    "        indexPromise = fetch(baseUrl).then((r) => (r.ok ? r.json() : [])).catch(() => []);",
    "      }",
    "      return indexPromise;",
    "    };",
    "    const renderMatches = (items, query) => {",
    "      results.replaceChildren();",
    "      if (!query) { results.hidden = true; return; }",
    "      const q = query.toLowerCase();",
    "      const hits = items.filter((item) => item.title.toLowerCase().includes(q) || item.summary.toLowerCase().includes(q) || item.path.toLowerCase().includes(q)).slice(0, 8);",
    "      if (hits.length === 0) {",
    "        const empty = document.createElement('p');",
    "        empty.className = 'search-empty';",
    "        empty.textContent = 'No matching pages.';",
    "        results.appendChild(empty);",
    "      } else {",
    "        const list = document.createElement('ul');",
    "        for (const hit of hits) {",
    "          const li = document.createElement('li');",
    "          const link = document.createElement('a');",
    "          link.href = rootPrefix + hit.path;",
    "          const title = document.createElement('span');",
    "          title.className = 'search-hit-title';",
    "          title.textContent = hit.title;",
    "          const path = document.createElement('span');",
    "          path.className = 'search-hit-path';",
    "          path.textContent = hit.path;",
    "          link.append(title, path);",
    "          li.appendChild(link);",
    "          list.appendChild(li);",
    "        }",
    "        results.appendChild(list);",
    "      }",
    "      results.hidden = false;",
    "    };",
    "    form.addEventListener('submit', (event) => event.preventDefault());",
    "    input.addEventListener('input', () => {",
    "      const q = input.value.trim();",
    "      if (!q) { results.hidden = true; return; }",
    "      void loadIndex().then((items) => renderMatches(items, input.value.trim()));",
    "    });",
    "    document.addEventListener('click', (event) => {",
    "      if (!form.contains(event.target)) results.hidden = true;",
    "    });",
    "    document.addEventListener('keydown', (event) => {",
    "      if (event.key === 'Escape' && !results.hidden) { results.hidden = true; }",
    "      if (event.key === '/' && document.activeElement?.tagName !== 'INPUT' && document.activeElement?.tagName !== 'TEXTAREA') {",
    "        event.preventDefault();",
    "        input.focus();",
    "      }",
    "    });",
    "  }",
    "  for (const pre of document.querySelectorAll('article pre')) {",
    "    const code = pre.querySelector('code');",
    "    if (!code || !navigator.clipboard) continue;",
    "    const btn = document.createElement('button');",
    "    btn.type = 'button';",
    "    btn.className = 'code-copy';",
    "    btn.textContent = 'Copy';",
    "    btn.setAttribute('aria-label', 'Copy code block');",
    "    let timer;",
    "    btn.addEventListener('click', () => {",
    "      void navigator.clipboard.writeText(code.textContent || '').then(() => {",
    "        btn.textContent = 'Copied';",
    "        clearTimeout(timer);",
    "        timer = setTimeout(() => { btn.textContent = 'Copy'; }, 1600);",
    "      }).catch(() => undefined);",
    "    });",
    "    pre.appendChild(btn);",
    "  }",
    "})();",
    "",
  ].join("\n");
}

function generatedStyleSheet(): string {
  return [
    ":root {",
    "  color-scheme: light dark;",
    "  --bg: #f7f7f8;",
    "  --surface: #ffffff;",
    "  --surface-subtle: #eff1f4;",
    "  --border: #dcdfe4;",
    "  --border-strong: #c2c8d1;",
    "  --text: #181a1f;",
    "  --title: #0d0f12;",
    "  --muted: #555c68;",
    "  --accent: #1d4ed8;",
    "  --accent-hover: #1e40af;",
    "  --accent-soft: #1d4ed814;",
    "  --visited: #1e40af;",
    "  --warn: #b45309;",
    "  --warn-bg: #fef3c7;",
    "  --shadow-sm: 0 1px 2px #0000000f;",
    "  --shadow-md: 0 8px 24px #00000018;",
    "  --sans: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', system-ui, sans-serif;",
    "  --mono: ui-monospace, 'SF Mono', Menlo, Consolas, monospace;",
    "  --header-h: 56px;",
    "  --sidebar-w: 268px;",
    "  --r-sm: 6px;",
    "  --r-md: 10px;",
    "}",
    "@media (prefers-color-scheme: dark) {",
    "  :root {",
    "    --bg: #0c0d0f;",
    "    --surface: #14161a;",
    "    --surface-subtle: #1b1e24;",
    "    --border: #ffffff17;",
    "    --border-strong: #ffffff2b;",
    "    --text: #dce0e6;",
    "    --title: #f5f7fa;",
    "    --muted: #98a1b0;",
    "    --accent: #60a5fa;",
    "    --accent-hover: #93c5fd;",
    "    --accent-soft: #60a5fa1a;",
    "    --visited: #93c5fd;",
    "    --warn: #f59e0b;",
    "    --warn-bg: #f59e0b1a;",
    "    --shadow-sm: 0 1px 2px #00000059;",
    "    --shadow-md: 0 12px 32px #00000080;",
    "  }",
    "}",
    "* { box-sizing: border-box; }",
    "html { background: var(--bg); color: var(--text); font: 400 16px/1.65 var(--sans); -webkit-font-smoothing: antialiased; }",
    "body { margin: 0; min-height: 100dvh; display: flex; flex-direction: column; overflow-x: clip; }",
    ":focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }",
    "[tabindex='-1']:focus, [tabindex='-1']:focus-visible { outline: none; }",
    ".skip-link { position: absolute; left: -9999px; top: 8px; z-index: 80; padding: 10px 14px; border-radius: var(--r-sm); background: var(--accent); color: #ffffff; font-weight: 600; text-decoration: none; }",
    ".skip-link:focus { left: 12px; }",
    ".site-header { position: sticky; top: 0; z-index: 40; min-height: var(--header-h); background: color-mix(in srgb, var(--bg) 90%, transparent); -webkit-backdrop-filter: blur(12px); backdrop-filter: blur(12px); border-bottom: 1px solid var(--border); }",
    ".site-header__inner { max-width: 1320px; margin: 0 auto; padding: 10px 24px; display: flex; align-items: center; gap: 16px; }",
    ".site-brand { display: inline-flex; align-items: center; gap: 10px; color: var(--title); font-size: 16px; font-weight: 700; letter-spacing: -0.02em; text-decoration: none; white-space: nowrap; }",
    ".site-brand__mark { display: inline-grid; place-items: center; width: 24px; height: 24px; border-radius: var(--r-sm); background: var(--title); color: var(--bg); font: 700 13px/1 var(--mono); }",
    ".site-header__links { display: flex; align-items: center; gap: 4px; margin-left: 8px; }",
    ".site-header__links a { padding: 6px 10px; border-radius: var(--r-sm); color: var(--muted); font-size: 14px; font-weight: 500; text-decoration: none; transition: color 120ms, background-color 120ms; }",
    ".site-header__links a:hover { color: var(--title); background: var(--surface-subtle); }",
    ".site-header__links a[aria-current='page'] { color: var(--title); background: var(--accent-soft); font-weight: 600; }",
    ".site-search { position: relative; margin-left: auto; width: min(320px, 100%); }",
    ".site-search input { width: 100%; height: 36px; padding: 0 12px; border: 1px solid var(--border); border-radius: var(--r-sm); background: var(--surface); color: var(--title); font: 400 14px/1.2 var(--sans); }",
    ".site-search input::placeholder { color: var(--muted); }",
    ".site-search input:focus { border-color: var(--accent); }",
    ".search-results { position: absolute; top: calc(100% + 6px); left: 0; right: 0; z-index: 50; max-height: 360px; overflow: auto; background: var(--surface); border: 1px solid var(--border-strong); border-radius: var(--r-md); box-shadow: var(--shadow-md); padding: 6px; }",
    ".search-results ul { list-style: none; margin: 0; padding: 0; display: grid; gap: 2px; }",
    ".search-results a { display: grid; gap: 2px; padding: 8px 10px; border-radius: var(--r-sm); text-decoration: none; color: var(--title); }",
    ".search-results a:hover { background: var(--surface-subtle); }",
    ".search-hit-title { font-size: 14px; font-weight: 600; }",
    ".search-hit-path { font: 400 12px/1.3 var(--mono); color: var(--muted); }",
    ".search-empty { margin: 0; padding: 10px; font-size: 13px; color: var(--muted); }",
    ".nav-toggle { display: none; align-items: center; justify-content: center; min-height: 44px; min-width: 44px; padding: 0 12px; border: 1px solid var(--border); border-radius: var(--r-sm); background: var(--surface); color: var(--title); font: 500 14px/1 var(--sans); cursor: pointer; }",
    ".docs-shell { flex: 1 1 auto; width: 100%; max-width: 1320px; margin: 0 auto; padding: 28px 24px 64px; display: grid; grid-template-columns: var(--sidebar-w) minmax(0, 1fr); gap: 48px; align-items: start; }",
    ".docs-sidebar { position: sticky; top: calc(var(--header-h) + 20px); max-height: calc(100dvh - var(--header-h) - 40px); overflow-y: auto; padding-right: 12px; }",
    ".nav-group { margin: 0 0 20px; }",
    ".nav-group__title { margin: 0 0 8px; padding: 0 10px; font-size: 12px; font-weight: 600; color: var(--muted); }",
    ".nav-group ul { list-style: none; margin: 0; padding: 0; display: grid; gap: 2px; }",
    ".nav-group a { display: block; padding: 6px 10px; border-radius: var(--r-sm); color: var(--text); font-size: 14px; line-height: 1.35; text-decoration: none; overflow-wrap: anywhere; transition: background-color 120ms, color 120ms; }",
    ".nav-group a:hover { background: var(--surface-subtle); color: var(--title); }",
    ".nav-group a[aria-current='page'] { background: var(--accent-soft); color: var(--accent); font-weight: 600; }",
    ".nav-group-fold > summary { cursor: pointer; padding: 6px 10px; font-size: 13px; font-weight: 600; color: var(--muted); border-radius: var(--r-sm); list-style: none; }",
    ".nav-group-fold > summary::-webkit-details-marker { display: none; }",
    ".nav-group-fold > summary::before { content: '▸ '; }",
    ".nav-group-fold[open] > summary::before { content: '▾ '; }",
    ".nav-group-fold > summary:hover { background: var(--surface-subtle); color: var(--title); }",
    ".docs-main { min-width: 0; }",
    "article { max-width: 68ch; }",
    "article h1, article h2, article h3, article h4 { color: var(--title); font-weight: 700; letter-spacing: -0.02em; line-height: 1.25; scroll-margin-top: calc(var(--header-h) + 20px); }",
    "article h1 { font-size: clamp(1.75rem, 3vw, 2.25rem); margin: 0 0 16px; }",
    "article h2 { font-size: 1.35rem; margin: 36px 0 12px; padding-top: 20px; border-top: 1px solid var(--border); }",
    "article h3 { font-size: 1.08rem; margin: 24px 0 8px; }",
    "article p { margin: 0 0 16px; }",
    "article ul, article ol { margin: 0 0 16px; padding-left: 22px; }",
    "article li { margin: 6px 0; }",
    "article a { color: var(--accent); text-decoration: underline; text-underline-offset: 3px; text-decoration-thickness: 1px; }",
    "article a:visited { color: var(--visited); }",
    "article a:hover { color: var(--accent-hover); }",
    "article hr { border: 0; border-top: 1px solid var(--border); margin: 28px 0; }",
    "article blockquote { margin: 0 0 18px; padding: 10px 16px; border-left: 3px solid var(--accent); background: var(--surface-subtle); border-radius: 0 var(--r-sm) var(--r-sm) 0; color: var(--text); }",
    "article blockquote p:last-child { margin-bottom: 0; }",
    "code, pre { font-family: var(--mono); }",
    "code { font-size: 0.875em; padding: 2px 6px; border-radius: 5px; background: var(--surface-subtle); border: 1px solid var(--border); color: var(--title); overflow-wrap: anywhere; }",
    "pre { position: relative; margin: 0 0 20px; padding: 16px 68px 16px 18px; overflow-x: auto; border-radius: var(--r-md); background: var(--surface); border: 1px solid var(--border); box-shadow: var(--shadow-sm); line-height: 1.5; }",
    "pre code { padding: 0; border: 0; background: transparent; font-size: 13.5px; color: var(--text); overflow-wrap: normal; }",
    ".code-copy { position: absolute; top: 8px; right: 8px; padding: 4px 10px; border: 1px solid var(--border); border-radius: var(--r-sm); background: var(--surface-subtle); color: var(--text); font: 500 12px/1.2 var(--sans); cursor: pointer; }",
    ".code-copy:hover { border-color: var(--border-strong); color: var(--title); }",
    ".code-copy:active { transform: translateY(1px); }",
    ".table-wrap { width: 100%; overflow-x: auto; margin: 0 0 20px; border: 1px solid var(--border); border-radius: var(--r-md); background: var(--surface); }",
    ".table-wrap table { width: 100%; min-width: 540px; border-collapse: collapse; font-size: 14px; text-align: left; }",
    ".table-wrap th, .table-wrap td { padding: 10px 14px; border-bottom: 1px solid var(--border); vertical-align: top; }",
    ".table-wrap th { font-weight: 600; color: var(--title); background: var(--surface-subtle); white-space: nowrap; }",
    ".table-wrap tr:last-child td { border-bottom: 0; }",
    ".table-wrap td code { white-space: nowrap; }",
    ".notice { border-left: 3px solid var(--warn); background: var(--warn-bg); border-radius: 0 var(--r-sm) var(--r-sm) 0; padding: 12px 16px; margin: 0 0 18px; }",
    ".unresolved { color: var(--muted); }",
    ".site-footer { border-top: 1px solid var(--border); background: var(--surface); color: var(--muted); font-size: 13px; }",
    ".site-footer__inner { max-width: 1320px; margin: 0 auto; padding: 20px 24px; display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 12px; }",
    ".site-footer a { color: var(--muted); text-decoration: none; }",
    ".site-footer a:hover { color: var(--title); text-decoration: underline; }",
    ".site-footer__links { display: flex; flex-wrap: wrap; gap: 16px; }",
    "@media (max-width: 960px) {",
    "  .docs-shell { grid-template-columns: 1fr; gap: 24px; padding: 20px 16px 48px; }",
    "  .nav-toggle { display: inline-flex; }",
    "  .site-header__links { display: none; }",
    "  .site-header__inner { padding: 8px 16px; gap: 10px; }",
    "  .site-search { flex: 1 1 auto; width: auto; }",
    "  .site-search input { height: 44px; font-size: 16px; }",
    "  .docs-sidebar { display: none; position: static; max-height: none; padding: 14px; border: 1px solid var(--border); border-radius: var(--r-md); background: var(--surface); }",
    "  .docs-sidebar.is-open { display: block; }",
    "  .nav-group a, .nav-group-fold > summary { min-height: 44px; display: flex; align-items: center; }",
    "}",
    "@media (prefers-reduced-motion: reduce) {",
    "  *, *::before, *::after { animation-duration: 0.001ms !important; transition-duration: 0.001ms !important; scroll-behavior: auto !important; }",
    "  .code-copy:active { transform: none !important; }",
    "}",
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
    .replace(/\[([^\]]+)\]\([^)\s]+\)/gu, "$1")
    .trim();
}

function documentDescription(content: string): string {
  const lines = content.split(/\r?\n/u);
  let inFence = false;
  for (const line of lines) {
    if (line.trimStart().startsWith("```")) {
      inFence = !inFence;
      continue;
    }
    if (inFence) continue;
    const trimmed = line.trim();
    if (
      trimmed === "" ||
      trimmed.startsWith("#") ||
      trimmed.startsWith("<!--") ||
      trimmed.startsWith("|") ||
      trimmed.startsWith("-") ||
      trimmed.startsWith("*") ||
      /^\d+\.\s/u.test(trimmed)
    ) {
      continue;
    }
    const text = plainText(trimmed);
    if (text.length > 0) {
      return text.length > 160 ? `${text.slice(0, 157)}...` : text;
    }
  }
  return "Xerxes documentation for the Bun-native TypeScript agent runtime.";
}

function headingSlug(raw: string, used: Set<string>): string {
  const base =
    plainText(raw)
      .toLowerCase()
      .replace(/[^a-z0-9]+/gu, "-")
      .replace(/^-+|-+$/gu, "") || "section";
  let candidate = base;
  let suffix = 2;
  while (used.has(candidate)) {
    candidate = `${base}-${suffix}`;
    suffix += 1;
  }
  used.add(candidate);
  return candidate;
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
  description = "Xerxes documentation for the Bun-native TypeScript agent runtime.",
): string {
  const styles = stylePaths
    .map(
      (path) =>
        `<link rel="stylesheet" href="${escapeHtml(relativeHref(outputPath, path))}">`,
    )
    .join("\n    ");
  const isReportDoc = (document: SourceDocument): boolean =>
    /(?:\b\d{4}-\d{2}-\d{2}\b|audit|report|benchmark|parity|plan|evidence|roadmap|repair|matrix|survives)/iu.test(
      `${document.sourcePath} ${document.title}`,
    );
  const startOrder = new Map<string, number>([
    ["index.html", 0],
    ["configuration-guide.md", 1],
    ["goal-mode.md", 2],
    ["BUN_BUILD.md", 3],
  ]);
  const startDocs = navigation
    .filter(
      (doc) =>
        doc.outputPath === "index.html" ||
        doc.sourcePath === "BUN_BUILD.md" ||
        doc.sourcePath === "configuration-guide.md" ||
        doc.sourcePath === "goal-mode.md",
    )
    .sort((a, b) => {
      const rankA = startOrder.get(a.outputPath) ?? startOrder.get(a.sourcePath) ?? 99;
      const rankB = startOrder.get(b.outputPath) ?? startOrder.get(b.sourcePath) ?? 99;
      return rankA - rankB;
    });
  const startPaths = new Set(startDocs.map((doc) => doc.outputPath));
  const apiDocs = navigation.filter((doc) => doc.kind === "typescript-api");
  const apiPaths = new Set(apiDocs.map((doc) => doc.outputPath));
  const reportDocs = navigation.filter(
    (doc) => !startPaths.has(doc.outputPath) && !apiPaths.has(doc.outputPath) && isReportDoc(doc),
  );
  const reportPaths = new Set(reportDocs.map((doc) => doc.outputPath));
  const guideDocs = navigation.filter(
    (doc) =>
      !startPaths.has(doc.outputPath) &&
      !apiPaths.has(doc.outputPath) &&
      !reportPaths.has(doc.outputPath),
  );
  const renderNavItem = (item: { readonly outputPath: string; readonly title: string }): string => {
    const href = relativeHref(outputPath, item.outputPath);
    const current = item.outputPath === outputPath ? ' aria-current="page"' : "";
    return `<li><a href="${escapeHtml(href)}"${current}>${escapeHtml(item.title)}</a></li>`;
  };
  const primarySection =
    startDocs.length > 0
      ? `<div class="nav-group"><p class="nav-group__title">Start here</p><ul>${startDocs.map(renderNavItem).join("")}</ul></div>`
      : "";
  const guidesSection =
    guideDocs.length > 0
      ? `<div class="nav-group"><p class="nav-group__title">Guides</p><ul>${guideDocs.map(renderNavItem).join("")}</ul></div>`
      : "";
  const referenceItems = [
    ...apiDocs.map((doc) => ({ outputPath: doc.outputPath, title: doc.title })),
    { outputPath: "sitemap.html", title: "Site map" },
    { outputPath: "documentation-report.html", title: "Build report" },
  ];
  const referenceSection = `<div class="nav-group"><p class="nav-group__title">Reference</p><ul>${referenceItems.map(renderNavItem).join("")}</ul></div>`;
  const reportsSection =
    reportDocs.length > 0
      ? `<div class="nav-group"><details class="nav-group-fold"${reportPaths.has(outputPath) ? " open" : ""}><summary>Architecture notes &amp; reports (${reportDocs.length})</summary><ul>${reportDocs.map(renderNavItem).join("")}</ul></details></div>`
      : "";
  const apiIndexDoc = apiDocs[0];
  const topLinks = [
    { outputPath: "index.html", title: "Docs" },
    ...(apiIndexDoc === undefined ? [] : [{ outputPath: apiIndexDoc.outputPath, title: "API" }]),
    { outputPath: "sitemap.html", title: "Site map" },
  ]
    .map((item) => {
      const href = relativeHref(outputPath, item.outputPath);
      const current = item.outputPath === outputPath ? ' aria-current="page"' : "";
      return `<a href="${escapeHtml(href)}"${current}>${escapeHtml(item.title)}</a>`;
    })
    .join("");
  const rootPrefix = (() => {
    const dir = posix.dirname(outputPath);
    if (dir === "." || dir === "") return "";
    return `${dir.split("/").map(() => "..").join("/")}/`;
  })();
  const indexUrl = relativeHref(outputPath, "assets/search-index.json");
  const scriptHref = relativeHref(outputPath, "assets/docs.js");
  const faviconHref = relativeHref(outputPath, "assets/favicon.svg");
  const pageTitle = `${title} · ${siteTitle}`;
  return [
    "<!doctype html>",
    '<html lang="en">',
    "<head>",
    '  <meta charset="utf-8">',
    '  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">',
    `  <meta name="description" content="${escapeHtml(description)}">`,
    '  <meta name="theme-color" media="(prefers-color-scheme: light)" content="#f7f7f8">',
    '  <meta name="theme-color" media="(prefers-color-scheme: dark)" content="#0c0d0f">',
    `  <meta property="og:title" content="${escapeHtml(pageTitle)}">`,
    `  <meta property="og:description" content="${escapeHtml(description)}">`,
    '  <meta property="og:type" content="website">',
    '  <meta name="twitter:card" content="summary">',
    `  <title>${escapeHtml(pageTitle)}</title>`,
    `  <link rel="icon" type="image/svg+xml" href="${escapeHtml(faviconHref)}">`,
    `    ${styles}`,
    "</head>",
    "<body>",
    '  <a class="skip-link" href="#main-content">Skip to main content</a>',
    '  <header class="site-header">',
    '    <div class="site-header__inner">',
    `      <a class="site-brand" href="${escapeHtml(relativeHref(outputPath, "index.html"))}"><span class="site-brand__mark" aria-hidden="true">X</span><span>${escapeHtml(siteTitle)}</span></a>`,
    `      <div class="site-header__links">${topLinks}</div>`,
    `      <form class="site-search" role="search" data-docs-search data-index-url="${escapeHtml(indexUrl)}" data-root-prefix="${escapeHtml(rootPrefix)}">`,
    '        <input type="search" aria-label="Search documentation" placeholder="Search docs (/)">',
    '        <div id="docs-search-results" class="search-results" role="region" aria-label="Search results" hidden></div>',
    "      </form>",
    '      <button type="button" class="nav-toggle" data-nav-toggle aria-expanded="false" aria-controls="docs-sidebar">Menu</button>',
    "    </div>",
    "  </header>",
    '  <div class="docs-shell">',
    '    <aside id="docs-sidebar" class="docs-sidebar">',
    `      <nav aria-label="Site">${primarySection}${guidesSection}${referenceSection}${reportsSection}</nav>`,
    "    </aside>",
    '    <main id="main-content" class="docs-main" tabindex="-1">',
    `      <article data-source="${escapeHtml(outputPath)}">`,
    body,
    "      </article>",
    "    </main>",
    "  </div>",
    '  <footer class="site-footer">',
    '    <div class="site-footer__inner">',
    "      <span>Xerxes-Agents documentation. Licensed under the Apache License, Version 2.0. Generated deterministically by the native Bun documentation builder.</span>",
    `      <span class="site-footer__links"><a href="${escapeHtml(relativeHref(outputPath, "sitemap.html"))}">Site map</a><a href="${escapeHtml(relativeHref(outputPath, "documentation-report.html"))}">Build report</a></span>`,
    "    </div>",
    "  </footer>",
    `  <script src="${escapeHtml(scriptHref)}" defer></script>`,
    "</body>",
    "</html>",
    "",
  ].join("\n");
}

function renderInline(value: string): string {
  const codeTokens: string[] = [];
  const withCodePlaceholders = value.replace(/`([^`]+)`/gu, (_match, code: string) => {
    const token = `\u0000CODE${codeTokens.length}\u0000`;
    codeTokens.push(`<code>${escapeHtml(code)}</code>`);
    return token;
  });
  const escaped = escapeHtml(withCodePlaceholders);
  const linked = escaped.replace(
    /\[([^\]]+)\]\(([^)\s]+)\)/gu,
    (_match, label: string, href: string) => {
      const destination = documentationHref(href);
      return `<a href="${destination}">${label}</a>`;
    },
  );
  const formatted = linked
    .replace(/\*\*([^*]+)\*\*/gu, "<strong>$1</strong>")
    .replace(/(?<![\w*/\\])\*([^\s*](?:[^*]*[^\s*])?)\*(?![\w*])/gu, "<em>$1</em>");
  return formatted.replace(/\u0000CODE(\d+)\u0000/gu, (_match, index: string) => {
    return codeTokens[Number(index)] ?? "";
  });
}

function isTableDivider(line: string | undefined): boolean {
  return line !== undefined && /^\s*\|?[\s:|-]*-{3,}[\s:|-]*\|?\s*$/u.test(line);
}

function splitTableRow(line: string): string[] {
  return line
    .trim()
    .replace(/^\|/u, "")
    .replace(/\|$/u, "")
    .split("|")
    .map((cell) => cell.trim());
}

function renderMarkdownBody(source: string): string {
  const lines = source.split(/\r?\n/u);
  const output: string[] = [];
  const paragraph: string[] = [];
  const usedHeadingIds = new Set<string>();
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

  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index]!;
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
    if (/^\s*<!--[\s\S]*-->\s*$/u.test(line)) {
      flushParagraph();
      flushList();
      continue;
    }
    if (line.trimStart().startsWith("|") && isTableDivider(lines[index + 1])) {
      flushParagraph();
      flushList();
      const headers = splitTableRow(line);
      const rows: string[][] = [];
      let rowIndex = index + 2;
      while (rowIndex < lines.length && lines[rowIndex]!.trimStart().startsWith("|")) {
        rows.push(splitTableRow(lines[rowIndex]!));
        rowIndex += 1;
      }
      const headHtml = `<thead><tr>${headers.map((cell) => `<th>${renderInline(cell)}</th>`).join("")}</tr></thead>`;
      const bodyHtml =
        rows.length > 0
          ? `<tbody>${rows.map((row) => `<tr>${row.map((cell) => `<td>${renderInline(cell)}</td>`).join("")}</tr>`).join("")}</tbody>`
          : "";
      output.push(`<div class="table-wrap" tabindex="0"><table>${headHtml}${bodyHtml}</table></div>`);
      index = rowIndex - 1;
      continue;
    }
    const heading = line.match(/^(#{1,6})\s+(.+)$/u);
    if (heading !== null) {
      flushParagraph();
      flushList();
      const level = heading[1]?.length ?? 1;
      const content = heading[2] ?? "";
      const idAttr = level === 2 ? ` id="${escapeHtml(headingSlug(content, usedHeadingIds))}"` : "";
      output.push(`<h${level}${idAttr}>${renderInline(content)}</h${level}>`);
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
