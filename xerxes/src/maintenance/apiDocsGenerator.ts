// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { randomUUID } from 'node:crypto'
import { chmod, mkdir, readFile, readdir, rename, rm, rmdir, stat, writeFile } from 'node:fs/promises'
import { basename, dirname, isAbsolute, join, relative, resolve, sep } from 'node:path'
import { fileURLToPath } from 'node:url'

import ts from 'typescript'

export const API_DOCS_MANIFEST_FILE = '.xerxes-typescript-api-docs.json'
export const API_DOCS_MANIFEST_VERSION = 1
export const API_DOCS_GENERATOR_NAME = 'xerxes-typescript-api-docs'
export const DEFAULT_TYPESCRIPT_PACKAGE_NAME = '@xerxes/runtime'

const PACKAGE_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../..')
const DEFAULT_SOURCE_DIRECTORY = join(PACKAGE_ROOT, 'src')
const DEFAULT_OUTPUT_DIRECTORY = resolve(PACKAGE_ROOT, '../docs/typescript-api')
const SKIPPED_DIRECTORIES = new Set(['.git', '.cache', '.turbo', 'coverage', 'dist', 'node_modules'])
const SOURCE_SUFFIXES = ['.d.mts', '.d.cts', '.d.ts', '.tsx', '.mts', '.cts', '.ts'] as const

export type ApiDocsChangeKind = 'created' | 'deleted' | 'unchanged' | 'updated'
export type TypeScriptExportKind = 'class' | 'const' | 'default' | 'enum' | 'function' | 'interface' | 'namespace'
  | 're-export' | 'type' | 'variable'

export interface TypeScriptModule {
  readonly absolutePath: string
  readonly documentationPath: string
  readonly moduleName: string
  readonly packagePath: string
  readonly relativePath: string
  readonly sourceRelativePath: string
}

export interface TypeScriptApiSymbol {
  readonly declaration: string
  readonly documentation?: string
  readonly kind: TypeScriptExportKind
  readonly name: string
}

export interface DocumentedTypeScriptModule extends TypeScriptModule {
  readonly exports: readonly TypeScriptApiSymbol[]
}

export interface TypeScriptApiDocsOptions {
  readonly atomic?: boolean
  readonly clean?: boolean
  readonly dryRun?: boolean
  readonly outputDirectory: string
  readonly packageName?: string
  readonly sourceDirectory: string
}

export interface ApiDocsChange {
  readonly action: ApiDocsChangeKind
  readonly path: string
}

export interface TypeScriptApiDocsResult {
  readonly changed: boolean
  readonly changes: readonly ApiDocsChange[]
  readonly dryRun: boolean
  readonly modules: readonly DocumentedTypeScriptModule[]
  readonly outputDirectory: string
}

export interface TypeScriptFormatterPort {
  format(paths: readonly string[], options: { readonly fix: boolean }): Promise<void>
}

export interface TypeScriptFormattingResult {
  readonly paths: readonly string[]
}

interface NormalizedOptions {
  readonly atomic: boolean
  readonly clean: boolean
  readonly dryRun: boolean
  readonly outputDirectory: string
  readonly packageName: string
  readonly sourceDirectory: string
}

interface ApiDocsManifest {
  readonly files: readonly string[]
  readonly generator: typeof API_DOCS_GENERATOR_NAME
  readonly packageName: string
  readonly version: typeof API_DOCS_MANIFEST_VERSION
}

interface PackagePage {
  readonly childPackages: readonly string[]
  readonly modulePaths: readonly string[]
  readonly packagePath: string
}

/** Error raised for an invalid source tree or an unsafe documentation destination. */
export class ApiDocsGenerationError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'ApiDocsGenerationError'
  }
}

/** Discover supported TypeScript source modules below one package source directory. */
export async function discoverTypeScriptModules(
  sourceDirectory: string,
  packageName = DEFAULT_TYPESCRIPT_PACKAGE_NAME,
): Promise<TypeScriptModule[]> {
  const root = resolve(sourceDirectory)
  const sourceStat = await safeStat(root)
  if (sourceStat === undefined || !sourceStat.isDirectory()) {
    throw new ApiDocsGenerationError(`TypeScript source directory does not exist: ${root}`)
  }

  const normalizedPackageName = normalizePackageName(packageName)
  const paths: string[] = []
  await collectSourceFiles(root, paths)

  return paths.sort(compareText).map(absolutePath => {
    const sourceRelativePath = toPosixPath(relative(root, absolutePath))
    const relativePath = removeSourceSuffix(sourceRelativePath)
    const packagePath = parentPath(relativePath)
    return {
      absolutePath,
      documentationPath: documentationPathFor(relativePath),
      moduleName: moduleNameFor(relativePath, normalizedPackageName),
      packagePath,
      relativePath,
      sourceRelativePath,
    }
  })
}

/** Parse one TypeScript module and return a deterministic summary of its public exports. */
export function extractTypeScriptExports(source: string, sourcePath = 'module.ts'): TypeScriptApiSymbol[] {
  const sourceFile = ts.createSourceFile(sourcePath, source, ts.ScriptTarget.Latest, true, scriptKindFor(sourcePath))
  const parseDiagnostics = (sourceFile as ts.SourceFile & { readonly parseDiagnostics: readonly ts.Diagnostic[] }).parseDiagnostics
  if (parseDiagnostics.length > 0) {
    const messages = parseDiagnostics
      .map(diagnostic => ts.flattenDiagnosticMessageText(diagnostic.messageText, ' '))
      .join('; ')
    throw new ApiDocsGenerationError(`Cannot document syntactically invalid TypeScript source (${sourcePath}): ${messages}`)
  }

  const symbols: TypeScriptApiSymbol[] = []
  for (const statement of sourceFile.statements) {
    collectExportedStatement(symbols, statement, sourceFile)
  }
  return symbols.sort(compareSymbols)
}

/** Document a discovered TypeScript module without evaluating or importing application code. */
export async function documentTypeScriptModule(module: TypeScriptModule): Promise<DocumentedTypeScriptModule> {
  const source = await readUtf8(module.absolutePath)
  return { ...module, exports: extractTypeScriptExports(source, module.sourceRelativePath) }
}

/**
 * Generate Markdown API pages and nested package indexes from a TypeScript source tree.
 *
 * The output directory is manifest-owned: a non-empty destination without this generator's
 * manifest is rejected instead of being overwritten. `clean` removes only stale files listed
 * in the prior manifest, never arbitrary files placed beside the generated documentation.
 */
export async function generateTypeScriptApiDocs(options: TypeScriptApiDocsOptions): Promise<TypeScriptApiDocsResult> {
  const normalized = normalizeOptions(options)
  assertSeparateDirectories(normalized.sourceDirectory, normalized.outputDirectory)

  const modules = await discoverTypeScriptModules(normalized.sourceDirectory, normalized.packageName)
  if (modules.length === 0) {
    throw new ApiDocsGenerationError(`No TypeScript modules found below: ${normalized.sourceDirectory}`)
  }

  const documentedModules: DocumentedTypeScriptModule[] = []
  for (const module of modules) documentedModules.push(await documentTypeScriptModule(module))

  const pages = buildDocumentationPages(documentedModules, normalized.packageName)
  const manifest = await readManifest(normalized.outputDirectory)
  await assertOutputOwnership(normalized.outputDirectory, manifest)

  const currentPaths = [...pages.keys()].sort(compareText)
  const currentPathSet = new Set(currentPaths)
  const previousPaths = manifest?.files ?? []
  const stalePaths = normalized.clean
    ? previousPaths.filter(path => !currentPathSet.has(path)).sort(compareText)
    : []
  const manifestPaths = normalized.clean
    ? currentPaths
    : [...new Set([...previousPaths, ...currentPaths])].sort(compareText)
  const manifestText = renderManifest(normalized.packageName, manifestPaths)
  const changes: ApiDocsChange[] = []

  for (const stalePath of stalePaths) {
    const deleted = await deleteGeneratedFile(normalized.outputDirectory, stalePath, normalized)
    if (deleted) changes.push({ action: 'deleted', path: outputPath(normalized.outputDirectory, stalePath) })
  }

  for (const relativePath of currentPaths) {
    const page = pages.get(relativePath)
    if (page === undefined) throw new ApiDocsGenerationError(`Missing generated page: ${relativePath}`)
    changes.push(await writeGeneratedFile(normalized.outputDirectory, relativePath, page, normalized))
  }
  changes.push(await writeGeneratedFile(normalized.outputDirectory, API_DOCS_MANIFEST_FILE, manifestText, normalized))

  return {
    changed: changes.some(change => change.action !== 'unchanged'),
    changes: changes.sort((left, right) => compareText(left.path, right.path)),
    dryRun: normalized.dryRun,
    modules: documentedModules,
    outputDirectory: normalized.outputDirectory,
  }
}

/**
 * Invoke a caller-supplied formatter for discovered TypeScript files.
 *
 * This is intentionally an injected boundary. The documentation generator never spawns
 * formatter commands, shells out to Bun, or delegates to Python; an application may wire a
 * formatter adapter here if formatting is desired as part of a larger workflow.
 */
export async function formatTypeScriptSources(
  sourceDirectory: string,
  formatter: TypeScriptFormatterPort,
  fix = true,
): Promise<TypeScriptFormattingResult> {
  const modules = await discoverTypeScriptModules(sourceDirectory)
  const paths = modules.map(module => module.absolutePath)
  await formatter.format(paths, { fix })
  return { paths }
}

/** Render one Markdown module page from parsed TypeScript exports. */
export function renderModulePage(module: DocumentedTypeScriptModule): string {
  const lines = [
    '<!-- Generated by Xerxes TypeScript API documentation. Do not edit directly. -->',
    '',
    `# ${module.moduleName}`,
    '',
    `Source: \`${module.sourceRelativePath}\``,
    '',
    '## Exports',
    '',
  ]

  if (module.exports.length === 0) {
    lines.push('_This module does not declare public exports._', '')
    return lines.join('\n')
  }

  for (const symbol of module.exports) {
    lines.push(`### \`${symbol.name}\` (${symbol.kind})`, '')
    if (symbol.documentation !== undefined) lines.push(symbol.documentation, '')
    const fence = codeFenceFor(symbol.declaration)
    lines.push(`${fence}ts`, symbol.declaration, fence, '')
  }
  return lines.join('\n')
}

/** Render a stable package index whose child packages always precede direct modules. */
export function renderPackageIndex(page: PackagePage, modules: readonly DocumentedTypeScriptModule[], packageName: string): string {
  const title = page.packagePath === '' ? `${packageName} API Reference` : `${packageName}/${page.packagePath} package`
  const lines = [
    '<!-- Generated by Xerxes TypeScript API documentation. Do not edit directly. -->',
    '',
    `# ${title}`,
    '',
  ]

  if (page.childPackages.length > 0) {
    lines.push('## Packages', '')
    for (const childPackage of page.childPackages) {
      const label = finalPathComponent(childPackage)
      lines.push(`- [\`${label}\`](${relativeIndexLink(page.packagePath, childPackage)})`)
    }
    lines.push('')
  }

  if (page.modulePaths.length > 0) {
    lines.push('## Modules', '')
    const modulesByPath = new Map(modules.map(module => [module.documentationPath, module]))
    for (const modulePath of page.modulePaths) {
      const module = modulesByPath.get(modulePath)
      if (module === undefined) throw new ApiDocsGenerationError(`Package index references unknown module: ${modulePath}`)
      lines.push(`- [\`${module.moduleName}\`](${relativeModuleLink(page.packagePath, module.documentationPath)})`)
    }
    lines.push('')
  }

  if (page.childPackages.length === 0 && page.modulePaths.length === 0) {
    lines.push('_No TypeScript modules were discovered in this package._', '')
  }
  return lines.join('\n')
}

/** Run the standalone docs command; formatting remains an application-owned injected boundary. */
export async function main(args: readonly string[] = process.argv.slice(2)): Promise<number> {
  try {
    const parsed = parseArguments(args)
    if (parsed === undefined) return 0
    const result = await generateTypeScriptApiDocs(parsed)
    const changed = result.changes.filter(change => change.action !== 'unchanged')
    const verb = result.dryRun ? 'Would update' : 'Updated'
    console.log(`${verb} TypeScript API docs for ${result.modules.length} modules.`)
    for (const change of changed) {
      const path = relative(result.outputDirectory, change.path) || change.path
      console.log(`  ${change.action.toUpperCase()}: ${path}`)
    }
    return 0
  } catch (error) {
    console.error(`typescript-api-docs: ${errorMessage(error)}`)
    return 1
  }
}

if (import.meta.main) {
  process.exitCode = await main()
}

function buildDocumentationPages(modules: readonly DocumentedTypeScriptModule[], packageName: string): Map<string, string> {
  const pages = new Map<string, string>()
  for (const module of modules) addPage(pages, module.documentationPath, renderModulePage(module))

  const packagePages = buildPackagePages(modules)
  for (const page of packagePages) {
    const indexPath = page.packagePath === '' ? 'index.md' : `${page.packagePath}/index.md`
    addPage(pages, indexPath, renderPackageIndex(page, modules, packageName))
  }
  return pages
}

function buildPackagePages(modules: readonly DocumentedTypeScriptModule[]): PackagePage[] {
  const children = new Map<string, Set<string>>()
  const modulePaths = new Map<string, Set<string>>()
  const packages = new Set<string>([''])

  for (const module of modules) {
    packages.add(module.packagePath)
    if (!modulePaths.has(module.packagePath)) modulePaths.set(module.packagePath, new Set())
    modulePaths.get(module.packagePath)!.add(module.documentationPath)

    const segments = module.packagePath === '' ? [] : module.packagePath.split('/')
    for (let index = 0; index < segments.length; index += 1) {
      const packagePath = segments.slice(0, index + 1).join('/')
      const parent = segments.slice(0, index).join('/')
      packages.add(packagePath)
      if (!children.has(parent)) children.set(parent, new Set())
      children.get(parent)!.add(packagePath)
    }
  }

  return [...packages]
    .sort(compareText)
    .map(packagePath => ({
      childPackages: [...(children.get(packagePath) ?? [])].sort(compareText),
      modulePaths: [...(modulePaths.get(packagePath) ?? [])].sort(compareText),
      packagePath,
    }))
}

function collectExportedStatement(symbols: TypeScriptApiSymbol[], statement: ts.Statement, sourceFile: ts.SourceFile): void {
  if (ts.isExportDeclaration(statement)) {
    collectReExportSymbols(symbols, statement, sourceFile)
    return
  }
  if (ts.isExportAssignment(statement)) {
    symbols.push(symbolFor(
      statement.isExportEquals ? 'export =' : 'default',
      'default',
      statement,
      sourceFile,
    ))
    return
  }
  if (ts.isVariableStatement(statement) && hasExportModifier(statement)) {
    const kind: TypeScriptExportKind = (statement.declarationList.flags & ts.NodeFlags.Const) !== 0 ? 'const' : 'variable'
    for (const declaration of statement.declarationList.declarations) {
      symbols.push(symbolFor(declaration.name.getText(sourceFile), kind, statement, sourceFile))
    }
    return
  }
  if (ts.isFunctionDeclaration(statement) && hasExportModifier(statement)) {
    symbols.push(symbolFor(exportedName(statement), 'function', statement, sourceFile))
    return
  }
  if (ts.isClassDeclaration(statement) && hasExportModifier(statement)) {
    symbols.push(symbolFor(exportedName(statement), 'class', statement, sourceFile))
    return
  }
  if (ts.isInterfaceDeclaration(statement) && hasExportModifier(statement)) {
    symbols.push(symbolFor(exportedName(statement), 'interface', statement, sourceFile))
    return
  }
  if (ts.isTypeAliasDeclaration(statement) && hasExportModifier(statement)) {
    symbols.push(symbolFor(exportedName(statement), 'type', statement, sourceFile))
    return
  }
  if (ts.isEnumDeclaration(statement) && hasExportModifier(statement)) {
    symbols.push(symbolFor(exportedName(statement), 'enum', statement, sourceFile))
    return
  }
  if (ts.isModuleDeclaration(statement) && hasExportModifier(statement)) {
    symbols.push(symbolFor(exportedName(statement), 'namespace', statement, sourceFile))
    return
  }
  if (ts.isImportEqualsDeclaration(statement) && hasExportModifier(statement)) {
    symbols.push(symbolFor(exportedName(statement), 'namespace', statement, sourceFile))
  }
}

function collectReExportSymbols(symbols: TypeScriptApiSymbol[], statement: ts.ExportDeclaration, sourceFile: ts.SourceFile): void {
  const clause = statement.exportClause
  if (clause === undefined) {
    symbols.push(symbolFor('*', 're-export', statement, sourceFile))
    return
  }
  if (ts.isNamespaceExport(clause)) {
    symbols.push(symbolFor(clause.name.text, 're-export', statement, sourceFile))
    return
  }
  for (const element of clause.elements) {
    symbols.push(symbolFor(element.name.text, 're-export', statement, sourceFile))
  }
}

function symbolFor(
  name: string,
  kind: TypeScriptExportKind,
  statement: ts.Node,
  sourceFile: ts.SourceFile,
): TypeScriptApiSymbol {
  const documentation = documentationFor(statement, sourceFile)
  return {
    declaration: statement.getText(sourceFile).trim(),
    ...(documentation === undefined ? {} : { documentation }),
    kind,
    name,
  }
}

function hasExportModifier(statement: { readonly modifiers?: ts.NodeArray<ts.ModifierLike> }): boolean {
  return statement.modifiers?.some(modifier => modifier.kind === ts.SyntaxKind.ExportKeyword) === true
}

function hasDefaultModifier(statement: { readonly modifiers?: ts.NodeArray<ts.ModifierLike> }): boolean {
  return statement.modifiers?.some(modifier => modifier.kind === ts.SyntaxKind.DefaultKeyword) === true
}

function exportedName(statement: { readonly modifiers?: ts.NodeArray<ts.ModifierLike>; readonly name?: ts.DeclarationName }): string {
  if (hasDefaultModifier(statement)) return 'default'
  return statement.name?.getText() ?? 'default'
}

function documentationFor(statement: ts.Node, sourceFile: ts.SourceFile): string | undefined {
  const leading = sourceFile.text.slice(statement.getFullStart(), statement.getStart(sourceFile))
  const match = /\/\*\*([\s\S]*?)\*\/\s*$/u.exec(leading)
  if (match === null) return undefined
  const body = match[1]
  if (body === undefined) return undefined
  const documentation = body
    .split(/\r?\n/u)
    .map(line => line.replace(/^\s*\* ?/u, '').trimEnd())
    .join('\n')
    .trim()
  return documentation === '' ? undefined : documentation
}

function normalizeOptions(options: TypeScriptApiDocsOptions): NormalizedOptions {
  const sourceDirectory = resolve(options.sourceDirectory)
  const outputDirectory = resolve(options.outputDirectory)
  return {
    atomic: options.atomic ?? true,
    clean: options.clean ?? true,
    dryRun: options.dryRun ?? false,
    outputDirectory,
    packageName: normalizePackageName(options.packageName ?? DEFAULT_TYPESCRIPT_PACKAGE_NAME),
    sourceDirectory,
  }
}

function normalizePackageName(packageName: string): string {
  const normalized = packageName.trim().replace(/\/+$/u, '')
  if (normalized === '') throw new ApiDocsGenerationError('packageName must not be empty.')
  return normalized
}

function documentationPathFor(relativePath: string): string {
  return relativePath === 'index' || relativePath.endsWith('/index') ? `${relativePath}.api.md` : `${relativePath}.md`
}

function moduleNameFor(relativePath: string, packageName: string): string {
  const withoutIndex = relativePath === 'index'
    ? ''
    : relativePath.endsWith('/index')
      ? relativePath.slice(0, -'/index'.length)
      : relativePath
  return withoutIndex === '' ? packageName : `${packageName}/${withoutIndex}`
}

function parentPath(path: string): string {
  const separatorIndex = path.lastIndexOf('/')
  return separatorIndex === -1 ? '' : path.slice(0, separatorIndex)
}

function scriptKindFor(path: string): ts.ScriptKind {
  const lower = path.toLowerCase()
  if (lower.endsWith('.tsx')) return ts.ScriptKind.TSX
  return ts.ScriptKind.TS
}

function removeSourceSuffix(path: string): string {
  const suffix = SOURCE_SUFFIXES.find(candidate => path.toLowerCase().endsWith(candidate))
  if (suffix === undefined) throw new ApiDocsGenerationError(`Unsupported TypeScript source suffix: ${path}`)
  return path.slice(0, -suffix.length)
}

function isSourceFile(name: string): boolean {
  const lower = name.toLowerCase()
  if (lower.endsWith('.d.ts') || lower.endsWith('.d.mts') || lower.endsWith('.d.cts')) return false
  return lower.endsWith('.ts') || lower.endsWith('.tsx') || lower.endsWith('.mts') || lower.endsWith('.cts')
}

async function collectSourceFiles(directory: string, paths: string[]): Promise<void> {
  const entries = await readdir(directory, { withFileTypes: true })
  for (const entry of entries.sort((left, right) => compareText(left.name, right.name))) {
    const path = join(directory, entry.name)
    if (entry.isDirectory()) {
      if (!SKIPPED_DIRECTORIES.has(entry.name)) await collectSourceFiles(path, paths)
      continue
    }
    if (entry.isFile() && isSourceFile(entry.name)) paths.push(path)
  }
}

function renderManifest(packageName: string, files: readonly string[]): string {
  const manifest: ApiDocsManifest = {
    files,
    generator: API_DOCS_GENERATOR_NAME,
    packageName,
    version: API_DOCS_MANIFEST_VERSION,
  }
  return `${JSON.stringify(manifest, null, 2)}\n`
}

async function readManifest(outputDirectory: string): Promise<ApiDocsManifest | undefined> {
  const path = outputPath(outputDirectory, API_DOCS_MANIFEST_FILE)
  let raw: string
  try {
    raw = await readUtf8(path)
  } catch (error) {
    if (isMissingFile(error)) return undefined
    throw error
  }

  let parsed: unknown
  try {
    parsed = JSON.parse(raw)
  } catch {
    throw new ApiDocsGenerationError(`Documentation manifest is not valid JSON: ${path}`)
  }
  if (!isManifest(parsed)) {
    throw new ApiDocsGenerationError(`Documentation manifest has an unsupported shape: ${path}`)
  }
  return parsed
}

function isManifest(value: unknown): value is ApiDocsManifest {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return false
  const record = value as Record<string, unknown>
  return record.generator === API_DOCS_GENERATOR_NAME
    && record.version === API_DOCS_MANIFEST_VERSION
    && typeof record.packageName === 'string'
    && Array.isArray(record.files)
    && record.files.every(file => typeof file === 'string' && isSafeRelativePath(file))
}

async function assertOutputOwnership(outputDirectory: string, manifest: ApiDocsManifest | undefined): Promise<void> {
  let entries
  try {
    entries = await readdir(outputDirectory)
  } catch (error) {
    if (isMissingFile(error)) return
    throw error
  }
  if (entries.length > 0 && manifest === undefined) {
    throw new ApiDocsGenerationError(
      `Refusing to write into non-empty documentation directory without a valid manifest: ${outputDirectory}`,
    )
  }
}

async function writeGeneratedFile(
  outputDirectory: string,
  relativePath: string,
  content: string,
  options: NormalizedOptions,
): Promise<ApiDocsChange> {
  const path = outputPath(outputDirectory, relativePath)
  let existing: string | undefined
  try {
    existing = await readUtf8(path)
  } catch (error) {
    if (!isMissingFile(error)) throw error
  }
  const action: ApiDocsChangeKind = existing === undefined ? 'created' : existing === content ? 'unchanged' : 'updated'
  if (action === 'unchanged' || options.dryRun) return { action, path }

  await mkdir(dirname(path), { recursive: true })
  if (!options.atomic) {
    await writeFile(path, content, 'utf8')
    return { action, path }
  }

  const existingStat = await safeStat(path)
  const mode = existingStat === undefined ? 0o644 : existingStat.mode & 0o777
  const temporary = join(dirname(path), `.${basename(path)}.${process.pid}.${randomUUID()}.tmp`)
  try {
    await writeFile(temporary, content, { encoding: 'utf8', mode })
    await chmod(temporary, mode)
    await rename(temporary, path)
  } catch (error) {
    await rm(temporary, { force: true })
    throw error
  }
  return { action, path }
}

async function deleteGeneratedFile(outputDirectory: string, relativePath: string, options: NormalizedOptions): Promise<boolean> {
  const path = outputPath(outputDirectory, relativePath)
  const targetStat = await safeStat(path)
  if (targetStat === undefined) return false
  if (!targetStat.isFile()) {
    throw new ApiDocsGenerationError(`Generated documentation path is no longer a file: ${path}`)
  }
  if (options.dryRun) return true
  await rm(path)
  await removeEmptyParents(dirname(path), outputDirectory)
  return true
}

async function removeEmptyParents(directory: string, outputDirectory: string): Promise<void> {
  let current = directory
  while (current !== outputDirectory) {
    const entries = await readdir(current)
    if (entries.length > 0) return
    await rmdir(current)
    const parent = dirname(current)
    if (parent === current) return
    current = parent
  }
}

function addPage(pages: Map<string, string>, path: string, content: string): void {
  if (!isSafeRelativePath(path)) throw new ApiDocsGenerationError(`Unsafe documentation path: ${path}`)
  if (pages.has(path)) throw new ApiDocsGenerationError(`Duplicate documentation path: ${path}`)
  pages.set(path, content)
}

function outputPath(outputDirectory: string, relativePath: string): string {
  if (!isSafeRelativePath(relativePath)) throw new ApiDocsGenerationError(`Unsafe documentation path: ${relativePath}`)
  const path = resolve(outputDirectory, relativePath)
  const pathRelativeToRoot = relative(outputDirectory, path)
  if (pathRelativeToRoot === '..' || pathRelativeToRoot.startsWith(`..${sep}`) || isAbsolute(pathRelativeToRoot)) {
    throw new ApiDocsGenerationError(`Documentation path escapes output directory: ${relativePath}`)
  }
  return path
}

function isSafeRelativePath(path: string): boolean {
  if (path === '' || path === '.' || isAbsolute(path)) return false
  const segments = path.split(/[\\/]/u)
  return segments.every(segment => segment !== '' && segment !== '.' && segment !== '..')
}

function assertSeparateDirectories(sourceDirectory: string, outputDirectory: string): void {
  if (isNestedPath(sourceDirectory, outputDirectory) || isNestedPath(outputDirectory, sourceDirectory)) {
    throw new ApiDocsGenerationError('sourceDirectory and outputDirectory must not overlap.')
  }
}

function isNestedPath(parent: string, candidate: string): boolean {
  const difference = relative(parent, candidate)
  return difference === '' || (difference !== '..' && !difference.startsWith(`..${sep}`) && !isAbsolute(difference))
}

function relativeIndexLink(currentPackage: string, childPackage: string): string {
  const target = `${childPackage}/index.md`
  return relativeOutputLink(currentPackage, target)
}

function relativeModuleLink(currentPackage: string, modulePath: string): string {
  return relativeOutputLink(currentPackage, modulePath)
}

function relativeOutputLink(currentPackage: string, targetPath: string): string {
  const from = currentPackage === '' ? '.' : currentPackage
  const link = toPosixPath(relative(from, targetPath))
  return link === '' ? './' : link
}

function finalPathComponent(path: string): string {
  const index = path.lastIndexOf('/')
  return index === -1 ? path : path.slice(index + 1)
}

function codeFenceFor(content: string): string {
  const matches = content.match(/`+/gu) ?? []
  const longest = matches.reduce((length, match) => Math.max(length, match.length), 0)
  return '`'.repeat(Math.max(3, longest + 1))
}

function compareSymbols(left: TypeScriptApiSymbol, right: TypeScriptApiSymbol): number {
  return compareText(left.name, right.name) || compareText(left.kind, right.kind) || compareText(left.declaration, right.declaration)
}

function compareText(left: string, right: string): number {
  if (left === right) return 0
  return left < right ? -1 : 1
}

function toPosixPath(path: string): string {
  return path.split(sep).join('/')
}

async function readUtf8(path: string): Promise<string> {
  return new TextDecoder('utf-8', { fatal: true, ignoreBOM: true }).decode(await readFile(path))
}

async function safeStat(path: string) {
  try {
    return await stat(path)
  } catch (error) {
    if (isMissingFile(error)) return undefined
    throw error
  }
}

function isMissingFile(error: unknown): boolean {
  return typeof error === 'object' && error !== null && 'code' in error && error.code === 'ENOENT'
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

function parseArguments(args: readonly string[]): TypeScriptApiDocsOptions | undefined {
  let atomic = true
  let clean = true
  let dryRun = false
  let outputDirectory = DEFAULT_OUTPUT_DIRECTORY
  let packageName = DEFAULT_TYPESCRIPT_PACKAGE_NAME
  let sourceDirectory = DEFAULT_SOURCE_DIRECTORY

  for (let index = 0; index < args.length; index += 1) {
    const argument = args[index]
    if (argument === undefined) continue
    if (argument === '--help' || argument === '-h') {
      console.log([
        'Usage: bun apiDocsGenerator.ts [--source <directory>] [--output <directory>] [--package <name>]',
        '       [--dry-run] [--no-clean] [--no-atomic]',
        '',
        'This command generates Markdown API docs only. Formatting is an injected application boundary:',
        'use formatTypeScriptSources(sourceDirectory, formatter) when a formatter adapter is available.',
      ].join('\n'))
      return undefined
    }
    if (argument === '--dry-run') {
      dryRun = true
      continue
    }
    if (argument === '--no-clean') {
      clean = false
      continue
    }
    if (argument === '--no-atomic') {
      atomic = false
      continue
    }
    if (argument === '--source' || argument === '--output' || argument === '--package') {
      const value = args[index + 1]
      if (value === undefined) throw new ApiDocsGenerationError(`${argument} requires a value.`)
      if (argument === '--source') sourceDirectory = value
      if (argument === '--output') outputDirectory = value
      if (argument === '--package') packageName = value
      index += 1
      continue
    }
    if (argument === '--format') {
      throw new ApiDocsGenerationError(
        '--format is not available in the standalone generator; inject a TypeScriptFormatterPort with formatTypeScriptSources().',
      )
    }
    throw new ApiDocsGenerationError(`Unknown option: ${argument}`)
  }

  return { atomic, clean, dryRun, outputDirectory, packageName, sourceDirectory }
}
