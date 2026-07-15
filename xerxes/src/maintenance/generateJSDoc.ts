// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { basename, extname } from 'node:path'
import * as ts from 'typescript'

import { errorMessage, parseFileArguments, readUtf8Text, writeTextFile, type WriteOptions } from './io.js'
import { applySourceEdits, type SourceEdit } from './pythonSource.js'

const SOURCE_EXTENSIONS = new Set(['.cts', '.mts', '.ts', '.tsx'])
const VOID_RETURN_TYPES = new Set(['never', 'Promise<void>', 'PromiseLike<void>', 'void'])

const NAME_PATTERNS: readonly { prefix: string; verb: string }[] = [
  { prefix: 'get', verb: 'Retrieve the ' },
  { prefix: 'set', verb: 'Set the ' },
  { prefix: 'is', verb: 'Check whether ' },
  { prefix: 'has', verb: 'Check whether ' },
  { prefix: 'can', verb: 'Determine whether ' },
  { prefix: 'should', verb: 'Determine whether ' },
  { prefix: 'find', verb: 'Find ' },
  { prefix: 'search', verb: 'Search for ' },
  { prefix: 'create', verb: 'Create ' },
  { prefix: 'build', verb: 'Build ' },
  { prefix: 'make', verb: 'Make ' },
  { prefix: 'generate', verb: 'Generate ' },
  { prefix: 'compute', verb: 'Compute ' },
  { prefix: 'calculate', verb: 'Calculate ' },
  { prefix: 'process', verb: 'Process ' },
  { prefix: 'handle', verb: 'Handle ' },
  { prefix: 'parse', verb: 'Parse ' },
  { prefix: 'format', verb: 'Format ' },
  { prefix: 'convert', verb: 'Convert ' },
  { prefix: 'validate', verb: 'Validate ' },
  { prefix: 'check', verb: 'Check ' },
  { prefix: 'load', verb: 'Load ' },
  { prefix: 'save', verb: 'Save ' },
  { prefix: 'delete', verb: 'Delete ' },
  { prefix: 'remove', verb: 'Remove ' },
  { prefix: 'add', verb: 'Add ' },
  { prefix: 'update', verb: 'Update ' },
  { prefix: 'clear', verb: 'Clear ' },
  { prefix: 'reset', verb: 'Reset ' },
]

export interface JSDocGenerationOptions {
  includeModuleDocumentation?: boolean
}

export interface JSDocGenerationResult {
  changed: boolean
  declarationsDocumented: number
  moduleDocumented: boolean
  text: string
  valid: boolean
}

export interface JSDocGenerationFileResult extends JSDocGenerationResult {
  path: string
}

export interface JSDocFileGenerationOptions extends JSDocGenerationOptions, WriteOptions {}

/** Explicit filesystem boundary for source-documentation generation. */
export interface SourceDocumentationFilePort {
  readText(path: string): Promise<string>
  writeText(path: string, text: string, options: WriteOptions): Promise<void>
}

const nativeFilePort: SourceDocumentationFilePort = {
  readText: readUtf8Text,
  writeText: writeTextFile,
}

/** Convert a TypeScript identifier to a lower-case human-readable phrase. */
export function humanizeName(name: string): string {
  return name
    .replace(/([a-z\d])([A-Z])/gu, '$1 $2')
    .replace(/([A-Z]+)([A-Z][a-z])/gu, '$1 $2')
    .replace(/[-_$]+/gu, ' ')
    .trim()
    .replace(/\s+/gu, ' ')
    .toLowerCase()
}

/** Derive a conservative sentence from a declaration name. */
export function describeFromName(
  name: string,
  options: { asynchronous?: boolean; property?: 'get' | 'set' } = {},
): string {
  const humanized = humanizeName(name)
  if (options.property === 'get') return `Return the ${humanized}.`
  if (options.property === 'set') return `Set the ${humanized}.`

  for (const pattern of NAME_PATTERNS) {
    const remainder = nameRemainder(name, pattern.prefix)
    if (remainder !== undefined) return asynchronousDescription(`${pattern.verb}${humanizeName(remainder)}.`, options)
  }

  if (name === 'constructor' || name === 'init' || name === '__init__') {
    return asynchronousDescription('Initialize the instance.', options)
  }
  if (name.startsWith('__') && name.endsWith('__')) {
    return asynchronousDescription(`Special method for ${humanizeName(name.slice(2, -2))}.`, options)
  }
  if (name.startsWith('_')) return asynchronousDescription(`Internal helper to ${humanized}.`, options)
  return asynchronousDescription(`${capitalize(humanized)}.`, options)
}

/** Add missing JSDoc comments to a syntactically valid TypeScript module. */
export function generateJSDoc(
  source: string,
  fileName = 'module.ts',
  options: JSDocGenerationOptions = {},
): JSDocGenerationResult {
  if (!isSyntacticallyValid(source, fileName)) return invalidResult(source)

  const sourceFile = ts.createSourceFile(fileName, source, ts.ScriptTarget.Latest, true, scriptKindFor(fileName))
  const lineEnding = source.includes('\r\n') ? '\r\n' : '\n'
  const insertions = new Map<number, string[]>()
  let declarationsDocumented = 0
  let moduleDocumented = false

  if (
    options.includeModuleDocumentation !== false
    && sourceFile.statements.length > 0
    && !hasModuleDocumentation(source, sourceFile)
  ) {
    const position = moduleDocumentationPosition(source, sourceFile)
    addInsertion(
      insertions,
      position,
      formatJSDoc(moduleDocumentationLines(sourceFile, fileName), indentationAt(source, position), lineEnding),
    )
    moduleDocumented = true
  }

  const visit = (node: ts.Node): void => {
    const lines = documentationLinesForNode(node, sourceFile)
    if (lines !== undefined && !hasExistingJSDoc(source, node)) {
      const declarationStart = node.getStart(sourceFile)
      const position = lineStartAt(source, declarationStart)
      addInsertion(insertions, position, formatJSDoc(lines, indentationAt(source, declarationStart), lineEnding))
      declarationsDocumented += 1
    }
    ts.forEachChild(node, visit)
  }
  sourceFile.forEachChild(visit)

  if (insertions.size === 0) {
    return { changed: false, declarationsDocumented: 0, moduleDocumented: false, text: source, valid: true }
  }

  const edits: SourceEdit[] = [...insertions.entries()].map(([start, texts]) => ({
    end: start,
    replacement: texts.join(''),
    start,
  }))
  const text = applySourceEdits(source, edits)
  if (!isSyntacticallyValid(text, fileName)) return invalidResult(source)
  return { changed: text !== source, declarationsDocumented, moduleDocumented, text, valid: true }
}

/** Transform explicitly named TypeScript files using injectable I/O and dry-run controls. */
export async function generateJSDocForFiles(
  paths: readonly string[],
  options: JSDocFileGenerationOptions = {},
  filePort: SourceDocumentationFilePort = nativeFilePort,
): Promise<JSDocGenerationFileResult[]> {
  const results: JSDocGenerationFileResult[] = []
  for (const path of paths) {
    assertTypeScriptPath(path)
    const source = await filePort.readText(path)
    const result = generateJSDoc(source, path, options)
    if (result.changed && options.dryRun !== true) await filePort.writeText(path, result.text, options)
    results.push({ ...result, path })
  }
  return results
}

export async function main(args: readonly string[] = process.argv.slice(2)): Promise<number> {
  try {
    const parsed = parseFileArguments(
      args,
      'Usage: bun generateJSDoc.ts [--dry-run] [--no-atomic] <file1.ts> [file2.ts ...]',
    )
    if (parsed === undefined) return 0
    if (parsed.paths.length === 0) throw new Error('At least one TypeScript source path is required.')

    const results = await generateJSDocForFiles(parsed.paths, parsed)
    let total = 0
    for (const result of results) {
      if (!result.valid) {
        console.log(`  SKIP (syntax error): ${result.path}`)
        continue
      }
      if (!result.changed) continue
      total += 1
      console.log(`  ${parsed.dryRun ? 'WOULD PROCESS' : 'PROCESSED'}: ${result.path}`)
    }
    console.log(`\nTotal files ${parsed.dryRun ? 'that would be modified' : 'modified'}: ${total}`)
    return 0
  } catch (error) {
    console.error(`generate-jsdoc: ${errorMessage(error)}`)
    return 1
  }
}

if (import.meta.main) {
  process.exitCode = await main()
}

function addInsertion(insertions: Map<number, string[]>, position: number, text: string): void {
  const existing = insertions.get(position)
  if (existing === undefined) {
    insertions.set(position, [text])
    return
  }
  existing.push(text)
}

function assertTypeScriptPath(path: string): void {
  if (!SOURCE_EXTENSIONS.has(extname(path).toLowerCase())) {
    throw new Error(`Expected a TypeScript source file (.ts, .tsx, .mts, or .cts): ${path}`)
  }
}

function asynchronousDescription(
  description: string,
  options: { asynchronous?: boolean; property?: 'get' | 'set' },
): string {
  if (options.asynchronous !== true || description.length === 0) return description
  return `Asynchronously ${description[0]!.toLowerCase()}${description.slice(1)}`
}

function capitalize(value: string): string {
  return value.length === 0 ? value : `${value[0]!.toUpperCase()}${value.slice(1)}`
}

function documentationLinesForNode(node: ts.Node, sourceFile: ts.SourceFile): readonly string[] | undefined {
  if (ts.isFunctionDeclaration(node)) {
    const name = node.name?.text
    return name === undefined ? undefined : functionDocumentationLines(node, name, sourceFile)
  }
  if (ts.isMethodDeclaration(node) || ts.isMethodSignature(node)) {
    const name = propertyName(node.name)
    return name === undefined ? undefined : functionDocumentationLines(node, name, sourceFile)
  }
  if (ts.isGetAccessorDeclaration(node)) {
    const name = propertyName(node.name)
    return name === undefined ? undefined : functionDocumentationLines(node, name, sourceFile, { property: 'get' })
  }
  if (ts.isSetAccessorDeclaration(node)) {
    const name = propertyName(node.name)
    return name === undefined ? undefined : functionDocumentationLines(node, name, sourceFile, { property: 'set' })
  }
  if (ts.isConstructorDeclaration(node)) {
    return functionDocumentationLines(node, 'constructor', sourceFile, { includeReturns: false })
  }
  if (ts.isClassDeclaration(node) || ts.isInterfaceDeclaration(node)) {
    return node.name === undefined ? undefined : classDocumentationLines(node, sourceFile)
  }
  if (ts.isTypeAliasDeclaration(node) || ts.isEnumDeclaration(node)) {
    return [`${capitalize(humanizeName(node.name.text))}.`]
  }
  if (ts.isPropertyDeclaration(node) || ts.isPropertySignature(node)) {
    const name = propertyName(node.name)
    return name === undefined ? undefined : [`The ${humanizeName(name)}.`]
  }
  if (ts.isVariableStatement(node)) return variableDocumentationLines(node, sourceFile)
  return undefined
}

function functionDocumentationLines(
  declaration: ts.SignatureDeclaration,
  name: string,
  sourceFile: ts.SourceFile,
  options: { includeReturns?: boolean; property?: 'get' | 'set' } = {},
): string[] {
  const asynchronous = ts.canHaveModifiers(declaration)
    && (ts.getModifiers(declaration)?.some(modifier => modifier.kind === ts.SyntaxKind.AsyncKeyword) ?? false)
  const descriptionOptions = options.property === undefined
    ? { asynchronous }
    : { asynchronous, property: options.property }
  const lines = [describeFromName(name, descriptionOptions)]
  const parameters = declaration.parameters.flatMap(parameterDocumentationLines)
  const includeReturns = options.includeReturns !== false
    && options.property !== 'set'
    && returnsValue(declaration, sourceFile)

  if (parameters.length > 0 || includeReturns) lines.push('')
  lines.push(...parameters)
  if (includeReturns) {
    lines.push(
      asynchronous ? '@returns Resolves with the result of the operation.' : '@returns Result of the operation.',
    )
  }
  return lines
}

function parameterDocumentationLines(parameter: ts.ParameterDeclaration): string[] {
  if (!ts.isIdentifier(parameter.name) || parameter.name.text === 'this') return []
  const name = parameter.name.text
  if (parameter.dotDotDotToken !== undefined) return [`@param ${name} - Additional positional arguments.`]

  const inputKind = parameter.questionToken !== undefined || parameter.initializer !== undefined
    ? 'Optional input'
    : 'Input'
  let description = `${inputKind}: ${humanizeName(name)}.`
  if (parameter.initializer !== undefined) description += ` Defaults to ${parameter.initializer.getText()}.`
  return [`@param ${name} - ${description}`]
}

function classDocumentationLines(
  declaration: ts.ClassDeclaration | ts.InterfaceDeclaration,
  sourceFile: ts.SourceFile,
): string[] {
  const name = declaration.name
  if (name === undefined) return []
  const lines = [`${capitalize(humanizeName(name.text))}.`]
  const heritage = heritageDocumentationLines(declaration, sourceFile)
  if (heritage.length > 0) lines.push('', ...heritage)
  return lines
}

function heritageDocumentationLines(
  declaration: ts.ClassDeclaration | ts.InterfaceDeclaration,
  sourceFile: ts.SourceFile,
): string[] {
  const lines: string[] = []
  for (const clause of declaration.heritageClauses ?? []) {
    const tag = clause.token === ts.SyntaxKind.ExtendsKeyword ? '@extends' : '@implements'
    for (const type of clause.types) lines.push(`${tag} ${type.getText(sourceFile)}`)
  }
  return lines
}

function hasExistingJSDoc(source: string, node: ts.Node): boolean {
  return (ts.getLeadingCommentRanges(source, node.getFullStart()) ?? []).some(range => isJSDocComment(source, range))
}

function hasModuleDocumentation(source: string, sourceFile: ts.SourceFile): boolean {
  const firstStatement = sourceFile.statements[0]
  if (firstStatement === undefined) return false
  return (ts.getLeadingCommentRanges(source, firstStatement.getFullStart()) ?? []).some(range => {
    return isJSDocComment(source, range) && /@module\b/u.test(source.slice(range.pos, range.end))
  })
}

function indentationAt(source: string, position: number): string {
  const lineStart = lineStartAt(source, position)
  const indentation = source.slice(lineStart, position).match(/^[\t ]*/u)
  return indentation?.[0] ?? ''
}

function lineStartAt(source: string, position: number): number {
  const lineStart = source.lastIndexOf('\n', Math.max(0, position - 1)) + 1
  return lineStart === 0 && source.startsWith('\uFEFF') ? 1 : lineStart
}

function isJSDocComment(source: string, range: ts.CommentRange): boolean {
  return source.slice(range.pos, range.end).startsWith('/**')
}

function isSyntacticallyValid(source: string, fileName: string): boolean {
  const output = ts.transpileModule(source, {
    compilerOptions: { jsx: ts.JsxEmit.Preserve, module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ESNext },
    fileName,
    reportDiagnostics: true,
  })
  return !(output.diagnostics ?? []).some(diagnostic => diagnostic.category === ts.DiagnosticCategory.Error)
}

function invalidResult(source: string): JSDocGenerationResult {
  return { changed: false, declarationsDocumented: 0, moduleDocumented: false, text: source, valid: false }
}

function moduleDocumentationLines(sourceFile: ts.SourceFile, fileName: string): string[] {
  const moduleName = basename(fileName).replace(/\.(?:cts|mts|ts|tsx)$/iu, '')
  const lines = [`${capitalize(humanizeName(moduleName))} module for Xerxes.`, '', `@module ${moduleName}`]
  const exports = publicExports(sourceFile)
  if (exports.length > 0) lines.push(`@remarks Exports: ${exports.join(', ')}.`)
  return lines
}

function moduleDocumentationPosition(source: string, sourceFile: ts.SourceFile): number {
  const firstStatement = sourceFile.statements[0]
  if (firstStatement === undefined) return source.length
  const firstJSDoc = (ts.getLeadingCommentRanges(source, firstStatement.getFullStart()) ?? []).find(range => {
    return isJSDocComment(source, range)
  })
  return firstJSDoc?.pos ?? firstStatement.getStart(sourceFile)
}

function nameRemainder(name: string, prefix: string): string | undefined {
  if (name.startsWith(`${prefix}_`)) return name.slice(prefix.length + 1)
  if (name.startsWith(prefix) && name.length > prefix.length && /[A-Z]/u.test(name[prefix.length]!)) {
    return name.slice(prefix.length)
  }
  return undefined
}

function propertyName(name: ts.PropertyName | undefined): string | undefined {
  if (name === undefined) return undefined
  if (ts.isIdentifier(name) || ts.isStringLiteral(name) || ts.isNumericLiteral(name)) return name.text
  return undefined
}

function publicExports(sourceFile: ts.SourceFile): string[] {
  const names: string[] = []
  for (const statement of sourceFile.statements) {
    if (!hasExportModifier(statement)) continue
    if (ts.isVariableStatement(statement)) {
      for (const declaration of statement.declarationList.declarations) {
        if (ts.isIdentifier(declaration.name)) names.push(declaration.name.text)
      }
      continue
    }
    if (
      (ts.isClassDeclaration(statement)
        || ts.isEnumDeclaration(statement)
        || ts.isFunctionDeclaration(statement)
        || ts.isInterfaceDeclaration(statement)
        || ts.isTypeAliasDeclaration(statement))
      && statement.name !== undefined
    ) {
      names.push(statement.name.text)
    }
  }
  if (names.length <= 10) return names
  return [...names.slice(0, 10), `... and ${names.length - 10} more`]
}

function hasExportModifier(node: ts.Node): boolean {
  return ts.canHaveModifiers(node)
    && (ts.getModifiers(node)?.some(modifier => modifier.kind === ts.SyntaxKind.ExportKeyword) ?? false)
}

function returnsValue(declaration: ts.SignatureDeclaration, sourceFile: ts.SourceFile): boolean {
  const annotation = declaration.type?.getText(sourceFile).replace(/\s+/gu, '')
  if (annotation !== undefined) return !VOID_RETURN_TYPES.has(annotation)
  const body = (declaration as ts.FunctionLikeDeclarationBase).body
  if (body === undefined) return false
  return ts.isBlock(body) ? containsValueReturn(body) : true
}

function containsValueReturn(node: ts.Node): boolean {
  let found = false
  const visit = (child: ts.Node): void => {
    if (found || (child !== node && ts.isFunctionLike(child))) return
    if (ts.isReturnStatement(child) && child.expression !== undefined) {
      found = true
      return
    }
    ts.forEachChild(child, visit)
  }
  ts.forEachChild(node, visit)
  return found
}

function scriptKindFor(fileName: string): ts.ScriptKind {
  return extname(fileName).toLowerCase() === '.tsx' ? ts.ScriptKind.TSX : ts.ScriptKind.TS
}

function variableDocumentationLines(
  node: ts.VariableStatement,
  sourceFile: ts.SourceFile,
): readonly string[] | undefined {
  const [declaration] = node.declarationList.declarations
  if (
    declaration === undefined
    || !ts.isIdentifier(declaration.name)
    || declaration.initializer === undefined
    || (!ts.isArrowFunction(declaration.initializer) && !ts.isFunctionExpression(declaration.initializer))
    || node.declarationList.declarations.length !== 1
  ) {
    return undefined
  }
  return functionDocumentationLines(declaration.initializer, declaration.name.text, sourceFile)
}

function formatJSDoc(lines: readonly string[], indentation: string, lineEnding: string): string {
  return [
    `${indentation}/**`,
    ...lines.map(line => (line === '' ? `${indentation} *` : `${indentation} * ${line}`)),
    `${indentation} */`,
    '',
  ].join(lineEnding)
}
