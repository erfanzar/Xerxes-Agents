// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { AgentSpecError } from '../core/errors.js'

export type YamlScalar = boolean | number | null | string
export type YamlMap = { readonly [key: string]: YamlValue }
export type YamlSequence = readonly YamlValue[]
export type YamlValue = YamlMap | YamlScalar | YamlSequence

interface Line {
  readonly content: string
  readonly indent: number
  readonly line: number
  readonly raw: string
}

/**
 * Parse the deliberately small YAML subset used by Xerxes configuration files.
 *
 * The agent-spec format only needs mappings, sequences, scalars, inline lists,
 * comments, and literal/folded block strings. Keeping this parser local avoids
 * making an agent definition depend on a third-party YAML runtime.
 */
export function parseYaml(content: string, source = '<yaml>'): YamlValue {
  const parser = new SimpleYamlParser(content, source)
  return parser.parse()
}

export function yamlMap(value: YamlValue, source: string): YamlMap {
  if (Array.isArray(value) || value === null || typeof value !== 'object') {
    throw new AgentSpecError(`Expected a mapping in ${source}`)
  }
  return value as YamlMap
}

class SimpleYamlParser {
  private readonly lines: readonly string[]
  private index = 0

  constructor(
    content: string,
    private readonly source: string,
  ) {
    this.lines = content.replace(/^\uFEFF/, '').replace(/\r\n/g, '\n').split('\n')
  }

  parse(): YamlValue {
    this.skipIgnorable()
    if (this.index >= this.lines.length) {
      return {}
    }
    if (this.trimmed(this.index) === '---') {
      this.index += 1
      this.skipIgnorable()
    }
    if (this.index >= this.lines.length) {
      return {}
    }
    const first = this.lineAt(this.index)
    if (first.indent !== 0) {
      this.fail(first.line, 'Top-level content must not be indented')
    }
    const value = this.parseBlock(0)
    this.skipIgnorable()
    if (this.index < this.lines.length && this.trimmed(this.index) !== '...') {
      const line = this.lineAt(this.index)
      this.fail(line.line, 'Unexpected trailing YAML content')
    }
    return value
  }

  private parseBlock(indent: number): YamlValue {
    this.skipIgnorable()
    const next = this.nextContentLine()
    if (!next || next.indent < indent) {
      return {}
    }
    if (next.indent > indent) {
      this.fail(next.line, `Unexpected indentation (expected ${indent} spaces)`)
    }
    return isSequenceItem(next.content) ? this.parseSequence(indent) : this.parseMapping(indent)
  }

  private parseMapping(indent: number): YamlMap {
    const output: Record<string, YamlValue> = {}
    while (true) {
      this.skipIgnorable()
      const line = this.nextContentLine()
      if (!line || line.indent < indent) {
        break
      }
      if (line.indent > indent) {
        this.fail(line.line, `Unexpected indentation (expected ${indent} spaces)`)
      }
      if (isSequenceItem(line.content)) {
        this.fail(line.line, 'Cannot mix a sequence with mapping entries at the same indentation')
      }
      const separator = mappingSeparator(line.content)
      if (separator < 0) {
        this.fail(line.line, 'Expected a mapping key followed by a colon')
      }
      const rawKey = line.content.slice(0, separator).trim()
      if (!rawKey) {
        this.fail(line.line, 'Mapping key must not be empty')
      }
      const key = String(parseScalar(rawKey))
      if (Object.hasOwn(output, key)) {
        this.fail(line.line, `Duplicate mapping key '${key}'`)
      }
      const rawValue = stripComment(line.content.slice(separator + 1)).trim()
      this.index += 1
      output[key] = this.valueAfterKey(rawValue, indent)
    }
    return output
  }

  private parseSequence(indent: number): YamlSequence {
    const output: YamlValue[] = []
    while (true) {
      this.skipIgnorable()
      const line = this.nextContentLine()
      if (!line || line.indent < indent) {
        break
      }
      if (line.indent > indent) {
        this.fail(line.line, `Unexpected indentation (expected ${indent} spaces)`)
      }
      if (!isSequenceItem(line.content)) {
        this.fail(line.line, 'Cannot mix mapping entries with a sequence at the same indentation')
      }
      const rawValue = stripComment(line.content.slice(1)).trim()
      this.index += 1
      if (!rawValue) {
        output.push(this.nestedValue(indent))
        continue
      }
      const separator = mappingSeparator(rawValue)
      if (separator < 0) {
        output.push(this.scalarOrBlock(rawValue, indent))
        continue
      }
      output.push(this.sequenceMapping(rawValue, indent, line.line))
    }
    return output
  }

  private sequenceMapping(rawValue: string, indent: number, lineNumber: number): YamlMap {
    const separator = mappingSeparator(rawValue)
    if (separator < 0) {
      this.fail(lineNumber, 'Expected a mapping key followed by a colon')
    }
    const key = String(parseScalar(rawValue.slice(0, separator).trim()))
    if (!key) {
      this.fail(lineNumber, 'Mapping key must not be empty')
    }
    const value = stripComment(rawValue.slice(separator + 1)).trim()
    const output: Record<string, YamlValue> = { [key]: this.valueAfterKey(value, indent) }
    this.skipIgnorable()
    const continuation = this.nextContentLine()
    if (!continuation || continuation.indent <= indent) {
      return output
    }
    const nested = this.parseBlock(continuation.indent)
    if (Array.isArray(nested) || nested === null || typeof nested !== 'object') {
      this.fail(continuation.line, 'A sequence mapping continuation must be a mapping')
    }
    for (const [nestedKey, nestedValue] of Object.entries(nested)) {
      if (Object.hasOwn(output, nestedKey)) {
        this.fail(continuation.line, `Duplicate mapping key '${nestedKey}'`)
      }
      output[nestedKey] = nestedValue
    }
    return output
  }

  private valueAfterKey(rawValue: string, indent: number): YamlValue {
    if (!rawValue) {
      return this.nestedValue(indent)
    }
    return this.scalarOrBlock(rawValue, indent)
  }

  private scalarOrBlock(rawValue: string, indent: number): YamlValue {
    if (/^[>|][+-]?$/.test(rawValue)) {
      return this.blockScalar(rawValue, indent)
    }
    return parseScalar(rawValue)
  }

  private nestedValue(parentIndent: number): YamlValue {
    this.skipIgnorable()
    const nested = this.nextContentLine()
    if (!nested || nested.indent <= parentIndent) {
      return null
    }
    return this.parseBlock(nested.indent)
  }

  private blockScalar(indicator: string, parentIndent: number): string {
    const blockLines: string[] = []
    let minimumIndent: number | undefined
    while (this.index < this.lines.length) {
      const raw = this.lines[this.index] ?? ''
      const indent = leadingWhitespace(raw)
      if (raw.trim() && indent <= parentIndent) {
        break
      }
      if (!raw.trim() && this.index + 1 < this.lines.length) {
        blockLines.push('')
        this.index += 1
        continue
      }
      if (!raw.trim() && this.index + 1 >= this.lines.length) {
        this.index += 1
        break
      }
      minimumIndent = Math.min(minimumIndent ?? indent, indent)
      blockLines.push(raw)
      this.index += 1
    }
    const contentIndent = minimumIndent ?? parentIndent + 1
    const lines = blockLines.map(line => line ? line.slice(contentIndent) : '')
    const separator = indicator.startsWith('>') ? ' ' : '\n'
    let value = lines.join(separator)
    if (indicator.endsWith('-')) {
      return value.replace(/\n+$/, '')
    }
    if (lines.length) {
      value += '\n'
    }
    return value
  }

  private nextContentLine(): Line | undefined {
    this.skipIgnorable()
    if (this.index >= this.lines.length) {
      return undefined
    }
    return this.lineAt(this.index)
  }

  private skipIgnorable(): void {
    while (this.index < this.lines.length) {
      const trimmed = this.trimmed(this.index)
      if (trimmed && !trimmed.startsWith('#')) {
        return
      }
      this.index += 1
    }
  }

  private lineAt(index: number): Line {
    const raw = this.lines[index] ?? ''
    const indent = leadingWhitespace(raw)
    if (raw.slice(0, indent).includes('\t')) {
      this.fail(index + 1, 'Tabs are not supported for indentation')
    }
    return { content: raw.slice(indent), indent, line: index + 1, raw }
  }

  private trimmed(index: number): string {
    return (this.lines[index] ?? '').trim()
  }

  private fail(line: number, message: string): never {
    throw new AgentSpecError(`Invalid YAML in ${this.source} at line ${line}: ${message}`)
  }
}

function leadingWhitespace(line: string): number {
  let index = 0
  while (index < line.length && (line[index] === ' ' || line[index] === '\t')) {
    index += 1
  }
  return index
}

function isSequenceItem(content: string): boolean {
  return content === '-' || content.startsWith('- ')
}

function mappingSeparator(value: string): number {
  let quote: '"' | "'" | undefined
  let bracketDepth = 0
  for (let index = 0; index < value.length; index += 1) {
    const character = value[index]
    if (quote) {
      if (character === quote && value[index - 1] !== '\\') {
        quote = undefined
      }
      continue
    }
    if (character === '"' || character === "'") {
      quote = character
      continue
    }
    if (character === '[' || character === '{') {
      bracketDepth += 1
      continue
    }
    if (character === ']' || character === '}') {
      bracketDepth = Math.max(0, bracketDepth - 1)
      continue
    }
    if (character === ':' && bracketDepth === 0 && (index + 1 === value.length || /\s/.test(value[index + 1] ?? ''))) {
      return index
    }
  }
  return -1
}

function stripComment(value: string): string {
  let quote: '"' | "'" | undefined
  for (let index = 0; index < value.length; index += 1) {
    const character = value[index]
    if (quote) {
      if (character === quote && value[index - 1] !== '\\') {
        quote = undefined
      }
      continue
    }
    if (character === '"' || character === "'") {
      quote = character
      continue
    }
    if (character === '#' && (index === 0 || /\s/.test(value[index - 1] ?? ''))) {
      return value.slice(0, index)
    }
  }
  return value
}

function parseScalar(rawValue: string): YamlValue {
  const value = stripComment(rawValue).trim()
  if (!value || value === '~' || /^null$/i.test(value)) {
    return null
  }
  if (/^(?:true|false)$/i.test(value)) {
    return /^true$/i.test(value)
  }
  if (/^-?(?:0|[1-9]\d*)(?:\.\d+)?$/.test(value)) {
    const numeric = Number(value)
    if (Number.isFinite(numeric)) {
      return numeric
    }
  }
  if (value.startsWith('[') && value.endsWith(']')) {
    const inner = value.slice(1, -1).trim()
    return inner ? splitFlowValues(inner).map(parseScalar) : []
  }
  if (value.startsWith('{') && value.endsWith('}')) {
    const map: Record<string, YamlValue> = {}
    const inner = value.slice(1, -1).trim()
    if (!inner) {
      return map
    }
    for (const entry of splitFlowValues(inner)) {
      const separator = mappingSeparator(entry)
      if (separator < 0) {
        throw new AgentSpecError(`Invalid inline YAML mapping entry '${entry}'`)
      }
      const key = String(parseScalar(entry.slice(0, separator).trim()))
      map[key] = parseScalar(entry.slice(separator + 1))
    }
    return map
  }
  if (value.startsWith('"') && value.endsWith('"')) {
    try {
      return JSON.parse(value) as string
    } catch {
      throw new AgentSpecError(`Invalid double-quoted YAML string '${value}'`)
    }
  }
  if (value.startsWith("'") && value.endsWith("'")) {
    return value.slice(1, -1).replace(/''/g, "'")
  }
  return value
}

function splitFlowValues(value: string): string[] {
  const values: string[] = []
  let start = 0
  let quote: '"' | "'" | undefined
  let depth = 0
  for (let index = 0; index < value.length; index += 1) {
    const character = value[index]
    if (quote) {
      if (character === quote && value[index - 1] !== '\\') {
        quote = undefined
      }
      continue
    }
    if (character === '"' || character === "'") {
      quote = character
      continue
    }
    if (character === '[' || character === '{') {
      depth += 1
      continue
    }
    if (character === ']' || character === '}') {
      depth = Math.max(0, depth - 1)
      continue
    }
    if (character === ',' && depth === 0) {
      values.push(value.slice(start, index).trim())
      start = index + 1
    }
  }
  values.push(value.slice(start).trim())
  return values.filter(Boolean)
}
