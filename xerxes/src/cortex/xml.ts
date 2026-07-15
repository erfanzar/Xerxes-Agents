// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Narrow XML reader for LLM-produced Cortex plans.
 *
 * It intentionally accepts only element tags, quoted attributes, and the five
 * predefined XML entities. DTDs, entities, processing instructions, comments,
 * and CDATA are rejected rather than delegated to a general XML parser. Cortex
 * plans are a small, known document format, so this avoids XML expansion and
 * parser-configuration hazards without adding a runtime dependency.
 */

export interface XmlElement {
  readonly attributes: Readonly<Record<string, string>>
  readonly inner: string
  readonly name: string
}

interface XmlTag {
  readonly attributes: Readonly<Record<string, string>>
  readonly closing: boolean
  readonly end: number
  readonly name: string
  readonly selfClosing: boolean
  readonly start: number
}

const NAME_PATTERN = /^[A-Za-z_][A-Za-z0-9_.:-]*$/
const FORBIDDEN_XML = /<\s*[!?]/

/** Extract the first complete named root element from a model response. */
export function parseXmlRoot(source: string, expectedName: string): XmlElement {
  if (!source.trim()) throw new Error('XML response is empty')
  if (FORBIDDEN_XML.test(source)) throw new Error('XML declarations, DTDs, entities, and processing instructions are not allowed')

  const rootName = expectedName.toLowerCase()
  const start = findRootStart(source, rootName)
  if (start < 0) throw new Error(`XML response does not contain a <${expectedName}> root`)
  const opening = readTag(source, start)
  if (opening.closing || opening.selfClosing || opening.name !== rootName) {
    throw new Error(`Expected a non-empty <${expectedName}> root element`)
  }

  let cursor = opening.end
  let depth = 1
  while (depth > 0) {
    const next = source.indexOf('<', cursor)
    if (next < 0) throw new Error(`Unclosed <${expectedName}> root element`)
    const tag = readTag(source, next)
    if (tag.name === rootName) {
      if (tag.closing) depth -= 1
      else if (!tag.selfClosing) depth += 1
    }
    if (depth === 0) {
      return {
        name: opening.name,
        attributes: opening.attributes,
        inner: source.slice(opening.end, tag.start),
      }
    }
    cursor = tag.end
  }
  throw new Error(`Unclosed <${expectedName}> root element`)
}

/** Return direct child elements in source order. Text between elements must be whitespace only. */
export function directXmlChildren(source: string): XmlElement[] {
  const children: XmlElement[] = []
  let cursor = 0
  while (cursor < source.length) {
    const next = source.indexOf('<', cursor)
    if (next < 0) {
      if (source.slice(cursor).trim()) throw new Error('Unexpected text between XML elements')
      break
    }
    if (source.slice(cursor, next).trim()) throw new Error('Unexpected text between XML elements')
    const opening = readTag(source, next)
    if (opening.closing) throw new Error(`Unexpected closing </${opening.name}> tag`)
    if (opening.selfClosing) {
      children.push({ name: opening.name, attributes: opening.attributes, inner: '' })
      cursor = opening.end
      continue
    }

    let depth = 1
    let scan = opening.end
    while (depth > 0) {
      const nestedStart = source.indexOf('<', scan)
      if (nestedStart < 0) throw new Error(`Unclosed <${opening.name}> element`)
      const nested = readTag(source, nestedStart)
      if (nested.name === opening.name) {
        if (nested.closing) depth -= 1
        else if (!nested.selfClosing) depth += 1
      }
      if (depth === 0) {
        children.push({
          name: opening.name,
          attributes: opening.attributes,
          inner: source.slice(opening.end, nested.start),
        })
        cursor = nested.end
        break
      }
      scan = nested.end
    }
  }
  return children
}

/** Find the first direct child with the requested name. */
export function xmlChild(children: readonly XmlElement[], name: string): XmlElement | undefined {
  return children.find(child => child.name === name.toLowerCase())
}

/** Decode plain text inside a plan field. Nested markup is intentionally unsupported. */
export function xmlText(element: XmlElement | undefined, fallback = ''): string {
  if (!element) return fallback
  if (element.inner.includes('<')) throw new Error(`<${element.name}> cannot contain nested markup`)
  return decodeEntities(element.inner).trim()
}

/** Parse comma- or whitespace-separated task/step identifiers. */
export function xmlIdentifiers(element: XmlElement | undefined): string[] {
  const text = xmlText(element)
  if (!text) return []
  return [...new Set(text.split(/[\s,]+/).map(value => value.trim()).filter(Boolean))]
}

/** Parse a conventional XML boolean field. */
export function xmlBoolean(element: XmlElement | undefined, fallback = false): boolean {
  const value = xmlText(element).toLowerCase()
  if (!value) return fallback
  if (value === 'true' || value === '1' || value === 'yes') return true
  if (value === 'false' || value === '0' || value === 'no') return false
  throw new Error(`Invalid XML boolean value: ${value}`)
}

function findRootStart(source: string, name: string): number {
  const pattern = new RegExp(`<${escapeRegExp(name)}(?=[\\s/>])`, 'i')
  const match = pattern.exec(source)
  return match?.index ?? -1
}

function readTag(source: string, start: number): XmlTag {
  if (source[start] !== '<') throw new Error('Expected an XML tag')
  const first = source[start + 1]
  if (!first || first === '!' || first === '?') throw new Error('Unsupported XML declaration or markup')

  let quote: '"' | "'" | undefined
  let end = start + 1
  for (; end < source.length; end += 1) {
    const current = source[end]
    if (!current) break
    if (quote) {
      if (current === quote) quote = undefined
      continue
    }
    if (current === '"' || current === "'") {
      quote = current
      continue
    }
    if (current === '>') break
  }
  if (end >= source.length || source[end] !== '>') throw new Error('Unterminated XML tag')
  if (quote) throw new Error('Unterminated XML attribute quote')

  let content = source.slice(start + 1, end).trim()
  const closing = content.startsWith('/')
  if (closing) content = content.slice(1).trim()
  const selfClosing = !closing && content.endsWith('/')
  if (selfClosing) content = content.slice(0, -1).trim()
  const space = content.search(/\s/)
  const rawName = space < 0 ? content : content.slice(0, space)
  const attributeSource = space < 0 ? '' : content.slice(space).trim()
  if (!NAME_PATTERN.test(rawName)) throw new Error(`Invalid XML tag name: ${rawName}`)
  if (closing && attributeSource) throw new Error(`Closing </${rawName}> tag cannot have attributes`)
  return {
    name: rawName.toLowerCase(),
    attributes: closing ? {} : parseAttributes(attributeSource),
    closing,
    selfClosing,
    start,
    end: end + 1,
  }
}

function parseAttributes(source: string): Readonly<Record<string, string>> {
  if (!source) return {}
  const values: Record<string, string> = {}
  let cursor = 0
  while (cursor < source.length) {
    while (/\s/.test(source[cursor] ?? '')) cursor += 1
    if (cursor >= source.length) break
    const nameMatch = /^[A-Za-z_][A-Za-z0-9_.:-]*/.exec(source.slice(cursor))
    if (!nameMatch) throw new Error('Invalid XML attribute name')
    const name = nameMatch[0]
    cursor += name.length
    while (/\s/.test(source[cursor] ?? '')) cursor += 1
    if (source[cursor] !== '=') throw new Error(`Attribute ${name} must use a quoted value`)
    cursor += 1
    while (/\s/.test(source[cursor] ?? '')) cursor += 1
    const quote = source[cursor]
    if (quote !== '"' && quote !== "'") throw new Error(`Attribute ${name} must use a quoted value`)
    cursor += 1
    const valueStart = cursor
    const valueEnd = source.indexOf(quote, cursor)
    if (valueEnd < 0) throw new Error(`Unterminated value for attribute ${name}`)
    if (Object.hasOwn(values, name)) throw new Error(`Duplicate XML attribute: ${name}`)
    values[name] = decodeEntities(source.slice(valueStart, valueEnd))
    cursor = valueEnd + 1
  }
  return values
}

function decodeEntities(source: string): string {
  const entityPattern = /&(amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+);/g
  const remainder = source.replace(entityPattern, '')
  if (remainder.includes('&')) throw new Error('Only predefined XML entities are allowed')
  return source.replace(entityPattern, (_whole, entity: string) => {
    if (entity === 'amp') return '&'
    if (entity === 'lt') return '<'
    if (entity === 'gt') return '>'
    if (entity === 'quot') return '"'
    if (entity === 'apos') return "'"
    const numeric = entity.startsWith('#x') ? Number.parseInt(entity.slice(2), 16) : Number.parseInt(entity.slice(1), 10)
    if (!Number.isSafeInteger(numeric) || numeric < 0 || numeric > 0x10ffff) throw new Error(`Invalid numeric XML entity: &${entity};`)
    return String.fromCodePoint(numeric)
  })
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}
