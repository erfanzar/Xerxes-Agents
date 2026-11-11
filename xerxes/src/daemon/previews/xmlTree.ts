// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * A small, non-validating XML reader for Office parts shown in previews.
 *
 * Element and attribute names keep only their local part (`w:p` is `p`), so
 * the transitional and strict OOXML namespaces read alike. Only the five
 * predefined entities and character references are decoded: a DOCTYPE (and
 * any entity it declares) is skipped, never expanded, and nothing external is
 * ever loaded. Comments and processing instructions are dropped; CDATA is
 * text. Mismatched closing tags close back to the nearest open element of
 * that name, so a damaged part still yields what it can.
 */

import { decodeXmlEntities, xmlLocalName } from '../../skills/powerpoint/xml.js'

export interface XmlNode {
  readonly name: string
  readonly attributes: Readonly<Record<string, string>>
  readonly children: readonly (XmlNode | string)[]
}

interface MutableNode {
  readonly name: string
  readonly attributes: Record<string, string>
  readonly children: (MutableNode | string)[]
}

/** Parsing stopped because the part has more elements than a preview reads. */
export class XmlTooComplexError extends Error {
  constructor(readonly limit: number) {
    super(`The document has more than ${limit} XML elements`)
    this.name = 'XmlTooComplexError'
  }
}

const ATTRIBUTE = /([^\s=/>]+)\s*=\s*(?:"([^"]*)"|'([^']*)')/g

/** Parse `source` into a tree under a synthetic root named `#document`. */
export function parseXmlTree(source: string, maxElements = 2_000_000): XmlNode {
  const root: MutableNode = { name: '#document', attributes: {}, children: [] }
  const stack: MutableNode[] = [root]
  let elements = 0
  let cursor = 0
  const text = (value: string): void => {
    if (value) stack[stack.length - 1]!.children.push(value)
  }
  while (cursor < source.length) {
    const open = source.indexOf('<', cursor)
    if (open < 0) { text(decodeXmlEntities(source.slice(cursor))); break }
    if (open > cursor) text(decodeXmlEntities(source.slice(cursor, open)))
    if (source.startsWith('<!--', open)) {
      const end = source.indexOf('-->', open + 4)
      cursor = end < 0 ? source.length : end + 3
      continue
    }
    if (source.startsWith('<![CDATA[', open)) {
      const end = source.indexOf(']]>', open + 9)
      text(source.slice(open + 9, end < 0 ? source.length : end))
      cursor = end < 0 ? source.length : end + 3
      continue
    }
    if (source.startsWith('<?', open)) {
      const end = source.indexOf('?>', open + 2)
      cursor = end < 0 ? source.length : end + 2
      continue
    }
    if (source.startsWith('<!', open)) {
      cursor = skipDeclaration(source, open)
      continue
    }
    const end = tagEnd(source, open + 1)
    if (end < 0) break
    const body = source.slice(open + 1, end)
    cursor = end + 1
    if (body.startsWith('/')) {
      const name = xmlLocalName(body.slice(1).trim())
      const at = stack.findLastIndex(node => node.name === name)
      if (at > 0) stack.length = at
      continue
    }
    const selfClosing = body.endsWith('/')
    const inner = selfClosing ? body.slice(0, -1) : body
    const nameEnd = inner.search(/[\s/]|$/)
    const name = xmlLocalName(inner.slice(0, nameEnd))
    if (!name) continue
    elements += 1
    if (elements > maxElements) throw new XmlTooComplexError(maxElements)
    const attributes: Record<string, string> = {}
    for (const match of inner.slice(nameEnd).matchAll(ATTRIBUTE)) {
      const qualified = match[1]!
      if (qualified === 'xmlns' || qualified.startsWith('xmlns:')) continue
      const local = xmlLocalName(qualified)
      if (!(local in attributes)) attributes[local] = decodeXmlEntities(match[2] ?? match[3] ?? '')
    }
    const node: MutableNode = { name, attributes, children: [] }
    stack[stack.length - 1]!.children.push(node)
    if (!selfClosing) stack.push(node)
  }
  return root
}

/** The index of the `>` that ends a tag opened before `from`, skipping quoted attribute values. */
function tagEnd(source: string, from: number): number {
  let quote = ''
  for (let index = from; index < source.length; index += 1) {
    const char = source[index]!
    if (quote) { if (char === quote) quote = '' }
    else if (char === '"' || char === "'") quote = char
    else if (char === '>') return index
  }
  return -1
}

/** Skip `<!DOCTYPE …>` including an internal subset in brackets; its declarations are never applied. */
function skipDeclaration(source: string, open: number): number {
  let depth = 0
  for (let index = open + 2; index < source.length; index += 1) {
    const char = source[index]
    if (char === '[') depth += 1
    else if (char === ']') depth = Math.max(0, depth - 1)
    else if (char === '>' && depth === 0) return index + 1
  }
  return source.length
}

/** Child elements of `node` with the given local name. */
export function childElements(node: XmlNode | undefined, name?: string): XmlNode[] {
  if (!node) return []
  const out: XmlNode[] = []
  for (const child of node.children) if (typeof child !== 'string' && (name === undefined || child.name === name)) out.push(child)
  return out
}

/** The first child element with the given local name. */
export function childElement(node: XmlNode | undefined, name: string): XmlNode | undefined {
  if (!node) return undefined
  for (const child of node.children) if (typeof child !== 'string' && child.name === name) return child
  return undefined
}

/** Follow a path of child names: `descend(doc, 'document', 'body')`. */
export function descend(node: XmlNode | undefined, ...names: string[]): XmlNode | undefined {
  let current = node
  for (const name of names) current = childElement(current, name)
  return current
}

/** The text directly inside `node` (not its descendants). */
export function ownText(node: XmlNode | undefined): string {
  if (!node) return ''
  let text = ''
  for (const child of node.children) if (typeof child === 'string') text += child
  return text
}
