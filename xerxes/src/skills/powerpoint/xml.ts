// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { OfficePackageError } from './errors.js'

/** Parsed relationship entry from an OOXML `.rels` part. */
export interface OoxmlRelationship {
  readonly id: string
  readonly target: string
  readonly targetMode?: string
  readonly type: string
}

/** Return the local component of a qualified XML name. */
export function xmlLocalName(name: string): string {
  const separator = name.lastIndexOf(':')
  return separator < 0 ? name : name.slice(separator + 1)
}

/** Escape text placed into a freshly-created XML attribute. */
export function escapeXmlAttribute(value: string): string {
  return value
    .replaceAll('&', '&amp;')
    .replaceAll('"', '&quot;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
}

/** Parse XML attributes without evaluating entities or loading external XML resources. */
export function parseXmlAttributes(source: string): Readonly<Record<string, string>> {
  const attributes: Record<string, string> = {}
  const expression = /([A-Za-z_][\w:.-]*)\s*=\s*(["'])([\s\S]*?)\2/g
  for (const match of source.matchAll(expression)) {
    const name = match[1]
    const value = match[3]
    if (name !== undefined && value !== undefined) attributes[name] = value
  }
  return attributes
}

/** Read only valid, internally addressable relationship records from an OOXML relationship XML payload. */
export function parseOoxmlRelationships(xml: string): readonly OoxmlRelationship[] {
  const relationships: OoxmlRelationship[] = []
  const expression = /<Relationship\b([\s\S]*?)(?:\/\s*>|>\s*<\/Relationship\s*>)/gi
  for (const match of xml.matchAll(expression)) {
    const attributes = parseXmlAttributes(match[1] ?? '')
    const id = attributes.Id
    const target = attributes.Target
    const type = attributes.Type
    if (!id || !target || !type) continue
    const targetMode = attributes.TargetMode
    relationships.push({
      id,
      target,
      type,
      ...(targetMode === undefined ? {} : { targetMode }),
    })
  }
  return relationships
}

/** Remove matching `<Relationship>` tags while retaining the rest of the XML byte-for-byte. */
export function removeOoxmlRelationships(
  xml: string,
  predicate: (relationship: OoxmlRelationship) => boolean,
): string {
  const expression = /\s*<Relationship\b([\s\S]*?)(?:\/\s*>|>\s*<\/Relationship\s*>)\s*/gi
  return xml.replace(expression, (whole, rawAttributes: string) => {
    const attributes = parseXmlAttributes(rawAttributes)
    const id = attributes.Id
    const target = attributes.Target
    const type = attributes.Type
    if (!id || !target || !type) return whole
    const targetMode = attributes.TargetMode
    const relationship: OoxmlRelationship = {
      id,
      target,
      type,
      ...(targetMode === undefined ? {} : { targetMode }),
    }
    return predicate(relationship) ? '\n' : whole
  })
}

/** Insert a child fragment immediately before the final named closing tag. */
export function appendXmlChild(xml: string, qualifiedName: string, child: string): string {
  const expression = new RegExp(`</${escapeRegExp(qualifiedName)}\\s*>`, 'gi')
  let match: RegExpExecArray | undefined
  for (const candidate of xml.matchAll(expression)) {
    match = candidate
  }
  if (match === undefined || match.index === undefined) {
    throw new OfficePackageError(`Expected closing XML tag </${qualifiedName}> was not found`)
  }
  return `${xml.slice(0, match.index)}${child}${xml.slice(match.index)}`
}

/** Insert a child fragment before the final closing tag with the requested local name. */
export function appendXmlChildByLocalName(xml: string, localName: string, child: string): string {
  const expression = new RegExp(`</(?:[A-Za-z_][\\w.-]*:)?${escapeRegExp(localName)}\\s*>`, 'gi')
  let match: RegExpExecArray | undefined
  for (const candidate of xml.matchAll(expression)) match = candidate
  if (match === undefined || match.index === undefined) {
    throw new OfficePackageError(`Expected closing XML tag for ${localName} was not found`)
  }
  return `${xml.slice(0, match.index)}${child}${xml.slice(match.index)}`
}

/** Replace an empty self-closing element by its explicit open/close form. */
export function expandEmptyXmlElement(xml: string, qualifiedName: string, child: string): string | undefined {
  const expression = new RegExp(`<${escapeRegExp(qualifiedName)}\\b([^>]*)/\\s*>`, 'i')
  const match = expression.exec(xml)
  if (match === null) return undefined
  const attributes = match[1] ?? ''
  return `${xml.slice(0, match.index)}<${qualifiedName}${attributes}>${child}</${qualifiedName}>${xml.slice(match.index + match[0].length)}`
}

/**
 * Remove presentation-indent whitespace and comments without touching text
 * contained in `*:t` elements. This matches the intent of the Python pack
 * helper while avoiding an XML parser with external entity expansion.
 */
export function condenseOoxmlXml(xml: string): string {
  const tokens = xml.match(/<!--[\s\S]*?-->|<!\[CDATA\[[\s\S]*?\]\]>|<\?[\s\S]*?\?>|<!DOCTYPE[\s\S]*?>|<[^>]+>|[^<]+/g)
  if (tokens === null) throw new OfficePackageError('XML payload could not be tokenized')
  const stack: string[] = []
  const output: string[] = []
  for (const token of tokens) {
    if (token.startsWith('<!--')) continue
    if (token.startsWith('<![CDATA[') || token.startsWith('<?') || token.startsWith('<!DOCTYPE')) {
      output.push(token)
      continue
    }
    if (token.startsWith('</')) {
      const name = /^<\/\s*([^\s>]+)/.exec(token)?.[1]
      if (name !== undefined) {
        const expected = stack.pop()
        if (expected !== undefined && expected !== name) {
          throw new OfficePackageError(`Malformed XML nesting: expected </${expected}> but received </${name}>`)
        }
      }
      output.push(token)
      continue
    }
    if (token.startsWith('<')) {
      const name = /^<\s*([^\s/>]+)/.exec(token)?.[1]
      if (name === undefined) throw new OfficePackageError(`Malformed XML tag: ${token.slice(0, 80)}`)
      if (!/\/\s*>$/.test(token)) stack.push(name)
      output.push(token)
      continue
    }
    const parent = stack.at(-1)
    if (!token.trim() && xmlLocalName(parent ?? '') !== 't') continue
    output.push(token)
  }
  if (stack.length) throw new OfficePackageError(`Malformed XML: unclosed <${stack.at(-1) ?? ''}>`)
  return output.join('')
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}
