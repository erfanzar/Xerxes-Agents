// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { OfficePackageError } from './errors.js'
import { xmlLocalName } from './xml.js'

/** A mutable XML attribute retained without resolving XML entities. */
export interface OoxmlTreeAttribute {
  name: string
  value: string
}

/** An element node in the small OOXML-only tree model used by redline helpers. */
export interface OoxmlTreeElement {
  attributes: OoxmlTreeAttribute[]
  children: OoxmlTreeNode[]
  kind: 'element'
  name: string
  selfClosing: boolean
}

/** An ordinary XML text node. */
export interface OoxmlTreeText {
  kind: 'text'
  value: string
}

/** A comment, declaration, processing instruction, or CDATA token retained verbatim. */
export interface OoxmlTreeRaw {
  kind: 'raw'
  value: string
}

export type OoxmlTreeNode = OoxmlTreeElement | OoxmlTreeRaw | OoxmlTreeText

/** Root container for one parsed OOXML XML part. */
export interface OoxmlTreeDocument {
  children: OoxmlTreeNode[]
}

/** Parse regular OOXML XML into a deliberately limited, entity-safe mutable tree. */
export function parseOoxmlTree(xml: string): OoxmlTreeDocument {
  const tokens = xml.match(/<!--[\s\S]*?-->|<!\[CDATA\[[\s\S]*?\]\]>|<\?[\s\S]*?\?>|<!DOCTYPE[\s\S]*?>|<[^>]+>|[^<]+/g)
  if (tokens === null) throw new OfficePackageError('XML payload could not be tokenized')
  const document: OoxmlTreeDocument = { children: [] }
  const stack: OoxmlTreeElement[] = []
  for (const token of tokens) {
    const children = stack.at(-1)?.children ?? document.children
    if (token.startsWith('<!--') || token.startsWith('<![CDATA[') || token.startsWith('<?') || token.startsWith('<!DOCTYPE')) {
      children.push({ kind: 'raw', value: token })
      continue
    }
    if (token.startsWith('</')) {
      const name = /^<\/\s*([^\s>]+)/.exec(token)?.[1]
      const open = stack.pop()
      if (name === undefined || open === undefined || open.name !== name) {
        throw new OfficePackageError(`Malformed XML close tag: ${token.slice(0, 80)}`)
      }
      continue
    }
    if (token.startsWith('<')) {
      const match = /^<\s*([A-Za-z_][\w:.-]*)([\s\S]*?)>$/.exec(token)
      if (match === null) throw new OfficePackageError(`Malformed XML open tag: ${token.slice(0, 80)}`)
      const rawAttributes = match[2] ?? ''
      const selfClosing = /\/\s*$/.test(rawAttributes)
      const attributes = parseTreeAttributes(rawAttributes.replace(/\/\s*$/, ''))
      const element: OoxmlTreeElement = {
        attributes,
        children: [],
        kind: 'element',
        name: match[1] ?? '',
        selfClosing,
      }
      children.push(element)
      if (!selfClosing) stack.push(element)
      continue
    }
    children.push({ kind: 'text', value: token })
  }
  if (stack.length) throw new OfficePackageError(`Malformed XML: unclosed <${stack.at(-1)?.name ?? ''}>`)
  return document
}

/** Serialize the small OOXML tree back to valid XML. */
export function serializeOoxmlTree(document: OoxmlTreeDocument): string {
  return document.children.map(serializeOoxmlNode).join('')
}

/** Serialize a node on its own for semantic XML comparisons. */
export function serializeOoxmlNode(node: OoxmlTreeNode): string {
  if (node.kind === 'text' || node.kind === 'raw') return node.value
  const attributes = node.attributes.map(attribute => ` ${attribute.name}="${attribute.value.replaceAll('"', '&quot;')}"`).join('')
  if (node.selfClosing && !node.children.length) return `<${node.name}${attributes}/>`
  return `<${node.name}${attributes}>${node.children.map(serializeOoxmlNode).join('')}</${node.name}>`
}

/** Find all descendant elements with the requested namespace-agnostic local name. */
export function findOoxmlElements(document: OoxmlTreeDocument | OoxmlTreeElement, localName: string): OoxmlTreeElement[] {
  const matches: OoxmlTreeElement[] = []
  collectOoxmlElements(document.children, localName, matches)
  return matches
}

/** Visit every element in depth-first document order. */
export function visitOoxmlElements(
  document: OoxmlTreeDocument | OoxmlTreeElement,
  visitor: (element: OoxmlTreeElement) => void,
): void {
  for (const child of document.children) {
    if (child.kind !== 'element') continue
    visitor(child)
    visitOoxmlElements(child, visitor)
  }
}

/** Return direct element children whose local name matches. */
export function directOoxmlElements(element: OoxmlTreeElement, localName: string): OoxmlTreeElement[] {
  return element.children.filter((child): child is OoxmlTreeElement => child.kind === 'element' && xmlLocalName(child.name) === localName)
}

/** Read an attribute by qualified name, then by namespace-agnostic local name. */
export function getOoxmlAttribute(element: OoxmlTreeElement, name: string): string | undefined {
  const exact = element.attributes.find(attribute => attribute.name === name)
  if (exact !== undefined) return exact.value
  const requestedLocalName = xmlLocalName(name)
  return element.attributes.find(attribute => xmlLocalName(attribute.name) === requestedLocalName)?.value
}

/** Add or replace a qualified XML attribute. */
export function setOoxmlAttribute(element: OoxmlTreeElement, name: string, value: string): void {
  const existing = element.attributes.find(attribute => attribute.name === name)
  if (existing !== undefined) {
    existing.value = value
    return
  }
  element.attributes.push({ name, value })
}

/** Delete every attribute whose qualified name matches the supplied predicate. */
export function removeOoxmlAttributes(element: OoxmlTreeElement, predicate: (name: string) => boolean): void {
  element.attributes = element.attributes.filter(attribute => !predicate(attribute.name))
}

/** Return text directly held by an element without interpreting child elements. */
export function directOoxmlText(element: OoxmlTreeElement): string {
  return element.children.filter((child): child is OoxmlTreeText => child.kind === 'text').map(child => child.value).join('')
}

/** Replace an element's direct text content with one text node. */
export function replaceDirectOoxmlText(element: OoxmlTreeElement, value: string): void {
  element.children = [{ kind: 'text', value }]
  element.selfClosing = false
}

/** Return true when nodes between indices contain no meaningful content or element. */
export function onlyIgnorableOoxmlNodesBetween(nodes: readonly OoxmlTreeNode[], left: number, right: number): boolean {
  for (let index = left + 1; index < right; index += 1) {
    const node = nodes[index]
    if (node === undefined) continue
    if (node.kind === 'element') return false
    if (node.kind === 'text' && node.value.trim()) return false
    if (node.kind === 'raw' && !node.value.startsWith('<!--')) return false
  }
  return true
}

/** Return the next sibling element index, skipping whitespace text and comments. */
export function nextOoxmlElementIndex(nodes: readonly OoxmlTreeNode[], current: number): number | undefined {
  for (let index = current + 1; index < nodes.length; index += 1) {
    const node = nodes[index]
    if (node === undefined) continue
    if (node.kind === 'element') return index
    if (node.kind === 'text' && !node.value.trim()) continue
    if (node.kind === 'raw' && node.value.startsWith('<!--')) continue
    return undefined
  }
  return undefined
}

/** Remove all descendants that match a local element name. */
export function removeOoxmlElements(document: OoxmlTreeDocument | OoxmlTreeElement, localName: string): void {
  document.children = document.children.filter(node => node.kind !== 'element' || xmlLocalName(node.name) !== localName)
  for (const child of document.children) {
    if (child.kind === 'element') removeOoxmlElements(child, localName)
  }
}

function collectOoxmlElements(nodes: readonly OoxmlTreeNode[], localName: string, matches: OoxmlTreeElement[]): void {
  for (const node of nodes) {
    if (node.kind !== 'element') continue
    if (xmlLocalName(node.name) === localName) matches.push(node)
    collectOoxmlElements(node.children, localName, matches)
  }
}

function parseTreeAttributes(source: string): OoxmlTreeAttribute[] {
  const attributes: OoxmlTreeAttribute[] = []
  const expression = /([A-Za-z_][\w:.-]*)\s*=\s*(["'])([\s\S]*?)\2/g
  for (const match of source.matchAll(expression)) {
    const name = match[1]
    const value = match[3]
    if (name !== undefined && value !== undefined) attributes.push({ name, value })
  }
  return attributes
}
