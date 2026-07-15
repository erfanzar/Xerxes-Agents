// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { AmbiguousTrackedChangeAuthorError, OfficePackagePartNotFoundError } from './errors.js'
import { type OfficePackagePort } from './package.js'
import { readOfficeZip } from './zip.js'
import {
  directOoxmlElements,
  directOoxmlText,
  findOoxmlElements,
  getOoxmlAttribute,
  nextOoxmlElementIndex,
  onlyIgnorableOoxmlNodesBetween,
  parseOoxmlTree,
  removeOoxmlAttributes,
  removeOoxmlElements,
  replaceDirectOoxmlText,
  serializeOoxmlNode,
  serializeOoxmlTree,
  setOoxmlAttribute,
  visitOoxmlElements,
  type OoxmlTreeElement,
} from './xmlTree.js'
import { xmlLocalName } from './xml.js'

const WORD_DOCUMENT_PART = 'word/document.xml'

/** Result shared by native run and tracked-change reduction operations. */
export interface RedlineSimplificationResult {
  readonly count: number
  readonly message: string
}

/** Result from an XML-only redline transform. */
export interface RedlineXmlResult extends RedlineSimplificationResult {
  readonly xml: string
}

/** Coalesce adjacent same-style Word runs in an XML string. */
export function mergeDocumentRunsXml(xml: string): RedlineXmlResult {
  const document = parseOoxmlTree(xml)
  removeOoxmlElements(document, 'proofErr')
  visitOoxmlElements(document, element => {
    if (xmlLocalName(element.name) === 'r') removeOoxmlAttributes(element, name => name.toLowerCase().includes('rsid'))
  })
  let count = 0
  visitOoxmlElements(document, element => {
    if (element.children.some(child => child.kind === 'element' && xmlLocalName(child.name) === 'r')) {
      count += mergeRunsInContainer(element)
    }
  })
  return { count, message: `Merged ${count} runs`, xml: serializeOoxmlTree(document) }
}

/** Apply `mergeDocumentRunsXml` to an unpacked DOCX package. */
export async function mergeDocumentRuns(packageDirectory: OfficePackagePort): Promise<RedlineSimplificationResult> {
  if (!await packageDirectory.hasPart(WORD_DOCUMENT_PART)) throw new OfficePackagePartNotFoundError(WORD_DOCUMENT_PART)
  const result = mergeDocumentRunsXml(await packageDirectory.readText(WORD_DOCUMENT_PART))
  await packageDirectory.writeText(WORD_DOCUMENT_PART, result.xml)
  return { count: result.count, message: result.message }
}

/** Merge adjacent same-author `w:ins` and `w:del` ranges in an XML string. */
export function simplifyDocumentRedlinesXml(xml: string): RedlineXmlResult {
  const document = parseOoxmlTree(xml)
  let count = 0
  for (const container of [...findOoxmlElements(document, 'p'), ...findOoxmlElements(document, 'tc')]) {
    count += mergeTrackedChangesInContainer(container, 'ins')
    count += mergeTrackedChangesInContainer(container, 'del')
  }
  return { count, message: `Simplified ${count} tracked changes`, xml: serializeOoxmlTree(document) }
}

/** Apply `simplifyDocumentRedlinesXml` to an unpacked DOCX package. */
export async function simplifyDocumentRedlines(packageDirectory: OfficePackagePort): Promise<RedlineSimplificationResult> {
  if (!await packageDirectory.hasPart(WORD_DOCUMENT_PART)) throw new OfficePackagePartNotFoundError(WORD_DOCUMENT_PART)
  const result = simplifyDocumentRedlinesXml(await packageDirectory.readText(WORD_DOCUMENT_PART))
  await packageDirectory.writeText(WORD_DOCUMENT_PART, result.xml)
  return { count: result.count, message: result.message }
}

/** Count inserted and deleted revision ranges grouped by `w:author`. */
export function getTrackedChangeAuthorsXml(xml: string): ReadonlyMap<string, number> {
  const document = parseOoxmlTree(xml)
  const authors = new Map<string, number>()
  for (const element of [...findOoxmlElements(document, 'ins'), ...findOoxmlElements(document, 'del')]) {
    const author = getOoxmlAttribute(element, 'w:author')
    if (author) authors.set(author, (authors.get(author) ?? 0) + 1)
  }
  return authors
}

/** Count tracked-change authors from `word/document.xml` in an unpacked package. */
export async function getTrackedChangeAuthors(packageDirectory: OfficePackagePort): Promise<ReadonlyMap<string, number>> {
  if (!await packageDirectory.hasPart(WORD_DOCUMENT_PART)) return new Map()
  return getTrackedChangeAuthorsXml(await packageDirectory.readText(WORD_DOCUMENT_PART))
}

/**
 * Infer the sole author that added new tracked changes relative to a packed
 * DOCX baseline. A baseline with no new changes returns `defaultAuthor`;
 * multiple authors fail rather than silently choosing one.
 */
export async function inferTrackedChangeAuthor(
  packageDirectory: OfficePackagePort,
  originalDocx: Uint8Array,
  defaultAuthor = 'Claude',
): Promise<string> {
  if (typeof defaultAuthor !== 'string' || !defaultAuthor.trim()) throw new TypeError('defaultAuthor must be non-empty')
  const modifiedAuthors = await getTrackedChangeAuthors(packageDirectory)
  if (!modifiedAuthors.size) return defaultAuthor
  const originalXml = readOfficeZip(originalDocx).get(WORD_DOCUMENT_PART)
  const originalAuthors = originalXml === undefined
    ? new Map<string, number>()
    : getTrackedChangeAuthorsXml(new TextDecoder().decode(originalXml))
  const newAuthors = [...modifiedAuthors.entries()]
    .filter(([author, count]) => count > (originalAuthors.get(author) ?? 0))
    .map(([author]) => author)
  if (!newAuthors.length) return defaultAuthor
  if (newAuthors.length === 1) return newAuthors[0] ?? defaultAuthor
  throw new AmbiguousTrackedChangeAuthorError(
    `Multiple authors added new changes: ${newAuthors.join(', ')}. Cannot infer which author to validate.`,
  )
}

function mergeRunsInContainer(container: OoxmlTreeElement): number {
  let count = 0
  for (let index = 0; index < container.children.length; index += 1) {
    const run = container.children[index]
    if (run?.kind !== 'element' || xmlLocalName(run.name) !== 'r') continue
    while (true) {
      const nextIndex = nextOoxmlElementIndex(container.children, index)
      const next = nextIndex === undefined ? undefined : container.children[nextIndex]
      if (nextIndex === undefined || next?.kind !== 'element' || xmlLocalName(next.name) !== 'r' || !canMergeRuns(run, next)) {
        break
      }
      moveRunContent(run, next)
      container.children.splice(nextIndex, 1)
      count += 1
    }
    consolidateRunText(run)
  }
  return count
}

function canMergeRuns(left: OoxmlTreeElement, right: OoxmlTreeElement): boolean {
  const leftProperties = directOoxmlElements(left, 'rPr')[0]
  const rightProperties = directOoxmlElements(right, 'rPr')[0]
  if ((leftProperties === undefined) !== (rightProperties === undefined)) return false
  if (leftProperties === undefined || rightProperties === undefined) return true
  return serializeOoxmlNode(leftProperties) === serializeOoxmlNode(rightProperties)
}

function moveRunContent(target: OoxmlTreeElement, source: OoxmlTreeElement): void {
  target.children.push(...source.children.filter(child => child.kind === 'element' && xmlLocalName(child.name) !== 'rPr'))
  target.selfClosing = false
}

function consolidateRunText(run: OoxmlTreeElement): void {
  const textIndices = run.children
    .map((node, index) => node.kind === 'element' && xmlLocalName(node.name) === 't' ? index : undefined)
    .filter((index): index is number => index !== undefined)
  for (let offset = textIndices.length - 1; offset > 0; offset -= 1) {
    const currentIndex = textIndices[offset]
    const previousIndex = textIndices[offset - 1]
    if (currentIndex === undefined || previousIndex === undefined) continue
    const current = run.children[currentIndex]
    const previous = run.children[previousIndex]
    if (current?.kind !== 'element' || previous?.kind !== 'element') continue
    if (!onlyIgnorableOoxmlNodesBetween(run.children, previousIndex, currentIndex)) continue
    const merged = directOoxmlText(previous) + directOoxmlText(current)
    replaceDirectOoxmlText(previous, merged)
    if (merged.startsWith(' ') || merged.endsWith(' ')) {
      setOoxmlAttribute(previous, 'xml:space', 'preserve')
    } else {
      removeOoxmlAttributes(previous, name => name === 'xml:space')
    }
    run.children.splice(currentIndex, 1)
  }
}

function mergeTrackedChangesInContainer(container: OoxmlTreeElement, kind: 'del' | 'ins'): number {
  let count = 0
  let trackedIndex = 0
  while (true) {
    const tracked = container.children
      .map((node, index) => node.kind === 'element' && xmlLocalName(node.name) === kind ? index : undefined)
      .filter((index): index is number => index !== undefined)
    const currentIndex = tracked[trackedIndex]
    const nextIndex = tracked[trackedIndex + 1]
    if (currentIndex === undefined || nextIndex === undefined) return count
    const current = container.children[currentIndex]
    const next = container.children[nextIndex]
    if (current?.kind !== 'element' || next?.kind !== 'element' || !canMergeTrackedChanges(container, currentIndex, nextIndex)) {
      trackedIndex += 1
      continue
    }
    current.children.push(...next.children)
    current.selfClosing = false
    container.children.splice(nextIndex, 1)
    count += 1
  }
}

function canMergeTrackedChanges(container: OoxmlTreeElement, currentIndex: number, nextIndex: number): boolean {
  const current = container.children[currentIndex]
  const next = container.children[nextIndex]
  if (current?.kind !== 'element' || next?.kind !== 'element') return false
  if (getOoxmlAttribute(current, 'w:author') !== getOoxmlAttribute(next, 'w:author')) return false
  return onlyIgnorableOoxmlNodesBetween(container.children, currentIndex, nextIndex)
}
