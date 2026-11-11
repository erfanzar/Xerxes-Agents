// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * A DOCX body as readable blocks: headings (from built-in heading styles,
 * their localized copies and outline levels), paragraphs with bold, italic
 * and underline runs, numbered and bulleted list items with the markers Word
 * would print, and tables as rows of cell text. Tracked deletions, hidden
 * text and field codes are left out; images and text boxes are not shown.
 */

import { relationshipTargetPartName } from '../../skills/powerpoint/package.js'
import { decodeXmlEntities, parseOoxmlRelationships } from '../../skills/powerpoint/xml.js'
import type { OfficeZipReader } from '../../skills/powerpoint/zip.js'
import { capCell, DOCUMENT_LIMITS, PREVIEW_PART_LIMIT, type DocumentLimits } from './limits.js'
import { childElement, childElements, descend, ownText, parseXmlTree, type XmlNode } from './xmlTree.js'

export interface DocumentRun {
  readonly text: string
  readonly bold?: true
  readonly italic?: true
  readonly underline?: true
}

export type DocumentBlock =
  | { readonly type: 'heading'; readonly level: number; readonly text: string }
  | { readonly type: 'paragraph'; readonly runs: DocumentRun[] }
  | { readonly type: 'list_item'; readonly ordered: boolean; readonly level: number; readonly marker: string; readonly runs: DocumentRun[] }
  | { readonly type: 'table'; readonly rows: string[][] }

export interface DocumentPreview {
  readonly blocks: DocumentBlock[]
  readonly truncated: boolean
}

/** The document part is missing or is not WordprocessingML. */
export class DocumentFormatError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'DocumentFormatError'
  }
}

interface Style {
  readonly name: string
  readonly basedOn?: string
  readonly outline?: number
  readonly numbering?: { readonly id: string; readonly level?: number }
  readonly bold?: boolean
  readonly italic?: boolean
}

interface Level { readonly format: string; readonly text?: string; readonly start: number }

interface Numbering {
  readonly levels: ReadonlyMap<string, ReadonlyMap<number, Level>>
  readonly instances: ReadonlyMap<string, { readonly abstract: string; readonly starts: ReadonlyMap<number, number> }>
}

const val = (node: XmlNode | undefined): string | undefined => node?.attributes.val

/** A toggle property (`<w:b/>`, `<w:b w:val="0"/>`) as on, off or unset. */
function toggle(node: XmlNode | undefined): boolean | undefined {
  if (!node) return undefined
  const value = val(node)
  return value === undefined || !['0', 'false', 'off', 'none'].includes(value.toLowerCase())
}

function numberAttribute(value: string | undefined): number | undefined {
  if (value === undefined) return undefined
  const number = Number.parseInt(value, 10)
  return Number.isFinite(number) ? number : undefined
}

function relsPartOf(part: string): string {
  const slash = part.lastIndexOf('/')
  return `${part.slice(0, slash + 1)}_rels/${part.slice(slash + 1)}.rels`
}

function partsByType(zip: OfficeZipReader, part: string, limit: number): Map<string, string> {
  const rels = relsPartOf(part)
  const bytes = zip.read(rels, limit)
  const out = new Map<string, string>()
  if (bytes === undefined) return out
  for (const rel of parseOoxmlRelationships(new TextDecoder().decode(bytes))) {
    if (rel.targetMode?.toLowerCase() === 'external') continue
    const type = rel.type.slice(rel.type.lastIndexOf('/') + 1)
    try { if (!out.has(type)) out.set(type, relationshipTargetPartName(rels, decodeXmlEntities(rel.target))) }
    catch { /* a target outside the package is not a part we can read */ }
  }
  return out
}

function readTree(zip: OfficeZipReader, part: string | undefined, limit: number): XmlNode | undefined {
  if (part === undefined) return undefined
  const bytes = zip.read(part, limit)
  return bytes === undefined ? undefined : parseXmlTree(new TextDecoder().decode(bytes))
}

function readStyles(tree: XmlNode | undefined): Map<string, Style> {
  const styles = new Map<string, Style>()
  for (const style of childElements(childElement(tree, 'styles'), 'style')) {
    const id = style.attributes.styleId
    if (!id) continue
    const pPr = childElement(style, 'pPr')
    const numPr = childElement(pPr, 'numPr')
    const numId = val(childElement(numPr, 'numId'))
    const rPr = childElement(style, 'rPr')
    const basedOn = val(childElement(style, 'basedOn'))
    const outline = numberAttribute(val(childElement(pPr, 'outlineLvl')))
    const level = numberAttribute(val(childElement(numPr, 'ilvl')))
    const bold = toggle(childElement(rPr, 'b'))
    const italic = toggle(childElement(rPr, 'i'))
    styles.set(id, {
      name: (val(childElement(style, 'name')) ?? id).toLowerCase(),
      ...(basedOn ? { basedOn } : {}),
      ...(outline === undefined ? {} : { outline }),
      ...(numId ? { numbering: { id: numId, ...(level === undefined ? {} : { level }) } } : {}),
      ...(bold === undefined ? {} : { bold }),
      ...(italic === undefined ? {} : { italic }),
    })
  }
  return styles
}

function readNumbering(tree: XmlNode | undefined): Numbering {
  const root = childElement(tree, 'numbering')
  const levels = new Map<string, Map<number, Level>>()
  for (const abstract of childElements(root, 'abstractNum')) {
    const id = abstract.attributes.abstractNumId
    if (id === undefined) continue
    const byLevel = new Map<number, Level>()
    for (const lvl of childElements(abstract, 'lvl')) {
      const index = numberAttribute(lvl.attributes.ilvl)
      if (index === undefined) continue
      const text = val(childElement(lvl, 'lvlText'))
      byLevel.set(index, { format: val(childElement(lvl, 'numFmt')) ?? 'decimal', start: numberAttribute(val(childElement(lvl, 'start'))) ?? 1, ...(text === undefined ? {} : { text }) })
    }
    levels.set(id, byLevel)
  }
  const instances = new Map<string, { abstract: string; starts: Map<number, number> }>()
  for (const num of childElements(root, 'num')) {
    const id = num.attributes.numId
    const abstract = val(childElement(num, 'abstractNumId'))
    if (id === undefined || abstract === undefined) continue
    const starts = new Map<number, number>()
    for (const override of childElements(num, 'lvlOverride')) {
      const index = numberAttribute(override.attributes.ilvl)
      const start = numberAttribute(val(childElement(override, 'startOverride')))
      if (index !== undefined && start !== undefined) starts.set(index, start)
    }
    instances.set(id, { abstract, starts })
  }
  return { levels, instances }
}

/** Walk a style and the styles it is based on, nearest first (cycles cut off). */
function* styleChain(styles: ReadonlyMap<string, Style>, id: string | undefined): Generator<[string, Style | undefined]> {
  const seen = new Set<string>()
  for (let current = id; current && !seen.has(current) && seen.size < 16; current = styles.get(current)?.basedOn) {
    seen.add(current)
    yield [current, styles.get(current)]
  }
}

function headingLevel(styles: ReadonlyMap<string, Style>, styleId: string | undefined, direct: number | undefined): number | undefined {
  if (direct !== undefined) return direct >= 0 && direct < 6 ? direct + 1 : undefined
  for (const [id, style] of styleChain(styles, styleId)) {
    const named = /^heading\s*([1-9])$/.exec(style?.name ?? id.toLowerCase())?.[1]
    if (named) return Math.min(6, Number(named))
    if ((style?.name ?? id.toLowerCase()) === 'title') return 1
    if (style?.outline !== undefined) return style.outline >= 0 && style.outline < 6 ? style.outline + 1 : undefined
  }
  return undefined
}

const BULLETS = ['•', '◦', '▪']

function roman(value: number): string {
  const table: [number, string][] = [[1000, 'm'], [900, 'cm'], [500, 'd'], [400, 'cd'], [100, 'c'], [90, 'xc'], [50, 'l'], [40, 'xl'], [10, 'x'], [9, 'ix'], [5, 'v'], [4, 'iv'], [1, 'i']]
  let out = ''
  let rest = Math.max(1, Math.min(3999, value))
  for (const [amount, letters] of table) while (rest >= amount) { out += letters; rest -= amount }
  return out
}

function letters(value: number): string {
  // Word's letter lists repeat the letter past z: a … z, aa … zz.
  const index = Math.max(1, value) - 1
  return String.fromCharCode(97 + (index % 26)).repeat(Math.floor(index / 26) + 1)
}

function formatCounter(value: number, format: string): string {
  switch (format) {
    case 'lowerLetter': return letters(value)
    case 'upperLetter': return letters(value).toUpperCase()
    case 'lowerRoman': return roman(value)
    case 'upperRoman': return roman(value).toUpperCase()
    case 'decimalZero': return String(value).padStart(2, '0')
    default: return String(value)
  }
}

/** Numbers per list instance and level, the way Word continues and restarts them. */
class ListCounters {
  private readonly counters = new Map<string, number[]>()

  constructor(private readonly numbering: Numbering) {}

  marker(numId: string, level: number): { ordered: boolean; marker: string } {
    const instance = this.numbering.instances.get(numId)
    const levels = instance ? this.numbering.levels.get(instance.abstract) : undefined
    const definition = (index: number): Level => levels?.get(index) ?? { format: index === 0 ? 'decimal' : 'bullet', start: 1 }
    const start = (index: number): number => instance?.starts.get(index) ?? definition(index).start
    const own = definition(level)
    if (own.format === 'bullet') return { ordered: false, marker: BULLETS[level % BULLETS.length]! }
    if (own.format === 'none') return { ordered: false, marker: '' }
    const values = this.counters.get(numId) ?? []
    for (let index = 0; index < level; index += 1) values[index] ??= start(index)
    values[level] = values[level] === undefined ? start(level) : values[level]! + 1
    values.length = level + 1
    this.counters.set(numId, values)
    const template = own.text ?? `%${level + 1}.`
    const marker = template.replace(/%([1-9])/g, (_, digit: string) => {
      const index = Number(digit) - 1
      return formatCounter(values[index] ?? start(index), definition(index).format)
    })
    return { ordered: true, marker }
  }
}

interface Context {
  readonly styles: ReadonlyMap<string, Style>
  readonly lists: ListCounters
}

/** Children of a container, looking through content controls and custom XML wrappers. */
function* contentOf(node: XmlNode): Generator<XmlNode> {
  for (const child of childElements(node)) {
    if (child.name === 'sdt') { const content = childElement(child, 'sdtContent'); if (content) yield* contentOf(content) }
    else if (child.name === 'customXml' || child.name === 'smartTag') yield* contentOf(child)
    else yield child
  }
}

function runFormat(rPr: XmlNode | undefined, styles: ReadonlyMap<string, Style>): { bold: boolean; italic: boolean; underline: boolean; hidden: boolean } {
  let bold = toggle(childElement(rPr, 'b'))
  let italic = toggle(childElement(rPr, 'i'))
  for (const [, style] of styleChain(styles, val(childElement(rPr, 'rStyle')))) {
    bold ??= style?.bold
    italic ??= style?.italic
  }
  const underline = childElement(rPr, 'u')
  return { bold: bold === true, italic: italic === true, underline: underline !== undefined && val(underline) !== 'none', hidden: toggle(childElement(rPr, 'vanish')) === true }
}

function runText(run: XmlNode): string {
  let text = ''
  for (const child of childElements(run)) {
    switch (child.name) {
      case 't': text += ownText(child); break
      case 'tab': case 'ptab': text += '\t'; break
      case 'br': case 'cr': text += '\n'; break
      case 'noBreakHyphen': text += '-'; break
      default: break
    }
  }
  return text
}

const RUN_CONTAINERS = new Set(['hyperlink', 'ins', 'smartTag', 'customXml', 'fldSimple', 'bdo', 'dir', 'moveTo', 'sdtContent', 'sdt'])

function paragraphRuns(paragraph: XmlNode, styles: ReadonlyMap<string, Style>): DocumentRun[] {
  const runs: DocumentRun[] = []
  const visit = (node: XmlNode): void => {
    for (const child of childElements(node)) {
      if (child.name === 'r') {
        const format = runFormat(childElement(child, 'rPr'), styles)
        const text = runText(child)
        if (format.hidden || !text) continue
        const run: DocumentRun = { text, ...(format.bold ? { bold: true } : {}), ...(format.italic ? { italic: true } : {}), ...(format.underline ? { underline: true } : {}) }
        const last = runs.at(-1)
        if (last && last.bold === run.bold && last.italic === run.italic && last.underline === run.underline) runs[runs.length - 1] = { ...last, text: last.text + text }
        else runs.push(run)
      } else if (RUN_CONTAINERS.has(child.name)) visit(child)
    }
  }
  visit(paragraph)
  return runs
}

const plain = (runs: readonly DocumentRun[]): string => runs.map(run => run.text).join('')

function cellText(cell: XmlNode, styles: ReadonlyMap<string, Style>): string {
  const lines: string[] = []
  for (const child of contentOf(cell)) {
    if (child.name === 'p') lines.push(plain(paragraphRuns(child, styles)))
    else if (child.name === 'tbl') lines.push(tableRows(child, styles, 50, 20).map(row => row.join(' | ')).join('\n'))
  }
  return lines.join('\n').replace(/\n{3,}/g, '\n\n').trim()
}

function tableRows(table: XmlNode, styles: ReadonlyMap<string, Style>, maxRows: number, maxColumns: number): string[][] {
  const rows: string[][] = []
  for (const row of contentOf(table)) {
    if (row.name !== 'tr') continue
    if (rows.length >= maxRows) break
    const cells: string[] = []
    for (const cell of contentOf(row)) {
      if (cell.name !== 'tc') continue
      cells.push(cellText(cell, styles))
      // A cell spanning columns keeps the columns after it aligned.
      const span = numberAttribute(val(descend(cell, 'tcPr', 'gridSpan'))) ?? 1
      for (let extra = 1; extra < Math.min(span, maxColumns); extra += 1) cells.push('')
    }
    rows.push(cells.slice(0, maxColumns))
  }
  const width = Math.max(0, ...rows.map(row => row.length))
  return rows.map(row => row.length < width ? [...row, ...Array<string>(width - row.length).fill('')] : row)
}

function paragraphBlock(paragraph: XmlNode, context: Context): DocumentBlock | undefined {
  const pPr = childElement(paragraph, 'pPr')
  const styleId = val(childElement(pPr, 'pStyle'))
  const runs = paragraphRuns(paragraph, context.styles)
  const text = plain(runs)
  if (!text.trim()) return undefined
  const level = headingLevel(context.styles, styleId, numberAttribute(val(childElement(pPr, 'outlineLvl'))))
  if (level !== undefined) return { type: 'heading', level, text: text.replace(/\s+/g, ' ').trim() }
  const numPr = childElement(pPr, 'numPr')
  let numId = val(childElement(numPr, 'numId'))
  let listLevel = numberAttribute(val(childElement(numPr, 'ilvl')))
  if (numId === undefined) {
    for (const [, style] of styleChain(context.styles, styleId)) {
      if (style?.numbering) { numId = style.numbering.id; listLevel ??= style.numbering.level; break }
    }
  }
  if (numId !== undefined && numId !== '0') {
    const depth = Math.max(0, Math.min(8, listLevel ?? 0))
    const { ordered, marker } = context.lists.marker(numId, depth)
    return { type: 'list_item', ordered, level: depth, marker, runs }
  }
  return { type: 'paragraph', runs }
}

const blockChars = (block: DocumentBlock): number => {
  switch (block.type) {
    case 'heading': return block.text.length
    case 'table': return block.rows.reduce((sum, row) => sum + row.reduce((width, cell) => width + cell.length, 0), 0)
    default: return plain(block.runs).length
  }
}

/** Read the body of a DOCX archive as preview blocks. */
export function readWordDocument(zip: OfficeZipReader, limits: DocumentLimits = DOCUMENT_LIMITS, partLimit = PREVIEW_PART_LIMIT): DocumentPreview {
  const documentPart = partsByType(zip, '', partLimit).get('officeDocument') ?? 'word/document.xml'
  const tree = readTree(zip, documentPart, partLimit)
  const body = descend(tree, 'document', 'body')
  if (!body) throw new DocumentFormatError('The document part is missing, so this is not a Word document')
  const related = partsByType(zip, documentPart, partLimit)
  const context: Context = {
    styles: readStyles(readTree(zip, related.get('styles'), partLimit)),
    lists: new ListCounters(readNumbering(readTree(zip, related.get('numbering'), partLimit))),
  }
  const blocks: DocumentBlock[] = []
  let budget = limits.maxChars
  let truncated = false
  for (const node of contentOf(body)) {
    let block: DocumentBlock | undefined
    if (node.name === 'p') block = paragraphBlock(node, context)
    else if (node.name === 'tbl') {
      const rows = tableRows(node, context.styles, limits.maxTableRows + 1, limits.maxTableColumns + 1)
      if (rows.length > limits.maxTableRows || rows.some(row => row.length > limits.maxTableColumns)) truncated = true
      const cut = rows.slice(0, limits.maxTableRows).map(row => row.slice(0, limits.maxTableColumns).map(cell => capCell(cell, 2000)))
      if (cut.length) block = { type: 'table', rows: cut }
    }
    if (!block) continue
    if (blocks.length >= limits.maxBlocks || budget - blockChars(block) < 0) { truncated = true; break }
    budget -= blockChars(block)
    blocks.push(block)
  }
  return { blocks, truncated }
}
