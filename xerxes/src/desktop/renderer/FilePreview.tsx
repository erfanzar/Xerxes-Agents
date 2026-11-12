// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * What the Files panel shows for a file that is not just text: CSV and TSV
 * as a table, XLSX as sheet tabs and cells, DOCX as headings, paragraphs,
 * lists and tables, images, PDFs in Chromium's own viewer, and HTML in a
 * sandboxed frame with scripts off. The daemon parses the file
 * (`workspace.filePreview` with `kinds`, ui/PROTOCOL.md); this file reads its
 * reply and draws it. Everything is read-only; text stays in the editor.
 */

import { useEffect, useMemo, useState, type ReactElement, type ReactNode } from 'react'

import { desktopError, record, text, type RpcRecord } from './desktopRpc.js'
import { isVscodeHost } from './host.js'

/** Kinds this renderer can draw, sent as `kinds` on every preview request. */
export const PREVIEW_KINDS = ['table', 'image', 'pdf', 'sheet', 'document', 'html'] as const
type RequestKind = (typeof PREVIEW_KINDS)[number]

/** The rendered view a file has besides its source, toggled by the panel's Source button. */
export type RenderedView = 'markdown' | 'table' | 'image' | 'html'

export function renderedViewOf(path: string): RenderedView | undefined {
  if (/\.(md|mdx|markdown)$/i.test(path)) return 'markdown'
  if (/\.(csv|tsv|tab)$/i.test(path)) return 'table'
  if (/\.svg$/i.test(path)) return 'image'
  if (/\.(html?|xhtml)$/i.test(path)) return 'html'
  return undefined
}

/** The kinds to ask for: in Source view, the file's rendered kind is left out so its text comes back. */
export function requestKinds(path: string, source: boolean): RequestKind[] {
  const view = renderedViewOf(path)
  return source && view && view !== 'markdown' ? PREVIEW_KINDS.filter(kind => kind !== view) : [...PREVIEW_KINDS]
}

/** A text file as the editor holds it (the panel's OpenFile). */
export interface PreviewText { readonly path: string; readonly content: string; readonly version: string | null; readonly truncated: boolean }

export interface DocumentRun { readonly text: string; readonly bold?: boolean; readonly italic?: boolean; readonly underline?: boolean }
export type DocumentBlock =
  | { readonly type: 'heading'; readonly level: number; readonly text: string }
  | { readonly type: 'paragraph'; readonly runs: readonly DocumentRun[] }
  | { readonly type: 'list_item'; readonly ordered: boolean; readonly level: number; readonly marker: string; readonly runs: readonly DocumentRun[] }
  | { readonly type: 'table'; readonly rows: readonly (readonly string[])[] }

export type RichPreview =
  | { readonly kind: 'table'; readonly path: string; readonly header: readonly string[]; readonly rows: readonly (readonly string[])[]; readonly totalRows: number; readonly totalColumns: number; readonly truncated: boolean; readonly encoding: string }
  | { readonly kind: 'sheet'; readonly path: string; readonly sheets: readonly { readonly name: string; readonly hidden: boolean }[]; readonly sheet: number; readonly rows: readonly (readonly string[])[]; readonly totalRows: number; readonly totalColumns: number; readonly truncated: boolean; readonly notice: string }
  | { readonly kind: 'document'; readonly path: string; readonly blocks: readonly DocumentBlock[]; readonly truncated: boolean }
  | { readonly kind: 'image'; readonly path: string; readonly mediaType: string; readonly data: string }
  | { readonly kind: 'pdf'; readonly path: string; readonly data: string }
  | { readonly kind: 'html'; readonly path: string; readonly content: string }
  | { readonly kind: 'unavailable'; readonly path: string; readonly reason: string; readonly message: string }

/** A reply as the panel uses it: text for the editor, a rich view, or both (HTML has source and a rendering). */
export interface PreviewReply { readonly file?: PreviewText; readonly rich?: RichPreview }

const strings = (value: unknown): string[] => Array.isArray(value) ? value.map(cell => typeof cell === 'string' ? cell : String(cell ?? '')) : []
const grid = (value: unknown): string[][] => Array.isArray(value) ? value.map(strings) : []
const count = (value: unknown, fallback = 0): number => typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : fallback

function runsOf(value: unknown): DocumentRun[] {
  if (!Array.isArray(value)) return []
  return value.flatMap(item => {
    if (!item || typeof item !== 'object') return []
    const run = item as RpcRecord
    return [{ text: text(run.text), ...(run.bold === true ? { bold: true } : {}), ...(run.italic === true ? { italic: true } : {}), ...(run.underline === true ? { underline: true } : {}) }]
  })
}

function blocksOf(value: unknown): DocumentBlock[] {
  if (!Array.isArray(value)) return []
  return value.flatMap((item): DocumentBlock[] => {
    if (!item || typeof item !== 'object') return []
    const block = item as RpcRecord
    switch (block.type) {
      case 'heading': return [{ type: 'heading', level: Math.min(6, Math.max(1, count(block.level, 1))), text: text(block.text) }]
      case 'paragraph': return [{ type: 'paragraph', runs: runsOf(block.runs) }]
      case 'list_item': return [{ type: 'list_item', ordered: block.ordered === true, level: Math.min(8, count(block.level)), marker: text(block.marker), runs: runsOf(block.runs) }]
      case 'table': return [{ type: 'table', rows: grid(block.rows) }]
      default: return []
    }
  })
}

/**
 * Read a `workspace.filePreview` reply. A typed `{ok:false,reason}` is a
 * preview with nothing to show (too large, not a format this can read); any
 * other failure throws. A reply without `kind` is an older runtime's text.
 */
export function previewReplyOf(path: string, value: unknown): PreviewReply {
  const reply = record(value)
  if (reply.ok === false) {
    if (typeof reply.reason === 'string') return { rich: { kind: 'unavailable', path, reason: reply.reason, message: text(reply.error) || 'This file cannot be previewed.' } }
    throw new Error(text(reply.error) || 'The file could not be read.')
  }
  const textFile = (): PreviewText => ({ path, content: text(reply.content), version: typeof reply.version === 'string' ? reply.version : null, truncated: reply.truncated === true })
  switch (reply.kind) {
    case undefined:
    case 'text':
      return { file: textFile() }
    case 'html':
      return { file: textFile(), rich: { kind: 'html', path, content: text(reply.content) } }
    case 'table':
      return { rich: { kind: 'table', path, header: strings(reply.header), rows: grid(reply.rows), totalRows: count(reply.total_rows), totalColumns: count(reply.total_columns), truncated: reply.truncated === true, encoding: text(reply.encoding) || 'utf-8' } }
    case 'sheet': {
      const sheets = Array.isArray(reply.sheets) ? reply.sheets.map(item => { const sheet = record(item); return { name: text(sheet.name), hidden: sheet.hidden === true } }) : []
      return { rich: { kind: 'sheet', path, sheets, sheet: count(reply.sheet), rows: grid(reply.rows), totalRows: count(reply.total_rows), totalColumns: count(reply.total_columns), truncated: reply.truncated === true, notice: text(reply.notice) } }
    }
    case 'document':
      return { rich: { kind: 'document', path, blocks: blocksOf(reply.blocks), truncated: reply.truncated === true } }
    case 'image':
      if (!/^image\/(png|jpeg|gif|webp|svg\+xml)$/.test(text(reply.media_type))) throw new Error('The runtime sent an image type the app does not show.')
      return { rich: { kind: 'image', path, mediaType: text(reply.media_type), data: text(reply.data) } }
    case 'pdf':
      return { rich: { kind: 'pdf', path, data: text(reply.data) } }
    default:
      return { rich: { kind: 'unavailable', path, reason: 'unsupported', message: 'Update the app to preview this kind of file.' } }
  }
}

/** Ask the runtime for a preview of `path` as the panel shows it. */
export async function requestPreview(
  bridge: { call(method: string, params?: RpcRecord): Promise<unknown> },
  sessionKey: string,
  path: string,
  options: { readonly source: boolean; readonly sheet: number },
): Promise<PreviewReply> {
  const reply = await bridge.call('workspace.filePreview', {
    path, kinds: requestKinds(path, options.source), ...(options.sheet ? { sheet: options.sheet } : {}),
    ...(sessionKey ? { session_key: sessionKey } : {}),
  })
  return previewReplyOf(path, reply)
}

const fileName = (path: string): string => path.split('/').filter(Boolean).at(-1) ?? path

/** "Showing 1,000 of 52,340 rows", or nothing when everything is shown. */
export function shownNote(shownRows: number, totalRows: number, shownColumns: number, totalColumns: number, cut: boolean): string {
  const parts: string[] = []
  if (shownRows < totalRows) parts.push(`Showing ${shownRows.toLocaleString('en-US')} of ${totalRows.toLocaleString('en-US')} rows`)
  if (shownColumns < totalColumns) parts.push(`${parts.length ? 'first' : 'Showing the first'} ${shownColumns.toLocaleString('en-US')} of ${totalColumns.toLocaleString('en-US')} columns`)
  if (!parts.length && cut) parts.push('Long cells are shortened')
  return parts.join(', ')
}

function Note({ children }: { children: ReactNode }): ReactElement {
  return <p className="file-preview-note" role="note">{children}</p>
}

function TableView({ preview }: { preview: Extract<RichPreview, { kind: 'table' }> }): ReactElement {
  const note = shownNote(preview.rows.length, preview.totalRows, preview.header.length, preview.totalColumns, preview.truncated)
  return <div className="file-grid-view">
    {(note || preview.encoding !== 'utf-8') && <Note>{[note, preview.encoding === 'windows-1252' ? 'Read as Windows-1252 text' : preview.encoding.startsWith('utf-16') ? 'Read as UTF-16 text' : ''].filter(Boolean).join('. ')}</Note>}
    <div className="file-grid" role="region" aria-label={`Table: ${fileName(preview.path)}`} tabIndex={0}>
      <table>
        <thead><tr><th className="file-grid__index" scope="col"><span className="sr-only">Row</span></th>{preview.header.map((cell, index) => <th key={index} scope="col">{cell}</th>)}</tr></thead>
        <tbody>{preview.rows.map((row, index) => <tr key={index}><th className="file-grid__index" scope="row">{index + 1}</th>{row.map((cell, column) => <td key={column} title={cell.length > 30 ? cell : undefined}>{cell}</td>)}</tr>)}</tbody>
      </table>
      {!preview.rows.length && <p className="file-preview-empty">{preview.header.length ? 'The table has a header row and no data rows.' : 'The file is empty.'}</p>}
    </div>
  </div>
}

/** Spreadsheet column letters: 0 is A, 26 is AA. */
export function columnName(index: number): string {
  let name = ''
  for (let value = index + 1; value > 0; value = Math.floor((value - 1) / 26)) name = String.fromCharCode(65 + ((value - 1) % 26)) + name
  return name
}

function SheetView({ preview, requested, choose }: { preview: Extract<RichPreview, { kind: 'sheet' }>; requested: number; choose: (index: number) => void }): ReactElement {
  const width = preview.rows[0]?.length ?? 0
  const note = shownNote(preview.rows.length, preview.totalRows, width, preview.totalColumns, preview.truncated)
  const loading = requested !== preview.sheet
  return <div className="file-grid-view">
    {preview.sheets.length > 1 && <div className="file-sheet-tabs" role="tablist" aria-label="Sheets">
      {preview.sheets.map((sheet, index) => <button key={index} role="tab" aria-selected={index === preview.sheet} aria-controls="file-sheet-grid"
        className={sheet.hidden ? 'is-hidden' : undefined} title={sheet.hidden ? `${sheet.name} (hidden sheet)` : sheet.name}
        onClick={() => { if (index !== requested) choose(index) }}
        onKeyDown={event => {
          if (event.key !== 'ArrowRight' && event.key !== 'ArrowLeft') return
          event.preventDefault()
          const next = (index + (event.key === 'ArrowRight' ? 1 : -1) + preview.sheets.length) % preview.sheets.length
          const tabs = event.currentTarget.parentElement?.querySelectorAll<HTMLButtonElement>('[role=tab]')
          tabs?.[next]?.focus()
          choose(next)
        }}>{sheet.name}</button>)}
    </div>}
    {loading && <p className="file-preview-note" role="status">Loading {preview.sheets[requested]?.name ?? 'sheet'}…</p>}
    {!loading && note && <Note>{note}</Note>}
    {preview.notice
      ? <p className="file-preview-empty" id="file-sheet-grid">{preview.notice}</p>
      : <div key={preview.sheet} className="file-grid file-grid--sheet" id="file-sheet-grid" role="region" aria-label={`Sheet ${preview.sheets[preview.sheet]?.name ?? ''}`} tabIndex={0} aria-busy={loading}>
        {preview.rows.length ? <table>
          <thead><tr><th className="file-grid__index" scope="col"><span className="sr-only">Row</span></th>{Array.from({ length: width }, (_, index) => <th key={index} scope="col">{columnName(index)}</th>)}</tr></thead>
          <tbody>{preview.rows.map((row, index) => <tr key={index}><th className="file-grid__index" scope="row">{index + 1}</th>{row.map((cell, column) => <td key={column} className={/^-?[\d,.]+%?$/.test(cell) ? 'is-number' : undefined} title={cell.length > 30 ? cell : undefined}>{cell}</td>)}</tr>)}</tbody>
        </table> : <p className="file-preview-empty">This sheet is empty.</p>}
      </div>}
  </div>
}

function Runs({ runs }: { runs: readonly DocumentRun[] }): ReactElement {
  return <>{runs.map((run, index) => {
    let node: ReactNode = run.text
    if (run.underline) node = <u>{node}</u>
    if (run.italic) node = <em>{node}</em>
    if (run.bold) node = <strong>{node}</strong>
    return <span key={index}>{node}</span>
  })}</>
}

/** Consecutive list items become one list; each keeps the marker Word prints and its depth. */
function groupBlocks(blocks: readonly DocumentBlock[]): (DocumentBlock | { readonly type: 'list'; readonly items: Extract<DocumentBlock, { type: 'list_item' }>[] })[] {
  const out: (DocumentBlock | { type: 'list'; items: Extract<DocumentBlock, { type: 'list_item' }>[] })[] = []
  for (const block of blocks) {
    const last = out.at(-1)
    if (block.type === 'list_item') {
      if (last?.type === 'list') last.items.push(block)
      else out.push({ type: 'list', items: [block] })
    } else out.push(block)
  }
  return out
}

function DocumentView({ preview }: { preview: Extract<RichPreview, { kind: 'document' }> }): ReactElement {
  return <article className="file-document file-docx" tabIndex={0} aria-label={`Document: ${fileName(preview.path)}`}>
    {preview.truncated && <Note>This document is long, so only its beginning is shown.</Note>}
    <div className="md md--document">
      {groupBlocks(preview.blocks).map((block, index) => {
        switch (block.type) {
          case 'heading': {
            const Heading = `h${block.level}` as 'h1'
            return <Heading key={index}>{block.text}</Heading>
          }
          case 'paragraph': return <p key={index} className="file-docx__paragraph"><Runs runs={block.runs} /></p>
          case 'list': return <ul key={index} className="file-docx__list">{block.items.map((item, at) => <li key={at} style={{ paddingInlineStart: `${1.6 + item.level * 1.4}em` }}>
            <span className="file-docx__marker" aria-hidden={item.ordered ? undefined : true}>{item.marker}</span><Runs runs={item.runs} />
          </li>)}</ul>
          case 'table': return <div key={index} className="md__table-scroll" role="region" aria-label="Table contents" tabIndex={0}><table className="md__table">
            <tbody>{block.rows.map((row, at) => <tr key={at}>{row.map((cell, column) => <td key={column} title={cell.length > 30 ? cell : undefined}>{cell}</td>)}</tr>)}</tbody>
          </table></div>
          default: return null
        }
      })}
      {!preview.blocks.length && <p className="file-preview-empty">This document has no text to show.</p>}
    </div>
  </article>
}

function ImagePreview({ preview }: { preview: Extract<RichPreview, { kind: 'image' }> }): ReactElement {
  const [actual, setActual] = useState(false)
  return <div className={'file-image' + (actual ? ' is-actual' : '')}>
    <button className="file-image__frame" aria-pressed={actual} title={actual ? 'Fit to the panel' : 'Show at actual size'} onClick={() => setActual(value => !value)}>
      {/* An SVG shown as an image never runs its scripts or loads what it links to. */}
      <img src={`data:${preview.mediaType};base64,${preview.data}`} alt={fileName(preview.path)} draggable={false} />
    </button>
  </div>
}

function decodeBase64(data: string): Uint8Array {
  const binary = atob(data)
  const bytes = new Uint8Array(binary.length)
  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index)
  return bytes
}

function PdfPreview({ preview }: { preview: Extract<RichPreview, { kind: 'pdf' }> }): ReactElement {
  const url = useMemo(() => {
    if (typeof URL.createObjectURL !== 'function') return ''
    try { return URL.createObjectURL(new Blob([decodeBase64(preview.data) as Uint8Array<ArrayBuffer>], { type: 'application/pdf' })) }
    catch { return '' }
  }, [preview.data])
  useEffect(() => () => { if (url) URL.revokeObjectURL(url) }, [url])
  // The VS Code webview's policy has no frame source for the viewer.
  if (isVscodeHost()) return <Unavailable message="PDFs cannot be shown inside VS Code. Open the file in its default app to read it." />
  if (!url) return <Unavailable message="This PDF could not be prepared for viewing." />
  return <iframe className="file-pdf" src={url} title={`PDF: ${fileName(preview.path)}`} />
}

function HtmlPreview({ preview }: { preview: Extract<RichPreview, { kind: 'html' }> }): ReactElement {
  return <div className="file-html">
    <Note>Scripts, forms and web content are off in this preview.</Note>
    {/* An empty sandbox: no scripts, forms, popups or navigation, and an opaque origin. The page's own CSP still applies, so nothing remote loads. */}
    <iframe className="file-html__frame" sandbox="" referrerPolicy="no-referrer" srcDoc={preview.content} title={`Web page: ${fileName(preview.path)}`} />
  </div>
}

function Unavailable({ message, children }: { message: string; children?: ReactNode }): ReactElement {
  return <div className="file-preview-unavailable" role="status"><p>{message}</p>{children}</div>
}

/** One rich preview. `actions` are the Open/Show buttons, repeated where a file cannot be shown. */
export function RichFilePreview({ preview, requestedSheet, chooseSheet, actions }: {
  preview: RichPreview; requestedSheet: number; chooseSheet: (index: number) => void; actions: ReactNode
}): ReactElement {
  switch (preview.kind) {
    case 'table': return <TableView preview={preview} />
    case 'sheet': return <SheetView preview={preview} requested={requestedSheet} choose={chooseSheet} />
    case 'document': return <DocumentView preview={preview} />
    case 'image': return <ImagePreview preview={preview} />
    case 'pdf': return <PdfPreview preview={preview} />
    case 'html': return <HtmlPreview preview={preview} />
    case 'unavailable': return <Unavailable message={preview.message}>{actions}</Unavailable>
  }
}

const SHOW_LABEL = typeof navigator !== 'undefined' && /Mac/i.test(navigator.platform || navigator.userAgent) ? 'Show in Finder' : 'Show in folder'

/** "Open in default app" and "Show in Finder" for the selected file. */
export function FileActions({ path, cwd, remote }: { path: string; cwd: string; remote: boolean }): ReactElement {
  const [error, setError] = useState('')
  useEffect(() => setError(''), [path])
  const bridge = typeof window === 'undefined' ? undefined : window.xerxes
  const unavailable = remote ? 'The file is on the SSH host, not on this computer' : !bridge?.fileAction || !cwd ? 'Update the app to open files from here' : ''
  const run = (action: 'open' | 'reveal'): void => {
    setError('')
    if (!bridge?.fileAction) return
    void bridge.fileAction(action, path.replace(/^\.\//, ''), cwd)
      .then(result => { if (!result.ok) setError(result.error) }, reason => setError(desktopError(reason)))
  }
  return <>
    <button className="file-browser__action" disabled={Boolean(unavailable)} title={unavailable || 'Open the file in the app your computer uses for it'} onClick={() => run('open')}>Open in default app</button>
    <button className="file-browser__action" disabled={Boolean(unavailable)} title={unavailable || `${SHOW_LABEL}: ${path}`} onClick={() => run('reveal')}>{SHOW_LABEL}</button>
    {error && <span className="file-browser__action-error" role="alert">{error}</span>}
  </>
}
