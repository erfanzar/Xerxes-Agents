// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { open, realpath } from 'node:fs/promises'
import { constants } from 'node:fs'
import { pathToFileURL } from 'node:url'
import type { LspAdapter, LspRequest } from '../tools/claudeTools/search.js'
import { WorkspacePathResolver } from '../tools/pathSafety.js'
import { LspConnection, type LspConnectionOptions } from './connection.js'

export interface LspConnectionPort {
  readonly connected: boolean
  request(method: string, params?: unknown, signal?: AbortSignal): Promise<unknown>
  notify(method: string, params?: unknown): Promise<void>
  close(): Promise<void>
}
export interface LspHostOptions extends Omit<LspConnectionOptions, 'onNotification' | 'onRequest'> {
  readonly languageId: string
  readonly connectionFactory?: (options: LspConnectionOptions) => LspConnectionPort
  /** Answers to workspace/configuration, by section (e.g. python → { pythonPath }). */
  readonly configuration?: Readonly<Record<string, unknown>>
}
/** One completion item, bounded and stripped to what an editor popup needs. */
export interface LspCompletionItem {
  readonly label: string
  readonly kind?: number
  readonly detail?: string
  readonly documentation?: string
  readonly insertText: string
  /** Replaced span on the request line, as UTF-16 character offsets. */
  readonly replace?: { readonly start: number; readonly end: number }
  readonly sortText?: string
}
export const LSP_COMPLETION_LIMIT = 100
interface Document { text: string; version: number; diagnostics?: unknown[]; diagnosticError?: string }
const record = (value: unknown): value is Record<string, unknown> => !!value && typeof value === 'object' && !Array.isArray(value)
const methods = { definition: ['textDocument/definition', 'definitionProvider'], references: ['textDocument/references', 'referencesProvider'], hover: ['textDocument/hover', 'hoverProvider'], symbols: ['textDocument/documentSymbol', 'documentSymbolProvider'] } as const

/** One explicitly configured language server, scoped to one canonical workspace.
 * This scopes client reads; the configured executable is trusted host code, not a sandbox.
 */
export class LspHost implements LspAdapter {
  private connection!: LspConnectionPort
  private capabilities: Record<string, unknown> = {}
  private readonly documents = new Map<string, Document>()
  private readonly diagnosticWaiters = new Map<string, () => void>()
  private sequence = 0
  private queue: Promise<void> = Promise.resolve()
  private pending = 0
  private closed = false
  private syncKind = 0
  private readonly paths: WorkspacePathResolver
  private constructor(private readonly root: string, private readonly languageId: string) { this.paths = new WorkspacePathResolver(root) }

  static async start(options: LspHostOptions, signal?: AbortSignal): Promise<LspHost> {
    if (signal?.aborted) throw new DOMException('LSP initialization cancelled', 'AbortError')
    if (!options.languageId.trim() || options.languageId.length > 128) throw new Error('Configure an LSP language ID')
    const root = await realpath(options.cwd)
    if (signal?.aborted) throw new DOMException('LSP initialization cancelled', 'AbortError')
    const host = new LspHost(root, options.languageId)
    host.connection = (options.connectionFactory ?? (config => new LspConnection(config)))({
      command: options.command, cwd: root,
      ...(options.args ? { args: options.args } : {}), ...(options.env ? { env: options.env } : {}),
      ...(options.timeoutMs ? { timeoutMs: options.timeoutMs } : {}),
      onNotification: (method, params) => host.notification(method, params),
      onRequest: (method, params) => {
        if (method === 'workspace/configuration') {
          const items = record(params) && Array.isArray(params.items) ? params.items : []
          return { result: items.map(item => (record(item) && typeof item.section === 'string' ? options.configuration?.[item.section] : undefined) ?? {}) }
        }
        if (method === 'window/workDoneProgress/create' || method === 'client/registerCapability' || method === 'client/unregisterCapability') return { result: null }
        return undefined
      },
    })
    try {
      const result = await host.connection.request('initialize', {
        processId: process.pid, rootUri: pathToFileURL(root).href,
        workspaceFolders: [{ uri: pathToFileURL(root).href, name: root.split(/[\\/]/).at(-1) }],
        capabilities: { general: { positionEncodings: ['utf-16'] },
          workspace: { configuration: true, workspaceFolders: true },
          textDocument: { synchronization: { dynamicRegistration: false }, publishDiagnostics: { versionSupport: true },
            completion: { completionItem: { snippetSupport: false, documentationFormat: ['plaintext', 'markdown'] }, contextSupport: true } } },
      }, signal)
      if (!record(result) || !record(result.capabilities)) throw new Error('Language server returned invalid initialization capabilities')
      host.capabilities = result.capabilities
      if (host.capabilities.positionEncoding !== undefined && host.capabilities.positionEncoding !== 'utf-16') throw new Error('Language server must support UTF-16 positions')
      const sync = host.capabilities.textDocumentSync
      host.syncKind = typeof sync === 'number' ? sync : record(sync) && sync.openClose === true && typeof sync.change === 'number' ? sync.change : 0
      if (![1, 2].includes(host.syncKind)) throw new Error('Language server must support document open/change synchronization')
      await host.connection.notify('initialized', {})
      if (signal?.aborted) throw new DOMException('LSP initialization cancelled', 'AbortError')
      return host
    } catch (error) { await host.connection.close(); throw error }
  }

  get connected(): boolean { return !this.closed && this.connection.connected }

  async execute(request: LspRequest, signal?: AbortSignal): Promise<unknown> {
    if (this.closed || !this.connection.connected) throw new Error('Language server unavailable; restart its host')
    if (!Object.hasOwn(methods, request.action) && request.action !== 'diagnostics') throw new Error('Unsupported LSP action; use definition, references, hover, symbols or diagnostics')
    if (!Number.isSafeInteger(request.line) || !Number.isSafeInteger(request.character) || request.line < 0 || request.character < 0) throw new Error('LSP positions must be nonnegative integers')
    if (signal?.aborted) throw new DOMException('LSP request cancelled', 'AbortError')
    if (request.diagnosticsWaitMs !== undefined && (!Number.isSafeInteger(request.diagnosticsWaitMs) || request.diagnosticsWaitMs < 0 || request.diagnosticsWaitMs > 5000)) throw new Error("LSP diagnostics wait must be 0–5000 milliseconds")
    if (this.pending >= 32) throw new Error('Language server workspace queue is full')
    this.pending++
    const operation = this.queue.then(async () => {
      if (signal?.aborted) throw new DOMException('LSP request cancelled', 'AbortError')
      if (this.closed || !this.connection.connected) throw new Error('Language server unavailable; restart its host')
      const path = await this.paths.resolve(request.filePath)
      const text = await readDocument(path)
      if (signal?.aborted) throw new DOMException('LSP request cancelled', 'AbortError')
      const uri = pathToFileURL(path).href
      let document = this.documents.get(uri)
      if (!document || document.text !== text) {
        if (this.sequence >= 2_147_483_647) throw new Error('LSP document version limit reached; restart its host')
        const version = ++this.sequence
        if (!document && this.documents.size >= 32) {
          const oldest = this.documents.keys().next().value!
          await this.connection.notify('textDocument/didClose', { textDocument: { uri: oldest } })
          this.documents.delete(oldest)
        }
        const old = document
        document = { text, version }
        this.documents.set(uri, document)
        try {
          await this.connection.notify(old ? 'textDocument/didChange' : 'textDocument/didOpen', old
            ? { textDocument: { uri, version }, contentChanges: [{ text }] }
            : { textDocument: { uri, version, languageId: this.languageId, text } })
        } catch (error) { this.documents.delete(uri); this.closed = true; await this.connection.close(); throw error }
      }
      if (request.action === 'diagnostics') {
        await this.waitForDiagnostics(uri, document, request.diagnosticsWaitMs ?? 0, signal)
        if (signal?.aborted) throw new DOMException("LSP request cancelled", "AbortError")
        if (this.closed || !this.connection.connected) throw new Error("Language server unavailable during diagnostics")
        if (text !== await readDocument(await this.paths.resolve(path))) throw new Error("Document changed during diagnostics; retry against the current version")
        if (signal?.aborted) throw new DOMException("LSP request cancelled", "AbortError")
        if (this.closed || !this.connection.connected) throw new Error("Language server unavailable during diagnostics")
        return {
        uri, version: document.version, fresh: document.diagnostics !== undefined,
        diagnostics: document.diagnostics ?? [],
        ...(document.diagnostics === undefined ? { reason: document.diagnosticError ?? 'No diagnostics published for this document version yet' } : {}),
      }
      }
      const [method, capability] = methods[request.action as keyof typeof methods]
      if (this.capabilities[capability] !== true && !record(this.capabilities[capability])) throw new Error(`Language server does not support ${request.action}`)
      const lines = text.split('\n')
      if (request.action !== 'symbols' && (request.line >= lines.length || request.character > lines[request.line]!.replace(/\r$/, '').length)) throw new Error('LSP position is outside the document')
      const result = await this.connection.request(method, { textDocument: { uri },
        ...(request.action === 'symbols' ? {} : { position: { line: request.line, character: request.character } }),
        ...(request.action === 'references' ? { context: { includeDeclaration: true } } : {}),
      }, signal)
      if (signal?.aborted) throw new DOMException('LSP request cancelled', 'AbortError')
      if (this.closed || text !== await readDocument(await this.paths.resolve(path))) throw new Error('Document or language-server state changed during navigation; retry against the current version')
      return result
    }).finally(() => { this.pending-- })
    this.queue = operation.then(() => {}, () => {})
    if (!signal) return operation
    return new Promise((resolve, reject) => {
      const abort = () => reject(new DOMException('LSP request cancelled', 'AbortError'))
      signal.addEventListener('abort', abort, { once: true })
      if (signal.aborted) abort()
      operation.then(resolve, reject).finally(() => signal.removeEventListener('abort', abort))
    })
  }

  /**
   * Completion at a position of the editor's current (possibly unsaved) text.
   * The text is synchronized as the document's content; the file on disk is
   * only used to resolve and scope the path.
   */
  async complete(request: { readonly filePath: string; readonly text: string; readonly line: number; readonly character: number }, signal?: AbortSignal): Promise<{ readonly items: readonly LspCompletionItem[]; readonly incomplete: boolean }> {
    if (this.closed || !this.connection.connected) throw new Error('Language server unavailable; restart its host')
    if (!this.capabilities.completionProvider) throw new Error('Language server does not support completion')
    if (typeof request.text !== 'string' || request.text.length > 1_048_576) throw new Error('Completion text exceeds 1 MiB')
    if (!Number.isSafeInteger(request.line) || !Number.isSafeInteger(request.character) || request.line < 0 || request.character < 0) throw new Error('LSP positions must be nonnegative integers')
    const lines = request.text.split('\n')
    if (request.line >= lines.length || request.character > lines[request.line]!.replace(/\r$/, '').length) throw new Error('LSP position is outside the document')
    if (signal?.aborted) throw new DOMException('LSP request cancelled', 'AbortError')
    if (this.pending >= 32) throw new Error('Language server workspace queue is full')
    this.pending++
    const operation = this.queue.then(async () => {
      if (signal?.aborted) throw new DOMException('LSP request cancelled', 'AbortError')
      if (this.closed || !this.connection.connected) throw new Error('Language server unavailable; restart its host')
      const path = await this.paths.resolve(request.filePath)
      const uri = pathToFileURL(path).href
      const old = this.documents.get(uri)
      if (!old || old.text !== request.text) {
        if (this.sequence >= 2_147_483_647) throw new Error('LSP document version limit reached; restart its host')
        const version = ++this.sequence
        if (!old && this.documents.size >= 32) {
          const oldest = this.documents.keys().next().value!
          await this.connection.notify('textDocument/didClose', { textDocument: { uri: oldest } })
          this.documents.delete(oldest)
        }
        this.documents.set(uri, { text: request.text, version })
        try {
          await this.connection.notify(old ? 'textDocument/didChange' : 'textDocument/didOpen', old
            ? { textDocument: { uri, version }, contentChanges: [{ text: request.text }] }
            : { textDocument: { uri, version, languageId: this.languageId, text: request.text } })
        } catch (error) { this.documents.delete(uri); this.closed = true; await this.connection.close(); throw error }
      }
      const result = await this.connection.request('textDocument/completion', {
        textDocument: { uri }, position: { line: request.line, character: request.character }, context: { triggerKind: 1 },
      }, signal)
      return completionItems(result, request.line)
    }).finally(() => { this.pending-- })
    this.queue = operation.then(() => {}, () => {})
    if (!signal) return operation
    return new Promise((resolve, reject) => {
      const abort = () => reject(new DOMException('LSP request cancelled', 'AbortError'))
      signal.addEventListener('abort', abort, { once: true })
      if (signal.aborted) abort()
      operation.then(resolve, reject).finally(() => signal.removeEventListener('abort', abort))
    })
  }

  async close(): Promise<void> { this.closed = true; for (const wake of this.diagnosticWaiters.values()) wake(); this.diagnosticWaiters.clear(); this.documents.clear(); await this.connection.close() }

  private waitForDiagnostics(uri: string, document: Document, milliseconds: number, signal?: AbortSignal): Promise<void> {
    if (!milliseconds || document.diagnostics !== undefined || document.diagnosticError || signal?.aborted || this.closed) return Promise.resolve()
    return new Promise(resolve => {
      const finish = () => {
        clearTimeout(timer)
        signal?.removeEventListener("abort", finish)
        if (this.diagnosticWaiters.get(uri) === finish) this.diagnosticWaiters.delete(uri)
        resolve()
      }
      const timer = setTimeout(finish, milliseconds)
      this.diagnosticWaiters.set(uri, finish)
      signal?.addEventListener("abort", finish, { once: true })
      if (signal?.aborted) finish()
    })
  }

  private notification(method: string, params: unknown): void {
    if (method !== 'textDocument/publishDiagnostics' || !record(params) || typeof params.uri !== 'string') return
    const document = this.documents.get(params.uri)
    if (!document || params.version !== document.version) return
    this.diagnosticWaiters.get(params.uri)?.()
    if (!Array.isArray(params.diagnostics) || params.diagnostics.length > 1000) { document.diagnosticError = 'Invalid or oversized diagnostics response'; delete document.diagnostics; return }
    const diagnostics: unknown[] = []
    for (const item of params.diagnostics) {
      if (!record(item) || typeof item.message !== 'string' || !record(item.range) || !position(item.range.start) || !position(item.range.end)
        || item.range.end.line < item.range.start.line || (item.range.end.line === item.range.start.line && item.range.end.character < item.range.start.character)) {
        document.diagnosticError = 'Malformed diagnostics response'; delete document.diagnostics; return
      }
      diagnostics.push({ message: item.message.slice(0, 8192), range: {
        start: { line: item.range.start.line, character: item.range.start.character },
        end: { line: item.range.end.line, character: item.range.end.character },
      },
        ...(typeof item.severity === 'number' && [1, 2, 3, 4].includes(item.severity) ? { severity: item.severity } : {}),
      })
    }
    document.diagnostics = diagnostics; delete document.diagnosticError
  }
}
/** Normalize a CompletionList / CompletionItem[] into at most LSP_COMPLETION_LIMIT editor items. */
export function completionItems(result: unknown, line: number): { readonly items: readonly LspCompletionItem[]; readonly incomplete: boolean } {
  const raw = Array.isArray(result) ? result : record(result) && Array.isArray(result.items) ? result.items : []
  const incomplete = record(result) && result.isIncomplete === true
  const clip = (value: unknown, limit: number): string | undefined => typeof value === 'string' && value ? value.slice(0, limit) : undefined
  const items: LspCompletionItem[] = []
  for (const item of raw) {
    if (!record(item) || typeof item.label !== 'string' || !item.label) continue
    const edit = record(item.textEdit) ? item.textEdit : undefined
    const range = edit && record(edit.range) ? edit.range : edit && record(edit.replace) ? edit.replace : undefined
    const sameLine = range && position(range.start) && position(range.end) && range.start.line === line && range.end.line === line
    const insertText = (edit && typeof edit.newText === 'string' ? edit.newText : typeof item.insertText === 'string' ? item.insertText : item.label).slice(0, 4096)
    const documentation = typeof item.documentation === 'string' ? item.documentation : record(item.documentation) ? item.documentation.value : undefined
    items.push({
      label: item.label.slice(0, 256), insertText,
      ...(typeof item.kind === 'number' && Number.isSafeInteger(item.kind) ? { kind: item.kind } : {}),
      ...(clip(item.detail, 512) ? { detail: clip(item.detail, 512)! } : {}),
      ...(clip(documentation, 4096) ? { documentation: clip(documentation, 4096)! } : {}),
      ...(sameLine && range && position(range.start) && position(range.end) ? { replace: { start: range.start.character, end: range.end.character } } : {}),
      ...(clip(item.sortText, 256) ? { sortText: clip(item.sortText, 256)! } : {}),
    })
  }
  items.sort((a, b) => (a.sortText ?? a.label).localeCompare(b.sortText ?? b.label))
  return { items: items.slice(0, LSP_COMPLETION_LIMIT), incomplete: incomplete || items.length > LSP_COMPLETION_LIMIT }
}
function position(value: unknown): value is { line: number; character: number } {
  return record(value) && Number.isSafeInteger(value.line) && Number.isSafeInteger(value.character) && Number(value.line) >= 0 && Number(value.character) >= 0
}
async function readDocument(path: string): Promise<string> {
  const file = await open(path, constants.O_RDONLY | constants.O_NOFOLLOW | constants.O_NONBLOCK)
  try {
    if (!(await file.stat()).isFile()) throw new Error('LSP document must be a regular file')
    const buffer = Buffer.alloc(1_048_577)
    let size = 0
    while (size < buffer.length) { const { bytesRead } = await file.read(buffer, size, buffer.length - size, null); if (!bytesRead) break; size += bytesRead }
    if (size > 1_048_576) throw new Error('LSP document exceeds 1 MiB')
    try { return new TextDecoder('utf-8', { fatal: true }).decode(buffer.subarray(0, size)) } catch { throw new Error('LSP document is not valid UTF-8') }
  } finally { await file.close() }
}
