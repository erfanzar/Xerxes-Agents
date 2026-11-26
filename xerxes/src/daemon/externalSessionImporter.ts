// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Strictly read-only discovery and transcript import from external coding agent
 * runtimes: Google Antigravity (`agy` / `antigravity-cli` & `antigravity` brain
 * transcripts), Anthropic Claude Code (`~/.claude/projects/<project>/<id>.jsonl`), and
 * OpenAI Codex (`~/.codex/state_5.sqlite` + `~/.codex/sessions/<date>/<rollout>.jsonl`).
 *
 * Never writes, renames, locks, or modifies any external runtime file or DB.
 */

import { Database } from 'bun:sqlite'
import { closeSync, existsSync, openSync, readdirSync, readFileSync, readSync, statSync, type Dirent } from 'node:fs'
import { homedir } from 'node:os'
import { basename, join, resolve } from 'node:path'

import type { AcpExternalAgentProvider } from '../acp/externalAgents.js'
import type { DaemonSession } from './runtime.js'

const VALID_SESSION_ID_RE = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/
const MAX_SCAN_SLICE_BYTES = 24 * 1024
const MAX_IMPORT_FILE_BYTES = 32 * 1024 * 1024
const MAX_IMPORT_MESSAGES = 4_000
const MAX_PREVIEW_CHARS = 160
const MAX_TITLE_CHARS = 100
const MAX_TOOL_OUTPUT_CHARS = 64 * 1024

export interface ExternalImportableSession {
  readonly id: string
  readonly runtime_id: string
  readonly runtime_name: string
  readonly provider: AcpExternalAgentProvider
  readonly title: string
  readonly cwd: string
  readonly updated_at: string
  readonly preview?: string
  readonly model?: string
  readonly thought_level?: string
  readonly turn_count?: number
  readonly message_count?: number
  readonly source?: string
  readonly already_imported?: boolean
}

export interface ImportedExternalTranscript {
  readonly title: string
  readonly cwd?: string
  readonly model?: string
  readonly thoughtLevel?: string
  readonly messages: DaemonSession['messages']
  readonly toolExecutions: DaemonSession['toolExecutions']
  readonly thinkingContent: string[]
  readonly turnCount: number
}

export interface DiscoverImportableSessionsOptions {
  readonly allProjects?: boolean
  readonly cwd?: string
  readonly env?: Readonly<Record<string, string | undefined>>
  readonly existingExternalKeys?: ReadonlySet<string>
  readonly homeDir?: string
  readonly limit?: number
  readonly query?: string
  readonly runtime?: string
}

export interface LoadExternalTranscriptOptions {
  readonly env?: Readonly<Record<string, string | undefined>>
  readonly externalSessionId: string
  readonly homeDir?: string
  readonly runtimeId: string
  readonly source?: string
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function resolveHomeDir(options: {
  readonly env?: Readonly<Record<string, string | undefined>>
  readonly homeDir?: string
}): string {
  const env = options.env ?? process.env
  return resolve(options.homeDir ?? env.HOME?.trim() ?? homedir())
}

function normalizePath(raw: string): string {
  const trimmed = raw.trim()
  if (!trimmed) return ''
  return resolve(trimmed).replace(/\/+$/, '') || '/'
}

function clipText(raw: string, limit: number): string {
  const compact = raw.replace(/\s+/g, ' ').trim()
  if (compact.length <= limit) return compact
  return `${compact.slice(0, limit - 1)}…`
}

function isSafeSessionId(id: string): boolean {
  const clean = id.trim()
  return Boolean(clean && VALID_SESSION_ID_RE.test(clean) && clean !== '.' && clean !== '..' && !clean.includes('..'))
}

function readHeadAndTail(filePath: string, sliceBytes = MAX_SCAN_SLICE_BYTES): {
  readonly head: string
  readonly size: number
  readonly tail: string
} | null {
  let fd: number | undefined
  try {
    const st = statSync(filePath)
    if (!st.isFile() || st.size === 0) return null
    fd = openSync(filePath, 'r')
    if (st.size <= sliceBytes * 2) {
      const buf = Buffer.allocUnsafe(st.size)
      const bytes = readSync(fd, buf, 0, st.size, 0)
      const text = buf.subarray(0, bytes).toString('utf8')
      return { head: text, tail: text, size: st.size }
    }
    const headBuf = Buffer.allocUnsafe(sliceBytes)
    const headBytes = readSync(fd, headBuf, 0, sliceBytes, 0)
    const tailBuf = Buffer.allocUnsafe(sliceBytes)
    const tailBytes = readSync(fd, tailBuf, 0, sliceBytes, Math.max(0, st.size - sliceBytes))
    return {
      head: headBuf.subarray(0, headBytes).toString('utf8'),
      tail: tailBuf.subarray(0, tailBytes).toString('utf8'),
      size: st.size,
    }
  } catch {
    return null
  } finally {
    if (fd !== undefined) {
      try {
        closeSync(fd)
      } catch {
        // Ignore close errors.
      }
    }
  }
}

function readBoundedFileText(filePath: string, maxBytes = MAX_IMPORT_FILE_BYTES): string | null {
  let fd: number | undefined
  try {
    const st = statSync(filePath)
    if (!st.isFile() || st.size === 0) return null
    if (st.size <= maxBytes) {
      return readFileSync(filePath, 'utf8')
    }
    fd = openSync(filePath, 'r')
    const buf = Buffer.allocUnsafe(maxBytes)
    const bytes = readSync(fd, buf, 0, maxBytes, Math.max(0, st.size - maxBytes))
    const raw = buf.subarray(0, bytes).toString('utf8')
    const firstNewline = raw.indexOf('\n')
    return firstNewline >= 0 ? raw.slice(firstNewline + 1) : raw
  } catch {
    return null
  } finally {
    if (fd !== undefined) {
      try {
        closeSync(fd)
      } catch {
        // Ignore close errors.
      }
    }
  }
}

function extractTagContent(text: string, tag: string): string | undefined {
  const open = `<${tag}>`
  const close = `</${tag}>`
  const start = text.indexOf(open)
  if (start < 0) return undefined
  const end = text.indexOf(close, start + open.length)
  if (end < 0) return text.slice(start + open.length).trim()
  return text.slice(start + open.length, end).trim()
}

/** Strip Antigravity XML wrappers (`<USER_REQUEST>`, `<ADDITIONAL_METADATA>`, `<USER_SETTINGS_CHANGE>`). */
export function cleanAgyUserPromptText(raw: string): string {
  const direct = extractTagContent(raw, 'USER_REQUEST')
  if (direct !== undefined) {
    return direct.trim()
  }
  return raw
    .replace(/<ADDITIONAL_METADATA>[\s\S]*?<\/ADDITIONAL_METADATA>/gi, '')
    .replace(/<USER_SETTINGS_CHANGE>[\s\S]*?<\/USER_SETTINGS_CHANGE>/gi, '')
    .replace(/<CONTEXT_SUMMARY>[\s\S]*?<\/CONTEXT_SUMMARY>/gi, '')
    .trim()
}

function extractAgyTaskHeading(sessionDir: string): string | undefined {
  try {
    const taskPath = join(sessionDir, 'task.md')
    if (!existsSync(taskPath)) return undefined
    const head = readHeadAndTail(taskPath, 2048)?.head ?? ''
    for (const line of head.split('\n')) {
      const m = /^#{1,2}\s+(.+)$/.exec(line.trim())
      if (!m || !m[1]) continue
      const candidate = m[1].trim()
      if (!candidate || /^tasks?$/i.test(candidate) || /^checklist$/i.test(candidate)) continue
      return clipText(candidate, MAX_TITLE_CHARS)
    }
  } catch {
    // Ignore unreadable task.md.
  }
  return undefined
}

function inferAgyThoughtLevel(model: string | undefined): string | undefined {
  if (!model) return undefined
  const lower = model.trim().toLowerCase()
  if (lower.endsWith('-xhigh') || lower.includes('extra-high')) return 'xhigh'
  if (lower.endsWith('-high')) return 'high'
  if (lower.endsWith('-medium')) return 'medium'
  if (lower.endsWith('-low')) return 'low'
  if (lower.endsWith('-max')) return 'max'
  return undefined
}

function extractAgyCwdFromText(text: string): string | undefined {
  const workspaceMatch = /Command Working Directory:\s*(\/[^\r\n]+)/.exec(text)
    ?? /Active Workspaces:\s*\n\s*-\s*(\/[^\r\n]+)/.exec(text)
  return workspaceMatch?.[1]?.trim() || undefined
}

function agyBrainRoots(homeDir: string): ReadonlyArray<{ readonly dir: string; readonly source: string }> {
  return [
    { dir: join(homeDir, '.gemini', 'antigravity-cli', 'brain'), source: 'antigravity-cli' },
    { dir: join(homeDir, '.gemini', 'antigravity', 'brain'), source: 'antigravity' },
  ]
}

function findAgyTranscriptFile(sessionDir: string): string | undefined {
  const logsDir = join(sessionDir, '.system_generated', 'logs')
  const full = join(logsDir, 'transcript_full.jsonl')
  if (existsSync(full)) return full
  const compact = join(logsDir, 'transcript.jsonl')
  if (existsSync(compact)) return compact
  return undefined
}

function discoverAntigravitySessions(homeDir: string): ExternalImportableSession[] {
  const byId = new Map<string, { readonly mtimeMs: number; readonly session: ExternalImportableSession }>()
  for (const { dir: rootDir, source } of agyBrainRoots(homeDir)) {
    let entries: string[]
    try {
      entries = readdirSync(rootDir)
    } catch {
      continue
    }
    for (const entry of entries) {
      if (!isSafeSessionId(entry)) continue
      const sessionDir = join(rootDir, entry)
      const transcriptPath = findAgyTranscriptFile(sessionDir)
      if (!transcriptPath) continue
      let st: ReturnType<typeof statSync>
      try {
        st = statSync(transcriptPath)
      } catch {
        continue
      }
      const prev = byId.get(entry)
      if (prev && prev.mtimeMs >= st.mtimeMs) continue

      const slice = readHeadAndTail(transcriptPath)
      if (!slice) continue

      let firstUserText = ''
      let hasExplicitUser = false
      let detectedCwd = extractAgyCwdFromText(slice.head) ?? ''
      let detectedModel = ''
      let lastPreview = ''
      let headUserCount = 0

      for (const line of slice.head.split('\n')) {
        const trimmed = line.trim()
        if (!trimmed.startsWith('{')) continue
        let parsed: unknown
        try {
          parsed = JSON.parse(trimmed)
        } catch {
          continue
        }
        if (!isRecord(parsed)) continue
        const stepType = typeof parsed.type === 'string' ? parsed.type : ''
        const stepSource = typeof parsed.source === 'string' ? parsed.source : ''
        if (stepType === 'USER_INPUT') {
          if (stepSource === 'USER_EXPLICIT') hasExplicitUser = true
          const content = typeof parsed.content === 'string' ? parsed.content : ''
          if (!detectedCwd) detectedCwd = extractAgyCwdFromText(content) ?? ''
          const cleaned = cleanAgyUserPromptText(content)
          if (cleaned) {
            headUserCount += 1
            if (!firstUserText) firstUserText = cleaned
            lastPreview = cleaned
          }
        } else if (stepType === 'PLANNER_RESPONSE') {
          if (typeof parsed.model === 'string' && parsed.model.trim()) {
            detectedModel = parsed.model.trim()
          }
          if (typeof parsed.content === 'string' && parsed.content.trim()) {
            lastPreview = parsed.content.trim()
          }
        }
      }

      for (const line of slice.tail.split('\n')) {
        const trimmed = line.trim()
        if (!trimmed.startsWith('{')) continue
        let parsed: unknown
        try {
          parsed = JSON.parse(trimmed)
        } catch {
          continue
        }
        if (!isRecord(parsed)) continue
        const stepType = typeof parsed.type === 'string' ? parsed.type : ''
        if (stepType === 'USER_INPUT' && parsed.source === 'USER_EXPLICIT') {
          hasExplicitUser = true
        }
        if (stepType === 'PLANNER_RESPONSE') {
          if (typeof parsed.model === 'string' && parsed.model.trim()) {
            detectedModel = parsed.model.trim()
          }
          if (typeof parsed.content === 'string' && parsed.content.trim()) {
            lastPreview = parsed.content.trim()
          }
        } else if (stepType === 'USER_INPUT' && typeof parsed.content === 'string') {
          const cleaned = cleanAgyUserPromptText(parsed.content)
          if (cleaned) lastPreview = cleaned
        }
      }

      if (!hasExplicitUser && !firstUserText) continue
      const taskHeading = extractAgyTaskHeading(sessionDir)
      const title = taskHeading || (firstUserText ? clipText(firstUserText.split('\n')[0] ?? firstUserText, MAX_TITLE_CHARS) : `Antigravity ${entry.slice(0, 8)}`)
      const thoughtLevel = inferAgyThoughtLevel(detectedModel)
      const session: ExternalImportableSession = {
        id: entry,
        runtime_id: 'antigravity',
        runtime_name: 'Antigravity',
        provider: 'antigravity',
        title,
        cwd: detectedCwd,
        updated_at: new Date(st.mtimeMs).toISOString(),
        ...(lastPreview ? { preview: clipText(lastPreview, MAX_PREVIEW_CHARS) } : {}),
        ...(detectedModel ? { model: detectedModel } : {}),
        ...(thoughtLevel ? { thought_level: thoughtLevel } : {}),
        ...(headUserCount > 0 ? { turn_count: headUserCount } : {}),
        source,
      }
      byId.set(entry, { mtimeMs: st.mtimeMs, session })
    }
  }
  return [...byId.values()]
    .sort((a, b) => b.mtimeMs - a.mtimeMs)
    .map(item => item.session)
}

export function loadAntigravityTranscript(
  externalSessionId: string,
  homeDir: string,
  preferredSource?: string,
): ImportedExternalTranscript | null {
  if (!isSafeSessionId(externalSessionId)) return null
  const roots = agyBrainRoots(homeDir)
  const ordered = preferredSource
    ? [...roots.filter(r => r.source === preferredSource), ...roots.filter(r => r.source !== preferredSource)]
    : roots
  let sessionDir: string | undefined
  let transcriptPath: string | undefined
  for (const { dir } of ordered) {
    const candidateDir = join(dir, externalSessionId)
    const candidateFile = findAgyTranscriptFile(candidateDir)
    if (candidateFile) {
      sessionDir = candidateDir
      transcriptPath = candidateFile
      break
    }
  }
  if (!sessionDir || !transcriptPath) return null

  const raw = readBoundedFileText(transcriptPath)
  if (!raw) return null

  const messages: DaemonSession['messages'] = []
  const toolExecutions: DaemonSession['toolExecutions'] = []
  const thinkingContent: string[] = []
  const pendingTools: Array<{ readonly id: string; readonly name: string; readonly inputs: Record<string, unknown> }> = []
  let detectedModel = ''
  let detectedCwd = ''
  let firstUserText = ''

  for (const line of raw.split('\n')) {
    const trimmed = line.trim()
    if (!trimmed.startsWith('{')) continue
    let parsed: unknown
    try {
      parsed = JSON.parse(trimmed)
    } catch {
      continue
    }
    if (!isRecord(parsed)) continue
    const stepType = typeof parsed.type === 'string' ? parsed.type : ''
    const stepSource = typeof parsed.source === 'string' ? parsed.source : ''
    const stepIndex = typeof parsed.step_index === 'number' ? parsed.step_index : messages.length

    if (stepType === 'USER_INPUT') {
      if (stepSource && stepSource !== 'USER_EXPLICIT' && stepSource !== 'USER') continue
      const contentRaw = typeof parsed.content === 'string' ? parsed.content : ''
      if (!detectedCwd) detectedCwd = extractAgyCwdFromText(contentRaw) ?? ''
      const clean = cleanAgyUserPromptText(contentRaw)
      if (!clean) continue
      pendingTools.length = 0
      if (!firstUserText) firstUserText = clean
      messages.push({ role: 'user', content: clean })
      continue
    }

    if (stepType === 'PLANNER_RESPONSE') {
      if (typeof parsed.model === 'string' && parsed.model.trim()) {
        detectedModel = parsed.model.trim()
      }
      const thinking = typeof parsed.thinking === 'string' ? parsed.thinking.trim() : ''
      const content = typeof parsed.content === 'string' ? parsed.content : ''
      const rawCalls = Array.isArray(parsed.tool_calls) ? parsed.tool_calls : []
      const wireCalls: Array<{
        readonly id: string
        readonly type: 'function'
        readonly function: { readonly name: string; readonly arguments: string }
      }> = []
      for (let idx = 0; idx < rawCalls.length; idx += 1) {
        const call = rawCalls[idx]
        if (!isRecord(call)) continue
        const name = typeof call.name === 'string' && call.name.trim() ? call.name.trim() : 'tool'
        const args = call.args
        let argsStr = '{}'
        let inputs: Record<string, unknown> = {}
        if (typeof args === 'string') {
          argsStr = args
          try {
            const parsedArgs = JSON.parse(args)
            if (isRecord(parsedArgs)) inputs = parsedArgs
          } catch {
            inputs = { raw: args }
          }
        } else if (isRecord(args)) {
          inputs = args
          try {
            argsStr = JSON.stringify(args)
          } catch {
            argsStr = '{}'
          }
        }
        const id = `agy-imp-${stepIndex}-${idx}`
        wireCalls.push({
          id,
          type: 'function',
          function: { name, arguments: argsStr },
        })
        pendingTools.push({ id, name, inputs })
      }
      if (!content && !thinking && wireCalls.length === 0) continue
      messages.push({
        role: 'assistant',
        content,
        ...(thinking ? { thinking } : {}),
        ...(wireCalls.length > 0 ? { tool_calls: wireCalls } : {}),
      })
      thinkingContent.push(thinking)
      continue
    }

    if (stepType === 'GENERIC' && pendingTools.length > 0) {
      const matched = pendingTools.shift()!
      const rawContent = typeof parsed.content === 'string' ? parsed.content : ''
      const bounded = rawContent.length > MAX_TOOL_OUTPUT_CHARS
        ? `${rawContent.slice(0, MAX_TOOL_OUTPUT_CHARS)}\n… (truncated)`
        : rawContent
      messages.push({
        role: 'tool',
        tool_call_id: matched.id,
        name: matched.name,
        content: bounded,
      })
      toolExecutions.push({
        toolCallId: matched.id,
        name: matched.name,
        inputs: matched.inputs,
        result: bounded,
        permitted: true,
        durationMs: 0,
      })
    }
  }

  if (messages.length > MAX_IMPORT_MESSAGES) {
    messages.splice(0, messages.length - MAX_IMPORT_MESSAGES)
  }
  const turnCount = messages.filter(m => m.role === 'user').length
  const title =
    extractAgyTaskHeading(sessionDir)
    || (firstUserText ? clipText(firstUserText.split('\n')[0] ?? firstUserText, MAX_TITLE_CHARS) : `Antigravity ${externalSessionId.slice(0, 8)}`)
  const thoughtLevel = inferAgyThoughtLevel(detectedModel)
  return {
    title,
    ...(detectedCwd ? { cwd: detectedCwd } : {}),
    ...(detectedModel ? { model: detectedModel } : {}),
    ...(thoughtLevel ? { thoughtLevel } : {}),
    messages,
    toolExecutions,
    thinkingContent,
    turnCount,
  }
}

function cleanClaudeUserText(raw: string): string {
  return raw
    .replace(/<system-reminder>[\s\S]*?<\/system-reminder>/gi, '')
    .replace(/<ide_opened_file>[\s\S]*?<\/ide_opened_file>/gi, '')
    .replace(/<ide_selection>[\s\S]*?<\/ide_selection>/gi, '')
    .replace(/<command-name>([\s\S]*?)<\/command-name>/gi, '$1')
    .replace(/<command-message>[\s\S]*?<\/command-message>/gi, '')
    .replace(/<command-args>([\s\S]*?)<\/command-args>/gi, ' $1')
    .replace(/<local-command-stdout>[\s\S]*?<\/local-command-stdout>/gi, '')
    .trim()
}

function discoverClaudeCodeSessions(homeDir: string): ExternalImportableSession[] {
  const projectsRoot = join(homeDir, '.claude', 'projects')
  let projectDirs: string[]
  try {
    projectDirs = readdirSync(projectsRoot)
  } catch {
    return []
  }
  const found: Array<{ readonly mtimeMs: number; readonly session: ExternalImportableSession }> = []
  for (const proj of projectDirs) {
    const projDir = join(projectsRoot, proj)
    let files: string[]
    try {
      files = readdirSync(projDir)
    } catch {
      continue
    }
    for (const file of files) {
      if (!file.endsWith('.jsonl') || file.startsWith('agent-')) continue
      const stem = file.slice(0, -'.jsonl'.length)
      if (!isSafeSessionId(stem)) continue
      const fullPath = join(projDir, file)
      let st: ReturnType<typeof statSync>
      try {
        st = statSync(fullPath)
      } catch {
        continue
      }
      const slice = readHeadAndTail(fullPath)
      if (!slice) continue

      let sessionId = stem
      let cwd = ''
      let customTitle = ''
      let firstUser = ''
      let lastPreview = ''
      let model = ''
      let userTurns = 0

      const inspectChunk = (chunk: string, isHead: boolean): void => {
        for (const line of chunk.split('\n')) {
          const trimmed = line.trim()
          if (!trimmed.startsWith('{')) continue
          let parsed: unknown
          try {
            parsed = JSON.parse(trimmed)
          } catch {
            continue
          }
          if (!isRecord(parsed) || parsed.isSidechain === true) continue
          if (typeof parsed.sessionId === 'string' && parsed.sessionId.trim()) {
            sessionId = parsed.sessionId.trim()
          }
          if (!cwd && typeof parsed.cwd === 'string' && parsed.cwd.trim()) {
            cwd = parsed.cwd.trim()
          }
          if (parsed.type === 'custom-title' && typeof parsed.customTitle === 'string' && parsed.customTitle.trim()) {
            customTitle = parsed.customTitle.trim()
          } else if (parsed.type === 'user' && parsed.isMeta !== true && isRecord(parsed.message)) {
            const content = parsed.message.content
            if (typeof content === 'string') {
              const cleaned = cleanClaudeUserText(content)
              if (cleaned) {
                if (isHead) userTurns += 1
                if (!firstUser) firstUser = cleaned
                lastPreview = cleaned
              }
            }
          } else if (parsed.type === 'assistant' && isRecord(parsed.message)) {
            if (typeof parsed.message.model === 'string' && parsed.message.model.trim()) {
              model = parsed.message.model.trim()
            }
            if (Array.isArray(parsed.message.content)) {
              for (const part of parsed.message.content) {
                if (isRecord(part) && part.type === 'text' && typeof part.text === 'string' && part.text.trim()) {
                  lastPreview = part.text.trim()
                }
              }
            }
          }
        }
      }

      inspectChunk(slice.head, true)
      if (slice.tail !== slice.head) inspectChunk(slice.tail, false)
      if (!firstUser && !customTitle) continue

      const title = clipText(customTitle || firstUser.split('\n')[0] || `Claude Code ${sessionId.slice(0, 8)}`, MAX_TITLE_CHARS)
      found.push({
        mtimeMs: st.mtimeMs,
        session: {
          id: sessionId,
          runtime_id: 'claude-code',
          runtime_name: 'Claude Code',
          provider: 'claude-code',
          title,
          cwd,
          updated_at: new Date(st.mtimeMs).toISOString(),
          ...(lastPreview ? { preview: clipText(lastPreview, MAX_PREVIEW_CHARS) } : {}),
          ...(model ? { model } : {}),
          ...(userTurns > 0 ? { turn_count: userTurns } : {}),
          source: proj,
        },
      })
    }
  }
  return found
    .sort((a, b) => b.mtimeMs - a.mtimeMs)
    .map(item => item.session)
}

export function loadClaudeCodeTranscript(
  externalSessionId: string,
  homeDir: string,
  preferredSource?: string,
): ImportedExternalTranscript | null {
  if (!isSafeSessionId(externalSessionId)) return null
  const projectsRoot = join(homeDir, '.claude', 'projects')
  let filePath: string | undefined
  if (preferredSource && isSafeSessionId(preferredSource)) {
    const direct = join(projectsRoot, preferredSource, `${externalSessionId}.jsonl`)
    if (existsSync(direct)) filePath = direct
  }
  if (!filePath) {
    let projectDirs: string[]
    try {
      projectDirs = readdirSync(projectsRoot)
    } catch {
      return null
    }
    for (const proj of projectDirs) {
      const candidate = join(projectsRoot, proj, `${externalSessionId}.jsonl`)
      if (existsSync(candidate)) {
        filePath = candidate
        break
      }
    }
  }
  if (!filePath) return null

  const raw = readBoundedFileText(filePath)
  if (!raw) return null

  const messages: DaemonSession['messages'] = []
  const toolExecutions: DaemonSession['toolExecutions'] = []
  const toolExecIndexById = new Map<string, number>()
  const thinkingContent: string[] = []
  let customTitle = ''
  let firstUser = ''
  let detectedCwd = ''
  let detectedModel = ''

  // Merge consecutive assistant JSONL chunks sharing the same `message.id`.
  let openAssistantMsgId = ''
  let openAssistantText = ''
  let openAssistantThinking = ''
  let openAssistantCalls: Array<{
    readonly id: string
    readonly type: 'function'
    readonly function: { readonly name: string; readonly arguments: string }
  }> = []

  const flushOpenAssistant = (): void => {
    if (!openAssistantText && !openAssistantThinking && openAssistantCalls.length === 0) {
      openAssistantMsgId = ''
      return
    }
    messages.push({
      role: 'assistant',
      content: openAssistantText,
      ...(openAssistantThinking ? { thinking: openAssistantThinking } : {}),
      ...(openAssistantCalls.length > 0 ? { tool_calls: openAssistantCalls } : {}),
    })
    thinkingContent.push(openAssistantThinking)
    openAssistantMsgId = ''
    openAssistantText = ''
    openAssistantThinking = ''
    openAssistantCalls = []
  }

  for (const line of raw.split('\n')) {
    const trimmed = line.trim()
    if (!trimmed.startsWith('{')) continue
    let parsed: unknown
    try {
      parsed = JSON.parse(trimmed)
    } catch {
      continue
    }
    if (!isRecord(parsed) || parsed.isSidechain === true) continue
    if (!detectedCwd && typeof parsed.cwd === 'string' && parsed.cwd.trim()) {
      detectedCwd = parsed.cwd.trim()
    }
    if (parsed.type === 'custom-title' && typeof parsed.customTitle === 'string' && parsed.customTitle.trim()) {
      customTitle = parsed.customTitle.trim()
      continue
    }
    if (parsed.type === 'assistant' && isRecord(parsed.message)) {
      if (typeof parsed.message.model === 'string' && parsed.message.model.trim()) {
        detectedModel = parsed.message.model.trim()
      }
      const msgId = typeof parsed.message.id === 'string' ? parsed.message.id : ''
      if (openAssistantMsgId && msgId && msgId !== openAssistantMsgId) {
        flushOpenAssistant()
      }
      openAssistantMsgId = msgId || openAssistantMsgId
      const content = parsed.message.content
      if (typeof content === 'string' && content) {
        openAssistantText += (openAssistantText ? '\n' : '') + content
      } else if (Array.isArray(content)) {
        for (const part of content) {
          if (!isRecord(part)) continue
          if (part.type === 'thinking' && typeof part.thinking === 'string' && part.thinking.trim()) {
            openAssistantThinking += (openAssistantThinking ? '\n' : '') + part.thinking.trim()
          } else if (part.type === 'text' && typeof part.text === 'string' && part.text) {
            openAssistantText += (openAssistantText ? '\n' : '') + part.text
          } else if (part.type === 'tool_use') {
            const callId = typeof part.id === 'string' && part.id.trim() ? part.id.trim() : `cc-${messages.length}-${openAssistantCalls.length}`
            const name = typeof part.name === 'string' && part.name.trim() ? part.name.trim() : 'tool'
            const inputs = isRecord(part.input) ? part.input : {}
            let argsStr = '{}'
            try {
              argsStr = JSON.stringify(part.input ?? {})
            } catch {
              argsStr = '{}'
            }
            if (!openAssistantCalls.some(c => c.id === callId)) {
              openAssistantCalls.push({
                id: callId,
                type: 'function',
                function: { name, arguments: argsStr },
              })
            }
            if (!toolExecIndexById.has(callId)) {
              toolExecIndexById.set(callId, toolExecutions.length)
              toolExecutions.push({
                toolCallId: callId,
                name,
                inputs,
                result: '',
                permitted: true,
                durationMs: 0,
              })
            }
          }
        }
      }
      continue
    }

    if (parsed.type === 'user' && parsed.isMeta !== true && isRecord(parsed.message)) {
      flushOpenAssistant()
      const content = parsed.message.content
      if (typeof content === 'string') {
        const cleaned = cleanClaudeUserText(content)
        if (cleaned) {
          if (!firstUser) firstUser = cleaned
          messages.push({ role: 'user', content: cleaned })
        }
      } else if (Array.isArray(content)) {
        const textParts: string[] = []
        for (const part of content) {
          if (!isRecord(part)) continue
          if (part.type === 'tool_result') {
            const callId = typeof part.tool_use_id === 'string' ? part.tool_use_id.trim() : ''
            const rawResult = typeof part.content === 'string'
              ? part.content
              : Array.isArray(part.content)
                ? part.content.map(c => (isRecord(c) && typeof c.text === 'string' ? c.text : '')).filter(Boolean).join('\n')
                : ''
            const bounded = rawResult.length > MAX_TOOL_OUTPUT_CHARS
              ? `${rawResult.slice(0, MAX_TOOL_OUTPUT_CHARS)}\n… (truncated)`
              : rawResult
            const isError = part.is_error === true
            const execIdx = callId ? toolExecIndexById.get(callId) : undefined
            const existingExec = execIdx !== undefined ? toolExecutions[execIdx] : undefined
            const toolName = existingExec && isRecord(existingExec) && typeof existingExec.name === 'string'
              ? existingExec.name
              : 'tool'
            if (existingExec && isRecord(existingExec)) {
              existingExec.result = bounded
              if (isError) existingExec.error = bounded.slice(0, 240) || 'Tool failed'
            }
            messages.push({
              role: 'tool',
              tool_call_id: callId || `cc-res-${messages.length}`,
              name: toolName,
              content: bounded,
              ...(isError ? { is_error: true } : {}),
            })
          } else if (part.type === 'text' && typeof part.text === 'string') {
            const cleaned = cleanClaudeUserText(part.text)
            if (cleaned) textParts.push(cleaned)
          }
        }
        if (textParts.length > 0) {
          const joined = textParts.join('\n')
          if (!firstUser) firstUser = joined
          messages.push({ role: 'user', content: joined })
        }
      }
    }
  }
  flushOpenAssistant()

  if (messages.length > MAX_IMPORT_MESSAGES) {
    messages.splice(0, messages.length - MAX_IMPORT_MESSAGES)
  }
  const turnCount = messages.filter(m => m.role === 'user').length
  const title = clipText(
    customTitle || (firstUser ? firstUser.split('\n')[0] ?? firstUser : '') || `Claude Code ${externalSessionId.slice(0, 8)}`,
    MAX_TITLE_CHARS,
  )
  return {
    title,
    ...(detectedCwd ? { cwd: detectedCwd } : {}),
    ...(detectedModel ? { model: detectedModel } : {}),
    messages,
    toolExecutions,
    thinkingContent,
    turnCount,
  }
}

function codexHomeDir(options: {
  readonly env?: Readonly<Record<string, string | undefined>>
  readonly homeDir?: string
}): string {
  const env = options.env ?? process.env
  const override = env.CODEX_HOME?.trim()
  if (override) return resolve(override)
  return join(resolveHomeDir(options), '.codex')
}

function isCodexBoilerplateUserText(text: string): boolean {
  const trimmed = text.trim()
  if (!trimmed) return true
  return (
    trimmed.startsWith('# AGENTS.md instructions for ')
    || trimmed.startsWith('<environment_context>')
    || trimmed.startsWith('<permissions instructions>')
    || trimmed.startsWith('<INSTRUCTIONS>')
    || trimmed.startsWith('<collaboration_mode>')
    || trimmed.startsWith('<turn_aborted>')
  )
}

function cleanCodexTitle(raw: string): string {
  const myRequestIdx = raw.indexOf('## My request:')
  const base = myRequestIdx >= 0 ? raw.slice(myRequestIdx + '## My request:'.length).trim() : raw.trim()
  return clipText(base.split('\n')[0] ?? base, MAX_TITLE_CHARS)
}

function collectCodexRolloutFiles(sessionsRoot: string, out: string[], depth = 0): void {
  if (depth > 4) return
  let entries: readonly Dirent[]
  try {
    entries = readdirSync(sessionsRoot, { withFileTypes: true })
  } catch {
    return
  }
  for (const entry of entries) {
    const full = join(sessionsRoot, String(entry.name))
    if (entry.isDirectory()) {
      collectCodexRolloutFiles(full, out, depth + 1)
    } else if (entry.isFile() && String(entry.name).endsWith('.jsonl')) {
      out.push(full)
    }
  }
}

function extractCodexIdFromFilename(filePath: string): string | undefined {
  const base = basename(filePath, '.jsonl')
  const uuidMatch = /([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$/i.exec(base)
  if (uuidMatch?.[1]) return uuidMatch[1]
  return isSafeSessionId(base) ? base : undefined
}

function discoverCodexSessions(options: {
  readonly env?: Readonly<Record<string, string | undefined>>
  readonly homeDir?: string
}): ExternalImportableSession[] {
  const codexHome = codexHomeDir(options)
  const byId = new Map<string, { readonly updatedMs: number; readonly session: ExternalImportableSession }>()

  // 1. Primary: strictly read-only query of Codex's state_5.sqlite threads index.
  const sqlitePath = join(codexHome, 'state_5.sqlite')
  if (existsSync(sqlitePath)) {
    let db: Database | undefined
    try {
      db = new Database(sqlitePath, { readonly: true })
      const rows = db
        .query(
          'SELECT id, title, cwd, updated_at, rollout_path, model, reasoning_effort, first_user_message, preview FROM threads WHERE COALESCE(archived, 0) = 0 ORDER BY updated_at DESC LIMIT 200',
        )
        .all() as Array<Record<string, unknown>>
      for (const row of rows) {
        const id = typeof row.id === 'string' ? row.id.trim() : ''
        if (!id || !isSafeSessionId(id)) continue
        const rolloutPath = typeof row.rollout_path === 'string' ? row.rollout_path.trim() : ''
        if (rolloutPath && !existsSync(rolloutPath)) continue
        const rawUpdated = typeof row.updated_at === 'number' ? row.updated_at : 0
        const updatedMs = rawUpdated > 0 && rawUpdated < 10_000_000_000 ? rawUpdated * 1000 : rawUpdated || Date.now()
        const rawTitle =
          (typeof row.title === 'string' && row.title.trim()) ||
          (typeof row.first_user_message === 'string' && row.first_user_message.trim()) ||
          `Codex ${id.slice(0, 8)}`
        const title = cleanCodexTitle(rawTitle)
        const cwd = typeof row.cwd === 'string' ? row.cwd.trim() : ''
        const model = typeof row.model === 'string' && row.model.trim() ? row.model.trim() : undefined
        const thoughtLevel =
          typeof row.reasoning_effort === 'string' && row.reasoning_effort.trim()
            ? row.reasoning_effort.trim()
            : undefined
        const preview =
          (typeof row.preview === 'string' && row.preview.trim()) ||
          (typeof row.first_user_message === 'string' && row.first_user_message.trim()) ||
          undefined
        byId.set(id, {
          updatedMs,
          session: {
            id,
            runtime_id: 'codex',
            runtime_name: 'Codex',
            provider: 'codex',
            title,
            cwd,
            updated_at: new Date(updatedMs).toISOString(),
            ...(preview ? { preview: clipText(cleanCodexTitle(preview), MAX_PREVIEW_CHARS) } : {}),
            ...(model ? { model } : {}),
            ...(thoughtLevel ? { thought_level: thoughtLevel } : {}),
            ...(rolloutPath ? { source: rolloutPath } : {}),
          },
        })
      }
    } catch {
      // Fall through to JSONL scan when SQLite schema differs or DB is unavailable.
    } finally {
      try {
        db?.close()
      } catch {
        // Ignore close errors.
      }
    }
  }

  // 2. Supplement / fallback: scan `$CODEX_HOME/sessions/**/*.jsonl`.
  const files: string[] = []
  collectCodexRolloutFiles(join(codexHome, 'sessions'), files)
  for (const filePath of files) {
    const inferredId = extractCodexIdFromFilename(filePath)
    if (inferredId && byId.has(inferredId)) continue
    let st: ReturnType<typeof statSync>
    try {
      st = statSync(filePath)
    } catch {
      continue
    }
    const slice = readHeadAndTail(filePath)
    if (!slice) continue
    let sessionId = inferredId ?? ''
    let cwd = ''
    let firstUser = ''
    let model = ''
    let effort = ''
    for (const line of slice.head.split('\n')) {
      const trimmed = line.trim()
      if (!trimmed.startsWith('{')) continue
      let parsed: unknown
      try {
        parsed = JSON.parse(trimmed)
      } catch {
        continue
      }
      if (!isRecord(parsed)) continue
      const payload = isRecord(parsed.payload) ? parsed.payload : undefined
      if (parsed.type === 'session_meta' && payload) {
        if (typeof payload.session_id === 'string' && payload.session_id.trim()) sessionId = payload.session_id.trim()
        else if (typeof payload.id === 'string' && payload.id.trim()) sessionId = payload.id.trim()
        if (typeof payload.cwd === 'string' && payload.cwd.trim()) cwd = payload.cwd.trim()
      } else if (parsed.type === 'turn_context' && payload) {
        if (typeof payload.model === 'string' && payload.model.trim()) model = payload.model.trim()
        if (typeof payload.effort === 'string' && payload.effort.trim()) effort = payload.effort.trim()
        if (!cwd && typeof payload.cwd === 'string' && payload.cwd.trim()) cwd = payload.cwd.trim()
      } else if (parsed.type === 'response_item' && payload && payload.type === 'message' && payload.role === 'user') {
        const text = extractCodexMessageParts(payload.content)
        if (text && !isCodexBoilerplateUserText(text) && !firstUser) {
          firstUser = text
        }
      }
    }
    if (!sessionId || !isSafeSessionId(sessionId) || byId.has(sessionId) || !firstUser) continue
    byId.set(sessionId, {
      updatedMs: st.mtimeMs,
      session: {
        id: sessionId,
        runtime_id: 'codex',
        runtime_name: 'Codex',
        provider: 'codex',
        title: cleanCodexTitle(firstUser),
        cwd,
        updated_at: new Date(st.mtimeMs).toISOString(),
        preview: clipText(firstUser, MAX_PREVIEW_CHARS),
        ...(model ? { model } : {}),
        ...(effort ? { thought_level: effort } : {}),
        source: filePath,
      },
    })
  }

  return [...byId.values()]
    .sort((a, b) => b.updatedMs - a.updatedMs)
    .map(item => item.session)
}

function extractCodexMessageParts(content: unknown): string {
  if (typeof content === 'string') return content.trim()
  if (!Array.isArray(content)) return ''
  const parts: string[] = []
  for (const item of content) {
    if (!isRecord(item)) continue
    if (typeof item.text === 'string' && item.text.trim()) {
      parts.push(item.text.trim())
    }
  }
  return parts.join('\n').trim()
}

export function loadCodexTranscript(
  externalSessionId: string,
  options: {
    readonly env?: Readonly<Record<string, string | undefined>>
    readonly homeDir?: string
    readonly preferredSource?: string
  } = {},
): ImportedExternalTranscript | null {
  if (!isSafeSessionId(externalSessionId)) return null
  const codexHome = codexHomeDir(options)
  let rolloutPath: string | undefined

  // Only accept `preferredSource` if it lives strictly under `$CODEX_HOME`.
  if (options.preferredSource) {
    const resolvedSrc = resolve(options.preferredSource)
    if (resolvedSrc.startsWith(`${codexHome}/`) && existsSync(resolvedSrc)) {
      rolloutPath = resolvedSrc
    }
  }
  if (!rolloutPath) {
    const sqlitePath = join(codexHome, 'state_5.sqlite')
    if (existsSync(sqlitePath)) {
      let db: Database | undefined
      try {
        db = new Database(sqlitePath, { readonly: true })
        const row = db
          .query('SELECT rollout_path FROM threads WHERE id = ? LIMIT 1')
          .get(externalSessionId) as Record<string, unknown> | null
        if (row && typeof row.rollout_path === 'string' && row.rollout_path.trim()) {
          const candidate = resolve(row.rollout_path.trim())
          if (candidate.startsWith(`${codexHome}/`) && existsSync(candidate)) {
            rolloutPath = candidate
          }
        }
      } catch {
        // Fall through to directory scan.
      } finally {
        try {
          db?.close()
        } catch {
          // Ignore close errors.
        }
      }
    }
  }
  if (!rolloutPath) {
    const files: string[] = []
    collectCodexRolloutFiles(join(codexHome, 'sessions'), files)
    rolloutPath = files.find(file => basename(file).includes(externalSessionId))
  }
  if (!rolloutPath) return null

  const raw = readBoundedFileText(rolloutPath)
  if (!raw) return null

  const messages: DaemonSession['messages'] = []
  const toolExecutions: DaemonSession['toolExecutions'] = []
  const toolExecIndexById = new Map<string, number>()
  const thinkingContent: string[] = []
  let detectedCwd = ''
  let detectedModel = ''
  let detectedEffort = ''
  let firstUser = ''
  let pendingThinking = ''

  for (const line of raw.split('\n')) {
    const trimmed = line.trim()
    if (!trimmed.startsWith('{')) continue
    let parsed: unknown
    try {
      parsed = JSON.parse(trimmed)
    } catch {
      continue
    }
    if (!isRecord(parsed)) continue
    const payload = isRecord(parsed.payload) ? parsed.payload : undefined
    if (!payload) continue

    if (parsed.type === 'session_meta') {
      if (!detectedCwd && typeof payload.cwd === 'string' && payload.cwd.trim()) {
        detectedCwd = payload.cwd.trim()
      }
      continue
    }
    if (parsed.type === 'turn_context') {
      if (typeof payload.model === 'string' && payload.model.trim()) detectedModel = payload.model.trim()
      if (typeof payload.effort === 'string' && payload.effort.trim()) detectedEffort = payload.effort.trim()
      if (!detectedCwd && typeof payload.cwd === 'string' && payload.cwd.trim()) detectedCwd = payload.cwd.trim()
      continue
    }
    if (parsed.type !== 'response_item') continue

    const itemType = typeof payload.type === 'string' ? payload.type : ''
    if (itemType === 'reasoning') {
      const summary = Array.isArray(payload.summary)
        ? payload.summary
            .map(s => (typeof s === 'string' ? s : isRecord(s) && typeof s.text === 'string' ? s.text : ''))
            .filter(Boolean)
            .join('\n')
        : ''
      if (summary) pendingThinking += (pendingThinking ? '\n' : '') + summary
      continue
    }
    if (itemType === 'message') {
      const role = typeof payload.role === 'string' ? payload.role : ''
      const text = extractCodexMessageParts(payload.content)
      if (!text) continue
      if (role === 'user') {
        if (isCodexBoilerplateUserText(text)) continue
        pendingThinking = ''
        if (!firstUser) firstUser = text
        messages.push({ role: 'user', content: text })
      } else if (role === 'assistant') {
        messages.push({
          role: 'assistant',
          content: text,
          ...(pendingThinking ? { thinking: pendingThinking } : {}),
        })
        thinkingContent.push(pendingThinking)
        pendingThinking = ''
      }
      continue
    }
    if (itemType === 'function_call' || itemType === 'custom_tool_call') {
      const callId = typeof payload.call_id === 'string' && payload.call_id.trim()
        ? payload.call_id.trim()
        : `codex-${messages.length}`
      const name = typeof payload.name === 'string' && payload.name.trim() ? payload.name.trim() : 'tool'
      const rawArgs = payload.arguments ?? payload.input ?? '{}'
      const argsStr = typeof rawArgs === 'string' ? rawArgs : JSON.stringify(rawArgs)
      let inputs: Record<string, unknown> = {}
      try {
        const parsedArgs = typeof rawArgs === 'string' ? JSON.parse(rawArgs) : rawArgs
        if (isRecord(parsedArgs)) inputs = parsedArgs
      } catch {
        inputs = { raw: argsStr }
      }
      messages.push({
        role: 'assistant',
        content: '',
        ...(pendingThinking ? { thinking: pendingThinking } : {}),
        tool_calls: [{ id: callId, type: 'function', function: { name, arguments: argsStr } }],
      })
      thinkingContent.push(pendingThinking)
      pendingThinking = ''
      toolExecIndexById.set(callId, toolExecutions.length)
      toolExecutions.push({
        toolCallId: callId,
        name,
        inputs,
        result: '',
        permitted: true,
        durationMs: 0,
      })
      continue
    }
    if (itemType === 'function_call_output' || itemType === 'custom_tool_call_output') {
      const callId = typeof payload.call_id === 'string' && payload.call_id.trim()
        ? payload.call_id.trim()
        : `codex-out-${messages.length}`
      const rawOut = typeof payload.output === 'string' ? payload.output : JSON.stringify(payload.output ?? '')
      const bounded = rawOut.length > MAX_TOOL_OUTPUT_CHARS
        ? `${rawOut.slice(0, MAX_TOOL_OUTPUT_CHARS)}\n… (truncated)`
        : rawOut
      const execIdx = toolExecIndexById.get(callId)
      const existingExec = execIdx !== undefined ? toolExecutions[execIdx] : undefined
      const toolName = existingExec && isRecord(existingExec) && typeof existingExec.name === 'string'
        ? existingExec.name
        : 'tool'
      if (existingExec && isRecord(existingExec)) {
        existingExec.result = bounded
      }
      messages.push({
        role: 'tool',
        tool_call_id: callId,
        name: toolName,
        content: bounded,
      })
    }
  }

  if (messages.length > MAX_IMPORT_MESSAGES) {
    messages.splice(0, messages.length - MAX_IMPORT_MESSAGES)
  }
  const turnCount = messages.filter(m => m.role === 'user').length
  return {
    title: firstUser ? cleanCodexTitle(firstUser) : `Codex ${externalSessionId.slice(0, 8)}`,
    ...(detectedCwd ? { cwd: detectedCwd } : {}),
    ...(detectedModel ? { model: detectedModel } : {}),
    ...(detectedEffort ? { thoughtLevel: detectedEffort } : {}),
    messages,
    toolExecutions,
    thinkingContent,
    turnCount,
  }
}

/**
 * Discover importable sessions across Antigravity, Claude Code, and Codex
 * (strictly read-only), optionally filtered by runtime, workspace cwd, or search query.
 */
export function discoverImportableExternalSessions(
  options: DiscoverImportableSessionsOptions = {},
): readonly ExternalImportableSession[] {
  const homeDir = resolveHomeDir(options)
  const runtimeFilter = (options.runtime ?? 'all').trim().toLowerCase()
  const includeAgy = runtimeFilter === '' || runtimeFilter === 'all' || runtimeFilter === 'antigravity'
  const includeClaude = runtimeFilter === '' || runtimeFilter === 'all' || runtimeFilter === 'claude-code'
  const includeCodex = runtimeFilter === '' || runtimeFilter === 'all' || runtimeFilter === 'codex'

  const all: ExternalImportableSession[] = []
  if (includeAgy) all.push(...discoverAntigravitySessions(homeDir))
  if (includeClaude) all.push(...discoverClaudeCodeSessions(homeDir))
  if (includeCodex) all.push(...discoverCodexSessions(options))

  const targetCwd = options.cwd ? normalizePath(options.cwd) : ''
  const needle = (options.query ?? '').trim().toLowerCase()
  const existingKeys = options.existingExternalKeys

  let filtered = all.map(item => {
    const key = `${item.runtime_id}:${item.id}`
    return existingKeys?.has(key) ? { ...item, already_imported: true } : item
  })

  if (targetCwd && options.allProjects !== true) {
    // Keep sessions matching targetCwd OR sessions whose cwd wasn't recorded on disk (e.g. AGY transcripts).
    const matching = filtered.filter(item => !item.cwd || normalizePath(item.cwd) === targetCwd)
    if (matching.length > 0) {
      filtered = matching
    }
  }

  if (needle) {
    filtered = filtered.filter(
      item =>
        item.title.toLowerCase().includes(needle)
        || item.id.toLowerCase().includes(needle)
        || item.cwd.toLowerCase().includes(needle)
        || item.runtime_name.toLowerCase().includes(needle)
        || (item.preview?.toLowerCase().includes(needle) ?? false)
        || (item.model?.toLowerCase().includes(needle) ?? false),
    )
  }

  filtered.sort((a, b) => Date.parse(b.updated_at || '0') - Date.parse(a.updated_at || '0'))
  const limit = Math.max(1, Math.min(200, options.limit ?? 80))
  return filtered.slice(0, limit)
}

/** Load the full chronological transcript of an external runtime session (strictly read-only). */
export function loadExternalSessionTranscript(
  options: LoadExternalTranscriptOptions,
): ImportedExternalTranscript | null {
  const homeDir = resolveHomeDir(options)
  const runtimeId = options.runtimeId.trim().toLowerCase()
  if (runtimeId === 'antigravity') {
    return loadAntigravityTranscript(options.externalSessionId, homeDir, options.source)
  }
  if (runtimeId === 'claude-code') {
    return loadClaudeCodeTranscript(options.externalSessionId, homeDir, options.source)
  }
  if (runtimeId === 'codex') {
    return loadCodexTranscript(options.externalSessionId, {
      ...(options.env ? { env: options.env } : {}),
      homeDir,
      ...(options.source ? { preferredSource: options.source } : {}),
    })
  }
  return null
}
