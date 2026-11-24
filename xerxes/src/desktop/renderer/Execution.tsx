// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { memo, useMemo, useState, type ReactElement } from 'react'
import type { ToolContentBlock, ToolItem } from './types.js'
import { useDesktopNavigation } from './DesktopPanels.js'
import { OutputViewer, readableOutput } from './OutputViewer.js'
import { Icon, type IconName } from './Icon.js'
import { StructuredResult, structuredOutput } from './StructuredResult.js'
import { CopyButton } from './CopyButton.js'
import { FileToolRow, fileToolOf, fileToolStats } from './FileToolRow.js'
import { ToolImages } from './ImageView.js'
import { chatImageTypeOf } from '../chatMediaShared.js'
import { ArtifactCard, artifactOfTool } from './ArtifactCard.js'
import { SwarmToolCard, isSwarmToolName, swarmPreviewOf } from './SwarmToolCard.js'
import { parseArgs } from './blocks.js'
import { commandGist } from './commandGist.js'

function record(text: string | undefined): Record<string, unknown> | null {
  if (typeof text !== 'string' || !text) return null
  try {
    const value: unknown = JSON.parse(text)
    return value !== null && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : null
  } catch { return null }
}

const AGY_COMMAND_ENVELOPE_RE =
  /^(?:Created At:[^\n]*\r?\n)?(?:Completed At:[^\n]*\r?\n)?Command exited with code (-?\d+)\.?(?:\r?\n+(?:Command output:\r?\n?)?|$)/i

/**
 * Strip Antigravity ACP's verbose command output envelope
 * (`Created At: ...\nCompleted At: ...\nCommand exited with code N.\n\nCommand output:\n`)
 * so terminal output renders cleanly alongside an accurate exit code.
 */
export function stripAgyCommandEnvelope(raw: string): {
  readonly stripped: boolean
  readonly text: string
  readonly exitCode: number | null
} {
  const match = AGY_COMMAND_ENVELOPE_RE.exec(raw)
  if (!match) return { stripped: false, text: raw, exitCode: null }
  const code = Number(match[1])
  return {
    stripped: true,
    text: raw.slice(match[0].length),
    exitCode: Number.isFinite(code) ? code : null,
  }
}

const failureByItem = new WeakMap<ToolItem, boolean>()
/** Share failure status with collapsed group headers without reparsing unchanged results. */
export function toolHasFailed(item: ToolItem): boolean {
  if (item.state === 'failed') return true
  const cached = failureByItem.get(item)
  if (cached !== undefined) return cached
  const parsed = record(item.output)
  let exitCode: number | null =
    typeof parsed?.exitCode === 'number'
      ? parsed.exitCode
      : typeof parsed?.exit_code === 'number'
        ? parsed.exit_code
        : null
  if (exitCode === null) {
    const rawText = typeof parsed?.output === 'string' ? parsed.output : item.output
    if (rawText) exitCode = stripAgyCommandEnvelope(rawText).exitCode
  }
  const failed = exitCode !== null && exitCode !== 0
  failureByItem.set(item, failed)
  return failed
}

function shellArgument(value: string): string {
  return /^[a-zA-Z0-9_./:=,+-]+$/.test(value) ? value : `'${value.replaceAll("'", "'\\''")}'`
}

export function executionView(item: ToolItem) {
  const parsedInput = parseArgs(item.input)
  const input = Object.keys(parsedInput).length > 0 ? parsedInput : null
  const output = record(item.output)
  const commandParts = Array.isArray(output?.command)
    ? output.command
    : input && typeof input.cmd === 'string' && Array.isArray(input.args)
      ? [input.cmd, ...input.args]
      : null
  const command = commandParts?.every((part): part is string => typeof part === 'string')
    ? commandParts.map(shellArgument).join(' ')
    : typeof input?.cmd === 'string'
      ? input.cmd
      : typeof input?.command === 'string'
        ? input.command
        : typeof input?.CommandLine === 'string'
          ? input.CommandLine
          : typeof input?.script === 'string'
            ? input.script
            : null
  const summary = typeof input?.toolSummary === 'string' && input.toolSummary.trim()
    ? input.toolSummary.trim()
    : null
  const rawExitCode =
    typeof output?.exitCode === 'number'
      ? output.exitCode
      : typeof output?.exit_code === 'number'
        ? output.exit_code
        : null
  let stdout: string | null = null
  let envelopeExitCode: number | null = null
  if (typeof output?.stdout === 'string') {
    const unwrapped = stripAgyCommandEnvelope(output.stdout)
    stdout = unwrapped.stripped ? unwrapped.text : output.stdout
    envelopeExitCode = unwrapped.exitCode
  } else if (typeof output?.output === 'string') {
    const unwrapped = stripAgyCommandEnvelope(output.output)
    if (command !== null || rawExitCode !== null || unwrapped.stripped) {
      stdout = unwrapped.stripped ? unwrapped.text : output.output
      envelopeExitCode = unwrapped.exitCode
    }
  } else if (!output && typeof item.output === 'string' && item.output) {
    const unwrapped = stripAgyCommandEnvelope(item.output)
    if (unwrapped.stripped) {
      stdout = unwrapped.text
      envelopeExitCode = unwrapped.exitCode
    }
  }
  return {
    command,
    summary,
    stdout,
    stderr: typeof output?.stderr === 'string' ? output.stderr : null,
    exitCode: rawExitCode ?? envelopeExitCode,
    cwd: typeof output?.cwd === 'string' ? output.cwd : null,
  }
}

export type SemanticToolCategory =
  | 'swarm'
  | 'shell'
  | 'file'
  | 'search'
  | 'web'
  | 'skill'
  | 'mcp'
  | 'memory'
  | 'schedule'
  | 'monitor'
  | 'agent'
  | 'plan'
  | 'other'

export interface SemanticToolSummary {
  readonly category: SemanticToolCategory
  readonly icon: IconName
  readonly badge: string
  readonly target: string
  readonly scope?: string
  readonly metaPill?: string
  readonly metaTone?: 'neutral' | 'ok' | 'failed' | 'working' | 'accent'
  readonly command?: string
  readonly exitCode?: number | null
}

function normalizedTail(rawName: string | undefined): string {
  const safe = typeof rawName === 'string' ? rawName : ''
  return (safe.split(/[.:/]/).pop() ?? safe)
    .trim()
    .replace(/Tool$/i, '')
    .replace(/([a-z0-9])([A-Z])/g, '$1_$2')
    .replace(/[\s-]+/g, '_')
    .toLowerCase()
}

function strVal(value: unknown): string {
  return typeof value === 'string' ? value.trim() : ''
}

function clipText(text: string, max = 88): string {
  const flat = text.replace(/\s+/g, ' ').trim()
  return flat.length > max ? `${flat.slice(0, max - 1)}…` : flat
}

function looksLikeRawJson(value: string): boolean {
  const trimmed = value.trim()
  return (trimmed.startsWith('{') && trimmed.endsWith('}'))
    || (trimmed.startsWith('[') && trimmed.endsWith(']'))
    || trimmed.startsWith('{"')
}

function safeCleanArg(raw: string | undefined): string {
  if (typeof raw !== 'string') return ''
  const trimmed = raw.trim()
  if (!trimmed || trimmed === '{}' || trimmed === '[]' || looksLikeRawJson(trimmed)) return ''
  return clipText(trimmed, 88)
}

function basenameOf(rawPath: string): string {
  const clean = rawPath.replace(/[\\/]+$/, '')
  return clean.split(/[\\/]/).pop() || clean
}

function formatUrlTarget(rawUrl: string): string {
  const trimmed = rawUrl.trim()
  if (!trimmed) return ''
  try {
    const parsed = new URL(trimmed)
    const path = parsed.pathname && parsed.pathname !== '/' ? parsed.pathname : ''
    const combined = `${parsed.hostname}${path}`
    return clipText(combined, 64)
  } catch {
    return clipText(trimmed, 64)
  }
}

function formatHumanLabel(raw: string): string {
  const spaced = raw
    .replace(/Tool$/i, '')
    .replace(/([a-z0-9])([A-Z])/g, '$1 $2')
    .replace(/[_.-]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
  return spaced ? spaced[0]!.toUpperCase() + spaced.slice(1) : 'Tool'
}

function parseMcpIdentifier(rawName: string): { readonly server: string; readonly tool: string } | null {
  const trimmed = rawName.trim()
  const doubleUnder = /^mcp__([^_]+(?:_[^_]+)*)__(.+)$/i.exec(trimmed)
  if (doubleUnder) {
    return { server: doubleUnder[1]!, tool: formatHumanLabel(doubleUnder[2]!) }
  }
  const dotted = /^mcp[.:/]([^.:/]+)[.:/](.+)$/i.exec(trimmed)
  if (dotted) {
    return { server: dotted[1]!, tool: formatHumanLabel(dotted[2]!) }
  }
  const singleMcp = /^mcp[.:/_-](.+)$/i.exec(trimmed)
  if (singleMcp) {
    return { server: 'mcp', tool: formatHumanLabel(singleMcp[1]!) }
  }
  return null
}

function countOutputMatches(output: string | undefined, unit: 'match' | 'file' | 'item' | 'result'): string | undefined {
  if (typeof output !== 'string') return undefined
  const trimmed = output.trim()
  if (!trimmed) return undefined
  const plural = (n: number): string => {
    if (n === 0) return unit === 'match' ? '0 matches' : `0 ${unit}s`
    if (n === 1) return `1 ${unit}`
    return unit === 'match' ? `${n} matches` : `${n} ${unit}s`
  }
  if (trimmed.startsWith('{') || trimmed.startsWith('[')) {
    try {
      const parsed: unknown = JSON.parse(trimmed)
      if (Array.isArray(parsed)) return plural(parsed.length)
      if (parsed && typeof parsed === 'object') {
        const rec = parsed as Record<string, unknown>
        for (const key of ['matches', 'files', 'results', 'entries', 'items', 'symbols', 'locations']) {
          if (Array.isArray(rec[key])) return plural((rec[key] as unknown[]).length)
        }
        for (const key of ['total', 'count', 'matchCount', 'totalMatches', 'total_matches']) {
          if (typeof rec[key] === 'number' && Number.isFinite(rec[key])) return plural(rec[key] as number)
        }
      }
    } catch {
      // Fall through to plain-text line counting.
    }
  }
  if (/^(no matches|0 matches|no files|nothing found|empty)/i.test(trimmed)) return plural(0)
  const explicitCount = /\b(\d+)\s+(match(?:es)?|files?|results?|entries|items?)\b/i.exec(trimmed)
  if (explicitCount && trimmed.split('\n').length <= 2) {
    return plural(Number(explicitCount[1]))
  }
  const lines = trimmed
    .split('\n')
    .map(l => l.trim())
    .filter(l => l.length > 0 && !/^(?:\.\.\.|\[(?:Grep|Glob|ListDir|Search)|Showing |Truncated )/i.test(l))
  return plural(lines.length)
}

function firstReadableArg(args: Record<string, unknown>, fallbackArg: string): string {
  const preferredKeys = [
    'title', 'toolSummary', 'name', 'skill', 'file_path', 'path', 'Path',
    'query', 'Query', 'pattern', 'Pattern', 'url', 'command', 'cmd', 'CommandLine',
    'target', 'id', 'key', 'topic', 'action', 'message', 'text',
  ]
  for (const key of preferredKeys) {
    const v = args[key]
    if (typeof v === 'string' && v.trim() && !looksLikeRawJson(v)) return clipText(v, 76)
    if (typeof v === 'number' || typeof v === 'boolean') return String(v)
  }
  for (const [, v] of Object.entries(args)) {
    if (typeof v === 'string' && v.trim() && !looksLikeRawJson(v)) return clipText(v, 76)
    if (typeof v === 'number' || typeof v === 'boolean') return String(v)
  }
  return safeCleanArg(fallbackArg)
}

/**
 * Classify any ToolItem into a clean semantic category with icon, action badge,
 * human-readable target (never raw JSON), optional scope, and result/exit pill.
 */
export function classifyToolItem(item: ToolItem, fallbackLabel?: string): SemanticToolSummary {
  const rawName = item.name || item.verb || ''
  const tail = normalizedTail(rawName)
  const args = parseArgs(item.input)
  const failed = toolHasFailed(item)

  // 1. Swarm
  if (isSwarmToolName(rawName)) {
    const preview = swarmPreviewOf(item)
    return {
      category: 'swarm',
      icon: 'tree',
      badge: preview.verb,
      target: preview.target,
      ...(failed ? { metaPill: 'failed', metaTone: 'failed' as const } : {}),
    }
  }

  // 2. File tools (used by live preview strips & ToolCallsSection)
  const fileTool = fileToolOf(item)
  if (fileTool) {
    const verb = fileTool.kind === 'read'
      ? 'Read'
      : fileTool.kind === 'edit'
        ? (fileTool.wholeFile ? 'Rewrite' : 'Edit')
        : fileTool.kind === 'write'
          ? (fileTool.append ? 'Append' : 'Write')
          : fileTool.kind === 'patch'
            ? 'Patch'
            : 'Outline'
    const stats = !failed && item.state !== 'working' ? fileToolStats(item) : null
    return {
      category: 'file',
      icon: fileTool.kind === 'read' ? 'file' : fileTool.kind === 'outline' ? 'listBullets' : 'changes',
      badge: verb,
      target: basenameOf(fileTool.path),
      ...(stats ? { metaPill: `+${stats.adds}${stats.dels > 0 ? ` −${stats.dels}` : ''}`, metaTone: 'ok' as const } : {}),
    }
  }

  // 3. Shell / Command
  const view = executionView(item)
  const isShell = /^(exec_?command|check_?command|kill_?command|list_?commands|bash|shell|run_?command|terminal|pty_\w+|read_?terminal_?output)$/.test(tail)
    || Boolean(view.command)
  if (isShell) {
    const cmd = view.command || strVal(args.command) || strVal(args.cmd) || strVal(args.CommandLine) || strVal(args.script) || safeCleanArg(item.arg)
    const exitMeta = view.exitCode !== null
      ? {
          metaPill: `exit ${view.exitCode}`,
          metaTone: (view.exitCode === 0 ? 'ok' : 'failed') as 'ok' | 'failed',
        }
      : {}
    return {
      category: 'shell',
      icon: 'terminal',
      badge: view.summary || fallbackLabel || 'Shell',
      target: cmd || (tail === 'list_commands' ? 'active commands' : 'command'),
      ...(view.cwd ? { scope: basenameOf(view.cwd) } : {}),
      ...exitMeta,
      ...(cmd ? { command: cmd } : {}),
      exitCode: view.exitCode,
    }
  }

  // 4. MCP tools (check before generic search/web so `mcp.github:search_issues` keeps MCP badge!)
  const mcp = parseMcpIdentifier(rawName)
  if (mcp || tail === 'mcp') {
    const server = mcp?.server || strVal(args.server) || 'server'
    const toolLabel = mcp?.tool || strVal(args.tool) || 'call'
    const argTarget = firstReadableArg(args, item.arg)
    const target = argTarget ? `${toolLabel} · ${argTarget}` : toolLabel
    return {
      category: 'mcp',
      icon: 'code',
      badge: 'MCP',
      target,
      scope: server,
    }
  }

  // 5. Search / Discovery (Grep, Ripgrep, Glob, Find, ListDir, LSP, History)
  if (/^(grep|grep_search|ripgrep|rg|search|search_history|find_by_name|tool_search)$/.test(tail)) {
    const pattern = strVal(args.pattern) || strVal(args.Pattern) || strVal(args.query) || strVal(args.Query) || strVal(args.regex) || strVal(args.text) || safeCleanArg(item.arg)
    const scope = strVal(args.path) || strVal(args.Path) || strVal(args.directory) || strVal(args.Directory) || strVal(args.dir) || strVal(args.include) || strVal(args.glob)
    const count = item.state === 'done' && !failed ? countOutputMatches(item.output, 'match') : undefined
    return {
      category: 'search',
      icon: 'search',
      badge: tail === 'search_history' ? 'History' : 'Search',
      target: pattern || 'workspace',
      ...(scope ? { scope } : {}),
      ...(count ? { metaPill: count, metaTone: 'neutral' as const } : {}),
    }
  }
  if (/^(glob|find|find_files)$/.test(tail)) {
    const pattern = strVal(args.pattern) || strVal(args.Pattern) || strVal(args.glob) || strVal(args.query) || strVal(args.Query) || safeCleanArg(item.arg)
    const scope = strVal(args.path) || strVal(args.Path) || strVal(args.directory) || strVal(args.Directory) || strVal(args.cwd)
    const count = item.state === 'done' && !failed ? countOutputMatches(item.output, 'file') : undefined
    return {
      category: 'search',
      icon: 'folder',
      badge: 'Find files',
      target: pattern || '*',
      ...(scope ? { scope } : {}),
      ...(count ? { metaPill: count, metaTone: 'neutral' as const } : {}),
    }
  }
  if (/^(list_dir|listdir|ls)$/.test(tail)) {
    const dir = strVal(args.path) || strVal(args.Path) || strVal(args.directory) || strVal(args.Directory) || strVal(args.dir) || safeCleanArg(item.arg) || '.'
    const count = item.state === 'done' && !failed ? countOutputMatches(item.output, 'item') : undefined
    return {
      category: 'search',
      icon: 'folder',
      badge: 'List',
      target: dir,
      ...(count ? { metaPill: count, metaTone: 'neutral' as const } : {}),
    }
  }
  if (/^(lsp|code_intel)$/.test(tail)) {
    const op = strVal(args.operation) || strVal(args.action) || strVal(args.method)
    const file = strVal(args.file_path) || strVal(args.path) || strVal(args.Path)
    const sym = strVal(args.symbol) || strVal(args.query) || strVal(args.Query)
    const line = typeof args.line === 'number' ? `:${args.line}` : ''
    const target = [op, sym || (file ? `${basenameOf(file)}${line}` : '')].filter(Boolean).join(' ') || safeCleanArg(item.arg) || 'inspect'
    const count = item.state === 'done' && !failed ? countOutputMatches(item.output, 'result') : undefined
    return {
      category: 'search',
      icon: 'code',
      badge: 'LSP',
      target,
      ...(file && sym ? { scope: basenameOf(file) } : {}),
      ...(count ? { metaPill: count, metaTone: 'neutral' as const } : {}),
    }
  }

  // 6. Web / Browser
  if (/^(web_search|search_web|google_search|duck_duck_go_search|duckduckgosearch)$/.test(tail)) {
    const query = strVal(args.query) || strVal(args.Query) || strVal(args.q) || safeCleanArg(item.arg)
    const domain = strVal(args.domain)
    const count = item.state === 'done' && !failed ? countOutputMatches(item.output, 'result') : undefined
    return {
      category: 'web',
      icon: 'cloud',
      badge: 'Web search',
      target: query || 'search',
      ...(domain ? { scope: domain } : {}),
      ...(count ? { metaPill: count, metaTone: 'neutral' as const } : {}),
    }
  }
  if (/^(web_fetch|read_url_content|web_scraper|url_analyzer|rss_reader|api_client)$/.test(tail)) {
    const url = strVal(args.url) || strVal(args.Url) || strVal(args.endpoint) || safeCleanArg(item.arg)
    const method = strVal(args.method).toUpperCase()
    return {
      category: 'web',
      icon: 'cloud',
      badge: tail === 'api_client' ? 'API' : 'Web fetch',
      target: formatUrlTarget(url) || 'request',
      ...(method ? { scope: method } : {}),
    }
  }
  if (/^(browser|computer_use)/.test(tail)) {
    const action = tail === 'computer_use'
      ? strVal(args.action)
      : tail.replace(/^browser_?/, '').replace(/_/g, ' ') || strVal(args.action)
    const rawTarget = strVal(args.url) || strVal(args.endpoint) || strVal(args.selector) || strVal(args.text)
    const formatted = rawTarget.startsWith('http') || rawTarget.startsWith('ws') ? formatUrlTarget(rawTarget) : clipText(rawTarget, 56)
    const target = [action, formatted].filter(Boolean).join(' · ') || safeCleanArg(item.arg) || 'session'
    return {
      category: 'web',
      icon: 'laptop',
      badge: 'Browser',
      target,
    }
  }

  // 7. Skill
  if (/^(skill|skill_manage|use_skill|load_skill|read_skill|run_skill)$/.test(tail)) {
    const skillName = strVal(args.skill) || strVal(args.name) || strVal(args.skill_name) || safeCleanArg(item.arg)
    const action = strVal(args.action)
    return {
      category: 'skill',
      icon: 'spark',
      badge: 'Skill',
      target: skillName || action || 'skill',
      ...(action && skillName ? { scope: action } : {}),
    }
  }

  // 8. Memory / Lessons / Reflections
  if (/^(memory|save_memory|recall_memory|workspace_memory|lesson_|reflection_)/.test(tail)) {
    const target = strVal(args.query) || strVal(args.topic) || strVal(args.title) || strVal(args.key) || strVal(args.action) || safeCleanArg(item.arg) || formatHumanLabel(tail)
    return {
      category: 'memory',
      icon: 'brain',
      badge: 'Memory',
      target,
    }
  }

  // 9. Schedule / Timer
  if (/^(schedule|manage_schedule|list_schedules|cron|timer)/.test(tail)) {
    const cron = strVal(args.cron) || strVal(args.cron_expression) || strVal(args.CronExpression)
    const durationSec = typeof args.duration_seconds === 'number'
      ? args.duration_seconds
      : typeof args.DurationSeconds === 'number'
        ? args.DurationSeconds
        : null
    const timing = cron || (durationSec !== null ? `in ${durationSec}s` : '')
    const name = strVal(args.name) || strVal(args.prompt) || strVal(args.Prompt) || strVal(args.action)
    const target = timing && name ? `${timing} · ${clipText(name, 48)}` : timing || clipText(name, 64) || safeCleanArg(item.arg) || 'schedules'
    return {
      category: 'schedule',
      icon: 'clock',
      badge: 'Schedule',
      target,
    }
  }

  // 10. Monitor / Watch
  if (/^(monitor_|list_monitors|stop_monitor|list_monitor_sources)/.test(tail)) {
    const target = strVal(args.file_path) || strVal(args.path) || strVal(args.source) || strVal(args.match) || strVal(args.terminal_id) || strVal(args.monitor_id) || safeCleanArg(item.arg) || formatHumanLabel(tail)
    return {
      category: 'monitor',
      icon: 'eye',
      badge: 'Monitor',
      target,
    }
  }

  // 11. Agent / Workflow / Plan
  if (/^(agent|spawn_agents|send_message|await_agents|peek_agent|check_agent_messages|workflow|handoff|task_)/.test(tail)) {
    const target = strVal(args.name) || strVal(args.agent) || strVal(args.title) || strVal(args.description) || safeCleanArg(item.arg)
    return {
      category: 'agent',
      icon: tail === 'workflow' ? 'lightning' : 'agents',
      badge: tail === 'workflow' ? 'Workflow' : formatHumanLabel(fallbackLabel || tail),
      target,
    }
  }
  if (/^(todo_write|plan|enter_plan_mode|exit_plan_mode)$/.test(tail)) {
    const target = strVal(args.title) || safeCleanArg(item.arg)
    return {
      category: 'plan',
      icon: 'plan',
      badge: 'Plan',
      target: target || formatHumanLabel(tail),
    }
  }

  // 12. Fallback — never show raw JSON in target!
  return {
    category: 'other',
    icon: 'tools',
    badge: fallbackLabel || formatHumanLabel(rawName),
    target: (item.title && item.title.trim()) || firstReadableArg(args, item.arg),
  }
}

/**
 * An agent's tool calls folded behind one header, closed by default. A busy
 * agent makes hundreds of calls; listing them all pushed its summary and
 * output far below. Rows are built only once the section is first opened.
 */
export function ToolCallsSection({ items, label = 'Tool calls' }: { items: readonly ToolItem[]; label?: string }): ReactElement {
  const [opened, setOpened] = useState(false)
  const runningItems = useMemo(() => items.filter(item => item.state === 'working'), [items])
  const failed = useMemo(() => items.filter(toolHasFailed).length, [items])
  const diffTotals = useMemo(() => {
    let adds = 0
    let dels = 0
    for (const item of items) {
      if (toolHasFailed(item)) continue
      const stats = fileToolStats(item)
      if (stats) {
        adds += stats.adds
        dels += stats.dels
      }
    }
    return { adds, dels }
  }, [items])

  return <div className="toolcalls-wrap">
    <details className="toolcalls" onToggle={event => { if (event.currentTarget.open) setOpened(true) }}>
      <summary>
        <span className="toolcalls__chev"><Icon name="chevron" size={12} /></span>
        <span className="toolcalls__title">{label}</span>
        <span className="toolcalls__count">{items.length}</span>
        {(diffTotals.adds > 0 || diffTotals.dels > 0) && (
          <span className="toolcalls__diff">
            <span className="add">+{diffTotals.adds}</span>
            {diffTotals.dels > 0 && <span className="del">−{diffTotals.dels}</span>}
          </span>
        )}
        {runningItems.length > 0 && <span className="toolcalls__tone" data-tone="working">{runningItems.length} running</span>}
        {failed > 0 && <span className="toolcalls__tone" data-tone="failed">{failed} failed</span>}
      </summary>
      {opened && (
        <div className="toolcalls__list tool-timeline__rail">
          {items.map(item => {
            const itemFailed = toolHasFailed(item)
            const stepState = itemFailed ? 'failed' : item.state
            return <div key={item.id} className="tool-timeline__step" data-kind="tool" data-state={stepState}>
              {(stepState === 'working' || stepState === 'failed') && <span className="tool-timeline__node" aria-hidden="true" />}
              <div className="tool-timeline__content">
                <ToolCallRow item={item} label={item.name.replaceAll('_', ' ')} />
              </div>
            </div>
          })}
        </div>
      )}
    </details>
    {!opened && runningItems.length > 0 && (
      <div className="tool-timeline__live-strip tool-timeline__live-strip--compact" aria-label="Active tool calls">
        {runningItems.slice(-2).map(item => {
          const info = classifyToolItem(item, item.name.replaceAll('_', ' '))
          const previewTarget = info.category === 'shell' && info.command ? commandGist(info.command) : info.target
          return <div key={item.id} className="tool-timeline__live-row" data-state="working" data-category={info.category}>
            <span className="tool-timeline__live-dot" aria-hidden="true" />
            <span className="tool-timeline__live-icon" aria-hidden="true"><Icon name={info.icon} size={12} /></span>
            <span className="tool-timeline__live-badge">{info.badge}</span>
            {previewTarget && <code className="tool-timeline__live-target">{info.category === 'shell' ? `$ ${previewTarget}` : previewTarget}</code>}
          </div>
        })}
      </div>
    )}
  </div>
}

export const ToolCallRow = memo(function ToolCallRow({ item, label }: { item: ToolItem; label: string }): ReactElement {
  // A saved artifact is a card with an Open button, not a tool row to decode.
  const artifact = artifactOfTool(item)
  if (artifact) return <ArtifactCard artifact={artifact} />
  // Swarm orchestration tools render as dedicated Antigravity-grade action cards.
  if (isSwarmToolName(item.name)) return <SwarmToolCard item={item} failed={toolHasFailed(item)} />
  const file = fileToolOf(item)
  // A read of an image file shows the picture (ToolImages), not its bytes as text lines.
  if (file?.kind === 'read' && chatImageTypeOf(file.path)) return <GenericToolRow item={item} label={label} />
  return file ? <FileToolRow item={item} tool={file} failed={toolHasFailed(item)} /> : <GenericToolRow item={item} label={label} />
})

const GenericToolRow = memo(function GenericToolRow({ item, label }: { item: ToolItem; label: string }): ReactElement {
  const [inspected, setInspected] = useState(false)
  const view = executionView(item)
  const failed = toolHasFailed(item)
  const semantic = classifyToolItem(item, label)
  const isShell = semantic.category === 'shell'
  const dur = item.dur && item.dur !== '0.0s' ? item.dur : ''
  const targetText = isShell ? (view.command || semantic.target) : semantic.target

  return <details
    className="toolrow execution-row"
    data-category={semantic.category}
    data-state={failed ? 'failed' : item.state}
    data-command={view.command ? '' : undefined}
    onToggle={event => { if (event.currentTarget.open) setInspected(true) }}
  >
    <summary>
      <span className="execution-row__main">
        <span className="execution-row__label" data-category={semantic.category} title={label}>
          <span className="execution-row__icon" aria-hidden="true">
            <Icon name={item.state === 'working' ? 'spinner' : semantic.icon} size={13} />
          </span>
          <span className="execution-row__badge">{isShell ? (view.summary || label) : semantic.badge}</span>
        </span>
        {targetText && (
          <code className={`execution-row__target${isShell ? ' execution-row__target--cmd' : ''}`} title={targetText}>
            {isShell && <span className="execution-row__prompt" aria-hidden="true">$ </span>}
            {targetText}
          </code>
        )}
        {semantic.scope && (
          <span className="execution-row__scope" title={semantic.scope}>{semantic.scope}</span>
        )}
        {semantic.metaPill && (
          <span className="execution-row__meta" data-tone={semantic.metaTone ?? 'neutral'}>{semantic.metaPill}</span>
        )}
        <span className="execution-row__trail">
          {item.diff && (
            <span className="execution-row__diff">
              <span className="add">+{item.diff.adds}</span> <span className="del">−{item.diff.dels}</span>
            </span>
          )}
          {(item.state === 'working' || failed || dur) && (
            <span className="execution-row__state">
              {item.state === 'working' ? 'Running' : failed ? 'Failed' : ''}{dur ? (item.state === 'working' || failed ? ` · ${dur}` : dur) : ''}
            </span>
          )}
          <span className="execution-row__disclosure"><Icon name="chevron" size={12} /></span>
        </span>
      </span>
      {failed && (item.error || view.stderr) && (
        <span className="execution-row__error">{item.error || view.stderr}</span>
      )}
    </summary>
    {inspected && <ExecutionDetails item={item} />}
  </details>
})

function ToolContentBlocks({ blocks }: { readonly blocks: readonly ToolContentBlock[] }): ReactElement | null {
  const open = useDesktopNavigation()
  const nonImage = blocks.filter(block => block.type !== 'image' && block.type !== 'text')
  if (nonImage.length === 0) return null
  return (
    <div className="execution__blocks" role="region" aria-label="Tool attachments">
      {nonImage.map((block, index) => {
        if (block.type === 'resource_link') {
          const isExternal = /^https?:\/\//i.test(block.uri)
          const label = block.title || block.name || block.uri
          return (
            <div key={`rlink:${block.uri}:${index}`} className="execution__block execution__block--link">
              <Icon name={isExternal ? 'cloud' : 'file'} size={12} />
              {isExternal ? (
                <button
                  type="button"
                  className="execution__block-link"
                  title={block.uri}
                  onClick={() => void window.xerxes?.openExternal?.(block.uri)}
                >
                  {label}
                </button>
              ) : (
                <button
                  type="button"
                  className="execution__block-link"
                  title={block.uri}
                  onClick={() => open('files', block.uri.replace(/^file:\/\//i, ''))}
                >
                  {label}
                </button>
              )}
              {block.description && <span className="execution__block-meta">{block.description}</span>}
            </div>
          )
        }
        if (block.type === 'resource') {
          if (block.mime_type?.toLowerCase().startsWith('image/') && block.blob) return null
          return (
            <div key={`res:${block.uri}:${index}`} className="execution__block execution__block--resource">
              <div className="execution__block-head">
                <Icon name="file" size={12} />
                <span title={block.uri}>{block.uri}</span>
                {block.mime_type && <span className="execution__block-meta">{block.mime_type}</span>}
              </div>
              {block.text && <OutputViewer text={block.text} compact />}
            </div>
          )
        }
        if (block.type === 'terminal') {
          return (
            <div key={`term:${block.terminal_id}:${index}`} className="execution__block execution__block--terminal">
              <Icon name="terminal" size={12} />
              <button
                type="button"
                className="execution__block-link"
                onClick={() => open('terminal')}
              >
                Terminal · {block.terminal_id}
              </button>
            </div>
          )
        }
        if (block.type === 'audio') {
          return (
            <div key={`audio:${index}`} className="execution__block execution__block--audio">
              <Icon name="activity" size={12} />
              <span>Audio ({block.mime_type})</span>
            </div>
          )
        }
        return null
      })}
    </div>
  )
}

export function ExecutionDetails({ item }: { item: ToolItem }): ReactElement {
  const [expanded, setExpanded] = useState(false)
  const view = executionView(item)
  const failed = toolHasFailed(item)
  const output = view.stdout ?? readableOutput(item.output)
  const structured = view.stdout === null && output === item.output ? structuredOutput(item.output) : null
  const hasBlocks = Boolean(item.contentBlocks && item.contentBlocks.some(b => b.type !== 'text'))
  const hasActions = Boolean(view.command || output || item.input)
  // A finished step says nothing about itself (Claude Code's): only running
  // or failed steps carry a status line.
  return <div className="execution">
    {(item.state === 'working' || failed) && <div className="execution__status" data-failed={failed || undefined}>
      <span>{item.state === 'working' ? 'Running' : 'Failed'}{view.exitCode !== null ? ` · Exit ${view.exitCode}` : ''}</span>
      {view.cwd && <span title={view.cwd}>{view.cwd}</span>}
    </div>}
    {view.command && <pre className="execution__command" title={view.cwd ?? undefined}>{view.command}</pre>}
    {output && (structured ? <><div className="execution__viewer-actions"><span>Result</span><button type="button" onClick={() => setExpanded(value => !value)} aria-expanded={expanded}>{expanded ? 'Compact view' : 'Expand result'}</button></div><div className="execution__viewer" data-expanded={expanded || undefined} tabIndex={0} role="region" aria-label="Tool output"><StructuredResult value={structured} /></div></> : <OutputViewer text={output} compact />)}
    {!failed && <ToolImages item={item} />}
    {!failed && item.contentBlocks && item.contentBlocks.length > 0 && <ToolContentBlocks blocks={item.contentBlocks} />}
    {view.stderr && <div className="execution__error"><strong>Standard error</strong><pre>{view.stderr}</pre></div>}
    {item.error && item.error !== output && item.error !== view.stderr && <div className="execution__error"><strong>Error</strong><pre>{item.error}</pre></div>}
    {!output && !item.error && !view.stderr && !hasBlocks && <p className="execution__empty">{item.state === 'working' ? 'Waiting for output…' : 'No output'}</p>}
    <div className="execution__footer">
      <details className="execution__raw"><summary>Raw details</summary><code>{item.name} · {item.id}</code><strong>Input</strong><pre>{item.input || '(no arguments)'}</pre><strong>Result</strong><pre>{item.output || '(no result)'}</pre></details>
      {hasActions && <div className="execution__actions">
        {view.command && <CopyButton text={view.command} label="Copy command" />}
        {/* glob, list_dir, memory, browser, skill and MCP results all land in
            the structured branch, which had no copy control — the output was
            reachable only by drag-selecting it. */}
        {output && <CopyButton text={output} label="Copy output" />}
        {item.input && <CopyButton text={item.input} label="Copy arguments" />}
      </div>}
    </div>
  </div>
}
