// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Session transcript → markdown export (mockup 08 "Export as markdown").
 *
 * Pure and offline: it renders exactly what the session record carries —
 * the same `messages` + `tool_executions` + `thinking_content` fold the
 * activity replay uses — and never invents tool rows or results for
 * records the daemon did not store.
 */

export interface ExportSession {
  readonly id?: unknown
  readonly title?: unknown
  readonly model?: unknown
  readonly cwd?: unknown
  readonly messages?: unknown
  readonly tool_executions?: unknown
  readonly thinking_content?: unknown
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === 'object' && !Array.isArray(value)

const asText = (value: unknown): string => (typeof value === 'string' ? value : '')

/** Fenced code block, widening the fence when the text itself contains one. */
function codeBlock(text: string): string {
  const fence = text.includes('```') ? '````' : '```'
  return `${fence}\n${text}\n${fence}`
}

/** The one argument a human wants to see, or compact JSON for the rest. */
function argSummary(inputs: unknown): string {
  if (isRecord(inputs)) {
    const cmd = inputs.cmd ?? inputs.command
    if (Array.isArray(cmd)) return cmd.map(String).join(' ')
    for (const key of ['cmd', 'command', 'file_path', 'path', 'pattern', 'query', 'url', 'skill']) {
      const value = inputs[key]
      if (typeof value === 'string' && value.trim()) return value
    }
  }
  const json = JSON.stringify(inputs ?? {})
  return json.length > 160 ? `${json.slice(0, 157)}…` : json
}

function toolLine(exec: Record<string, unknown>): string {
  const name = asText(exec.name) || 'tool'
  const pieces = [`**\`${name}\`** \`${argSummary(exec.inputs ?? exec.arguments)}\``]
  if (typeof exec.durationMs === 'number') pieces.push(`_( ${(exec.durationMs / 1000).toFixed(1)}s )_`)
  const error = asText(exec.error)
  if (error) pieces.push(`⚠️ ${error}`)
  return `- ${pieces.join(' — ')}`
}

/**
 * Render the session record as markdown. Ordering matches the activity
 * replay: reasoning precedes its answer, tool activity is reconstructed
 * from `tool_executions` in message order.
 */
export function sessionToMarkdown(session: ExportSession, exportedAt = new Date()): string {
  const title = asText(session.title) || (asText(session.id) ? `Session ${asText(session.id)}` : 'Session')
  const meta = [
    `Exported from Xerxes · ${exportedAt.toISOString().slice(0, 16).replace('T', ' ')}`,
    asText(session.id) && `session \`${asText(session.id)}\``,
    asText(session.model) && `model \`${asText(session.model)}\``,
    asText(session.cwd) && `cwd \`${asText(session.cwd)}\``,
  ].filter(Boolean).join(' · ')

  const out: string[] = [`# ${title}`, `> ${meta}`]
  const messages = Array.isArray(session.messages) ? session.messages : []
  const executions = Array.isArray(session.tool_executions) ? session.tool_executions : []
  const thinking = Array.isArray(session.thinking_content) ? session.thinking_content : []
  let executionIndex = 0
  let assistantIndex = 0

  for (const message of messages) {
    if (!isRecord(message)) continue
    const role = asText(message.role)

    if (role === 'tool') {
      // The stored result: its call lives in tool_executions at the same
      // order. Without a stored execution there is nothing honest to show.
      const exec = executions[executionIndex]
      executionIndex += 1
      if (!isRecord(exec)) continue
      out.push(toolLine(exec))
      const result = asText(message.content)
      if (result.trim()) out.push(codeBlock(result))
      continue
    }
    if (role === 'system') continue

    const turnThinking = role === 'assistant' ? asText(thinking[assistantIndex]) : ''
    if (role === 'assistant') assistantIndex += 1
    if (turnThinking.trim()) {
      out.push(`**Thinking**\n\n> ${turnThinking.trim().replace(/\n/g, '\n> ')}`)
    }

    const content = message.content
    if (typeof content === 'string') {
      if (!content.trim()) continue
      out.push(role === 'user' ? `## You\n\n${content}` : `## Agent\n\n${content}`)
      continue
    }
    if (!Array.isArray(content)) continue

    const userParts: string[] = []
    for (const part of content) {
      if (!isRecord(part)) continue
      if (part.type === 'tool_use') {
        out.push(toolLine({ name: part.name, inputs: part.input ?? part.arguments }))
        continue
      }
      if (part.type === 'tool_result') {
        const result = asText(part.content)
        if (result.trim()) out.push(codeBlock(result))
        continue
      }
      const think = asText(part.think) || asText(part.thinking)
      const text = asText(part.text)
      if (think.trim()) out.push(`**Thinking**\n\n> ${think.trim().replace(/\n/g, '\n> ')}`)
      if (text.trim()) {
        if (role === 'user') userParts.push(text)
        else out.push(`## Agent\n\n${text}`)
      }
    }
    if (userParts.length) out.push(`## You\n\n${userParts.join('\n')}`)
  }
  return `${out.join('\n\n')}\n`
}
