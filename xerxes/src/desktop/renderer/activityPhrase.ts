// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { commandGist } from './commandGist.js'
import type { Block, ToolItem } from './types.js'

/**
 * Plain-language status for an activity group.
 *
 * The group header used to say "Working" while anything ran and "Used 12
 * tools · Exec command, Read file" after — the first says nothing about what
 * is happening, the second is the tool registry read aloud. These phrases say
 * what the agent is doing ("Running pytest tests/", "Editing store.ts") and,
 * once finished, what it did ("Ran 4 commands, edited 2 files").
 */

type Family = 'command' | 'read' | 'edit' | 'search' | 'web' | 'spawn' | 'message' | 'wait' | 'agents' | 'plan' | 'other'

const FAMILIES: ReadonlyArray<readonly [Family, RegExp]> = [
  ['command', /^(exec_?command|check_?command|bash|shell|run_?command|pty_\w+|read_?terminal_?output|kill_?command)$/],
  ['edit', /^(file_?edit|edit|write_?file|append_?file|notebook_?edit|multi_?edit|apply_?patch)$/],
  ['read', /^(read_?file|read|list_?dir|ls|view)$/],
  ['search', /^(grep|glob|search|find|lsp|tool_?search|search_?history)$/],
  ['web', /^(web_?scraper|web_?fetch|web_?search|google_?search|duck_?duck_?go_?search|api_?client|url_?analyzer|rss_?reader|browser\w*)$/],
  ['spawn', /^(agent|spawn_?agents|task_?create|handoff)$/],
  ['message', /^(send_?message)$/],
  ['wait', /^(await_?agents)$/],
  ['agents', /^(check_?agent_?messages|peek_?agent|task_?(get|list|output|stop|update)|reset_?agent)$/],
  ['plan', /^(todo_?write|plan|enter_?plan_?mode|exit_?plan_?mode)$/],
]

/** `FileEditTool` / `exec_command` / `mcp.server:GrepTool` → a family. */
function familyOf(name: string): Family {
  const tail = (name.split(/[.:]/).pop() ?? name)
    .replace(/Tool$/, '')
    .replace(/([a-z0-9])([A-Z])/g, '$1_$2')
    .replace(/[\s-]+/g, '_')
    .toLowerCase()
  for (const [family, pattern] of FAMILIES) if (pattern.test(tail)) return family
  return 'other'
}

function clip(text: string, max = 56): string {
  const flat = text.replace(/\s+/g, ' ').trim()
  return flat.length > max ? `${flat.slice(0, max - 1)}…` : flat
}

function basename(path: string): string {
  return path.replace(/[\\/]+$/, '').split(/[\\/]/).pop() || path
}

/** Present-tense phrase for one running call: "Running pytest tests/". */
export function toolPhrase(item: Pick<ToolItem, 'name' | 'verb' | 'arg' | 'path'>): string {
  const target = item.path || item.arg
  switch (familyOf(item.name || item.verb)) {
    case 'command': return target ? `Running ${clip(commandGist(item.arg))}` : 'Running a command'
    case 'edit': return target ? `Editing ${basename(target)}` : 'Editing a file'
    case 'read': return target ? `Reading ${basename(target)}` : 'Reading files'
    case 'search': return item.arg ? `Searching for ${clip(item.arg, 40)}` : 'Searching the code'
    case 'web': return item.arg ? `Looking up ${clip(item.arg, 44)}` : 'Searching the web'
    case 'spawn': return 'Spawning agents'
    case 'message': return item.arg ? `Messaging ${clip(item.arg, 32)}` : 'Messaging an agent'
    case 'wait': return 'Waiting on agents'
    case 'agents': return 'Checking on agents'
    case 'plan': return 'Updating the plan'
    default: {
      const label = (item.verb || item.name).replace(/[_-]+/g, ' ').trim()
      return label ? `Using ${label}` : 'Working'
    }
  }
}

/**
 * What the group is doing right now. The newest running call wins; with
 * nothing running, live reasoning or still-working subagents explain the
 * wait, and otherwise the model is deciding its next step.
 */
export function liveActivityPhrase(blocks: readonly Block[]): string {
  const tools = blocks.flatMap(block => block.kind === 'tools' ? block.items : [])
  const running = tools.findLast(item => item.state === 'working')
  if (running) return toolPhrase(running)
  const last = blocks.at(-1)
  if (last?.kind === 'thinking' && last.streaming) return 'Thinking'
  const working = blocks.reduce((total, block) => total + (block.kind === 'agents'
    ? block.members.filter(member => member.status === 'working' || member.status === 'running').length
    : 0), 0)
  if (working > 0) return `Waiting on ${working} agent${working === 1 ? '' : 's'}`
  return 'Thinking'
}

const PAST: Record<Family, (count: number) => string> = {
  command: n => `ran ${n} command${n === 1 ? '' : 's'}`,
  edit: n => `edited ${n} file${n === 1 ? '' : 's'}`,
  read: n => `read ${n} file${n === 1 ? '' : 's'}`,
  search: n => `ran ${n} search${n === 1 ? '' : 'es'}`,
  web: n => `made ${n} web lookup${n === 1 ? '' : 's'}`,
  spawn: n => n === 1 ? 'started an agent' : `started ${n} agents`,
  message: n => `sent ${n} message${n === 1 ? '' : 's'}`,
  wait: () => 'waited on agents',
  agents: () => 'checked on agents',
  plan: () => 'updated the plan',
  other: n => `used ${n} other tool${n === 1 ? '' : 's'}`,
}

/**
 * Past-tense summary of a finished group: "Ran 4 commands, edited 2 files".
 * Edits and reads count distinct files, not calls — five edits to one file
 * is one edited file. The three largest families lead; the rest collapse.
 */
export function activitySummary(blocks: readonly Block[]): string {
  const tools = blocks.flatMap(block => block.kind === 'tools' ? block.items : [])
  if (tools.length === 0) {
    const spawned = blocks.reduce((total, block) => total + (block.kind === 'agents' ? block.members.length : 0), 0)
    if (spawned > 0) return spawned === 1 ? 'Started an agent' : `Started ${spawned} agents`
    if (blocks.some(block => block.kind === 'thinking')) return 'Thought it through'
    return blocks.some(block => block.kind === 'notice' && block.error) ? 'Runtime error' : 'Runtime notice'
  }
  const counts = new Map<Family, Set<string>>()
  for (const item of tools) {
    const family = familyOf(item.name || item.verb)
    const bucket = counts.get(family) ?? new Set<string>()
    bucket.add(family === 'edit' || family === 'read' ? (item.path || item.arg || item.id) : item.id)
    counts.set(family, bucket)
  }
  // One SpawnAgents call can start many agents; the card knows how many.
  const members = blocks.reduce((total, block) => total + (block.kind === 'agents' ? block.members.length : 0), 0)
  const ranked = [...counts.entries()]
    .map(([family, ids]) => [family, family === 'spawn' ? Math.max(ids.size, members) : ids.size] as const)
    .sort((a, b) => (a[0] === 'other' ? 1 : b[0] === 'other' ? -1 : b[1] - a[1]))
  const shown = ranked.slice(0, 3).map(([family, count]) => PAST[family](count))
  const rest = ranked.slice(3).reduce((total, [, count]) => total + count, 0)
  if (rest > 0) shown.push(`${rest} more`)
  const sentence = shown.length > 1 ? `${shown.slice(0, -1).join(', ')} and ${shown.at(-1)}` : shown[0]!
  return sentence[0]!.toUpperCase() + sentence.slice(1)
}

/** Approval card heading: the question being asked, not the registry id. */
export function approvalTitle(name: string): string {
  switch (familyOf(name)) {
    case 'command': return 'Run this command?'
    case 'edit': return 'Change this file?'
    case 'read': return 'Read this file?'
    case 'search': return 'Search the workspace?'
    case 'web': return 'Reach the network?'
    case 'spawn': return 'Start agents?'
    case 'message': return 'Send this message?'
    case 'plan': return 'Update the plan?'
    default: {
      const label = (name.split(/[.:]/).pop() ?? name).replace(/Tool$/, '').replace(/([a-z0-9])([A-Z])/g, '$1 $2').replace(/[_-]+/g, ' ').trim().toLowerCase()
      return label ? `Allow ${label}?` : 'Allow this action?'
    }
  }
}
