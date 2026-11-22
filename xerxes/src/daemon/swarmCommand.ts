// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type {
  SwarmBoardMessage,
  SwarmChannelSummary,
  SwarmStatusSnapshot,
} from '../swarm/types.js'

export const SWARM_USAGE =
  'Usage: /immortals [<task>|board|status|channels|kill <name|all>|stop]'

export type SwarmCommand =
  | { readonly kind: 'status'; readonly openBoard: boolean }
  | { readonly kind: 'channels' }
  | { readonly kind: 'kill'; readonly target: string }
  | { readonly kind: 'invalid-kill' }
  | { readonly kind: 'activate'; readonly task: string }

/**
 * Parse `/immortals` (or legacy `/swarm`) arguments across daemon, TUI, Desktop, and CLI surfaces.
 *
 * Only exact control words (`board`, `status`, `channels`, `stop`, `kill <target>`)
 * are treated as control actions; any other text is treated as an Immortals task prompt.
 */
export function parseSwarmCommand(rawInput: string): SwarmCommand {
  const input = rawInput.trim()
  if (!input) {
    return { kind: 'status', openBoard: true }
  }
  const lower = input.toLowerCase()
  if (lower === 'board') {
    return { kind: 'status', openBoard: true }
  }
  if (lower === 'status') {
    return { kind: 'status', openBoard: false }
  }
  if (lower === 'channels') {
    return { kind: 'channels' }
  }
  if (lower === 'stop') {
    return { kind: 'kill', target: 'all' }
  }
  if (lower === 'kill') {
    return { kind: 'invalid-kill' }
  }
  if (/^kill\s+/iu.test(input)) {
    const target = input.slice(4).trim()
    return target ? { kind: 'kill', target } : { kind: 'invalid-kill' }
  }
  return { kind: 'activate', task: input }
}

export function renderSwarmStatusText(
  status: SwarmStatusSnapshot,
  messages: readonly SwarmBoardMessage[] = [],
): string {
  const working = status.members.filter(member => member.state === 'working').length
  const waiting = status.members.filter(member => member.state === 'waiting').length
  const stopped = status.members.filter(member => member.state === 'stopped').length
  const killed = status.members.filter(member => member.state === 'killed').length
  const title = status.title.trim() || '(untitled Immortals)'

  const lines: string[] = [
    `Immortals: ${title}`,
    `Active: ${status.active ? 'yes' : 'no'} · Head: #${status.head} · Members: ${working} working, ${waiting} waiting${stopped ? `, ${stopped} stopped` : ''}${killed ? `, ${killed} killed` : ''}`,
  ]

  if (status.overview) {
    lines.push('')
    lines.push('Overview:')
    const overviewBody = status.overview.summary?.trim() || status.overview.markdown.trim()
    for (const line of overviewBody.split(/\r?\n/).slice(0, 8)) {
      lines.push(`  ${line}`)
    }
  }

  lines.push('')
  if (!status.members.length) {
    lines.push('Roster: (no members yet)')
  } else {
    lines.push(`Roster (${status.members.length}):`)
    for (const member of status.members) {
      const parent = member.added_by ? ` (by @${member.added_by})` : ''
      const readInfo = `read:#${member.last_read_seq}${member.last_post_seq !== undefined ? ` post:#${member.last_post_seq}` : ''}`
      const errInfo = member.error ? ` error: ${member.error}` : ''
      lines.push(
        `  - ${member.name} [${member.role}] ${member.state} · model:${member.model_display_name || member.model} · ws:${member.workspace} · ${readInfo}${parent}${errInfo}`,
      )
    }
  }

  lines.push('')
  if (!messages.length) {
    lines.push('Recent board messages: (none)')
  } else {
    const tail = messages.slice(-10)
    lines.push(`Recent board messages (${tail.length} of #${status.head}):`)
    for (const msg of tail) {
      const snippet = msg.text.replace(/\s+/gu, ' ').trim().slice(0, 140)
      lines.push(`  #${msg.seq} [${msg.time}] @${msg.agent} (#${msg.channel}): ${snippet}`)
    }
  }

  lines.push('')
  lines.push(SWARM_USAGE)
  return lines.join('\n')
}

export function renderSwarmChannelsText(
  channels: readonly SwarmChannelSummary[],
  title?: string,
): string {
  const header = title?.trim() ? `Immortals channels — ${title.trim()}` : 'Immortals channels'
  if (!channels.length) {
    return `${header}\n  (no channels)`
  }
  const lines = [`${header} (${channels.length}):`]
  for (const ch of channels) {
    const archivedTag = ch.archived ? ' [archived]' : ''
    const topicPart = ch.topic ? ` — ${ch.topic}` : ''
    lines.push(
      `  #${ch.name}${archivedTag} (${ch.subscriber_count} subscriber${ch.subscriber_count === 1 ? '' : 's'}, by @${ch.created_by})${topicPart}`,
    )
  }
  return lines.join('\n')
}

/**
 * Strip `@` immediately preceding known swarm member names, group prefixes,
 * `all`, or `#channel` references inside UI narrator feedback so user feedback
 * wakes only `@narrator` as `@narrator user feedback: <sanitized>`.
 */
export function formatSwarmNarratorFeedback(
  rawText: string,
  status: Pick<SwarmStatusSnapshot, 'members' | 'channels'>,
): string {
  const trimmed = rawText.trim()
  const memberTokens = new Set<string>(['all', 'coordinator'])
  for (const member of status.members) {
    memberTokens.add(member.name.toLowerCase())
    if (member.group) {
      memberTokens.add(member.group.toLowerCase())
    }
  }
  const channelTokens = new Set<string>(['main'])
  for (const ch of status.channels) {
    channelTokens.add(ch.name.toLowerCase())
  }

  const sanitized = trimmed
    .replace(/@#([a-z0-9_-]+)/giu, (full, ch: string) => (
      channelTokens.has(ch.toLowerCase()) ? `#${ch}` : full
    ))
    .replace(/@([a-z0-9][a-z0-9_-]*)/giu, (full, name: string) => (
      memberTokens.has(name.toLowerCase()) ? name : full
    ))

  return `@narrator user feedback: ${sanitized}`
}
