// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import {
  COORDINATOR_NAME,
  MAIN_CHANNEL,
  SWARM_OPEN_BOARD_MARKDOWN,
  WILDCARD_CHANNEL,
  type SwarmMemberSnapshot,
} from './types.js'

export function buildSwarmWorkerPromptOverlay(member: SwarmMemberSnapshot): string {
  const parentName = member.added_by ?? COORDINATOR_NAME
  const channelSummary = member.channels.includes(WILDCARD_CHANNEL)
    ? 'all channels (*)'
    : member.channels.map(ch => `#${ch}`).join(', ')
  const workspaceNote = member.workspace === 'inherit'
    ? `Shared repository workspace (${member.workspace_path ?? '.'}) — coordinate file ownership carefully so you never overwrite another member's edits.`
    : member.workspace === 'share'
      ? `Isolated Git worktree (${member.workspace_path ?? '.'}, mode: share) — edit and build freely here, and ensure your changes are ready to land before your parent kills this workspace.`
      : `Isolated repository clone (${member.workspace_path ?? '.'}, mode: branch) — edit and test in isolation, and report how to integrate your changes before cleanup.`

  const roleSpecificLines: string[] = []
  if (member.role === 'narrator') {
    roleSpecificLines.push(
      '- Role: Live Overview Curator (`narrator`). Read board progress across all channels and maintain a clear, up-to-date Markdown synthesis with `swarm_overview`. Do not post conversational chatter with `post_message`; incorporate any `@narrator user feedback: ...` notes into the next overview update, then park with `swarm_wait` (`mentions_only: false`).',
    )
  } else if (member.role === 'monitor') {
    roleSpecificLines.push(
      '- Role: Stall Monitor (`monitor`). Observe roster and board activity with `read_board{status: true}` and `read_board{since}`. Stay silent while work is progressing; post a concise `@<member> [stall] ...` nudge citing `#<seq>` only when every worker is idle with unfinished work, and stop once `[done]` is posted.',
    )
  }

  return [
    '<swarm_worker>',
    `You are "${member.name}", a member of a Xerxes Immortals multi-agent force collaborating over a shared durable message board.`,
    ...(member.group ? [`Group: "${member.group}" (messages naming @${member.group} wake every living member of your group).`] : []),
    `Added by: @${parentName} (report final deliverables or blockers to @${parentName}).`,
    `Subscribed channels: ${channelSummary} (default channel: #${MAIN_CHANNEL}).`,
    `Workspace: ${workspaceNote}`,
    ...roleSpecificLines,
    '',
    'Board protocol:',
    `- Read before acting on wakeups: call \`read_board\` with \`since\` set to your highest read sequence number (currently #${member.last_read_seq}). Reference earlier posts by \`#<seq>\` instead of repeating long text.`,
    '- `@<name>` is a wakeup doorbell, never a courtesy CC: every `@mention` wakes that member for a turn. Mention only the specific peer who needs to act next; avoid `@all` unless broadcasting a force-wide contract change, and never use `@#main`.',
    '- No greetings or empty acknowledgements: go straight to your assigned work and post concrete findings, artifacts, or blockers.',
    `- Final deliverable format: begin your completion post with \`@${parentName} [done] <concise summary>\` and call \`swarm_wait\` in the same turn with \`since\` set to the highest sequence number you have read. Never poll \`read_board\` in a loop.`,
    '- Helper hierarchy: you may spawn sub-workers with `swarm{action:"add"}` for genuinely parallel slices, and you may `model` or `kill` only workers you added yourself (always integrate their `share`/`branch` workspace work before `kill`).',
    '</swarm_worker>',
  ].join('\n')
}

export function buildSwarmCoordinatorPromptOverlay(params: {
  readonly sessionId: string
  readonly title?: string
  readonly active: boolean
  readonly head: number
  readonly lastReadSeq: number
}): string {
  const titleLine = params.title ? `Immortals title: "${params.title}".` : 'Set a short descriptive title with `swarm{action:"title", title:"..."}` when starting.'
  const boardToolsNote = params.active
    ? `Board head is #${params.head} (your last read is #${params.lastReadSeq}). Use \`read_board\`, \`post_message\`, and \`swarm_wait\` to coordinate without polling.`
    : 'Board tools (`post_message`, `read_board`, `swarm_wait`) unlock automatically on the turn after your first `swarm{action:"add"}`.'

  return [
    '<swarm_coordinator>',
    `You are "${COORDINATOR_NAME}" for Xerxes Immortals session "${params.sessionId}". ${titleLine}`,
    boardToolsNote,
    '- Decompose work into non-overlapping file/module slices, spawn focused workers (`swarm{action:"add"}`), and pair implementation slices with a read-only verifier where appropriate.',
    '- When `immortals-narrator` / `immortals-monitor` skills are available, spawn `narrator` and `monitor` in the opening wave (`workspace: "inherit"`). Relay user feedback to `@narrator` as `@narrator user feedback: <verbatim>` (or `@narrator user feedback (default): ...`) while stripping `@` before member/group names or `all`.',
    '- Use `@<name>` only to wake the member that must act next, cite board history by `#<seq>`, and on worker error/quota wakeups inspect `swarm{action:"models"}` and call `swarm{action:"model", name, model, reason}` in place (never `kill`+`add`).',
    '- Before `swarm{action:"kill"}`, land any `share`/`branch` workspace work. Leave `monitor` and `narrator` alive unless asked to stop them, post one closing `[done] <summary>` with NO `@` mentions to quiet `monitor`, and end replies with the board link on its own line:',
    SWARM_OPEN_BOARD_MARKDOWN,
    '</swarm_coordinator>',
  ].join('\n')
}

export function buildWorkerStartupPrompt(params: {
  readonly memberName: string
  readonly briefSeq: number
  readonly channel: string
  readonly effectiveBrief: string
}): string {
  return [
    `You have just started. Your assignment (recorded on the board as #${params.briefSeq} in #${params.channel} for @${params.memberName}):`,
    '',
    params.effectiveBrief,
  ].join('\n')
}

export function buildWorkerWakePrompt(params: {
  readonly head: number
  readonly wakeSeq: number
  readonly wakeAuthor: string
  readonly wakeChannel: string
  readonly lastReadSeq: number
  readonly revived?: boolean
}): string {
  const prefix = params.revived
    ? '[swarm] Revived from stopped state by new board activity'
    : '[swarm] New board activity'
  return `${prefix} (#${params.wakeSeq} from @${params.wakeAuthor} in #${params.wakeChannel}; board head is #${params.head}, your last read is #${params.lastReadSeq}). Call read_board with since:${params.lastReadSeq} to read what arrived.`
}

export function buildWorkerModelSwitchPrompt(params: {
  readonly previousModel: string
  readonly nextModel: string
  readonly lastReadSeq: number
  readonly reason?: string
}): string {
  const suffix = params.reason ? ` (${params.reason})` : ''
  return `[swarm] Your active model was switched from "${params.previousModel}" to "${params.nextModel}"${suffix}. Your conversation history is intact; resume any unfinished work, or call swarm_wait with since:${params.lastReadSeq} if you have nothing left to do.`
}
