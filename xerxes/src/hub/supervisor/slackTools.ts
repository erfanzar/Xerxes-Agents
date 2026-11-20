// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHash } from 'node:crypto'
import { ToolRegistry } from '../../executors/toolRegistry.js'
import type { JsonObject } from '../../types/toolCalls.js'
import { teamId, userId } from '../slack/wire.js'
import { SupervisorDenied } from './actions.js'

export interface SupervisorSlackPort {
  contacts(ownerId: string): readonly { teamId: string; peerId: string; mode: 'auto' | 'draft' }[]
  prepare(ownerId: string, target: { teamId: string; peerId: string }, text: string, signal: AbortSignal, requestId: string): Promise<'drafted' | 'sent'>
  isOwner(ownerId: string): boolean
}

export const SLACK_LIST_TOOL = 'slack_list_contacts'
export const SLACK_PREPARE_TOOL = 'slack_prepare_dm'

/** Separate from the fixed supervisor registry. Never passes Slack/profile/draft text into the model. */
export function createPrivateSlackTools(port: SupervisorSlackPort) {
  const registry = new ToolRegistry()
  const authorize = (owner: unknown): string => {
    if (typeof owner !== 'string' || !port.isOwner(owner)) throw new SupervisorDenied('Slack tools require a private turn from the current Hub owner.')
    return owner
  }
  registry.register({ type: 'function', function: {
    name: SLACK_LIST_TOOL, description: 'List approved Slack DM destinations by exact IDs and delivery mode. No Slack messages are available to this model.',
    parameters: { type: 'object', properties: {}, additionalProperties: false },
  } }, (args, context) => {
    const owner = authorize(context.metadata.slackOwner)
    if (Object.keys(args).length !== 0) throw new Error('Slack contacts accepts no arguments.')
    try {
      return { contacts: port.contacts(owner)
        .filter(row => teamId(row.teamId) && userId(row.peerId) && (row.mode === 'draft' || row.mode === 'auto'))
        .map(row => ({ team_id: row.teamId, user_id: row.peerId, mode: row.mode })) }
    } catch {
      throw new Error('Slack contacts are unavailable.')
    }
  })
  registry.register({ type: 'function', function: {
    name: SLACK_PREPARE_TOOL, description: 'Only on a direct request to message an approved Slack contact. Prepare a plain-text DM to the exact approved IDs. Draft mode requires review in Slack settings; auto mode may send immediately. Never retry an uncertain delivery.',
    parameters: { type: 'object', properties: {
      team_id: { type: 'string' }, user_id: { type: 'string' }, text: { type: 'string', minLength: 1, maxLength: 4000 },
    }, required: ['team_id', 'user_id', 'text'], additionalProperties: false },
  } }, async (args: JsonObject, context, signal) => {
    const owner = authorize(context.metadata.slackOwner)
    const { team_id, user_id, text } = args
    if (Object.keys(args).length !== 3 || !teamId(team_id) || !userId(user_id) || typeof text !== 'string' || !text.trim() || text.length > 4000 || signal === undefined) {
      throw new Error('An exact Slack team ID, user ID and bounded message are required.')
    }
    signal.throwIfAborted()
    try {
      const causalId = context.metadata.slackCausalId
      if (typeof causalId !== 'string') throw new Error('A direct request is required.')
      const requestId = createHash('sha256').update(JSON.stringify([causalId, team_id, user_id, text])).digest('hex')
      const state = await port.prepare(owner, { teamId: team_id, peerId: user_id }, text, signal, requestId)
      return { state: state === 'sent' ? 'sent' : 'drafted' }
    } catch {
      // Nothing from Slack, including upstream exception messages, is model-visible.
      throw new Error('Slack could not complete the operation. Check the Slack panel; do not retry blindly.')
    }
  })
  return registry
}
