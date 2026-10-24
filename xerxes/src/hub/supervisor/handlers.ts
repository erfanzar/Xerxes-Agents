// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { AuthenticatedContext } from '../identityPort.js'
import { HubProtocolError, readId, type HubMethod } from '../protocol.js'
import type { HubMethodContext, HubMethodHandler } from '../server.js'
import { actionToWire, type ActionService } from './actions.js'
import { MAX_MESSAGE_CHARS, messageToWire, type MessageSource, type SupervisorThreads, type ThreadMessage, type ThreadRecord, type UserPrincipal } from './threads.js'

/**
 * The four `supervisor.*` Hub methods. Request and response shapes match what
 * `client/hubController.ts` already sends and parses:
 *
 * - `supervisor.thread.open {thread_id?, private?, fresh?}` -> `{thread_id, messages, confirmations, ...}`
 * - `supervisor.message.submit {message_id, text, source, thread_id?}` -> `{messages, confirmations}`
 * - `supervisor.confirmation.respond {confirmation_id, decision, response_id, thread_id?}` -> `{confirmation}`
 * - `supervisor.action.status {confirmation_id}` -> `{confirmation}`
 *
 * Every method needs a signed-in person; an agent principal is refused. The
 * model's reply is not part of the submit response: it arrives as a
 * `supervisor.thread_event`, because a turn can outlast a request.
 */

export type SupervisorHandlers = Readonly<Partial<Record<HubMethod, HubMethodHandler>>>

export interface SupervisorHandlerCore {
  readonly threads: Pick<SupervisorThreads, 'ensureShared' | 'privateFor' | 'access' | 'messagesFor' | 'participants' | 'isWorking'>
  readonly actions: Pick<ActionService, 'cardsFor' | 'respond' | 'status'>
  /** The model id shown in the thread; a function so a hot-swapped model shows at once. */
  readonly modelId: string | (() => string)
  /** Whether XERXES has a model to run on. `unconfigured` makes `supervisor.message.submit` refuse. */
  readonly configStatus?: () => { readonly status: string; readonly message?: string }
  usage(): { readonly tokens: number; readonly costUsd: number; readonly calls: number; readonly windowStart: string }
  submit(
    principal: UserPrincipal,
    input: { readonly threadId: string; readonly text: string; readonly source: MessageSource; readonly clientMessageId: string },
  ): { readonly message: ThreadMessage; readonly duplicate: boolean }
}

function exactKeys(params: Readonly<Record<string, unknown>>, allowed: readonly string[]): void {
  for (const key of Object.keys(params)) {
    if (!allowed.includes(key)) throw new HubProtocolError('invalid_request', `unknown parameter ${JSON.stringify(key.slice(0, 40))}`)
  }
}

/** Only a signed-in person uses the supervisor conversation. An agent principal never does. */
function requireUser(context: AuthenticatedContext): UserPrincipal {
  if (context.principal.kind !== 'user') throw new HubProtocolError('forbidden', 'only a signed-in person can use the supervisor')
  return context.principal
}

function snapshot(core: SupervisorHandlerCore, thread: ThreadRecord, principal: UserPrincipal): Record<string, unknown> {
  return {
    thread_id: thread.id,
    kind: thread.kind,
    title: thread.title,
    messages: core.threads.messagesFor(thread.id, principal.id).map(messageToWire),
    confirmations: core.actions.cardsFor(thread.id).map(actionToWire),
    participants: thread.kind === 'shared' ? core.threads.participants(thread.id).map(person => ({ id: person.id, name: person.displayName })) : [],
    model: typeof core.modelId === 'function' ? core.modelId() : core.modelId,
    config_status: core.configStatus?.().status ?? 'configured',
    busy: core.threads.isWorking(thread.id),
    usage: core.usage(),
  }
}

export function createSupervisorHandlers(core: SupervisorHandlerCore): SupervisorHandlers {
  const open: HubMethodHandler = (ctx: HubMethodContext, params) => {
    const principal = requireUser(ctx.context)
    exactKeys(params, ['thread_id', 'private', 'fresh'])
    let id: string
    if (params.thread_id !== undefined) id = readId(params.thread_id, 'thread_id')
    else if (params.private === true) id = core.threads.privateFor(principal, params.fresh === true).id
    else id = core.threads.ensureShared().id
    return snapshot(core, core.threads.access(id, principal), principal)
  }

  const submit: HubMethodHandler = (ctx, params) => {
    const principal = requireUser(ctx.context)
    exactKeys(params, ['message_id', 'text', 'source', 'thread_id'])
    // Nothing is stored and no model runs: say what to do instead of queueing a turn that cannot answer.
    const config = core.configStatus?.()
    if (config?.status === 'unconfigured') {
      throw new HubProtocolError('unconfigured', config.message ?? "Choose XERXES's model in XERXES settings")
    }
    const clientMessageId = readId(params.message_id, 'message_id')
    const text = typeof params.text === 'string' ? params.text.trim() : ''
    if (text === '' || [...text].length > MAX_MESSAGE_CHARS) {
      throw new HubProtocolError('invalid_request', `text must be 1 to ${MAX_MESSAGE_CHARS} characters`)
    }
    if (params.source !== undefined && params.source !== 'text' && params.source !== 'voice') {
      throw new HubProtocolError('invalid_request', "source must be 'text' or 'voice'")
    }
    const threadId = params.thread_id === undefined ? core.threads.ensureShared().id : readId(params.thread_id, 'thread_id')
    const { message, duplicate } = core.submit(principal, { threadId, text, source: params.source === 'voice' ? 'voice' : 'text', clientMessageId })
    return {
      thread_id: threadId,
      duplicate,
      messages: [messageToWire(message)],
      confirmations: core.actions.cardsFor(threadId).map(actionToWire),
    }
  }

  const respond: HubMethodHandler = async (ctx, params) => {
    const action = await core.actions.respond(ctx.context, params)
    return { thread_id: action.threadId, confirmation: actionToWire(action) }
  }

  const status: HubMethodHandler = async (ctx, params) => {
    const action = await core.actions.status(ctx.context, params)
    return { thread_id: action.threadId, confirmation: actionToWire(action) }
  }

  return {
    'supervisor.thread.open': open,
    'supervisor.message.submit': submit,
    'supervisor.confirmation.respond': respond,
    'supervisor.action.status': status,
  }
}

/**
 * Answers for a Hub whose supervisor is not configured: a clear `unsupported`
 * with the setup step, never an empty thread and never a fallback model.
 */
export function createUnsupportedSupervisorHandlers(reason: string): SupervisorHandlers {
  const refuse: HubMethodHandler = async () => {
    throw new HubProtocolError('unsupported', reason)
  }
  return {
    'supervisor.thread.open': refuse,
    'supervisor.message.submit': refuse,
    'supervisor.confirmation.respond': refuse,
    'supervisor.action.status': refuse,
    'supervisor.autopilot.get': refuse,
    'supervisor.autopilot.set': refuse,
  }
}
