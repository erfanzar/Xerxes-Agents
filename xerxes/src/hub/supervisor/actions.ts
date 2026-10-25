// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { AsyncLocalStorage } from 'node:async_hooks'
import { createHash, randomUUID } from 'node:crypto'
import type { Database } from 'bun:sqlite'

import { canonicalJson } from '../../security/hubAuthEnvelope.js'
import { hasScope, type HubGrant, type HubPrincipal, type HubScope } from '../../security/principal.js'
import type { JsonObject } from '../../types/toolCalls.js'
import type { AuthenticatedContext, HubIdentityPort } from '../identityPort.js'
import { HubProtocolError, type HubCommandAction, type HubCommandResult } from '../protocol.js'
import {
  SUPERVISOR_DISPLAY_NAME, SUPERVISOR_PRINCIPAL_ID, SupervisorAudit, supervisorActor, supervisorRequestHash,
} from './audit.js'
import {
  SHARED_THREAD_ID, delegatedScopes, isSupervisorThreadId, messageToWire, type ScopeReader, type SupervisorEventType, type SupervisorThreads, type UserPrincipal,
} from './threads.js'

/**
 * Confirmed actions.
 *
 * A mutation tool never runs anything. It creates an action intent (a
 * confirmation card) and returns `pending_confirmation`. A signed-in human with
 * the scope the action needs on that target confirms or cancels it. The card is
 * bound to the canonical arguments, the target revision, the runtime instance,
 * an expiry and a single-use idempotency key, and is re-validated at execution.
 *
 * Confirmation is its own interaction. It never writes to the approval store
 * and never answers a permission or question. The confirm route is not a model
 * tool and refuses agent principals structurally.
 *
 * Execution goes through `HubCommandRouter` with an agent actor that carries
 * `requested_by` and `confirmed_by` (attribution only). The agent's authority
 * for that one call is the confirming human's effective scopes inside the agent
 * ceiling, handed to the router through `SupervisorGrantOverlay`.
 *
 * An outcome of `unknown` is never retried by anything in this module.
 */

export const ACTION_TOOLS = ['task.start', 'task.steer', 'task.stop', 'agent.message', 'goal.set', 'model.set'] as const
export type ActionTool = (typeof ACTION_TOOLS)[number]

export function isActionTool(name: string): name is ActionTool {
  return (ACTION_TOOLS as readonly string[]).includes(name)
}

/** The scope a person needs on the target to confirm each action. */
export const ACTION_SCOPE: Readonly<Record<ActionTool, HubScope>> = Object.freeze({
  'task.start': 'session.submit',
  'task.steer': 'session.steer',
  'task.stop': 'session.cancel',
  'agent.message': 'session.steer',
  'goal.set': 'goal.write',
  'model.set': 'session.configure',
})

export const UNKNOWN_OUTCOME_MESSAGE = 'Could not confirm whether this was applied. Check the task before trying again. XERXES will not retry it.'
export const DEFAULT_ACTION_EXPIRY_MS = 10 * 60 * 1000
const SELF_NAMES: ReadonlySet<string> = new Set(['xerxes-supervisor', 'supervisor', 'xerxes'])

/** An access refusal for the supervisor (as opposed to bad arguments). The runner stops the turn on it. */
export class SupervisorDenied extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'SupervisorDenied'
  }
}

export type ActionState = 'pending' | 'confirming' | 'accepted' | 'rejected' | 'unknown' | 'cancelled' | 'expired'

export interface ActionRecord {
  readonly id: string
  readonly threadId: string
  readonly tool: ActionTool
  readonly workspaceId: string
  readonly sessionId?: string
  readonly args: JsonObject
  readonly argsHash: string
  readonly summary: string
  readonly targets: readonly string[]
  readonly instanceId?: string
  readonly targetRevision?: string
  readonly requestedBy: string
  readonly requestedByName: string
  readonly idempotencyKey: string
  readonly state: ActionState
  readonly confirmedBy?: string
  readonly confirmedByName?: string
  readonly outcome?: Readonly<Record<string, unknown>>
  readonly errorCode?: string
  readonly errorMessage?: string
  readonly createdAt: number
  readonly expiresAt: number
  readonly updatedAt: number
  /** Run by XERXES Autopilot without a Confirm tap; `confirmedBy` is the enabling Hub owner. */
  readonly auto: boolean
}

/**
 * The Hub owner Autopilot acts for, when Autopilot is active for a workspace.
 * `wake` is true for a turn the Hub started by itself (an Autopilot wake-up):
 * such a turn may only steer, stop or message without a Confirm tap.
 */
export interface AutopilotGrant {
  readonly owner: UserPrincipal
  readonly wake: boolean
}

/** What the turn that proposes an action looked like. */
export interface ActionTurnInfo {
  /** Set when the Hub started the turn (an Autopilot or Watch wake-up). */
  readonly trigger?: 'autopilot' | 'watch'
  /** Whether a user message by anyone but the requester was in the turn's context. Unknown counts as true. */
  readonly othersInContext?: boolean
  /** The one task a Hub wake-up is about. A wake's action on anything else is a normal Confirm card. */
  readonly target?: { readonly workspaceId: string; readonly sessionId: string }
}

/** The only actions an Autopilot wake-up runs without a Confirm tap. */
export const WAKE_AUTO_TOOLS: ReadonlySet<ActionTool> = new Set<ActionTool>(['task.steer', 'task.stop', 'agent.message'])

export type AutopilotBudgetTake =
  | { readonly ok: true }
  | { readonly ok: false; readonly notify: boolean; readonly limit: number }

/** What the actions need to know about the world, supplied by projections in production and fakes in tests. */
export interface ActionTargets {
  instanceId(workspaceId: string): string | undefined
  /** A token that changes when the task changes in a way that could make a proposal wrong. Undefined: no such task. */
  revision(workspaceId: string, sessionId: string): string | undefined
  hasGoal(workspaceId: string, sessionId: string): boolean | undefined
  online(workspaceId: string): boolean
  label(workspaceId: string): string | undefined
  title(workspaceId: string, sessionId: string): string | undefined
}

export type ActionExecutor = (context: AuthenticatedContext, params: Record<string, unknown>) => Promise<HubCommandResult>

/**
 * The supervisor principal has no stored grants. For the duration of one
 * confirmed dispatch it is given the confirming human's delegated scopes, and
 * only inside that call. Outside it the supervisor can do nothing through the
 * router (fail closed).
 */
export class SupervisorGrantOverlay {
  private readonly storage = new AsyncLocalStorage<readonly HubGrant[]>()

  run<T>(grants: readonly HubGrant[], fn: () => T): T {
    return this.storage.run(grants, fn)
  }

  current(): readonly HubGrant[] {
    return this.storage.getStore() ?? []
  }

  /** The identity view to give the command router. */
  wrapIdentity(identity: Pick<HubIdentityPort, 'grantsFor' | 'audit'>): Pick<HubIdentityPort, 'grantsFor' | 'audit'> {
    return {
      grantsFor: principalId => (principalId === SUPERVISOR_PRINCIPAL_ID ? this.current() : identity.grantsFor(principalId)),
      audit: record => identity.audit(record),
    }
  }
}

export interface ActionServiceOptions {
  readonly db: Database
  readonly threads: Pick<SupervisorThreads, 'append' | 'emit' | 'get'>
  readonly identity: Pick<HubIdentityPort, 'grantsFor' | 'audit'>
  readonly targets: ActionTargets
  readonly execute: ActionExecutor
  readonly overlay: SupervisorGrantOverlay
  /** Throws when the person may not use the thread (not_found for someone else's private thread). */
  readonly access: (threadId: string, principal: UserPrincipal) => void
  readonly clock?: () => number
  readonly expiryMs?: number
  readonly maxPendingPerThread?: number
  /** A repeat steer of the same target inside this window is refused. Default 5 s. */
  readonly steerCooldownMs?: number
  /** `model.set` is offered only when the runtime guarantees the change is session-only. */
  readonly modelSetSessionOnly?: boolean
  /**
   * XERXES Autopilot: the owner an action proposed by `requester` in this workspace runs for,
   * without a Confirm tap. Undefined: confirm as usual.
   */
  readonly autopilot?: (input: {
    readonly workspaceId: string; readonly threadId: string; readonly requester: UserPrincipal
    readonly trigger: boolean; readonly othersInContext: boolean
  }) => AutopilotGrant | undefined
  /** Per-workspace cap on autonomous actions. Past it, actions become cards. */
  readonly autopilotBudget?: { take(workspaceId: string): AutopilotBudgetTake }
  /**
   * Posts an Autopilot notice to the shared thread, limited to people who can read the
   * target (and the owner). Absent: the notice goes to the action's own thread.
   */
  readonly autopilotNotice?: (input: { readonly workspaceId: string; readonly sessionId?: string; readonly ownerId: string; readonly text: string }) => void
  /** Minimal-text push to the owner about an autonomous action. Best effort. */
  readonly push?: (userId: string, message: { readonly title: string; readonly body: string; readonly url: string }) => Promise<unknown>
  /** XERXES Watch: a watch wake proposed an action that waits for a Confirm (Autopilot is not active). Best effort. */
  readonly onWatchCard?: (action: ActionRecord) => void
  readonly onError?: (error: unknown, context: string) => void
}

interface ActionRow {
  id: string; thread_id: string; tool: string; workspace_id: string; session_id: string | null; args: string; args_hash: string
  summary: string; targets: string; instance_id: string | null; target_revision: string | null; requested_by: string
  requested_by_name: string; idempotency_key: string; state: string; confirmed_by: string | null; confirmed_by_name: string | null
  outcome: string | null; error_code: string | null; error_message: string | null; created_at: number; expires_at: number; updated_at: number
  auto: number
}

const STATES: ReadonlySet<string> = new Set(['pending', 'confirming', 'accepted', 'rejected', 'unknown', 'cancelled', 'expired'])

function clip(text: string, max: number): string {
  const flat = text.replace(/\s+/g, ' ').trim()
  return flat.length <= max ? flat : `${flat.slice(0, Math.max(0, max - 1))}…`
}

function toRecord(row: ActionRow): ActionRecord {
  return {
    id: row.id,
    threadId: row.thread_id,
    tool: row.tool as ActionTool,
    workspaceId: row.workspace_id,
    ...(row.session_id === null ? {} : { sessionId: row.session_id }),
    args: JSON.parse(row.args) as JsonObject,
    argsHash: row.args_hash,
    summary: row.summary,
    targets: JSON.parse(row.targets) as string[],
    ...(row.instance_id === null ? {} : { instanceId: row.instance_id }),
    ...(row.target_revision === null ? {} : { targetRevision: row.target_revision }),
    requestedBy: row.requested_by,
    requestedByName: row.requested_by_name,
    idempotencyKey: row.idempotency_key,
    state: STATES.has(row.state) ? (row.state as ActionState) : 'unknown',
    ...(row.confirmed_by === null ? {} : { confirmedBy: row.confirmed_by }),
    ...(row.confirmed_by_name === null ? {} : { confirmedByName: row.confirmed_by_name }),
    ...(row.outcome === null ? {} : { outcome: JSON.parse(row.outcome) as Record<string, unknown> }),
    ...(row.error_code === null ? {} : { errorCode: row.error_code }),
    ...(row.error_message === null ? {} : { errorMessage: row.error_message }),
    createdAt: row.created_at,
    expiresAt: row.expires_at,
    updatedAt: row.updated_at,
    auto: row.auto === 1,
  }
}

/** The card as the clients' `parseConfirmation` reads it. */
export function actionToWire(action: ActionRecord): Record<string, unknown> {
  const state = action.state === 'accepted' ? 'confirmed'
    : action.state === 'rejected' || action.state === 'unknown' ? 'failed'
    : action.state
  const message = action.state === 'unknown' ? UNKNOWN_OUTCOME_MESSAGE : action.state === 'rejected' ? action.errorMessage : undefined
  return {
    id: action.id,
    confirmation_id: action.id,
    action_id: action.id,
    thread_id: action.threadId,
    tool: action.tool,
    summary: action.summary,
    targets: [...action.targets],
    requested_by: action.requestedByName,
    ...(action.confirmedByName === undefined ? {} : { confirmed_by: action.confirmedByName }),
    ...(action.auto ? { auto: true } : {}),
    expires_at: new Date(action.expiresAt).toISOString(),
    state,
    ...(message === undefined ? {} : { message }),
    ...(action.outcome === undefined ? {} : { outcome: action.outcome }),
  }
}

interface StepResult {
  readonly state: 'accepted' | 'rejected' | 'unknown'
  readonly result?: Readonly<Record<string, unknown>>
  readonly error?: { readonly code: string; readonly message: string }
}
interface DispatchOutcome {
  readonly state: 'accepted' | 'rejected' | 'unknown'
  readonly detail: Record<string, unknown>
  readonly error?: { readonly code: string; readonly message: string }
}

function sha256(text: string): string {
  return createHash('sha256').update(text).digest('hex')
}

function stringArg(args: JsonObject, key: string): string | undefined {
  const value = args[key]
  return typeof value === 'string' ? value : undefined
}

export class ActionService {
  private readonly db: Database
  private readonly clock: () => number

  constructor(private readonly options: ActionServiceOptions) {
    this.db = options.db
    this.clock = options.clock ?? Date.now
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS hub_supervisor_actions (
        id TEXT PRIMARY KEY, thread_id TEXT NOT NULL, tool TEXT NOT NULL, workspace_id TEXT NOT NULL, session_id TEXT,
        args TEXT NOT NULL, args_hash TEXT NOT NULL, summary TEXT NOT NULL, targets TEXT NOT NULL,
        instance_id TEXT, target_revision TEXT, requested_by TEXT NOT NULL, requested_by_name TEXT NOT NULL,
        idempotency_key TEXT NOT NULL UNIQUE, state TEXT NOT NULL, confirmed_by TEXT, confirmed_by_name TEXT,
        response_id TEXT, outcome TEXT, error_code TEXT, error_message TEXT,
        created_at INTEGER NOT NULL, expires_at INTEGER NOT NULL, updated_at INTEGER NOT NULL
      );
      CREATE INDEX IF NOT EXISTS hub_supervisor_actions_thread ON hub_supervisor_actions(thread_id, created_at);
    `)
    // Databases created before Autopilot have no `auto` column; add it once.
    const columns = this.db.query<{ name: string }, []>('PRAGMA table_info(hub_supervisor_actions)').all()
    if (!columns.some(column => column.name === 'auto')) {
      this.db.exec('ALTER TABLE hub_supervisor_actions ADD COLUMN auto INTEGER NOT NULL DEFAULT 0')
    }
    // A dispatch that was in flight when the Hub stopped has an unknowable outcome. Never replay it.
    this.db.query(`UPDATE hub_supervisor_actions SET state = 'unknown', error_code = 'unavailable',
      error_message = 'The Hub restarted before the outcome was recorded.', updated_at = ? WHERE state = 'confirming'`).run(this.clock())
  }

  // ── lookup ─────────────────────────────────────────────────────────

  get(id: string): ActionRecord | undefined {
    const row = this.db.query<ActionRow, [string]>('SELECT * FROM hub_supervisor_actions WHERE id = ?').get(id)
    return row ? toRecord(row) : undefined
  }

  private load(id: string): ActionRecord {
    const action = this.get(id)
    if (!action) throw new HubProtocolError('not_found', 'no such confirmation')
    return action
  }

  /** Cards the thread still shows: waiting ones, plus anything settled in the last half hour. */
  cardsFor(threadId: string): ActionRecord[] {
    const now = this.clock()
    const rows = this.db.query<ActionRow, [string, number, number]>(
      `SELECT * FROM hub_supervisor_actions WHERE thread_id = ? AND (state IN ('pending','confirming') OR updated_at > ?) ORDER BY created_at DESC LIMIT ?`,
    ).all(threadId, now - 30 * 60 * 1000, 20)
    return rows.map(toRecord).reverse()
  }

  pendingCount(threadId: string): number {
    return this.db.query<{ n: number }, [string, number]>("SELECT COUNT(*) AS n FROM hub_supervisor_actions WHERE thread_id = ? AND state = 'pending' AND expires_at > ?")
      .get(threadId, this.clock())?.n ?? 0
  }

  // ── transitions ────────────────────────────────────────────────────

  /** Atomic compare-and-set. Exactly one caller wins a given transition. */
  private transition(id: string, from: readonly ActionState[], to: ActionState, patch: { confirmedBy?: string; confirmedByName?: string; responseId?: string; auto?: boolean } = {}): boolean {
    const placeholders = from.map(() => '?').join(',')
    const result = this.db.query(
      `UPDATE hub_supervisor_actions SET state = ?, confirmed_by = COALESCE(?, confirmed_by), confirmed_by_name = COALESCE(?, confirmed_by_name),
       response_id = COALESCE(?, response_id), auto = MAX(auto, ?), updated_at = ? WHERE id = ? AND state IN (${placeholders})`,
    ).run(to, patch.confirmedBy ?? null, patch.confirmedByName ?? null, patch.responseId ?? null, patch.auto === true ? 1 : 0, this.clock(), id, ...from)
    return result.changes === 1
  }

  private finish(id: string, state: ActionState, outcome: Record<string, unknown> | undefined, error?: { code: string; message: string }): ActionRecord {
    this.db.query('UPDATE hub_supervisor_actions SET state = ?, outcome = ?, error_code = ?, error_message = ?, updated_at = ? WHERE id = ?').run(
      state, outcome === undefined ? null : JSON.stringify(outcome), error?.code ?? null, error?.message ?? null, this.clock(), id,
    )
    return this.load(id)
  }

  private announce(action: ActionRecord, notice?: string): void {
    const { threads } = this.options
    if (notice !== undefined) {
      const { message } = threads.append({ threadId: action.threadId, role: 'system', text: notice, causalMessageId: action.id })
      threads.emit(action.threadId, 'supervisor.thread_event', { message: messageToWire(message) })
    }
    threads.emit(action.threadId, 'supervisor.confirmation', { confirmation: actionToWire(action) })
  }

  private auditEntry(action: ActionRecord, stage: Parameters<SupervisorAudit['record']>[0]['stage'], decision: 'allowed' | 'denied', outcome: Parameters<SupervisorAudit['record']>[0]['outcome'], confirmedBy?: string) {
    return {
      stage,
      actor: supervisorActor(action.requestedBy, confirmedBy),
      actionId: action.id,
      tool: action.tool,
      workspaceId: action.workspaceId,
      ...(action.sessionId === undefined ? {} : { sessionId: action.sessionId }),
      requestHash: action.argsHash,
      decision,
      outcome,
    } as const
  }

  private get audit(): SupervisorAudit {
    return new SupervisorAudit(this.options.identity, this.options.onError)
  }

  /** Expire every waiting card whose time is up. */
  async sweep(): Promise<number> {
    const rows = this.db.query<ActionRow, [number]>("SELECT * FROM hub_supervisor_actions WHERE state = 'pending' AND expires_at <= ?").all(this.clock())
    let count = 0
    for (const row of rows) {
      if (!this.transition(row.id, ['pending'], 'expired')) continue
      count += 1
      const action = this.load(row.id)
      await this.audit.recordBestEffort(this.auditEntry(action, 'expired', 'allowed', 'n/a'))
      this.announce(action)
    }
    return count
  }

  // ── intent ─────────────────────────────────────────────────────────

  async createIntent(input: {
    readonly threadId: string
    readonly tool: ActionTool
    readonly args: JsonObject
    readonly requester: UserPrincipal
    readonly reader: ScopeReader
    readonly turn?: ActionTurnInfo
  }): Promise<ActionRecord> {
    const { tool, args, requester, reader } = input
    const { targets } = this.options
    const workspaceId = stringArg(args, 'workspace_id')
    if (workspaceId === undefined) throw new Error('workspace_id is required')
    const sessionId = stringArg(args, 'task_id')
    const label = targets.label(workspaceId)
    if (label === undefined) throw new Error(`unknown workspace ${JSON.stringify(workspaceId)}`)
    if (tool === 'model.set' && this.options.modelSetSessionOnly !== true) {
      throw new Error('unsupported: this runtime cannot guarantee a session-only model change, so XERXES cannot propose one')
    }
    // No self-targeting and no recursive supervisors.
    for (const key of ['preset', 'agent_id'] as const) {
      const value = stringArg(args, key)
      if (value !== undefined && SELF_NAMES.has(value.toLowerCase())) throw new Error('refused: the supervisor cannot target or start a supervisor')
    }
    if (sessionId !== undefined && isSupervisorThreadId(sessionId)) throw new Error('refused: that is a supervisor thread, not a task')
    if (sessionId !== undefined && targets.revision(workspaceId, sessionId) === undefined) throw new Error(`no such task ${JSON.stringify(sessionId)} in this workspace`)
    if (sessionId === undefined && tool !== 'task.start') throw new Error('task_id is required')

    const scope = ACTION_SCOPE[tool]
    const accessTarget = { workspaceId, ...(sessionId === undefined ? {} : { sessionId }) }
    if (!reader.canRead(accessTarget) || !reader.scopes(accessTarget).has(scope)) {
      throw new SupervisorDenied(`you cannot ${tool} here: ${scope} is required on this target for everyone in this thread`)
    }
    if (tool === 'goal.set' && sessionId !== undefined) {
      const expected = args.expected_revision
      const hasGoal = targets.hasGoal(workspaceId, sessionId)
      if (hasGoal !== undefined && ((expected === 0 && hasGoal) || (typeof expected === 'number' && expected > 0 && !hasGoal))) {
        throw new Error('stale: expected_revision does not match the current goal; read the task digest and propose again')
      }
    }

    const argsJson = canonicalJson(args)
    const argsHash = supervisorRequestHash(tool, workspaceId, sessionId, args)
    const now = this.clock()
    const existing = this.db.query<ActionRow, [string, string, string]>(
      "SELECT * FROM hub_supervisor_actions WHERE thread_id = ? AND args_hash = ? AND tool = ? AND state = 'pending' AND expires_at > " + now,
    ).get(input.threadId, argsHash, tool)
    if (existing) return toRecord(existing)
    if (this.pendingCount(input.threadId) >= (this.options.maxPendingPerThread ?? 10)) {
      throw new Error('too many confirmations are waiting in this thread; ask the team to confirm or cancel some first')
    }
    if ((tool === 'task.steer' || tool === 'agent.message') && sessionId !== undefined) {
      const cooldown = this.options.steerCooldownMs ?? 5000
      const burst = this.db.query<{ n: number }, [string, string, number]>(
        "SELECT COUNT(*) AS n FROM hub_supervisor_actions WHERE workspace_id = ? AND session_id = ? AND tool IN ('task.steer','agent.message') AND state != 'cancelled' AND created_at > ?",
      ).get(workspaceId, sessionId, now - cooldown)?.n ?? 0
      if (burst > 0) throw new Error('this task was just steered; wait a few seconds before proposing another message')
    }

    const id = `act-${randomUUID()}`
    const title = sessionId === undefined ? undefined : targets.title(workspaceId, sessionId)
    const record: ActionRecord = {
      id,
      threadId: input.threadId,
      tool,
      workspaceId,
      ...(sessionId === undefined ? {} : { sessionId }),
      args,
      argsHash,
      summary: summarize(tool, args, title),
      targets: [sessionId === undefined ? label : `${label} / ${title ?? sessionId}`],
      ...(targets.instanceId(workspaceId) === undefined ? {} : { instanceId: targets.instanceId(workspaceId)! }),
      ...(sessionId === undefined ? {} : { targetRevision: targets.revision(workspaceId, sessionId)! }),
      requestedBy: requester.id,
      requestedByName: requester.displayName,
      idempotencyKey: randomUUID(),
      state: 'pending',
      createdAt: now,
      expiresAt: now + (this.options.expiryMs ?? DEFAULT_ACTION_EXPIRY_MS),
      updatedAt: now,
      auto: false,
    }
    // XERXES Autopilot: decide before anything is stored whether this runs without a Confirm tap.
    const grant = this.options.autopilot?.({
      workspaceId, threadId: input.threadId, requester,
      // Any Hub-started turn counts as a wake: it can never fall into the "owner alone" rule.
      trigger: input.turn?.trigger !== undefined, othersInContext: input.turn?.othersInContext ?? true,
    })
    const ownerGrants = grant === undefined ? [] : this.options.identity.grantsFor(grant.owner.id)
    // A Hub wake-up may act without Confirm only on the exact task it is about. No target: nothing auto-runs.
    const wakeTarget = input.turn?.trigger === undefined ? undefined : input.turn.target
    const offTarget = input.turn?.trigger !== undefined
      && (wakeTarget === undefined || wakeTarget.workspaceId !== workspaceId || wakeTarget.sessionId !== sessionId)
    let held: { readonly text: string; readonly push: string } | undefined
    let eligible = false
    if (grant !== undefined) {
      const owner = grant.owner
      if (grant.wake && !WAKE_AUTO_TOOLS.has(tool)) {
        held = { text: `Autopilot needs you: confirm ${record.summary}`, push: 'Autopilot needs you to confirm an action.' }
      } else if (offTarget) {
        held = { text: `Autopilot needs you: confirm ${record.summary} (not the task this wake-up is about)`, push: 'Autopilot needs you to confirm an action.' }
      } else if (!hasScope(owner, ownerGrants, accessTarget, scope, now)) {
        held = { text: `Autopilot needs you: ${owner.displayName} cannot ${tool} here, so this waits for a person to confirm: ${record.summary}`, push: 'Autopilot needs you.' }
      } else {
        eligible = true
      }
    }

    // Fail closed: no audit record, no card.
    try {
      await this.audit.record(this.auditEntry(record, 'intent', 'allowed', 'n/a'))
    } catch (error) {
      this.options.onError?.(error, 'supervisor audit intent')
      throw new Error('the audit log is unavailable; no action was proposed')
    }
    let auto = false
    if (eligible && grant !== undefined) {
      const budget = this.options.autopilotBudget?.take(workspaceId) ?? { ok: true }
      if (budget.ok) auto = true
      else if (budget.notify) {
        held = {
          text: `Autopilot needs you: it reached its limit of ${budget.limit} autonomous actions this hour in ${label}, so actions wait for a person to confirm: ${record.summary}`,
          push: 'Autopilot needs you.',
        }
      }
    }
    // `auto` is stored with the card, and an automatic card is never announced as pending:
    // nobody sees Confirm/Cancel on it, and the compare-and-set below still has a single winner.
    this.db.query(`INSERT INTO hub_supervisor_actions (id, thread_id, tool, workspace_id, session_id, args, args_hash, summary, targets,
      instance_id, target_revision, requested_by, requested_by_name, idempotency_key, state, created_at, expires_at, updated_at, auto)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?)`).run(
      record.id, record.threadId, tool, workspaceId, sessionId ?? null, argsJson, argsHash, record.summary, JSON.stringify(record.targets),
      record.instanceId ?? null, record.targetRevision ?? null, requester.id, requester.displayName, record.idempotencyKey, now, record.expiresAt, now,
      auto ? 1 : 0,
    )
    if (auto && grant !== undefined) return await this.confirmAndDispatch(record.id, grant.owner, ownerGrants, undefined, true)
    this.announce(record)
    if (grant === undefined && input.turn?.trigger === 'watch') {
      try { this.options.onWatchCard?.(record) } catch (error) { this.options.onError?.(error, 'watch card') }
    }
    if (held !== undefined && grant !== undefined) {
      this.autopilotNote(record, grant.owner.id, held.text)
      await this.pushOwner(grant.owner.id, held.push)
    }
    return record
  }

  /** An Autopilot notice: in the action's own thread unless it is the shared one, and to the shared thread for readers of the target. */
  private autopilotNote(action: ActionRecord, ownerId: string, text: string): void {
    const { threads } = this.options
    const hook = this.options.autopilotNotice
    if (action.threadId !== SHARED_THREAD_ID || hook === undefined) {
      const { message } = threads.append({ threadId: action.threadId, role: 'system', text, causalMessageId: action.id })
      threads.emit(action.threadId, 'supervisor.thread_event', { message: messageToWire(message) })
    }
    if (hook === undefined) return
    try {
      hook({ workspaceId: action.workspaceId, ...(action.sessionId === undefined ? {} : { sessionId: action.sessionId }), ownerId, text })
    } catch (error) {
      this.options.onError?.(error, 'autopilot notice')
    }
  }

  private async pushOwner(ownerId: string, body: string): Promise<void> {
    if (this.options.push === undefined) return
    try {
      // Lock-screen text stays minimal: no task title, no arguments.
      await this.options.push(ownerId, { title: 'XERXES', body, url: '/xerxes' })
    } catch (error) {
      this.options.onError?.(error, 'autopilot action push')
    }
  }

  // ── respond ────────────────────────────────────────────────────────

  async respond(context: AuthenticatedContext, params: Readonly<Record<string, unknown>>): Promise<ActionRecord> {
    const allowed = ['confirmation_id', 'action_id', 'decision', 'response_id', 'thread_id']
    for (const key of Object.keys(params)) {
      if (!allowed.includes(key)) throw new HubProtocolError('invalid_request', `unknown parameter ${JSON.stringify(key.slice(0, 40))}`)
    }
    const id = typeof params.confirmation_id === 'string' ? params.confirmation_id : params.action_id
    if (typeof id !== 'string' || id.length === 0 || id.length > 200) throw new HubProtocolError('invalid_request', 'confirmation_id is required')
    const decision = params.decision
    if (decision !== 'confirm' && decision !== 'cancel') throw new HubProtocolError('invalid_request', "decision must be 'confirm' or 'cancel'")
    const responseId = typeof params.response_id === 'string' ? params.response_id.slice(0, 200) : undefined

    const action = this.load(id)
    // Agents are refused structurally, before anything else is looked at.
    if (context.principal.kind !== 'user') {
      await this.audit.recordBestEffort({ ...this.auditEntry(action, 'denied', 'denied', 'rejected'), actor: context.actor })
      throw new HubProtocolError('forbidden', 'only a signed-in person can confirm or cancel an action')
    }
    const human = context.principal
    this.options.access(action.threadId, human)
    if (typeof params.thread_id === 'string' && params.thread_id !== action.threadId) throw new HubProtocolError('not_found', 'no such confirmation')

    const target = { workspaceId: action.workspaceId, ...(action.sessionId === undefined ? {} : { sessionId: action.sessionId }) }
    const grants = this.options.identity.grantsFor(human.id)
    const now = this.clock()
    if (decision === 'confirm' && !hasScope(human, grants, target, ACTION_SCOPE[action.tool], now)) {
      await this.audit.recordBestEffort({ ...this.auditEntry(action, 'denied', 'denied', 'rejected', human.id) })
      throw new HubProtocolError('forbidden', `not permitted: confirming ${action.tool} requires ${ACTION_SCOPE[action.tool]} on this target`)
    }

    await this.sweep()
    const current = this.load(id)
    if (current.state === 'expired') throw new HubProtocolError('conflict', 'This confirmation expired. Ask XERXES again if you still want it.')
    // Replays and lost races report the true state; nothing runs twice.
    if (current.state !== 'pending') return current

    if (decision === 'cancel') {
      if (!this.transition(id, ['pending'], 'cancelled', { confirmedBy: human.id, confirmedByName: human.displayName, ...(responseId === undefined ? {} : { responseId }) })) return this.load(id)
      const cancelled = this.load(id)
      await this.audit.recordBestEffort({ ...this.auditEntry(cancelled, 'cancel', 'allowed', 'n/a', human.id), actor: supervisorActor(cancelled.requestedBy) })
      this.announce(cancelled, `Cancelled by ${human.displayName}: ${cancelled.summary}`)
      return cancelled
    }

    return this.confirmAndDispatch(id, human, grants, responseId, false)
  }

  /**
   * The single confirm path shared by a person's Confirm tap and XERXES Autopilot:
   * single-winner compare-and-set, fail-closed audit, stale checks at execution,
   * dispatch under the confirmer's delegated scopes, settle. Never retries `unknown`.
   */
  private async confirmAndDispatch(id: string, human: UserPrincipal, grants: readonly HubGrant[], responseId: string | undefined, auto: boolean): Promise<ActionRecord> {
    // Only one confirmer wins this compare-and-set; everyone else sees the winner's state.
    if (!this.transition(id, ['pending'], 'confirming', { confirmedBy: human.id, confirmedByName: human.displayName, ...(responseId === undefined ? {} : { responseId }), auto })) {
      const lost = this.load(id)
      if (lost.state === 'expired') throw new HubProtocolError('conflict', 'This confirmation expired. Ask XERXES again if you still want it.')
      return lost
    }
    const confirming = this.load(id)
    this.announce(confirming)

    try {
      await this.audit.record(this.auditEntry(confirming, 'confirm', 'allowed', 'n/a', human.id))
    } catch (error) {
      this.options.onError?.(error, 'supervisor audit confirm')
      return this.settle(this.finish(id, 'rejected', { reason: 'audit_unavailable' }, { code: 'unavailable', message: 'The audit log is unavailable, so nothing was sent. Ask XERXES again.' }), human)
    }

    const stale = this.staleReason(confirming)
    if (stale !== undefined) {
      return this.settle(this.finish(id, 'rejected', { reason: 'stale' }, { code: 'conflict', message: stale }), human)
    }

    try {
      await this.audit.record(this.auditEntry(confirming, 'dispatch', 'allowed', 'n/a', human.id))
    } catch (error) {
      this.options.onError?.(error, 'supervisor audit dispatch')
      return this.settle(this.finish(id, 'rejected', { reason: 'audit_unavailable' }, { code: 'unavailable', message: 'The audit log is unavailable, so nothing was sent. Ask XERXES again.' }), human)
    }

    const outcome = await this.dispatch(confirming, human, grants)
    return this.settle(this.finish(id, outcome.state, outcome.detail, outcome.error), human)
  }

  /** Status of one card for someone allowed to see the thread. */
  async status(context: AuthenticatedContext, params: Readonly<Record<string, unknown>>): Promise<ActionRecord> {
    for (const key of Object.keys(params)) {
      if (!['confirmation_id', 'action_id', 'thread_id'].includes(key)) throw new HubProtocolError('invalid_request', `unknown parameter ${JSON.stringify(key.slice(0, 40))}`)
    }
    const id = typeof params.confirmation_id === 'string' ? params.confirmation_id : params.action_id
    if (typeof id !== 'string' || id.length === 0 || id.length > 200) throw new HubProtocolError('invalid_request', 'confirmation_id is required')
    if (context.principal.kind !== 'user') throw new HubProtocolError('forbidden', 'only a signed-in person can read action status')
    const action = this.load(id)
    this.options.access(action.threadId, context.principal)
    await this.sweep()
    return this.load(id)
  }

  private async settle(action: ActionRecord, human: UserPrincipal): Promise<ActionRecord> {
    const outcome = action.state === 'accepted' ? 'accepted' : action.state === 'rejected' ? 'rejected' : 'unknown'
    await this.audit.recordBestEffort(this.auditEntry(action, 'outcome', 'allowed', outcome, human.id))
    if (action.auto) {
      const notice = action.state === 'accepted' ? `Autopilot did: ${action.summary} (on behalf of ${human.displayName})`
        : action.state === 'rejected' ? `Autopilot needs you: this was not done: ${action.summary} ${action.errorMessage ?? ''}`.trim()
        : `Autopilot needs you: the outcome is unknown: ${action.summary} Check the task; XERXES will not retry it.`
      this.announce(action)
      // Only readers of the target see what Autopilot did.
      this.autopilotNote(action, human.id, notice)
      await this.pushOwner(human.id, action.state === 'accepted' ? 'Autopilot acted for you. Open XERXES to see what it did.' : 'Autopilot needs you.')
      return action
    }
    const notice = action.state === 'accepted' ? `Done: ${action.summary} (confirmed by ${human.displayName}).`
      : action.state === 'rejected' ? `Not done: ${action.summary} ${action.errorMessage ?? ''}`.trim()
      : `Outcome unknown: ${action.summary} Check the task before trying again; XERXES will not retry it.`
    this.announce(action, notice)
    return action
  }

  private staleReason(action: ActionRecord): string | undefined {
    const { targets } = this.options
    if (this.clock() >= action.expiresAt) return 'This confirmation expired.'
    if (!targets.online(action.workspaceId)) return 'The workspace is not online. Nothing was sent. Ask XERXES again when it is back.'
    if (targets.instanceId(action.workspaceId) !== action.instanceId) return 'The workspace runtime restarted since this was proposed. Nothing was sent. Ask XERXES again.'
    if (action.sessionId !== undefined) {
      const revision = targets.revision(action.workspaceId, action.sessionId)
      if (revision === undefined) return 'The task no longer exists. Nothing was sent.'
      if (revision !== action.targetRevision) return 'The task changed since this was proposed. Nothing was sent. Ask XERXES again.'
      if (action.tool === 'goal.set') {
        const expected = action.args.expected_revision
        const hasGoal = targets.hasGoal(action.workspaceId, action.sessionId)
        if (hasGoal !== undefined && ((expected === 0 && hasGoal) || (typeof expected === 'number' && expected > 0 && !hasGoal))) {
          return 'The goal changed since this was proposed. Nothing was sent. Ask XERXES again.'
        }
      }
    }
    return undefined
  }

  // ── dispatch ───────────────────────────────────────────────────────

  private async dispatch(action: ActionRecord, confirmer: UserPrincipal, confirmerGrants: readonly HubGrant[]): Promise<DispatchOutcome> {
    const target = { workspaceId: action.workspaceId, ...(action.sessionId === undefined ? {} : { sessionId: action.sessionId }) }
    const scopes = delegatedScopes(confirmer, confirmerGrants, target, this.clock())
    const resource = action.sessionId === undefined
      ? ({ type: 'workspace', workspaceId: action.workspaceId } as const)
      : ({ type: 'session', workspaceId: action.workspaceId, sessionId: action.sessionId } as const)
    const delegated: HubGrant[] = [{ principalId: SUPERVISOR_PRINCIPAL_ID, resource, effect: 'allow', scopes: [...scopes] }]
    const principal: HubPrincipal = { kind: 'agent', id: SUPERVISOR_PRINCIPAL_ID, displayName: SUPERVISOR_DISPLAY_NAME, onBehalfOf: action.requestedBy }
    const context: AuthenticatedContext = {
      principal,
      grants: delegated,
      browserSessionId: `supervisor:${action.id}`,
      actor: supervisorActor(action.requestedBy, confirmer.id),
    }
    let step = 0
    const send = async (command: HubCommandAction, sessionId: string | undefined, args: Record<string, unknown>): Promise<StepResult> => {
      const params = {
        workspace_id: action.workspaceId,
        ...(sessionId === undefined ? {} : { session_id: sessionId }),
        command_id: `${action.idempotencyKey}:${step++}`,
        action: command,
        args,
      }
      try {
        const result = await this.options.overlay.run(delegated, () => this.options.execute(context, params))
        const state = result.state === 'accepted' || result.state === 'rejected' ? result.state : 'unknown'
        return {
          state,
          ...(result.result === undefined ? {} : { result: result.result }),
          ...(result.error === undefined ? {} : { error: { code: result.error.code, message: result.error.message } }),
        }
      } catch (error) {
        // The router refuses before sending anything when it throws a protocol error.
        if (error instanceof HubProtocolError) return { state: 'rejected', error: { code: error.code, message: error.message } }
        this.options.onError?.(error, 'supervisor dispatch')
        return { state: 'unknown', error: { code: 'internal', message: 'The dispatch failed after it may have been sent.' } }
      }
    }
    const done = (result: StepResult, detail: Record<string, unknown>): DispatchOutcome => ({
      state: result.state, detail, ...(result.error === undefined ? {} : { error: result.error }),
    })
    const args = action.args
    const text = stringArg(args, 'text') ?? ''

    switch (action.tool) {
      case 'task.start': {
        let sessionId = action.sessionId
        let opened = false
        if (sessionId === undefined) {
          const preset = stringArg(args, 'preset')
          const open = await send('session.open', undefined, preset === undefined ? {} : { agent_id: preset })
          if (open.state !== 'accepted') return done(open, { step: 'open' })
          const created = open.result?.session_id
          if (typeof created !== 'string' || created === '') {
            return { state: 'unknown', detail: { step: 'open' }, error: { code: 'internal', message: 'The new task id was not reported.' } }
          }
          sessionId = created
          opened = true
        }
        const submit = await send('turn.submit', sessionId, { text })
        return done(submit, { session_id: sessionId, session_opened: opened })
      }
      case 'task.steer': {
        const steer = await send('turn.steer', action.sessionId, { text })
        return done(steer, { delivered: steer.result?.delivered === true })
      }
      case 'task.stop': {
        // Pause goal continuation first so the stop is not undone by the next round.
        let paused = false
        if (scopes.has('goal.write')) {
          const pause = await send('goal.set', action.sessionId, { input: 'pause' })
          paused = pause.state === 'accepted'
        }
        const cancel = await send('turn.cancel', action.sessionId, {})
        const cascade = args.cascade === true
        return done(cancel, {
          requested: true,
          stopped: cancel.state === 'accepted' && cancel.result?.cancelled === true,
          goal_paused: paused,
          cascade_requested: cascade,
          // The Hub has no allowlisted child-interrupt route; this is reported, never claimed.
          cascade_applied: false,
        })
      }
      case 'agent.message': {
        const message = await send('subagent.message', action.sessionId, { task: stringArg(args, 'agent_id') ?? '', message: text })
        return done(message, { delivered: message.result?.delivered === true })
      }
      case 'goal.set': {
        const goal = await send('goal.set', action.sessionId, { input: goalInput(args) })
        return done(goal, {})
      }
      case 'model.set': {
        const profile = stringArg(args, 'provider_profile')
        const model = await send('model.set', action.sessionId, { model: stringArg(args, 'model') ?? '', ...(profile === undefined ? {} : { provider_profile: profile }) })
        return done(model, {})
      }
    }
  }
}

/** The goal text sent to the runtime: the objective, with the criteria folded in as plain lines. */
export function goalInput(args: JsonObject): string {
  const objective = stringArg(args, 'objective') ?? ''
  const criteria = Array.isArray(args.criteria) ? args.criteria : []
  const lines = criteria.flatMap(item => {
    if (typeof item !== 'object' || item === null || Array.isArray(item)) return []
    const description = (item as Record<string, unknown>).description
    return typeof description === 'string' ? [`- ${description}`] : []
  })
  const text = lines.length === 0 ? objective : `${objective}\n\nDone when:\n${lines.join('\n')}`
  return text.length <= 4000 ? text : text.slice(0, 4000)
}

function summarize(tool: ActionTool, args: JsonObject, title: string | undefined): string {
  const task = title === undefined ? 'the task' : `“${clip(title, 60)}”`
  const text = clip(stringArg(args, 'text') ?? '', 160)
  switch (tool) {
    case 'task.start': return args.task_id === undefined ? `Start a new task: ${text}` : `Start a new turn in ${task}: ${text}`
    case 'task.steer': return `Steer ${task}: ${text}`
    case 'task.stop': return `Pause the goal and stop the current turn of ${task}${args.cascade === true ? ' and ask its agents to stop' : ''}.`
    case 'agent.message': return `Message agent ${clip(stringArg(args, 'agent_id') ?? '', 60)} in ${task}: ${text}`
    case 'goal.set': return `Set the goal of ${task}: ${clip(stringArg(args, 'objective') ?? '', 160)}`
    case 'model.set': return `Change the model of ${task} to ${clip(stringArg(args, 'model') ?? '', 80)} (${clip(stringArg(args, 'provider_profile') ?? '', 60)}), for this task only.`
  }
}

export type { SupervisorEventType }
