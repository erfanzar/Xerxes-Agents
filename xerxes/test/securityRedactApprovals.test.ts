// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdtempSync, readFileSync, rmSync, statSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { expect, test } from 'bun:test'

import { AuditEmitter, InMemoryCollector, TurnStartEvent } from '../src/audit/index.js'
import { DaemonInteractionBoard } from '../src/daemon/interactions.js'
import {
  ApprovalRecord,
  ApprovalScope,
  ApprovalStore,
  approvalArgumentsHash,
} from '../src/security/approvals.js'
import { REDACTED, redactPayload, redactString } from '../src/security/redact.js'
import type { PermissionRequest } from '../src/streaming/events.js'

/** Stands in for an approvals file that was truncated, hand-edited, or is unreadable. */
class UnreadableApprovalStore extends ApprovalStore {
  override check(): boolean | undefined {
    throw new Error('approvals.json is not valid JSON')
  }
}

function execRequest(requestId: string, argv: readonly string[]): PermissionRequest {
  return {
    requestId,
    description: 'run a command',
    inputs: { argv: [...argv] },
    toolCall: {
      id: 'call-' + requestId,
      type: 'function',
      function: { name: 'exec_command', arguments: { argv: [...argv] } },
    },
  }
}

test('redaction removes credential, token, PII, and sensitive-field values without mutating input', () => {
  const text = 'api_key=abcdefgh12345678 Authorization: Bearer secret-token@example.com sk-abcdefghijklmnop user@example.com'
  expect(redactString(text)).not.toContain('abcdefgh12345678')
  expect(redactString(text)).not.toContain('user@example.com')

  const payload = { api_key: 'secret', nested: { authorization: 'Bearer no-leak', email: 'person@example.com' } }
  const redacted = redactPayload(payload) as Record<string, unknown>
  expect(redacted).toEqual({ api_key: REDACTED, nested: { authorization: REDACTED, email: REDACTED } })
  expect(payload.api_key).toBe('secret')
})

test('field redaction rewrites only the secret, never the separator or the surrounding prose', () => {
  // Each of these used to come back with its separator normalized to "=" and
  // any trailing quote orphaned, so a redacted log line no longer matched the
  // line the program emitted and read as corruption rather than redaction.
  expect(redactString('store the api_key: abcdefgh12345678 in Vault, not here'))
    .toBe('store the api_key: ' + REDACTED + ' in Vault, not here')
  expect(redactString('api-key="zzzzzzzzzzzz"')).toBe('api-key="' + REDACTED + '"')
  expect(redactString('Authorization:  Bearer   tok.en-value trailing'))
    .toBe('Authorization:  Bearer   ' + REDACTED + ' trailing')
  expect(redactString('set password = hunter2seventeen now')).toBe('set password = ' + REDACTED + ' now')
})

test('audit previews redact credentials before records reach collectors', () => {
  const collector = new InMemoryCollector()
  const emitter = new AuditEmitter({ collector })
  emitter.emitTurnStart({ prompt: 'Authorization: Bearer super-secret-token' })
  emitter.emitToolCallAttempt({ toolName: 'request', args: { api_key: 'should-not-appear' } })

  expect((collector.getEvents()[0] as TurnStartEvent).promptPreview).toContain(REDACTED)
  expect((collector.getEvents()[0] as TurnStartEvent).promptPreview).not.toContain('super-secret-token')
  expect(JSON.stringify(collector.getEvents()[1]?.toRecord())).not.toContain('should-not-appear')
})

test('approval records select the newest matching scope and hash arguments deterministically', () => {
  const store = new ApprovalStore({ now: () => new Date('2026-07-13T00:00:00.000Z') })
  const hash = approvalArgumentsHash({ b: 2, a: 1 })
  expect(hash).toBe(approvalArgumentsHash({ a: 1, b: 2 }))
  store.add({ toolName: 'Bash', scope: ApprovalScope.ONCE, granted: true, sessionId: 'one', argsHash: hash })
  store.add({ toolName: 'Bash', scope: ApprovalScope.SESSION, granted: false, sessionId: 'two' })
  store.add(new ApprovalRecord({ toolName: 'ReadFile', scope: ApprovalScope.ALWAYS, granted: true }))

  expect(store.check('Bash', 'one', hash)).toBeTrue()
  expect(store.check('Bash', 'one', 'other')).toBeUndefined()
  expect(store.check('Bash', 'two')).toBeFalse()
  expect(store.check('ReadFile', 'any')).toBeTrue()
  expect(store.clearSession('one')).toBe(1)
  expect(store.check('Bash', 'one', hash)).toBeUndefined()
})

test('always and session approvals only match the arguments they were granted for', () => {
  const store = new ApprovalStore({ now: () => new Date('2026-07-13T00:00:00.000Z') })
  const listing = approvalArgumentsHash({ argv: ['ls'] })
  const destructive = approvalArgumentsHash({ argv: ['rm', '-rf', '/'] })
  store.add({ toolName: 'exec_command', scope: ApprovalScope.ALWAYS, granted: true, argsHash: listing })
  store.add({ toolName: 'Bash', scope: ApprovalScope.SESSION, granted: true, sessionId: 'one', argsHash: listing })

  expect(store.check('exec_command', 'any', listing)).toBeTrue()
  expect(store.check('exec_command', 'any', destructive)).toBeUndefined()
  expect(store.check('Bash', 'one', listing)).toBeTrue()
  expect(store.check('Bash', 'one', destructive)).toBeUndefined()
})

test('records written before argument scoping keep granting tool-wide', () => {
  const store = new ApprovalStore({ now: () => new Date('2026-07-13T00:00:00.000Z') })
  store.add(new ApprovalRecord({ toolName: 'exec_command', scope: ApprovalScope.ALWAYS, granted: true }))
  store.add({ toolName: 'Bash', scope: ApprovalScope.SESSION, granted: false, sessionId: 'one' })

  expect(store.check('exec_command', 'any', approvalArgumentsHash({ argv: ['ls'] }))).toBeTrue()
  expect(store.check('exec_command', 'any', approvalArgumentsHash({ argv: ['rm', '-rf', '/'] }))).toBeTrue()
  expect(store.check('Bash', 'one', approvalArgumentsHash({ argv: ['ls'] }))).toBeFalse()
})

test('an always allow click authorizes only the argv the user saw', async () => {
  const store = new ApprovalStore()
  const board = new DaemonInteractionBoard({ approvalStore: store })
  const approved = board.permissionBroker('session-one').request(execRequest('exec-1', ['ls']))
  expect(board.respondPermission('exec-1', 'always')).toBe(true)
  await expect(approved).resolves.toBe('approve_for_session')

  const repeated = board.permissionBroker('session-two').request(execRequest('exec-2', ['ls']))
  await expect(repeated).resolves.toBe('approve')

  const escalated = board.permissionBroker('session-one').request(execRequest('exec-3', ['rm', '-rf', '/']))
  expect(board.pendingPermissionIds()).toEqual(['exec-3'])
  expect(board.respondPermission('exec-3', 'reject')).toBe(true)
  await expect(escalated).resolves.toBe('reject')
})

test('an approve-for-session grant does not carry to a different argv in the same session', async () => {
  const board = new DaemonInteractionBoard()
  const approved = board.permissionBroker('session-one').request(execRequest('exec-1', ['ls']))
  expect(board.respondPermission('exec-1', 'approve_for_session')).toBe(true)
  await expect(approved).resolves.toBe('approve_for_session')

  await expect(board.permissionBroker('session-one').request(execRequest('exec-2', ['ls']))).resolves.toBe('approve')

  const escalated = board.permissionBroker('session-one').request(execRequest('exec-3', ['rm', '-rf', '/']))
  expect(board.pendingPermissionIds()).toEqual(['exec-3'])
  expect(board.respondPermission('exec-3', 'reject')).toBe(true)
  await expect(escalated).resolves.toBe('reject')
})

test('an unreadable approvals file prompts the user instead of throwing through the permission path', async () => {
  const failures: unknown[] = []
  const board = new DaemonInteractionBoard({
    approvalStore: new UnreadableApprovalStore(),
    onApprovalStoreError: error => failures.push(error),
  })
  const pending = board.permissionBroker('session-one').request(execRequest('exec-1', ['ls']))

  expect(board.pendingPermissionIds()).toEqual(['exec-1'])
  expect(failures).toHaveLength(1)
  expect(board.respondPermission('exec-1', 'approve')).toBe(true)
  await expect(pending).resolves.toBe('approve')
})

test('always approvals persist atomically in the Python-readable snake-case format', () => {
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-approvals-'))
  const path = join(directory, 'nested', 'approvals.json')
  try {
    const store = new ApprovalStore({ persistencePath: path, now: () => new Date('2026-07-13T00:00:00.000Z') })
    store.add({ toolName: 'WriteFile', scope: ApprovalScope.ALWAYS, granted: true })
    store.add({ toolName: 'ReadFile', scope: ApprovalScope.SESSION, granted: false, sessionId: 'session' })
    expect(JSON.parse(readFileSync(path, 'utf8'))).toEqual([{
      tool_name: 'WriteFile', scope: 'always', granted: true, session_id: '', args_hash: '', created_at: '2026-07-13T00:00:00.000Z',
    }])
    expect(statSync(path).mode & 0o777).toBe(0o600)
    expect(new ApprovalStore({ persistencePath: path }).check('WriteFile', 'fresh')).toBeTrue()
  } finally {
    rmSync(directory, { force: true, recursive: true })
  }
})

test('always approvals merge with the on-disk file so concurrent daemons cannot clobber each other', () => {
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-approvals-merge-'))
  const path = join(directory, 'approvals.json')
  try {
    const first = new ApprovalStore({ persistencePath: path, now: () => new Date('2026-07-13T00:00:00.000Z') })
    const second = new ApprovalStore({ persistencePath: path, now: () => new Date('2026-07-13T00:00:01.000Z') })
    first.add({ toolName: 'WriteFile', scope: ApprovalScope.ALWAYS, granted: true })
    // `second` loaded before `first` flushed; its flush must union, not overwrite.
    second.add({ toolName: 'Bash', scope: ApprovalScope.ALWAYS, granted: false })

    const persisted = JSON.parse(readFileSync(path, 'utf8')) as Array<{ tool_name: string }>
    expect(persisted.map(record => record.tool_name).sort()).toEqual(['Bash', 'WriteFile'])
    expect(statSync(path).mode & 0o777).toBe(0o600)

    // Re-adding an identical decision stays deduplicated after the merge.
    second.add({ toolName: 'Bash', scope: ApprovalScope.ALWAYS, granted: false })
    const repersisted = JSON.parse(readFileSync(path, 'utf8')) as Array<{ tool_name: string }>
    expect(repersisted.map(record => record.tool_name).sort()).toEqual(['Bash', 'WriteFile'])

    const fresh = new ApprovalStore({ persistencePath: path })
    expect(fresh.check('WriteFile', 'any')).toBeTrue()
    expect(fresh.check('Bash', 'any')).toBeFalse()
  } finally {
    rmSync(directory, { force: true, recursive: true })
  }
})
