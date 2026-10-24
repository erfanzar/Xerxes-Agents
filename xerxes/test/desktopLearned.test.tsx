// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, expect, test } from 'bun:test'
import { createElement, isValidElement, type FormEvent, type ReactElement, type ReactNode } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { BlockBuilder } from '../src/desktop/renderer/blocks.js'
import { isGroupedActivity } from '../src/desktop/renderer/activityGroups.js'
import { Store, type XerxesLike } from '../src/desktop/renderer/store.js'
import type { DaemonEvent } from '../src/desktop/renderer/types.js'
import { LearnedBadge, LearnedLessons, LearnedPanel, LearningReviewControls, ProposalCard, LessonCard, ReflectionCard } from '../src/desktop/renderer/LearnedPanel.js'
import { readLearningList, reflectionNotice } from '../src/desktop/renderer/learning.js'
import { createBridge } from '../src/desktop/bridgeApi.js'
import type { StoredReflectionProposal } from '../src/runtime/reflection.js'
import type { Lesson } from '../src/memory/lessons.js'

const proposal: StoredReflectionProposal = {
  id: 'p1', workspace: '/repo', status: 'pending', createdAt: 1, updatedAt: 1,
  proposal: { kind: 'skill_create', target: 'check-first', rationale: 'Use [OBSERVATION] to verify.', evidence: ['[OBSERVATION] Tests passed'], skill: { description: 'Verify changes', instructions: 'Run the focused checks.', files: { 'references/check.md': 'Exact reference content' } } },
}
const lesson: Lesson = { id: 'l1', workspace: '/repo', cue: 'Before delivery', lesson: 'Verify the current tree.', confidence: 0.9, evidence: { kind: 'test', command: 'bun test', exitCode: 0, summary: '4 pass' }, agentId: 'default', createdAt: 1, reviewStatus: 'pending' }
const notice = { category: 'reflection', type: 'proposal', message: 'A review is ready', session_id: 's1', reflection: { id: 'p1', kind: 'skill_create', target: 'check-first', rationale: 'Use [OBSERVATION] to verify.', evidence_count: 1 } }
const list = () => ({ ok: true, workspace: '/repo', proposals: [proposal], lessons: [lesson], counts: { pending_proposals: 1, pending_lessons: 1 } })
const originalWindow = globalThis.window
class Bridge implements XerxesLike {
  calls: { method: string; params: Record<string, unknown> }[] = []
  handlers = new Set<(event: DaemonEvent) => void>()
  response: (method: string, params: Record<string, unknown>) => Record<string, unknown> | Promise<Record<string, unknown>> = method => method === 'learning.list' ? list() : method === 'learning.review' ? { ok: true, report: 'Applied check-first.', proposal: { ...proposal, status: 'applied' } } : method === 'initialize' ? { ok: true, session_id: 's1', cwd: '/repo', session: { id: 's1', key: 'desktop-test', cwd: '/repo' } } : { ok: true, sessions: [] }
  call(method: string, params: Record<string, unknown> = {}) { this.calls.push({ method, params }); return Promise.resolve(this.response(method, params)) }
  push(type: string, payload: Record<string, unknown>) { for (const handler of this.handlers) handler({ type, payload }) }
}
async function setup() {
  const bridge = new Bridge()
  ;(globalThis as { window: unknown }).window = { xerxes: { onEvent: (handler: (event: DaemonEvent) => void) => { bridge.handlers.add(handler); return () => bridge.handlers.delete(handler) } } }
  const store = new Store(); store.start(bridge); await Bun.sleep(0)
  return { store, bridge }
}
afterEach(() => { if (originalWindow === undefined) delete (globalThis as { window?: unknown }).window; else (globalThis as { window: unknown }).window = originalWindow })

test('opening a session fetches learned items and the panel lists exact proposal content', async () => {
  const { store, bridge } = await setup()
  expect(bridge.calls.some(call => call.method === 'learning.list' && call.params.session_key === 'desktop-test')).toBe(true)
  expect(store.getSnapshot().learningCounts).toEqual({ pending_proposals: 1, pending_lessons: 1 })
  const html = renderToStaticMarkup(createElement(LearnedPanel, { snap: store.getSnapshot(), actions: store }))
  for (const text of ['Proposals', 'Lessons', 'New skill', 'check-first', 'Use [OBSERVATION]', '[OBSERVATION] Tests passed', 'Verify changes', 'Run the focused checks.', 'references/check.md', 'Exact reference content', 'Approve', 'Reject']) expect(html).toContain(text)
})

test('approve is sent once while busy, includes session and target, and refreshes', async () => {
  const { store, bridge } = await setup()
  const pending = Promise.withResolvers<Record<string, unknown>>()
  bridge.response = method => method === 'learning.review' ? pending.promise : list()
  const before = bridge.calls.filter(call => call.method === 'learning.list').length
  const first = store.reviewLearning('proposal', 'p1', 'approve')
  const second = store.reviewLearning('proposal', 'p1', 'approve')
  expect(store.getSnapshot().learningReviewBusy).toBe(true)
  expect(bridge.calls.filter(call => call.method === 'learning.review')).toEqual([{ method: 'learning.review', params: { session_key: 'desktop-test', target: 'proposal', id: 'p1', action: 'approve' } }])
  pending.resolve({ ok: true, report: 'Applied check-first.' }); await Promise.all([first, second])
  expect(store.getSnapshot().learningReviewBusy).toBe(false)
  expect(store.getSnapshot().learningResults?.['proposal:p1']?.report).toBe('Applied check-first.')
  expect(bridge.calls.filter(call => call.method === 'learning.list').length).toBeGreaterThan(before)
})

test('reject and rollback require a nonblank reason; rollback is approved-lessons only', async () => {
  const { store, bridge } = await setup()
  await store.reviewLearning('proposal', 'p1', 'reject', '   ')
  await store.reviewLearning('lesson', 'l1', 'rollback', 'Incorrect lesson')
  await store.reviewLearning('proposal', 'p1', 'rollback', 'Not allowed')
  expect(bridge.calls.filter(call => call.method === 'learning.review')).toHaveLength(0)
  await store.reviewLearning('proposal', 'p1', 'reject', ' Not reusable ')
  expect(bridge.calls.find(call => call.method === 'learning.review')?.params).toMatchObject({ action: 'reject', reason: 'Not reusable' })
  bridge.response = method => method === 'learning.list' ? { ...list(), lessons: [{ ...lesson, reviewStatus: 'approved' }] } : { ok: true, report: 'Rolled back.' }
  await store.refreshLearning()
  await store.reviewLearning('lesson', 'l1', 'rollback', '')
  expect(bridge.calls.filter(call => call.method === 'learning.review')).toHaveLength(1)
  await store.reviewLearning('lesson', 'l1', 'rollback', 'Outdated')
  expect(bridge.calls.filter(call => call.method === 'learning.review').at(-1)?.params).toMatchObject({ target: 'lesson', id: 'l1', action: 'rollback', reason: 'Outdated' })
})

test('lessons show confidence, reported test evidence and status-appropriate controls', async () => {
  const { store } = await setup()
  const props = { actions: store, snap: store.getSnapshot() }
  const pending = renderToStaticMarkup(createElement(LessonCard, { ...props, lesson }))
  expect(pending).toContain('90%'); expect(pending).toContain('reported'); expect(pending).toContain('bun test'); expect(pending).toContain('exit code 0'); expect(pending).toContain('Approve'); expect(pending).not.toContain('Roll back')
  const approved = renderToStaticMarkup(createElement(LessonCard, { ...props, lesson: { ...lesson, reviewStatus: 'approved' } }))
  expect(approved).toContain('Roll back'); expect(approved).not.toContain('>Approve<'); expect(approved).not.toContain('>Reject<')
})

test('workspace broadcasts update counts immediately and refresh; other workspaces are ignored', async () => {
  const { store, bridge } = await setup()
  const pending = Promise.withResolvers<Record<string, unknown>>()
  bridge.response = () => pending.promise
  bridge.push('learning_changed', { workspace: '/elsewhere', counts: { pending_proposals: 100, pending_lessons: 0 } })
  expect(store.getSnapshot().learningCounts?.pending_proposals).toBe(1)
  bridge.push('learning_changed', { workspace: '/repo', counts: { pending_proposals: 3, pending_lessons: 4 } })
  expect(store.getSnapshot().learningCounts).toEqual({ pending_proposals: 3, pending_lessons: 4 })
  pending.resolve(list()); await Bun.sleep(0)
})

test('reflection notification stays a distinct visible card through block finalization', async () => {
  const { store, bridge } = await setup()
  bridge.push('notification', { ...notice, session_id: 'other' })
  expect(store.getSnapshot().blocks.some(block => block.kind === 'reflection')).toBe(false)
  bridge.push('notification', notice)
  const card = store.getSnapshot().blocks.find(block => block.kind === 'reflection')
  expect(card?.kind).toBe('reflection')
  if (card?.kind !== 'reflection') throw new Error('Missing card')
  expect(isGroupedActivity(card)).toBe(false)
  const html = renderToStaticMarkup(createElement(ReflectionCard, { reflection: card.reflection, snap: store.getSnapshot(), actions: store }))
  for (const text of ['New skill', 'check-first', 'Review', 'Approve', 'Reject']) expect(html).toContain(text)
  const builder = new BlockBuilder(); builder.push('text_part', { text: 'Finished' }); builder.push('notification', notice)
  const id = builder.snapshot(true).find(block => block.kind === 'reflection')?.id
  builder.finalize(); expect(builder.all().find(block => block.kind === 'reflection')?.id).toBe(id)
})

test('/xerxes alone opens Learned locally; arguments still reach the daemon', async () => {
  const { store, bridge } = await setup()
  bridge.calls.length = 0
  expect(await store.submit(' /xerxes  ')).toBe(true)
  expect(store.getSnapshot().learningOpen).toBe(true)
  expect(bridge.calls.some(call => call.method === 'slash' || call.method === 'turn.submit')).toBe(false)
  await store.submit('/xerxes lessons')
  expect(bridge.calls.find(call => call.method === 'slash')?.params.command).toBe('/xerxes lessons')
})

test('review failures preserve reports and lint details inline, with no retry actions for terminal proposals', async () => {
  const { store, bridge } = await setup()
  bridge.response = method => method === 'learning.review' ? { ok: false, error: 'Lint refused', report: 'Missing description: add a discovery description.', proposal: { ...proposal, status: 'failed' } } : { ...list(), proposals: [{ ...proposal, status: 'failed' }] }
  await store.reviewLearning('proposal', 'p1', 'approve')
  const html = renderToStaticMarkup(createElement(ProposalCard, { stored: { ...proposal, status: 'failed' }, snap: store.getSnapshot(), actions: store }))
  expect(html).toContain('Lint refused'); expect(html).toContain('Missing description'); expect(html).not.toContain('>Approve<'); expect(html).not.toContain('>Reject<')
})

interface NativeProps {
  children?: ReactNode
  onClick?: () => void
  onSubmit?: (event: FormEvent<HTMLFormElement>) => void
  disabled?: boolean
  required?: boolean
  name?: string
}
function elements(node: ReactNode): ReactElement<NativeProps>[] {
  if (Array.isArray(node)) return node.flatMap(elements)
  if (!isValidElement<NativeProps>(node)) return []
  return [node, ...elements(node.props.children)]
}

test('the actual Review and Approve button callbacks open the target and send one review', async () => {
  const { store, bridge } = await setup()
  const reflection = reflectionNotice(notice.reflection)!
  const view = ReflectionCard({ reflection, snap: store.getSnapshot(), actions: store })
  const review = elements(view).find(element => element.type === 'button' && element.props.children === 'Review')!
  review.props.onClick!()
  expect(store.getSnapshot().learningOpen).toBe(true)
  expect(store.getSnapshot().learningProposalId).toBe('p1')
  const controls = LearningReviewControls({ target: 'proposal', id: 'p1', snap: store.getSnapshot(), actions: store })
  const approve = elements(controls).find(element => element.type === 'button' && element.props.children === 'Approve')!
  approve.props.onClick!(); approve.props.onClick!(); await Bun.sleep(0)
  expect(bridge.calls.filter(call => call.method === 'learning.review')).toHaveLength(1)
  expect(store.getSnapshot().learningResults?.['proposal:p1']?.report).toContain('Applied')
})

test('the Reject form requires a reason and its submit callback sends that reason directly', async () => {
  const { store, bridge } = await setup()
  const controls = LearningReviewControls({ target: 'proposal', id: 'p1', snap: store.getSnapshot(), actions: store })
  const input = elements(controls).find(element => element.type === 'input')!
  expect(input.props.required).toBe(true)
  const form = elements(controls).find(element => element.type === 'form')!
  let prevented = false
  const submit = (value: string) => form.props.onSubmit!({ preventDefault: () => { prevented = true }, currentTarget: { elements: { namedItem: () => ({ value }) } } } as unknown as FormEvent<HTMLFormElement>)
  submit(' '); expect(bridge.calls.filter(call => call.method === 'learning.review')).toHaveLength(0)
  submit('Not reusable'); await Bun.sleep(0)
  expect(prevented).toBe(true)
  expect(bridge.calls.find(call => call.method === 'learning.review')?.params).toMatchObject({ target: 'proposal', id: 'p1', action: 'reject', reason: 'Not reusable' })
})

test('badge markup shows pending sum; all review buttons are disabled while refreshing or reviewing', async () => {
  const { store, bridge } = await setup()
  bridge.push('learning_changed', { workspace: '/repo', counts: { pending_proposals: 3, pending_lessons: 4 } })
  const html = renderToStaticMarkup(createElement(LearnedBadge, { snap: store.getSnapshot() }))
  expect(html).toContain('7 pending learned items'); expect(html).toContain('>7<')
  for (const update of [{ learningLoading: true }, { learningReviewBusy: true }, { connection: 'offline' as const }]) {
    const controls = LearningReviewControls({ target: 'proposal', id: 'p1', snap: { ...store.getSnapshot(), ...update }, actions: store })
    expect(elements(controls).filter(element => element.type === 'button').every(button => button.props.disabled)).toBe(true)
  }
})

test('lessons are grouped Pending then Approved and tool evidence is distinct from reported tests', async () => {
  const { store } = await setup()
  const approved: Lesson = { ...lesson, id: 'l2', cue: 'Before editing', reviewStatus: 'approved', evidence: { kind: 'tool_call', toolCallId: 't1', toolName: 'ReadFile', summary: 'Read the current source' } }
  const snap = { ...store.getSnapshot(), learning: { ...readLearningList(list()), lessons: [approved, lesson] } }
  const html = renderToStaticMarkup(createElement(LearnedLessons, { snap, actions: store }))
  expect(html.indexOf('Pending')).toBeLessThan(html.indexOf('Approved'))
  expect(html.indexOf('Before delivery')).toBeLessThan(html.indexOf('Before editing'))
  expect(html).toContain('ReadFile — Read the current source'); expect(html).toContain('Test (reported): bun test')
})

test('agent YAML, skill updates, and acknowledged plugin ideas retain exact previews and no terminal actions', async () => {
  const { store } = await setup()
  const base = { ...proposal.proposal, kind: 'plugin_idea', plugin: { description: '<script>not executable</script>' } } as const
  const plugin = renderToStaticMarkup(createElement(ProposalCard, { stored: { ...proposal, status: 'acknowledged', proposal: base }, snap: store.getSnapshot(), actions: store }))
  expect(plugin).toContain('Plugin idea'); expect(plugin).toContain('/plugin-creator'); expect(plugin).toContain('&lt;script&gt;'); expect(plugin).not.toContain('<script>'); expect(plugin).not.toContain('>Approve<')
  const agent = renderToStaticMarkup(createElement(ProposalCard, { stored: { ...proposal, proposal: { kind: 'agent_update', target: 'reviewer', rationale: 'Keep evidence', evidence: ['[ACTION] ReadFile'], agent: { yaml: 'name: reviewer\ntools: [ReadFile]\n' } } }, snap: store.getSnapshot(), actions: store }))
  expect(agent).toContain('Agent update'); expect(agent).toContain('name: reviewer\ntools: [ReadFile]')
  if (proposal.proposal.kind !== 'skill_create') throw new Error('Expected skill fixture')
  const update = renderToStaticMarkup(createElement(ProposalCard, { stored: { ...proposal, proposal: { ...proposal.proposal, kind: 'skill_update' } }, snap: store.getSnapshot(), actions: store }))
  expect(update).toContain('Skill update'); expect(update).toContain('Exact reference content')
  for (const status of ['applying', 'applied', 'failed', 'rejected', 'acknowledged'] as const) {
    const html = renderToStaticMarkup(createElement(ProposalCard, { stored: { ...proposal, status }, snap: store.getSnapshot(), actions: store }))
    expect(html).not.toContain('>Approve<'); expect(html).not.toContain('>Reject<')
  }
})

test('a removed lesson keeps its rollback report visible and list failures are not empty success', async () => {
  const { store, bridge } = await setup()
  bridge.response = method => method === 'learning.list' ? { ...list(), lessons: [{ ...lesson, reviewStatus: 'approved' }] } : { ok: true, report: 'Rolled back l1: outdated.' }
  await store.refreshLearning()
  bridge.response = method => method === 'learning.list' ? { ...list(), lessons: [] } : { ok: true, report: 'Rolled back l1: outdated.' }
  await store.reviewLearning('lesson', 'l1', 'rollback', 'outdated')
  expect(renderToStaticMarkup(createElement(LearnedPanel, { snap: store.getSnapshot(), actions: store }))).toContain('Rolled back l1: outdated.')
  bridge.response = () => ({ ok: false, error: 'Learning store unavailable' })
  await store.refreshLearning()
  const html = renderToStaticMarkup(createElement(LearnedPanel, { snap: store.getSnapshot(), actions: store }))
  expect(html).toContain('role="alert"'); expect(html).toContain('Learning store unavailable')
})

test('newer learning refreshes win over old replies, including a broadcast arriving mid-request', async () => {
  const { store, bridge } = await setup()
  const old = Promise.withResolvers<Record<string, unknown>>()
  bridge.response = () => old.promise
  const refreshing = store.refreshLearning()
  bridge.response = () => ({ ...list(), counts: { pending_proposals: 8, pending_lessons: 9 } })
  bridge.push('learning_changed', { workspace: '/repo', counts: { pending_proposals: 8, pending_lessons: 9 } })
  await Bun.sleep(0)
  old.resolve(list()); await refreshing
  expect(store.getSnapshot().learningCounts).toEqual({ pending_proposals: 8, pending_lessons: 9 })
})

test('malformed and cross-workspace items fail closed; invalid notifications never create approval cards', () => {
  for (const update of [{ proposals: [proposal, proposal] }, { lessons: [{ ...lesson, workspace: '/other' }] }, { lessons: [{ ...lesson, confidence: 2 }] }, { proposals: [{ ...proposal, status: 'surprise' }] }, { counts: { pending_proposals: -1, pending_lessons: 0 } }]) expect(() => readLearningList({ ...list(), ...update })).toThrow()
  expect(reflectionNotice({ ...notice.reflection, evidence_count: -1 })).toBeNull()
  const builder = new BlockBuilder(); builder.push('notification', { ...notice, reflection: { ...notice.reflection, kind: 'arbitrary' } })
  expect(builder.all().some(block => block.kind === 'reflection')).toBe(false)
})

test('each sub-tab shows its own pending count, and the panel opens on the one with something waiting', async () => {
  const { store } = await setup()
  const base = { ...store.getSnapshot(), learning: { ...readLearningList(list()), lessons: [] } }
  // Seen live: the header said 1, the person opened Lessons, saw nothing pending and nothing said why.
  const proposalsOnly = renderToStaticMarkup(createElement(LearnedPanel, { snap: { ...base, learningCounts: { pending_proposals: 1, pending_lessons: 0 } }, actions: store }))
  expect(proposalsOnly).toContain('aria-label="1 pending proposals"')
  expect(proposalsOnly).not.toContain('pending lessons"')
  expect(proposalsOnly).toContain('aria-label="Reflection proposals"')
  const lessonsOnly = renderToStaticMarkup(createElement(LearnedPanel, { snap: { ...base, learning: { ...readLearningList(list()), proposals: [] }, learningCounts: { pending_proposals: 0, pending_lessons: 2 } }, actions: store }))
  expect(lessonsOnly).toContain('aria-label="2 pending lessons"')
  expect(lessonsOnly).toContain('aria-label="Lessons"')
})

test('empty views explain evidence and human approval instead of implying automatic changes', async () => {
  const { store } = await setup()
  const snap = { ...store.getSnapshot(), learning: { ...readLearningList(list()), proposals: [], lessons: [] } }
  const proposals = renderToStaticMarkup(createElement(LearnedPanel, { snap, actions: store }))
  expect(proposals).toContain('No proposals yet.'); expect(proposals).toContain('Nothing is applied without approval.')
  const lessons = renderToStaticMarkup(createElement(LearnedLessons, { snap, actions: store }))
  expect(lessons).toContain('No pending lessons.'); expect(lessons).toContain('No approved lessons.'); expect(lessons).toContain('for approval')
})

test('transport failures are inline and still refresh the list; unknown proposals cannot be approved', async () => {
  const { store, bridge } = await setup()
  await store.reviewLearning('proposal', 'unknown', 'approve')
  expect(bridge.calls.some(call => call.method === 'learning.review')).toBe(false)
  bridge.response = method => { if (method === 'learning.review') throw new Error('Review connection lost'); return list() }
  await store.reviewLearning('proposal', 'p1', 'approve')
  expect(store.getSnapshot().learningReviewBusy).toBe(false)
  expect(store.getSnapshot().learningResults?.['proposal:p1']?.error).toBe('Review connection lost')
  expect(bridge.calls.at(-1)?.method).toBe('learning.list')
})

test('old session learning replies and review reports cannot contaminate the next workspace', async () => {
  const { store, bridge } = await setup()
  const oldList = Promise.withResolvers<Record<string, unknown>>(), oldReview = Promise.withResolvers<Record<string, unknown>>()
  bridge.response = method => method === 'learning.review' ? oldReview.promise : oldList.promise
  const reviewing = store.reviewLearning('proposal', 'p1', 'approve')
  const refreshing = store.refreshLearning()
  bridge.response = method => method === 'initialize'
    ? { ok: true, session_id: 's2', cwd: '/other', session: { id: 's2', key: 'desktop-other', cwd: '/other' } }
    : method === 'learning.list' ? { ok: true, workspace: '/other', proposals: [], lessons: [], counts: { pending_proposals: 0, pending_lessons: 0 } } : { ok: true, sessions: [] }
  await (store as unknown as { initialize(extra: Record<string, unknown>): Promise<unknown> }).initialize({})
  await Bun.sleep(0)
  oldList.resolve(list()); oldReview.resolve({ ok: true, report: 'Old workspace applied.' })
  await Promise.all([reviewing, refreshing])
  expect(store.getSnapshot().learning?.workspace).toBe('/other')
  expect(store.getSnapshot().learningCounts).toEqual({ pending_proposals: 0, pending_lessons: 0 })
  expect(store.getSnapshot().learningResults).toEqual({})
  expect(store.getSnapshot().learningReviewBusy).toBe(false)
})

test('the preload bridge accepts both learning methods and forwards the broadcast payload unchanged', async () => {
  const calls: unknown[][] = [], handlers = new Map<string, (value: unknown) => void>()
  const bridge = createBridge({ invoke: async (...args) => { calls.push(args); return { ok: true } }, on: (channel, handler) => { handlers.set(channel, handler); return () => handlers.delete(channel) }, setZoomFactor: () => {} })
  await bridge.call('learning.list', { session_key: 'desktop-test' })
  await bridge.call('learning.review', { session_key: 'desktop-test', target: 'proposal', id: 'p1', action: 'reject', reason: 'Not reusable' })
  expect(calls[0]).toEqual(['daemon:call', 'learning.list', { session_key: 'desktop-test' }])
  expect(calls[1]).toEqual(['daemon:call', 'learning.review', { session_key: 'desktop-test', target: 'proposal', id: 'p1', action: 'reject', reason: 'Not reusable' }])
  const received: unknown[] = []
  bridge.onEvent(event => received.push(event))
  const payload = { workspace: '/repo', counts: { pending_proposals: 1, pending_lessons: 2 } }
  handlers.get('daemon:event')!({ type: 'learning_changed', payload })
  expect(received).toEqual([{ type: 'learning_changed', payload }])
})
