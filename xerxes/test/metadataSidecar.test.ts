// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, beforeEach, expect, test } from 'bun:test'
import { appendFile, mkdtemp, readFile, rm, stat, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { createGoal, getGoal } from '../src/runtime/goalDomain.js'
import { buildSessionExport, listSavedSessions } from '../src/runtime/sessionExport.js'
import { DaemonTranscriptStore, normalizeDaemonTranscript, type DaemonTranscript } from '../src/session/daemonTranscript.js'
import { joinMetadata, sidecarPathFor, splitMetadata } from '../src/session/metadataSidecar.js'

const ID = 'abcdef012345'
const MESSAGES = [{ role: 'user', content: 'hi' }, { role: 'assistant', content: 'hello' }]
let directory: string
let store: DaemonTranscriptStore

beforeEach(async () => {
  directory = await mkdtemp(join(tmpdir(), 'xerxes-metadata-sidecar-'))
  store = new DaemonTranscriptStore({ directory, currentProjectDirectory: directory })
})
afterEach(async () => { await rm(directory, { recursive: true, force: true }) })

const row = (index: number, output: string, status = 'completed') => ({ id: `agent-${index}`, status, last_output: output })

function transcriptOf(metadata: Record<string, unknown>): DaemonTranscript {
  return normalizeDaemonTranscript(
    { session_id: ID, turn_count: 1, cwd: directory, messages: MESSAGES, metadata },
    { currentProjectDirectory: directory, requestedSessionKey: ID },
  )!
}

const sidecar = () => sidecarPathFor(store.pathFor(ID))
const sidecarLines = async () => (await readFile(sidecar(), 'utf8')).split('\n').filter(Boolean).length
const rawRecord = async () => JSON.parse(await readFile(store.pathFor(ID), 'utf8')) as Record<string, unknown>
const agents = (transcript: DaemonTranscript | undefined) => transcript?.metadata.xerxes_subagent_snapshots_v1 as Record<string, unknown>[]

test('agents, goal history and workflow runs are kept whole, but out of the transcript file', async () => {
  const metadata: Record<string, unknown> = {
    title: 'kept inline',
    xerxes_subagent_snapshots_v1: [row(0, 'a'), row(1, 'b'), row(2, 'c')],
    xerxes_workflow_runs_v1: { run1: { name: 'audit', script: 'return 1' } },
  }
  createGoal(metadata, ID, { objective: 'keep everything' }, 1_000)
  await store.save(transcriptOf(metadata))

  const raw = await rawRecord()
  expect(raw.metadata).toEqual({ title: 'kept inline' })
  expect(Object.keys(raw.metadata_sidecar as object).sort()).toEqual(['goal_changes', 'xerxes_subagent_snapshots_v1', 'xerxes_workflow_runs_v1'])
  // The refs come after the history, so a listing's head read never pays for them.
  expect(Object.keys(raw).at(-1)).toBe('metadata_sidecar')
  expect((await store.readHeader(ID)).kind).toBe('header')

  const loaded = await new DaemonTranscriptStore({ directory, currentProjectDirectory: directory }).load(ID)
  expect(agents(loaded)).toEqual([row(0, 'a'), row(1, 'b'), row(2, 'c')])
  expect(loaded?.metadata.xerxes_workflow_runs_v1).toEqual({ run1: { name: 'audit', script: 'return 1' } })
  expect(getGoal(loaded!.metadata, ID)?.objective).toBe('keep everything')
  expect(loaded?.extra).not.toHaveProperty('metadata_sidecar')
})

test('a save writes only the records that changed', async () => {
  await store.save(transcriptOf({ xerxes_subagent_snapshots_v1: [row(0, 'a'), row(1, 'b'), row(2, 'c')] }))
  expect(await sidecarLines()).toBe(3)

  const loaded = (await store.load(ID))!
  const rows = [...agents(loaded)]
  rows[1] = { ...rows[1], status: 'failed' }
  await store.save({ ...loaded, metadata: { ...loaded.metadata, xerxes_subagent_snapshots_v1: rows } })
  expect(await sidecarLines()).toBe(4)

  // Nothing changed: the sidecar is not touched at all.
  const again = (await store.load(ID))!
  await store.save(again)
  expect(await sidecarLines()).toBe(4)
  expect(agents(await store.load(ID))).toEqual([row(0, 'a'), row(1, 'b', 'failed'), row(2, 'c')])
})

test('an older transcript with inline metadata loads as is and moves out on its next save', async () => {
  await writeFile(store.pathFor(ID), JSON.stringify({
    format: 'xerxes-daemon-session', schema_version: 2, session_id: ID, turn_count: 1, generation: 3, cwd: directory,
    metadata: { title: 'old', xerxes_subagent_snapshots_v1: [row(0, 'from before the sidecar')] },
    messages: MESSAGES,
  }))
  const loaded = (await store.load(ID))!
  expect(agents(loaded)).toEqual([row(0, 'from before the sidecar')])
  await store.save(loaded)
  expect((await rawRecord()).metadata).toEqual({ title: 'old' })
  expect(agents(await store.load(ID))).toEqual([row(0, 'from before the sidecar')])
})

test('a torn final sidecar line is ignored and the next append starts on a fresh line', async () => {
  await store.save(transcriptOf({ xerxes_subagent_snapshots_v1: [row(0, 'a')] }))
  await appendFile(sidecar(), '{"h":"0123456789abcdef0123456789abcdef","v":{"id":"tor')
  const loaded = (await store.load(ID))!
  expect(agents(loaded)).toEqual([row(0, 'a')])
  await store.save({ ...loaded, metadata: { ...loaded.metadata, xerxes_subagent_snapshots_v1: [row(0, 'a'), row(1, 'b')] } })
  const reread = new DaemonTranscriptStore({ directory, currentProjectDirectory: directory })
  expect(agents(await reread.load(ID))).toEqual([row(0, 'a'), row(1, 'b')])
})

test('superseded versions are reclaimed, never a record the session still holds', async () => {
  const round = (n: number) => Array.from({ length: 30 }, (_, index) => row(index, `${index}:${n}:`.padEnd(40_000, 'x')))
  await store.save(transcriptOf({ xerxes_subagent_snapshots_v1: round(0) }))
  for (const n of [1, 2]) {
    const loaded = (await store.load(ID))!
    await store.save({ ...loaded, metadata: { ...loaded.metadata, xerxes_subagent_snapshots_v1: round(n) } })
  }
  // Three generations of 1.2 MB were written; only the live one remains.
  expect(await sidecarLines()).toBe(30)
  expect((await stat(sidecar())).size).toBeLessThan(1_300_000)
  const reread = new DaemonTranscriptStore({ directory, currentProjectDirectory: directory })
  expect(agents(await reread.load(ID))).toEqual(round(2))
})

test('removing a session removes its sidecar; an export carries the metadata whole', async () => {
  await store.save(transcriptOf({ title: 'exported', xerxes_subagent_snapshots_v1: [row(0, 'a'), row(1, 'b')] }))
  const [saved] = await listSavedSessions({ storeDir: directory })
  const exported = await buildSessionExport(saved!)
  expect(exported.metadata.xerxes_subagent_snapshots_v1).toEqual([row(0, 'a'), row(1, 'b')])
  expect(exported.metadata).not.toHaveProperty('metadata_sidecar')

  expect(await store.remove(ID)).toBe(true)
  expect(await Bun.file(sidecar()).exists()).toBe(false)
})

test('split stores identical records once; join keeps an inline value and counts missing records', () => {
  const split = splitMetadata({ other: 1, xerxes_subagent_snapshots_v1: [row(0, 'same'), row(0, 'same')] })
  expect(split.metadata).toEqual({ other: 1 })
  expect(split.entries.size).toBe(1)

  const values = new Map([...split.entries].map(([hash, json]) => [hash, JSON.parse(json) as unknown]))
  const joined: Record<string, unknown> = {}
  expect(joinMetadata(joined, split.refs, values)).toBe(0)
  expect(joined.xerxes_subagent_snapshots_v1).toEqual([row(0, 'same'), row(0, 'same')])

  const inline: Record<string, unknown> = { xerxes_subagent_snapshots_v1: ['inline wins'] }
  expect(joinMetadata(inline, split.refs, new Map())).toBe(0)
  expect(inline.xerxes_subagent_snapshots_v1).toEqual(['inline wins'])

  const missing: Record<string, unknown> = {}
  expect(joinMetadata(missing, split.refs, new Map())).toBe(2)
  expect(missing.xerxes_subagent_snapshots_v1).toEqual([])
})
