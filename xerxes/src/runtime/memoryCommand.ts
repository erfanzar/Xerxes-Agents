// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdir } from 'node:fs/promises'
import { dirname, join } from 'node:path'

import { xerxesHome } from '../daemon/paths.js'
import { GovernedMemoryStore } from '../memory/governedMemoryStore.js'

export type MemoryCommandAction = 'record' | 'review' | 'classify' | 'correct' | 'expire' | 'list'

export interface MemoryCommandOptions {
  readonly action: MemoryCommandAction
  readonly id?: string
  readonly content?: string
  readonly source?: string
  readonly agentId?: string
  readonly sensitivity?: 'public' | 'internal' | 'confidential' | 'restricted'
  readonly originalId?: string
  readonly newId?: string
  readonly reason?: string
  readonly directory?: string
}

export interface MemoryCommandResult {
  readonly ok: boolean
  readonly message?: string
  readonly error?: string
}

export async function runMemoryCommand(options: MemoryCommandOptions): Promise<MemoryCommandResult> {
  const directory = options.directory ?? join(xerxesHome(), 'governed-memory')
  await mkdir(dirname(directory), { recursive: true })
  const store = new GovernedMemoryStore({ directory })

  switch (options.action) {
    case 'record': {
      if (!options.id || !options.content || !options.source || !options.agentId) {
        return { ok: false, error: 'record requires --id, --content, --source, and --agent-id' }
      }
      await store.record({
        id: options.id,
        content: options.content,
        source: options.source,
        agentId: options.agentId,
        sensitivity: options.sensitivity,
      })
      return { ok: true, message: `recorded memory ${options.id}` }
    }
    case 'review': {
      if (!options.id) return { ok: false, error: 'review requires --id' }
      const status = options.content as 'approved' | 'rejected' | undefined
      if (!status || !['approved', 'rejected'].includes(status)) {
        return { ok: false, error: 'review requires --content to be approved or rejected' }
      }
      await store.review(options.id, status)
      return { ok: true, message: `reviewed ${options.id} as ${status}` }
    }
    case 'classify': {
      if (!options.id || !options.sensitivity) return { ok: false, error: 'classify requires --id and --sensitivity' }
      await store.classify(options.id, options.sensitivity)
      return { ok: true, message: `classified ${options.id} as ${options.sensitivity}` }
    }
    case 'correct': {
      if (!options.originalId || !options.newId || !options.content || !options.reason) {
        return { ok: false, error: 'correct requires --original-id, --new-id, --content, and --reason' }
      }
      await store.correct({ originalId: options.originalId, newId: options.newId, content: options.content, reason: options.reason })
      return { ok: true, message: `corrected ${options.originalId} as ${options.newId}` }
    }
    case 'expire': {
      if (!options.id) return { ok: false, error: 'expire requires --id' }
      await store.expire(options.id)
      return { ok: true, message: `expired memory ${options.id}` }
    }
    case 'list': {
      const state = await store.load()
      const lines = Array.from(state.records.values()).map(r => `${r.id}\t${r.reviewStatus}\t${r.sensitivity}\t${r.content.slice(0, 40)}`)
      return { ok: true, message: lines.join('\n') || 'no records' }
    }
  }
}
