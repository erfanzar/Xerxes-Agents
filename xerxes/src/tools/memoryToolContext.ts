// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHash } from 'node:crypto'
import { existsSync } from 'node:fs'
import { join, resolve } from 'node:path'

import type { ToolExecutionContext } from '../executors/toolRegistry.js'
import { ContextualMemory } from '../memory/contextualMemory.js'
import { LongTermMemory } from '../memory/longTermMemory.js'
import { RAGStorage, SQLiteStorage } from '../memory/storage.js'
import type { MemoryToolContext } from './memoryTools.js'

export interface MemoryToolContextResolverOptions {
  /**
   * Directory holding one durable recall database per project. It must sit
   * outside every AgentMemory scope root, or agent_memory_* lists and
   * searches the databases as text.
   */
  readonly directory: string
  /** Project used when the turn carries no `metadata.project_root`. */
  readonly defaultProjectRoot: string
}

export interface MemoryToolContextResolver {
  readonly prune: (sessionId: string) => void
  readonly resolve: (context: ToolExecutionContext) => MemoryToolContext
}

interface ProjectStore {
  readonly dbPath: string
  readonly sqlite: SQLiteStorage
  readonly storage: RAGStorage
  readonly longTerms: Map<string, LongTermMemory>
}

interface SessionMemory {
  readonly memory: ContextualMemory
  readonly projectRoot: string
}

/**
 * Host wiring for save_memory/search_memory. The working tier stays per
 * session, but the long-term tier is one durable store per agent and project
 * shared by every session. It used to be a fresh in-process store per session
 * (SQLiteStorage only writes when WRITE_MEMORY=1), so entries the tool
 * promised were durable vanished on session eviction or daemon restart.
 */
export function createMemoryToolContextResolver(options: MemoryToolContextResolverOptions): MemoryToolContextResolver {
  const sessions = new Map<string, SessionMemory>()
  const projects = new Map<string, ProjectStore>()

  // A memory wipe removes the database files while this process still holds
  // them open; writes through the old handle would land on an unlinked inode
  // and report success. Drop everything bound to that project and reopen.
  const discardIfRemoved = (projectRoot: string): void => {
    const project = projects.get(projectRoot)
    if (!project || existsSync(project.dbPath)) return
    project.sqlite.close()
    projects.delete(projectRoot)
    for (const [key, session] of sessions) {
      if (session.projectRoot === projectRoot) sessions.delete(key)
    }
  }
  const projectFor = (projectRoot: string): ProjectStore => {
    discardIfRemoved(projectRoot)
    let project = projects.get(projectRoot)
    if (!project) {
      const name = createHash('sha256').update(projectRoot).digest('hex').slice(0, 24)
      const dbPath = join(options.directory, `${name}.sqlite`)
      const sqlite = new SQLiteStorage({ dbPath, writeEnabled: true })
      project = { dbPath, sqlite, storage: new RAGStorage(sqlite), longTerms: new Map() }
      projects.set(projectRoot, project)
    }
    return project
  }
  const longTermFor = (agentId: string, projectRoot: string): LongTermMemory => {
    const project = projectFor(projectRoot)
    let longTerm = project.longTerms.get(agentId)
    if (!longTerm) {
      longTerm = new LongTermMemory({ storage: project.storage, ownerId: agentId })
      project.longTerms.set(agentId, longTerm)
    }
    return longTerm
  }
  return {
    prune(sessionId) {
      const prefix = `${sessionId}:`
      for (const key of sessions.keys()) {
        if (key.startsWith(prefix)) sessions.delete(key)
      }
    },
    resolve(context) {
      const agentId = context.agentId ?? 'default'
      const key = (context.sessionId ?? 'sessionless') + ':' + agentId
      const existing = sessions.get(key)
      if (existing) discardIfRemoved(existing.projectRoot)
      let session = sessions.get(key)
      if (!session) {
        const projectRoot = context.metadata?.project_root
        const root = resolve(typeof projectRoot === 'string' && projectRoot.trim() ? projectRoot : options.defaultProjectRoot)
        session = { memory: new ContextualMemory({ longTerm: longTermFor(agentId, root) }), projectRoot: root }
        sessions.set(key, session)
      }
      return { agentId, memory: session.memory }
    },
  }
}
