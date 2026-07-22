// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

export interface AcpSession {
  cancelled: boolean
  cwd: string
  readonly metadata: Record<string, unknown>
  modelOverride: string | null
  readonly sessionId: string
  title: string
}

export interface AcpSessionOptions {
  readonly metadata?: Record<string, unknown>
  readonly model?: string | null
  readonly title?: string
}

export interface ExistingAcpSessionRecord {
  readonly sessionId: string
}

/** Raised when attaching a session id that is already live in the store. */
export class AcpSessionConflictError extends Error {
  constructor(readonly sessionId: string) {
    super(`ACP session already exists: ${sessionId}`)
    this.name = 'AcpSessionConflictError'
  }
}

/** In-memory registry for live ACP sessions. */
export class AcpSessionStore {
  private readonly sessions = new Map<string, AcpSession>()

  create(cwd: string, options: AcpSessionOptions = {}): AcpSession {
    const session: AcpSession = {
      sessionId: createSessionId(),
      cwd,
      modelOverride: options.model ?? null,
      title: options.title ?? '',
      cancelled: false,
      metadata: { ...options.metadata },
    }
    this.sessions.set(session.sessionId, session)
    return session
  }

  attachExisting(record: ExistingAcpSessionRecord, cwd: string): AcpSession {
    if (this.sessions.has(record.sessionId)) {
      // Silently replacing the live record would drop its model override,
      // title, cancellation state, and metadata.
      throw new AcpSessionConflictError(record.sessionId)
    }
    const session: AcpSession = {
      sessionId: record.sessionId,
      cwd,
      modelOverride: null,
      title: '',
      cancelled: false,
      metadata: {},
    }
    this.sessions.set(session.sessionId, session)
    return session
  }

  get(sessionId: string): AcpSession | undefined {
    return this.sessions.get(sessionId)
  }

  list(): readonly AcpSession[] {
    return [...this.sessions.values()]
  }

  cancel(sessionId: string): boolean {
    const session = this.sessions.get(sessionId)
    if (!session) {
      return false
    }
    session.cancelled = true
    return true
  }

  drop(sessionId: string): boolean {
    return this.sessions.delete(sessionId)
  }

  setModel(sessionId: string, model: string | null): boolean {
    const session = this.sessions.get(sessionId)
    if (!session) {
      return false
    }
    session.modelOverride = model
    return true
  }
}

function createSessionId(): string {
  return crypto.randomUUID().replaceAll('-', '')
}
