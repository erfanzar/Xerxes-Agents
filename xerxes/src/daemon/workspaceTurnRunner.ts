// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { resolveProjectDirectory as resolve } from './paths.js'
import type { DaemonSession, TurnRunner, TurnRunControls } from './runtime.js'
import type { DaemonWorkspaces, WorkspaceResources } from './workspaceResources.js'

/** Pin a runner to its workspace for the whole turn, including concurrent turns. */
export class WorkspaceTurnRunner implements TurnRunner {
  readonly managesSessionState = true
  private readonly runners = new Map<string, TurnRunner>()

  constructor(
    private readonly workspaces: Pick<DaemonWorkspaces, 'get' | 'peek'>,
    private readonly create: (cwd: string, resources: WorkspaceResources) => TurnRunner | undefined,
  ) {}

  private runner(cwd: string, resources: WorkspaceResources): TurnRunner {
    const root = resolve(cwd)
    const cached = this.runners.get(root)
    if (cached) return cached
    const runner = this.create(root, resources)
    if (!runner) throw new Error('Configure a supported provider before starting a turn')
    this.runners.set(root, runner)
    return runner
  }

  toolInventory(session: DaemonSession) {
    const resources = this.workspaces.peek(session.cwd)
    if (!resources) return []
    const root = resolve(session.cwd)
    // A listing must not pin a full runner (tool registry, agent maps,
    // bootstrap-prompt cache) into this map for a workspace that may never
    // run a turn — cache only runners a real turn created.
    const runner = this.runners.get(root) ?? this.create(root, resources)
    if (!runner) return []
    return runner.toolInventory?.(session) ?? []
  }

  dropSession(sessionId: string): void {
    for (const runner of this.runners.values()) runner.dropSession?.(sessionId)
  }

  dropWorkspace(cwd: string): void {
    this.runners.delete(resolve(cwd))
  }

  async *run(session: DaemonSession, text: string, signal: AbortSignal, controls?: TurnRunControls) {
    const resources = await this.workspaces.get(session.cwd)
    signal.throwIfAborted()
    yield* this.runner(session.cwd, resources).run(session, text, signal, controls)
  }
}
