// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { resolveProjectDirectory as resolve } from './paths.js'
import { AgentPresetRoster } from '../agents/presets.js'
import { SkillRegistry, defaultSkillDiscoveryRoots, trustedHashWorkspaceSkills } from '../extensions/skills.js'
import { MCPManager } from '../mcp/manager.js'
import { startConfiguredMcpServers } from '../mcp/configured.js'

export interface WorkspaceResources {
  readonly skillRegistry: SkillRegistry
  readonly mcpManager: MCPManager
  readonly agentPresetRoster: AgentPresetRoster
}

/** One daemon owns independent project capabilities; no process-wide cwd mutation. */
export class DaemonWorkspaces {
  private readonly pending = new Map<string, Promise<WorkspaceResources>>()
  private readonly ready = new Map<string, WorkspaceResources>()
  private closed = false

  constructor(private readonly options: {
    readonly beforeOpen?: (cwd: string) => Promise<void>
    readonly home: string
    readonly allowWorkspace: boolean
    readonly report: (message: string) => void
  }) {}

  peek(cwd: string): WorkspaceResources | undefined { return this.ready.get(resolve(cwd)) }

  get(cwd: string): Promise<WorkspaceResources> {
    if (this.closed) return Promise.reject(new Error('Daemon workspace resources are closed'))
    const root = resolve(cwd)
    const existing = this.pending.get(root)
    if (existing) return existing
    const loading = this.load(root).then(resources => {
      // A released load may finish after its replacement. Only the current
      // generation may publish resources used by peek() and close().
      if (this.pending.get(root) === loading) this.ready.set(root, resources)
      return resources
    })
    this.pending.set(root, loading)
    loading.catch(() => { if (this.pending.get(root) === loading) this.pending.delete(root) })
    return loading
  }

  private async load(root: string): Promise<WorkspaceResources> {
    await this.options.beforeOpen?.(root)
    const resources = {
      skillRegistry: new SkillRegistry({ workspaceTrust: trustedHashWorkspaceSkills() }),
      mcpManager: new MCPManager(),
      agentPresetRoster: new AgentPresetRoster({ projectDirectory: root }),
    }
    try {
      await resources.skillRegistry.refresh(...defaultSkillDiscoveryRoots({ cwd: root }))
      await startConfiguredMcpServers(resources.mcpManager, { ...this.options, workspace: root })
      return resources
    } catch (error) {
      await resources.mcpManager.disconnectAll()
      throw error
    }
  }

  /**
   * Drop one workspace's resources when its last session is gone. Without
   * this a long-lived shared daemon keeps every project it ever saw — each
   * with its own eagerly-started MCP server processes — alive until process
   * exit. A later get() for the same root simply reloads. The pending entry
   * must go too: a resolved load stays memoized there, and a get() that found
   * only the pending entry would hand back the very manager this release just
   * disconnected, forever.
   */
  async release(cwd: string): Promise<void> {
    const root = resolve(cwd)
    const pending = this.pending.get(root)
    this.pending.delete(root)
    // Capture the doomed resources up front. A get() that arrives after the
    // release decision starts a NEW load (pending was deleted above); its
    // manager belongs to the workspace's next life and must never be
    // disconnected here, even if it lands in `ready` before this method
    // resumes — so only the captured reference is ever torn down.
    let resources = this.ready.get(root)
    if (!resources && pending) {
      // Do not tear down a load still in flight mid-load — awaiting it lets
      // startConfiguredMcpServers finish; the resulting manager is then an
      // orphan with no session, which is exactly what release disposes.
      resources = await pending.catch(() => undefined)
    }
    if (!resources) return
    if (this.ready.get(root) === resources) this.ready.delete(root)
    try {
      await resources.mcpManager.disconnectAll()
    } catch (error) {
      this.options.report(`Disconnecting MCP servers for '${root}' failed: ${error instanceof Error ? error.message : String(error)}`)
    }
  }

  async close(): Promise<void> {
    this.closed = true
    await Promise.allSettled(this.pending.values())
    await Promise.all([...this.ready.values()].map(resource => resource.mcpManager.disconnectAll()))
    this.ready.clear()
    this.pending.clear()
  }
}
