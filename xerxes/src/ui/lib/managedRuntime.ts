// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

export interface ManagedRuntimeClient {
  start(): Promise<void>
  request<T>(method: string, params: Record<string, unknown>): Promise<T>
  close(): void
}

/**
 * Opt-in for runtimes too old to have `runtime.restart_if_idle`. Without it
 * such a runtime is an error the user had to clear by hand (ssh in, stop it,
 * start the new build). With it, the runtime is asked to `shutdown` — never
 * killed — and only after every kind of running work is checked idle, the
 * same one-time migration the desktop offers for a local runtime.
 */
export interface LegacyRuntimeMigration {
  /** The pid of a runtime whose status does not report one; undefined when it cannot be confirmed. */
  pidFallback?: () => Promise<number | undefined>
}

export interface ManagedRuntimeOutcome {
  busy: boolean
  /** What the busy runtime is waiting for, as the daemon reported it. */
  blockers?: string[]
  waitingForGoalRound?: boolean
}

export interface ManagedRuntimeOptions {
  /** The person confirmed "restart now": running work stops instead of being waited out. */
  force?: boolean
}

/** Upgrade through the daemon's atomic idle guard, never by killing a process. */
export async function prepareManagedRuntime(
  create: (verifyBuild: boolean) => ManagedRuntimeClient,
  expectedBuildId: string,
  waitForExit: (pid: number) => Promise<void>,
  legacy?: LegacyRuntimeMigration,
  options: ManagedRuntimeOptions = {},
): Promise<ManagedRuntimeOutcome> {
  // Self-contained on purpose: this function is inlined into the script the
  // remote host runs, so these helpers cannot live at module level.
  const restarting = (reply: Record<string, unknown>) => reply.ok === false && /^Runtime is restarting/i.test(String(reply.error ?? ''))
  const busy = (reply: Record<string, unknown>): ManagedRuntimeOutcome => ({ busy: true,
    ...(Array.isArray(reply.blockers) ? { blockers: (reply.blockers as unknown[]).filter((item): item is string => typeof item === 'string') } : {}),
    ...(reply.waiting_for_goal_round === true ? { waitingForGoalRound: true } : {}) })
  let client = create(false)
  try {
    await client.start()
    const status = await client.request<Record<string, unknown>>('runtime.status', {})
    if (status.daemon_build_id === expectedBuildId) return { busy: false }
    let pid = Number.isSafeInteger(status.pid) && Number(status.pid) > 0 ? Number(status.pid) : undefined
    if (pid === undefined && legacy?.pidFallback) pid = await legacy.pidFallback()
    if (pid === undefined) throw new Error('Remote runtime identity is unavailable; left it running.')
    // Another client (a second window on this host, a TUI) may already have
    // asked this runtime to restart; it then refuses every request. That is
    // the restart this one wants too: wait for it instead of failing.
    let result = restarting(status) ? status : await client.request<Record<string, unknown>>('runtime.restart_if_idle', options.force ? { force: true } : {})
    if (options.force && result.busy === true && result.force_supported !== true) {
      // A runtime older than "restart now" ignores force; its shutdown stops
      // the running work, which the person confirmed before this was sent.
      result = await client.request<Record<string, unknown>>('shutdown', {})
    }
    if (result.busy === true) return busy(result)
    if (result.ok !== true && legacy && /^Unknown method/i.test(String(result.error ?? ''))) {
      // Self-contained on purpose: this function is inlined into the script
      // the remote host runs, so it cannot call helpers from this module.
      if (status.active_subagents !== undefined && status.active_subagents !== 0) return { busy: true }
      if (status.channels_configured === true) return { busy: true }
      const list = await client.request<Record<string, unknown>>('session.active_list', {})
      if (list.ok !== true || !Array.isArray(list.sessions)) throw new Error('Could not check running work on the old runtime; left it running.')
      for (const value of list.sessions as unknown[]) {
        const session = value && typeof value === 'object' ? value as Record<string, unknown> : undefined
        if (!session || typeof session.key !== 'string') throw new Error('Could not identify a session on the old runtime; left it running.')
        if (session.status !== 'idle' || session.active_turn_id) return { busy: true }
        const terminals = await client.request<Record<string, unknown>>('terminal.list', { session_key: session.key })
        const monitors = await client.request<Record<string, unknown>>('monitor.list', { session_key: session.key })
        const known = (reply: Record<string, unknown>) => !/^Unknown method/i.test(String(reply.error ?? ''))
        if (known(terminals) && (terminals.ok !== true || !Array.isArray(terminals.terminals))) throw new Error('Could not check terminals on the old runtime; left it running.')
        if (known(monitors) && (monitors.ok !== true || !Array.isArray(monitors.monitors))) throw new Error('Could not check monitors on the old runtime; left it running.')
        if (Array.isArray(terminals.terminals) && (terminals.terminals as Array<Record<string, unknown> | null>).some(row => row?.running)) return { busy: true }
        if (Array.isArray(monitors.monitors) && (monitors.monitors as Array<Record<string, unknown> | null>).some(row => row?.state === 'watching')) return { busy: true }
      }
      result = await client.request<Record<string, unknown>>('shutdown', {})
    }
    if (result.busy === true) return busy(result)
    if (result.ok !== true && !restarting(result)) throw new Error(String(result.error || 'Remote runtime cannot restart safely; left it running.'))
    client.close()
    await waitForExit(pid)
    client = create(true)
    await client.start()
    const updated = await client.request<Record<string, unknown>>('runtime.status', {})
    if (updated.daemon_build_id !== expectedBuildId) throw new Error('Remote runtime restarted with an unexpected build.')
    return { busy: false }
  } finally { client.close() }
}
