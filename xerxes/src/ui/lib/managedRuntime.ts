// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

export interface ManagedRuntimeClient {
  start(): Promise<void>
  request<T>(method: string, params: Record<string, unknown>): Promise<T>
  close(): void
}

/** Upgrade through the daemon's atomic idle guard, never by killing a process. */
export async function prepareManagedRuntime(
  create: (verifyBuild: boolean) => ManagedRuntimeClient,
  expectedBuildId: string,
  waitForExit: (pid: number) => Promise<void>,
): Promise<{ busy: boolean }> {
  let client = create(false)
  try {
    await client.start()
    const status = await client.request<Record<string, unknown>>('runtime.status', {})
    if (status.daemon_build_id === expectedBuildId) return { busy: false }
    const pid = status.pid
    if (!Number.isSafeInteger(pid) || Number(pid) <= 0) throw new Error('Remote runtime identity is unavailable; left it running.')
    const result = await client.request<Record<string, unknown>>('runtime.restart_if_idle', {})
    if (result.busy === true) return { busy: true }
    if (result.ok !== true) throw new Error(String(result.error || 'Remote runtime cannot restart safely; left it running.'))
    client.close()
    await waitForExit(Number(pid))
    client = create(true)
    await client.start()
    const updated = await client.request<Record<string, unknown>>('runtime.status', {})
    if (updated.daemon_build_id !== expectedBuildId) throw new Error('Remote runtime restarted with an unexpected build.')
    return { busy: false }
  } finally { client.close() }
}
