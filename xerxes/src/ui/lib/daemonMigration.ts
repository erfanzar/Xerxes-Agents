// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

type Reply = Record<string, unknown>

function processAlive(pid: number): boolean {
  try { process.kill(pid, 0); return true }
  catch (error) { return (error as NodeJS.ErrnoException).code !== 'ESRCH' }
}

/** Both clients use the server's atomic idle check; neither kills a shared process. */
export async function releaseIdleProjectDaemon(
  request: (method: string) => Promise<Reply>,
  isAlive: (pid: number) => boolean = processAlive,
): Promise<boolean> {
  const status = await request('runtime.status')
  if (typeof status.pid !== 'number' || !Number.isSafeInteger(status.pid) || status.pid <= 0) return false
  let result: Reply
  try { result = await request('runtime.restart_if_idle') }
  catch (error) {
    // Older servers cannot prove idle atomically. Preserve their ownership.
    if (error instanceof Error && /unknown method|method not found/i.test(error.message)) return false
    throw error
  }
  if (result.busy === true) return false
  if (result.ok !== true) throw new Error(typeof result.error === 'string' ? result.error : 'Could not migrate the previous daemon safely')
  const deadline = Date.now() + 5000
  while (isAlive(status.pid)) {
    if (Date.now() >= deadline) throw new Error('Previous daemon is still shutting down. Reconnect shortly; existing work was left untouched.')
    await new Promise(resolve => setTimeout(resolve, 25))
  }
  return true
}
