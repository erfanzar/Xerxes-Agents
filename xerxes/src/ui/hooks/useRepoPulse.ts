// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
//
// The working tree's current state, polled on the same cadence and with the
// same shared cache as `useGitBranch` — two components asking for it in the
// same frame cost one `git` invocation, not two.
import { execFile } from 'node:child_process'
import { promisify } from 'node:util'

import { useEffect, useState } from 'react'

import { EMPTY_PULSE, parseNumstat, parseStatusPorcelainV2, type RepoPulse } from '../lib/repoPulse.js'

const TTL_MS = 15_000
const TIMEOUT_MS = 700

const pexec = promisify(execFile)
const cache = new Map<string, { at: number; pulse: RepoPulse }>()
const inflight = new Map<string, Promise<RepoPulse>>()

const run = async (cwd: string, args: readonly string[]): Promise<string> => {
  try {
    const { stdout } = await pexec('git', ['-C', cwd, ...args], { timeout: TIMEOUT_MS })

    return stdout
  } catch {
    return ''
  }
}

const resolvePulse = async (cwd: string): Promise<RepoPulse> => {
  const [status, numstat] = await Promise.all([
    run(cwd, ['status', '--porcelain=v2', '--branch', '--untracked-files=normal']),
    run(cwd, ['diff', '--numstat', 'HEAD'])
  ])

  if (!status) {
    return EMPTY_PULSE
  }

  return { ...EMPTY_PULSE, ...parseStatusPorcelainV2(status), ...parseNumstat(numstat) }
}

const fetchPulse = (cwd: string): Promise<RepoPulse> => {
  const pending = inflight.get(cwd)

  if (pending) {
    return pending
  }

  const next = resolvePulse(cwd).finally(() => inflight.delete(cwd))
  inflight.set(cwd, next)

  return next
}

/**
 * Poll the repository at `cwd`. Returns the last known pulse immediately, so
 * a re-render never blanks the statusbar while git is answering.
 */
export function useRepoPulse(cwd: string): RepoPulse {
  const [pulse, setPulse] = useState<RepoPulse>(() => cache.get(cwd)?.pulse ?? EMPTY_PULSE)

  useEffect(() => {
    if (!cwd) {
      return
    }

    let cancelled = false

    const tick = async () => {
      const hit = cache.get(cwd)

      if (hit && Date.now() - hit.at < TTL_MS) {
        if (!cancelled) {
          setPulse(hit.pulse)
        }

        return
      }

      const next = await fetchPulse(cwd)
      cache.set(cwd, { at: Date.now(), pulse: next })

      if (!cancelled) {
        setPulse(next)
      }
    }

    void tick()
    const id = setInterval(() => void tick(), TTL_MS)
    id.unref?.()

    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [cwd])

  return pulse
}
