// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */

import { testRender } from '@opentui/react/test-utils'
import { act } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { GatewayProvider } from '../app/gatewayContext.js'
import type { GatewayServices } from '../app/interfaces.js'
import { getOverlayState, resetOverlayState } from '../app/overlayStore.js'
import { agentRetryCommands } from '../app/slash/commands/agentRetry.js'
import type { SlashRunCtx } from '../app/slash/types.js'
import type { GatewayClient } from '../gatewayClient.js'
import { subagentFailed, subagentRetryable } from '../lib/agentRetry.js'
import { AgentPanelOverlay } from '../opentui/agentPanel.js'
import { DEFAULT_THEME } from '../theme.js'
import type { SubagentProgress } from '../types.js'

const agent = (overrides: Partial<SubagentProgress> = {}): SubagentProgress => ({
  agentType: 'researcher',
  depth: 0,
  goal: 'Audit authentication policy boundaries',
  id: 'agent-1',
  index: 0,
  name: 'dead-worker',
  notes: [],
  parentId: null,
  status: 'failed',
  taskCount: 1,
  thinking: [],
  toolCount: 1,
  tools: ['ReadFile'],
  ...overrides
})

const servicesWith = (rpc: GatewayServices['rpc']): GatewayServices =>
  ({ gw: {} as GatewayClient, rpc }) as unknown as GatewayServices

const flushRetry = async (setup: { flush: () => Promise<void> }) => {
  await act(async () => {
    await Bun.sleep(0)
    await Bun.sleep(0)
  })
  await setup.flush()
}

describe('agent retry model', () => {
  it('treats every terminal state as retryable and active states as off-limits', () => {
    expect(subagentRetryable('failed')).toBe(true)
    expect(subagentRetryable('error')).toBe(true)
    expect(subagentRetryable('interrupted')).toBe(true)
    expect(subagentRetryable('timeout')).toBe(true)
    expect(subagentRetryable('completed')).toBe(true)
    expect(subagentRetryable('running')).toBe(false)
    expect(subagentRetryable('queued')).toBe(false)
    expect(subagentFailed('failed')).toBe(true)
    expect(subagentFailed('completed')).toBe(false)
  })
})

describe('OpenTUI agents overlay retry action', () => {
  it('retries the selected dead agent through the subagent.retry RPC', async () => {
    const rpc = vi.fn(async () => ({ ok: true, agent: { name: 'dead-worker', status: 'idle' } }))
    const setup = await testRender(
      <GatewayProvider value={servicesWith(rpc)}>
        <AgentPanelOverlay history={[]} liveAgents={[agent()]} onClose={() => undefined} t={DEFAULT_THEME} />
      </GatewayProvider>,
      { height: 24, width: 80 }
    )

    try {
      await setup.flush()
      expect(setup.captureCharFrame()).toContain('press r to retry this agent')

      act(() => setup.mockInput.pressKey('r'))
      await flushRetry(setup)
      expect(rpc).toHaveBeenCalledWith('subagent.retry', { task: 'dead-worker' })
      // The framed card clips a long note at 80 columns; assert the verdict,
      // not the full sentence.
      expect(setup.captureCharFrame()).toContain('retry accepted')
    } finally {
      act(() => setup.renderer.destroy())
    }
  })

  it('shows the daemon error honestly when the retry is rejected', async () => {
    const rpc = vi.fn(async () => ({ ok: false, error: 'spawned agent not found' }))
    const setup = await testRender(
      <GatewayProvider value={servicesWith(rpc)}>
        <AgentPanelOverlay history={[]} liveAgents={[agent()]} onClose={() => undefined} t={DEFAULT_THEME} />
      </GatewayProvider>,
      { height: 24, width: 80 }
    )

    try {
      await setup.flush()
      act(() => setup.mockInput.pressKey('r'))
      await flushRetry(setup)

      expect(rpc).toHaveBeenCalledWith('subagent.retry', { task: 'dead-worker' })
      expect(setup.captureCharFrame()).toContain('retry failed: spawned agent not found')
    } finally {
      act(() => setup.renderer.destroy())
    }
  })

  it('refuses to retry a running agent without calling the daemon', async () => {
    const rpc = vi.fn(async () => ({ ok: true, agent: { status: 'idle' } }))
    const setup = await testRender(
      <GatewayProvider value={servicesWith(rpc)}>
        <AgentPanelOverlay
          history={[]}
          liveAgents={[agent({ status: 'running' })]}
          onClose={() => undefined}
          t={DEFAULT_THEME}
        />
      </GatewayProvider>,
      { height: 24, width: 80 }
    )

    try {
      await setup.flush()
      act(() => setup.mockInput.pressKey('r'))
      await flushRetry(setup)

      expect(rpc).not.toHaveBeenCalled()
      expect(setup.captureCharFrame()).toContain('cannot retry: agent is still running')
    } finally {
      act(() => setup.renderer.destroy())
    }
  })

  it('prefers the dead agent for selection and moves with arrow keys', async () => {
    const rpc = vi.fn(async () => ({ ok: true, agent: { status: 'idle' } }))
    const liveAgents = [
      agent({ id: 'agent-done', name: 'done-worker', status: 'completed', title: 'Done worker' }),
      agent({ id: 'agent-dead', name: 'dead-worker', status: 'failed', title: 'Dead worker' })
    ]
    const setup = await testRender(
      <GatewayProvider value={servicesWith(rpc)}>
        <AgentPanelOverlay history={[]} liveAgents={liveAgents} onClose={() => undefined} t={DEFAULT_THEME} />
      </GatewayProvider>,
      { height: 30, width: 80 }
    )

    try {
      // Two agents do not fit in 30 rows, so the selected row is only on screen
      // after the overlay scrolls it into view — which needs the layout pass that
      // follows the mounting commit. flushRetry settles that extra frame.
      await flushRetry(setup)
      // The dead agent is pre-selected as the most likely retry target.
      expect(setup.captureCharFrame()).toContain('press r to retry this agent')

      // FAILED is its own group now and sorts LAST: a run that broke has
      // already spent its money and does not outrank a result waiting to be
      // read. So the dead agent sits below the completed one, and reaching
      // the completed agent means moving up. (Right opens the inspector
      // rather than advancing the selection.)
      act(() => setup.mockInput.pressKey('ARROW_UP'))
      await flushRetry(setup)
      expect(setup.captureCharFrame()).toContain('press r to run this agent again')

      act(() => setup.mockInput.pressKey('r'))
      await flushRetry(setup)
      expect(rpc).toHaveBeenCalledWith('subagent.retry', { task: 'done-worker' })
    } finally {
      act(() => setup.renderer.destroy())
    }
  })
})

const agentsCommand = agentRetryCommands.find(command => command.name === 'agents')!

const flush = async () => {
  await Promise.resolve()
  await Promise.resolve()
}

function makeSlashCtx(rpc: ReturnType<typeof vi.fn>) {
  const sys: string[] = []
  const ctx = {
    gateway: {
      gw: { request: vi.fn(async () => ({ output: 'daemon agents output' })) },
      rpc
    },
    guarded: (fn: (value: never) => void) => (value: never) => fn(value),
    guardedErr: (error: unknown) => {
      throw error instanceof Error ? error : new Error(String(error))
    },
    stale: () => false,
    transcript: {
      page: () => undefined,
      sys: (text: string) => sys.push(text)
    }
  } as unknown as SlashRunCtx

  return { ctx, sys }
}

describe('/agents retry slash command', () => {
  afterEach(() => resetOverlayState())

  it('prints usage when the retry target is missing', async () => {
    const rpc = vi.fn()
    const { ctx, sys } = makeSlashCtx(rpc)

    agentsCommand.run('retry', ctx, '/agents retry')
    await flush()

    expect(sys).toEqual(['usage: /agents retry <name-or-id> [follow-up message]'])
    expect(rpc).not.toHaveBeenCalled()
  })

  it('retries a dead agent by name and reports the resumed status', async () => {
    const rpc = vi.fn(async () => ({ ok: true, agent: { name: 'dead-worker', status: 'idle', title: 'Dead worker' } }))
    const { ctx, sys } = makeSlashCtx(rpc)

    agentsCommand.run('retry dead-worker', ctx, '/agents retry dead-worker')
    await flush()

    expect(rpc).toHaveBeenCalledWith('subagent.retry', { task: 'dead-worker' })
    expect(sys[0]).toContain('retrying agent `dead-worker`')
    expect(sys[1]).toContain('agent `Dead worker` resumed (idle)')
  })

  it('forwards an optional follow-up message with the retry', async () => {
    const rpc = vi.fn(async () => ({ ok: true, agent: { name: 'dead-worker', status: 'running' } }))
    const { ctx } = makeSlashCtx(rpc)

    agentsCommand.run('retry dead-worker focus on the auth boundary', ctx, '/agents retry dead-worker …')
    await flush()

    expect(rpc).toHaveBeenCalledWith('subagent.retry', {
      message: 'focus on the auth boundary',
      task: 'dead-worker'
    })
  })

  it('surfaces a daemon rejection as an honest error', async () => {
    const rpc = vi.fn(async () => ({ ok: false, error: 'spawned agent not found' }))
    const { ctx, sys } = makeSlashCtx(rpc)

    agentsCommand.run('retry ghost', ctx, '/agents retry ghost')
    await flush()

    expect(sys.at(-1)).toBe('retry failed: spawned agent not found')
  })

  it('keeps the stock dashboard behavior for non-retry forms', async () => {
    const rpc = vi.fn()
    const { ctx, sys } = makeSlashCtx(rpc)

    agentsCommand.run('', ctx, '/agents')
    expect(getOverlayState().agents).toBe(true)

    agentsCommand.run('status', ctx, '/agents status')
    await flush()
    expect(ctx.gateway.gw.request).toHaveBeenCalledWith('slash.exec', expect.objectContaining({ command: 'agents' }))
    expect(sys).toContain('daemon agents output')

    agentsCommand.run('bogus', ctx, '/agents bogus')
    expect(sys.at(-1)).toContain('usage: /agents')
  })
})
