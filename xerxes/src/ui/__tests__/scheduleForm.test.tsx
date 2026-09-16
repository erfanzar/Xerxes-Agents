// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import { testRender } from '@opentui/react/test-utils'
import { act } from 'react'
import { expect, it, vi } from 'vitest'
import { GatewayProvider } from '../app/gatewayContext.js'
import type { GatewayServices } from '../app/interfaces.js'
import { ScheduleForm } from '../opentui/scheduleForm.js'
import { DARK_THEME } from '../theme.js'
it('previews timing without saving and ignores obsolete responses', async () => {
  let resolveOld!: (value: unknown) => void
  const rpc = vi.fn().mockImplementationOnce(() => new Promise(resolve => { resolveOld = resolve }))
    .mockResolvedValue({ ok: false, error: 'Enter a future ISO time' })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><ScheduleForm t={DARK_THEME} onClose={() => {}} onSaved={() => {}} /></GatewayProvider>, { width: 110, height: 35 })
  try {
    await vi.waitFor(() => expect(rpc).toHaveBeenCalledWith('schedule.preview', { timezone: 'UTC', schedule: '0 9 * * *' }))
    act(() => screen.mockInput.pressKey('TAB'))
    await screen.flush()
    act(() => screen.mockInput.pressArrow('right'))
    await screen.flush()
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).toContain('Enter a future ISO time') })
    await act(async () => resolveOld({ ok: true, next_run_at: '2099-01-01T09:00:00.000Z' }))
    await screen.flush()
    expect(screen.captureCharFrame()).not.toContain('2099-01-01')
    expect(screen.captureCharFrame()).toContain('Paused · enable when ready')
    expect(rpc.mock.calls.every(call => call[0] === 'schedule.preview')).toBe(true)
  } finally { act(() => screen.renderer.destroy()) }
})

it('shows the next eligible UTC instant before saving', async () => {
  const rpc = vi.fn(async () => ({ ok: true, next_run_at: '2099-01-01T09:00:00.000Z' }))
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><ScheduleForm t={DARK_THEME} onClose={() => {}} onSaved={() => {}} /></GatewayProvider>, { width: 110, height: 35 })
  try {
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).toContain('2099-01-01T09:00:00.000Z') })
  } finally { act(() => screen.renderer.destroy()) }
})
it.each([[150, 40], [40, 18]])('creates a paused schedule at %ix%i', async (width, height) => {
  const saved = vi.fn()
  const rpc = vi.fn(async () => ({ ok: true, job: { id: 'created' } }))
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><ScheduleForm t={DARK_THEME} onClose={() => {}} onSaved={saved} /></GatewayProvider>, { width, height })
  try {
    await screen.flush()
    await act(async () => screen.mockInput.typeText('Review changes'))
    await screen.flush()
    act(() => screen.mockInput.pressKey('RETURN'))
    await screen.flush()
    await vi.waitFor(() => expect(saved).toHaveBeenCalledWith('created'))
    expect(rpc).toHaveBeenCalledWith('schedule.create', { timezone: "UTC", missed_run_policy: "coalesce", misfire_grace_seconds: 300, deliver: "none", recipient: "", max_model_calls: null, max_runs: null, expires_at: null, target: "independent", timeout_seconds: 300, max_retries: 3, prompt: 'Review changes', schedule: '0 9 * * *', paused: true })
  } finally { act(() => screen.renderer.destroy()) }
})
it('retains an edited prompt after stale revision rejection', async () => {
  const rpc = vi.fn(async () => ({ ok: false, error: 'Schedule changed; refresh before editing' }))
  const saved = vi.fn()
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><ScheduleForm t={DARK_THEME} initial={{ id: 'job', revision: 'old', prompt: 'Review', schedule: '0 8 * * *', timezone: 'America/New_York', paused: false }} onClose={() => {}} onSaved={saved} /></GatewayProvider>, { width: 110, height: 35 })
  try {
    await screen.flush()
    act(() => screen.mockInput.pressKey('RETURN'))
    await screen.flush()
    await vi.waitFor(() => expect(screen.captureCharFrame()).toContain('Schedule changed'))
    expect(screen.captureCharFrame()).toContain('Review')
    expect(saved).not.toHaveBeenCalled()
    expect(rpc).toHaveBeenCalledWith('schedule.update', { timezone: "America/New_York", missed_run_policy: "coalesce", misfire_grace_seconds: 300, deliver: "none", recipient: "", max_model_calls: null, max_runs: null, expires_at: null, target: "independent", timeout_seconds: 300, max_retries: 3, schedule_id: 'job', revision: 'old', prompt: 'Review', schedule: '0 8 * * *', paused: false })
  } finally { act(() => screen.renderer.destroy()) }
})

it('edits an interval schedule without converting its timing mode', async () => {
  const rpc = vi.fn(async () => ({ ok: true, job: { id: 'interval' } }))
  const saved = vi.fn()
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><ScheduleForm t={DARK_THEME} initial={{ id: 'interval', revision: 'v1', prompt: 'Watch build', schedule: '', interval_seconds: 30, paused: true }} onClose={() => {}} onSaved={saved} /></GatewayProvider>, { width: 110, height: 35 })
  try {
    await screen.flush()
    expect(screen.captureCharFrame()).toContain('Interval seconds: 30')
    act(() => screen.mockInput.pressKey('RETURN'))
    await screen.flush()
    await vi.waitFor(() => expect(saved).toHaveBeenCalledWith('interval'))
    expect(rpc).toHaveBeenCalledWith('schedule.update', { timezone: "UTC", schedule_id: 'interval', revision: 'v1', prompt: 'Watch build', interval_seconds: 30, paused: true, missed_run_policy: "coalesce", misfire_grace_seconds: 300, deliver: "none", recipient: "", max_model_calls: null, max_runs: null, expires_at: null, target: "independent", timeout_seconds: 300, max_retries: 3 })
  } finally { act(() => screen.renderer.destroy()) }
})

it('lets the keyboard select and edit the recurring timezone', async () => {
  const rpc = vi.fn(async () => ({ ok: true, job: { id: 'zone' } }))
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><ScheduleForm t={DARK_THEME} initial={{ id: 'zone', revision: 'v1', prompt: 'Review', schedule: '0 9 * * *', timezone: '', paused: true }} onClose={() => {}} onSaved={() => {}} /></GatewayProvider>, { width: 150, height: 40 })
  try {
    await screen.flush()
    for (let index = 0; index < 6; index++) {
      act(() => screen.mockInput.pressKey('TAB'))
      await screen.flush()
    }
    expect(screen.captureCharFrame()).toContain('› Recurring timezone:')
    await act(async () => screen.mockInput.typeText('Asia/Tokyo'))
    await screen.flush()
    act(() => screen.mockInput.pressKey('RETURN'))
    await screen.flush()
    await vi.waitFor(() => expect(rpc).toHaveBeenCalledWith('schedule.update', expect.objectContaining({ timezone: 'Asia/Tokyo', schedule: '0 9 * * *' })))
  } finally { act(() => screen.renderer.destroy()) }
})

it('loads and saves the selected missed-run policy using the keyboard', async () => {
  const saved = vi.fn()
  const rpc = vi.fn(async () => ({ ok: true, job: { id: 'policy' }, next_run_at: '2099-01-01T09:00:00.000Z' }))
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><ScheduleForm t={DARK_THEME} initial={{ id: 'policy', revision: 'v1', prompt: 'Review', schedule: '0 9 * * *', paused: true, missed_run_policy: 'skip', misfire_grace_seconds: 45 }} onClose={() => {}} onSaved={saved} /></GatewayProvider>, { width: 110, height: 35 })
  try {
    await screen.flush()
    expect(screen.captureCharFrame()).toContain('Missed runs: Skip overdue occurrences')
    expect(screen.captureCharFrame()).toContain('Lateness allowance seconds: 45')
    for (let index = 0; index < 7; index++) {
      act(() => screen.mockInput.pressKey('TAB'))
      await screen.flush()
    }
    act(() => screen.mockInput.pressArrow('right'))
    await screen.flush()
    expect(screen.captureCharFrame()).toContain('Missed runs: Run once after returning')
    act(() => screen.mockInput.pressKey('RETURN'))
    await vi.waitFor(() => expect(saved).toHaveBeenCalledWith('policy'))
    expect(rpc).toHaveBeenCalledWith('schedule.update', expect.objectContaining({ missed_run_policy: 'coalesce', misfire_grace_seconds: 45 }))
  } finally { act(() => screen.renderer.destroy()) }
})

it.each([[110, 35], [40, 18]])('selects a configured delivery channel and saves its recipient at %ix%i', async (width, height) => {
  const saved = vi.fn()
  const rpc = vi.fn(async (method: string) => method === 'schedule.options'
    ? { ok: true, destinations: [{ name: 'none', enabled: true }, { name: 'recording', enabled: false }] }
    : { ok: true, job: { id: 'delivery' }, next_run_at: '2099-01-01T09:00:00.000Z' })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><ScheduleForm t={DARK_THEME} initial={{ id: 'delivery', revision: 'v1', prompt: 'Review', schedule: '0 9 * * *', paused: true }} onClose={() => {}} onSaved={saved} /></GatewayProvider>, { width, height })
  try {
    await screen.flush()
    for (let index = 0; index < 9; index++) {
      act(() => screen.mockInput.pressKey('TAB'))
      await screen.flush()
    }
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).toContain('recording') })
    act(() => screen.mockInput.pressArrow('down'))
    await screen.flush()
    expect(screen.captureCharFrame()).toContain('Delivery channel: recording')
    act(() => screen.mockInput.pressKey('TAB'))
    await screen.flush()
    await act(async () => screen.mockInput.typeText('room-42'))
    await screen.flush()
    act(() => screen.mockInput.pressKey('RETURN'))
    await vi.waitFor(() => expect(saved).toHaveBeenCalledWith('delivery'))
    expect(rpc).toHaveBeenCalledWith('schedule.update', expect.objectContaining({ deliver: 'recording', recipient: 'room-42', paused: true }))
    act(() => screen.mockInput.pressKey('TAB', { shift: true }))
    await screen.flush()
    act(() => screen.mockInput.pressArrow('up'))
    await screen.flush()
    act(() => screen.mockInput.pressKey('RETURN'))
    await vi.waitFor(() => expect(rpc).toHaveBeenCalledWith('schedule.update', expect.objectContaining({ deliver: 'none', recipient: '' })))
  } finally { act(() => screen.renderer.destroy()) }
})

it.each(['max_model_calls', 'max_runs', 'expires_at'] as const)('saves and removes %s without changing other settings', async limit => {
  const rpc = vi.fn(async () => ({ ok: true, job: { id: 'limited' }, next_run_at: '2099-01-01T09:00:00.000Z' }))
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><ScheduleForm t={DARK_THEME} initial={{ id: 'limited', revision: 'v1', prompt: 'Review', schedule: '0 9 * * *', paused: true }} onClose={() => {}} onSaved={() => {}} /></GatewayProvider>, { width: 80, height: 18 })
  try {
    await screen.flush()
    act(() => screen.mockInput.pressKey('TAB', { shift: true })); await screen.flush()
    act(() => screen.mockInput.pressKey('TAB', { shift: true })); await screen.flush()
    act(() => screen.mockInput.pressKey('TAB', { shift: true })); await screen.flush()
    act(() => screen.mockInput.pressKey('TAB', { shift: true })); await screen.flush()
    if (limit !== 'expires_at') { act(() => screen.mockInput.pressKey('TAB', { shift: true })); await screen.flush() }
    if (limit === 'max_model_calls') { act(() => screen.mockInput.pressKey('TAB', { shift: true })); await screen.flush() }
    await act(async () => screen.mockInput.typeText(limit === 'expires_at' ? '2099-01-01T00:00:00Z' : '3'))
    await screen.flush()
    act(() => screen.mockInput.pressKey('RETURN'))
    await vi.waitFor(() => expect(rpc).toHaveBeenCalledWith('schedule.update', expect.objectContaining({ [limit]: limit === 'expires_at' ? '2099-01-01T00:00:00Z' : 3 })))
    // The RPC spy observes submission before React commits busy/focus changes.
    // Flush inside the retry so a stale pre-save frame cannot pass this check
    // while the editor is still disabled and discard the following keystrokes.
    await vi.waitFor(async () => {
      await screen.flush()
      expect(screen.captureCharFrame()).not.toContain('Saving…')
    })
    for (let i = 0; i < (limit === 'expires_at' ? 20 : 1); i++) act(() => screen.mockInput.pressKey('BACKSPACE'))
    await screen.flush()
    act(() => screen.mockInput.pressKey('RETURN'))
    await vi.waitFor(() => expect(rpc).toHaveBeenCalledWith('schedule.update', expect.objectContaining({ [limit]: null })))
  } finally { act(() => screen.renderer.destroy()) }
})
it('selects a bounded follow-up in this conversation', async () => {
  const rpc = vi.fn(async () => ({ ok: true, job: { id: 'followup' }, next_run_at: '2099-01-01T00:00:00Z' }))
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><ScheduleForm t={DARK_THEME} initial={{ id: 'followup', revision: 'v1', prompt: 'Check back', schedule: '', interval_seconds: 60, paused: true, max_runs: 3, expires_at: '2099-01-01T00:00:00Z' }} onClose={() => {}} onSaved={() => {}} /></GatewayProvider>, { width: 80, height: 18 })
  try {
    await screen.flush()
    act(() => screen.mockInput.pressKey('TAB', { shift: true })); await screen.flush()
    act(() => screen.mockInput.pressKey('TAB', { shift: true })); await screen.flush()
    act(() => screen.mockInput.pressKey('TAB', { shift: true })); await screen.flush()
    act(() => screen.mockInput.pressArrow('right')); await screen.flush()
    expect(screen.captureCharFrame()).toContain('This conversation')
    act(() => screen.mockInput.pressKey('RETURN'))
    await vi.waitFor(() => expect(rpc).toHaveBeenCalledWith('schedule.update', expect.objectContaining({ target: 'session', max_runs: 3, expires_at: '2099-01-01T00:00:00Z' })))
  } finally { act(() => screen.renderer.destroy()) }
})
it('creates a paused conversation loop with explicit bounded defaults', async () => {
  const rpc = vi.fn(async () => ({ ok: true, job: { id: 'loop' }, next_run_at: '2099-01-01T00:00:00Z' }))
  const before = Date.now()
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><ScheduleForm t={DARK_THEME} followup onClose={() => {}} onSaved={() => {}} /></GatewayProvider>, { width: 80, height: 18 })
  try {
    await screen.flush()
    await act(async () => screen.mockInput.typeText('Check deployment'))
    await screen.flush(); act(() => screen.mockInput.pressKey('RETURN'))
    await vi.waitFor(() => expect(rpc).toHaveBeenCalledWith('schedule.create', expect.objectContaining({ target: 'session', paused: true, interval_seconds: 600, max_runs: 10, prompt: 'Check deployment' })))
    const request = (rpc.mock.calls as unknown as Array<[string, Record<string, unknown>]>).find(call => call[0] === 'schedule.create')![1]
    expect(Date.parse(String(request.expires_at))).toBeGreaterThanOrEqual(before + 86400000)
    expect(Date.parse(String(request.expires_at))).toBeLessThanOrEqual(Date.now() + 86400000)
  } finally { act(() => screen.renderer.destroy()) }
})

it.each([[220, 65], [40, 18]])('edits and clears a follow-up stop condition at %ix%i', async (width, height) => {
  const rpc = vi.fn(async () => ({ ok: true, job: { id: 'check' }, next_run_at: '2099-01-01T00:00:00Z' }))
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><ScheduleForm t={DARK_THEME}
    initial={{ id: 'check', revision: 'v1', prompt: 'Check deployment', schedule: '', paused: true, interval_seconds: 60, target_session_id: 'owner', max_runs: 3, expires_at: '2099-01-01T00:00:00Z', stop_condition: 'Healthy' }}
    onClose={() => {}} onSaved={() => {}} /></GatewayProvider>, { width, height })
  try {
    await screen.flush()
    act(() => screen.mockInput.pressKey('TAB', { shift: true })); await screen.flush()
    act(() => screen.mockInput.pressKey('TAB', { shift: true })); await screen.flush()
    expect(screen.captureCharFrame()).toContain('Stop condition')
    act(() => screen.mockInput.pressKey('END')); await screen.flush()
    for (let i = 0; i < 7; i++) act(() => screen.mockInput.pressKey('BACKSPACE'))
    await screen.flush()
    act(() => screen.mockInput.pressKey('RETURN'))
    await vi.waitFor(() => expect(rpc).toHaveBeenCalledWith('schedule.update', expect.objectContaining({ stop_condition: null, target: 'session' })))
    await vi.waitFor(async () => {
      await screen.flush()
      expect(screen.captureCharFrame()).not.toContain('Saving…')
    })
    await act(async () => screen.mockInput.typeText('All checks pass'))
    await screen.flush()
    act(() => screen.mockInput.pressKey('RETURN'))
    await vi.waitFor(() => expect(rpc).toHaveBeenCalledWith('schedule.update', expect.objectContaining({ stop_condition: 'All checks pass' })))
  } finally { act(() => screen.renderer.destroy()) }
})

it.each([[220, 65], [40, 18]])('edits, validates and clears the lifetime token threshold at %ix%i', async (width, height) => {
  const rpc = vi.fn(async () => ({ ok: true, job: { id: 'budget' }, next_run_at: '2099-01-01T00:00:00Z' }))
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><ScheduleForm t={DARK_THEME}
    initial={{ id: 'budget', revision: 'v1', prompt: 'Check', schedule: '', paused: true, max_total_tokens: 100 }}
    onClose={() => {}} onSaved={() => {}} /></GatewayProvider>, { width, height })
  try {
    await screen.flush()
    act(() => screen.mockInput.pressKey('TAB', { shift: true })); await screen.flush()
    expect(screen.captureCharFrame()).toContain('Lifetime token')
    act(() => screen.mockInput.pressKey('END')); await screen.flush()
    for (let i = 0; i < 3; i++) act(() => screen.mockInput.pressKey('BACKSPACE'))
    await screen.flush()
    await act(async () => screen.mockInput.typeText('bad')); await screen.flush()
    act(() => screen.mockInput.pressKey('RETURN')); await screen.flush()
    expect(rpc.mock.calls.some(call => (call as unknown[])[0] === 'schedule.update')).toBe(false)
    for (let i = 0; i < 3; i++) act(() => screen.mockInput.pressKey('BACKSPACE'))
    await screen.flush()
    act(() => screen.mockInput.pressKey('RETURN'))
    await vi.waitFor(() => expect(rpc).toHaveBeenCalledWith('schedule.update', expect.objectContaining({ max_total_tokens: null })))
    await vi.waitFor(async () => {
      await screen.flush()
      expect(screen.captureCharFrame()).not.toContain('Saving…')
    })
    await act(async () => screen.mockInput.typeText('250')); await screen.flush()
    act(() => screen.mockInput.pressKey('RETURN'))
    await vi.waitFor(() => expect(rpc).toHaveBeenCalledWith('schedule.update', expect.objectContaining({ max_total_tokens: 250 })))
  } finally { act(() => screen.renderer.destroy()) }
})
