// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { EventEmitter } from 'node:events'
import { writeFileSync, existsSync, readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { execFileSync } from 'node:child_process'
import type { spawn } from 'node:child_process'
import { describe, expect, it, vi } from 'vitest'
import { connectRemoteMachine, parseRemoteMachine, remoteMachineCommand } from '../lib/machineHandoff.js'

const machine = { alias: 'gpu', target: 'me@server', workspacePath: "/work/it's a project; $(touch nope)" }
const child = () => {
  const event = Object.assign(new EventEmitter(), { stdout: new EventEmitter(), stderr: new EventEmitter(), kill: vi.fn() })
  event.kill.mockImplementation(() => { queueMicrotask(() => event.emit('close', null)); return true })
  return event
}

function tunnelAddress(args: string[]): string {
  const control = args[args.indexOf('-S') + 1]!
  writeFileSync(control, '')
  return join(dirname(control), 'rpc.sock') + ':/remote/rpc.sock'
}
function forwardReply() {
  const reply = child()
  queueMicrotask(() => reply.emit('close', 0))
  return reply
}

describe('local TUI over SSH', () => {
  it('forwards a private socket and launches the renderer locally, then cleans up', async () => {
    const setup = child(), tunnel = child(), local = child()
    let socket = ''
    const onSessionId = vi.fn()
    const launch = vi.fn((binary: string, args: string[], options?: { env?: NodeJS.ProcessEnv }) => {
      if (args.includes('-O')) return forwardReply()
      if (args.includes('-M')) {
        socket = tunnelAddress(args).split(':')[0]!
        writeFileSync(socket, '')
        return tunnel
      }
      if (binary === 'ssh') {
        queueMicrotask(() => { setup.stdout.emit('data', 'XERXES_REMOTE_READY {"socketPath":"/remote/rpc.sock","projectDir":"/remote/project"}\n'); setup.emit('close', 0) })
        return setup
      }
      writeFileSync(options!.env!.XERXES_TUI_ACTIVE_SESSION_FILE!, JSON.stringify({ session_id: 'remote-session-1' }))
      queueMicrotask(() => local.emit('close', 0))
      return local
    })
    const suspend = vi.fn(async (action: () => Promise<void>) => action())
    await connectRemoteMachine(machine, { spawnProcess: launch as unknown as typeof spawn, suspend, tuiEntry: '/local/entry.js', onSessionId })
    expect(onSessionId).toHaveBeenCalledWith('remote-session-1')
    expect(launch.mock.calls[0]?.[1]).not.toContain('-t')
    expect(launch.mock.calls[1]?.[1]).toContain('-N')
    // Both subprocesses must override permissive settings inherited from an
    // SSH alias; setup is as sensitive as the long-lived transport.
    for (const call of launch.mock.calls.slice(0, 2)) {
      expect(call[1]).toEqual(expect.arrayContaining(['StrictHostKeyChecking=yes', 'ForwardAgent=no', 'ForwardX11=no']))
      const args = call[1]!
      const destination = args.indexOf('--')
      // ssh -G only evaluates configuration: no connection or credentials.
      const effective = execFileSync('ssh', ['-G', ...args.slice(0, destination),
        '-o', 'StrictHostKeyChecking=no', '-o', 'ForwardAgent=yes', '-o', 'ForwardX11=yes',
        '-F', '/dev/null', 'example.invalid'], { encoding: 'utf8' })
      expect(effective).toMatch(/^stricthostkeychecking true$/m)
      expect(effective).toMatch(/^forwardagent no$/m)
      expect(effective).toMatch(/^forwardx11 no$/m)
    }
    expect(launch).toHaveBeenLastCalledWith(process.execPath, ['/local/entry.js'], expect.objectContaining({ stdio: 'inherit', env: expect.objectContaining({ XERXES_REMOTE_SOCKET: socket, XERXES_PROJECT_DIR: '/remote/project', XERXES_TUI_RESUME: '' }) }))
    expect(suspend).toHaveBeenCalledOnce()
    expect(tunnel.kill).toHaveBeenCalledWith('SIGTERM')
    expect(existsSync(socket)).toBe(false)
    expect(remoteMachineCommand(machine)).toContain('XERXES_REMOTE_READY')
  })
  it('rejects malformed machine responses before spawning', () => {
    for (const target of ['-oProxyCommand=bad', 'server;bad', 'server\nother']) expect(() => parseRemoteMachine({ ...machine, target })).toThrow('Invalid remote')
    expect(() => parseRemoteMachine({ ...machine, workspacePath: 'relative' })).toThrow()
  })
  it('never exposes arbitrary setup diagnostics or malformed handshake contents', async () => {
    for (const [output, code, expected] of [
      ['private-sentinel-credential', 1, 'Remote setup failed'],
      ['Permission denied private-sentinel-credential', 255, 'SSH authentication failed'],
      ['XERXES_REMOTE_READY {private-sentinel-credential}', 0, 'invalid daemon address'],
    ] as const) {
      const setup = child()
      const launch = vi.fn(() => {
        queueMicrotask(() => { setup.stdout.emit('data', output); setup.stderr.emit('data', output); setup.emit('close', code) })
        return setup
      })
      const error = await connectRemoteMachine(machine, { spawnProcess: launch as unknown as typeof spawn }).catch(error => error as Error)
      expect(error).toBeInstanceOf(Error)
      expect((error as Error).message).toContain(expected)
      expect(String(error)).not.toContain('private-sentinel-credential')
      expect(launch).toHaveBeenCalledOnce()
    }
  })
  it('replaces a dropped tunnel without restarting setup or the live renderer', async () => {
    const setup = child(), tunnel = child(), replacement = child(), local = child()
    let restored = false
    let tunnels = 0
    const addresses: string[] = []
    const launch = vi.fn((binary: string, args: string[]) => {
      if (args.includes('-O')) return forwardReply()
      if (args.includes('-M')) {
        const address = tunnelAddress(args)
        addresses.push(address)
        writeFileSync(address.split(':')[0]!, '')
        if (++tunnels === 2) {
          expect(local.kill).not.toHaveBeenCalled()
          queueMicrotask(() => local.emit('close', 0))
          return replacement
        }
        return tunnel
      }
      if (binary === 'ssh') {
        queueMicrotask(() => { setup.stdout.emit('data', 'XERXES_REMOTE_READY {"socketPath":"/remote/rpc.sock","projectDir":"/remote"}\n'); setup.emit('close', 0) })
        return setup
      }
      queueMicrotask(() => tunnel.emit('close', 255))
      return local
    })
    await connectRemoteMachine(machine, { spawnProcess: launch as unknown as typeof spawn, suspend: async action => { try { await action() } finally { restored = true } } })
    expect(addresses[0]).toBe(addresses[1])
    expect(launch.mock.calls.filter(call => call[0] !== 'ssh')).toHaveLength(1)
    expect(launch.mock.calls.filter(call => call[0] === 'ssh' && !call[1].includes('-M') && !call[1].includes('-O'))).toHaveLength(1)
    expect(local.kill).not.toHaveBeenCalled()
    expect(replacement.kill).toHaveBeenCalledWith('SIGTERM')
    expect(restored).toBe(true)
  })
  it('reports setup errors without suspending the local renderer', async () => {
    const suspend = vi.fn()
    const launch = () => { const proc = child(); queueMicrotask(() => proc.emit('error', new Error('missing ssh'))); return proc }
    await expect(connectRemoteMachine(machine, { spawnProcess: launch as unknown as typeof spawn, suspend })).rejects.toThrow('Could not start SSH')
    expect(suspend).not.toHaveBeenCalled()
  })
  it('leaves the renderer available after authentication failure and never restarts it on exit', async () => {
    const { reconnectRemoteMachine } = await import('../lib/machineHandoff.js')
    const setup = child(), tunnel = child(), local = child()
    const launch = vi.fn((binary: string, args: string[], options?: { env?: NodeJS.ProcessEnv }) => {
      if (args.includes('-O')) return forwardReply()
      if (args.includes('-M')) { writeFileSync(tunnelAddress(args).split(':')[0]!, ''); return tunnel }
      if (binary === 'ssh') {
        queueMicrotask(() => { setup.stdout.emit('data', 'XERXES_REMOTE_READY {"socketPath":"/remote/rpc.sock","projectDir":"/remote"}\n'); setup.emit('close', 0) })
        return setup
      }
      queueMicrotask(() => {
        tunnel.stderr.emit('data', 'Permission denied. sensitive-remote-diagnostic')
        tunnel.emit('close', 255)
        expect(local.kill).not.toHaveBeenCalled()
        const status = readFileSync(options!.env!.XERXES_REMOTE_STATUS_FILE!, 'utf8')
        expect(status).toContain('authentication failed')
        expect(status).not.toContain('sensitive-remote-diagnostic')
        // The user, not transport loss, ends this renderer.
        local.emit('close', 0)
      })
      return local
    })
    await expect(reconnectRemoteMachine(machine, { spawnProcess: launch as unknown as typeof spawn, suspend: action => action() })).rejects.toThrow('authentication failed')
    expect(launch).toHaveBeenCalledTimes(4)
    expect(local.kill).not.toHaveBeenCalled()
  })
  it('cancels a pending tunnel retry when the remote renderer exits', async () => {
    const setup = child(), tunnel = child(), local = child()
    const launch = vi.fn((binary: string, args: string[]) => {
      if (args.includes('-O')) return forwardReply()
      if (args.includes('-M')) { writeFileSync(tunnelAddress(args).split(':')[0]!, ''); return tunnel }
      if (binary === 'ssh') {
        queueMicrotask(() => { setup.stdout.emit('data', 'XERXES_REMOTE_READY {"socketPath":"/remote/rpc.sock","projectDir":"/remote"}\n'); setup.emit('close', 0) })
        return setup
      }
      queueMicrotask(() => { tunnel.emit('close', 255); local.emit('close', 0) })
      return local
    })
    await expect(connectRemoteMachine(machine, { spawnProcess: launch as unknown as typeof spawn, suspend: action => action() })).rejects.toThrow('reconnection failed')
    await new Promise(resolve => setTimeout(resolve, 300))
    expect(launch).toHaveBeenCalledTimes(4)
  })
  it('cancels before setup without starting any process', async () => {
    const controller = new AbortController(); controller.abort()
    const launch = vi.fn()
    await expect(connectRemoteMachine(machine, { signal: controller.signal, spawnProcess: launch as unknown as typeof spawn })).rejects.toThrow('cancelled')
    expect(launch).not.toHaveBeenCalled()
  })
})

it('retries transient drops and resumes the last remote session', async () => {
  const { reconnectRemoteMachine } = await import('../lib/machineHandoff.js')
  let calls = 0
  const connect = vi.fn(async (_machine, options) => {
    if (++calls === 1) { options?.onSessionId?.('session-123'); throw new Error('SSH tunnel closed (255). Connection reset') }
  }) as unknown as typeof connectRemoteMachine
  const wait = vi.fn(async () => {})
  const progress = vi.fn()
  await reconnectRemoteMachine(machine, { onProgress: progress }, connect, wait)
  expect(connect).toHaveBeenLastCalledWith(machine, expect.objectContaining({ resumeSessionId: 'session-123' }))
  expect(wait).toHaveBeenCalledWith(2000, undefined)
  expect(progress).toHaveBeenCalledWith(expect.stringContaining('Retry 1/3'))
})
it('bounds retries and does not retry authentication failures or cancellation', async () => {
  const { reconnectRemoteMachine } = await import('../lib/machineHandoff.js')
  const connect = vi.fn(async () => { throw new Error('SSH tunnel closed (255)') })
  const wait = vi.fn(async () => {})
  await expect(reconnectRemoteMachine(machine, {}, connect, wait)).rejects.toThrow('SSH tunnel closed')
  expect(connect).toHaveBeenCalledTimes(4)
  const denied = vi.fn(async () => { throw new Error('SSH tunnel closed (255). Permission denied') })
  await expect(reconnectRemoteMachine(machine, {}, denied, wait)).rejects.toThrow('Permission denied')
  expect(denied).toHaveBeenCalledOnce()
  const controller = new AbortController()
  const cancelWait = async () => { controller.abort(); controller.signal.throwIfAborted() }
  connect.mockClear()
  await expect(reconnectRemoteMachine(machine, { signal: controller.signal }, connect, cancelWait)).rejects.toThrow()
  expect(connect).toHaveBeenCalledOnce()
})

it('does not launch a local TUI if the tunnel drops while suspending the parent renderer', async () => {
  const setup = child(), tunnel = child(), local = child()
  const launch = vi.fn((binary: string, args: string[]) => {
    if (args.includes('-O')) return forwardReply()
      if (args.includes('-M')) { writeFileSync(tunnelAddress(args).split(':')[0]!, ''); return tunnel }
    if (binary === 'ssh') {
      queueMicrotask(() => { setup.stdout.emit('data', 'XERXES_REMOTE_READY {"socketPath":"/remote/rpc.sock","projectDir":"/remote"}\n'); setup.emit('close', 0) })
      return setup
    }
    queueMicrotask(() => local.emit('close', 0)); return local
  })
  await expect(connectRemoteMachine(machine, { spawnProcess: launch as unknown as typeof spawn, suspend: async action => { tunnel.emit('close', 255); await action() } })).rejects.toThrow('SSH tunnel closed')
  expect(launch).toHaveBeenCalledTimes(3)
})
it('cancels remote setup promptly even if SSH ignores termination', async () => {
  const setup = child()
  setup.kill.mockImplementation(() => true)
  const controller = new AbortController()
  const work = connectRemoteMachine(machine, { signal: controller.signal, spawnProcess: (() => { queueMicrotask(() => controller.abort()); return setup }) as unknown as typeof spawn }).then(() => 'success', () => 'cancelled')
  try {
    const result = await Promise.race([work, new Promise<string>(resolve => setTimeout(() => resolve('hung'), 100))])
    expect(result).toBe('cancelled')
  } finally { setup.emit('close', null); await work }
})

for (const outcome of ['open', 'cancel', 'disconnect', 'failure'] as const) it(`prepares before terminal suspension and cleans up after ${outcome}`, async () => {
  const setup=child(),tunnel=child(),local=child(),controller=new AbortController()
  const close=vi.fn(async()=>{})
  const order:string[]=[]
  const launch=vi.fn((binary:string,args:string[],options?:{env?:NodeJS.ProcessEnv})=>{
    if(args.includes('-O'))return forwardReply()
    if(args.includes('-M')){writeFileSync(tunnelAddress(args).split(':')[0]!, '');return tunnel}
    if(binary==='ssh'){
      queueMicrotask(()=>{setup.stdout.emit('data','XERXES_REMOTE_READY {"socketPath":"/remote/rpc.sock","projectDir":"/remote/project"}\n');setup.emit('close',0)})
      return setup
    }
    order.push('renderer')
    expect(options?.env?.XERXES_TUI_RESUME).toBe('prepared-id')
    expect(options?.env?.XERXES_TUI_PREPARED_SESSION_KEY).toBe('tui:prepared-key')
    queueMicrotask(()=>local.emit('close',0))
    return local
  })
  const run=connectRemoteMachine(machine,{spawnProcess:launch as unknown as typeof spawn,signal:controller.signal,
    suspend:async action=>{order.push('suspend');await action()},
    prepare:async remote=>{
      order.push('review')
      expect(remote.projectDir).toBe('/remote/project')
      expect(existsSync(remote.socketPath)).toBe(true)
      if(outcome==='failure')throw new Error('Review setup failed')
      if(outcome==='cancel')controller.abort()
      if(outcome==='disconnect'){
        tunnel.emit('close',255)
        expect(remote.signal.aborted).toBe(true)
      }
      return {sessionId:'prepared-id',sessionKey:'tui:prepared-key',close}
    }})
  if(outcome==='open')await run
  else await expect(run).rejects.toThrow()
  expect(order).toEqual(outcome==='open'?['review','suspend','renderer']:['review'])
  expect(close).toHaveBeenCalledTimes(outcome==='failure'?0:1)
  expect(tunnel.kill).toHaveBeenCalled()
})
