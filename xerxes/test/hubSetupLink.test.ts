// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, describe, expect, test } from 'bun:test'
import { readFile, stat, writeFile } from 'node:fs/promises'
import { join } from 'node:path'

import { daemonProjects, runHubCli } from '../src/hub/cli.js'
import type { CommandResult, CommandRunner } from '../src/hub/commandRunner.js'
import { HubLinkError, runHubLink, runLinkTunnel, validateSshTarget } from '../src/hub/hubLink.js'
import { HubSetupError, parseTailscaleStatus, runHubSetup } from '../src/hub/hubSetup.js'
import { hubPaths } from '../src/hub/paths.js'
import { installUserService, renderLaunchAgent, renderSystemdUnit, UserServiceError } from '../src/hub/serviceUnits.js'
import { HubSettingsError, hubSettingsPath, loadHubSettings, saveHubSettings, settingsEnvironment } from '../src/hub/settings.js'
import { WorkspaceRegistry } from '../src/hub/workspaceRegistry.js'
import { disposeAll, registerCleanup, tempDir } from './hubHarness.js'

afterEach(disposeAll)

function fakeRunner(handler: (argv: readonly string[]) => Partial<CommandResult> | undefined) {
  const calls: string[][] = []
  const run: CommandRunner = async argv => {
    calls.push([...argv])
    return { code: 0, stdout: '', stderr: '', timedOut: false, ...(handler(argv) ?? {}) }
  }
  return { run, calls }
}
const noSleep = async (): Promise<void> => {}

describe('hub settings', () => {
  test('round-trips with 0600, environment variables win, and a broken file is refused untouched', async () => {
    const dir = await tempDir('xh-settings-')
    const paths = { stateDir: dir }
    expect(await loadHubSettings(paths)).toEqual({})
    await saveHubSettings(paths, { publicOrigin: 'https://hub.tail1.ts.net', port: 8787 })
    expect(await loadHubSettings(paths)).toEqual({ publicOrigin: 'https://hub.tail1.ts.net', port: 8787 })
    expect((await stat(hubSettingsPath(paths))).mode & 0o777).toBe(0o600)
    expect(settingsEnvironment({ XERXES_HUB_PORT: '9000' }, await loadHubSettings(paths))).toMatchObject({ XERXES_HUB_PORT: '9000', XERXES_HUB_PUBLIC_ORIGINS: 'https://hub.tail1.ts.net' })
    await writeFile(hubSettingsPath(paths), '{nope')
    await expect(loadHubSettings(paths)).rejects.toThrow(HubSettingsError)
    expect(await readFile(hubSettingsPath(paths), 'utf8')).toBe('{nope')
    await writeFile(hubSettingsPath(paths), JSON.stringify({ version: 1, public_origin: 'http://x/path' }))
    await expect(loadHubSettings(paths)).rejects.toThrow('exact origin')
    await writeFile(hubSettingsPath(paths), JSON.stringify({ version: 1, secret: 'x' }))
    await expect(loadHubSettings(paths)).rejects.toThrow('unknown setting')
  })
})

describe('user service units', () => {
  const spec = { name: 'svc', label: 'dev.x.svc', description: 'Test 100%', argv: ['/opt/bun', '/a/$HOME/%h', 'run'], environment: { PATH: '/usr/bin' }, logPath: '/tmp/x.log' }

  test('systemd units quote words and never expand % or $', () => {
    const unit = renderSystemdUnit(spec)
    expect(unit).toContain('Description=Test 100%%')
    expect(unit).toContain('ExecStart="/opt/bun" "/a/$$HOME/%%h" "run"')
    expect(unit).toContain('Environment="PATH=/usr/bin"')
    expect(unit).toContain('Restart=always')
  })

  test('launchd agents escape XML and keep the job alive', () => {
    const plist = renderLaunchAgent({ ...spec, argv: ['/opt/bun', '<&>'] })
    expect(plist).toContain('<string>&lt;&amp;&gt;</string>')
    expect(plist).toContain('<key>KeepAlive</key><true/>')
    expect(plist).toContain('<key>Label</key><string>dev.x.svc</string>')
  })

  test('refuses relative executables, control characters, bad names and unsupported platforms', async () => {
    expect(() => renderSystemdUnit({ ...spec, argv: ['bun'] })).toThrow(UserServiceError)
    expect(() => renderSystemdUnit({ ...spec, argv: ['/opt/bun', 'a\nb'] })).toThrow(UserServiceError)
    expect(() => renderSystemdUnit({ ...spec, name: 'Bad Name' })).toThrow(UserServiceError)
    expect(() => renderSystemdUnit({ ...spec, environment: { 'bad-name': 'x' } })).toThrow(UserServiceError)
    const home = await tempDir('xh-unit-')
    await expect(installUserService(spec, { platform: 'win32', home, run: fakeRunner(() => undefined).run, sleep: noSleep })).rejects.toThrow(/Linux .*macOS/)
  })

  test('a failed systemctl step is reported, not ignored', async () => {
    const home = await tempDir('xh-unit-')
    const fake = fakeRunner(argv => (argv[2] === 'restart' ? { code: 1, stderr: 'Failed to connect to bus' } : undefined))
    await expect(installUserService({ ...spec, logPath: join(home, 'x.log') }, { platform: 'linux', home, run: fake.run, sleep: noSleep })).rejects.toThrow('Failed to connect to bus')
  })
})

describe('hub setup', () => {
  test('reads the tailnet name, waits for Serve approval, saves settings, creates the owner, installs the service and prints a pairing link; rerunning keeps the owner', async () => {
    const home = await tempDir('xh-setup-')
    const paths = hubPaths({ XERXES_HOME: join(home, '.xerxes') })
    let serveCalls = 0
    const fake = fakeRunner(argv => {
      if (argv[0] !== 'tailscale') return undefined
      if (argv[1] === 'status') return { stdout: JSON.stringify({ BackendState: 'Running', Self: { DNSName: 'hub.tail1.ts.net.' } }) }
      if (argv[1] === 'serve') {
        serveCalls += 1
        return serveCalls === 1 ? { code: 1, stderr: 'Serve is not enabled on your tailnet.\nTo enable, visit:\n\n  https://login.tailscale.com/f/serve?node=abc\n' } : {}
      }
      return undefined
    })
    const lines: string[] = []
    const deps = { platform: 'linux' as const, home, run: fake.run, sleep: noSleep, tailscale: 'tailscale', fetchHealth: async () => true, out: (line: string) => lines.push(line), err: (line: string) => lines.push(line) }
    const options = { paths, environment: { USER: 'erfan', PATH: '/usr/bin' }, executable: '/opt/bun', scriptPath: '/opt/xerxes/cli.js', port: 8787 }
    const result = await runHubSetup(options, deps)
    expect(result.origin).toBe('https://hub.tail1.ts.net')
    expect(serveCalls).toBe(2)
    expect(fake.calls).toContainEqual(['tailscale', 'serve', '--bg', '--https=443', 'http://127.0.0.1:8787'])
    const text = lines.join('\n')
    expect(text).toContain('https://login.tailscale.com/f/serve?node=abc')
    expect(text).toContain('Created the owner "erfan"')
    expect(text).toContain('https://hub.tail1.ts.net/pair#')
    expect(text).toContain(`xerxes hub pair confirm ${result.pairingId}`)
    expect(await loadHubSettings(paths)).toEqual({ publicOrigin: 'https://hub.tail1.ts.net', host: '127.0.0.1', port: 8787 })
    const unit = await readFile(join(home, '.config', 'systemd', 'user', 'xerxes-hub.service'), 'utf8')
    expect(unit).toContain('ExecStart="/opt/bun" "/opt/xerxes/cli.js" "hub" "run"')
    expect(fake.calls).toContainEqual(['systemctl', '--user', 'restart', 'xerxes-hub.service'])

    lines.length = 0
    const again = await runHubSetup({ ...options, pair: false }, deps)
    expect(again.ownerId).toBe(result.ownerId)
    expect(again.pairingId).toBeUndefined()
    expect(lines.join('\n')).toContain('The owner already exists')
  })

  test('stops with the next step when Tailscale is missing, signed out or has no name', async () => {
    const home = await tempDir('xh-setup-')
    const paths = hubPaths({ XERXES_HOME: home })
    const run = (handler: (argv: readonly string[]) => Partial<CommandResult> | undefined) =>
      runHubSetup({ paths, environment: {}, executable: '/opt/bun', scriptPath: '/opt/cli.js' },
        { platform: 'linux', home, run: fakeRunner(handler).run, sleep: noSleep, tailscale: 'tailscale', fetchHealth: async () => true, out: () => {}, err: () => {} })
    await expect(run(argv => (argv[1] === 'version' ? { code: 127 } : undefined))).rejects.toThrow(/not installed/)
    await expect(run(argv => (argv[1] === 'status' ? { stdout: JSON.stringify({ BackendState: 'NeedsLogin' }) } : undefined))).rejects.toThrow(HubSetupError)
    await expect(run(argv => (argv[1] === 'status' ? { stdout: JSON.stringify({ BackendState: 'Running', Self: {} }) } : undefined))).rejects.toThrow(/MagicDNS/)
    // Nothing was written before Tailscale was ready.
    expect(await loadHubSettings(paths)).toEqual({})
  })

  test('parses tailscale status', () => {
    expect(parseTailscaleStatus(JSON.stringify({ BackendState: 'Running', Self: { DNSName: 'a.b.ts.net.' } }))).toEqual({ backendState: 'Running', dnsName: 'a.b.ts.net' })
    expect(() => parseTailscaleStatus('nope')).toThrow(HubSetupError)
  })
})

describe('hub link', () => {
  const linkOptions = (home: string) => ({
    target: 'xerxes@hub', localSocket: '/Users/me/.xerxes/daemon/global-1.sock',
    executable: '/opt/bun', scriptPath: '/opt/cli.js', environment: { PATH: '/usr/bin' }, logPath: join(home, 'link.log'),
    machine: 'Mac', remoteXerxes: '~/.bun/bin/bun ~/x/cli.js',
  })

  const PEM = '-----BEGIN PUBLIC KEY-----\nMCowBQYDK2VwAyEAabc\n-----END PUBLIC KEY-----\n'
  const hubHandler = (argv: readonly string[]): Partial<CommandResult> | undefined => {
    const command = argv.at(-1) ?? ''
    if (command.startsWith('mkdir -p')) return { stdout: '/home/xerxes\n' }
    if (command.includes("'machines' 'add-local'")) return { stdout: JSON.stringify({ machine_id: 'machine-1', created: true, removed_workspace_ids: ['old-1', 'old-2'] }) + '\n' }
    if (command.includes("'info' '--json'")) return { stdout: JSON.stringify({ hub_id: 'hub-abc', fingerprint: 'f', public_key_pem: PEM, public_origin: null, version: '1' }) + '\n' }
    return undefined
  }

  test('prepares the Hub host, installs the tunnel service, registers ONE machine, replaces old entries and enrolls the Hub key locally (idempotent)', async () => {
    const home = await tempDir('xh-link-')
    const fake = fakeRunner(hubHandler)
    const lines: string[] = []
    const enrolled: unknown[] = []
    const deps = {
      platform: 'linux' as const, home, run: fake.run, sleep: noSleep, ssh: '/usr/bin/ssh', hostname: 'mac.local', socketExists: () => true, out: (line: string) => { lines.push(line) },
      enrollLocal: async (input: unknown) => { enrolled.push(input) },
    }
    const result = await runHubLink(linkOptions(home), deps)
    expect(result).toMatchObject({ machineId: 'machine-1', remoteSocket: '/home/xerxes/.xerxes/tunnels/mac-daemon.sock', enrolled: true })
    const adds = fake.calls.map(call => call.at(-1)!).filter(command => command.includes('add-local'))
    expect(adds).toEqual(["~/.bun/bin/bun ~/x/cli.js 'hub' 'machines' 'add-local' 'Mac' '--socket' '/home/xerxes/.xerxes/tunnels/mac-daemon.sock' '--json'"])
    expect(lines.join('\n')).toContain('replaced 2 older per-project entries')
    expect(enrolled).toEqual([{ socketPath: '/Users/me/.xerxes/daemon/global-1.sock', hubId: 'hub-abc', publicKeyPem: PEM }])
    expect(fake.calls.some(call => call.at(-1) === "test -S '/home/xerxes/.xerxes/tunnels/mac-daemon.sock'")).toBe(true)
    // Every remote call is non-interactive, never multiplexed, and ends option parsing before the target.
    for (const call of fake.calls.filter(row => row[0] === '/usr/bin/ssh')) {
      expect(call).toContain('BatchMode=yes')
      expect(call).toContain('ControlMaster=no')
      expect(call).toContain('ControlPath=none')
      expect(call[call.indexOf('--') + 1]).toBe('xerxes@hub')
    }
    // Linking again registers the same machine and enrolls again without error.
    await runHubLink(linkOptions(home), deps)
    expect(enrolled).toHaveLength(2)
    const unit = await readFile(join(home, '.config', 'systemd', 'user', 'xerxes-hub-link-xerxes-hub.service'), 'utf8')
    expect(unit).toContain('"link-tunnel" "xerxes@hub" "--remote-socket" "/home/xerxes/.xerxes/tunnels/mac-daemon.sock" "--local-socket" "/Users/me/.xerxes/daemon/global-1.sock"')
  })

  test('re-linking under another machine name keeps the existing tunnel socket, so the Hub entry is not stranded', async () => {
    const home = await tempDir('xh-relink-')
    const fake = fakeRunner(hubHandler)
    const deps = {
      platform: 'linux' as const, home, run: fake.run, sleep: noSleep, ssh: '/usr/bin/ssh', hostname: 'Erfans-MacBook-Pro.local', socketExists: () => true, out: () => {},
      enrollLocal: async () => {},
    }
    const first = await runHubLink(linkOptions(home), deps)
    expect(first.remoteSocket).toBe('/home/xerxes/.xerxes/tunnels/mac-daemon.sock')
    // The app links without --machine: the name now comes from the hostname.
    const { machine: _machine, ...withoutMachine } = linkOptions(home)
    const second = await runHubLink(withoutMachine, deps)
    expect(second.remoteSocket).toBe('/home/xerxes/.xerxes/tunnels/mac-daemon.sock')
    const adds = fake.calls.map(call => call.at(-1)!).filter(command => command.includes('add-local'))
    expect(adds.at(-1)).toContain("'--socket' '/home/xerxes/.xerxes/tunnels/mac-daemon.sock'")
    // A fresh machine with no service yet still gets a socket named after it.
    const other = await tempDir('xh-fresh-')
    expect((await runHubLink(withoutMachine, { ...deps, home: other })).remoteSocket).toBe('/home/xerxes/.xerxes/tunnels/erfans-macbook-pro-daemon.sock')
  })

  test('fails with the next step: no local daemon, bad target, no Xerxes on the Hub host, tunnel never appears', async () => {
    const home = await tempDir('xh-link-')
    const deps = (handler: (argv: readonly string[]) => Partial<CommandResult> | undefined, socketExists = true) =>
      ({ platform: 'linux' as const, home, run: fakeRunner(handler).run, sleep: noSleep, ssh: '/usr/bin/ssh', hostname: 'mac', socketExists: () => socketExists, out: () => {}, enrollLocal: async () => {} })
    const prepared = (argv: readonly string[]) => ((argv.at(-1) ?? '').startsWith('mkdir -p') ? { stdout: '/home/xerxes\n' } : undefined)
    await expect(runHubLink(linkOptions(home), deps(prepared, false))).rejects.toThrow(/No Xerxes daemon/)
    await expect(runHubLink({ ...linkOptions(home), target: '-oProxyCommand=x' }, deps(prepared))).rejects.toThrow(HubLinkError)
    await expect(runHubLink(linkOptions(home), deps(argv => prepared(argv) ?? ((argv.at(-1) ?? '').includes('add-local') ? { code: 127 } : undefined)))).rejects.toThrow(/not installed/)
    await expect(runHubLink({ ...linkOptions(home), socketWaitMs: 0 }, deps(argv => prepared(argv) ?? ((argv.at(-1) ?? '').startsWith('test -S') ? { code: 1 } : undefined)))).rejects.toThrow(/never appeared/)
    expect(() => validateSshTarget('user@host name')).toThrow(HubLinkError)
    expect(validateSshTarget('xerxes@209.38.102.184')).toBe('xerxes@209.38.102.184')
  })

  test('the tunnel runner clears a stale forwarded socket, then forwards this daemon', async () => {
    const fake = fakeRunner(() => undefined)
    let attached: readonly string[] = []
    const code = await runLinkTunnel({ ssh: '/usr/bin/ssh', target: 'xerxes@hub', remoteSocket: '/home/x/.xerxes/tunnels/m.sock', localSocket: '/Users/me/d.sock' },
      { run: fake.run, runAttached: async argv => { attached = argv; return 255 } })
    expect(code).toBe(255)
    expect(fake.calls[0]!.at(-1)).toBe("rm -f '/home/x/.xerxes/tunnels/m.sock'")
    expect(attached).toContain('ExitOnForwardFailure=yes')
    // A shared ControlMaster would take the forward and let this process exit at once.
    expect(attached).toContain('ControlMaster=no')
    expect(attached).toContain('ControlPath=none')
    expect(attached[attached.indexOf('-R') + 1]).toBe('/home/x/.xerxes/tunnels/m.sock:/Users/me/d.sock')
  })
})

describe('hub CLI helpers', () => {
  test('add-local updates the entry for the same project and socket instead of duplicating it', async () => {
    const home = await tempDir('xh-cli-')
    const paths = hubPaths({ XERXES_HOME: home })
    const quiet = { paths, stdout: () => {}, stderr: () => {} }
    expect(await runHubCli(['workspaces', 'add-local', '/srv/a', '--socket', '/tmp/t.sock'], quiet)).toBe(0)
    expect(await runHubCli(['workspaces', 'add-local', '/srv/a', '--socket', '/tmp/t.sock', '--label', 'Renamed'], quiet)).toBe(0)
    expect(await runHubCli(['workspaces', 'add-local', '/srv/a'], quiet)).toBe(0)
    const rows = (await new WorkspaceRegistry(paths.workspacesPath).load()).workspaces
    expect(rows).toHaveLength(2)
    expect(rows[0]).toMatchObject({ label: 'Renamed', endpoint: { kind: 'local', socketPath: '/tmp/t.sock' }, connection: { autoConnect: true } })
    expect(rows[1]?.endpoint).not.toHaveProperty('socketPath')
  })

  test('daemonProjects lists the absolute project directories of open chats with one summary call', async () => {
    const dir = await tempDir('xh-dp-')
    const socketPath = join(dir, 'd.sock')
    const seen: unknown[] = []
    const server = Bun.listen<undefined>({ unix: socketPath, socket: {
      data(socket, bytes) {
        const frame = JSON.parse(bytes.toString().trim()) as { id: number }
        seen.push(frame)
        socket.write(JSON.stringify({ jsonrpc: '2.0', id: frame.id, result: { ok: true, sessions: [{ cwd: '/b' }, { cwd: '/a' }, { cwd: '/b' }, { cwd: 'rel' }, {}] } }) + '\n')
      },
    } })
    registerCleanup(() => { server.stop(true) })
    expect(await daemonProjects(socketPath)).toEqual(['/a', '/b'])
    expect(seen).toMatchObject([{ method: 'session.active_list', params: { history_limit: 0 } }])
    await expect(daemonProjects(join(dir, 'missing.sock'), 500)).rejects.toBeDefined()
  })
})
