// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { ComputerUseUnavailableError } from '../src/tools/computerUse/backend.js'
import {
  MacOSComputerUsePort,
  createMacOSComputerUseToolOptions,
  type MacOSCommandResult,
  type MacOSComputerUsePortOptions,
  type MacOSCommandRunner,
} from '../src/tools/computerUse/macosPort.js'
import {
  JXA_CURSOR_POSITION,
  JXA_FOCUS_APP,
  JXA_LIST_APPS,
  JXA_SCREEN_INFO,
} from '../src/tools/computerUse/macosScripts.js'

test('output-producing JXA scripts use explicit return (bare expressions print nothing)', () => {
  for (const script of [JXA_CURSOR_POSITION, JXA_SCREEN_INFO, JXA_LIST_APPS, JXA_FOCUS_APP]) {
    expect(script).toContain('return')
  }
  // NSNumber boxing breaks strict equality in JXA; the app filter must coerce.
  expect(JXA_LIST_APPS).toContain('Number(app.activationPolicy)')
})

const PNG_BYTES = new Uint8Array([0x89, 0x50, 0x4e, 0x47])
const PNG_B64 = 'iVBORw=='

interface FakeRunner {
  readonly calls: string[][]
  readonly runner: MacOSCommandRunner
}

function fakeRunner(overrides: { readonly failCode?: number; readonly stdout?: string; readonly stderr?: string } = {}): FakeRunner {
  const calls: string[][] = []
  const runner: MacOSCommandRunner = async argv => {
    calls.push([...argv])
    const executable = argv[0] ?? ''
    const script = argv[4] ?? ''
    if (overrides.failCode !== undefined) {
      return { code: overrides.failCode, stderr: overrides.stderr ?? 'boom', stdout: '' }
    }
    if (executable.endsWith('osascript') && script.includes('NSScreen')) {
      return { code: 0, stderr: '', stdout: '1800,1169,2\n' }
    }
    if (executable.endsWith('sips') && argv.includes('-g')) {
      return { code: 0, stderr: '', stdout: '/tmp/xerxes-cua-test.png\n  pixelWidth: 3600\n  pixelHeight: 2338\n' }
    }
    if (script.includes('activateWithOptions')) {
      return { code: 0, stderr: '', stdout: overrides.stdout ?? 'ok' }
    }
    if (script.includes('runningApplications')) {
      return { code: 0, stderr: '', stdout: overrides.stdout ?? 'Finder, Safari, Terminal' }
    }
    if (script.includes('CGEventGetLocation')) {
      return { code: 0, stderr: '', stdout: overrides.stdout ?? '1045.5,303.25\n' }
    }
    return { code: 0, stderr: '', stdout: overrides.stdout ?? '' }
  }
  return { calls, runner }
}

function makePort(fake: FakeRunner, extra: MacOSComputerUsePortOptions = {}): MacOSComputerUsePort {
  return new MacOSComputerUsePort({
    fileExists: () => true,
    platform: 'darwin',
    readFile: async () => PNG_BYTES,
    removeFile: async () => undefined,
    runner: fake.runner,
    tmpDir: '/tmp',
    uniqueId: () => 'test',
    ...extra,
  })
}

test('macOS port reports availability only on darwin with all system tools present', () => {
  const fake = fakeRunner()
  expect(makePort(fake).isAvailable()).toBe(true)
  expect(makePort(fake, { platform: 'linux' }).isAvailable()).toBe(false)
  expect(makePort(fake, { fileExists: path => !path.includes('sips') }).isAvailable()).toBe(false)
})

test('capture downscales to logical points capped at the max edge and cleans up', async () => {
  const fake = fakeRunner()
  const removed: string[] = []
  const port = makePort(fake, { removeFile: async path => { removed.push(path) } })

  const capture = await port.capture({ mode: 'vision' })
  expect(capture.mode).toBe('vision')
  expect(capture.width).toBe(1568)
  expect(capture.height).toBe(1018)
  expect(capture.pngB64).toBe(PNG_B64)
  expect(capture.pngBytesLength).toBe(PNG_BYTES.length)
  expect(capture.elements).toEqual([])

  const [shot, dims, , resize] = fake.calls
  expect(shot?.[0]).toBe('/usr/sbin/screencapture')
  expect(shot).toContain('/tmp/xerxes-cua-test.png')
  expect(dims?.[0]).toBe('/usr/bin/sips')
  expect(resize).toEqual(['/usr/bin/sips', '-z', '1018', '1568', '/tmp/xerxes-cua-test.png'])
  expect(removed).toEqual(['/tmp/xerxes-cua-test.png'])
})

test('capture mode ax performs no screenshot work', async () => {
  const fake = fakeRunner()
  const capture = await makePort(fake).capture({ mode: 'ax' })
  expect(capture).toEqual({ elements: [], height: 0, mode: 'ax', width: 0 })
  expect(fake.calls).toEqual([])
})

test('clicks map captured-image pixels back to logical points with button and count', async () => {
  const fake = fakeRunner()
  const port = makePort(fake)
  await port.capture({ mode: 'vision' })

  const result = await port.click({ button: 'left', captureAfter: false, clickCount: 1, x: 784, y: 509 })
  expect(result.ok).toBe(true)
  const click = fake.calls.at(-1)
  expect(click?.slice(0, 4)).toEqual(['/usr/bin/osascript', '-l', 'JavaScript', '-e'])
  expect(click?.slice(5)).toEqual(['900', '584', 'left', '1'])

  await port.doubleClick({ captureAfter: false, x: 10, y: 20 })
  expect(fake.calls.at(-1)?.[8]).toBe('2')
  await port.tripleClick({ captureAfter: false, x: 10, y: 20 })
  expect(fake.calls.at(-1)?.[8]).toBe('3')
  await port.rightClick({ captureAfter: false, x: 10, y: 20 })
  expect(fake.calls.at(-1)?.[7]).toBe('right')
  await port.middleClick({ captureAfter: false, x: 10, y: 20 })
  expect(fake.calls.at(-1)?.[7]).toBe('middle')
})

test('element-only targets fail with an actionable message instead of pretending', async () => {
  const fake = fakeRunner()
  const port = makePort(fake)
  const result = await port.click({ button: 'left', captureAfter: false, clickCount: 1, element: 3 })
  expect(result.ok).toBe(false)
  expect(result.message).toContain('x/y coordinates')
  expect(fake.calls).toEqual([])

  const drag = await port.drag({ captureAfter: false, startElement: 1, endX: 5, endY: 6 })
  expect(drag.ok).toBe(false)
  const value = await port.setValue({ captureAfter: false, element: 2, value: 'x' })
  expect(value.ok).toBe(false)
  expect(value.message).toContain('accessibility')
})

test('mouse_move and drag emit CoreGraphics events with computed coordinates', async () => {
  const fake = fakeRunner()
  const port = makePort(fake)
  await port.capture({ mode: 'vision' })

  await port.mouseMove({ captureAfter: false, x: 784, y: 509 })
  expect(fake.calls.at(-1)?.slice(5)).toEqual(['900', '584'])

  const drag = await port.drag({ captureAfter: false, startX: 0, startY: 0, endX: 1568, endY: 1018 })
  expect(drag.ok).toBe(true)
  expect(fake.calls.at(-1)?.slice(5)).toEqual(['0', '0', '1800', '1169', '24'])
})

test('scroll converts pixel deltas to signed wheel lines and moves first when x/y given', async () => {
  const fake = fakeRunner()
  const port = makePort(fake)

  await port.scroll({ captureAfter: false, dx: -40, dy: 80 })
  expect(fake.calls.at(-1)?.slice(5)).toEqual(['-2', '1'])

  await port.scroll({ captureAfter: false, dx: 0, dy: -400, x: 5, y: 5 })
  const scroll = fake.calls.at(-1)
  const move = fake.calls.at(-2)
  expect(move?.[4]).toContain('MouseMoved')
  expect(scroll?.slice(5)).toEqual(['10', '0'])
})

test('type passes text through argv only, never interpolated into script source', async () => {
  const fake = fakeRunner()
  const port = makePort(fake)
  const payload = 'hello "; rm -rf /; echo "'
  const result = await port.type({ captureAfter: false, text: payload })
  expect(result.ok).toBe(true)
  const call = fake.calls.at(-1)
  expect(call?.at(-1)).toBe(payload)
  expect(call?.[3]).not.toContain(payload)
})

test('key chords resolve modifiers and key codes, unknown keys fail clearly', async () => {
  const fake = fakeRunner()
  const port = makePort(fake)

  const chord = await port.key({ captureAfter: false, key: 'command+shift+t' })
  expect(chord.ok).toBe(true)
  expect(fake.calls.at(-1)?.slice(5)).toEqual(['17', '1', '1', '0', '0'])

  await port.key({ captureAfter: false, key: 'Enter' })
  expect(fake.calls.at(-1)?.slice(5)).toEqual(['36', '0', '0', '0', '0'])

  const bad = await port.key({ captureAfter: false, key: 'command+wat' })
  expect(bad.ok).toBe(false)
  expect(bad.message).toContain('unknown key')
})

test('wait, list_apps, focus_app, and cursor_position report structured results', async () => {
  const fake = fakeRunner()
  const port = makePort(fake, { sleep: async () => undefined })

  expect((await port.wait(5)).message).toBe('waited 5ms')

  const apps = await port.listApps()
  expect(apps.ok).toBe(true)

  const focused = await port.focusApp('Safari')
  expect(focused.ok).toBe(true)

  const position = await port.cursorPosition()
  expect(position.ok).toBe(true)
})

test('list_apps parses the CSV output and focus_app reports missing apps', async () => {
  const appsFake = fakeRunner({ stdout: 'Finder, Safari, Terminal\n' })
  const apps = await makePort(appsFake).listApps()
  expect(apps.meta).toEqual({ apps: ['Finder', 'Safari', 'Terminal'] })

  const missingFake = fakeRunner({ stdout: 'not found' })
  const missing = await makePort(missingFake).focusApp('Nope')
  expect(missing.ok).toBe(false)
  expect(missing.message).toContain('Nope')

  const cursorFake = fakeRunner({ stdout: '1045.5,303.25\n' })
  const cursor = await makePort(cursorFake).cursorPosition()
  expect(cursor.meta).toEqual({ x: 1046, y: 303 })
})

test('backend failures surface permission hints, and capture failure raises unavailable', async () => {
  const denied = fakeRunner({ failCode: 1, stderr: 'System Events got an error: not allowed assistive access' })
  const port = makePort(denied)

  const click = await port.click({ button: 'left', captureAfter: false, clickCount: 1, x: 1, y: 1 })
  expect(click.ok).toBe(false)
  expect(click.message).toContain('Privacy & Security')

  await expect(port.capture({ mode: 'vision' })).rejects.toBeInstanceOf(ComputerUseUnavailableError)
})

test('capture_after attaches a fresh screenshot to a successful action', async () => {
  const fake = fakeRunner()
  const port = makePort(fake)
  const result = await port.click({ button: 'left', captureAfter: true, clickCount: 1, x: 1, y: 1 })
  expect(result.ok).toBe(true)
  expect(result.capture?.pngB64).toBe(PNG_B64)
  expect(fake.calls.some(call => call[0] === '/usr/sbin/screencapture')).toBe(true)
})

test('tool options default on when the backend can run, with explicit opt-out and force-enable paths', () => {
  // Default-on: no flags on a capable host registers the tool.
  expect(createMacOSComputerUseToolOptions({}, {}, () => true)).toBeDefined()
  // A host without the zero-install backend stays silent unless forced.
  expect(createMacOSComputerUseToolOptions({}, {}, () => false)).toBeUndefined()
  // Force-enable on an unsupported host registers the tool so calls error explicitly.
  expect(createMacOSComputerUseToolOptions({ computer_use_enabled: true }, {}, () => false)).toBeDefined()
  expect(createMacOSComputerUseToolOptions({}, { XERXES_COMPUTER_USE: '1' }, () => false)).toBeDefined()
  // Explicit opt-out always wins.
  expect(createMacOSComputerUseToolOptions({ computer_use_enabled: false }, {}, () => true)).toBeUndefined()
  expect(createMacOSComputerUseToolOptions({}, { XERXES_COMPUTER_USE: 'off' }, () => true)).toBeUndefined()
  expect(createMacOSComputerUseToolOptions({}, { XERXES_COMPUTER_USE: 'false' }, () => true)).toBeUndefined()
  // Non-macos backends are host-injected, never auto-built here.
  expect(createMacOSComputerUseToolOptions({ computer_use_backend: 'cua-mcp' }, {}, () => true)).toBeUndefined()
})
