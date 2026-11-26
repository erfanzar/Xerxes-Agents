// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, readFile, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

import { Shell } from '../src/desktop/renderer/App.js'
import { Markdown, parseEmbedLine } from '../src/desktop/renderer/markdown.js'
import type { Snapshot } from '../src/desktop/renderer/store.js'
import {
  ArtifactWidgetEmbed,
  WidgetFrame,
  WidgetOpenArtifactContext,
  WidgetSessionContext,
  wrapWidgetHtml,
} from '../src/desktop/renderer/WidgetFrame.js'
import { XERXES_3D_RUNTIME_SCRIPT } from '../src/desktop/renderer/xerxes3dRuntime.js'
import { lintHarness } from '../src/extensions/harnessLint.js'
import { parseSkillMarkdown, SkillRegistry } from '../src/extensions/skills.js'
import { ToolRegistry } from '../src/executors/toolRegistry.js'
import { ArtifactStore } from '../src/session/artifacts.js'
import {
  CLAUDE_WORKFLOW_TOOL_DEFINITIONS,
  ClaudeWorkflowTools,
} from '../src/tools/claudeTools/workflow.js'
import { ValidationError } from '../src/core/errors.js'

const RENDERER_DIR = join(import.meta.dir, '..', 'src', 'desktop', 'renderer')
const SKILL_PATH = join(import.meta.dir, '..', 'skills', 'generative-ui', 'SKILL.md')

function baseSnapshot(overrides: Partial<Snapshot>): Snapshot {
  return {
    connection: 'online',
    cwd: '/repo',
    model: 'kimi',
    currentAgentPreset: 'default',
    agentPresets: [],
    models: [],
    reasoningLevels: [],
    contextTokens: null,
    contextMax: null,
    ttftMs: null,
    tokensPerSecond: null,
    llmDurationMs: 0,
    llmSteps: 0,
    toolDurationMs: 0,
    toolSteps: 0,
    inputTokens: 0,
    outputTokens: 0,
    metricPhase: null,
    metricPhaseStartedAt: null,
    cacheHitRate: null,
    sessions: [],
    live: [],
    fleet: [],
    skillSuggestions: [],
    creatorTrace: [],
    backgroundJobs: [],
    currentId: '',
    currentTitle: '',
    sessionKey: 'sess-1',
    branch: '',
    daemonWarning: null,
    costUsd: null,
    mcpStatus: {},
    goal: '',
    approval: null,
    question: null,
    interactionHidden: false,
    planMode: false,
    turnActive: false,
    turnFailed: false,
    turnSeconds: 0,
    blocks: [],
    error: null,
    tab: 'activity',
    turnCount: 0,
    queue: [],
    changes: [],
    plan: null,
    log: [],
    failed: null,
    settingsOpen: false,
    settingsTab: 'general',
    paletteOpen: false,
    commands: [],
    pickerOpen: false,
    wsMenuOpen: false,
    sessionMenu: null,
    streamThinking: true,
    taskModalOpen: false,
    providers: [],
    providerModels: {},
    providerModelLoading: [],
    providerModelWarnings: {},
    providerTypes: [],
    permissionMode: '',
    snippets: {},
    ...overrides,
  }
}

test('XERXES_3D_RUNTIME_SCRIPT executes cleanly and renders all 3D primitives, particles, labels, and orbit controls', () => {
  const calls: string[] = []
  const listeners = new Map<string, Array<(event: unknown) => void>>()
  const mockCtx = {
    fillStyle: '',
    strokeStyle: '',
    lineWidth: 1,
    lineJoin: 'round',
    lineCap: 'round',
    font: '',
    textAlign: 'center',
    textBaseline: 'middle',
    setTransform: () => calls.push('setTransform'),
    clearRect: () => calls.push('clearRect'),
    fillRect: () => calls.push('fillRect'),
    beginPath: () => calls.push('beginPath'),
    moveTo: () => calls.push('moveTo'),
    lineTo: () => calls.push('lineTo'),
    closePath: () => calls.push('closePath'),
    fill: () => calls.push('fill'),
    stroke: () => calls.push('stroke'),
    arc: () => calls.push('arc'),
    rect: () => calls.push('rect'),
    roundRect: () => calls.push('roundRect'),
    fillText: () => calls.push('fillText'),
    measureText: (s: string) => ({ width: s.length * 6 }),
    save: () => calls.push('save'),
    restore: () => calls.push('restore'),
    setLineDash: () => calls.push('setLineDash'),
    createRadialGradient: () => ({ addColorStop: () => {} }),
  }
  const mockCanvas = {
    width: 600,
    height: 360,
    clientWidth: 600,
    clientHeight: 360,
    style: { cursor: '', touchAction: '' },
    getBoundingClientRect: () => ({ width: 600, height: 360 }),
    getContext: (kind: string) => (kind === '2d' ? mockCtx : null),
    addEventListener: (type: string, fn: (event: unknown) => void) => {
      const list = listeners.get(type) ?? []
      list.push(fn)
      listeners.set(type, list)
    },
    removeEventListener: (type: string, fn: (event: unknown) => void) => {
      const list = listeners.get(type) ?? []
      listeners.set(type, list.filter(item => item !== fn))
    },
  }
  const fakeWindow: Record<string, unknown> = {
    devicePixelRatio: 2,
    addEventListener: () => {},
    removeEventListener: () => {},
  }

  // Evaluate the self-contained runtime IIFE against the isolated fake window.
  const runner = new Function('window', 'globalThis', XERXES_3D_RUNTIME_SCRIPT) as (w: unknown, g: unknown) => void
  runner(fakeWindow, fakeWindow)

  const engine = fakeWindow.Xerxes3D as {
    __version: string
    createScene: (canvas: unknown, opts?: Record<string, unknown>) => Record<string, (...args: unknown[]) => unknown>
  }
  expect(engine.__version).toBe('1.0')

  const scene = engine.createScene(mockCanvas, { background: 'cosmic', distance: 5, autoRotate: 0.3 })
  scene.grid({ size: 6, divisions: 6 })
  scene.axes({ size: 2 })
  scene.box({ size: 0.6, position: [-1.5, 0, 0] })
  scene.sphere({ radius: 0.45, position: [1.5, 0, 0] })
  scene.torus({ radius: 0.8, tube: 0.2, position: [0, 0, -1.2] })
  scene.torusKnot({ radius: 1.0, tube: 0.25, color: '#a855f7', colorB: '#fbbf24' })
  scene.icosahedron({ radius: 0.4, detail: 1, position: [0, 1.4, 0] })
  scene.cylinder({ radiusTop: 0.2, radiusBottom: 0.3, height: 0.8, position: [0, -1.2, 0] })
  scene.cone({ radius: 0.3, height: 0.7, position: [1, -1, 0] })
  scene.plane({ width: 1, height: 1, position: [0, -1.4, 0] })
  scene.mesh({ vertices: [[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces: [[0, 1, 2]] }, { doubleSided: true })
  scene.line([[0, 0, 0], [1, 1, 1]], { dashed: true })
  scene.particles({ count: 24, radius: 2.2, glow: true })
  scene.label([0, 1.6, 0], 'Simurgh Knot')

  let tickCount = 0
  scene.animate(() => { tickCount += 1 })
  expect(tickCount).toBe(1)
  expect(calls.filter(c => c === 'fill').length).toBeGreaterThan(100)
  expect(calls.filter(c => c === 'arc').length).toBeGreaterThan(20)
  expect(calls).toContain('fillText')

  // Orbit drag + zoom + double-click reset update camera cleanly.
  const beforeCam = scene.getCamera() as { azimuth: number; distance: number }
  listeners.get('pointerdown')?.[0]?.({ clientX: 100, clientY: 100 })
  listeners.get('pointermove')?.[0]?.({ clientX: 160, clientY: 120 })
  listeners.get('pointerup')?.[0]?.({})
  listeners.get('wheel')?.[0]?.({ deltaY: 80, preventDefault: () => {} })
  const movedCam = scene.getCamera() as { azimuth: number; distance: number }
  expect(movedCam.azimuth).not.toBe(beforeCam.azimuth)
  expect(movedCam.distance).toBeGreaterThan(beforeCam.distance)

  listeners.get('dblclick')?.[0]?.({})
  const resetCam = scene.getCamera() as { azimuth: number; distance: number }
  expect(resetCam.azimuth).toBeCloseTo(beforeCam.azimuth, 5)
  expect(resetCam.distance).toBeCloseTo(beforeCam.distance, 5)

  scene.dispose()
  expect(listeners.get('pointerdown')).toEqual([])
})

function decodeDataScripts(html: string): string {
  const decoded: string[] = []
  for (const match of html.matchAll(/src="data:text\/javascript;base64,([^"]+)"/g)) {
    decoded.push(Buffer.from(match[1] ?? '', 'base64').toString('utf8'))
  }
  return decoded.join('\n')
}

test('wrapWidgetHtml injects offline CSP, theme tokens, Xerxes3D runtime, and inline auto-height bridge', () => {
  const bare = wrapWidgetHtml('<div class="x-card">Hello</div><script>window.__demo = 1;</script>', {
    mode: 'inline',
    theme: 'dark',
    embedId: 'emb-42',
  })
  expect(bare).toContain('<!doctype html>')
  expect(bare).toContain(
    "default-src 'none'; script-src 'unsafe-inline' data:; style-src 'unsafe-inline'; img-src data: blob:; font-src data:;",
  )
  expect(bare).toContain('--background:')
  expect(bare).toContain('--primary:')
  expect(bare).toContain('--chart-6:')
  expect(bare).toContain('html.light')
  expect(bare).toContain('src="data:text/javascript;base64,')
  const bareScripts = decodeDataScripts(bare)
  expect(bareScripts).toContain('global.Xerxes3D')
  expect(bareScripts).toContain('xerxes-widget-theme')
  expect(bareScripts).toContain("window.dispatchEvent(new Event('themechange'))")
  expect(bareScripts).toContain("type: 'xerxes-widget-height', embedId: 'emb-42'")
  expect(bareScripts).toContain('window.__demo = 1;')
  expect(bare).toContain('<div class="x-card">Hello</div>')

  // Full document in light panel mode preserves existing HTML, uses @layer xerxes-defaults
  // without clobbering standard root font-size or height: 100%, and omits inline height bridge.
  const full = wrapWidgetHtml(
    '<!doctype html><html lang="en"><head><title>Demo</title></head><body><canvas id="c"></canvas></body></html>',
    { mode: 'panel', theme: 'light', embedId: 'panel-1' },
  )
  expect(full).toContain('class="light"')
  expect(full).toContain('data-theme="light"')
  expect(full).toContain('<title>Demo</title>')
  expect(full).toContain('@layer xerxes-defaults')
  expect(full).not.toContain('font-size: 13.5px')
  expect(full).not.toMatch(/[^-]height:\s*100%/)
  const fullScripts = decodeDataScripts(full)
  expect(fullScripts).toContain('global.Xerxes3D')
  expect(fullScripts).not.toContain('xerxes-widget-height')

  // Inline HTML event attributes (onclick/oninput/onchange/javascript: href) are rewritten out of
  // tags into externalized data: scripts so Chromium CSP3 inheritance (`script-src 'self' data:`)
  // never blocks clicks or sliders, while bridge script installs memory storage + dynamic upgrader.
  const interactive = wrapWidgetHtml(
    `<button id="b" onclick="selectTab(&quot;metrics&quot;)">Metrics</button>`
    + `<input id="r" type="range" oninput="setVal(this.value)" />`
    + `<a id="a" href="javascript:jump(2)">Jump</a>`
    + `<template><button onclick="fromTemplate()">Tpl</button></template>`
    + `<script>const tpl = '<button onclick="keepInScript()">Keep</button>';</script>`,
    { mode: 'panel', theme: 'dark', embedId: 'int-1' },
  )
  expect(interactive).not.toMatch(/<button id="b"[^>]*\bonclick=/i)
  expect(interactive).not.toMatch(/<input[^>]*\boninput=/i)
  expect(interactive).not.toMatch(/<a[^>]*href="javascript:/i)
  expect(interactive).toContain('<template><button onclick="fromTemplate()">Tpl</button></template>')
  expect(interactive).toContain('data-xev="e0"')
  expect(interactive).toContain('data-xev="e1"')
  expect(interactive).toContain('data-xev="e2"')
  const intScripts = decodeDataScripts(interactive)
  expect(intScripts).toContain('selectTab("metrics")')
  expect(intScripts).toContain('setVal(this.value)')
  expect(intScripts).toContain('jump(2)')
  expect(intScripts).toContain('keepInScript()')
  expect(intScripts).toContain("installMemoryStorage('localStorage')")
  expect(intScripts).toContain('bindDynamicEl')
  expect(intScripts).toContain('flushDynamicBindings')
})

test('WidgetFrame uses strict sandbox="allow-scripts", detects 3D badge, avoids literal hex colors, and revokes blob URLs', async () => {
  const html3d = renderToStaticMarkup(
    createElement(
      WidgetOpenArtifactContext.Provider,
      { value: () => {} },
      createElement(WidgetFrame, {
        html: '<canvas id="c"></canvas><script>Xerxes3D.createScene("#c").torusKnot()</script>',
        title: 'Cosmic Torus Knot',
        mode: 'inline',
        artifactId: 'art3d00001',
      }),
    ),
  )
  expect(html3d).toContain('sandbox="allow-scripts"')
  expect(html3d).not.toContain('allow-same-origin')
  expect(html3d).toContain('data-3d="true"')
  expect(html3d).toContain('>3D<')
  expect(html3d).toContain('Cosmic Torus Knot')
  expect(html3d).toContain('Open in panel')
  expect(html3d).toContain('Source')
  expect(html3d).toContain('Reload')
  expect(html3d.match(/#[0-9a-fA-F]{6}\b/g) ?? []).toEqual([])

  const htmlPlain = renderToStaticMarkup(
    createElement(WidgetFrame, {
      html: '<div class="x-card">Interactive calculator</div>',
      title: 'Calculator',
      mode: 'panel',
    }),
  )
  expect(htmlPlain).toContain('sandbox="allow-scripts"')
  expect(htmlPlain).toContain('widget-frame--panel')
  expect(htmlPlain).toContain('widget-frame__panel-expand')
  expect(htmlPlain).not.toContain('widget-frame__bar')
  expect(htmlPlain).not.toContain('data-3d="true"')
  expect(htmlPlain).not.toContain('Open in panel')
  expect(htmlPlain.match(/#[0-9a-fA-F]{6}\b/g) ?? []).toEqual([])

  const source = await readFile(join(RENDERER_DIR, 'WidgetFrame.tsx'), 'utf8')
  expect(source).toContain('URL.createObjectURL')
  expect(source).toContain('URL.revokeObjectURL')
  expect(source).not.toContain('allow-same-origin')
  expect(source).toContain("'artifacts.read'")
})

test('ArtifactWidgetEmbed renders loading state cleanly with context and zero hex literals', () => {
  const html = renderToStaticMarkup(
    createElement(
      WidgetSessionContext.Provider,
      { value: 'sess-9' },
      createElement(
        WidgetOpenArtifactContext.Provider,
        { value: () => {} },
        createElement(ArtifactWidgetEmbed, { artifactId: 'abc1234567', title: 'Orbital Explorer' }),
      ),
    ),
  )
  expect(html).toContain('data-artifact-id="abc1234567"')
  expect(html).toContain('Orbital Explorer')
  expect(html).toContain('Open in panel')
  expect(html.match(/#[0-9a-fA-F]{6}\b/g) ?? []).toEqual([])
})

test('markdown renders <xerxes-embed>, <agent-embed>, ```widget, ```html-widget, and ```3d inline while keeping ```html as code', () => {
  expect(parseEmbedLine('<xerxes-embed artifact="a1b2c3d4"></xerxes-embed>')).toEqual({ artifactId: 'a1b2c3d4' })
  expect(parseEmbedLine('<xerxes-embed artifact="a1b2c3d4" title="My Sim"></xerxes-embed>')).toEqual({
    artifactId: 'a1b2c3d4',
    title: 'My Sim',
  })
  expect(parseEmbedLine('<agent-embed src="artifact://z9y8x7w6"></agent-embed>')).toEqual({ artifactId: 'z9y8x7w6' })
  expect(
    parseEmbedLine('<agent-embed src="file:///Users/demo/.gemini/antigravity-cli/brain/83077c62-1572-4755-a9d0-b21533b609c1/orbital%20scene.html" title="Orbital"></agent-embed>'),
  ).toEqual({
    artifactId: 'agy:orbital scene.html',
    title: 'Orbital',
  })
  expect(parseEmbedLine('<agent-embed src="artifact://agy:widget.html"></agent-embed>')).toEqual({
    artifactId: 'agy:widget.html',
  })
  expect(parseEmbedLine('<agent-embed src="file:///etc/passwd"></agent-embed>')).toBeNull()
  expect(parseEmbedLine('<agent-embed src="file:///Users/demo/.gemini/antigravity-cli/brain/sess-1/..%2Fsecret.html"></agent-embed>')).toBeNull()
  expect(parseEmbedLine('<agent-embed src="file:///Users/demo/.gemini/antigravity-cli/brain/sess-1/notes.md"></agent-embed>')).toBeNull()
  expect(parseEmbedLine('<agent-embed src="https://evil.example/x.html"></agent-embed>')).toBeNull()

  const md = [
    'Here is an interactive artifact:',
    '',
    '<xerxes-embed artifact="art12345" title="Simurgh Scene"></xerxes-embed>',
    '',
    '<agent-embed src="artifact://art67890"></agent-embed>',
    '',
    '```widget Interactive Slider',
    '<div class="x-row"><input type="range" /></div>',
    '```',
    '',
    '```html-widget',
    '<div class="x-card">Widget HTML</div>',
    '```',
    '',
    '```3d Quick Knot',
    'const s = Xerxes3D.createScene(canvas); s.torusKnot(); s.animate();',
    '```',
    '',
    'And a plain HTML code sample:',
    '',
    '```html',
    '<div class="example">Plain code only</div>',
    '```',
  ].join('\n')

  const rendered = renderToStaticMarkup(createElement(Markdown, { text: md }))
  expect(rendered).toContain('data-artifact-id="art12345"')
  expect(rendered).toContain('Simurgh Scene')
  expect(rendered).toContain('data-artifact-id="art67890"')
  expect(rendered).toContain('Interactive Slider')
  expect(rendered).toContain('Quick Knot')
  expect(rendered).toContain('data-3d="true"')
  expect(rendered).toContain('sandbox="allow-scripts"')
  // Plain ```html stays a highlighted code block and does NOT spawn an extra iframe.
  expect(rendered).toContain('data-lang="html"')
  expect((rendered.match(/<iframe /g) ?? []).length).toBe(3)

  // Streaming unclosed widget fence does not prematurely mount an iframe.
  const unclosed = renderToStaticMarkup(createElement(Markdown, { text: '```widget\n<div>partial' }))
  expect(unclosed).not.toContain('<iframe ')
  expect(unclosed).toContain('data-lang="widget"')

  // Standalone <agent-embed> pointing at an Antigravity brain image renders inline via ChatImage.
  const imgEmbed = renderToStaticMarkup(
    createElement(Markdown, {
      text: '<agent-embed src="file:///Users/demo/.gemini/antigravity-cli/brain/sess-1/render.png" title="3D Render"></agent-embed>',
    }),
  )
  expect(imgEmbed).toContain('Loading image: 3D Render')
  expect(imgEmbed).toContain('/Users/demo/.gemini/antigravity-cli/brain/sess-1/render.png')
})

test('ArtifactTool accepts optional embed boolean and returns inline embed hint for html artifacts', async () => {
  const artifactDef = CLAUDE_WORKFLOW_TOOL_DEFINITIONS.find(tool => tool.function.name === 'ArtifactTool')
  expect(artifactDef).toBeDefined()
  const props = (artifactDef!.function.parameters as { properties: Record<string, { type?: string }> }).properties
  expect(props.embed?.type).toBe('boolean')

  const dir = await mkdtemp(join(tmpdir(), 'xerxes-genui-'))
  try {
    const registry = new ToolRegistry()
    const workflow = new ClaudeWorkflowTools({ artifactStore: new ArtifactStore({ root: dir }) }, registry)
    const created = await workflow.execute(
      'ArtifactTool',
      {
        action: 'create',
        title: '3D Knot',
        kind: 'html',
        embed: true,
        content: '<canvas id="c"></canvas><script>Xerxes3D.createScene("#c").torusKnot()</script>',
      },
      { agentId: 'default', sessionId: 'sess-genui' },
    )
    expect(typeof created).toBe('string')
    const textResult = created as string
    expect(textResult).toMatch(/^Created artifact "3D Knot" \(id [a-z0-9]+, version 1\)\./)
    expect(textResult).toContain('<xerxes-embed artifact="')

    await expect(
      workflow.execute(
        'ArtifactTool',
        { action: 'create', title: 'Bad embed', kind: 'html', embed: 'yes', content: '<p>x</p>' },
        { agentId: 'default', sessionId: 'sess-genui' },
      ),
    ).rejects.toThrow(ValidationError)
  } finally {
    await rm(dir, { recursive: true, force: true })
  }
})

test('bundled generative-ui skill loads cleanly with zero lint findings', async () => {
  const raw = await readFile(SKILL_PATH, 'utf8')
  expect(lintHarness([{ path: SKILL_PATH, content: raw }])).toEqual([])

  const parsed = parseSkillMarkdown(raw, SKILL_PATH)
  expect(parsed.metadata.name).toBe('generative-ui')
  expect(parsed.metadata.tags).toContain('3d')
  expect(parsed.instructions).toContain('License: Apache-2.0')
  expect(parsed.instructions).toContain('window.Xerxes3D')
  expect(parsed.instructions).toContain('<xerxes-embed artifact=')
  expect(parsed.instructions).toContain('ArtifactTool')

  const registry = new SkillRegistry()
  await registry.discover(join(import.meta.dir, '..', 'skills'))
  expect(registry.get('generative-ui')).toBeDefined()
})

test('WorkspaceGate and Offline render Xerxes brand emblem and AgentOrb with zero hex literals', () => {
  const gateHtml = renderToStaticMarkup(createElement(Shell, { snap: baseSnapshot({ noWorkspace: true }) }))
  expect(gateHtml).toContain('wsgate__mark')
  expect(gateHtml).toContain('./logo-128.png')
  expect(gateHtml.match(/#[0-9a-fA-F]{6}\b/g) ?? []).toEqual([])

  const offlineHtml = renderToStaticMarkup(
    createElement(Shell, { snap: baseSnapshot({ connection: 'offline', blocks: [] }) }),
  )
  expect(offlineHtml).toContain('offline__mark')
  expect(offlineHtml).toContain('class="agent-orb"')
  expect(offlineHtml.match(/#[0-9a-fA-F]{6}\b/g) ?? []).toEqual([])
})
