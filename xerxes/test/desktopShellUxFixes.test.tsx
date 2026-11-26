// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, describe, expect, test } from 'bun:test'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

import { ActionDialog } from '../src/desktop/renderer/ActionDialog.js'
import {
  CONVERSATION_COLUMN_WIDTH,
  FailedCard,
  FilesChangedCard,
  QuestionCard,
  Shell,
  THREAD_DETAILS_MIN_GUTTER,
  TaskModal,
  isDetailsInlineEligible,
  pruneRailTabsForContext,
} from '../src/desktop/renderer/App.js'
import { isAntigravityAcpSession } from '../src/desktop/renderer/GitPanel.js'
import { AppUpdatePrompt } from '../src/desktop/renderer/AppUpdatePrompt.js'
import { RASTER_BRAND_MIN_SIZE, RuntimeBrandIcon, XerxesLogoMark } from '../src/desktop/renderer/BrandMark.js'
import { Markdown, clearMarkdownCache, hasOpenStreamingBlock, splitTableRow } from '../src/desktop/renderer/markdown.js'
import { Store, appUpdateView, type Snapshot } from '../src/desktop/renderer/store.js'
import { T3ModelPicker } from '../src/desktop/renderer/T3ModelPicker.js'
import { T3Sidebar, clampSidebarPopoverPosition } from '../src/desktop/renderer/T3Sidebar.js'
import { ThreadDetailsCard } from '../src/desktop/renderer/ThreadDetailsCard.js'
import { parseThreadInboxState } from '../src/desktop/renderer/threadInbox.js'
import {
  REPIN_BOTTOM_THRESHOLD_PX,
  captureHistoryScrollAnchor,
  nextTranscriptFollowState,
  restoreHistoryScrollAnchor,
} from '../src/desktop/renderer/transcriptScroll.js'

const RENDERER_DIR = join(import.meta.dir, '../src/desktop/renderer')
const holder = globalThis as { document?: unknown; window?: unknown }
const origDocument = holder.document
const origWindow = holder.window

function stampStyle(style: string | null): void {
  holder.document = {
    documentElement: {
      dataset: {},
      getAttribute: (name: string) => (name === 'data-style' ? style : null),
    },
  }
}

function makeSnapshot(overrides: Partial<Snapshot> = {}): Snapshot {
  return {
    ...new Store().getSnapshot(),
    connection: 'online',
    noWorkspace: false,
    cwd: '/workspace/Xerxes-Agents',
    branch: 'main',
    model: 'claude-sonnet-4-6',
    models: [
      { id: 'claude-sonnet-4-6', label: 'Claude Sonnet 4.6', provider: 'Anthropic' },
      { id: 'claude-opus-4-6', label: 'Claude Opus 4.6', provider: 'Anthropic' },
    ],
    currentId: 'sess-1',
    currentTitle: 'Fix desktop shell UX',
    sessionKey: 'key-1',
    permissionMode: 'auto',
    ...overrides,
  }
}

afterEach(() => {
  holder.document = origDocument
  holder.window = origWindow
  clearMarkdownCache()
})

describe('Fix 1: GlobalKeys, Store.toggleSideQuestion & TaskModal', () => {
  test('Store.toggleSideQuestion opens and closes side question cleanly', () => {
    const s = new Store()
    expect(s.getSnapshot().sideQuestionOpen).toBeFalsy()
    s.toggleSideQuestion()
    expect(s.getSnapshot().sideQuestionOpen).toBe(true)
    const rev1 = s.getSnapshot().sideQuestionOpenRevision ?? 0
    expect(rev1).toBeGreaterThan(0)
    s.toggleSideQuestion()
    expect(s.getSnapshot().sideQuestionOpen).toBe(false)
    s.toggleSideQuestion()
    expect(s.getSnapshot().sideQuestionOpen).toBe(true)
    expect(s.getSnapshot().sideQuestionOpenRevision).toBe(rev1 + 1)
  })

  test('TaskModal renders btn--solid CTA, icon instead of raw Unicode square, accessible labels, and backdrop/Escape handlers', () => {
    const html = renderToStaticMarkup(createElement(TaskModal, { snap: makeSnapshot({ taskModalOpen: true }) }))
    expect(html).toContain('class="btn btn--solid"')
    expect(html).toContain('Start task')
    expect(html).not.toContain('▣')
    expect(html).toContain('for="task-objective"')
    expect(html).toContain('id="task-objective"')
    expect(html).toContain('for="task-model"')
    expect(html).toContain('id="task-model"')

    const appSrc = readFileSync(join(RENDERER_DIR, 'App.tsx'), 'utf8')
    const taskModalSlice = appSrc.slice(appSrc.indexOf('function TaskModal('), appSrc.indexOf('function copyText('))
    expect(taskModalSlice).toContain('handleDialogEscape(event, () => store.closeTaskModal()')
    expect(taskModalSlice).toContain('event.target === event.currentTarget && !starting')
    const globalKeysSlice = appSrc.slice(appSrc.indexOf('function GlobalKeys('))
    expect(globalKeysSlice).toContain('store.toggleSideQuestion()')
    expect(globalKeysSlice).toContain('[role="dialog"][aria-modal="true"]')
    expect(globalKeysSlice).toContain('if (store.getSnapshot().settingsOpen) store.closeSettings()')
  })
})

describe('Fix 2: Sidebar Popovers Portaled & Viewport-Clamped', () => {
  test('clampSidebarPopoverPosition prevents -31px left clip at default 270px sidebar width and flips snooze menus upward at viewport bottom', () => {
    // Scope popover (230px wide) anchored to Filter icon at x=171..199 in 270px sidebar.
    const scopePos = clampSidebarPopoverPosition({
      anchor: { left: 171, right: 199, top: 56, bottom: 84 },
      popup: { width: 230, height: 240 },
      viewport: { width: 1440, height: 900 },
      align: 'right',
    })
    expect(scopePos.left).toBe(8)
    expect(scopePos.top).toBe(90)

    // Snooze menu (186px wide, 170px tall) opened on a thread row near the bottom of an 800px window flips upward.
    const snoozePos = clampSidebarPopoverPosition({
      anchor: { left: 210, right: 232, top: 740, bottom: 762 },
      popup: { width: 186, height: 170 },
      viewport: { width: 1440, height: 800 },
      align: 'right',
    })
    expect(snoozePos.left).toBe(46)
    expect(snoozePos.top).toBe(740 - 170 - 6)
  })

  test('T3Sidebar portals .t3-scope-pop, .t3-add-pop, and .t3-snooze-menu through PickerLayer', () => {
    const src = readFileSync(join(RENDERER_DIR, 'T3Sidebar.tsx'), 'utf8')
    expect(src).toContain("import { PickerLayer } from './Overlays.js'")
    expect(src).toMatch(/<PickerLayer>\s*<div ref=\{scopePopRef\} className="t3-scope-pop"/)
    expect(src).toMatch(/<PickerLayer>\s*<div ref=\{addProjectPopRef\} className="t3-add-pop"/)
    expect(src).toMatch(/<PickerLayer>\s*<div ref=\{snoozeRef\} className="t3-snooze-menu"/)
  })
})

describe('Fix 3: T3ModelPicker snap.pickerOpen sync, keyboard navigation & safe rail browsing', () => {
  test('opens when snap.pickerOpen is true and renders combobox ARIA + keyboard focus attributes', () => {
    const html = renderToStaticMarkup(createElement(T3ModelPicker, { snap: makeSnapshot({ pickerOpen: true }) }))
    expect(html).toContain('data-t3-model-picker="true"')
    expect(html).toContain('role="combobox"')
    expect(html).toContain('aria-activedescendant=')
    expect(html).toContain('data-option-index="0"')
    expect(html).toContain('is-focused')
  })

  test('calls store.closePicker() on close, binds ArrowDown/ArrowUp/Enter, and never mutates live ACP runtime on rail clicks', () => {
    const src = readFileSync(join(RENDERER_DIR, 'T3ModelPicker.tsx'), 'utf8')
    expect(src).toContain('store.closePicker()')
    expect(src).toContain("event.key === 'ArrowDown'")
    expect(src).toContain("event.key === 'ArrowUp'")
    expect(src).toContain("event.key === 'Enter'")

    const railStart = src.indexOf('const selectRailProvider =')
    const railEnd = src.indexOf('const activeCursor =')
    const railFn = src.slice(railStart, railEnd)
    expect(railFn.length).toBeGreaterThan(20)
    expect(railFn).not.toContain('disconnectAcpAgent')
    expect(railFn).not.toContain('connectAcpAgent')
  })
})

describe('Fix 4: Topbar Breadcrumb double-click rename & dedicated actions trigger', () => {
  test('separates double-clickable thread title from Thread actions menu button', () => {
    stampStyle('vibes')
    const html = renderToStaticMarkup(createElement(Shell, { snap: makeSnapshot() }))
    expect(html).toContain('title="Double-click to rename thread"')
    expect(html).toContain('aria-label="Thread actions"')
    expect(html).toContain('t3-breadcrumb__actions')
    expect(html).not.toContain('Double-click to rename · Click for thread actions')
  })
})

describe('Fix 5: Markdown streaming fences/math, <ol>, and pipe-in-backticks tables', () => {
  const caret = createElement('span', { className: 'caret' })

  test('hasOpenStreamingBlock tracks unclosed code fences and display math blocks', () => {
    expect(hasOpenStreamingBlock(['intro', '```ts', 'const x = 1'])).toBe(true)
    expect(hasOpenStreamingBlock(['intro', '```ts', 'const x = 1', '```', 'after'])).toBe(false)
    expect(hasOpenStreamingBlock(['~~~py', 'def f():'])).toBe(true)
    expect(hasOpenStreamingBlock(['````md', '```ts', 'code', '```', 'still inside'])).toBe(true)
    expect(hasOpenStreamingBlock(['$$', '\\int_0^1 x dx'])).toBe(true)
    expect(hasOpenStreamingBlock(['$$', '\\int_0^1 x dx', '$$', 'done'])).toBe(false)
    expect(hasOpenStreamingBlock(['\\[', 'a^2 + b^2 = c^2'])).toBe(true)
    expect(hasOpenStreamingBlock(['$$x^2$$', 'plain text'])).toBe(false)
  })

  test('streaming unclosed fenced code and display math never collapse into a single-line <p>', () => {
    const codeHtml = renderToStaticMarkup(
      createElement(Markdown, {
        text: 'Example:\n\n```ts\nfunction sum(a: number, b: number) {\n  return a + b',
        tail: caret,
      }),
    )
    expect(codeHtml).toContain('class="md__code"')
    expect(codeHtml).toContain('function')
    expect(codeHtml).not.toContain('<p>function sum')

    const mathHtml = renderToStaticMarkup(
      createElement(Markdown, {
        text: 'Formula:\n\n$$\nE = mc^2\n+ p^2',
        tail: caret,
      }),
    )
    expect(mathHtml).toContain('class="md__mathblock"')
    expect(mathHtml).not.toContain('<p>$$')
  })

  test('ordered lists emit <ol> with start attribute when != 1 and separate from <ul>', () => {
    const html = renderToStaticMarkup(
      createElement(Markdown, {
        text: '1. First\n2. Second\n\n3. Third\n4. Fourth\n- Bullet',
      }),
    )
    expect(html).toContain('<ol><li><span class="md__item">First</span></li><li><span class="md__item">Second</span></li></ol>')
    expect(html).toContain('<ol start="3"><li><span class="md__item">Third</span></li><li><span class="md__item">Fourth</span></li></ol>')
    expect(html).toContain('<ul><li><span class="md__item">Bullet</span></li></ul>')
  })

  test('splitTableRow preserves pipes inside backticks and escaped pipes', () => {
    expect(splitTableRow('| Type | Fallback |')).toEqual(['Type', 'Fallback'])
    expect(splitTableRow('| `string | number` | `a \\| b` |')).toEqual(['`string | number`', '`a | b`'])
    expect(splitTableRow('| foo \\| bar | baz |')).toEqual(['foo | bar', 'baz'])

    const tableHtml = renderToStaticMarkup(
      createElement(Markdown, {
        text: '| Type | Notes |\n| --- | --- |\n| `string | number` | union \\| pipe |',
      }),
    )
    expect(tableHtml).toContain('<td><code>string | number</code></td>')
    expect(tableHtml).toContain('<td>union | pipe</td>')
  })
})

describe('Fix 6: QuestionCard ⌘↵ submission and valid ARIA', () => {
  test('renders sibling radio/checkbox toggle button beside freeform input (no <input> inside <label role="radio">)', () => {
    stampStyle('vibes')
    const html = renderToStaticMarkup(
      createElement(QuestionCard, {
        plan: null,
        question: {
          requestId: 'q-1',
          items: [
            {
              id: 'approach',
              question: 'Which approach should we take?',
              options: ['Option A — Fast', 'Option B — Thorough'],
              allowFreeform: true,
              placeholder: 'Describe another approach',
            },
          ],
        },
      }),
    )
    expect(html).not.toContain('<label class="vq__opt vq__opt--other"')
    expect(html).toContain('class="vq__other-toggle"')
    expect(html).toContain('aria-label="Other answer"')
    expect(html).toContain('class="vq__other-input"')
    expect(html).toContain('<textarea')

    const appSrc = readFileSync(join(RENDERER_DIR, 'App.tsx'), 'utf8')
    const qSlice = appSrc.slice(appSrc.indexOf('export function QuestionCard('), appSrc.indexOf('function PlanReviewCard('))
    expect(qSlice).toContain('handleCardKeyDown')
    expect(qSlice).toContain('if (modEnter)')

    // Multi-question cards render dedicated numbered badges (never Markdown <ol> list indentation),
    // full-width Other textareas, and a clear remaining-question status indicator in the footer.
    const multiHtml = renderToStaticMarkup(
      createElement(QuestionCard, {
        plan: null,
        question: {
          requestId: 'q-multi',
          items: [
            { id: 'q1', question: 'Which commit should be exported?', options: ['Current main', 'Exact commit'], allowFreeform: true },
            { id: 'q2', question: 'How should we handle Ray?', options: ['Stop Ray', 'Leave running'], allowFreeform: true },
          ],
        },
      }),
    )
    expect(multiHtml).toContain('class="vq__num"')
    expect(multiHtml).not.toContain('<ol><li><span class="md__item">Which commit')
    expect(multiHtml).toContain('0 of 2 answered · Answer #1 to submit')
    expect(multiHtml.match(/class="vq__other-input"/g)).toHaveLength(2)

    const vibesCss = readFileSync(join(RENDERER_DIR, 'vibes.css'), 'utf8')
    expect(vibesCss).toContain('.vq__item+.vq__item{border-top:1px solid var(--x-separator)}')
    expect(vibesCss).toContain('.vq__other-input{flex:1 1 auto;width:100%;min-width:0;')
  })
})

describe('Fix 7: Stale filesExpanded & railTabs pruning across context changes', () => {
  test('pruneRailTabsForContext prunes files:* on cwd change and docs:* on session change', () => {
    const tabs = [
      { panel: 'terminal' as const },
      { panel: 'files' as const, item: 'src/a.ts', title: 'a.ts' },
      { panel: 'docs' as const, item: 'art-1', title: 'Spec' },
    ]
    const onCwd = pruneRailTabsForContext(tabs, 'files:src/a.ts', { cwdChanged: true, sessionChanged: false })
    expect(onCwd.tabs.map(t => (t.item ? `${t.panel}:${t.item}` : t.panel))).toEqual(['terminal', 'docs:art-1'])
    expect(onCwd.activeKey).toBe('docs:art-1')

    const onSession = pruneRailTabsForContext(tabs, 'docs:art-1', { cwdChanged: false, sessionChanged: true })
    expect(onSession.tabs.map(t => (t.item ? `${t.panel}:${t.item}` : t.panel))).toEqual(['terminal', 'files:src/a.ts'])
    expect(onSession.activeKey).toBe('files:src/a.ts')

    const onlyScoped = pruneRailTabsForContext([{ panel: 'files', item: 'src/a.ts' }], 'files:src/a.ts', {
      cwdChanged: true,
      sessionChanged: false,
    })
    expect(onlyScoped.tabs).toEqual([])
    expect(onlyScoped.activeKey).toBeNull()
  })
})

describe('Fix 8: ThreadDetailsCard overlap math, agent gating & T3Icon standardization', () => {
  test('isDetailsInlineEligible requires sidebarWidth + 728 + 2*276 and rejects agentOpen', () => {
    const base = {
      vibesStyle: true,
      isVscode: false,
      noWorkspace: false,
      railOpen: false,
      railLauncherOpen: false,
      agentOpen: false,
      focused: false,
      sidebarWidth: 270,
    }
    const minWithSidebar = 270 + CONVERSATION_COLUMN_WIDTH + 2 * THREAD_DETAILS_MIN_GUTTER
    expect(minWithSidebar).toBe(1550)
    expect(isDetailsInlineEligible({ ...base, windowWidth: 1220 })).toBe(false)
    expect(isDetailsInlineEligible({ ...base, windowWidth: 1440 })).toBe(false)
    expect(isDetailsInlineEligible({ ...base, windowWidth: 1518 })).toBe(false)
    expect(isDetailsInlineEligible({ ...base, windowWidth: 1550 })).toBe(true)
    expect(isDetailsInlineEligible({ ...base, focused: true, windowWidth: 1280 })).toBe(true)
    expect(isDetailsInlineEligible({ ...base, windowWidth: 1800, agentOpen: true })).toBe(false)
  })

  test('ThreadDetailsCard uses strictly T3Icon with zero Phosphor Icon imports', () => {
    const src = readFileSync(join(RENDERER_DIR, 'ThreadDetailsCard.tsx'), 'utf8')
    expect(src).not.toContain("from './Icon.js'")
    expect(src).not.toContain('<Icon ')
    const html = renderToStaticMarkup(
      createElement(ThreadDetailsCard, {
        snap: makeSnapshot({
          swarm: { active: true, session_id: 's1', title: 'S', board: 'local', open_board: '', head: 1, overview: null, channels: [], members: [] },
          todos: [{ id: '1', content: 'Verify', status: 'completed' }],
        }),
        mode: 'popover',
        onClose: () => {},
        onNavigate: () => {},
        onOpenTerminal: () => {},
      }),
    )
    expect(html).toContain('t3-icon')
    expect(html).not.toContain('app-icon')
    // Default quick-open rows under Immortals: Artifacts, Terminal, Activity in order
    const immortalsIdx = html.indexOf('>Immortals<')
    const artifactsIdx = html.indexOf('>Artifacts<')
    const terminalIdx = html.indexOf('>Terminal<')
    const activityIdx = html.indexOf('>Activity<')
    expect(immortalsIdx).toBeGreaterThan(-1)
    expect(artifactsIdx).toBeGreaterThan(immortalsIdx)
    expect(terminalIdx).toBeGreaterThan(artifactsIdx)
    expect(activityIdx).toBeGreaterThan(terminalIdx)
  })
})

describe('Fix 9: Transcript Scroll anchoring, deduplication & trackpad disengagement', () => {
  test('captureHistoryScrollAnchor + restoreHistoryScrollAnchor preserve reading offset across prepends', () => {
    let anchorTop = 140
    const anchorNode = {
      dataset: { historyAnchor: 'blk-42' },
      getBoundingClientRect: () => ({ top: anchorTop, bottom: anchorTop + 40 }),
    }
    const container = {
      scrollTop: 80,
      scrollHeight: 1200,
      getBoundingClientRect: () => ({ top: 100, bottom: 700 }),
      querySelectorAll: () => [anchorNode] as unknown as NodeListOf<HTMLElement>,
    }
    const snap = captureHistoryScrollAnchor(container)
    expect(snap.anchorId).toBe('blk-42')
    expect(snap.anchorOffset).toBe(40)

    // Prepending 50 older groups shifts the anchored node down by 1600px.
    anchorTop = 1740
    container.scrollHeight = 2800
    expect(restoreHistoryScrollAnchor(container, snap)).toBe(1680)
  })

  test('nextTranscriptFollowState unpins on 1px upward scroll and keeps repin threshold at 24px', () => {
    expect(REPIN_BOTTOM_THRESHOLD_PX).toBe(24)
    const up1px = nextTranscriptFollowState({
      wasFollowing: true,
      prevScrollTop: 1000,
      scrollTop: 999,
      clientHeight: 600,
      scrollHeight: 1600,
      lastFollowTop: 1000,
    })
    expect(up1px.following).toBe(false)

    const appSrc = readFileSync(join(RENDERER_DIR, 'App.tsx'), 'utf8')
    expect(appSrc).toContain('automatic: !snap.historyError && hiddenGroupCount === 0')
    const tsSrc = readFileSync(join(RENDERER_DIR, 'transcriptScroll.ts'), 'utf8')
    expect(tsSrc).toContain('pendingPromise.current')
  })
})

describe('Fix 10: ActionDialog, OutputViewer, FindBar, AppUpdatePrompt, SessionMenu & Brand Cohesion', () => {
  test('ActionDialog portals inside .atelier and uses btn--solid primary CTA', () => {
    const html = renderToStaticMarkup(
      createElement(ActionDialog, {
        title: 'Confirm action',
        confirmLabel: 'Proceed',
        onConfirm: () => {},
        onCancel: () => {},
        children: createElement('p', null, 'Details'),
      }),
    )
    expect(html).toContain('class="atelier action-dialog-layer"')
    expect(html).toContain('class="btn btn--solid"')
    expect(html).not.toContain('btn-primary')
  })

  test('OutputViewer uses copyToClipboard with timer cleanup; FindBar collapses selection on query change', () => {
    const outputSrc = readFileSync(join(RENDERER_DIR, 'OutputViewer.tsx'), 'utf8')
    expect(outputSrc).toContain("from './CopyButton.js'")
    expect(outputSrc).toContain('copyToClipboard(text)')
    expect(outputSrc).toContain('clearTimeout(copyTimer.current)')

    const findSrc = readFileSync(join(RENDERER_DIR, 'FindBar.tsx'), 'utf8')
    expect(findSrc).toContain('window.getSelection()?.removeAllRanges()')
    expect(findSrc).toContain('lastQuery.current')
  })

  test('AppUpdatePrompt guards missing total bytes and registers window Escape listener', () => {
    const unknownTotal = renderToStaticMarkup(
      createElement(AppUpdatePrompt, {
        update: appUpdateView({ phase: 'downloading', release: { version: '0.9.0' } }),
      }),
    )
    expect(unknownTotal).toContain('Downloading…')
    expect(unknownTotal).not.toContain('( of )')

    const partialOnly = renderToStaticMarkup(
      createElement(AppUpdatePrompt, {
        update: appUpdateView({ phase: 'downloading', received: 12_582_912, release: { version: '0.9.0' } }),
      }),
    )
    expect(partialOnly).toContain('Downloading… 12 MB')
    expect(partialOnly).not.toContain(' of ')

    const promptSrc = readFileSync(join(RENDERER_DIR, 'AppUpdatePrompt.tsx'), 'utf8')
    expect(promptSrc).toContain("window.addEventListener('keydown', onWindowKeyDown)")
  })

  test('SessionMenu clamps viewport height when Delete confirmation expands', () => {
    const appSrc = readFileSync(join(RENDERER_DIR, 'App.tsx'), 'utf8')
    const menuSlice = appSrc.slice(appSrc.indexOf('function SessionMenu('), appSrc.indexOf('// ── Statusline'))
    expect(menuSlice).toContain('confirmingDelete, menu?.pending, menu?.error')
    expect(menuSlice).toContain('new ResizeObserver(place)')
  })

  test('Brand cohesion: sidebar wordmark is Xerxes Agents and micro brand icons render crisp vector glyphs instead of smudged PNGs', () => {
    const sidebarHtml = renderToStaticMarkup(
      createElement(T3Sidebar, {
        snap: makeSnapshot(),
        page: null,
        sections: [],
        allRows: [],
        filter: '',
        setFilter: () => {},
        inbox: parseThreadInboxState(null),
        undo: null,
        onUndo: () => {},
        onDismissUndo: () => {},
        onSetViewMode: () => {},
        onToggleProjectScope: () => {},
        onClearProjectScope: () => {},
        onTogglePin: () => {},
        onSettle: () => {},
        onSnooze: () => {},
        onToggleShelf: () => {},
        canGoBack: false,
        canGoForward: false,
        onGoBack: () => {},
        onGoForward: () => {},
        onOpen: () => {},
        sidebarToggle: createElement('button', { type: 'button' }, 'Toggle'),
        runtimeStatus: createElement('span', null, 'Connected'),
      }),
    )
    expect(sidebarHtml).toContain('<span class="t3-wordmark__brand">Xerxes</span><span class="t3-wordmark__sub">Agents</span>')
    expect(sidebarHtml).not.toContain('Xerxes Code')

    expect(RASTER_BRAND_MIN_SIZE).toBe(0)
    const microIcon = renderToStaticMarkup(createElement(RuntimeBrandIcon, { provider: 'xerxes', size: 13 }))
    expect(microIcon).toContain('./logo-128.png')
    expect(microIcon).not.toContain('<svg')

    const railIcon = renderToStaticMarkup(createElement(RuntimeBrandIcon, { provider: 'xerxes', size: 18 }))
    expect(railIcon).toContain('./logo-128.png')

    const heroLogo = renderToStaticMarkup(createElement(XerxesLogoMark, { size: 64 }))
    expect(heroLogo).toContain('./logo-128.png')
    expect(heroLogo).toContain('decoding="async"')

    // VS Code host also renders the real Xerxes phoenix icon everywhere (never the split-X SVG).
    holder.document = {
      documentElement: {
        dataset: { host: 'vscode' },
        getAttribute: (name: string) => (name === 'data-host' ? 'vscode' : name === 'data-style' ? 'vibes' : null),
      },
    }
    const vscodeLogo = renderToStaticMarkup(createElement(XerxesLogoMark, { size: 44 }))
    expect(vscodeLogo).toContain('./logo-128.png')
    expect(vscodeLogo).not.toContain('<svg')
    const vscodeWelcome = renderToStaticMarkup(createElement(Shell, { snap: makeSnapshot({ blocks: [] }) }))
    expect(vscodeWelcome).toContain('What should Xerxes do')
    expect(vscodeWelcome).toContain('./logo-128.png')
    expect(vscodeWelcome).not.toContain('data-emblem="xerxes-mark"')
  })
})

describe('Fix 11: Production audit UI/UX regressions (ACP saved state, FilesChangedCard, FailedCard, Reveal)', () => {
  test('isAntigravityAcpSession recognizes disconnected savedAcpBinding and T3ModelPicker displays saved ACP model', () => {
    const savedAgy = {
      session_id: 's-agy',
      agent_id: 'antigravity',
      name: 'Antigravity',
      provider: 'antigravity' as const,
      transport: 'acp' as const,
      cwd: '/workspace/Xerxes-Agents',
      connected: false,
      current_model: 'gemini-3-pro',
      current_model_name: 'Gemini 3 Pro',
    }
    const snap = makeSnapshot({
      model: 'claude-opus-4-6',
      acpBinding: null,
      savedAcpBinding: savedAgy,
    })
    expect(isAntigravityAcpSession(snap)).toBe(true)

    stampStyle('vibes')
    const pickerHtml = renderToStaticMarkup(createElement(T3ModelPicker, { snap }))
    expect(pickerHtml).toContain('Gemini 3 Pro')
    expect(pickerHtml).not.toContain('claude-opus')
  })

  test('FilesChangedCard renders files edited across multiple turns on earlier turn checkpoints', () => {
    stampStyle('vibes')
    const html = renderToStaticMarkup(
      createElement(FilesChangedCard, {
        group: [{ kind: 'checkpoint', id: 10, turn: 1, adds: 4, dels: 1 }],
        changes: [
          {
            path: '/workspace/Xerxes-Agents/src/app.ts',
            adds: 9,
            dels: 2,
            isNew: false,
            hunks: [],
            turn: 2,
            turns: [1, 2],
          },
        ],
        cwd: '/workspace/Xerxes-Agents',
      }),
    )
    expect(html).toContain('Files changed in turn 1')
    expect(html).toContain('1 changed file')
    expect(html).toContain('>src<')
  })

  test('FailedCard keeps Retry enabled even when failed.lastUser is empty', () => {
    const html = renderToStaticMarkup(
      createElement(FailedCard, {
        failed: { error: 'ACP exited unexpectedly', turn: 3, lastUser: '' },
      }),
    )
    expect(html).toContain('Retry')
    expect(html).not.toContain('disabled=""')
  })
})
