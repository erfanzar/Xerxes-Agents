// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, test } from 'bun:test'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'

import { planBrowserViewPlacement } from '../src/desktop/main/browserTabPlacement.js'

const desktopDir = join(import.meta.dir, '../src/desktop')

describe('BrowserTab visibility, stacking, and macOS repaint recovery', () => {
  test('nudges height by 1px and invalidates when re-showing a hidden view at the exact same size', () => {
    const rect = { x: 340, y: 92, width: 1080, height: 810 }
    const plan = planBrowserViewPlacement({
      shown: true,
      hostHidden: false,
      hasError: false,
      bounds: rect,
      wasVisible: false,
      lastPaintedSize: { width: 1080, height: 810 },
    })

    expect(plan).toEqual({
      action: 'show',
      bounds: rect,
      nudgeBounds: { x: 340, y: 92, width: 1080, height: 809 },
      invalidate: true,
    })
  })

  test('does not nudge when already visible (continuous resize) or when dimensions changed', () => {
    const rect = { x: 340, y: 92, width: 1080, height: 810 }
    const whileVisible = planBrowserViewPlacement({
      shown: true,
      hostHidden: false,
      hasError: false,
      bounds: rect,
      wasVisible: true,
      lastPaintedSize: { width: 1080, height: 810 },
    })
    expect(whileVisible).toEqual({
      action: 'show',
      bounds: rect,
      invalidate: false,
    })

    const resizedAfterHide = planBrowserViewPlacement({
      shown: true,
      hostHidden: false,
      hasError: false,
      bounds: { x: 20, y: 92, width: 1400, height: 810 },
      wasVisible: false,
      lastPaintedSize: { width: 1080, height: 810 },
    })
    expect(resizedAfterHide).toEqual({
      action: 'show',
      bounds: { x: 20, y: 92, width: 1400, height: 810 },
      invalidate: true,
    })
  })

  test('hides when host is occluded/hidden, panel is hidden, error is active, or bounds are empty', () => {
    const rect = { x: 340, y: 92, width: 1080, height: 810 }
    expect(planBrowserViewPlacement({
      shown: true,
      hostHidden: true,
      hasError: false,
      bounds: rect,
      wasVisible: true,
      lastPaintedSize: { width: 1080, height: 810 },
    })).toEqual({ action: 'hide' })

    expect(planBrowserViewPlacement({
      shown: false,
      hostHidden: false,
      hasError: false,
      bounds: null,
      wasVisible: true,
      lastPaintedSize: { width: 1080, height: 810 },
    })).toEqual({ action: 'hide' })

    expect(planBrowserViewPlacement({
      shown: true,
      hostHidden: false,
      hasError: true,
      bounds: rect,
      wasVisible: true,
      lastPaintedSize: { width: 1080, height: 810 },
    })).toEqual({ action: 'hide' })
  })

  test('main.ts stacks workspace view before syncWindowVisibility and uses setHostHidden to preserve bounds', () => {
    const mainSrc = readFileSync(join(desktopDir, 'main.ts'), 'utf8')
    const activateStart = mainSrc.indexOf('function activateSurface(id: number): void')
    expect(activateStart).toBeGreaterThan(-1)
    const activateBody = mainSrc.slice(activateStart, activateStart + 700)
    const addChildPos = activateBody.indexOf('surface.host.contentView.addChildView(surface.view)')
    const syncVisPos = activateBody.indexOf('syncWindowVisibility(surface.host, true)')
    expect(addChildPos).toBeGreaterThan(-1)
    expect(syncVisPos).toBeGreaterThan(addChildPos)

    expect(mainSrc).toContain('browserTab?.setHostHidden(hidden)')
    expect(mainSrc).not.toContain('if (hidden) browserTab?.setBounds(null)')
  })

  test('BrowserTab disables backgroundThrottling on show and BrowserPanel ignores ancestor/hidden overlays', () => {
    const browserTabSrc = readFileSync(join(desktopDir, 'main', 'browserTab.ts'), 'utf8')
    const browserPanelSrc = readFileSync(join(desktopDir, 'renderer', 'BrowserPanel.tsx'), 'utf8')
    const panelsSrc = readFileSync(join(desktopDir, 'renderer', 'DesktopPanels.tsx'), 'utf8')

    expect(browserTabSrc).toContain('view.webContents.setBackgroundThrottling(false)')
    expect(browserTabSrc).toContain('setHostHidden(hidden: boolean): void')
    expect(browserTabSrc).toContain('planBrowserViewPlacement(')

    expect(browserPanelSrc).toContain('el.contains(box)')
    expect(browserPanelSrc).toContain("window.addEventListener('focus', onWake)")
    expect(browserPanelSrc).toContain("document.addEventListener('visibilitychange', onWake)")
    expect(browserPanelSrc).toContain('sync(!next)')

    expect(panelsSrc).toContain("panel === 'browser' ? `${currentKey}:${snap.cwd}`")
  })

  test('model picker action buttons allow multi-line wrapping with vertical padding instead of fixed 28px clipping', () => {
    const vibesCss = readFileSync(join(desktopDir, 'renderer', 'vibes.css'), 'utf8')
    const primaryRule = /\.t3-mpop__primary\{([^}]*)\}/.exec(vibesCss)?.[1] ?? ''
    const secondaryRule = /\.t3-mpop__secondary\{([^}]*)\}/.exec(vibesCss)?.[1] ?? ''
    expect(primaryRule).toContain('min-height:30px')
    expect(primaryRule).toContain('height:auto')
    expect(primaryRule).toContain('padding:6px 12px')
    expect(secondaryRule).toContain('min-height:30px')
    expect(secondaryRule).toContain('height:auto')
    expect(secondaryRule).toContain('padding:6px 12px')
  })

  test('ConnectionBanner renders above .chat__body so the floating ThreadDetailsCard sits below it without covering Retry now', () => {
    const appSrc = readFileSync(join(desktopDir, 'renderer', 'App.tsx'), 'utf8')
    const vibesCss = readFileSync(join(desktopDir, 'renderer', 'vibes.css'), 'utf8')
    const atelierCss = readFileSync(join(desktopDir, 'renderer', 'atelier.css'), 'utf8')

    const bannerIdx = appSrc.indexOf('<ConnectionBanner snap={snap} />')
    const chatBodyIdx = appSrc.indexOf('<div className="chat__body">')
    const detailsIdx = appSrc.indexOf('<ThreadDetailsCard', chatBodyIdx)
    expect(bannerIdx).toBeGreaterThan(-1)
    expect(chatBodyIdx).toBeGreaterThan(bannerIdx)
    expect(detailsIdx).toBeGreaterThan(chatBodyIdx)

    expect(atelierCss).toContain('.chat__body{position:relative;display:flex;flex:1 1 auto;flex-direction:column;min-height:0;overflow:hidden}')
    expect(atelierCss).toContain('.connection-status{flex:none;')
    expect(vibesCss).toContain('.t3-details-card{position:absolute;top:4px;right:16px;')
  })
})
