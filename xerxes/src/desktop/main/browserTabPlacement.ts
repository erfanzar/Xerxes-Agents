// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Pure placement and visibility state transitions for the desktop Browser tab's
 * native Electron `WebContentsView`.
 *
 * Kept free of `electron` imports so the visibility, host-occlusion, and macOS
 * compositor repaint-nudge rules can be exhaustively verified under Bun tests.
 */

export interface BrowserViewRect {
  readonly x: number
  readonly y: number
  readonly width: number
  readonly height: number
}

export interface BrowserViewSize {
  readonly width: number
  readonly height: number
}

export interface BrowserPlacementInput {
  readonly shown: boolean
  readonly hostHidden: boolean
  readonly hasError: boolean
  readonly bounds: BrowserViewRect | null
  readonly wasVisible: boolean
  readonly lastPaintedSize: BrowserViewSize | null
}

export type BrowserPlacementPlan =
  | {
      readonly action: 'hide'
    }
  | {
      readonly action: 'show'
      readonly bounds: BrowserViewRect
      /**
       * On macOS Electron, re-showing a hidden `WebContentsView` at the exact
       * same dimensions can leave its `CALayer`/`IOSurface` uncomposited until
       * a size change allocates a new `viz::LocalSurfaceId`. When present,
       * apply `nudgeBounds` first and restore `bounds` on the next frame.
       */
      readonly nudgeBounds?: BrowserViewRect
      readonly invalidate: boolean
    }

/**
 * Decide whether the native browser view should be shown or hidden and whether
 * a hidden -> visible transition at identical dimensions needs a 1px compositor
 * repaint nudge to prevent macOS blank/black views.
 */
export function planBrowserViewPlacement(input: BrowserPlacementInput): BrowserPlacementPlan {
  const { shown, hostHidden, hasError, bounds, wasVisible, lastPaintedSize } = input
  if (!shown || hostHidden || hasError || !bounds || bounds.width <= 0 || bounds.height <= 0) {
    return { action: 'hide' }
  }
  const sameSize = lastPaintedSize !== null
    && lastPaintedSize.width === bounds.width
    && lastPaintedSize.height === bounds.height
  if (!wasVisible && sameSize && bounds.height > 1) {
    return {
      action: 'show',
      bounds,
      nudgeBounds: {
        x: bounds.x,
        y: bounds.y,
        width: bounds.width,
        height: bounds.height - 1,
      },
      invalidate: true,
    }
  }
  return {
    action: 'show',
    bounds,
    invalidate: !wasVisible,
  }
}
