// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
// Small prop-translation layer for controller-adjacent presentation code.
// Most layout names map directly to OpenTUI; this file handles the few that
// differ and enforces non-shrinking stacked rows.
import { createTextAttributes } from '@opentui/core'
import type { ReactNode } from 'react'

type TextWrap = 'truncate-end' | 'wrap' | 'wrap-trim'
type BorderStyle = 'double' | 'round' | 'single'

const BORDER_STYLE_MAP: Record<BorderStyle, 'double' | 'heavy' | 'rounded' | 'single'> = {
  round: 'rounded',
  single: 'single',
  double: 'double'
}

export interface BoxProps {
  children?: ReactNode
  flexDirection?: 'column' | 'row'
  flexGrow?: number
  flexShrink?: number
  flexWrap?: 'nowrap' | 'wrap'
  alignItems?: 'center' | 'flex-end' | 'flex-start' | 'stretch'
  alignSelf?: 'center' | 'flex-end' | 'flex-start' | 'stretch'
  justifyContent?: 'center' | 'flex-end' | 'flex-start' | 'space-between' | 'space-around'
  position?: 'absolute' | 'relative'
  overflow?: 'hidden' | 'visible'
  top?: number | string
  left?: number | string
  right?: number | string
  bottom?: number | string
  width?: number | string
  height?: number | string
  minWidth?: number | string
  minHeight?: number | string
  maxWidth?: number | string
  maxHeight?: number | string
  padding?: number
  paddingX?: number
  paddingY?: number
  paddingTop?: number
  paddingBottom?: number
  paddingLeft?: number
  paddingRight?: number
  margin?: number
  marginX?: number
  marginY?: number
  marginTop?: number
  marginBottom?: number
  marginLeft?: number
  marginRight?: number
  gap?: number
  borderStyle?: BorderStyle
  borderColor?: string
  backgroundColor?: string
  // Compatibility-only no-op. Native floating overlays own their stacking.
  opaque?: boolean
  // OpenTUI exposes pointer-down rather than a synthetic click event.
  onClick?: (event: unknown) => void
  onMouseDown?: (event: unknown) => void
  onMouseUp?: (event: unknown) => void
  onMouseEnter?: (event: unknown) => void
  onMouseLeave?: (event: unknown) => void
  onMouseDrag?: (event: unknown) => void
}

export function Box({
  borderStyle,
  flexDirection,
  flexGrow,
  flexShrink,
  onClick,
  onMouseEnter,
  onMouseLeave,
  ...rest
}: BoxProps) {
  const border = borderStyle ? BORDER_STYLE_MAP[borderStyle] : undefined
  // Default non-growing boxes to flexShrink:0 (grok's discipline). Yoga's
  // default flexShrink:1 lets a column shrink its stacked children below
  // their content height when the parent overflows, collapsing rows into
  // overlapping cells (the corruption). Growing regions (flexGrow set) keep
  // Yoga's default so they can still flex to fill space.
  const resolvedShrink = flexShrink ?? (flexGrow ? undefined : 0)

  return (
    <box
      // This controller-adjacent wrapper accepts a slightly wider set of
      // dimension strings than OpenTUI's public type surface.
      {...(rest as Record<string, unknown>)}
      // Xerxes' Box contract defaults to rows, while Yoga defaults to columns.
      // Keep the explicit row default so unmarked horizontal chrome remains
      // side-by-side.
      border={Boolean(border)}
      borderStyle={border}
      flexDirection={flexDirection ?? 'row'}
      flexGrow={flexGrow}
      flexShrink={resolvedShrink}
      onMouseDown={(onClick ?? rest.onMouseDown) as never}
      onMouseOver={onMouseEnter as never}
      onMouseOut={onMouseLeave as never}
    />
  )
}

export interface TextProps {
  children?: ReactNode
  color?: string
  dimColor?: boolean
  bold?: boolean
  italic?: boolean
  underline?: boolean
  strikethrough?: boolean
  inverse?: boolean
  wrap?: TextWrap
}

const hasStyle = (
  bold?: boolean,
  italic?: boolean,
  underline?: boolean,
  strike?: boolean,
  inverse?: boolean,
  dim?: boolean
) => Boolean(bold || italic || underline || strike || inverse || dim)

export function Text({ children, color, dimColor, bold, italic, underline, strikethrough, inverse, wrap }: TextProps) {
  // Mirror grok-cli's text usage: a plain <text fg={...}> with flexShrink={0}.
  // flexShrink={0} is the load-bearing bit — without it, Yoga's default
  // flexShrink:1 shrinks stacked text rows below their content height when a
  // column overflows, collapsing adjacent rows into the SAME terminal cells
  // (the "two rows interleave / spaces vanish" corruption). Only set
  // `attributes`/`wrapMode`/`truncate` when actually asked — grok leaves them
  // unset on the vast majority of text, and the always-on versions were part
  // of what tripped the renderer.
  const styled = hasStyle(bold, italic, underline, strikethrough, inverse, dimColor)
  const attributes = styled
    ? createTextAttributes({ bold, italic, underline, dim: dimColor, inverse, strikethrough })
    : undefined

  return (
    <text
      attributes={attributes}
      fg={color}
      flexShrink={0}
      truncate={wrap === 'truncate-end' ? true : undefined}
      wrapMode={wrap === 'truncate-end' ? 'none' : wrap === 'wrap' || wrap === 'wrap-trim' ? 'word' : undefined}
    >
      {children}
    </text>
  )
}

// Inline styled run. OpenTUI forbids nesting <text> inside <text> (a
// TextNodeRenderable child must be a string/span/StyledText, not another
// text block) — so any inline colour change WITHIN a line must be a <span>,
// not a nested <Text>. Use inside a <Text>: <Text><Span color=x>a</Span>b</Text>.
export function Span({
  children,
  color,
  dimColor,
  bold
}: {
  children?: ReactNode
  color?: string
  dimColor?: boolean
  bold?: boolean
}) {
  const attributes = bold || dimColor ? createTextAttributes({ bold, dim: dimColor }) : undefined

  return (
    <span attributes={attributes} fg={color}>
      {children}
    </span>
  )
}

// This compatibility boundary can wrap mixed boxes and text, so selection is
// controlled by the native leaf renderables rather than this container.
export function NoSelect({ children, ...rest }: BoxProps) {
  return <Box {...rest}>{children}</Box>
}
