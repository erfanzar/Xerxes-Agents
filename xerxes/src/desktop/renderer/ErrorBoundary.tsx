// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * The renderer validates daemon payloads strictly on purpose — `record()`
 * and `records()` throw rather than degrade a typed failure into a
 * successful empty state. Several panels run those validators inside
 * render (the catalog list, the diff line fold), so one unexpected field
 * from a daemon of a different build used to unmount the whole tree and
 * leave a black window with no message and no way back.
 *
 * This boundary keeps the failure local: a panel that throws is replaced by
 * its error and a retry, and only a throw from the shell itself takes the
 * window — and even then it says what happened and offers a reload.
 */

import { Component, Fragment, type ErrorInfo, type ReactNode } from 'react'

interface Props {
  readonly children: ReactNode
  /** Names the surface in the fallback, e.g. "Skills & tools". */
  readonly label?: string
  /** Remount the subtree instead of reloading the window. */
  readonly onRetry?: () => void
}

interface State {
  readonly error: Error | null
  /** Remounts children on retry so a cleared error cannot re-render stale state. */
  readonly attempt: number
}

export class ErrorBoundary extends Component<Props, State> {
  override state: State = { error: null, attempt: 0 }

  static getDerivedStateFromError(error: unknown): Partial<State> {
    return { error: error instanceof Error ? error : new Error(String(error)) }
  }

  override componentDidCatch(error: unknown, info: ErrorInfo): void {
    // The main process collects renderer console output; keep the component
    // trace, which the message alone does not carry.
    console.error('[xerxes] render failed', this.props.label ?? 'shell', error, info.componentStack)
  }

  private retry = (): void => {
    this.setState(previous => ({ error: null, attempt: previous.attempt + 1 }))
    this.props.onRetry?.()
  }

  override render(): ReactNode {
    const { error } = this.state
    // A Fragment, never a wrapper element: the shell's layout selectors are
    // direct-child (`.app__body > .chat`), so an extra node — even
    // display:contents — silently unhooks them.
    if (!error) return <Fragment key={this.state.attempt}>{this.props.children}</Fragment>
    const scoped = Boolean(this.props.label)
    return (
      <div className={'render-error' + (scoped ? ' render-error--panel' : '')} role="alert">
        <strong>{scoped ? `${this.props.label} could not be shown` : 'Xerxes hit an unexpected error'}</strong>
        <p>
          {scoped
            ? 'This usually means the runtime answered with something this app build does not understand. Everything else keeps working.'
            : 'Your session and any running task are safe in the runtime — reloading reconnects to them.'}
        </p>
        <pre>{error.message}</pre>
        <div className="render-error__actions">
          <button className="btn" onClick={this.retry}>Try again</button>
          <button className="btn" onClick={() => window.location.reload()}>Reload window</button>
        </div>
      </div>
    )
  }
}
