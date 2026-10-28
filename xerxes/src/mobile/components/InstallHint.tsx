// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { useState, type ReactElement } from 'react'

const DISMISSED_KEY = 'xerxes-install-hint-dismissed'

export interface InstallEnvironment {
  readonly standalone: boolean
  readonly ios: boolean
  readonly dismissed: boolean
}

/** Read once from the browser; tests pass an environment instead. */
export function browserInstallEnvironment(): InstallEnvironment {
  if (typeof window === 'undefined') return { standalone: true, ios: false, dismissed: true }
  const nav = window.navigator as Navigator & { standalone?: boolean }
  const standalone = nav.standalone === true || window.matchMedia?.('(display-mode: standalone)').matches === true
  const ios = /iPhone|iPad|iPod/u.test(nav.userAgent) || (nav.platform === 'MacIntel' && nav.maxTouchPoints > 1)
  let dismissed = false
  try { dismissed = window.localStorage.getItem(DISMISSED_KEY) === 'yes' } catch { /* storage is optional */ }
  return { standalone, ios, dismissed }
}

/**
 * Shown only in a browser tab, never in the installed app. On iPhone the Home Screen app also keeps
 * its own sign-in, separate from Safari, so this is how the phone stays signed in.
 */
export function InstallHint({ env = browserInstallEnvironment() }: { env?: InstallEnvironment | undefined }): ReactElement | null {
  const [hidden, setHidden] = useState(env.standalone || env.dismissed)
  if (hidden) return null
  const dismiss = (): void => {
    setHidden(true)
    try { window.localStorage.setItem(DISMISSED_KEY, 'yes') } catch { /* storage is optional */ }
  }
  return (
    <aside className="install-hint" aria-label="Install Xerxes">
      <p>
        <strong>Install Xerxes as an app.</strong>{' '}
        {env.ios
          ? <>Tap <span aria-label="Share">Share</span>, then <strong>Add to Home Screen</strong>, and open Xerxes from there. Pair from the installed app: Safari keeps a separate sign-in.</>
          : <>Use your browser menu: <strong>Install app</strong> or <strong>Add to Home screen</strong>.</>}
      </p>
      <button type="button" className="btn btn-quiet" onClick={dismiss}>Not now</button>
    </aside>
  )
}
