// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { useEffect, useRef, useState, type ReactElement } from 'react'

import { checkPairLink } from '../repairLink.js'
import { cameraErrorMessage, startQrScanner, type RunningScanner } from '../repairScanner.js'

/** Open a checked pairing link in this same app (same cookie store), which boots the pair page fresh. */
function openInPlace(href: string): void {
  const before = window.location.pathname
  window.location.assign(href)
  // A hash-only change does not reload; the pair page reads its ticket only at boot.
  if (before === '/pair') window.location.reload()
}

/**
 * Signed-out screen with an in-app re-pair: scan the desktop's pairing QR code with the
 * camera here (on iPhone the Camera app would open Safari, whose sign-in is separate
 * from the Home Screen app), or paste the link.
 */
export function SignedOut({ reason, openLink = openInPlace }: { reason: string; openLink?: (href: string) => void }): ReactElement {
  const [scanning, setScanning] = useState(false)
  const [message, setMessage] = useState<string | undefined>(undefined)
  const [pasted, setPasted] = useState('')
  const video = useRef<HTMLVideoElement>(null)

  const open = (raw: string): boolean => {
    const check = checkPairLink(raw, window.location.origin)
    if (!check.ok) { setMessage(check.message); return false }
    setMessage(undefined)
    openLink(check.href)
    return true
  }

  useEffect(() => {
    const element = video.current
    if (!scanning || element === null) return undefined
    let cancelled = false
    let running: RunningScanner | undefined
    let last = ''
    startQrScanner(element, text => {
      if (cancelled || text === last) return
      last = text
      if (open(text)) { cancelled = true; running?.stop() }
    }).then(scanner => {
      if (cancelled) scanner.stop()
      else running = scanner
    }).catch((error: unknown) => {
      if (cancelled) return
      setMessage(cameraErrorMessage(error))
      setScanning(false)
    })
    return () => { cancelled = true; running?.stop() }
    // open() only reads stable setters and props.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scanning])

  return (
    <section aria-labelledby="revoked-title" className="repair">
      <h1 id="revoked-title" className="page-title">Signed out</h1>
      <p>{reason}</p>
      <ol className="repair-steps">
        <li>On the desktop, open the Hub and choose <strong>Pair a phone</strong>.</li>
        <li>Tap <strong>Scan pairing QR code</strong> below and point this phone at the code.</li>
        <li>If the desktop asks, check that both screens show the same code and confirm it.</li>
      </ol>
      {scanning ? (
        <div className="repair-scanner">
          <video ref={video} className="repair-video" muted playsInline aria-label="Camera view for the pairing QR code" />
          <p role="status">Point the camera at the QR code on the desktop.</p>
          <button type="button" className="btn" onClick={() => setScanning(false)}>Cancel</button>
        </div>
      ) : (
        <button type="button" className="btn btn-primary repair-scan" onClick={() => { setMessage(undefined); setScanning(true) }}>Scan pairing QR code</button>
      )}
      <p className="repair-note">Scan here rather than with the iPhone Camera app: the Camera app opens Safari, which signs in Safari instead of this app.</p>
      <details className="repair-paste">
        <summary>Paste the pairing link instead</summary>
        <form onSubmit={event => { event.preventDefault(); open(pasted) }}>
          <label className="field">
            <span>Pairing link</span>
            <input type="text" inputMode="url" autoComplete="off" autoCapitalize="off" autoCorrect="off" spellCheck={false} value={pasted} onChange={event => setPasted(event.target.value)} />
          </label>
          <button type="submit" className="btn">Open link</button>
        </form>
      </details>
      {message === undefined ? null : <p role="alert" className="field-error">{message}</p>}
      <p className="repair-note">This page will not reconnect on its own.</p>
    </section>
  )
}
