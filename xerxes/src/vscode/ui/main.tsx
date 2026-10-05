// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Entry for the VS Code chat view. Colours, fonts and light/dark come from
 * the VS Code theme (vscode.css maps the app's tokens onto `--vscode-*`), so
 * the app's own appearance settings and backgrounds are not applied here.
 */

import { createRoot } from 'react-dom/client'

import { ErrorBoundary } from '../../desktop/renderer/ErrorBoundary.js'
import { VscodeApp } from './VscodeApp.js'

const root = document.documentElement

/** VS Code marks the body vscode-light / vscode-dark / vscode-high-contrast(-light); follow it live. */
const followTheme = (): void => {
  const classes = document.body.classList
  const light = classes.contains('vscode-light') || classes.contains('vscode-high-contrast-light')
  root.setAttribute('data-theme', light ? 'light' : 'dark')
  root.toggleAttribute('data-high-contrast', classes.contains('vscode-high-contrast') || classes.contains('vscode-high-contrast-light'))
}
followTheme()
new MutationObserver(followTheme).observe(document.body, { attributes: true, attributeFilter: ['class'] })

// No focus-based animation pause here: the view stays in sight while the
// editor has focus, and a frozen spinner beside it reads as a hung task.

const container = document.getElementById('root')
if (container) createRoot(container).render(<ErrorBoundary><VscodeApp /></ErrorBoundary>)
