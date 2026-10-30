// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Entry for the VS Code chat view. Light/dark follows the VS Code theme. The
 * Style setting (Xerxes: Style) picks the look: Neo (the default) draws its
 * own colours and fonts with neo.css; Classic maps the app's tokens onto the
 * VS Code theme (vscode.css). High-contrast themes always keep VS Code's
 * colours. Backgrounds and colour themes are desktop-only.
 */

import { createRoot } from 'react-dom/client'

import { applyStyle, isStyleChoice, type StyleChoice } from '../../desktop/renderer/appearance.js'
import { ErrorBoundary } from '../../desktop/renderer/ErrorBoundary.js'
import { VscodeApp } from './VscodeApp.js'

const root = document.documentElement
/** The saved setting, stamped into the page by the extension and updated live. */
const stamped = root.getAttribute('data-style-choice')
let chosenStyle: StyleChoice = isStyleChoice(stamped) ? stamped : 'neo'

/** VS Code marks the body vscode-light / vscode-dark / vscode-high-contrast(-light); follow it live. */
const followTheme = (): void => {
  const classes = document.body.classList
  const light = classes.contains('vscode-light') || classes.contains('vscode-high-contrast-light')
  const highContrast = classes.contains('vscode-high-contrast') || classes.contains('vscode-high-contrast-light')
  root.setAttribute('data-theme', light ? 'light' : 'dark')
  root.toggleAttribute('data-high-contrast', highContrast)
  applyStyle(highContrast ? 'classic' : chosenStyle, root)
}
followTheme()
new MutationObserver(followTheme).observe(document.body, { attributes: true, attributeFilter: ['class'] })
window.xerxes.onStyle?.(style => {
  chosenStyle = style
  root.setAttribute('data-style-choice', style)
  followTheme()
})

// No focus-based animation pause here: the view stays in sight while the
// editor has focus, and a frozen spinner beside it reads as a hung task.

const container = document.getElementById('root')
if (container) createRoot(container).render(<ErrorBoundary><VscodeApp /></ErrorBoundary>)
