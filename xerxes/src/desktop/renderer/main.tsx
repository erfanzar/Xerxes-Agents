// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createRoot } from 'react-dom/client'

import { App } from './App.js'

// Theme: follow the system only while the user has not pinned an explicit
// choice (Settings writes data-user-theme alongside data-theme).
const root = document.documentElement
const applyTheme = (): void => {
  if (root.hasAttribute('data-user-theme')) return
  root.setAttribute(
    'data-theme',
    window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark',
  )
}
applyTheme()
window.matchMedia('(prefers-color-scheme: light)').addEventListener('change', applyTheme)

const container = document.getElementById('root')
if (container) {
  createRoot(container).render(<App />)
}
