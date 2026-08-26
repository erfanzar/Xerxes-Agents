// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'

import { App } from './App.js'

const container = document.getElementById('root')
if (!container) throw new Error('desktop renderer: #root is missing from the document')

createRoot(container).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
