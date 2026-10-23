// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { defineConfig } from 'vitest/config'

// No DOM environment is installed (neither jsdom nor happy-dom), so these tests
// are pure reducers, fakes and react-dom/server renders in the node environment.
export default defineConfig({
  oxc: { jsx: { runtime: 'automatic' } },
  test: {
    environment: 'node',
    include: ['src/mobile/__tests__/**/*.test.{ts,tsx}'],
    exclude: ['dist/**', 'node_modules/**', '**/node_modules/**']
  }
})
