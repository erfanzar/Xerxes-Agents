// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
    ...(process.platform === 'win32' ? { maxWorkers: 2 } : {}),
    include: ['src/ui/**/*.test.{ts,tsx}'],
    exclude: ['dist/**', 'node_modules/**', '**/node_modules/**']
  }
})
