// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { installServiceWorker, type SwScope } from './swCore.js'

/** Replaced at build time by scripts/buildMobile.ts with the hashed shell asset list and its version. */
declare const __XERXES_SHELL__: readonly string[]
declare const __XERXES_SHELL_VERSION__: string

installServiceWorker(self as unknown as SwScope, { version: __XERXES_SHELL_VERSION__, shell: __XERXES_SHELL__ })
