// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * The Xerxes release this source tree builds. Kept in a dependency-free module
 * so clients (desktop main, TUI) can name their own release, for example to
 * install the matching runtime on a remote host, without importing the daemon.
 * Keep in sync with xerxes/package.json "version" (the release bump does).
 */
export const XERXES_VERSION = '0.8.13'
