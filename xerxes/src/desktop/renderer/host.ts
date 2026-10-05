// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** True inside the VS Code extension's webview (its page sets data-host="vscode"). */
export function isVscodeHost(): boolean {
  return typeof document !== 'undefined' && document.documentElement.dataset.host === 'vscode'
}
