// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
export function shortcutLabel(keys: string, platform = typeof window === 'undefined' ? 'darwin' : window.xerxes?.platform ?? 'darwin'): string {
  if (platform === 'darwin') return keys
  return keys.replaceAll('⇧', 'Shift+').replaceAll('⌥', 'Alt+').replaceAll('⌘', 'Ctrl+').replaceAll('⏎', 'Enter').replace(/\+\s+/g, '+')
}
