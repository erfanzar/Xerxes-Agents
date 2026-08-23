// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Terminal width the agents rail needs before it mounts (mockup 11: the rail
 * lives at ≥96 cols; below it F6 is the only path).
 */
export const AGENT_SIDEBAR_BREAKPOINT = 96

/** Show the wide-terminal rail only after delegation has produced something to inspect. */
export const shouldShowAgentSidebar = (terminalWidth: number, agentCount = 0): boolean =>
  agentCount > 0 && terminalWidth >= AGENT_SIDEBAR_BREAKPOINT

/**
 * Whether the rail should actually be mounted right now.
 *
 * The F6 overlay renders the same records from the same store, so mounting both
 * listed every agent twice — once dimmed behind the backdrop, once in the panel.
 * Distinct from {@link shouldShowAgentSidebar}, which answers only whether the
 * rail *fits*; that is the question the footer hint wants, so it stays stable
 * while the overlay is open instead of flipping and flipping back.
 */
export const shouldMountAgentSidebar = (
  terminalWidth: number,
  agentCount: number,
  overlayOpen: boolean
): boolean => !overlayOpen && shouldShowAgentSidebar(terminalWidth, agentCount)

export const agentSidebarWidth = (terminalWidth: number): number =>
  Math.max(38, Math.min(48, Math.floor(terminalWidth * 0.3)))

/** Width actually owned by the transcript/composer after an active rail. */
export const agentContentWidth = (terminalWidth: number, agentCount = 0): number =>
  shouldShowAgentSidebar(terminalWidth, agentCount)
    ? Math.max(1, terminalWidth - agentSidebarWidth(terminalWidth))
    : terminalWidth
