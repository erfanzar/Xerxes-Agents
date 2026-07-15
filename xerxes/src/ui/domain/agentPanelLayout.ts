// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

export const AGENT_SIDEBAR_BREAKPOINT = 118

/** Show the wide-terminal rail only after delegation has produced something to inspect. */
export const shouldShowAgentSidebar = (terminalWidth: number, agentCount = 0): boolean =>
  agentCount > 0 && terminalWidth >= AGENT_SIDEBAR_BREAKPOINT

export const agentSidebarWidth = (terminalWidth: number): number =>
  Math.max(38, Math.min(48, Math.floor(terminalWidth * 0.3)))

/** Width actually owned by the transcript/composer after an active rail. */
export const agentContentWidth = (terminalWidth: number, agentCount = 0): number =>
  shouldShowAgentSidebar(terminalWidth, agentCount)
    ? Math.max(1, terminalWidth - agentSidebarWidth(terminalWidth))
    : terminalWidth
