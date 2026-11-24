// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Crisp 24-grid stroke glyphs matching T3 Code's Lucide chrome iconography.
 * Every glyph inherits `currentColor` (zero hard-coded hex colors) and stays
 * inline so the Electron CSP (`default-src 'none'`) needs no remote assets.
 */

import type { ReactElement } from 'react'
import type { RailPanel } from './railTabs.js'

export type T3IconName =
  | 'panelLeft'
  | 'panelBottom'
  | 'panelRight'
  | 'squareMenu'
  | 'squarePen'
  | 'folderPlus'
  | 'folder'
  | 'folderOpen'
  | 'trash'
  | 'edit'
  | 'files'
  | 'fileText'
  | 'alarmClock'
  | 'timer'
  | 'check'
  | 'checkCheck'
  | 'checkCircle'
  | 'alertCircle'
  | 'undo'
  | 'refresh'
  | 'copy'
  | 'arrowLeft'
  | 'arrowRight'
  | 'sliders'
  | 'filterLines'
  | 'inbox'
  | 'folders'
  | 'layers'
  | 'maximize'
  | 'minimize'
  | 'gitPr'
  | 'gitBranch'
  | 'gitCommitPush'
  | 'fileDiff'
  | 'barChart'
  | 'activity'
  | 'arrowUp'
  | 'stopSquare'
  | 'pin'
  | 'pinOff'
  | 'chevronDown'
  | 'chevronUp'
  | 'chevronRight'
  | 'chevronsUpDown'
  | 'search'
  | 'compass'
  | 'close'
  | 'sparkles'
  | 'terminal'
  | 'terminalSquare'
  | 'worktree'
  | 'paperclip'
  | 'star'
  | 'shieldCheck'
  | 'monitor'
  | 'server'
  | 'globe'
  | 'messagesSquare'
  | 'bot'
  | 'network'
  | 'codeEditor'
  | 'plus'
  | 'circleArrowUp'
  | 'settingsGear'
  | 'workflow'
  | 'image'

/** Cohesive T3 Code Lucide icon mapping for every Right Panel surface. */
export function railT3IconOf(panel: RailPanel): T3IconName {
  switch (panel) {
    case 'browser': return 'globe'
    case 'terminal': return 'terminalSquare'
    case 'files': return 'files'
    case 'review': return 'fileDiff'
    case 'btw': return 'messagesSquare'
    case 'subagents': return 'bot'
    case 'swarm': return 'network'
    case 'docs': return 'layers'
    case 'background': return 'timer'
    case 'outputs': return 'fileText'
    case 'activity': return 'activity'
    case 'usage': return 'barChart'
  }
}

export function T3Icon({
  name,
  size = 15,
  className,
}: {
  readonly name: T3IconName
  readonly size?: number
  readonly className?: string
}): ReactElement {
  return (
    <svg
      className={className ? `t3-icon ${className}` : 't3-icon'}
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.8}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
    >
      {glyphOf(name)}
    </svg>
  )
}

function glyphOf(name: T3IconName): ReactElement {
  switch (name) {
    case 'panelLeft':
      return (
        <>
          <rect width="18" height="18" x="3" y="3" rx="2.5" />
          <path d="M9 3v18" />
        </>
      )
    case 'panelBottom':
      return (
        <>
          <rect width="18" height="18" x="3" y="3" rx="2.5" />
          <path d="M3 15h18" />
        </>
      )
    case 'panelRight':
      return (
        <>
          <rect width="18" height="18" x="3" y="3" rx="2.5" />
          <path d="M15 3v18" />
        </>
      )
    case 'squareMenu':
      return (
        <>
          <rect width="18" height="18" x="3" y="3" rx="2.5" />
          <path d="M7 8h10" />
          <path d="M7 12h10" />
          <path d="M7 16h10" />
        </>
      )
    case 'squarePen':
      return (
        <>
          <path d="M12 3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
          <path d="M18.375 2.625a1 1 0 0 1 3 3l-9.013 9.014a2 2 0 0 1-.853.505l-2.873.84a.5.5 0 0 1-.62-.62l.84-2.873a2 2 0 0 1 .506-.852z" />
        </>
      )
    case 'folderPlus':
      return (
        <>
          <path d="M12 10v6" />
          <path d="M9 13h6" />
          <path d="M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9L9.6 3.9A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2Z" />
        </>
      )
    case 'folder':
      return (
        <path d="M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9L9.6 3.9A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2Z" />
      )
    case 'folderOpen':
      return (
        <>
          <path d="m6 14 1.5-2.9A2 2 0 0 1 9.24 10H20a2 2 0 0 1 1.94 2.5l-1.54 6a2 2 0 0 1-1.95 1.5H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h3.9a2 2 0 0 1 1.69.9l.81 1.2a2 2 0 0 0 1.67.9H18a2 2 0 0 1 2 2v2" />
        </>
      )
    case 'trash':
      return (
        <>
          <path d="M3 6h18" />
          <path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6" />
          <path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2" />
        </>
      )
    case 'edit':
      return (
        <>
          <path d="M17 3a2.85 2.83 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z" />
        </>
      )
    case 'alarmClock':
      return (
        <>
          <circle cx="12" cy="13" r="8" />
          <path d="M12 9v4l2 2" />
          <path d="M5 3 2 6" />
          <path d="m22 6-3-3" />
          <path d="M6.38 18.7 4 21" />
          <path d="M17.64 18.67 20 21" />
        </>
      )
    case 'check':
      return <path d="M20 6 9 17l-5-5" />
    case 'checkCheck':
      return (
        <>
          <path d="M18 6 7 17l-5-5" />
          <path d="m22 10-7.5 7.5L13 16" />
        </>
      )
    case 'undo':
      return (
        <>
          <path d="M3 7v6h6" />
          <path d="M21 17a9 9 0 0 0-9-9 9 9 0 0 0-6 2.3L3 13" />
        </>
      )
    case 'arrowLeft':
      return (
        <>
          <path d="m12 19-7-7 7-7" />
          <path d="M19 12H5" />
        </>
      )
    case 'arrowRight':
      return (
        <>
          <path d="M5 12h14" />
          <path d="m12 5 7 7-7 7" />
        </>
      )
    case 'sliders':
      return (
        <>
          <line x1="21" x2="14" y1="4" y2="4" />
          <line x1="10" x2="3" y1="4" y2="4" />
          <line x1="21" x2="12" y1="12" y2="12" />
          <line x1="8" x2="3" y1="12" y2="12" />
          <line x1="21" x2="16" y1="20" y2="20" />
          <line x1="12" x2="3" y1="20" y2="20" />
          <line x1="14" x2="14" y1="2" y2="6" />
          <line x1="8" x2="8" y1="10" y2="14" />
          <line x1="16" x2="16" y1="18" y2="22" />
        </>
      )
    case 'filterLines':
      return (
        <>
          <path d="M3 6h18" />
          <path d="M6 12h12" />
          <path d="M10 18h4" />
        </>
      )
    case 'inbox':
      return (
        <>
          <polyline points="22 12 16 12 14 15 10 15 8 12 2 12" />
          <path d="M5.45 5.11 2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z" />
        </>
      )
    case 'folders':
      return (
        <>
          <path d="M20 17a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3.9a2 2 0 0 1-1.69-.9l-.81-1.2a2 2 0 0 0-1.67-.9H8a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2Z" />
          <path d="M2 8v11a2 2 0 0 0 2 2h14" />
        </>
      )
    case 'maximize':
      return (
        <>
          <polyline points="15 3 21 3 21 9" />
          <polyline points="9 21 3 21 3 15" />
          <line x1="21" x2="14" y1="3" y2="10" />
          <line x1="3" x2="10" y1="21" y2="14" />
        </>
      )
    case 'minimize':
      return (
        <>
          <polyline points="4 14 10 14 10 20" />
          <polyline points="20 10 14 10 14 4" />
          <line x1="14" x2="21" y1="10" y2="3" />
          <line x1="3" x2="10" y1="21" y2="14" />
        </>
      )
    case 'gitPr':
      return (
        <>
          <circle cx="18" cy="18" r="3" />
          <circle cx="6" cy="6" r="3" />
          <path d="M13 6h3a2 2 0 0 1 2 2v7" />
          <line x1="6" x2="6" y1="9" y2="21" />
        </>
      )
    case 'gitBranch':
      return (
        <>
          <line x1="6" x2="6" y1="3" y2="15" />
          <circle cx="18" cy="6" r="3" />
          <circle cx="6" cy="18" r="3" />
          <path d="M18 9a9 9 0 0 1-9 9" />
        </>
      )
    case 'gitCommitPush':
      return (
        <>
          <circle cx="12" cy="14" r="3" />
          <line x1="3" x2="9" y1="14" y2="14" />
          <line x1="15" x2="21" y1="14" y2="14" />
          <path d="M12 11V4" />
          <path d="m9 7 3-3 3 3" />
        </>
      )
    case 'fileDiff':
      return (
        <>
          <rect width="16" height="18" x="4" y="3" rx="2" />
          <path d="M9 10h6" />
          <path d="M12 7v6" />
          <path d="M9 17h6" />
        </>
      )
    case 'barChart':
      return (
        <>
          <line x1="18" x2="18" y1="20" y2="10" />
          <line x1="12" x2="12" y1="20" y2="4" />
          <line x1="6" x2="6" y1="20" y2="14" />
        </>
      )
    case 'arrowUp':
      return (
        <>
          <path d="m5 12 7-7 7 7" />
          <path d="M12 19V5" />
        </>
      )
    case 'stopSquare':
      return <rect x="6" y="6" width="12" height="12" rx="2.5" fill="currentColor" stroke="none" />
    case 'pin':
      return (
        <>
          <path d="M12 17v5" />
          <path d="M9 10.76a2 2 0 0 1-1.11 1.79l-1.78.9A2 2 0 0 0 5 15.24V16a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-.76a2 2 0 0 0-1.11-1.79l-1.78-.9A2 2 0 0 1 15 10.76V7a1 1 0 0 1 1-1 2 2 0 0 0 0-4H8a2 2 0 0 0 0 4 1 1 0 0 1 1 1z" />
        </>
      )
    case 'pinOff':
      return (
        <>
          <path d="M12 17v5" />
          <path d="M15 9.34V7a1 1 0 0 1 1-1 2 2 0 0 0 0-4H7.89" />
          <path d="m2 2 20 20" />
          <path d="M9 9v1.76a2 2 0 0 1-1.11 1.79l-1.78.9A2 2 0 0 0 5 15.24V16a1 1 0 0 0 1 1h11" />
        </>
      )
    case 'chevronDown':
      return <path d="m6 9 6 6 6-6" />
    case 'chevronUp':
      return <path d="m18 15-6-6-6 6" />
    case 'chevronRight':
      return <path d="m9 18 6-6-6-6" />
    case 'chevronsUpDown':
      return (
        <>
          <path d="m7 15 5 5 5-5" />
          <path d="m7 9 5-5 5 5" />
        </>
      )
    case 'search':
      return (
        <>
          <circle cx="11" cy="11" r="7" />
          <path d="m20 20-4.35-4.35" />
        </>
      )
    case 'close':
      return (
        <>
          <path d="M18 6 6 18" />
          <path d="m6 6 12 12" />
        </>
      )
    case 'sparkles':
      return (
        <>
          <path d="M9.937 15.5A2 2 0 0 0 8.5 14.063l-6.135-1.582a.5.5 0 0 1 0-.962L8.5 9.936A2 2 0 0 0 9.937 8.5l1.582-6.135a.5.5 0 0 1 .963 0L14.063 8.5A2 2 0 0 0 15.5 9.937l6.135 1.581a.5.5 0 0 1 0 .964L15.5 14.063a2 2 0 0 0-1.437 1.437l-1.582 6.135a.5.5 0 0 1-.963 0z" />
        </>
      )
    case 'terminal':
      return (
        <>
          <polyline points="4 17 10 11 4 5" />
          <line x1="12" x2="20" y1="19" y2="19" />
        </>
      )
    case 'worktree':
      return (
        <>
          <circle cx="6" cy="6" r="3" />
          <circle cx="18" cy="6" r="3" />
          <circle cx="6" cy="18" r="3" />
          <path d="M6 9v6" />
          <path d="M18 9a6 6 0 0 1-6 6H9" />
        </>
      )
    case 'paperclip':
      return (
        <path d="m21.44 11.05-9.19 9.19a6 6 0 0 1-8.49-8.49l8.57-8.57A4 4 0 1 1 18 8.84l-8.59 8.57a2 2 0 0 1-2.83-2.83l8.49-8.48" />
      )
    case 'star':
      return (
        <polygon
          points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"
          fill="currentColor"
          stroke="none"
        />
      )
    case 'shieldCheck':
      return (
        <>
          <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10" />
          <path d="m9 12 2 2 4-4" />
        </>
      )
    case 'monitor':
      return (
        <>
          <rect width="20" height="14" x="2" y="3" rx="2" />
          <line x1="8" x2="16" y1="21" y2="21" />
          <line x1="12" x2="12" y1="17" y2="21" />
        </>
      )
    case 'server':
      return (
        <>
          <rect width="20" height="8" x="2" y="2" rx="2" ry="2" />
          <rect width="20" height="8" x="2" y="14" rx="2" ry="2" />
          <line x1="6" x2="6.01" y1="6" y2="6" />
          <line x1="6" x2="6.01" y1="18" y2="18" />
        </>
      )
    case 'codeEditor':
      return (
        <>
          <polyline points="16 18 22 12 16 6" />
          <polyline points="8 6 2 12 8 18" />
        </>
      )
    case 'plus':
      return (
        <>
          <path d="M5 12h14" />
          <path d="M12 5v14" />
        </>
      )
    case 'circleArrowUp':
      return (
        <>
          <circle cx="12" cy="12" r="9.5" />
          <path d="m16 12-4-4-4 4" />
          <path d="M12 16V8" />
        </>
      )
    case 'settingsGear':
      return (
        <>
          <circle cx="12" cy="12" r="3" />
          <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9c.26.604.852.997 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
        </>
      )
    case 'workflow':
      return (
        <>
          <rect width="7" height="7" x="3" y="3" rx="1.5" />
          <path d="M7 10v4a2 2 0 0 0 2 2h5" />
          <rect width="7" height="7" x="14" y="14" rx="1.5" />
        </>
      )
    case 'globe':
      return (
        <>
          <circle cx="12" cy="12" r="9.5" />
          <path d="M2.5 12h19" />
          <path d="M12 2.5a14.5 14.5 0 0 0 0 19 14.5 14.5 0 0 0 0-19" />
        </>
      )
    case 'terminalSquare':
      return (
        <>
          <rect width="18" height="18" x="3" y="3" rx="2.5" />
          <path d="m7.5 15 3-3-3-3" />
          <path d="M12.5 15h4" />
        </>
      )
    case 'files':
      return (
        <>
          <path d="M15 2H9a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h9a2 2 0 0 0 2-2V7Z" />
          <path d="M14 2v5h5" />
          <path d="M4 6v13a2 2 0 0 0 2 2h9" />
        </>
      )
    case 'fileText':
      return (
        <>
          <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5Z" />
          <path d="M14 2v6h6" />
          <path d="M8 13h8" />
          <path d="M8 17h5" />
        </>
      )
    case 'messagesSquare':
      return (
        <>
          <path d="M14 9a2 2 0 0 1-2 2H6l-4 4V4a2 2 0 0 1 2-2h8a2 2 0 0 1 2 2z" />
          <path d="M18 9h2a2 2 0 0 1 2 2v11l-4-4h-6a2 2 0 0 1-2-2v-1" />
        </>
      )
    case 'bot':
      return (
        <>
          <path d="M12 8V4H8" />
          <rect width="16" height="12" x="4" y="8" rx="2.5" />
          <path d="M2 14h2" />
          <path d="M20 14h2" />
          <path d="M15 13v2" />
          <path d="M9 13v2" />
        </>
      )
    case 'network':
      return (
        <>
          <rect x="9" y="2.5" width="6" height="5.5" rx="1.2" />
          <rect x="2.5" y="16" width="6" height="5.5" rx="1.2" />
          <rect x="15.5" y="16" width="6" height="5.5" rx="1.2" />
          <path d="M12 8v4" />
          <path d="M5.5 16v-2.5a1.5 1.5 0 0 1 1.5-1.5h10a1.5 1.5 0 0 1 1.5 1.5V16" />
        </>
      )
    case 'layers':
      return (
        <>
          <path d="m12 2 9 5-9 5-9-5 9-5Z" />
          <path d="m3 12 9 5 9-5" />
          <path d="m3 17 9 5 9-5" />
        </>
      )
    case 'timer':
      return (
        <>
          <line x1="10" x2="14" y1="2.5" y2="2.5" />
          <line x1="12" x2="15" y1="14" y2="11" />
          <circle cx="12" cy="14" r="7.5" />
        </>
      )
    case 'activity':
      return (
        <path d="M22 12h-4l-3 8-6-16-3 8H2" />
      )
    case 'checkCircle':
      return (
        <>
          <circle cx="12" cy="12" r="9.5" />
          <path d="m8.5 12.2 2.4 2.4 4.8-5.1" />
        </>
      )
    case 'alertCircle':
      return (
        <>
          <circle cx="12" cy="12" r="9.5" />
          <line x1="12" x2="12" y1="8" y2="12.5" />
          <line x1="12" x2="12.01" y1="16" y2="16" />
        </>
      )
    case 'copy':
      return (
        <>
          <rect width="13" height="13" x="9" y="9" rx="2" />
          <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
        </>
      )
    case 'refresh':
      return (
        <>
          <path d="M21 12a9 9 0 1 1-2.64-6.36" />
          <path d="M21 3v6h-6" />
        </>
      )
    case 'compass':
      return (
        <>
          <circle cx="12" cy="12" r="9.5" />
          <polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76" />
        </>
      )
    case 'image':
      return (
        <>
          <rect width="18" height="18" x="3" y="3" rx="2" ry="2" />
          <circle cx="9" cy="9" r="2" />
          <path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21" />
        </>
      )
  }
}
