// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { useEffect, useRef, useState, type ReactElement } from 'react'
import { desktopCall, desktopError, records, text, type RpcRecord } from './desktopRpc.js'
import { Icon, type IconName } from './Icon.js'

const isDirectoryRow = (row: RpcRecord): boolean => row.meta === 'dir' || row.kind === 'directory'
const rowName = (row: RpcRecord): string => (text(row.label) || text(row.value)).replace(/\/$/, '').split('/').at(-1) ?? ''

/** Folders first, then files; each case-insensitive by name (dotfiles sort by the name after the dot). */
export function sortTreeRows(rows: readonly RpcRecord[]): RpcRecord[] {
  const key = (row: RpcRecord): string => rowName(row).replace(/^\./, '').toLowerCase()
  return [...rows].sort((a, b) => Number(isDirectoryRow(b)) - Number(isDirectoryRow(a)) || key(a).localeCompare(key(b)) || rowName(a).localeCompare(rowName(b)))
}

/** Compact file age ("17h", "4d"); empty when the runtime sent no time. */
export function fileAgeOf(iso: string, now = Date.now()): string {
  const at = Date.parse(iso)
  if (!iso || !Number.isFinite(at)) return ''
  const s = Math.max(0, Math.round((now - at) / 1000))
  return s < 60 ? 'now' : s < 3600 ? `${Math.floor(s / 60)}m` : s < 86400 ? `${Math.floor(s / 3600)}h` : `${Math.floor(s / 86400)}d`
}

/** The file's type, for a type-coloured icon (the icon is coloured, not the name). */
export function fileTypeOf(name: string): { icon: IconName; type: string } {
  const lower = name.toLowerCase()
  const ext = lower.includes('.') ? lower.slice(lower.lastIndexOf('.') + 1) : ''
  if (lower === 'dockerfile' || lower.startsWith('dockerfile.')) return { icon: 'stack', type: 'docker' }
  if (lower === 'license' || lower.startsWith('license.')) return { icon: 'file', type: 'plain' }
  if (['ts', 'tsx', 'js', 'jsx', 'mjs', 'cjs'].includes(ext)) return { icon: 'file', type: ext.startsWith('t') ? 'ts' : 'js' }
  if (['md', 'mdx', 'txt', 'rst'].includes(ext)) return { icon: 'note', type: 'doc' }
  if (['json', 'jsonc'].includes(ext)) return { icon: 'file', type: 'json' }
  if (['yml', 'yaml', 'toml', 'ini', 'env'].includes(ext)) return { icon: 'settings', type: 'config' }
  if (['css', 'scss', 'less'].includes(ext)) return { icon: 'file', type: 'css' }
  if (['lock', 'lockb'].includes(ext)) return { icon: 'shield', type: 'lock' }
  if (['png', 'jpg', 'jpeg', 'gif', 'svg', 'webp', 'ico', 'icns'].includes(ext)) return { icon: 'eye', type: 'image' }
  if (['sh', 'bash', 'zsh'].includes(ext)) return { icon: 'terminal', type: 'shell' }
  if (['py', 'rs', 'go', 'rb', 'java', 'c', 'cpp', 'h'].includes(ext)) return { icon: 'file', type: 'code' }
  return { icon: 'file', type: 'plain' }
}

/** Lazy, independent branches: expanding a folder keeps its siblings in place. */
export function WorkspaceFileTree({ sessionKey, path = './', selected, select, level = 1, changed }: {
  sessionKey: string; path?: string; selected: string; select: (path: string) => void; level?: number
  /** Session-changed files, workspace-relative without `./`, mapped to 'A' (added) or 'M' (modified). */
  changed?: ReadonlyMap<string, 'A' | 'M'>
}): ReactElement {
  const [rows, setRows] = useState<RpcRecord[]>([])
  const cachedRows = useRef<RpcRecord[]>([])
  const [offset, setOffset] = useState(0)
  const [more, setMore] = useState(false)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [retry, setRetry] = useState(0)
  useEffect(() => {
    let current = true
    setLoading(true); setError('')
    void desktopCall(window.xerxes, sessionKey, 'complete', { path_prefix: path, path_offset: offset })
      .then(result => {
        if (!current) return
        const next = records(result.completions)
        const fresh = offset ? next.filter(row => !cachedRows.current.some(existing => existing.value === row.value)) : next
        if (offset && next.length && !fresh.length) {
          setError('This workspace runtime cannot page folders. Filter by filename, or update the runtime and reopen Files.')
          setMore(false)
        } else setMore(next.length === 50)
        cachedRows.current = sortTreeRows(offset ? [...cachedRows.current, ...fresh] : fresh)
        setRows(cachedRows.current)
      })
      .catch(reason => { if (current) setError(desktopError(reason)) })
      .finally(() => { if (current) setLoading(false) })
    return () => { current = false }
  }, [sessionKey, path, offset, retry])
  return <div role={level === 1 ? 'tree' : 'group'} aria-label={level === 1 ? 'Workspace files' : undefined}>
    {rows.map(row => <FileTreeEntry key={text(row.value)} row={row} sessionKey={sessionKey} selected={selected} select={select} level={level} {...(changed ? { changed } : {})} />)}
    {loading && <p className="studio-muted" role="status">Loading folder…</p>}
    {error && <div role="alert">{error} <button onClick={() => setRetry(value => value + 1)}>Retry folder</button></div>}
    {!loading && !error && !rows.length && <p className="studio-muted">Empty folder</p>}
    {more && !loading && !error && <button onClick={() => setOffset(rows.length)}>Load more entries</button>}
  </div>
}

function FileTreeEntry({ row, sessionKey, selected, select, level, changed }: {
  row: RpcRecord; sessionKey: string; selected: string; select: (path: string) => void; level: number
  changed?: ReadonlyMap<string, 'A' | 'M'>
}): ReactElement {
  const path = text(row.value).replace(/^@/, '')
  const directory = isDirectoryRow(row)
  const name = rowName(row)
  const type = directory ? null : fileTypeOf(name)
  const rel = path.replace(/^\.\//, '').replace(/\/$/, '')
  const status = !directory ? changed?.get(rel) : undefined
  // A folder holding a changed file carries the small dot.
  const holdsChange = directory && changed !== undefined && [...changed.keys()].some(file => file.startsWith(rel + '/'))
  const [open, setOpen] = useState(false)
  const [loaded, setLoaded] = useState(false)
  const expand = () => { setOpen(value => !value); setLoaded(true) }
  return <div role="none">
    <button role="treeitem" aria-level={level} aria-expanded={directory ? open : undefined} aria-selected={!directory && selected === path}
      className="file-browser__entry" style={{ paddingLeft: 7 + (level - 1) * 16 }}
      title={(directory ? (open ? 'Collapse folder: ' : 'Expand folder: ') : 'Select file: ') + path}
      onClick={() => directory ? expand() : select(path)}
      onKeyDown={event => {
        if (directory && event.key === 'ArrowRight') { event.preventDefault(); setOpen(true); setLoaded(true) }
        else if (directory && event.key === 'ArrowLeft') { event.preventDefault(); setOpen(false) }
        else if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
          event.preventDefault()
          const entries = [...event.currentTarget.closest('[role=tree]')!.querySelectorAll<HTMLButtonElement>('[role=treeitem]')].filter(item => !item.closest('[hidden]'))
          entries[entries.indexOf(event.currentTarget) + (event.key === 'ArrowDown' ? 1 : -1)]?.focus()
        }
      }}>
      <span className="file-tree__icon" data-type={type?.type ?? 'folder'}><Icon name={directory ? 'folder' : type!.icon} size={16} /></span><span className="file-tree__name" data-change={status === 'A' ? 'added' : status === 'M' ? 'modified' : undefined}>{name}</span>
      {status && <span className="file-browser__badge file-tree__badge" data-change={status === 'A' ? 'added' : 'modified'} aria-label={status === 'A' ? 'Added' : 'Modified'}>{status}</span>}
      {holdsChange && <span className="file-tree__dot" aria-label="Contains changes" />}
      {!directory && !status && fileAgeOf(text(row.modified_at)) && <span className="file-tree__age">{fileAgeOf(text(row.modified_at))}</span>}
      {directory && <span className={'file-tree__chevron' + (open ? ' is-open' : '')}><Icon name="chevron" size={12} /></span>}
    </button>
    {directory && loaded && <div hidden={!open}><WorkspaceFileTree sessionKey={sessionKey} path={path.endsWith('/') ? path : path + '/'} selected={selected} select={select} level={level + 1} {...(changed ? { changed } : {})} /></div>}
  </div>
}
