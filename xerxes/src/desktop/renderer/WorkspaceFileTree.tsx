// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { useEffect, useRef, useState, type ReactElement } from 'react'
import { desktopCall, desktopError, records, text, type RpcRecord } from './desktopRpc.js'
import { Icon } from './Icon.js'

/** Lazy, independent branches: expanding a folder keeps its siblings in place. */
export function WorkspaceFileTree({ sessionKey, path = './', selected, select, level = 1 }: {
  sessionKey: string; path?: string; selected: string; select: (path: string) => void; level?: number
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
        cachedRows.current = offset ? [...cachedRows.current, ...fresh] : fresh
        setRows(cachedRows.current)
      })
      .catch(reason => { if (current) setError(desktopError(reason)) })
      .finally(() => { if (current) setLoading(false) })
    return () => { current = false }
  }, [sessionKey, path, offset, retry])
  return <div role={level === 1 ? 'tree' : 'group'} aria-label={level === 1 ? 'Workspace files' : undefined}>
    {rows.map(row => <FileTreeEntry key={text(row.value)} row={row} sessionKey={sessionKey} selected={selected} select={select} level={level} />)}
    {loading && <p className="studio-muted" role="status">Loading folder…</p>}
    {error && <div role="alert">{error} <button onClick={() => setRetry(value => value + 1)}>Retry folder</button></div>}
    {!loading && !error && !rows.length && <p className="studio-muted">Empty folder</p>}
    {more && !loading && !error && <button onClick={() => setOffset(rows.length)}>Load more entries</button>}
  </div>
}

function FileTreeEntry({ row, sessionKey, selected, select, level }: {
  row: RpcRecord; sessionKey: string; selected: string; select: (path: string) => void; level: number
}): ReactElement {
  const path = text(row.value).replace(/^@/, '')
  const directory = row.meta === 'dir' || row.kind === 'directory'
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
      <Icon name={directory ? 'folder' : 'file'} size={16} /><span>{(text(row.label) || path).replace(/\/$/, '').split('/').at(-1)}</span>
      {directory && <span className={'file-tree__chevron' + (open ? ' is-open' : '')}><Icon name="chevron" size={12} /></span>}
    </button>
    {directory && loaded && <div hidden={!open}><WorkspaceFileTree sessionKey={sessionKey} path={path.endsWith('/') ? path : path + '/'} selected={selected} select={select} level={level + 1} /></div>}
  </div>
}
