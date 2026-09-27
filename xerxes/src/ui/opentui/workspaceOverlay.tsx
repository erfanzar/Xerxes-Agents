// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import { useKeyboard, useTerminalDimensions } from '@opentui/react'
import type { ScrollBoxRenderable } from '@opentui/core'
import { useEffect, useMemo, useRef, useState } from 'react'
import { useOptionalGateway } from '../app/gatewayContext.js'
import { patchOverlayState } from '../app/overlayStore.js'
import { listWorkspaces, inspectWorkspace, checkWorkspaceApply, applyWorkspaceReview, type WorkspaceApplyCheck, type WorkspaceRecord, type WorkspaceReview } from '../lib/workspaces.js'
import { parseUnifiedDiff } from '../lib/gitDiff.js'
import type { Theme } from '../theme.js'
import { overlayPanelSize } from './overlayLayout.js'
import { DiffRow } from './diffPanel.js'
import { Box, Text } from './primitives.js'
import { DialogHeader, DialogFooter, DialogEmpty } from './dialogChrome.js'
import { WorkspaceRecovery } from './workspaceRecovery.js'

export function WorkspaceOverlay({ t }: { t: Theme }) {
  const gateway = useOptionalGateway()
  const size = overlayPanelSize(useTerminalDimensions(), { maxWidth: 220, minWidth: 32 })
  const [rows, setRows] = useState<WorkspaceRecord[]>([])
  const [selected, setSelected] = useState('')
  const [review, setReview] = useState<WorkspaceReview | null>(null)
  const [pages, setPages] = useState<Array<string | undefined>>([])
  const [cursor, setCursor] = useState<string | undefined>()
  const [next, setNext] = useState<string | undefined>()
  const [refresh, setRefresh] = useState(0)
  const [listError, setListError] = useState('')
  const [detailError, setDetailError] = useState('')
  const [loading, setLoading] = useState(true)
  const [focusDiff, setFocusDiff] = useState(false)
  const [integration, setIntegration] = useState('')
  const [checking, setChecking] = useState(false)
  const [checked, setChecked] = useState<WorkspaceApplyCheck | null>(null)
  const [confirming, setConfirming] = useState(false)
  const [recovery, setRecovery] = useState(false)
  const applying = useRef(false)
  const checkEpoch = useRef(0)
  const scroll = useRef<ScrollBoxRenderable | null>(null)
  useEffect(() => {
    let current = true
    setLoading(true)
    setListError('')
    if (!gateway) { setListError('Workspace review requires a connected daemon.'); setLoading(false); return }
    void listWorkspaces(gateway.rpc, cursor).then(page => {
      if (!current) return
      setRows(page.records); setNext(page.next)
      setSelected(id => page.records.some(row => row.id === id) ? id : page.records[0]?.id ?? '')
    }).catch(error => { if (current) { setListError(String(error)); setRows([]); setNext(undefined); setSelected('') } })
      .finally(() => { if (current) setLoading(false) })
    return () => { current = false }
  }, [gateway, cursor, refresh])
  useEffect(() => {
    let current = true
    checkEpoch.current++; setIntegration(''); setChecking(false); setChecked(null); setConfirming(false)
    setReview(null); setDetailError(''); scroll.current?.scrollTo(0)
    if (!selected || !gateway) return
    void inspectWorkspace(gateway.rpc, selected).then(value => { if (current) setReview(value) })
      .catch(error => { if (current) setDetailError(String(error)) })
    return () => { current = false; checkEpoch.current++ }
  }, [gateway, selected, refresh])
  const checkIntegration = () => {
    if (!gateway || !review || checking || applying.current) return
    const epoch = ++checkEpoch.current
    setChecking(true); setIntegration(''); setChecked(null); setConfirming(false); scroll.current?.scrollTo(0)
    void checkWorkspaceApply(gateway.rpc, review).then(check => { if (epoch === checkEpoch.current) { setIntegration(check.message); setChecked(check) } })
      .catch(error => { if (epoch === checkEpoch.current) setIntegration(String(error)) })
      .finally(() => { if (epoch === checkEpoch.current) setChecking(false) })
  }
  const applyIntegration = () => {
    if (!gateway || !review || !checked?.destinationState || applying.current || !confirming) return
    applying.current = true
    const epoch = ++checkEpoch.current
    setConfirming(false); setChecking(true); setIntegration('Applying reviewed changes…')
    void applyWorkspaceReview(gateway.rpc, review, checked).then(message => { if (epoch === checkEpoch.current) setIntegration(message) })
      .catch(error => { if (epoch === checkEpoch.current) setIntegration(String(error)) })
      .finally(() => { applying.current = false; if (epoch === checkEpoch.current) { setChecking(false); setChecked(null) } })
  }
  useKeyboard(key => {
    if (recovery) return
    if (key.eventType === 'release') return
    if (!['escape', 'up', 'down', 'left', 'right', 'tab', 'return', 'pageup', 'pagedown', 'home', 'end', 'c', 'r', 'n', 'p', 'a', 'y', 'i'].includes(key.name)) return
    key.preventDefault(); key.stopPropagation()
    if (applying.current) return
    if (confirming) { if (key.name === 'y') applyIntegration(); else if (key.name === 'escape' || key.name === 'n') setConfirming(false); return }
    if (key.name === 'i') { setRecovery(true); return }
    if (key.name === 'a') { if (checked?.destinationState) { setConfirming(true); scroll.current?.scrollTo(0) } return }
    if (key.name === 'escape') { patchOverlayState({ workspaces: false }); return }
    if (key.name === 'tab' || key.name === 'return') { setFocusDiff(value => !value); return }
    if (key.name === 'c') { checkIntegration(); return }
    if (key.name === 'r') { setRefresh(value => value + 1); return }
    if (key.name === 'n') { if (next && !loading) { setPages(values => [...values, cursor]); setCursor(next) } return }
    if (key.name === 'p') { if (pages.length && !loading) { setCursor(pages.at(-1)); setPages(values => values.slice(0, -1)) } return }
    if (key.name === 'up' || key.name === 'down') {
      const direction = key.name === 'down' ? 1 : -1
      if (focusDiff) scroll.current?.scrollBy(direction)
      else setSelected(id => rows[Math.max(0, Math.min(rows.length - 1, rows.findIndex(row => row.id === id) + direction))]?.id ?? '')
    } else if (key.name === 'home') scroll.current?.scrollTo(0)
    else if (key.name === 'end') scroll.current?.scrollTo(Number.MAX_SAFE_INTEGER)
    else if (key.name === 'left' || key.name === 'right') scroll.current?.scrollBy({ x: key.name === 'right' ? 12 : -12, y: 0 })
    else scroll.current?.scrollBy((key.name === 'pageup' ? -1 : 1) * Math.max(1, size.height - 9))
  })
  const parsed = useMemo(() => parseUnifiedDiff(review?.diff ?? ''), [review?.diff])
  const wide = size.width >= 100
  const codeWidth = Math.max(1, size.width - (wide ? 46 : 8), ...parsed.lines.map(line => Bun.stringWidth(line.text) + 18))
  const listHeight = wide ? size.height - 7 : Math.max(2, Math.floor((size.height - 7) / 4))
  const count = Math.max(1, Math.floor(listHeight / 2))
  const index = rows.findIndex(row => row.id === selected)
  const start = Math.max(0, index - count + 1)
  if (recovery) return <box position="absolute" left={0} top={0} width="100%" height="100%" zIndex={150} backgroundColor="#000000cc" alignItems="center" justifyContent="center"><WorkspaceRecovery t={t} width={size.width} height={size.height} close={() => { setRecovery(false); setRefresh(value => value + 1) }} /></box>
  return <box position="absolute" left={0} top={0} width="100%" height="100%" zIndex={150} backgroundColor="#000000cc" alignItems="center" justifyContent="center">
    <Box width={!rows.length && !confirming ? Math.min(88, size.width) : size.width} height={!rows.length && !confirming ? Math.min(24, size.height) : size.height} paddingX={1} flexDirection="column" borderStyle="round" borderColor={t.color.border} backgroundColor={t.color.overlayBg}>
      <DialogHeader t={t} title={<>Agent workspaces · Page {pages.length + 1}</>} subtitle="Review isolated changes before bringing them into your project." />
      <Box onMouseDown={() => { if (!applying.current && !confirming) setRecovery(true) }}><Text color={t.ds.secondary}>Review changes · I recovery</Text></Box>
      {integration ? <Text color={t.color.warn} wrap="wrap">{integration}</Text> : null}
      {confirming ? <Box flexDirection="column" flexShrink={0}><Text color={t.color.warn} wrap="wrap">Apply to {checked?.destination}? Files will change. Index and agent workspace stay intact.</Text><Box onMouseDown={applyIntegration}><Text color={t.color.accent}>Y confirm · Esc cancel</Text></Box></Box> : null}
      {listError ? <Text color={t.color.warn} wrap="wrap">{listError}</Text> : null}
      {!rows.length ? <DialogEmpty t={t} title={loading ? "Loading workspaces…" : "No retained workspaces."} description="Isolated agent changes will be collected here." symbol="▱" /> : (<Box flexDirection={wide ? 'row' : 'column'} flexGrow={1} minHeight={0}>
        <Box width={wide ? 38 : '100%'} height={wide ? '100%' : listHeight} flexShrink={0} flexDirection="column">
          {rows.slice(start, start + count).map(row => <Box key={row.id} flexDirection="column" flexShrink={0} backgroundColor={row.id === selected ? t.color.selectionBg : undefined} onMouseDown={() => { if (!applying.current && !confirming) { setSelected(row.id); setFocusDiff(false) } }}>
            <Text color={row.error ? t.color.warn : t.color.text} wrap="truncate-end">{row.id === selected ? '› ' : '  '}{row.taskId}</Text>
            <Text color={t.ds.secondary} wrap="truncate-end">  {row.error ? 'Unavailable · inspect error' : row.id}</Text>
          </Box>)}
          {!rows.length ? <DialogEmpty t={t} title={loading ? 'Loading workspaces…' : 'No retained workspaces.'} description="Isolated agent changes will be collected here." symbol="▱" /> : null}
        </Box>
        <scrollbox ref={scroll} scrollX style={{ flexGrow: 1, flexShrink: 1, minHeight: 0 }} contentOptions={{ flexDirection: 'column' }}>
          {detailError ? <Text color={t.color.warn} wrap="wrap">{detailError}</Text> : null}
          {review ? <Box flexDirection="column" flexShrink={0} width={codeWidth}>
            <Text bold color={t.color.text} wrap="wrap">{review.taskId}</Text>
            {review.setup ? <Text color={t.ds.secondary} wrap="wrap">Workspace setup · {review.setup}</Text> : null}
            {review.reviewId ? <Box onMouseDown={checkIntegration}><Text color={t.color.accent}>{checking ? 'Checking integration…' : 'C · Check apply (no files changed)'}</Text></Box> : null}
            {checked?.destinationState && !checking ? <Box onMouseDown={() => setConfirming(true)}><Text color={t.color.accent}>A · Apply reviewed changes</Text></Box> : null}
            <Text color={t.ds.secondary} wrap="wrap">{review.path}</Text>
            <Text color={t.ds.secondary} wrap="wrap">Base {review.base.slice(0, 12)} · HEAD {review.head.slice(0, 12)}{review.snapshotTree ? ' · captured working files' : ''}</Text>
            <Text color={t.ds.secondary} wrap="wrap">{review.status || 'Clean checkout'}</Text>
            <Text color={t.color.accent}>{parsed.files} files · +{parsed.insertions} −{parsed.deletions}</Text>
            {parsed.truncated ? <Text color={t.color.warn} wrap="wrap">Diff display truncated; this is not the complete change.</Text> : null}
            <Box flexDirection="column" width={codeWidth}>{parsed.lines.map((line, i) => <DiffRow key={i} line={line} t={t} />)}</Box>
            {!review.diff ? <Text color={t.ds.secondary}>No changes from starting state.</Text> : null}
          </Box> : selected && !detailError ? <Text color={t.ds.secondary}>Loading review…</Text> : null}
        </scrollbox>
      </Box>)}

      <DialogFooter t={t}><Text color={t.ds.secondary}>{size.width < 70 ? '↑↓ select · Tab/PgUp/PgDn diff' : '↑↓ select · Tab diff · PgUp/PgDn scroll'}</Text>
      <Text color={t.ds.secondary}>{size.width < 70 ? 'R refresh · N/P · Esc close' : 'R refresh · N/P pages · Esc close'}</Text></DialogFooter>
    </Box>
  </box>
}
