// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Interactive Immortals Graph (`Map` clean topology overview +
 * `Mentions` directed graph + Timeline Replay + Live `Export GIF`) for Xerxes
 * Desktop & VS Code.
 */

import {
  memo,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
  type ReactElement,
} from 'react'

import { Icon } from './Icon.js'
import { usePerfPaused } from './pageVisibility.js'
import {
  computeSwarmTimeline,
  exportSwarmGraphGif,
  layoutBattlefieldGraph,
  layoutMentionsGraph,
  normalizeSwarmGraphMode,
  type BattlefieldZone,
  type ExportSwarmGraphGifResult,
  type PositionedMentionEdge,
  type PositionedMentionNode,
  type SwarmGraphMemberNode,
  type SwarmGraphModeInput,
  type SwarmGraphViewMode,
  type SwarmReplayPacing,
  type SwarmReplaySpeed,
} from './swarmGraphModel.js'
import type { SwarmBoardMessage, SwarmStatusSnapshot } from './swarmState.js'
import { XerxesLogoMark } from './BrandMark.js'
import { SimurghSprite } from './XerxesSprites.js'

export interface SwarmGraphProps {
  readonly status: SwarmStatusSnapshot | null
  readonly messages: readonly SwarmBoardMessage[]
  readonly onInspectMember?: (name: string) => void
  readonly initialMode?: SwarmGraphModeInput
  readonly initialStep?: number
  readonly initialHideInactive?: boolean
  readonly initialExpandAll?: boolean
  readonly initialPacing?: SwarmReplayPacing
  readonly initialSpeed?: SwarmReplaySpeed
  /** Optional initial pinned member name in Mentions mode for deterministic tracing. */
  readonly initialPinnedMember?: string | null
  /** Optional custom GIF exporter for deterministic testing. */
  readonly onExportGif?: () => Promise<ExportSwarmGraphGifResult> | ExportSwarmGraphGifResult
}

const SPEED_OPTIONS: readonly SwarmReplaySpeed[] = [0.5, 1, 2, 4]
const DENSE_MENTIONS_EDGE_THRESHOLD = 14

function roleCaption(role: SwarmGraphMemberNode['role']): string {
  if (role === 'coordinator') return 'Coordinator'
  if (role === 'narrator') return 'Narrator'
  if (role === 'monitor') return 'Monitor'
  return 'Worker'
}

function stateCaption(state: SwarmGraphMemberNode['state']): string {
  if (state === 'error') return 'needs attention'
  return state
}

function isActivateKey(event: KeyboardEvent<SVGElement>): boolean {
  return event.key === 'Enter' || event.key === ' '
}

export const SwarmGraph = memo(function SwarmGraph({
  status,
  messages,
  onInspectMember,
  initialMode = 'battlefield',
  initialStep,
  initialHideInactive = false,
  initialExpandAll = false,
  initialPacing = 'even',
  initialSpeed = 1,
  initialPinnedMember = null,
  onExportGif,
}: SwarmGraphProps): ReactElement {
  const perfPaused = usePerfPaused()
  const timeline = useMemo(() => computeSwarmTimeline(status, messages), [status, messages])
  const totalPosts = timeline.totalPosts

  const clampedInitialStep =
    initialStep === undefined
      ? totalPosts
      : Math.max(0, Math.min(totalPosts, Math.trunc(initialStep)))

  const [mode, setMode] = useState<SwarmGraphViewMode>(() => normalizeSwarmGraphMode(initialMode))
  const [step, setStep] = useState<number>(clampedInitialStep)
  const [pinnedToLive, setPinnedToLive] = useState<boolean>(
    initialStep === undefined || clampedInitialStep >= totalPosts,
  )
  const [playing, setPlaying] = useState<boolean>(false)
  const [hideInactive, setHideInactive] = useState<boolean>(initialHideInactive)
  const [expandAll, setExpandAll] = useState<boolean>(initialExpandAll)
  const [pacing, setPacing] = useState<SwarmReplayPacing>(initialPacing)
  const [speed, setSpeed] = useState<SwarmReplaySpeed>(initialSpeed)
  const [hoveredMember, setHoveredMember] = useState<string | null>(null)
  const [pinnedMember, setPinnedMember] = useState<string | null>(initialPinnedMember)
  const [exportStatus, setExportStatus] = useState<'idle' | 'exporting' | 'saved' | 'error'>('idle')
  const [exportNote, setExportNote] = useState<string>('')
  const [stageWidth, setStageWidth] = useState<number>(680)

  const stageRef = useRef<HTMLDivElement>(null)
  const markerPrefix = useId().replace(/[^a-zA-Z0-9_-]/g, '')

  // Follow live head automatically whenever pinned to Live.
  useEffect(() => {
    if (pinnedToLive) {
      setStep(totalPosts)
    } else if (step > totalPosts) {
      setStep(totalPosts)
    }
  }, [totalPosts, pinnedToLive, step])

  // Observe container width so SVG scales crisply across side-rail & expanded views.
  useEffect(() => {
    const node = stageRef.current
    if (!node || typeof ResizeObserver === 'undefined') return
    const update = (): void => {
      const next = Math.round(node.clientWidth)
      if (next >= 300) setStageWidth(next)
    }
    update()
    const observer = new ResizeObserver(update)
    observer.observe(node)
    return () => observer.disconnect()
  }, [])

  // Replay timer respecting Even pacing vs Real time, playback speed, and window pause.
  useEffect(() => {
    if (!playing || perfPaused) return
    if (step >= totalPosts) {
      setPlaying(false)
      setPinnedToLive(true)
      return
    }
    const delay = timeline.stepDelayMs(step + 1, pacing, speed)
    const timer = setTimeout(() => {
      setStep(prev => {
        const next = Math.min(totalPosts, prev + 1)
        if (next >= totalPosts) {
          setPlaying(false)
          setPinnedToLive(true)
        }
        return next
      })
    }, delay)
    return () => clearTimeout(timer)
  }, [playing, perfPaused, step, totalPosts, pacing, speed, timeline])

  // Reset transient "Saved GIF" toast after 3.2s.
  useEffect(() => {
    if (exportStatus !== 'saved' && exportStatus !== 'error') return
    const timer = setTimeout(() => {
      setExportStatus('idle')
      setExportNote('')
    }, 3200)
    return () => clearTimeout(timer)
  }, [exportStatus])

  const effectiveStep = pinnedToLive ? totalPosts : Math.max(0, Math.min(totalPosts, step))
  const frame = useMemo(() => timeline.frameAt(effectiveStep), [timeline, effectiveStep])
  const isLive = effectiveStep === totalPosts && (pinnedToLive || !playing)

  const mentionsLayout = useMemo(
    () =>
      mode === 'mentions'
        ? layoutMentionsGraph(frame, {
            hideInactive,
            expandAll,
            width: Math.max(460, stageWidth),
            height: 410,
          })
        : null,
    [mode, frame, hideInactive, expandAll, stageWidth],
  )

  const battlefieldLayout = useMemo(
    () =>
      mode === 'battlefield'
        ? layoutBattlefieldGraph(frame, {
            width: Math.max(660, stageWidth),
            height: 510,
          })
        : null,
    [mode, frame, stageWidth],
  )

  const effectivePinnedMember = useMemo(() => {
    if (!pinnedMember || !mentionsLayout) return null
    return mentionsLayout.nodes.some(n => n.member.name === pinnedMember) ? pinnedMember : null
  }, [pinnedMember, mentionsLayout])

  const activeFocusMember = hoveredMember ?? effectivePinnedMember
  const isDenseMentions =
    mode === 'mentions' &&
    mentionsLayout !== null &&
    mentionsLayout.edges.length > DENSE_MENTIONS_EDGE_THRESHOLD

  const hoveredNode = useMemo(
    () => (hoveredMember ? frame.members.find(m => m.name === hoveredMember) ?? null : null),
    [hoveredMember, frame.members],
  )

  const connectedToFocus = useMemo(() => {
    if (!activeFocusMember || !mentionsLayout) return null
    const related = new Set<string>([activeFocusMember])
    for (const pe of mentionsLayout.edges) {
      if (pe.edge.from === activeFocusMember || pe.edge.to === activeFocusMember) {
        related.add(pe.edge.from)
        related.add(pe.edge.to)
      }
    }
    return related
  }, [activeFocusMember, mentionsLayout])

  const focusedEdgeCount = useMemo(() => {
    if (!activeFocusMember || !mentionsLayout) return 0
    let count = 0
    for (const pe of mentionsLayout.edges) {
      if (pe.edge.from === activeFocusMember || pe.edge.to === activeFocusMember) {
        count += 1
      }
    }
    return count
  }, [activeFocusMember, mentionsLayout])

  const orderedMentionEdges = useMemo(() => {
    if (!mentionsLayout) return []
    const ambient: PositionedMentionEdge[] = []
    const emphasized: PositionedMentionEdge[] = []
    for (const pe of mentionsLayout.edges) {
      const involvesFocus =
        activeFocusMember !== null &&
        (pe.edge.from === activeFocusMember || pe.edge.to === activeFocusMember)
      const isHighlighted = involvesFocus || (!isLive && pe.edge.isActive)
      if (isHighlighted) emphasized.push(pe)
      else ambient.push(pe)
    }
    return [...ambient, ...emphasized]
  }, [mentionsLayout, activeFocusMember, isLive])

  const jumpToStep = (next: number): void => {
    const clamped = Math.max(0, Math.min(totalPosts, Math.trunc(next)))
    setPlaying(false)
    setStep(clamped)
    setPinnedToLive(clamped >= totalPosts)
  }

  const togglePlay = (): void => {
    if (totalPosts === 0) return
    if (playing) {
      setPlaying(false)
      return
    }
    if (effectiveStep >= totalPosts) {
      setStep(0)
      setPinnedToLive(false)
    } else {
      setPinnedToLive(false)
    }
    setPlaying(true)
  }

  const jumpToLive = (): void => {
    setPlaying(false)
    setStep(totalPosts)
    setPinnedToLive(true)
  }

  const handleExportGif = async (): Promise<void> => {
    if (totalPosts === 0 || exportStatus === 'exporting') return
    setExportStatus('exporting')
    setExportNote('')
    try {
      const result = onExportGif
        ? await onExportGif()
        : exportSwarmGraphGif({
            timeline,
            mode,
            pacing,
            speed,
            hideInactive,
            expandAll,
            title: status?.title || 'XERXES IMMORTALS',
          })
      if (result.ok) {
        setExportStatus('saved')
        setExportNote(`Saved ${result.filename}`)
      } else {
        setExportStatus('error')
        setExportNote(result.error || 'GIF export failed')
      }
    } catch (failure) {
      setExportStatus('error')
      setExportNote(failure instanceof Error ? failure.message : 'GIF export failed')
    }
  }

  const progressPct = totalPosts > 0 ? Math.round((effectiveStep / totalPosts) * 1000) / 10 : 100
  const hasContent = frame.members.length > 0 || totalPosts > 0

  const renderMentionEdge = (pe: PositionedMentionEdge): ReactElement => {
    const involvesFocus =
      activeFocusMember !== null &&
      (pe.edge.from === activeFocusMember || pe.edge.to === activeFocusMember)
    const dimmed = activeFocusMember !== null && !involvesFocus
    const isHighlighted = involvesFocus || (!isLive && pe.edge.isActive)
    const isAmbientDense = isDenseMentions && activeFocusMember === null && !isHighlighted
    const markerId = isHighlighted
      ? `${markerPrefix}-arrow-active`
      : pe.dashed
        ? `${markerPrefix}-arrow-broadcast`
        : `${markerPrefix}-arrow-mention`

    const showEdgeCountBadge =
      !isAmbientDense &&
      !pe.dashed &&
      (pe.edge.count > 1 || (involvesFocus && focusedEdgeCount <= 12))
    const highFanoutFocus = involvesFocus && focusedEdgeCount > 16

    return (
      <g
        key={pe.edge.id}
        className="swarm-graph__edge"
        data-kind={pe.edge.kind}
        data-active={isHighlighted ? 'true' : undefined}
        data-dense={isAmbientDense ? 'true' : undefined}
        data-dimmed={dimmed ? 'true' : undefined}
      >
        <path
          d={pe.path}
          fill="none"
          strokeWidth={
            isHighlighted
              ? highFanoutFocus
                ? 1.65
                : Math.max(2.1, pe.strokeWidth)
              : isAmbientDense
                ? 1.3
                : Math.max(1.35, pe.strokeWidth)
          }
          strokeDasharray={pe.dashed ? '4 3' : undefined}
          markerEnd={`url(#${markerId})`}
        />
        {showEdgeCountBadge && (
          <g className="swarm-graph__edge-label" transform={`translate(${pe.labelX}, ${pe.labelY})`}>
            <rect x={-9} y={-7} width={18} height={14} rx={4} />
            <text x={0} y={0} textAnchor="middle" dominantBaseline="central">
              {pe.edge.count}
            </text>
          </g>
        )}
      </g>
    )
  }

  const renderMentionNode = (node: PositionedMentionNode): ReactElement => {
    const { member } = node
    const dimmed = connectedToFocus !== null && !connectedToFocus.has(member.name)
    const isPinned = effectivePinnedMember === member.name
    const x = node.x - node.width / 2
    const y = node.y - node.height / 2
    const spriteSize = 14
    const spriteX = x + 10
    const spriteY = node.y - spriteSize / 2
    const dotCx = spriteX + spriteSize + 2.5
    const dotCy = spriteY + 2
    const labelX = spriteX + spriteSize + 9
    const pillRx = Math.max(node.rx, Math.round(node.height / 2))
    const togglePin = (): void => {
      setPinnedMember(cur => (cur === member.name ? null : member.name))
    }

    const badgeText = member.allCount > 0 ? `@all ${member.allCount}` : ''
    const badgeW = badgeText ? Math.max(38, badgeText.length * 6.3 + 12) : 0

    return (
      <g
        key={member.name}
        className="swarm-graph__node"
        role="button"
        tabIndex={0}
        aria-pressed={isPinned}
        aria-label={`${member.name} (${roleCaption(member.role)}, ${stateCaption(member.state)}, ${member.postCount} posts)`}
        data-coordinator={member.isCoordinator ? 'true' : undefined}
        data-state={member.state}
        data-no-mention={member.postedWithoutMention ? 'true' : undefined}
        data-active-sender={!isLive && member.isActiveSender ? 'true' : undefined}
        data-hovered={hoveredMember === member.name ? 'true' : undefined}
        data-pinned={isPinned ? 'true' : undefined}
        data-dimmed={dimmed ? 'true' : undefined}
        onMouseEnter={() => setHoveredMember(member.name)}
        onMouseLeave={() => setHoveredMember(cur => (cur === member.name ? null : cur))}
        onFocus={() => setHoveredMember(member.name)}
        onBlur={() => setHoveredMember(cur => (cur === member.name ? null : cur))}
        onClick={event => {
          event.stopPropagation()
          togglePin()
        }}
        onKeyDown={event => {
          if (isActivateKey(event)) {
            event.preventDefault()
            event.stopPropagation()
            togglePin()
          }
        }}
      >
        <title>
          {`@${member.name} · ${roleCaption(member.role)} · ${stateCaption(member.state)} · ${member.postCount} posts`}
        </title>
        <rect
          className="swarm-graph__pill"
          x={x}
          y={y}
          width={node.width}
          height={node.height}
          rx={pillRx}
          strokeDasharray={member.postedWithoutMention ? '3 3' : undefined}
        />
        <g transform={`translate(${spriteX}, ${spriteY})`}>
          <SimurghSprite
            role={member.spriteRole}
            state={member.spriteState}
            size={spriteSize}
            animate={false}
          />
        </g>
        <circle
          className="swarm-graph__node-dot"
          data-state={member.state}
          cx={dotCx}
          cy={dotCy}
          r={2.5}
        />
        <text
          className="swarm-graph__node-label"
          x={labelX}
          y={node.y}
          dominantBaseline="central"
        >
          {node.displayLabel}
        </text>
        {badgeText && (
          <g
            className="swarm-graph__all-badge"
            transform={`translate(${x + node.width - badgeW + 6}, ${y - 10})`}
          >
            <rect x={0} y={0} width={badgeW} height={15} rx={7.5} />
            <text x={badgeW / 2} y={7.5} textAnchor="middle" dominantBaseline="central">
              {badgeText}
            </text>
          </g>
        )}
      </g>
    )
  }

  const renderBattlefieldZone = (zone: BattlefieldZone): ReactElement => {
    const isCoord = zone.id === 'coordinator'
    const coordSlot = isCoord ? zone.slots[0] : undefined
    const clickableCoord = isCoord && coordSlot && Boolean(onInspectMember)

    if (isCoord) {
      const progress = Math.max(0, Math.min(1, zone.progressRatio ?? 0))
      const active = Math.max(0, Math.min(1 - progress, zone.activeRatio ?? 0))
      const trackX = zone.x + 16
      const trackY = zone.y + 68
      const trackW = zone.width - 32
      const doneW = progress > 0 ? Math.max(4, Math.round(trackW * progress)) : 0
      const activeW = active > 0 ? Math.max(3, Math.round(trackW * active)) : 0
      const pctLabel = `${Math.round(progress * 100)}%`
      const textX = zone.x + 38
      const rightEdge = zone.x + zone.width - 16

      return (
        <g
          key={zone.id}
          className="swarm-battlefield__zone swarm-battlefield__zone--command"
          data-zone={zone.id}
          data-tone={zone.tone}
          data-hovered={coordSlot && hoveredMember === coordSlot.member.name ? 'true' : undefined}
        >
          <rect
            className="swarm-battlefield__zone-outer"
            x={zone.x}
            y={zone.y}
            width={zone.width}
            height={zone.height}
            rx={zone.radius}
          />

          <g
            className="swarm-battlefield__coord"
            role={clickableCoord ? 'button' : undefined}
            tabIndex={clickableCoord ? 0 : undefined}
            aria-label={`Coordinator (${zone.sublabel})`}
            onMouseEnter={() => coordSlot && setHoveredMember(coordSlot.member.name)}
            onMouseLeave={() => setHoveredMember(null)}
            onFocus={() => coordSlot && setHoveredMember(coordSlot.member.name)}
            onBlur={() => setHoveredMember(null)}
            onClick={() => coordSlot && onInspectMember?.(coordSlot.member.name)}
            onKeyDown={event => {
              if (clickableCoord && coordSlot && isActivateKey(event)) {
                event.preventDefault()
                onInspectMember?.(coordSlot.member.name)
              }
            }}
          >
            <title>
              {coordSlot
                ? `@${coordSlot.member.name} · Coordinator · ${zone.sublabel}`
                : `Coordinator · ${zone.sublabel}`}
            </title>

            {coordSlot && (
              <g transform={`translate(${zone.x + 16}, ${zone.y + 17})`}>
                <SimurghSprite
                  role="coordinator"
                  state={coordSlot.member.spriteState}
                  size={14}
                  animate={false}
                />
              </g>
            )}

            <text
              className="swarm-battlefield__coord-title"
              x={textX}
              y={zone.y + 24}
              dominantBaseline="central"
            >
              {zone.label}
            </text>

            <text
              className="swarm-battlefield__status-tag"
              x={rightEdge - 12}
              y={zone.y + 24}
              textAnchor="end"
              dominantBaseline="central"
            >
              {zone.statusTag}
            </text>
            <circle
              className="swarm-battlefield__status-dot"
              data-tone={zone.tone}
              cx={rightEdge - 3}
              cy={zone.y + 24}
              r={3}
            />

            <text
              className="swarm-battlefield__coord-sub"
              x={textX}
              y={zone.y + 46}
              dominantBaseline="central"
            >
              {zone.sublabel}
            </text>
            {frame.totalWorkers > 0 && (
              <>
                <text
                  className="swarm-battlefield__coord-pct"
                  x={rightEdge}
                  y={zone.y + 46}
                  textAnchor="end"
                  dominantBaseline="central"
                >
                  {pctLabel}
                </text>

                <rect
                  className="swarm-battlefield__progress-track"
                  x={trackX}
                  y={trackY}
                  width={trackW}
                  height={3.5}
                  rx={1.75}
                />
                {doneW > 0 && (
                  <rect
                    className="swarm-battlefield__progress-done"
                    x={trackX}
                    y={trackY}
                    width={doneW}
                    height={3.5}
                    rx={1.75}
                  />
                )}
                {activeW > 0 && (
                  <rect
                    className="swarm-battlefield__progress-active"
                    x={trackX + doneW}
                    y={trackY}
                    width={Math.min(activeW, Math.max(0, trackW - doneW))}
                    height={3.5}
                    rx={doneW > 0 ? 0 : 1.75}
                  />
                )}
              </>
            )}
          </g>
        </g>
      )
    }

    const overflowLabel = zone.overflowCount > 0 ? `+${zone.overflowCount} more` : ''

    return (
      <g
        key={zone.id}
        className="swarm-battlefield__zone"
        data-zone={zone.id}
        data-tone={zone.tone}
      >
        <rect
          className="swarm-battlefield__zone-outer"
          x={zone.x}
          y={zone.y}
          width={zone.width}
          height={zone.height}
          rx={zone.radius}
        />

        <circle
          className="swarm-battlefield__status-dot"
          data-tone={zone.tone}
          cx={zone.x + 16}
          cy={zone.y + 17}
          r={3}
        />
        <text
          className="swarm-battlefield__zone-title"
          x={zone.x + 26}
          y={zone.y + 17}
          dominantBaseline="central"
        >
          {zone.label}
        </text>
        <text
          className="swarm-battlefield__zone-count"
          x={zone.x + zone.width - 14}
          y={zone.y + 17}
          textAnchor="end"
          dominantBaseline="central"
        >
          {zone.statusTag}
        </text>
        <line
          className="swarm-battlefield__zone-divider"
          x1={zone.x}
          y1={zone.y + 34}
          x2={zone.x + zone.width}
          y2={zone.y + 34}
        />

        {/* Clean member rows */}
        {zone.slots.map(slot => {
          const clickable = Boolean(onInspectMember)
          return (
            <g
              key={slot.member.name}
              className="swarm-battlefield__member"
              role={clickable ? 'button' : undefined}
              tabIndex={clickable ? 0 : undefined}
              aria-label={`@${slot.member.name} (${roleCaption(slot.member.role)}, ${stateCaption(slot.member.state)})`}
              data-state={slot.member.state}
              data-hovered={hoveredMember === slot.member.name ? 'true' : undefined}
              onMouseEnter={() => setHoveredMember(slot.member.name)}
              onMouseLeave={() => setHoveredMember(cur => (cur === slot.member.name ? null : cur))}
              onFocus={() => setHoveredMember(slot.member.name)}
              onBlur={() => setHoveredMember(cur => (cur === slot.member.name ? null : cur))}
              onClick={() => onInspectMember?.(slot.member.name)}
              onKeyDown={event => {
                if (clickable && isActivateKey(event)) {
                  event.preventDefault()
                  onInspectMember?.(slot.member.name)
                }
              }}
            >
              <title>
                {`@${slot.member.name} · ${roleCaption(slot.member.role)} · ${stateCaption(slot.member.state)}${slot.member.currentActivity ? ` · ${slot.member.currentActivity}` : ''}`}
              </title>
              <rect
                className="swarm-battlefield__slot-plate"
                x={zone.x + 6}
                y={slot.y - 12}
                width={zone.width - 12}
                height={24}
                rx={6}
              />
              <g transform={`translate(${zone.x + 14}, ${slot.y - slot.spriteSize / 2})`}>
                <SimurghSprite
                  role={slot.member.spriteRole}
                  state={slot.member.spriteState}
                  size={slot.spriteSize}
                  animate={false}
                />
              </g>
              <text
                className="swarm-battlefield__member-label"
                x={zone.x + 36}
                y={slot.y}
                dominantBaseline="central"
              >
                {slot.displayLabel}
              </text>
              {slot.caption && (
                <text
                  className="swarm-battlefield__member-caption"
                  x={zone.x + zone.width - 26}
                  y={slot.y}
                  textAnchor="end"
                  dominantBaseline="central"
                >
                  {slot.caption}
                </text>
              )}
              <circle
                className="swarm-battlefield__member-dot"
                data-state={slot.member.state}
                cx={zone.x + zone.width - 16}
                cy={slot.y}
                r={3}
              />
            </g>
          )
        })}

        {zone.overflowCount > 0 && (
          <text
            className="swarm-battlefield__overflow-text"
            x={zone.x + 14}
            y={zone.y + zone.height - 14}
            dominantBaseline="central"
          >
            {overflowLabel}
          </text>
        )}
      </g>
    )
  }

  return (
    <div
      className="swarm-graph"
      data-mode={mode}
      onKeyDown={event => {
        if (event.key === 'Escape' && effectivePinnedMember !== null) {
          event.preventDefault()
          event.stopPropagation()
          setPinnedMember(null)
        }
      }}
    >
      {/* ── Replay Transport Bar ── */}
      <div className="swarm-graph__replay" role="region" aria-label="Immortals timeline replay">
        <div className="swarm-graph__transport">
          <button
            type="button"
            className="swarm-graph__tbtn"
            aria-label="Jump to start"
            title="Jump to start"
            disabled={totalPosts === 0 || effectiveStep === 0}
            onClick={() => jumpToStep(0)}
          >
            <span aria-hidden="true">|‹</span>
          </button>
          <button
            type="button"
            className="swarm-graph__tbtn"
            aria-label="Previous step"
            title="Previous post"
            disabled={totalPosts === 0 || effectiveStep === 0}
            onClick={() => jumpToStep(effectiveStep - 1)}
          >
            <span aria-hidden="true">‹</span>
          </button>
          <button
            type="button"
            className={`swarm-graph__tbtn swarm-graph__tbtn--play${playing ? ' is-playing' : ''}`}
            aria-label={playing ? 'Pause replay' : 'Play replay'}
            title={playing ? 'Pause replay' : 'Play replay'}
            disabled={totalPosts === 0}
            onClick={togglePlay}
          >
            <Icon name={playing ? 'pause' : 'play'} size={12} />
          </button>
          <button
            type="button"
            className="swarm-graph__tbtn"
            aria-label="Next step"
            title="Next post"
            disabled={totalPosts === 0 || effectiveStep >= totalPosts}
            onClick={() => jumpToStep(effectiveStep + 1)}
          >
            <span aria-hidden="true">›</span>
          </button>
        </div>

        <input
          type="range"
          className="swarm-graph__slider"
          aria-label="Replay timeline"
          aria-valuetext={totalPosts === 0 ? 'No posts' : isLive ? 'Live' : `Post ${effectiveStep} of ${totalPosts}`}
          min={0}
          max={Math.max(1, totalPosts)}
          step={1}
          value={totalPosts === 0 ? 0 : effectiveStep}
          disabled={totalPosts === 0}
          style={{ ['--swarm-progress' as string]: `${progressPct}%` }}
          onChange={event => jumpToStep(Number(event.target.value))}
        />

        <button
          type="button"
          className="swarm-graph__live"
          data-live={isLive ? 'true' : undefined}
          aria-pressed={isLive}
          title={isLive ? 'Following live Immortals updates' : 'Jump back to live head'}
          onClick={jumpToLive}
        >
          <span className="swarm-graph__live-dot" aria-hidden="true" />
          <span>Live</span>
        </button>
      </div>

      {/* ── Controls Row ── */}
      <div className="swarm-graph__controls">
        <div className="swarm-graph__summary-col">
          <div className="swarm-graph__summary">
            {isLive ? (
              <>
                <strong>
                  {totalPosts} {totalPosts === 1 ? 'post' : 'posts'}
                </strong>
                {totalPosts > 0 && (
                  <span className="swarm-graph__duration"> over {timeline.durationLabel}</span>
                )}
              </>
            ) : (
              <>
                <strong>
                  #{frame.step} / {totalPosts}
                </strong>
                <span className="swarm-graph__duration"> · {frame.elapsedLabel}</span>
                {frame.currentMessage && (
                  <span className="swarm-graph__step-author"> · @{frame.currentMessage.agent}</span>
                )}
              </>
            )}
          </div>

          <button
            type="button"
            className="swarm-graph__export"
            aria-label="Export Immortals graph as animated GIF"
            title="Export GIF"
            disabled={totalPosts === 0 || exportStatus === 'exporting'}
            onClick={() => void handleExportGif()}
          >
            <Icon name="download" size={12} />
            <span>
              {exportStatus === 'exporting'
                ? 'Exporting…'
                : exportStatus === 'saved'
                  ? 'Saved GIF'
                  : exportStatus === 'error'
                    ? 'Export failed'
                    : 'Export GIF'}
            </span>
          </button>
          {exportNote && (
            <span className="swarm-graph__export-note" role={exportStatus === 'error' ? 'alert' : 'status'}>
              {exportNote}
            </span>
          )}
        </div>

        <div className="swarm-graph__options">
          <div className="swarm-graph__mode" role="radiogroup" aria-label="Graph view mode">
            <button
              type="button"
              role="radio"
              className={`swarm-graph__mode-btn${mode === 'battlefield' ? ' is-selected' : ''}`}
              aria-checked={mode === 'battlefield'}
              onClick={() => setMode('battlefield')}
            >
              Map
            </button>
            <button
              type="button"
              role="radio"
              className={`swarm-graph__mode-btn${mode === 'mentions' ? ' is-selected' : ''}`}
              aria-checked={mode === 'mentions'}
              onClick={() => setMode('mentions')}
            >
              Mentions
            </button>
          </div>

          {mode === 'mentions' && (
            <>
              <button
                type="button"
                role="switch"
                aria-checked={hideInactive}
                className={`swarm-graph__toggle${hideInactive ? ' is-on' : ''}`}
                onClick={() => setHideInactive(prev => !prev)}
              >
                <span className="swarm-graph__switch" aria-hidden="true">
                  <span className="swarm-graph__switch-thumb" />
                </span>
                <span>Hide inactive</span>
              </button>
              <button
                type="button"
                role="switch"
                aria-checked={expandAll}
                className={`swarm-graph__toggle${expandAll ? ' is-on' : ''}`}
                onClick={() => setExpandAll(prev => !prev)}
              >
                <span className="swarm-graph__switch" aria-hidden="true">
                  <span className="swarm-graph__switch-thumb" />
                </span>
                <span>Expand @all</span>
              </button>
            </>
          )}

          <select
            className="swarm-graph__select"
            aria-label="Replay pacing"
            value={pacing}
            onChange={event => setPacing(event.target.value as SwarmReplayPacing)}
          >
            <option value="even">Even pacing</option>
            <option value="realtime">Real time</option>
          </select>

          <select
            className="swarm-graph__select swarm-graph__select--speed"
            aria-label="Replay speed"
            value={String(speed)}
            onChange={event => {
              const parsed = Number(event.target.value) as SwarmReplaySpeed
              if (SPEED_OPTIONS.includes(parsed)) setSpeed(parsed)
            }}
          >
            {SPEED_OPTIONS.map(opt => (
              <option key={opt} value={String(opt)}>{opt}×</option>
            ))}
          </select>
        </div>
      </div>

      {/* ── Dense Mentions Trace Bar (pin/clear focus cleanly without hiding top rows) ── */}
      {mode === 'mentions' && (effectivePinnedMember !== null || isDenseMentions) && (
        <div className="swarm-graph__tracebar" role="status">
          {effectivePinnedMember !== null ? (
            <span className="swarm-graph__trace-pill" data-pinned="true">
              <span>
                Tracing <strong>@{effectivePinnedMember}</strong>
              </span>
              {onInspectMember && (
                <>
                  <span className="swarm-graph__trace-sep" aria-hidden="true">·</span>
                  <button
                    type="button"
                    className="swarm-graph__trace-action"
                    onClick={() => onInspectMember(effectivePinnedMember)}
                  >
                    Inspect
                  </button>
                </>
              )}
              <span className="swarm-graph__trace-sep" aria-hidden="true">·</span>
              <button
                type="button"
                className="swarm-graph__trace-clear"
                onClick={() => setPinnedMember(null)}
              >
                Clear
              </button>
            </span>
          ) : (
            <span className="swarm-graph__trace-hint">
              Hover or click a member to trace mentions
            </span>
          )}
        </div>
      )}

      {/* ── Interactive Graph Stage ── */}
      {(() => {
        const svgMinWidth = mode === 'battlefield' && battlefieldLayout
          ? Math.min(battlefieldLayout.width, 560)
          : mentionsLayout
            ? Math.min(mentionsLayout.width, 440)
            : 0
        const needsHorizontalScroll = svgMinWidth > 0 && stageWidth < svgMinWidth
        return (
      <div
        className="swarm-graph__stage"
        ref={stageRef}
        style={needsHorizontalScroll ? { justifyContent: 'flex-start' } : undefined}
        onClick={() => {
          if (effectivePinnedMember !== null) setPinnedMember(null)
        }}
      >
        {!hasContent ? (
          <div className="swarm-graph__empty">
            <XerxesLogoMark size={48} className="swarm-graph__logo" decorative />
            <p className="swarm-graph__empty-title">Immortals Graph ready</p>
            <p className="swarm-graph__empty-sub">
              Start an Immortals run with <code>/immortals &lt;task&gt;</code> to see members, progress and mentions here.
            </p>
          </div>
        ) : mode === 'battlefield' && battlefieldLayout ? (
          <svg
            className="swarm-graph__svg swarm-graph__svg--battlefield"
            viewBox={`0 0 ${battlefieldLayout.width} ${battlefieldLayout.height}`}
            style={{ minWidth: `${svgMinWidth}px`, margin: 'auto' }}
            role="img"
            aria-label={`Immortals map showing ${frame.doneCount} of ${frame.totalWorkers} workers completed`}
          >
            <defs>
              <marker
                id={`${markerPrefix}-bf-done`}
                viewBox="0 0 10 10"
                refX="8.2"
                refY="5"
                markerWidth="6.2"
                markerHeight="6.2"
                orient="auto-start-reverse"
              >
                <path d="M 1 1.7 L 9 5 L 1 8.3 L 3 5 z" className="swarm-graph__marker--done" />
              </marker>
              <marker
                id={`${markerPrefix}-bf-msg`}
                viewBox="0 0 10 10"
                refX="8.2"
                refY="5"
                markerWidth="6.2"
                markerHeight="6.2"
                orient="auto-start-reverse"
              >
                <path d="M 1 1.7 L 9 5 L 1 8.3 L 3 5 z" className="swarm-graph__marker--msg" />
              </marker>
              <marker
                id={`${markerPrefix}-bf-wait`}
                viewBox="0 0 10 10"
                refX="8.2"
                refY="5"
                markerWidth="6.2"
                markerHeight="6.2"
                orient="auto-start-reverse"
              >
                <path d="M 1 1.7 L 9 5 L 1 8.3 L 3 5 z" className="swarm-graph__marker--wait" />
              </marker>
            </defs>

            {/* ── Clean Sector Cards & Coordinator ── */}
            <g className="swarm-battlefield__zones">
              {battlefieldLayout.zones.map(renderBattlefieldZone)}
            </g>

            {/* ── Handoff Connectors ── */}
            <g className="swarm-battlefield__dispatches">
              {battlefieldLayout.dispatches.map(dispatch => {
                const markerId =
                  dispatch.kind === 'tasks_done'
                    ? `${markerPrefix}-bf-done`
                    : dispatch.kind === 'waiting_on'
                      ? `${markerPrefix}-bf-wait`
                      : `${markerPrefix}-bf-msg`
                return (
                  <g
                    key={dispatch.id}
                    className="swarm-battlefield__vector"
                    data-kind={dispatch.kind}
                  >
                    <path
                      className="swarm-battlefield__vector-path"
                      d={dispatch.path}
                      fill="none"
                      strokeDasharray={dispatch.dashed ? '5 4' : undefined}
                      markerEnd={`url(#${markerId})`}
                    />
                  </g>
                )
              })}
            </g>
          </svg>
        ) : mentionsLayout ? (
          <svg
            className="swarm-graph__svg swarm-graph__svg--mentions"
            viewBox={`0 0 ${mentionsLayout.width} ${mentionsLayout.height}`}
            style={{ minWidth: `${svgMinWidth}px`, margin: 'auto' }}
            role="img"
            aria-label={`Immortals Mentions graph with ${mentionsLayout.nodes.length} members and ${mentionsLayout.edges.length} links`}
          >
            <defs>
              <marker
                id={`${markerPrefix}-arrow-mention`}
                viewBox="0 0 10 10"
                refX="8"
                refY="5"
                markerWidth="6.5"
                markerHeight="6.5"
                orient="auto-start-reverse"
              >
                <path d="M 1 2 L 9 5 L 1 8 z" className="swarm-graph__marker--mention" />
              </marker>
              <marker
                id={`${markerPrefix}-arrow-active`}
                viewBox="0 0 10 10"
                refX="8"
                refY="5"
                markerWidth="6.8"
                markerHeight="6.8"
                orient="auto-start-reverse"
              >
                <path d="M 1 2 L 9 5 L 1 8 z" className="swarm-graph__marker--active" />
              </marker>
              <marker
                id={`${markerPrefix}-arrow-broadcast`}
                viewBox="0 0 10 10"
                refX="8"
                refY="5"
                markerWidth="6"
                markerHeight="6"
                orient="auto-start-reverse"
              >
                <path d="M 1 2 L 9 5 L 1 8 z" className="swarm-graph__marker--broadcast" />
              </marker>
            </defs>

            <g className="swarm-graph__edges-layer">
              {orderedMentionEdges.map(renderMentionEdge)}
            </g>
            <g className="swarm-graph__nodes-layer">
              {mentionsLayout.nodes.map(renderMentionNode)}
            </g>
          </svg>
        ) : null}

        {hoveredNode && (
          <div className="swarm-graph__tooltip" role="tooltip">
            <div className="swarm-graph__tooltip-head">
              <SimurghSprite
                role={hoveredNode.spriteRole}
                state={hoveredNode.spriteState}
                size={16}
                animate={false}
              />
              <strong>@{hoveredNode.name}</strong>
              <span className="swarm-graph__tooltip-role">{roleCaption(hoveredNode.role)}</span>
              <span className="swarm-graph__tooltip-state" data-state={hoveredNode.state}>
                {stateCaption(hoveredNode.state)}
              </span>
            </div>
            <div className="swarm-graph__tooltip-meta">
              {hoveredNode.model && <span>{hoveredNode.model}</span>}
              <span>{hoveredNode.postCount} posts</span>
              {hoveredNode.allCount > 0 && <span>@all {hoveredNode.allCount}</span>}
              {mode === 'mentions' ? (
                <span>{effectivePinnedMember === hoveredNode.name ? 'Click to clear trace' : 'Click to trace'}</span>
              ) : (
                onInspectMember && <span>Click to inspect</span>
              )}
            </div>
            {hoveredNode.currentActivity && (
              <div className="swarm-graph__tooltip-act">{hoveredNode.currentActivity}</div>
            )}
          </div>
        )}
      </div>
        )
      })()}

      {/* ── Bottom Legend ── */}
      <footer className="swarm-graph__legend" aria-label="Graph legend">
        {mode === 'mentions' ? (
          <>
            <span className="swarm-graph__legend-item">
              <span className="swarm-graph__swatch swarm-graph__swatch--arrow" aria-hidden="true">→</span>
              <span>@mention, width = count</span>
            </span>
            <span className="swarm-graph__legend-item">
              <span className="swarm-graph__swatch swarm-graph__swatch--dashed-arrow" aria-hidden="true">⋯→</span>
              <span>via @all</span>
            </span>
            <span className="swarm-graph__legend-item">
              <span className="swarm-graph__swatch swarm-graph__swatch--dashed-pill" aria-hidden="true" />
              <span>posted with no mention</span>
            </span>
            <span className="swarm-graph__legend-item">
              <span className="swarm-graph__legend-badge" aria-hidden="true">@all 2</span>
              <span>@all posts (drawn by Expand @all)</span>
            </span>
          </>
        ) : (
          <>
            <span className="swarm-graph__legend-item">
              <span className="swarm-graph__swatch swarm-graph__swatch--done" aria-hidden="true" />
              <span>Completed</span>
            </span>
            <span className="swarm-graph__legend-item">
              <span className="swarm-graph__swatch swarm-graph__swatch--sent" aria-hidden="true" />
              <span>Dispatch sent</span>
            </span>
            <span className="swarm-graph__legend-item">
              <span className="swarm-graph__swatch swarm-graph__swatch--wait" aria-hidden="true">
                <svg width="20" height="10" viewBox="0 0 20 10" fill="none" aria-hidden="true">
                  <line x1="1" y1="5" x2="14" y2="5" stroke="var(--x-meta)" strokeWidth="1.5" strokeDasharray="3 2.5" strokeLinecap="round" />
                  <path d="M 12 2.2 L 18 5 L 12 7.8 L 13.5 5 z" fill="var(--x-meta)" />
                </svg>
              </span>
              <span>Awaiting handoff</span>
            </span>
          </>
        )}
      </footer>
    </div>
  )
})
