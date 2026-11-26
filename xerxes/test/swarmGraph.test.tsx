// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, test } from 'bun:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

import { SwarmGraph } from '../src/desktop/renderer/SwarmGraph.js'
import {
  chamferedOctagonVertices,
  computeSwarmTimeline,
  exportSwarmGraphGif,
  formatSwarmDuration,
  isPointInsideChamferedRect,
  layoutBattlefieldGraph,
  layoutHiveGraph,
  layoutMentionsGraph,
  normalizeSwarmGraphMode,
  parseSwarmTimestampMs,
  type CanvasLike,
} from '../src/desktop/renderer/swarmGraphModel.js'
import type { SwarmBoardMessage, SwarmMemberSnapshot, SwarmStatusSnapshot } from '../src/desktop/renderer/swarmState.js'

function makeMember(
  name: string,
  overrides: Partial<SwarmMemberSnapshot> = {},
): SwarmMemberSnapshot {
  return {
    name,
    role: name === 'coordinator' ? 'coordinator' : 'worker',
    ...(name === 'coordinator' ? {} : { added_by: 'coordinator' }),
    model: 'claude-sonnet-4-6',
    model_display_name: 'Claude Sonnet 4.6',
    workspace: 'share',
    channels: ['main'],
    state: 'waiting',
    killed: false,
    last_read_seq: 1,
    ...overrides,
  }
}

function makeMockCanvas(width: number, height: number): {
  readonly canvas: CanvasLike
  readonly fillRects: number
  readonly strokes: number
  readonly texts: string[]
} {
  const rgba = new Uint8ClampedArray(width * height * 4)
  let fillRectCount = 0
  let strokeCount = 0
  const texts: string[] = []
  let fillStyle = '#000000'
  let strokeStyle = '#ffffff'
  let pathPoints: Array<readonly [number, number]> = []

  const parseColor = (color: string): readonly [number, number, number] => {
    const hex = /^#([0-9a-f]{6})$/i.exec(color)
    if (hex?.[1]) {
      const n = Number.parseInt(hex[1], 16)
      return [(n >> 16) & 0xff, (n >> 8) & 0xff, n & 0xff]
    }
    const rgb = /^rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)/i.exec(color)
    if (rgb) {
      return [Number(rgb[1]), Number(rgb[2]), Number(rgb[3])]
    }
    return [24, 28, 36]
  }

  const paintDot = (cx: number, cy: number, color: string): void => {
    const [r, g, b] = parseColor(color)
    const ix = Math.round(cx)
    const iy = Math.round(cy)
    if (ix < 0 || iy < 0 || ix >= width || iy >= height) return
    const idx = (iy * width + ix) * 4
    rgba[idx] = r
    rgba[idx + 1] = g
    rgba[idx + 2] = b
    rgba[idx + 3] = 255
  }

  const ctx = {
    get fillStyle() {
      return fillStyle
    },
    set fillStyle(val: string) {
      fillStyle = String(val)
    },
    get strokeStyle() {
      return strokeStyle
    },
    set strokeStyle(val: string) {
      strokeStyle = String(val)
    },
    lineWidth: 1,
    lineCap: 'butt',
    lineJoin: 'miter',
    globalAlpha: 1,
    font: '',
    textAlign: 'left',
    textBaseline: 'alphabetic',
    save: () => undefined,
    restore: () => undefined,
    translate: () => undefined,
    scale: () => undefined,
    beginPath: () => {
      pathPoints = []
    },
    closePath: () => undefined,
    moveTo: (x: number, y: number) => {
      pathPoints.push([x, y])
    },
    lineTo: (x: number, y: number) => {
      pathPoints.push([x, y])
    },
    quadraticCurveTo: (cpx: number, cpy: number, x: number, y: number) => {
      pathPoints.push([cpx, cpy], [x, y])
    },
    arc: (x: number, y: number, r: number) => {
      pathPoints.push([x + r, y], [x - r, y], [x, y + r], [x, y - r])
    },
    rect: () => undefined,
    clip: () => undefined,
    stroke: () => {
      strokeCount += 1
      for (const [px, py] of pathPoints) paintDot(px, py, strokeStyle)
    },
    fill: () => {
      for (const [px, py] of pathPoints) paintDot(px, py, fillStyle)
    },
    setLineDash: () => undefined,
    fillRect: (x: number, y: number, w: number, h: number) => {
      fillRectCount += 1
      const [r, g, b] = parseColor(fillStyle)
      const x0 = Math.max(0, Math.floor(x))
      const y0 = Math.max(0, Math.floor(y))
      const x1 = Math.min(width, Math.ceil(x + w))
      const y1 = Math.min(height, Math.ceil(y + h))
      for (let py = y0; py < y1; py += 1) {
        for (let px = x0; px < x1; px += 1) {
          const idx = (py * width + px) * 4
          rgba[idx] = r
          rgba[idx + 1] = g
          rgba[idx + 2] = b
          rgba[idx + 3] = 255
        }
      }
    },
    fillText: (text: string) => {
      texts.push(String(text))
    },
    getImageData: () => ({
      width,
      height,
      data: new Uint8ClampedArray(rgba),
    }),
  } as unknown as CanvasRenderingContext2D

  return {
    canvas: {
      width,
      height,
      getContext: () => ctx,
    },
    get fillRects() {
      return fillRectCount
    },
    get strokes() {
      return strokeCount
    },
    texts,
  }
}

const activeSwarmStatus: SwarmStatusSnapshot = {
  active: true,
  session_id: 'sess-graph-1',
  title: 'Cosmic Simurgh Swarm Graph',
  board: 'local',
  open_board: '[Open the Swarm UI](sidecar://swarm/swarm_board/?auto_open=false)',
  head: 7,
  overview: null,
  channels: [{ name: 'main', topic: 'Shared swarm board', created_by: 'coordinator', archived: false, subscriber_count: 6 }],
  members: [
    makeMember('coordinator', { role: 'coordinator', state: 'waiting', model: 'claude-opus-4-6', model_display_name: 'Claude Opus 4.6' }),
    makeMember('brand-sprites', { state: 'working', current_activity: 'Generating Cosmic Simurgh GIFs' }),
    makeMember('swarm-graph', { state: 'working', current_activity: 'Rendering Hive honeycomb' }),
    makeMember('graph-verifier', { state: 'waiting' }),
    makeMember('narrator', { role: 'narrator', state: 'waiting' }),
    makeMember('schema-worker', { state: 'killed', killed: true }),
    makeMember('flaky-worker', { state: 'stopped', error: 'Quota exceeded on provider' }),
    makeMember('idle-retired', { state: 'killed', killed: true }),
  ],
}

const activeSwarmMessages: readonly SwarmBoardMessage[] = [
  {
    seq: 1,
    agent: 'coordinator',
    channel: 'main',
    text: '@all Kickoff: @brand-sprites owns sprites, @swarm-graph owns Hive & Mentions.',
    mentions: ['brand-sprites', 'swarm-graph'],
    channel_mentions: [],
    broadcast: true,
    time: '2026-11-22 01:00:00',
  },
  {
    seq: 2,
    agent: 'coordinator',
    channel: 'main',
    text: '@schema-worker Land the graph types first.',
    mentions: ['schema-worker'],
    channel_mentions: [],
    broadcast: false,
    time: '2026-11-22 01:05:10',
  },
  {
    seq: 3,
    agent: 'schema-worker',
    channel: 'main',
    text: '@swarm-graph [done] Landed graph model schema (#2).',
    mentions: ['swarm-graph'],
    channel_mentions: [],
    broadcast: false,
    time: '2026-11-22 01:14:08',
  },
  {
    seq: 4,
    agent: 'brand-sprites',
    channel: 'main',
    text: '@swarm-graph Exported SimurghSprite and encodeAnimatedGif contracts.',
    mentions: ['swarm-graph'],
    channel_mentions: [],
    broadcast: false,
    time: '2026-11-22 01:22:00',
  },
  {
    seq: 5,
    agent: 'swarm-graph',
    channel: 'main',
    text: '@brand-sprites Thanks! Confirmed pixel grid parity.',
    mentions: ['brand-sprites'],
    channel_mentions: [],
    broadcast: false,
    time: '2026-11-22 01:25:30',
  },
  {
    seq: 6,
    agent: 'narrator',
    channel: 'main',
    text: 'Overview updated with latest progress milestones.',
    mentions: [],
    channel_mentions: [],
    broadcast: false,
    time: '2026-11-22 01:30:00',
  },
  {
    seq: 7,
    agent: 'brand-sprites',
    channel: 'main',
    text: '@swarm-graph @graph-verifier [done] Brand pack complete.',
    mentions: ['swarm-graph', 'graph-verifier'],
    channel_mentions: [],
    broadcast: false,
    time: '2026-11-22 22:02:00',
  },
]

describe('Swarm Graph model: timestamps, replay timeline, Mentions layout & Battlefield command map', () => {
  test('parses SQL, ISO, and clock-only timestamps and formats durations accurately', () => {
    expect(formatSwarmDuration(0)).toBe('0s')
    expect(formatSwarmDuration(42_000)).toBe('42s')
    expect(formatSwarmDuration((14 * 60 + 8) * 1000)).toBe('14m 08s')
    expect(formatSwarmDuration((21 * 60 + 2) * 60 * 1000)).toBe('21h 02m')

    expect(parseSwarmTimestampMs('2026-11-22 01:00:00')).not.toBeNull()
    expect(parseSwarmTimestampMs('2026-11-22T01:00:00Z')).not.toBeNull()
    expect(parseSwarmTimestampMs('07:10:01')).toBe((7 * 3600 + 10 * 60 + 1) * 1000)
    expect(parseSwarmTimestampMs('')).toBeNull()

    // Midnight rollover for clock-only timestamps
    const rolloverTimeline = computeSwarmTimeline(null, [
      { seq: 1, agent: 'coordinator', channel: 'main', text: '@w1 start', mentions: ['w1'], channel_mentions: [], broadcast: false, time: '23:50:00' },
      { seq: 2, agent: 'w1', channel: 'main', text: '@coordinator [done]', mentions: ['coordinator'], channel_mentions: [], broadcast: false, time: '00:10:00' },
    ])
    expect(rolloverTimeline.durationLabel).toBe('20m 00s')
  })

  test('computeSwarmTimeline tracks historical replay frames, @all counts, no-mention authors, and live buckets', () => {
    const timeline = computeSwarmTimeline(activeSwarmStatus, activeSwarmMessages)
    expect(timeline.totalPosts).toBe(7)
    expect(timeline.durationLabel).toBe('21h 02m')

    // Step 0: coordinator present, no edges yet
    const f0 = timeline.frameAt(0)
    expect(f0.isLive).toBe(false)
    expect(f0.edges).toHaveLength(0)
    expect(f0.buckets.coordinator?.name).toBe('coordinator')

    // Step 3: schema-worker posted [done] -> transitioned to done bucket during replay
    const f3 = timeline.frameAt(3)
    expect(f3.activeSender).toBe('schema-worker')
    expect(f3.activeIsDone).toBe(true)
    expect(f3.buckets.done.map(m => m.name)).toContain('schema-worker')
    expect(f3.allCounts.coordinator).toBe(1)

    // Step 6: narrator posted with no mention
    const f6 = timeline.frameAt(6)
    expect(f6.postedWithoutMention.has('narrator')).toBe(true)
    expect(f6.postedWithoutMention.has('coordinator')).toBe(false)

    // Live frame (step 7): uses live status buckets (working, waiting, done, error)
    const live = timeline.frameAt(7)
    expect(live.isLive).toBe(true)
    expect(live.buckets.coordinator?.name).toBe('coordinator')
    expect(live.buckets.working.map(m => m.name).sort()).toEqual(['brand-sprites', 'swarm-graph'])
    expect(live.buckets.waiting.map(m => m.name).sort()).toEqual(['graph-verifier', 'narrator'])
    expect(live.buckets.done.map(m => m.name).sort()).toEqual(['idle-retired', 'schema-worker'])
    expect(live.buckets.error.map(m => m.name)).toEqual(['flaky-worker'])
    expect(live.doneCount).toBe(2)
    expect(live.totalWorkers).toBe(7)

    // Bidirectional edges brand-sprites <-> swarm-graph accumulated accurately
    const bToS = live.edges.find(e => e.from === 'brand-sprites' && e.to === 'swarm-graph')
    const sToB = live.edges.find(e => e.from === 'swarm-graph' && e.to === 'brand-sprites')
    expect(bToS?.count).toBe(2)
    expect(sToB?.count).toBe(1)

    // Step pacing scales cleanly with speed
    const dEven1x = timeline.stepDelayMs(4, 'even', 1)
    const dEven4x = timeline.stepDelayMs(4, 'even', 4)
    expect(dEven4x).toBeLessThan(dEven1x)
    const dReal = timeline.stepDelayMs(7, 'realtime', 1)
    expect(dReal).toBeGreaterThan(dEven1x)
  })

  test('layoutMentionsGraph places 24+ workers without node overlap, offsets bidirectional curves, and respects Hide inactive / Expand @all', () => {
    const manyMembers: SwarmMemberSnapshot[] = [
      makeMember('coordinator', { role: 'coordinator', state: 'waiting' }),
      ...Array.from({ length: 24 }, (_, idx) =>
        makeMember(`worker-service-${String(idx + 1).padStart(2, '0')}`, {
          state: idx < 4 ? 'working' : 'killed',
          killed: idx >= 4,
        }),
      ),
    ]
    const manyStatus: SwarmStatusSnapshot = {
      ...activeSwarmStatus,
      members: manyMembers,
    }
    const timeline = computeSwarmTimeline(manyStatus, activeSwarmMessages)
    const frame = timeline.frameAt(timeline.totalPosts)

    const layout = layoutMentionsGraph(frame, {
      hideInactive: false,
      expandAll: false,
      width: 720,
      height: 440,
    })

    // Coordinator sits centered at top
    const coord = layout.nodes.find(n => n.member.isCoordinator)!
    expect(coord.x).toBe(360)
    expect(coord.y).toBeLessThan(60)

    // Verify ZERO pairwise node overlaps across all 28+ nodes
    for (let i = 0; i < layout.nodes.length; i += 1) {
      const a = layout.nodes[i]!
      for (let j = i + 1; j < layout.nodes.length; j += 1) {
        const b = layout.nodes[j]!
        const overlapX = Math.abs(a.x - b.x) < (a.width + b.width) / 2 - 1
        const overlapY = Math.abs(a.y - b.y) < (a.height + b.height) / 2 - 1
        expect(overlapX && overlapY).toBe(false)
      }
    }

    // Bidirectional edges brand-sprites -> swarm-graph vs swarm-graph -> brand-sprites have separated control points
    const e1 = layout.edges.find(e => e.edge.from === 'brand-sprites' && e.edge.to === 'swarm-graph')!
    const e2 = layout.edges.find(e => e.edge.from === 'swarm-graph' && e.edge.to === 'brand-sprites')!
    expect(Math.hypot(e1.cx - e2.cx, e1.cy - e2.cy)).toBeGreaterThan(10)

    // Expand @all adds dashed broadcast edges
    const expanded = layoutMentionsGraph(frame, {
      hideInactive: false,
      expandAll: true,
      width: 720,
      height: 440,
    })
    expect(expanded.edges.filter(e => e.dashed).length).toBeGreaterThan(10)

    // Hide inactive hides zero-edge retired workers
    const compact = layoutMentionsGraph(frame, {
      hideInactive: true,
      expandAll: false,
      width: 720,
      height: 440,
    })
    expect(compact.hiddenCount).toBeGreaterThan(10)
    expect(compact.nodes.length).toBeLessThan(layout.nodes.length)
  })

  test('layoutBattlefieldGraph produces clean rounded 2-tier and multi-state dev-tool topology layouts (zero military labels or radar rings)', () => {
    expect(normalizeSwarmGraphMode('battlefield')).toBe('battlefield')
    expect(normalizeSwarmGraphMode('hive')).toBe('battlefield')
    expect(normalizeSwarmGraphMode('mentions')).toBe('mentions')
    expect(normalizeSwarmGraphMode(undefined)).toBe('battlefield')

    // 1. All-done 27-worker swarm -> clean 2-tier vertical stack (Completed over Coordinator)
    const done27Status: SwarmStatusSnapshot = {
      ...activeSwarmStatus,
      members: [
        makeMember('coordinator', { role: 'coordinator', state: 'waiting' }),
        ...Array.from({ length: 27 }, (_, idx) =>
          makeMember(`done-worker-${idx + 1}`, { state: 'killed', killed: true }),
        ),
      ],
    }
    const doneTimeline = computeSwarmTimeline(done27Status, [])
    const doneLayout = layoutBattlefieldGraph(doneTimeline.frameAt(0), { width: 680, height: 440 })

    expect(doneLayout.zones.map(z => z.id)).toEqual(['done', 'coordinator'])
    const completedZone = doneLayout.zones.find(z => z.id === 'done')!
    const coordZone = doneLayout.zones.find(z => z.id === 'coordinator')!
    expect(completedZone.label).toBe('Completed')
    expect(completedZone.cx).toBeCloseTo(coordZone.cx, 1)
    expect(completedZone.cy).toBeLessThan(coordZone.cy)
    expect(completedZone.slots).toHaveLength(6)
    expect(completedZone.overflowCount).toBe(21)
    expect(completedZone.tone).toBe('emerald')
    expect(completedZone.chamfer).toBe(0)
    expect(completedZone.radius).toBe(10)
    expect(coordZone.tone).toBe('emerald')
    expect(coordZone.sublabel).toBe('27 of 27 completed')
    expect(coordZone.statusTag).toBe('Completed')
    expect(coordZone.progressRatio).toBe(1)
    expect(coordZone.chamfer).toBe(0)
    expect(coordZone.radius).toBe(10)
    // Backwards-compatible chamfer helper still works, while layout uses zero radar rings
    expect(chamferedOctagonVertices(10, 10, 200, 100, 12)).toHaveLength(8)
    expect(doneLayout.rings).toHaveLength(0)

    // Backwards-compatible alias produces identical layout
    const aliasLayout = layoutHiveGraph(doneTimeline.frameAt(0), { width: 680, height: 440 })
    expect(aliasLayout.zones.map(z => z.id)).toEqual(['done', 'coordinator'])

    // 2. Multi-state active formation (Coordinator + Completed + Active + Waiting + Errors)
    const activeTimeline = computeSwarmTimeline(activeSwarmStatus, activeSwarmMessages)
    const activeLayout = layoutBattlefieldGraph(activeTimeline.frameAt(activeTimeline.totalPosts), {
      width: 720,
      height: 460,
    })
    expect(activeLayout.zones.map(z => z.id).sort()).toEqual([
      'coordinator',
      'done',
      'error',
      'waiting',
      'working',
    ])
    expect(activeLayout.zones.find(z => z.id === 'working')?.label).toBe('Active')
    expect(activeLayout.zones.find(z => z.id === 'waiting')?.label).toBe('Waiting')
    expect(activeLayout.zones.find(z => z.id === 'done')?.label).toBe('Completed')
    expect(activeLayout.zones.find(z => z.id === 'error')?.label).toBe('Errors')
    expect(activeLayout.dispatches.some(d => d.kind === 'waiting_on')).toBe(true)
    expect(activeLayout.dispatches.some(d => d.kind === 'tasks_done')).toBe(true)
    expect(activeLayout.dispatches.some(d => d.kind === 'message_sent')).toBe(true)

    // Verify zero pairwise zone overlap
    for (let i = 0; i < activeLayout.zones.length; i += 1) {
      const a = activeLayout.zones[i]!
      for (let j = i + 1; j < activeLayout.zones.length; j += 1) {
        const b = activeLayout.zones[j]!
        const overlapX = Math.abs(a.cx - b.cx) < (a.width + b.width) / 2 - 1
        const overlapY = Math.abs(a.cy - b.cy) < (a.height + b.height) / 2 - 1
        expect(overlapX && overlapY).toBe(false)
      }
    }
  })

  test('Map connector vectors never cross non-endpoint cards and vertical member rows never collide', () => {
    const fourZoneStatus: SwarmStatusSnapshot = {
      ...activeSwarmStatus,
      members: [
        makeMember('coordinator', { role: 'coordinator', state: 'waiting' }),
        makeMember('acp-bridge', { state: 'killed', killed: true }),
        makeMember('brand-sprites', { state: 'killed', killed: true }),
        makeMember('swarm-graph', { state: 'killed', killed: true }),
        makeMember('desktop-wire', { state: 'working' }),
        makeMember('narrator', { role: 'narrator', state: 'working' }),
        makeMember('tool-ux', { state: 'working' }),
        makeMember('graph-verifier', { state: 'waiting' }),
        makeMember('monitor', { role: 'monitor', state: 'waiting' }),
      ],
    }
    const fourZoneMessages: readonly SwarmBoardMessage[] = [
      {
        seq: 1,
        agent: 'coordinator',
        channel: 'main',
        text: '@all Kickoff: @brand-sprites, @acp-bridge, @swarm-graph, @tool-ux, @desktop-wire.',
        mentions: ['brand-sprites', 'acp-bridge', 'swarm-graph', 'tool-ux', 'desktop-wire'],
        channel_mentions: [],
        broadcast: true,
        time: '2026-11-22 02:00:00',
      },
      {
        seq: 2,
        agent: 'acp-bridge',
        channel: 'main',
        text: '@desktop-wire [done] ACP bridge ready.',
        mentions: ['desktop-wire'],
        channel_mentions: [],
        broadcast: false,
        time: '2026-11-22 02:10:00',
      },
      {
        seq: 3,
        agent: 'brand-sprites',
        channel: 'main',
        text: '@swarm-graph [done] Minimal icon pack ready.',
        mentions: ['swarm-graph'],
        channel_mentions: [],
        broadcast: false,
        time: '2026-11-22 02:15:00',
      },
      {
        seq: 4,
        agent: 'swarm-graph',
        channel: 'main',
        text: '@coordinator [done] Map & Mentions ready.',
        mentions: ['coordinator'],
        channel_mentions: [],
        broadcast: false,
        time: '2026-11-22 02:20:00',
      },
    ]

    const timeline = computeSwarmTimeline(fourZoneStatus, fourZoneMessages)
    const layout = layoutBattlefieldGraph(timeline.frameAt(timeline.totalPosts), { width: 700, height: 450 })

    // 1. Connectors stay strictly outside all non-endpoint cards
    expect(layout.dispatches.length).toBeGreaterThanOrEqual(2)
    for (const dispatch of layout.dispatches) {
      const nonEndpoints = layout.zones.filter(z => z.id !== dispatch.from && z.id !== dispatch.to)
      for (let step = 1; step <= 23; step += 1) {
        const t = step / 24
        const omt = 1 - t
        const px = omt * omt * dispatch.sx + 2 * omt * t * dispatch.cx + t * t * dispatch.ex
        const py = omt * omt * dispatch.sy + 2 * omt * t * dispatch.cy + t * t * dispatch.ey
        for (const other of nonEndpoints) {
          expect(isPointInsideChamferedRect(px, py, other, 3)).toBe(false)
        }
      }
    }

    // 2. 6-member list rows stay strictly separated vertically and inside their card bounds
    const sixPlusStatus: SwarmStatusSnapshot = {
      ...fourZoneStatus,
      members: [
        ...fourZoneStatus.members,
        makeMember('worker-extra-1', { state: 'killed', killed: true }),
        makeMember('worker-extra-2', { state: 'killed', killed: true }),
        makeMember('worker-extra-3', { state: 'killed', killed: true }),
        makeMember('worker-extra-4', { state: 'killed', killed: true }),
      ],
    }
    const sixLayout = layoutBattlefieldGraph(
      computeSwarmTimeline(sixPlusStatus, fourZoneMessages).frameAt(fourZoneMessages.length),
      { width: 700, height: 460 },
    )
    const sixCompleted = sixLayout.zones.find(z => z.id === 'done')!
    expect(sixCompleted.slots).toHaveLength(6)
    expect(sixCompleted.overflowCount).toBe(1)
    expect(new Set(sixCompleted.slots.map(s => Math.round(s.y))).size).toBe(6)

    for (const zone of [...layout.zones, ...sixLayout.zones]) {
      for (let i = 0; i < zone.slots.length; i += 1) {
        const a = zone.slots[i]!
        expect(a.spriteSize).toBe(14)
        expect(isPointInsideChamferedRect(a.x, a.y, zone, -10)).toBe(true)
        for (let j = i + 1; j < zone.slots.length; j += 1) {
          const b = zone.slots[j]!
          expect(Math.abs(a.y - b.y)).toBeGreaterThanOrEqual(24)
        }
      }
    }
  })

  test('exportSwarmGraphGif encodes genuine looping GIF89a binaries for Map (plus hive alias) and Mentions and handles missing canvas gracefully', () => {
    const timeline = computeSwarmTimeline(activeSwarmStatus, activeSwarmMessages)

    for (const mode of ['battlefield', 'mentions'] as const) {
      let downloadedBytes: Uint8Array | null = null
      let downloadedName = ''
      const mock = makeMockCanvas(320, 240)

      const result = exportSwarmGraphGif({
        timeline,
        mode,
        pacing: 'even',
        speed: 2,
        hideInactive: false,
        expandAll: mode === 'mentions',
        title: 'Cosmic Simurgh Swarm',
        width: 320,
        height: 240,
        maxFrames: 6,
        createCanvas: () => mock.canvas,
        download: (bytes, filename) => {
          downloadedBytes = bytes
          downloadedName = filename
          return true
        },
      })

      expect(result.ok).toBe(true)
      expect(result.frames).toBe(6)
      expect(result.bytes).toBeGreaterThan(800)
      expect(downloadedName).toBe(`xerxes-immortals-cosmic-simurgh-swarm-${mode}.gif`)
      if (mode === 'battlefield') {
        expect(mock.texts.some(line => line === 'XERXES IMMORTALS · MAP')).toBe(true)
      } else {
        expect(mock.texts.some(line => line.startsWith('XERXES IMMORTALS'))).toBe(true)
      }
      expect(downloadedBytes).not.toBeNull()
      // Verify GIF89a magic header, NETSCAPE2.0 loop, and 0x3B trailer
      const header = new TextDecoder('ascii').decode(downloadedBytes!.slice(0, 6))
      expect(header).toBe('GIF89a')
      expect(downloadedBytes![downloadedBytes!.length - 1]).toBe(0x3b)
      expect(mock.strokes).toBeGreaterThan(20)
      expect(mock.fillRects).toBeGreaterThanOrEqual(6)
    }

    // Legacy 'hive' mode input exports canonical '-battlefield.gif' filename
    const legacyMock = makeMockCanvas(320, 240)
    const legacyResult = exportSwarmGraphGif({
      timeline,
      mode: 'hive',
      pacing: 'even',
      speed: 2,
      hideInactive: false,
      expandAll: false,
      title: 'Imperial Campaign',
      width: 320,
      height: 240,
      maxFrames: 4,
      createCanvas: () => legacyMock.canvas,
      download: () => true,
    })
    expect(legacyResult.ok).toBe(true)
    expect(legacyResult.filename).toBe('xerxes-immortals-imperial-campaign-battlefield.gif')

    const headlessResult = exportSwarmGraphGif({
      timeline,
      mode: 'battlefield',
      pacing: 'even',
      speed: 1,
      hideInactive: false,
      expandAll: false,
      createCanvas: () => null,
    })
    expect(headlessResult.ok).toBe(false)
    expect(headlessResult.error).toContain('2D canvas unavailable')
  })
})

describe('SwarmGraph React component rendering', () => {
  test('renders Mentions mode with replay bar, toggles, Export GIF, @all badge, monochrome role icons, and Mentions legend', () => {
    const html = renderToStaticMarkup(
      createElement(SwarmGraph, {
        status: activeSwarmStatus,
        messages: activeSwarmMessages,
        initialMode: 'mentions',
        onInspectMember: () => undefined,
      }),
    )

    expect(html).toContain('aria-label="Immortals timeline replay"')
    expect(html).toContain('aria-label="Jump to start"')
    expect(html).toContain('aria-label="Play replay"')
    expect(html).toContain('aria-label="Replay timeline"')
    expect(html).toContain('7 posts')
    expect(html).toContain('over 21h 02m')
    expect(html).toContain('Export GIF')
    expect(html).toContain('>Map<')
    expect(html).not.toContain('Battlefield')
    expect(html).toContain('Mentions')
    expect(html).not.toContain('>Hive<')
    expect(html).toContain('Hide inactive')
    expect(html).toContain('Expand @all')
    expect(html).toContain('Even pacing')
    expect(html).toContain('Real time')
    expect(html).toContain('@all 1')
    expect(html).toContain('data-no-mention="true"')
    expect(html).toContain('@mention, width = count')
    expect(html).toContain('via @all')
    expect(html).toContain('posted with no mention')
    expect(html).toContain('xerxes-simurgh-sprite')

    // Design rule: zero hardcoded 6-digit hex colors anywhere
    expect(html.match(/#[0-9a-fA-F]{6}\b/g) ?? []).toEqual([])
  })

  test('renders Map mode with clean rounded rx=10 cards, sentence-case dev-tool labels, zero chamfered polygons, and zero hardcoded hex colors', () => {
    const html = renderToStaticMarkup(
      createElement(SwarmGraph, {
        status: activeSwarmStatus,
        messages: activeSwarmMessages,
        initialMode: 'battlefield',
        initialStep: 3,
      }),
    )

    expect(html).toContain('data-mode="battlefield"')
    expect(html).toContain('swarm-graph__svg--battlefield')
    expect(html).toContain('#3 / 7')
    expect(html).toContain('@schema-worker')
    expect(html).toContain('class="swarm-battlefield__zone-outer"')
    expect(html).toContain('rx="10"')
    // Zero chamfered polygons, nested icon boxes, or video-game tactical HUD chrome
    expect(html).not.toContain('<polygon')
    expect(html).not.toContain('swarm-battlefield__emblem-box')
    expect(html).not.toContain('COMMAND')
    expect(html).not.toContain('BATTLEFIELD')
    expect(html).not.toContain('VANGUARD')
    expect(html).not.toContain('VICTORY')
    expect(html).not.toContain('READY')
    expect(html).not.toContain('<animateMotion')
    expect(html).not.toContain('<animate')
    expect(html).not.toContain('xerxes-simurgh-orbit')
    expect(html).not.toContain('xerxes-astrolabe-ring')
    expect(html).not.toContain('xerxes-emblem-breath')
    expect(html).toContain('Coordinator')
    expect(html).toContain('Completed')
    expect(html).toContain('Dispatch sent')
    expect(html).toContain('Awaiting handoff')
    expect(html).not.toContain('Hide inactive')
    expect(html).not.toContain('>Hive<')

    // Legacy 'hive' prop normalizes transparently to 'battlefield'
    const legacyHtml = renderToStaticMarkup(
      createElement(SwarmGraph, {
        status: activeSwarmStatus,
        messages: activeSwarmMessages,
        initialMode: 'hive',
      }),
    )
    expect(legacyHtml).toContain('data-mode="battlefield"')
    expect(legacyHtml).toContain('Active')
    expect(legacyHtml).toContain('Waiting')
    expect(legacyHtml).toContain('Completed')
    expect(legacyHtml).toContain('Errors')
    expect(legacyHtml).not.toContain('Vanguard')
    expect(legacyHtml).not.toContain('Reserve')
    expect(legacyHtml).not.toContain('Victory')
    expect(legacyHtml).not.toContain('Contested')
    expect(legacyHtml).not.toContain('<polygon')

    // Zero hardcoded 6-digit hex colors anywhere in rendered Map SVG/HTML
    expect(legacyHtml.match(/#[0-9a-fA-F]{6}\b/g) ?? []).toEqual([])

    const emptyHtml = renderToStaticMarkup(
      createElement(SwarmGraph, {
        status: null,
        messages: [],
      }),
    )
    expect(emptyHtml).toContain('Immortals Graph ready')
    expect(emptyHtml).toContain('/immortals &lt;task&gt;')
    expect(emptyHtml).not.toContain('Battlefield')
  })

  test('declutters dense 57-member Mentions graphs with ambient edges, safe-centering CSS, and click-to-pin tracing', async () => {
    const members = [
      makeMember('coordinator', { role: 'coordinator', state: 'waiting' }),
      ...Array.from({ length: 56 }, (_, i) =>
        makeMember(`slice-${String(i + 1).padStart(2, '0')}`, { state: 'stopped' }),
      ),
    ]
    const messages: SwarmBoardMessage[] = members.slice(1, 30).map((m, idx) => ({
      seq: idx + 1,
      agent: idx % 2 === 0 ? 'coordinator' : m.name,
      channel: 'main',
      text: idx % 2 === 0 ? `@${m.name} own this slice` : `@coordinator [done] finished ${m.name}`,
      mentions: idx % 2 === 0 ? [m.name] : ['coordinator'],
      channel_mentions: [],
      broadcast: false,
      time: `2026-11-22 10:${String(idx).padStart(2, '0')}:00`,
    }))
    // Add cross-worker mentions so edge count exceeds 14 with count > 1
    messages.push(
      {
        seq: 31,
        agent: 'coordinator',
        channel: 'main',
        text: '@slice-01 follow up check',
        mentions: ['slice-01'],
        channel_mentions: [],
        broadcast: false,
        time: '2026-11-22 10:31:00',
      },
      {
        seq: 32,
        agent: 'slice-01',
        channel: 'main',
        text: '@slice-02 handoff review',
        mentions: ['slice-02'],
        channel_mentions: [],
        broadcast: false,
        time: '2026-11-22 10:32:00',
      },
    )
    const denseStatus: SwarmStatusSnapshot = {
      active: true,
      session_id: 'sess-dense-57',
      title: '57-Member Full Repo Study',
      board: 'local',
      open_board: '[Open the Swarm UI](sidecar://swarm/swarm_board/?auto_open=false)',
      head: messages.length,
      overview: null,
      channels: [{ name: 'main', topic: 'Study board', created_by: 'coordinator', archived: false, subscriber_count: 57 }],
      members,
    }

    // Unfocused dense Mentions: ambient edges carry data-dense="true", suppress noisy count badges, show trace hint
    const ambientHtml = renderToStaticMarkup(
      createElement(SwarmGraph, {
        status: denseStatus,
        messages,
        initialMode: 'mentions',
      }),
    )
    expect(ambientHtml).toContain('data-dense="true"')
    expect(ambientHtml).toContain('Hover or click a member to trace mentions')
    expect(ambientHtml).not.toContain('swarm-graph__edge-label')

    // Pinned member in dense Mentions: Tracing pill appears, connected edges light up, unrelated nodes dim
    const pinnedHtml = renderToStaticMarkup(
      createElement(SwarmGraph, {
        status: denseStatus,
        messages,
        initialMode: 'mentions',
        initialPinnedMember: 'slice-01',
      }),
    )
    expect(pinnedHtml).toContain('Tracing <strong>@slice-01</strong>')
    expect(pinnedHtml).toContain('>Clear<')
    expect(pinnedHtml).toContain('data-pinned="true"')
    expect(pinnedHtml).toContain('data-active="true"')
    expect(pinnedHtml).toContain('data-dimmed="true"')
    expect(pinnedHtml).toContain('swarm-graph__edge-label')

    // Safe vertical centering CSS prevents top-clipping @coordinator above scrollTop=0
    const atelierCss = await Bun.file(new URL('../src/desktop/renderer/atelier.css', import.meta.url)).text()
    expect(atelierCss).toContain('.swarm-graph__stage{position:relative;flex:1 1 auto;min-height:260px;overflow:auto;display:flex;align-items:flex-start;justify-content:center;padding:8px 12px}')
    expect(atelierCss).toContain('.swarm-graph__svg{display:block;width:100%;height:auto;max-width:860px;margin:auto 0;flex-shrink:0;overflow:visible;user-select:none}')
    expect(atelierCss).toContain(".swarm-graph__edge[data-dense='true']{opacity:.62}")
    expect(atelierCss).toContain('.swarm-graph__edge path{stroke:color-mix(in srgb,var(--x-secondary) 82%,transparent);stroke-linecap:round;')
  })
})
