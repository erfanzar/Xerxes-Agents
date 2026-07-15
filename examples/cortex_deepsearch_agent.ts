// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Native Cortex deep-search workflow.
 *
 * This port deliberately keeps search behind an injected port: the default
 * catalog is deterministic, while a real application can pass a policy-aware
 * search/scrape implementation without giving the example ambient network or
 * credential access.
 */

import {
  Cortex,
  type CortexAgent,
  type CortexOutput,
} from '../xerxes/src/index.js'
import { ProcessType } from '../xerxes/src/cortex/core/enums.js'
import {
  divider,
  optionValue,
  positiveIntegerOption,
  runMain,
  writeJsonWhenRequested,
} from './native_demo_support.js'

const TRACK_LIBRARY = [
  ['landscape-overview', 'Map terminology, main actors, and the shape of the topic.'],
  ['recent-developments', 'Find recent launches, upgrades, funding, and material shifts.'],
  ['primary-sources', 'Prioritize official documentation, repositories, papers, and statements.'],
  ['benchmarks-and-data', 'Collect measurable evidence, evaluations, and datasets.'],
  ['open-source-ecosystem', 'Inspect libraries, integrations, and implementation patterns.'],
  ['real-world-adoption', 'Look for case studies and production usage reports.'],
  ['risks-and-limitations', 'Surface tradeoffs, failure modes, and contested claims.'],
  ['future-direction', 'Identify roadmaps, open questions, and likely next steps.'],
] as const

export interface ResearchTrack {
  readonly focus: string
  readonly name: string
}

export interface DeepSearchPort {
  analyze(topic: string, plan: string, evidence: readonly string[]): Promise<string> | string
  plan(topic: string, tracks: readonly ResearchTrack[]): Promise<string> | string
  research(topic: string, track: ResearchTrack): Promise<string> | string
  write(topic: string, plan: string, analysis: string): Promise<string> | string
}

export interface DeepSearchRun {
  readonly analysis: string
  readonly output: CortexOutput
  readonly plan: string
  readonly report: string
  readonly tracks: readonly ResearchTrack[]
}

/** Choose repeatable tracks, expanding with a distinct second pass when needed. */
export function buildResearchTracks(workerCount: number): ResearchTrack[] {
  if (!Number.isInteger(workerCount) || workerCount < 1) throw new Error('workerCount must be a positive integer')
  return Array.from({ length: workerCount }, (_, index) => {
    const [name, focus] = TRACK_LIBRARY[index % TRACK_LIBRARY.length] ?? TRACK_LIBRARY[0]
    const pass = Math.floor(index / TRACK_LIBRARY.length)
    return pass === 0
      ? { name, focus }
      : { name: `${name}-pass-${pass + 1}`, focus: `${focus} Use sources not used in earlier passes.` }
  })
}

/** Build the native four-stage Cortex graph without an implicit web client. */
export function createDeepSearchCortex(
  topic: string,
  tracks: readonly ResearchTrack[],
  port: DeepSearchPort,
): Cortex {
  const strategist: CortexAgent = {
    id: 'strategist',
    role: 'Search Strategist',
    execute: () => port.plan(topic, tracks),
  }
  const researchers: CortexAgent[] = tracks.map(track => ({
    id: `researcher-${track.name}`,
    role: `Researcher: ${track.name}`,
    execute: () => port.research(topic, track),
  }))
  const analyst: CortexAgent = {
    id: 'analyst',
    role: 'Research Analyst',
    execute: context => port.analyze(
      topic,
      context.dependencyOutputs.get('plan')?.output ?? '',
      tracks.map(track => context.dependencyOutputs.get(`research-${track.name}`)?.output ?? ''),
    ),
  }
  const writer: CortexAgent = {
    id: 'writer',
    role: 'Technical Report Writer',
    execute: context => port.write(
      topic,
      context.dependencyOutputs.get('plan')?.output ?? '',
      context.dependencyOutputs.get('analysis')?.output ?? '',
    ),
  }
  return new Cortex({
    process: ProcessType.PARALLEL,
    maxParallel: tracks.length,
    agents: [strategist, ...researchers, analyst, writer],
    tasks: [
      {
        id: 'plan',
        agentId: strategist.id,
        description: `Create a research plan for ${topic}.`,
        expectedOutput: 'A bounded research plan.',
      },
      ...tracks.map(track => ({
        id: `research-${track.name}`,
        agentId: `researcher-${track.name}`,
        dependencies: ['plan'],
        description: `Collect evidence for ${track.name}: ${track.focus}`,
        expectedOutput: `Evidence for ${track.name}.`,
      })),
      {
        id: 'analysis',
        agentId: analyst.id,
        dependencies: tracks.map(track => `research-${track.name}`),
        contextTaskIds: ['plan'],
        description: 'Synthesize findings, uncertainties, and gaps.',
        expectedOutput: 'An evidence-bounded analysis.',
      },
      {
        id: 'report',
        agentId: writer.id,
        dependencies: ['analysis'],
        contextTaskIds: ['plan'],
        description: 'Write the final markdown report.',
        expectedOutput: 'A concise cited report.',
      },
    ],
  })
}

export async function runDeepSearch(
  topic: string,
  workerCount = 4,
  port: DeepSearchPort = demoDeepSearchPort(),
): Promise<DeepSearchRun> {
  const tracks = buildResearchTracks(workerCount)
  const output = await createDeepSearchCortex(topic, tracks, port).kickoff()
  const result = byTask(output)
  return {
    output,
    tracks,
    plan: result.get('plan') ?? '',
    analysis: result.get('analysis') ?? '',
    report: result.get('report') ?? output.rawOutput,
  }
}

/** Local evidence catalog used only by the no-network default demonstration. */
export function demoDeepSearchPort(): DeepSearchPort {
  return {
    plan: (topic, tracks) => [
      `# Deep-search plan: ${topic}`,
      '',
      ...tracks.map((track, index) => `${index + 1}. **${track.name}** — ${track.focus}`),
      '',
      'Validation: distinguish primary evidence from commentary and state unresolved questions.',
    ].join('\n'),
    research: (topic, track) => [
      `### ${track.name}`,
      `- Scope: ${track.focus}`,
      `- Demonstration evidence: a host-provided search port should collect primary sources about ${topic}.`,
      '- Confidence: illustrative only; the local catalog never claims live web verification.',
    ].join('\n'),
    analyze: (topic, _plan, evidence) => [
      `# Analysis: ${topic}`,
      '',
      `Reviewed ${evidence.filter(Boolean).length} independent tracks.`,
      'Strong conclusion: the workflow maintains a dependency barrier before synthesis.',
      'Open question: inject a real, policy-approved research port before treating any result as current evidence.',
    ].join('\n'),
    write: (topic, _plan, analysis) => [
      `# Deep-search report: ${topic}`,
      '',
      analysis,
      '',
      '## Method note',
      'This run used the deterministic local demonstration source. Replace it with an injected search/scrape port for a live investigation.',
    ].join('\n'),
  }
}

async function main(): Promise<void> {
  const args = Bun.argv.slice(2)
  const topic = optionValue(args, '--topic') ?? 'Bun-native multi-agent orchestration'
  const workers = positiveIntegerOption(args, '--researchers', 4)
  divider('CORTEX DEEP-SEARCH (native Bun demonstration)')
  console.log(`Topic: ${topic}\nParallel researchers: ${workers}\nMode: injected local research port (no network)`)
  const result = await runDeepSearch(topic, workers)
  console.log(`\n${result.report}`)
  const written = await writeJsonWhenRequested(args, 'outputs/cortex_deepsearch/report.json', result)
  if (written) console.log(`\nWrote requested artifact: ${written}`)
}

function byTask(output: CortexOutput): Map<string, string> {
  return new Map(output.taskOutputs.map(item => [item.taskId, item.output]))
}

if (import.meta.main) runMain(main)
