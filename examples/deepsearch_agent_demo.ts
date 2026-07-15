// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * A stateful, tool-oriented deep-search demonstration.
 *
 * The original Python example mixed its workflow with ambient provider and web
 * setup. This Bun version makes the search boundary explicit and ships a local
 * demonstration catalog so every analysis step is runnable offline.
 */

import {
  divider,
  optionValue,
  positiveIntegerOption,
  runMain,
  writeJsonWhenRequested,
} from './native_demo_support.js'

export interface SearchResult {
  readonly excerpt: string
  readonly source: string
  readonly title: string
  readonly url: string
}

export interface SearchPort {
  search(query: string, maxResults: number): Promise<readonly SearchResult[]> | readonly SearchResult[]
}

export interface SearchAnalysis {
  readonly commonTerms: readonly string[]
  readonly resultCount: number
  readonly sources: readonly string[]
}

export interface KnowledgeGraph {
  readonly edges: readonly { readonly from: string; readonly to: string }[]
  readonly nodes: readonly string[]
}

export class DeepSearchSession {
  private readonly results: SearchResult[] = []
  private readonly queries: string[] = []

  constructor(readonly topic: string, private readonly searchPort: SearchPort) {}

  initializeSearchSession(): string {
    this.results.length = 0
    this.queries.length = 0
    return `Initialized Bun-native deep-search session for: ${this.topic}`
  }

  generateSearchQueries(strategy = 'comprehensive', maxQueries = 8): string[] {
    if (!Number.isInteger(maxQueries) || maxQueries < 1) throw new Error('maxQueries must be a positive integer')
    const candidates = [
      this.topic,
      `${this.topic} official documentation`,
      `${this.topic} architecture`,
      `${this.topic} benchmarks`,
      `${this.topic} production case studies`,
      `${this.topic} risks limitations`,
      `${this.topic} open source`,
      `${this.topic} recent developments`,
      `${this.topic} ${strategy} analysis`,
    ]
    this.queries.splice(0, this.queries.length, ...candidates.slice(0, maxQueries))
    return [...this.queries]
  }

  async executeSearchBatch(maxResultsPerQuery = 5): Promise<readonly SearchResult[]> {
    if (!Number.isInteger(maxResultsPerQuery) || maxResultsPerQuery < 1) {
      throw new Error('maxResultsPerQuery must be a positive integer')
    }
    const batches = await Promise.all(this.queries.map(query => this.searchPort.search(query, maxResultsPerQuery)))
    const unique = new Map<string, SearchResult>()
    for (const result of batches.flat()) unique.set(result.url, result)
    this.results.splice(0, this.results.length, ...unique.values())
    return this.results
  }

  analyzeSearchResults(minResults = 1): SearchAnalysis {
    if (this.results.length < minResults) {
      throw new Error(`Need at least ${minResults} results before analysis; found ${this.results.length}`)
    }
    const terms = wordFrequency(this.results.flatMap(result => [result.title, result.excerpt]))
    return {
      resultCount: this.results.length,
      sources: [...new Set(this.results.map(result => result.source))].sort(),
      commonTerms: [...terms.entries()].sort((left, right) => right[1] - left[1]).slice(0, 8).map(([term]) => term),
    }
  }

  extractEntitiesFromResults(): string[] {
    return [...new Set(this.results.flatMap(result => extractCapitalizedTerms(`${result.title} ${result.excerpt}`)))].sort()
  }

  classifyContentSentiment(): { readonly neutral: number; readonly negative: number; readonly positive: number } {
    const totals = { positive: 0, negative: 0, neutral: 0 }
    for (const result of this.results) {
      const text = `${result.title} ${result.excerpt}`.toLowerCase()
      const score = countMatches(text, ['reliable', 'improved', 'successful', 'safe'])
        - countMatches(text, ['risk', 'failure', 'limited', 'unsafe'])
      if (score > 0) totals.positive += 1
      else if (score < 0) totals.negative += 1
      else totals.neutral += 1
    }
    return totals
  }

  generateContentSummary(maxLength = 500): string {
    if (!Number.isInteger(maxLength) || maxLength < 40) throw new Error('maxLength must be an integer of at least 40')
    const sourceText = this.results.map(result => `${result.title}: ${result.excerpt}`).join(' ')
    return sourceText.length <= maxLength ? sourceText : `${sourceText.slice(0, maxLength - 1).trimEnd()}…`
  }

  buildKnowledgeGraph(): KnowledgeGraph {
    const entities = this.extractEntitiesFromResults()
    return {
      nodes: [this.topic, ...entities],
      edges: entities.map(entity => ({ from: this.topic, to: entity })),
    }
  }

  generateResearchReport(): string {
    const analysis = this.analyzeSearchResults()
    const sentiment = this.classifyContentSentiment()
    return [
      `# Research report: ${this.topic}`,
      '',
      `## Coverage\n- Queries: ${this.queries.length}\n- Unique sources: ${analysis.resultCount}\n- Publishers: ${analysis.sources.join(', ') || 'none'}`,
      `## Common terms\n${analysis.commonTerms.map(term => `- ${term}`).join('\n') || '- none'}`,
      `## Sentiment signals\n- Positive: ${sentiment.positive}\n- Neutral: ${sentiment.neutral}\n- Negative: ${sentiment.negative}`,
      '## Evidence',
      ...this.results.map(result => `- [${result.title}](${result.url}) — ${result.excerpt}`),
      '',
      '> Local demo data is illustrative only. Inject a real search port before treating this report as live research.',
    ].join('\n')
  }

  snapshot(): { readonly queries: readonly string[]; readonly results: readonly SearchResult[]; readonly topic: string } {
    return { topic: this.topic, queries: [...this.queries], results: [...this.results] }
  }
}

export function demoSearchPort(): SearchPort {
  const catalog: readonly SearchResult[] = [
    {
      title: 'Bun native runtime guide',
      source: 'Example documentation',
      url: 'https://example.test/bun-runtime',
      excerpt: 'A reliable local-first runtime approach can reduce deployment complexity and improve startup behavior.',
    },
    {
      title: 'Cortex orchestration architecture',
      source: 'Example engineering notes',
      url: 'https://example.test/cortex',
      excerpt: 'Dependency-aware parallel workers isolate research, analysis, and report-writing stages with explicit boundaries.',
    },
    {
      title: 'Agent safety limitations',
      source: 'Example safety review',
      url: 'https://example.test/risks',
      excerpt: 'Any live search workflow has risks: stale evidence, incomplete sources, and unsafe tool access require review.',
    },
  ]
  return {
    search: (_query, maxResults) => catalog.slice(0, maxResults),
  }
}

async function main(): Promise<void> {
  const args = Bun.argv.slice(2)
  const topic = optionValue(args, '--topic') ?? 'Bun-native agent orchestration'
  const session = new DeepSearchSession(topic, demoSearchPort())
  divider('DEEP-SEARCH AGENT (native Bun workflow)')
  console.log(session.initializeSearchSession())
  const queries = session.generateSearchQueries(optionValue(args, '--strategy') ?? 'comprehensive', positiveIntegerOption(args, '--queries', 6))
  console.log(`Generated ${queries.length} queries.`)
  const results = await session.executeSearchBatch(3)
  console.log(`Collected ${results.length} illustrative results.\n`)
  console.log(session.generateResearchReport())
  const written = await writeJsonWhenRequested(args, 'outputs/deepsearch_agent/session.json', {
    snapshot: session.snapshot(),
    graph: session.buildKnowledgeGraph(),
  })
  if (written) console.log(`\nWrote requested artifact: ${written}`)
}

function wordFrequency(chunks: readonly string[]): Map<string, number> {
  const frequency = new Map<string, number>()
  for (const word of chunks.join(' ').toLowerCase().match(/[a-z][a-z-]{3,}/g) ?? []) {
    frequency.set(word, (frequency.get(word) ?? 0) + 1)
  }
  return frequency
}

function extractCapitalizedTerms(value: string): string[] {
  return value.match(/\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*/g) ?? []
}

function countMatches(value: string, terms: readonly string[]): number {
  return terms.reduce((total, term) => total + (value.includes(term) ? 1 : 0), 0)
}

if (import.meta.main) runMain(main)
