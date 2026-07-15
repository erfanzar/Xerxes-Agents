// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** Scenario 4: real-time native stream events plus an in-memory research notebook. */

import {
  ShortTermMemory,
  Xerxes,
  type StreamEvent,
} from '../src/typescript/src/index.js'
import { divider, exampleLlm, runMain } from './native_demo_support.js'

export interface KnowledgeEntry {
  readonly connections: readonly string[]
  readonly facts: readonly string[]
  readonly topic: string
}

export class ResearchNotebook {
  readonly memory = new ShortTermMemory({ capacity: 500 })
  private readonly topics = new Map<string, KnowledgeEntry>()

  extractKeyPoints(text: string, maxPoints = 5): string[] {
    const points = text.split(/[.!?]+/).map(sentence => sentence.trim()).filter(sentence => sentence.length > 20).slice(0, maxPoints)
    for (const point of points) this.memory.save(point, { kind: 'key_point' }, { agentId: 'research_assistant', memoryType: 'semantic' })
    return points
  }

  buildKnowledgeGraph(topic: string, facts: readonly string[]): KnowledgeEntry {
    const existing = this.topics.get(topic) ?? { topic, facts: [], connections: [] }
    const knownFacts = [...new Set([...existing.facts, ...facts])]
    const topicWords = new Set(topic.toLowerCase().split(/\s+/))
    const connections = [...this.topics.values()]
      .filter(entry => entry.topic !== topic && entry.topic.toLowerCase().split(/\s+/).some(word => topicWords.has(word)))
      .map(entry => entry.topic)
    const entry = { topic, facts: knownFacts, connections }
    this.topics.set(topic, entry)
    for (const fact of facts) this.memory.save(`${topic}: ${fact}`, { kind: 'fact' }, { agentId: 'research_assistant', memoryType: 'long_term' })
    return entry
  }

  synthesizeResearch(topics: readonly string[]): string {
    const facts = topics.flatMap(topic => this.topics.get(topic)?.facts ?? [])
    const themes = mostFrequentTerms(facts).slice(0, 5)
    return [
      '# Research synthesis',
      `Topics analyzed: ${topics.join(', ')}`,
      `Total facts gathered: ${facts.length}`,
      `Common themes: ${themes.join(', ') || 'none yet'}`,
      'Insight: this demonstration records facts only through explicit notebook APIs.',
    ].join('\n')
  }

  createResearchOutline(topic: string, depth: 'basic' | 'comprehensive' | 'moderate' = 'moderate'): string {
    const count = { basic: 3, moderate: 5, comprehensive: 8 }[depth]
    const sections = ['Introduction', 'Core concepts', 'Current state', 'Applications', 'Challenges', 'Future directions', 'Related topics', 'Conclusion']
    return [`# Research outline: ${topic}`, ...sections.slice(0, count).map((section, index) => `${index + 1}. ${section}`)].join('\n')
  }

  trackResearchProgress(topic: string): string {
    const entry = this.topics.get(topic)
    const factCount = entry?.facts.length ?? 0
    const completeness = Math.min(100, factCount * 20)
    return `Research progress for ${topic}: ${factCount} facts, ${completeness}% illustrative completeness.`
  }
}

/** Consume the real native stream event contract and return its rendered text. */
export async function streamResearchResponse(runtime: Xerxes, prompt: string): Promise<{ readonly events: readonly StreamEvent[]; readonly output: string }> {
  const events: StreamEvent[] = []
  const stream = runtime.runStream(prompt, { sessionId: 'streaming-research-demo' })
  while (true) {
    const next = await stream.next()
    if (next.done) return { events, output: next.value.output }
    events.push(next.value)
  }
}

async function main(): Promise<void> {
  const args = Bun.argv.slice(2)
  const notebook = new ResearchNotebook()
  const runtime = new Xerxes({
    model: 'gpt-4o-mini',
    coreTools: false,
    memory: notebook.memory,
    memoryMinChars: 1,
    llm: exampleLlm(args, () => [
      'Bun-native streaming emits incremental text events while the host owns memory and research tools.',
      'A reliable study should separate cited evidence from synthesized conclusions.',
    ].join(' ')),
    systemPrompt: 'You are a concise research assistant. State the boundary between demonstration data and verified evidence.',
  })
  divider('SCENARIO 4: STREAMING RESEARCH ASSISTANT')
  const streamed = await streamResearchResponse(runtime, 'Explain a safe research workflow for Bun-native agents.')
  for (const event of streamed.events) if (event.type === 'text') process.stdout.write(event.text)
  console.log(`\n\nStreamed event types: ${streamed.events.map(event => event.type).join(', ')}`)
  const facts = notebook.extractKeyPoints(streamed.output)
  const graph = notebook.buildKnowledgeGraph('Bun-native agents', facts)
  console.log(`\n${notebook.createResearchOutline(graph.topic)}`)
  console.log(`\n${notebook.trackResearchProgress(graph.topic)}`)
  console.log(`\n${notebook.synthesizeResearch([graph.topic])}`)
}

function mostFrequentTerms(values: readonly string[]): string[] {
  const terms = new Map<string, number>()
  for (const word of values.join(' ').toLowerCase().match(/[a-z][a-z-]{4,}/g) ?? []) {
    terms.set(word, (terms.get(word) ?? 0) + 1)
  }
  return [...terms.entries()].sort((left, right) => right[1] - left[1]).map(([word]) => word)
}

if (import.meta.main) runMain(main)
