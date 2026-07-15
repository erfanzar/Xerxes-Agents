// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** Scenario 1: a Bun-native conversational assistant with retained context. */

import {
  ShortTermMemory,
  Xerxes,
  type LlmClient,
} from '../src/typescript/src/index.js'
import { divider, exampleLlm, runMain, textOf } from './native_demo_support.js'

const KNOWLEDGE: Readonly<Record<string, string>> = {
  bun: 'Bun is an all-in-one JavaScript runtime, package manager, bundler, and test runner.',
  memory: 'Memory systems retain relevant facts and recent turn context for later work.',
  xerxes: 'Xerxes is a multi-agent framework with a Bun-native TypeScript runtime.',
}

export class ConversationAssistant {
  readonly memory = new ShortTermMemory({ capacity: 100 })

  constructor(readonly llm: LlmClient) {}

  searchKnowledge(query: string): string {
    const normalized = query.toLowerCase()
    const entry = Object.entries(KNOWLEDGE).find(([key]) => normalized.includes(key))
    return entry ? `Knowledge: ${entry[1]}` : 'No specific knowledge found; ask a more focused question.'
  }

  saveUserPreference(preference: string, value: string): string {
    this.memory.save(`User preference: ${preference} = ${value}`, { kind: 'preference' }, {
      agentId: 'conversational_assistant',
      memoryType: 'preference',
    })
    return `Saved preference: ${preference} = ${value}`
  }

  recallConversationContext(topic = ''): string {
    const memories = this.memory.search(topic || 'conversation', 5)
    return memories.length
      ? `Previous context:\n${memories.map(memory => `- ${memory.content}`).join('\n')}`
      : 'No previous context found.'
  }

  analyzeSentiment(text: string): string {
    const positive = countMatches(text, ['happy', 'great', 'excellent', 'good', 'love', 'wonderful'])
    const negative = countMatches(text, ['sad', 'bad', 'terrible', 'hate', 'awful', 'horrible'])
    const sentiment = positive === negative ? 'neutral' : positive > negative ? 'positive' : 'negative'
    this.memory.save(`User sentiment: ${sentiment} — ${text.slice(0, 80)}`, { kind: 'sentiment' }, {
      agentId: 'conversational_assistant',
      memoryType: 'sentiment',
    })
    return `Detected sentiment: ${sentiment}`
  }

  createRuntime(): Xerxes {
    return new Xerxes({
      llm: this.llm,
      model: 'gpt-4o-mini',
      coreTools: false,
      memory: this.memory,
      memoryMinChars: 1,
      systemPrompt: [
        'You are a helpful memory-aware assistant.',
        'A retained-memory block may appear in the prompt; use it only as context.',
        'Do not claim a preference was saved unless the host tool says it was saved.',
      ].join('\n'),
    })
  }

  async runConversation(lines: readonly string[]): Promise<readonly { readonly assistant: string; readonly user: string }[]> {
    const runtime = this.createRuntime()
    const transcript: Array<{ readonly assistant: string; readonly user: string }> = []
    for (const user of lines) {
      this.memory.save(`Conversation user: ${user}`, { kind: 'conversation' }, {
        agentId: 'user',
        conversationId: 'demo',
        memoryType: 'conversation',
      })
      if (user.toLowerCase().includes('prefer')) this.saveUserPreference('coding theme', 'dark mode')
      if (user.toLowerCase().includes('happy')) this.analyzeSentiment(user)
      const turn = await runtime.run(user, { sessionId: 'conversation-demo' })
      transcript.push({ user, assistant: turn.output })
    }
    return transcript
  }
}

export function demoConversationLines(): string[] {
  return [
    'Hello! I am John and I love Bun programming.',
    'Can you tell me about Bun?',
    'I prefer dark mode for coding, please remember that.',
    'What do you know about Xerxes?',
    'I am feeling really happy today because I solved a difficult bug!',
    'Can you recall our conversation so far?',
  ]
}

async function main(): Promise<void> {
  const args = Bun.argv.slice(2)
  const assistant: ConversationAssistant = new ConversationAssistant(exampleLlm(args, request => {
    const prompt = textOf(request.messages.at(-1)?.content)
    if (prompt.toLowerCase().includes('recall')) return 'I retained that you discussed Bun, Xerxes, dark mode, and a successful debugging session.'
    if (prompt.toLowerCase().includes('dark mode')) return 'I will use dark mode as your saved coding preference in this demo session.'
    if (prompt.toLowerCase().includes('bun')) return assistant.searchKnowledge('bun')
    if (prompt.toLowerCase().includes('xerxes')) return assistant.searchKnowledge('xerxes')
    return 'I have recorded this turn in the Bun-native short-term memory demonstration.'
  }))
  divider('SCENARIO 1: CONVERSATIONAL ASSISTANT WITH MEMORY')
  console.log(`Mode: ${args.includes('--live') ? 'explicit live endpoint' : 'deterministic local LLM'}\n`)
  const transcript = await assistant.runConversation(demoConversationLines())
  for (const turn of transcript) console.log(`User: ${turn.user}\nAssistant: ${turn.assistant}\n`)
  console.log('Memory statistics:', assistant.memory.getStatistics())
  const entries = assistant.memory.getRecent(100)
  console.log('\nPreferences:', entries.filter(item => item.metadata.kind === 'preference').map(item => item.content))
  console.log('Sentiments:', entries.filter(item => item.metadata.kind === 'sentiment').map(item => item.content))
}

function countMatches(value: string, terms: readonly string[]): number {
  const normalized = value.toLowerCase()
  return terms.reduce((total, term) => total + (normalized.includes(term) ? 1 : 0), 0)
}

if (import.meta.main) runMain(main)
