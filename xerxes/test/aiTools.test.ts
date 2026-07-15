// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { ToolRegistry } from '../src/executors/toolRegistry.js'
import {
  AI_TOOL_DEFINITIONS,
  EntityExtractor,
  TextEmbedder,
  TextSimilarity,
  entityExtractor,
  registerAiTools,
  textClassifier,
  textEmbedder,
  textSimilarity,
  textSummarizer,
} from '../src/tools/aiTools.js'
import type { JsonObject, ToolCall } from '../src/types/toolCalls.js'

function call(name: string, arguments_: JsonObject): ToolCall {
  return {
    id: crypto.randomUUID(),
    type: 'function',
    function: { name, arguments: arguments_ },
  }
}

test('TextEmbedder produces deterministic dependency-free TF-IDF vectors and honors truncation', async () => {
  const inputs: JsonObject = {
    text: ['Alpha beta beta', 'Beta gamma'],
    method: 'tfidf',
  }
  const first = await textEmbedder(inputs)
  const second = await textEmbedder(inputs)

  expect(first).toEqual(second)
  expect(first.shape).toEqual([2, 3])
  expect(first.features).toEqual(['beta', 'alpha', 'gamma'])
  expect((first.embeddings as number[][])[0]).toHaveLength(3)

  const truncated = await TextEmbedder.staticCall('one two three', 'tfidf', undefined, 3)
  expect(truncated.shape).toEqual([1, 1])
  expect(truncated.features).toEqual(['one'])
})

test('TextEmbedder uses only injected advanced providers and reports absent ones explicitly', async () => {
  const missing = await textEmbedder({ text: 'hello', method: 'openai' })
  expect(missing.error).toContain('injected openaiEmbeddings provider')

  const response = await textEmbedder(
    { text: ['one', 'two'], method: 'openai', model_name: 'fixture-embed' },
    {
      openaiEmbeddings: {
        embed(request) {
          expect(request).toEqual({ model: 'fixture-embed', texts: ['one', 'two'] })
          return {
            embeddings: [[1, 0], [0, 1]],
            model: 'fixture-embed',
            usage: { total_tokens: 2 },
          }
        },
      },
    },
  )
  expect(response).toEqual({
    embeddings: [[1, 0], [0, 1]],
    shape: [2, 2],
    model: 'fixture-embed',
    usage: { total_tokens: 2 },
  })
})

test('TextSimilarity supports deterministic lexical methods and injected semantic scores', async () => {
  expect(await textSimilarity({
    text1: 'machine learning',
    text2: 'deep learning',
    method: 'cosine',
  })).toMatchObject({
    similarity: 0.5,
    method: 'cosine',
    interpretation: 'Low similarity',
  })
  expect(await textSimilarity({
    text1: 'alpha beta',
    text2: 'beta gamma',
    method: 'jaccard',
  })).toMatchObject({
    similarity: 1 / 3,
    common_words: ['beta'],
  })
  expect(await TextSimilarity.staticCall('kitten', 'sitting', 'levenshtein')).toMatchObject({
    similarity: 4 / 7,
    distance: 3,
  })

  const semantic = await textSimilarity(
    { text1: 'cat', text2: 'feline', method: 'semantic' },
    {
      semanticSimilarity: {
        model: 'fixture-semantic',
        similarity: () => 0.82,
      },
    },
  )
  expect(semantic).toMatchObject({
    similarity: 0.82,
    method: 'semantic',
    model: 'fixture-semantic',
    interpretation: 'High similarity',
  })
  expect((await textSimilarity({ text1: 'cat', text2: 'feline', method: 'semantic' })).error).toContain(
    'requires an injected',
  )
})

test('TextClassifier and TextSummarizer expose stable local analyses', () => {
  expect(textClassifier({
    text: 'I love this excellent product, but the market is difficult.',
    method: 'sentiment',
  })).toMatchObject({
    sentiment: 'positive',
    positive_score: 2,
    negative_score: 0,
  })
  expect(textClassifier({
    text: 'Software programming data algorithm code.',
    method: 'topic',
  })).toMatchObject({
    topic: 'technology',
    confidence: 1,
  })
  expect(textClassifier({ text: 'not classified' })).toEqual({
    error: 'categories required for keyword classification',
  })

  const summary = textSummarizer({
    text: 'Xerxes builds agents. Xerxes builds reliable agents. Tea is warm.',
    max_sentences: 2,
  })
  expect(summary.summary).toContain('Xerxes')
  expect(summary.original_length).toBeGreaterThan(0)

  expect(textSummarizer({
    text: 'Agents need durable memory. Agents need reliable tools.',
    method: 'keywords',
  })).toMatchObject({
    keywords: ['agents', 'need', 'durable', 'memory', 'reliable', 'tools'],
    key_phrases: ['agents need', 'durable memory', 'memory agents', 'need durable', 'need reliable'],
  })
  expect(textSummarizer({ text: 'Two words. Four words live here.', method: 'statistics' })).toMatchObject({
    summary: {
      total_words: 6,
      total_sentences: 2,
      longest_sentence: 4,
      shortest_sentence: 2,
    },
  })
})

test('EntityExtractor returns bounded first-seen entities and rejects unsupported types', () => {
  const result = EntityExtractor.staticCall(
    'Ada Lovelace emailed ada@example.com at 09:30 on 2026-07-13. Visit https://example.com/docs. #Xerxes @ada paid $12.50.',
    ['emails', 'urls', 'dates', 'times', 'hashtags', 'mentions', 'currency', 'names'],
  )
  expect(result).toMatchObject({
    entities: {
      emails: ['ada@example.com'],
      urls: ['https://example.com/docs'],
      dates: ['2026-07-13'],
      times: ['09:30'],
      hashtags: ['#Xerxes'],
      mentions: ['@ada'],
      currency: ['$12.50'],
      names: ['Ada Lovelace'],
    },
    total_entities: 8,
  })
  expect(entityExtractor({ text: 'hello', entity_types: ['unknown'] })).toEqual({
    error: 'Unknown entity types: unknown',
  })
})

test('AI tools register under Python-compatible public names', async () => {
  const registry = new ToolRegistry()
  registerAiTools(registry)
  expect(registry.definitions().map(definition => definition.function.name)).toEqual(
    AI_TOOL_DEFINITIONS.map(definition => definition.function.name),
  )

  const classified = JSON.parse(await registry.execute(call('TextClassifier', {
    text: 'software and data',
    method: 'topic',
  }), { metadata: {} })) as JsonObject
  expect(classified.topic).toBe('technology')
})
