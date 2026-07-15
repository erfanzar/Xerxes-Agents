// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ValidationError } from '../core/errors.js'
import { ToolRegistry } from '../executors/toolRegistry.js'
import type { JsonObject, ToolDefinition } from '../types/toolCalls.js'

const DEFAULT_MAX_LENGTH = 512
const DEFAULT_MAX_SENTENCES = 3
const DEFAULT_SENTENCE_TRANSFORMER_MODEL = 'all-MiniLM-L6-v2'
const DEFAULT_OPENAI_EMBEDDING_MODEL = 'text-embedding-ada-002'
const MAX_EMBEDDING_FEATURES = 100
const MAX_RETURNED_FEATURES = 20
const MAX_RETURNED_ENTITIES = 20
const MAX_LEVENSHTEIN_CHARACTERS = 20_000
const MAX_TEXT_LENGTH = 1_000_000

const SENTENCE_STOP_WORDS = new Set([
  'a',
  'an',
  'and',
  'are',
  'at',
  'but',
  'by',
  'for',
  'have',
  'has',
  'had',
  'in',
  'is',
  'it',
  'of',
  'on',
  'or',
  'the',
  'to',
  'was',
  'were',
  'with',
])

const POSITIVE_WORDS = [
  'good',
  'great',
  'excellent',
  'amazing',
  'wonderful',
  'fantastic',
  'love',
  'best',
  'happy',
  'joy',
]

const NEGATIVE_WORDS = [
  'bad',
  'terrible',
  'awful',
  'horrible',
  'hate',
  'worst',
  'sad',
  'angry',
  'frustrating',
  'disappointing',
]

const LANGUAGE_INDICATORS: Readonly<Record<string, readonly string[]>> = {
  english: ['the', 'is', 'and', 'to', 'of', 'in', 'that', 'it', 'with', 'for'],
  spanish: ['el', 'la', 'de', 'que', 'en', 'los', 'las', 'por', 'con', 'para'],
  french: ['le', 'de', 'la', 'et', 'les', 'des', 'en', 'un', 'une', 'pour'],
  german: ['der', 'die', 'und', 'das', 'ist', 'den', 'dem', 'mit', 'zu', 'ein'],
  italian: ['il', 'di', 'la', 'che', 'e', 'le', 'della', 'per', 'con', 'del'],
}

const TOPIC_KEYWORDS: Readonly<Record<string, readonly string[]>> = {
  technology: [
    'computer',
    'software',
    'hardware',
    'internet',
    'digital',
    'data',
    'algorithm',
    'programming',
    'code',
    'app',
  ],
  business: [
    'company',
    'market',
    'sales',
    'revenue',
    'profit',
    'customer',
    'product',
    'service',
    'management',
    'strategy',
  ],
  science: [
    'research',
    'study',
    'experiment',
    'hypothesis',
    'theory',
    'discovery',
    'analysis',
    'evidence',
    'method',
    'result',
  ],
  health: [
    'medical',
    'health',
    'doctor',
    'patient',
    'treatment',
    'disease',
    'medicine',
    'hospital',
    'symptom',
    'diagnosis',
  ],
  education: [
    'student',
    'teacher',
    'school',
    'learn',
    'education',
    'course',
    'class',
    'university',
    'study',
    'knowledge',
  ],
}

const ENTITY_TYPES = [
  'emails',
  'urls',
  'phone_numbers',
  'dates',
  'times',
  'numbers',
  'hashtags',
  'mentions',
  'currency',
  'names',
] as const

type EntityType = (typeof ENTITY_TYPES)[number]

export interface TextEmbeddingRequest {
  readonly model: string
  readonly texts: readonly string[]
}

export interface TextEmbeddingResponse {
  readonly embeddings: readonly (readonly number[])[]
  readonly model?: string
  readonly usage?: JsonObject
}

/** Injectable embedding port for OpenAI, sentence-transformers, or a local model host. */
export interface TextEmbeddingProvider {
  embed(request: TextEmbeddingRequest, signal?: AbortSignal): TextEmbeddingResponse | Promise<TextEmbeddingResponse>
}

export interface SemanticSimilarityRequest {
  readonly text1: string
  readonly text2: string
}

/** Injectable direct semantic-similarity port for providers that do not expose embeddings. */
export interface SemanticSimilarityProvider {
  similarity(request: SemanticSimilarityRequest, signal?: AbortSignal): number | Promise<number>
  readonly model?: string
}

/**
 * Provider hooks are deliberately explicit. The built-in methods never load
 * model weights or inspect environment variables on their own.
 */
export interface AiToolProviders {
  readonly openaiEmbeddings?: TextEmbeddingProvider
  readonly semanticSimilarity?: SemanticSimilarityProvider
  readonly sentenceTransformerEmbeddings?: TextEmbeddingProvider
}

export const TEXT_EMBEDDER_DEFINITION: ToolDefinition = {
  type: 'function',
  function: {
    name: 'TextEmbedder',
    description: 'Create deterministic TF-IDF embeddings or use an explicitly injected embedding provider.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        text: {
          description: 'A text string or an array of text strings.',
          anyOf: [{ type: 'string' }, { type: 'array', items: { type: 'string' } }],
        },
        method: { type: 'string', enum: ['tfidf', 'sentence-transformers', 'openai'], default: 'tfidf' },
        model_name: { type: 'string', description: 'Optional provider model override.' },
        max_length: { type: 'integer', minimum: 0, maximum: MAX_TEXT_LENGTH, default: DEFAULT_MAX_LENGTH },
      },
      required: ['text'],
    },
  },
}

export const TEXT_SIMILARITY_DEFINITION: ToolDefinition = {
  type: 'function',
  function: {
    name: 'TextSimilarity',
    description: 'Calculate cosine, Jaccard, Levenshtein, or injected semantic similarity for two text strings.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        text1: { type: 'string' },
        text2: { type: 'string' },
        method: { type: 'string', enum: ['cosine', 'jaccard', 'levenshtein', 'semantic'], default: 'cosine' },
      },
      required: ['text1', 'text2'],
    },
  },
}

export const TEXT_CLASSIFIER_DEFINITION: ToolDefinition = {
  type: 'function',
  function: {
    name: 'TextClassifier',
    description: 'Classify text by deterministic keyword, sentiment, language, or topic heuristics.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        text: { type: 'string' },
        categories: { type: 'array', items: { type: 'string' } },
        method: { type: 'string', enum: ['keyword', 'sentiment', 'language', 'topic'], default: 'keyword' },
      },
      required: ['text'],
    },
  },
}

export const TEXT_SUMMARIZER_DEFINITION: ToolDefinition = {
  type: 'function',
  function: {
    name: 'TextSummarizer',
    description: 'Create extractive summaries, keywords, or deterministic text statistics.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        text: { type: 'string' },
        method: { type: 'string', enum: ['extractive', 'keywords', 'statistics'], default: 'extractive' },
        max_sentences: { type: 'integer', minimum: 1, default: DEFAULT_MAX_SENTENCES },
        max_length: { type: 'integer', minimum: 1, maximum: MAX_TEXT_LENGTH },
      },
      required: ['text'],
    },
  },
}

export const ENTITY_EXTRACTOR_DEFINITION: ToolDefinition = {
  type: 'function',
  function: {
    name: 'EntityExtractor',
    description: 'Extract deterministic regex-based emails, URLs, numbers, dates, names, and social entities.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        text: { type: 'string' },
        entity_types: { type: 'array', items: { type: 'string', enum: ENTITY_TYPES } },
      },
      required: ['text'],
    },
  },
}

export const AI_TOOL_DEFINITIONS: readonly ToolDefinition[] = [
  TEXT_EMBEDDER_DEFINITION,
  TEXT_SIMILARITY_DEFINITION,
  TEXT_CLASSIFIER_DEFINITION,
  TEXT_SUMMARIZER_DEFINITION,
  ENTITY_EXTRACTOR_DEFINITION,
]

/** Register the complete AI text-processing family under its Python-compatible tool names. */
export function registerAiTools(registry: ToolRegistry, providers: AiToolProviders = {}, agentId = 'default'): void {
  registry.register(TEXT_EMBEDDER_DEFINITION, (inputs, _context, signal) => textEmbedder(inputs, providers, signal), agentId)
  registry.register(TEXT_SIMILARITY_DEFINITION, (inputs, _context, signal) => textSimilarity(inputs, providers, signal), agentId)
  registry.register(TEXT_CLASSIFIER_DEFINITION, textClassifier, agentId)
  registry.register(TEXT_SUMMARIZER_DEFINITION, textSummarizer, agentId)
  registry.register(ENTITY_EXTRACTOR_DEFINITION, entityExtractor, agentId)
}

/**
 * Create one vector per input string. TF-IDF runs locally; advanced methods
 * only run through an explicit provider hook.
 */
export async function textEmbedder(
  inputs: JsonObject,
  providers: AiToolProviders = {},
  signal?: AbortSignal,
): Promise<JsonObject> {
  const texts = embeddingInputTexts(inputs)
  const maxLength = boundedInteger(inputs, 'max_length', DEFAULT_MAX_LENGTH, 0, MAX_TEXT_LENGTH)
  const method = optionalText(inputs, 'method') ?? 'tfidf'
  const modelName = optionalText(inputs, 'model_name')
  const truncated = texts.map(text => text.slice(0, maxLength))

  if (method === 'tfidf') {
    return tfidfEmbeddings(truncated)
  }
  if (method === 'sentence-transformers') {
    const provider = providers.sentenceTransformerEmbeddings
    if (!provider) {
      return failure('sentence-transformers embeddings require an injected sentenceTransformerEmbeddings provider')
    }
    return embeddingsFromProvider(
      provider,
      truncated,
      modelName ?? DEFAULT_SENTENCE_TRANSFORMER_MODEL,
      'sentence-transformers',
      signal,
    )
  }
  if (method === 'openai') {
    const provider = providers.openaiEmbeddings
    if (!provider) {
      return failure('OpenAI embeddings require an injected openaiEmbeddings provider')
    }
    return embeddingsFromProvider(provider, truncated, modelName ?? DEFAULT_OPENAI_EMBEDDING_MODEL, 'OpenAI', signal)
  }
  return failure('Unknown embedding method: ' + method)
}

/** Compute a deterministic lexical score, edit-distance score, or injected semantic score. */
export async function textSimilarity(
  inputs: JsonObject,
  providers: AiToolProviders = {},
  signal?: AbortSignal,
): Promise<JsonObject> {
  const text1 = textValue(inputs, 'text1')
  const text2 = textValue(inputs, 'text2')
  const method = optionalText(inputs, 'method') ?? 'cosine'

  if (method === 'cosine') {
    return similarityResult(cosineWordSimilarity(text1, text2), 'cosine', '0 to 1 (1 = identical)')
  }
  if (method === 'jaccard') {
    const words1 = new Set(splitWhitespaceWords(text1))
    const words2 = new Set(splitWhitespaceWords(text2))
    const commonWords = [...words1].filter(word => words2.has(word)).sort()
    const union = new Set([...words1, ...words2])
    const score = union.size === 0 ? 0 : commonWords.length / union.size
    const result = similarityResult(score, 'jaccard', '0 to 1 (1 = identical)')
    result.common_words = commonWords.slice(0, MAX_RETURNED_FEATURES)
    return result
  }
  if (method === 'levenshtein') {
    if (text1.length > MAX_LEVENSHTEIN_CHARACTERS || text2.length > MAX_LEVENSHTEIN_CHARACTERS) {
      return failure('Levenshtein inputs must each contain at most ' + String(MAX_LEVENSHTEIN_CHARACTERS) + ' characters')
    }
    const distance = levenshteinDistance(text1, text2)
    const maximumLength = Math.max(text1.length, text2.length)
    const score = maximumLength === 0 ? 1 : 1 - distance / maximumLength
    const result = similarityResult(score, 'levenshtein', '0 to 1 (1 = identical)')
    result.distance = distance
    return result
  }
  if (method === 'semantic') {
    return semanticSimilarity(text1, text2, providers, signal)
  }
  return failure('Unknown similarity method: ' + method)
}

/** Run the deterministic text classifier used by the TextClassifier tool. */
export function textClassifier(inputs: JsonObject): JsonObject {
  const text = textValue(inputs, 'text')
  const method = optionalText(inputs, 'method') ?? 'keyword'

  if (method === 'keyword') {
    const categories = optionalStringArray(inputs, 'categories')
    if (categories.length === 0) {
      return failure('categories required for keyword classification')
    }
    const lower = text.toLowerCase()
    const scores: JsonObject = {}
    let total = 0
    let topCategory = categories[0] ?? ''
    let topScore = -1
    for (const category of categories) {
      const score = splitWhitespaceWords(category).reduce((count, word) => count + (lower.includes(word) ? 1 : 0), 0)
      scores[category] = score
      total += score
      if (score > topScore) {
        topCategory = category
        topScore = score
      }
    }
    return {
      category: topCategory,
      confidence: total === 0 ? 0 : topScore / total,
      scores,
    }
  }

  if (method === 'sentiment') {
    const lower = text.toLowerCase()
    const positiveScore = POSITIVE_WORDS.reduce((count, word) => count + (lower.includes(word) ? 1 : 0), 0)
    const negativeScore = NEGATIVE_WORDS.reduce((count, word) => count + (lower.includes(word) ? 1 : 0), 0)
    const total = positiveScore + negativeScore
    if (positiveScore > negativeScore) {
      return {
        sentiment: 'positive',
        confidence: total === 0 ? 0.5 : positiveScore / total,
        positive_score: positiveScore,
        negative_score: negativeScore,
      }
    }
    if (negativeScore > positiveScore) {
      return {
        sentiment: 'negative',
        confidence: total === 0 ? 0.5 : negativeScore / total,
        positive_score: positiveScore,
        negative_score: negativeScore,
      }
    }
    return {
      sentiment: 'neutral',
      confidence: 0.5,
      positive_score: positiveScore,
      negative_score: negativeScore,
    }
  }

  if (method === 'language') {
    const words = splitWhitespaceWords(text)
    const scores: JsonObject = {}
    let language = 'english'
    let highest = -1
    for (const [candidate, indicators] of Object.entries(LANGUAGE_INDICATORS)) {
      const indicatorsSet = new Set(indicators)
      const score = words.reduce((count, word) => count + (indicatorsSet.has(word) ? 1 : 0), 0)
      scores[candidate] = score
      if (score > highest) {
        language = candidate
        highest = score
      }
    }
    return {
      language,
      confidence: words.length === 0 ? 0 : highest / words.length,
      scores,
    }
  }

  if (method === 'topic') {
    const lower = text.toLowerCase()
    const scores: JsonObject = {}
    let topic = 'technology'
    let highest = -1
    let total = 0
    for (const [candidate, keywords] of Object.entries(TOPIC_KEYWORDS)) {
      const score = keywords.reduce((count, keyword) => count + (lower.includes(keyword) ? 1 : 0), 0)
      scores[candidate] = score
      total += score
      if (score > highest) {
        topic = candidate
        highest = score
      }
    }
    return {
      topic,
      confidence: total === 0 ? 0 : highest / total,
      scores,
    }
  }

  return failure('Unknown classification method: ' + method)
}

/** Produce a local extractive summary, keyword list, or text statistics. */
export function textSummarizer(inputs: JsonObject): JsonObject {
  const text = textValue(inputs, 'text')
  const method = optionalText(inputs, 'method') ?? 'extractive'
  const maxSentences = boundedInteger(inputs, 'max_sentences', DEFAULT_MAX_SENTENCES, 1, Number.MAX_SAFE_INTEGER)
  const maxLength = optionalBoundedInteger(inputs, 'max_length', 1, MAX_TEXT_LENGTH)

  if (method === 'extractive') {
    const sentences = splitSentences(text)
    if (sentences.length === 0) {
      return failure('No sentences found in text')
    }
    const wordFrequency = frequency(
      tokenize(text).filter(word => !SENTENCE_STOP_WORDS.has(word)),
    )
    const scored = sentences.map((sentence, index) => {
      const words = tokenize(sentence)
      const score = words.length === 0
        ? 0
        : words.reduce((total, word) => total + (wordFrequency.get(word) ?? 0), 0) / words.length
      return { index, score, sentence }
    })
    scored.sort((left, right) => right.score - left.score || left.index - right.index)
    let summary = scored
      .slice(0, maxSentences)
      .map(item => item.sentence)
      .join('. ')
    if (!summary.endsWith('.')) {
      summary += '.'
    }
    if (maxLength !== undefined && summary.length > maxLength) {
      summary = summary.slice(0, maxLength) + '...'
    }
    return {
      summary,
      original_length: text.length,
      summary_length: summary.length,
      compression_ratio: text.length === 0 ? 0 : summary.length / text.length,
    }
  }

  if (method === 'keywords') {
    const words = tokenize(text).filter(word => !SENTENCE_STOP_WORDS.has(word) && word.length > 3)
    const keywords = mostFrequent(frequency(words), 10)
    const rawWords = tokenize(text)
    const phrases: string[] = []
    for (let index = 0; index + 1 < rawWords.length; index += 1) {
      const first = rawWords[index]
      const second = rawWords[index + 1]
      if (first !== undefined && second !== undefined && !SENTENCE_STOP_WORDS.has(first) && !SENTENCE_STOP_WORDS.has(second)) {
        phrases.push(first + ' ' + second)
      }
    }
    const keyPhrases = mostFrequent(frequency(phrases), 5)
    return {
      keywords,
      key_phrases: keyPhrases,
      summary: 'Key topics: ' + keywords.slice(0, 5).join(', '),
    }
  }

  if (method === 'statistics') {
    const sentences = splitSentences(text)
    const words = text.split(/\s+/u).filter(Boolean)
    const unique = new Set(words.map(word => word.toLowerCase()))
    const sentenceLengths = sentences.map(sentence => sentence.split(/\s+/u).filter(Boolean).length)
    return {
      summary: {
        total_characters: text.length,
        total_words: words.length,
        unique_words: unique.size,
        vocabulary_richness: words.length === 0 ? 0 : unique.size / words.length,
        total_sentences: sentences.length,
        avg_sentence_length: sentences.length === 0 ? 0 : words.length / sentences.length,
        longest_sentence: sentenceLengths.length === 0 ? 0 : Math.max(...sentenceLengths),
        shortest_sentence: sentenceLengths.length === 0 ? 0 : Math.min(...sentenceLengths),
      },
    }
  }

  return failure('Unknown summarization method: ' + method)
}

/** Extract a bounded, first-seen-ordered set of requested entities. */
export function entityExtractor(inputs: JsonObject): JsonObject {
  const text = textValue(inputs, 'text')
  const requested = optionalStringArray(inputs, 'entity_types')
  const entityTypes = requested.length === 0 ? ENTITY_TYPES : requested
  const unknown = entityTypes.filter(type => !isEntityType(type))
  if (unknown.length > 0) {
    return failure('Unknown entity types: ' + unknown.join(', '))
  }

  const entities: JsonObject = {}
  let totalEntities = 0
  for (const entityType of entityTypes) {
    if (!isEntityType(entityType)) continue
    const matches = extractEntityType(text, entityType)
    entities[entityType] = matches
    totalEntities += matches.length
  }
  return { entities, total_entities: totalEntities }
}

/**
 * Python-compatible class facade for embedders that previously called static_call.
 * New TypeScript callers should prefer the lowercase functions above.
 */
export class TextEmbedder {
  static staticCall(
    text: string | readonly string[],
    method = 'tfidf',
    modelName?: string,
    maxLength = DEFAULT_MAX_LENGTH,
    providers: AiToolProviders = {},
    signal?: AbortSignal,
  ): Promise<JsonObject> {
    const inputs: JsonObject = {
      text: typeof text === 'string' ? text : [...text],
      method,
      max_length: maxLength,
    }
    if (modelName !== undefined) inputs.model_name = modelName
    return textEmbedder(inputs, providers, signal)
  }
}

/** Python-compatible class facade for text similarity. */
export class TextSimilarity {
  static staticCall(
    text1: string,
    text2: string,
    method = 'cosine',
    providers: AiToolProviders = {},
    signal?: AbortSignal,
  ): Promise<JsonObject> {
    return textSimilarity({ text1, text2, method }, providers, signal)
  }
}

/** Python-compatible class facade for deterministic classification. */
export class TextClassifier {
  static staticCall(text: string, categories?: readonly string[], method = 'keyword'): JsonObject {
    const inputs: JsonObject = { text, method }
    if (categories !== undefined) inputs.categories = [...categories]
    return textClassifier(inputs)
  }
}

/** Python-compatible class facade for local summaries. */
export class TextSummarizer {
  static staticCall(text: string, method = 'extractive', maxSentences = DEFAULT_MAX_SENTENCES, maxLength?: number): JsonObject {
    const inputs: JsonObject = { text, method, max_sentences: maxSentences }
    if (maxLength !== undefined) inputs.max_length = maxLength
    return textSummarizer(inputs)
  }
}

/** Python-compatible class facade for entity extraction. */
export class EntityExtractor {
  static staticCall(text: string, entityTypes?: readonly string[]): JsonObject {
    const inputs: JsonObject = { text }
    if (entityTypes !== undefined) inputs.entity_types = [...entityTypes]
    return entityExtractor(inputs)
  }
}

async function embeddingsFromProvider(
  provider: TextEmbeddingProvider,
  texts: readonly string[],
  model: string,
  label: string,
  signal?: AbortSignal,
): Promise<JsonObject> {
  try {
    const response = await provider.embed({ model, texts }, signal)
    const embeddings = validatedEmbeddings(response.embeddings, texts.length)
    const result: JsonObject = {
      embeddings,
      shape: [embeddings.length, embeddings[0]?.length ?? 0],
      model: response.model ?? model,
    }
    if (response.usage !== undefined) result.usage = response.usage
    return result
  } catch (error) {
    return failure(label + ' embedding failed: ' + errorText(error))
  }
}

async function semanticSimilarity(
  text1: string,
  text2: string,
  providers: AiToolProviders,
  signal?: AbortSignal,
): Promise<JsonObject> {
  const direct = providers.semanticSimilarity
  if (direct) {
    try {
      const score = await direct.similarity({ text1, text2 }, signal)
      return semanticResult(score, direct.model)
    } catch (error) {
      return failure('semantic similarity provider failed: ' + errorText(error))
    }
  }

  const embeddingsProvider = providers.sentenceTransformerEmbeddings
  if (!embeddingsProvider) {
    return failure(
      'semantic similarity requires an injected semanticSimilarity or sentenceTransformerEmbeddings provider',
    )
  }
  try {
    const response = await embeddingsProvider.embed({
      model: DEFAULT_SENTENCE_TRANSFORMER_MODEL,
      texts: [text1, text2],
    }, signal)
    const embeddings = validatedEmbeddings(response.embeddings, 2)
    const first = embeddings[0]
    const second = embeddings[1]
    if (!first || !second) {
      return failure('semantic similarity provider returned fewer than two embeddings')
    }
    return semanticResult(cosineVectors(first, second), response.model ?? DEFAULT_SENTENCE_TRANSFORMER_MODEL)
  } catch (error) {
    return failure('semantic similarity provider failed: ' + errorText(error))
  }
}

function semanticResult(score: number, model?: string): JsonObject {
  if (!Number.isFinite(score) || score < -1 || score > 1) {
    return failure('semantic similarity provider must return a finite score from -1 to 1')
  }
  const result = similarityResult(score, 'semantic', '-1 to 1 (1 = identical)')
  if (model !== undefined) result.model = model
  return result
}

function tfidfEmbeddings(texts: readonly string[]): JsonObject {
  const tokenLists = texts.map(tokenize)
  const totalFrequency = frequency(tokenLists.flat())
  const documentFrequency = new Map<string, number>()
  for (const tokens of tokenLists) {
    for (const word of new Set(tokens)) {
      documentFrequency.set(word, (documentFrequency.get(word) ?? 0) + 1)
    }
  }
  const features = mostFrequent(totalFrequency, MAX_EMBEDDING_FEATURES)
  const embeddings = tokenLists.map(tokens => {
    const counts = frequency(tokens)
    const raw = features.map(feature => {
      const count = counts.get(feature) ?? 0
      if (count === 0 || tokens.length === 0) return 0
      const termFrequency = count / tokens.length
      const inverseDocumentFrequency = Math.log((1 + texts.length) / (1 + (documentFrequency.get(feature) ?? 0))) + 1
      return termFrequency * inverseDocumentFrequency
    })
    return normalizeVector(raw)
  })
  return {
    embeddings,
    shape: [embeddings.length, features.length],
    features: features.slice(0, MAX_RETURNED_FEATURES),
  }
}

function validatedEmbeddings(value: readonly (readonly number[])[], expectedLength: number): number[][] {
  if (!Array.isArray(value)) {
    throw new Error('provider response embeddings must be an array')
  }
  if (value.length !== expectedLength) {
    throw new Error('provider returned ' + String(value.length) + ' embeddings for ' + String(expectedLength) + ' inputs')
  }
  let dimension: number | undefined
  const vectors: number[][] = []
  for (const vector of value) {
    if (!Array.isArray(vector)) {
      throw new Error('provider response embedding must be an array of numbers')
    }
    if (dimension === undefined) dimension = vector.length
    if (vector.length !== dimension) {
      throw new Error('provider response embeddings must all have the same dimension')
    }
    const copy: number[] = []
    for (const entry of vector) {
      if (typeof entry !== 'number' || !Number.isFinite(entry)) {
        throw new Error('provider response embeddings must contain finite numbers')
      }
      copy.push(entry)
    }
    vectors.push(copy)
  }
  return vectors
}

function similarityResult(similarity: number, method: string, scale: string): JsonObject {
  return {
    similarity,
    method,
    scale,
    interpretation: similarityInterpretation(similarity),
  }
}

function similarityInterpretation(similarity: number): string {
  if (similarity > 0.9) return 'Very high similarity'
  if (similarity > 0.7) return 'High similarity'
  if (similarity > 0.5) return 'Moderate similarity'
  if (similarity > 0.3) return 'Low similarity'
  return 'Very low similarity'
}

function cosineWordSimilarity(text1: string, text2: string): number {
  const left = frequency(splitWhitespaceWords(text1))
  const right = frequency(splitWhitespaceWords(text2))
  const vocabulary = new Set([...left.keys(), ...right.keys()])
  let dot = 0
  let leftNorm = 0
  let rightNorm = 0
  for (const word of vocabulary) {
    const leftValue = left.get(word) ?? 0
    const rightValue = right.get(word) ?? 0
    dot += leftValue * rightValue
    leftNorm += leftValue * leftValue
    rightNorm += rightValue * rightValue
  }
  if (leftNorm === 0 || rightNorm === 0) return 0
  return dot / Math.sqrt(leftNorm * rightNorm)
}

function cosineVectors(left: readonly number[], right: readonly number[]): number {
  if (left.length !== right.length || left.length === 0) {
    throw new Error('semantic embedding vectors must have one non-zero shared dimension')
  }
  let dot = 0
  let leftNorm = 0
  let rightNorm = 0
  for (let index = 0; index < left.length; index += 1) {
    const leftValue = left[index] ?? 0
    const rightValue = right[index] ?? 0
    dot += leftValue * rightValue
    leftNorm += leftValue * leftValue
    rightNorm += rightValue * rightValue
  }
  if (leftNorm === 0 || rightNorm === 0) return 0
  return dot / Math.sqrt(leftNorm * rightNorm)
}

function levenshteinDistance(first: string, second: string): number {
  if (first === second) return 0
  let longer = first
  let shorter = second
  if (shorter.length > longer.length) {
    longer = second
    shorter = first
  }
  let previous = Array.from({ length: shorter.length + 1 }, (_value, index) => index)
  for (let row = 0; row < longer.length; row += 1) {
    const current: number[] = [row + 1]
    const source = longer[row] ?? ''
    for (let column = 0; column < shorter.length; column += 1) {
      const target = shorter[column] ?? ''
      const insertion = (previous[column + 1] ?? 0) + 1
      const deletion = (current[column] ?? 0) + 1
      const substitution = (previous[column] ?? 0) + (source === target ? 0 : 1)
      current.push(Math.min(insertion, deletion, substitution))
    }
    previous = current
  }
  return previous[previous.length - 1] ?? 0
}

function extractEntityType(text: string, entityType: EntityType): string[] {
  switch (entityType) {
    case 'emails':
      return patternMatches(text, /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b/gu)
    case 'urls':
      return uniqueLimited(
        patternMatches(text, /https?:\/\/[^\s<>"']+/giu).map(url => url.replace(/[),.!?;:]+$/u, '')),
      )
    case 'phone_numbers':
      return patternMatches(
        text,
        /\+?[(]?[0-9]{1,4}[)]?[-\s.]?[(]?[0-9]{1,4}[)]?[-\s.]?[0-9]{1,4}[-\s.]?[0-9]{1,9}/gu,
      )
    case 'dates':
      return patternMatches(text, /\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2})\b/gu)
    case 'times':
      return patternMatches(text, /\b\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AP]M)?\b/giu)
    case 'numbers':
      return patternMatches(text, /\b\d+(?:\.\d+)?\b/gu)
    case 'hashtags':
      return patternMatches(text, /#\w+/gu)
    case 'mentions':
      return patternMatches(text, /(?<![A-Za-z0-9._%+-])@\w+/gu)
    case 'currency':
      return patternMatches(text, /[$€£¥][\d,]+(?:\.\d{2})?/gu)
    case 'names':
      return patternMatches(text, /\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b/gu)
  }
}

function patternMatches(text: string, pattern: RegExp): string[] {
  const matches: string[] = []
  for (const match of text.matchAll(pattern)) {
    const value = match[0]
    if (value !== undefined && value) matches.push(value)
  }
  return uniqueLimited(matches)
}

function uniqueLimited(values: readonly string[]): string[] {
  const seen = new Set<string>()
  const result: string[] = []
  for (const value of values) {
    if (!value || seen.has(value)) continue
    seen.add(value)
    result.push(value)
    if (result.length >= MAX_RETURNED_ENTITIES) break
  }
  return result
}

function isEntityType(value: string): value is EntityType {
  return (ENTITY_TYPES as readonly string[]).includes(value)
}

function splitSentences(text: string): string[] {
  return text.split(/[.!?]+/u).map(sentence => sentence.trim()).filter(Boolean)
}

function tokenize(text: string): string[] {
  return text.toLowerCase().match(/[\p{L}\p{N}]+(?:['’-][\p{L}\p{N}]+)*/gu) ?? []
}

function splitWhitespaceWords(text: string): string[] {
  return text.toLowerCase().split(/\s+/u).filter(Boolean)
}

function frequency(values: readonly string[]): Map<string, number> {
  const result = new Map<string, number>()
  for (const value of values) {
    result.set(value, (result.get(value) ?? 0) + 1)
  }
  return result
}

function mostFrequent(values: ReadonlyMap<string, number>, maximum: number): string[] {
  return [...values.entries()]
    .sort((left, right) => right[1] - left[1] || lexicalCompare(left[0], right[0]))
    .slice(0, maximum)
    .map(([value]) => value)
}

function lexicalCompare(left: string, right: string): number {
  if (left < right) return -1
  if (left > right) return 1
  return 0
}

function normalizeVector(vector: readonly number[]): number[] {
  const magnitude = Math.sqrt(vector.reduce((total, value) => total + value * value, 0))
  return magnitude === 0 ? [...vector] : vector.map(value => value / magnitude)
}

function embeddingInputTexts(inputs: JsonObject): string[] {
  const value = inputs.text
  if (typeof value === 'string') return [value]
  if (!Array.isArray(value)) {
    throw new ValidationError('text', 'must be a string or an array of strings', value)
  }
  const texts: string[] = []
  for (const item of value) {
    if (typeof item !== 'string') {
      throw new ValidationError('text', 'must be a string or an array of strings', value)
    }
    texts.push(item)
  }
  return texts
}

function textValue(inputs: JsonObject, field: string): string {
  const value = inputs[field]
  if (typeof value !== 'string') {
    throw new ValidationError(field, 'must be a string', value)
  }
  return value
}

function optionalText(inputs: JsonObject, field: string): string | undefined {
  const value = inputs[field]
  if (value === undefined) return undefined
  if (typeof value !== 'string') {
    throw new ValidationError(field, 'must be a string', value)
  }
  return value
}

function optionalStringArray(inputs: JsonObject, field: string): string[] {
  const value = inputs[field]
  if (value === undefined) return []
  if (!Array.isArray(value)) {
    throw new ValidationError(field, 'must be an array of strings', value)
  }
  const strings: string[] = []
  for (const item of value) {
    if (typeof item !== 'string') {
      throw new ValidationError(field, 'must be an array of strings', value)
    }
    strings.push(item)
  }
  return strings
}

function boundedInteger(
  inputs: JsonObject,
  field: string,
  defaultValue: number,
  minimum: number,
  maximum: number,
): number {
  const value = inputs[field]
  if (value === undefined) return defaultValue
  if (typeof value !== 'number' || !Number.isInteger(value) || value < minimum || value > maximum) {
    throw new ValidationError(field, 'must be an integer between ' + String(minimum) + ' and ' + String(maximum), value)
  }
  return value
}

function optionalBoundedInteger(
  inputs: JsonObject,
  field: string,
  minimum: number,
  maximum: number,
): number | undefined {
  const value = inputs[field]
  if (value === undefined) return undefined
  if (typeof value !== 'number' || !Number.isInteger(value) || value < minimum || value > maximum) {
    throw new ValidationError(field, 'must be an integer between ' + String(minimum) + ' and ' + String(maximum), value)
  }
  return value
}

function failure(error: string): JsonObject {
  return { error }
}

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}
