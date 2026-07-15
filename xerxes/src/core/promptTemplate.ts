// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

export const SEP = '  '

export const PromptSection = {
  SYSTEM: 'system',
  PERSONA: 'persona',
  RULES: 'rules',
  FUNCTIONS: 'functions',
  TOOLS: 'tools',
  EXAMPLES: 'examples',
  CONTEXT: 'context',
  HISTORY: 'history',
  PROMPT: 'prompt',
} as const

export type PromptSection = (typeof PromptSection)[keyof typeof PromptSection]

export type PromptSections = Readonly<Partial<Record<PromptSection, string>>>

export const DEFAULT_PROMPT_SECTIONS: Readonly<Record<PromptSection, string>> = Object.freeze({
  [PromptSection.SYSTEM]: 'SYSTEM:',
  [PromptSection.PERSONA]: 'PERSONA:',
  [PromptSection.RULES]: 'RULES:',
  [PromptSection.FUNCTIONS]: 'FUNCTIONS:',
  [PromptSection.TOOLS]: 'TOOLS:\n' + SEP + 'When using tools, follow this format:',
  [PromptSection.EXAMPLES]: 'EXAMPLES:\n' + SEP,
  [PromptSection.CONTEXT]: 'CONTEXT:\n',
  [PromptSection.HISTORY]: 'HISTORY:\n' + SEP + 'Conversation so far:\n',
  [PromptSection.PROMPT]: 'PROMPT:\n',
})

export const DEFAULT_PROMPT_SECTION_ORDER: readonly PromptSection[] = Object.freeze([
  PromptSection.SYSTEM,
  PromptSection.RULES,
  PromptSection.FUNCTIONS,
  PromptSection.TOOLS,
  PromptSection.EXAMPLES,
  PromptSection.CONTEXT,
  PromptSection.HISTORY,
  PromptSection.PROMPT,
])

export interface PromptTemplateOptions {
  readonly sectionOrder?: readonly PromptSection[]
  readonly sections?: PromptSections
}

/**
 * Ordered prompt-section metadata shared by legacy prompt assembly code.
 *
 * The streaming runtime normally passes a complete system prompt directly,
 * while callers that need structured assembly can render one deterministic
 * prompt through this portable representation.
 */
export class PromptTemplate {
  readonly sectionOrder: readonly PromptSection[]
  readonly sections: Readonly<Record<PromptSection, string>>

  constructor(options: PromptTemplateOptions = {}) {
    this.sections = Object.freeze({
      ...DEFAULT_PROMPT_SECTIONS,
      ...options.sections,
    })
    this.sectionOrder = Object.freeze([...(options.sectionOrder ?? DEFAULT_PROMPT_SECTION_ORDER)])
  }

  /** Render only supplied non-blank sections in their configured order. */
  render(values: PromptSections, separator = '\n\n'): string {
    const parts: string[] = []
    for (const section of this.sectionOrder) {
      const content = values[section]?.trim()
      if (!content) continue
      const header = this.sections[section]?.trim()
      parts.push(header ? header + '\n' + content : content)
    }
    return parts.join(separator)
  }
}
